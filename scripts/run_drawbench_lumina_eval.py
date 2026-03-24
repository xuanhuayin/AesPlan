#!/usr/bin/env python3
"""
scripts/run_drawbench_lumina_eval.py
=====================================
Main paper experiments: AesPlan vs baselines on Lumina, DrawBench 109 prompts.

Variants:
  baseline      — full 30-step inference (quality upper bound)
  dpcache       — uniform DP schedule + cache reuse at skip (DPCache-style)
  diff_cfg_only — uniform DP schedule + diff CFG at skip  (isolates diff CFG)
  aes_cost_only — AesMask DP schedule + cache reuse       (isolates DP schedule)
  aesplan_full  — AesMask DP schedule + diff CFG at skip  (full method)

Metrics: ImageReward, Aesthetic, CLIP (no HPS to avoid OOM/swap)
Output:  outputs/drawbench_lumina_eval/summary.json

Usage:
  python scripts/run_drawbench_lumina_eval.py [--prompts 109] [--seeds 1] [--budget 10]
  python scripts/run_drawbench_lumina_eval.py --load_calib outputs/drawbench_lumina_eval/calibration
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ACCELAES_ROOT = Path("/home/runkai/xuanhua/AccelAes")
AESDIT_ROOT   = Path("/home/runkai/xuanhua/AesDiT")
sys.path.insert(0, str(ACCELAES_ROOT))
sys.path.insert(0, str(ACCELAES_ROOT / "src"))
sys.path.insert(0, str(AESDIT_ROOT / "src"))

import numpy as np
import torch

from src.models.dit_wrapper import LuminaDiTWrapper
from aesplan.calibration import AesPlanCalibrator, CalibrationResult
from aesplan.inference import generate_aesplan


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
CALIB_PROMPTS = [
    "A stunning portrait of a woman with intricate floral headdress, photorealistic",
    "A majestic eagle soaring above misty mountain peaks, ultra detailed feathers",
    "A beautiful butterfly resting on a vibrant red rose, macro photography",
    "A detailed painting of a knight in ornate golden armor, fantasy art",
    "A photorealistic portrait of a young girl with wildflowers in her hair",
]

def load_drawbench(path: str, max_prompts: int = None) -> list:
    with open(path) as f:
        data = json.load(f)
    prompts = [p for cat in data.values() for p in cat]
    if max_prompts:
        prompts = prompts[:max_prompts]
    return prompts


# ---------------------------------------------------------------------------
# Metrics  (load to CPU, no HPS to avoid OOM)
# ---------------------------------------------------------------------------
class _IRMetric:
    def __init__(self, device):
        import ImageReward as ir
        self.model = ir.load("ImageReward-v1.0", device=device)
    @torch.no_grad()
    def score(self, img, prompt):
        return float(self.model.score(prompt, img))

class _AesMetric:
    def __init__(self, device):
        import open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device)
        self.model.eval()
        cache = str(ACCELAES_ROOT / "src" / ".cache"); os.makedirs(cache, exist_ok=True)
        wp = os.path.join(cache, "aesthetic_mlp_l14.pth")
        if not os.path.exists(wp):
            torch.hub.download_url_to_file(
                "https://github.com/christophschuhmann/improved-aesthetic-predictor"
                "/raw/main/sac+logos+ava1-l14-linearMSE.pth", wp)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(768,1024), torch.nn.Dropout(0.2),
            torch.nn.Linear(1024,128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128,64),  torch.nn.Dropout(0.1),
            torch.nn.Linear(64,16),   torch.nn.Linear(16,1),
        ).to(device)
        state = torch.load(wp, map_location=device, weights_only=True)
        self.mlp.load_state_dict({k.replace("layers.", ""): v for k, v in state.items()})
        self.mlp.eval()
    @torch.no_grad()
    def score(self, img):
        t = self.preprocess(img).unsqueeze(0).to(self.device)
        f = self.model.encode_image(t); f = f / f.norm(dim=-1, keepdim=True)
        return self.mlp(f.float()).item()

class _CLIPMetric:
    def __init__(self, device):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=device)
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.model.eval(); self.device = device
    @torch.no_grad()
    def score(self, img, prompt):
        t = self.preprocess(img).unsqueeze(0).to(self.device)
        tok = self.tokenizer([prompt]).to(self.device)
        fi = self.model.encode_image(t); fi = fi / fi.norm(dim=-1, keepdim=True)
        ft = self.model.encode_text(tok); ft = ft / ft.norm(dim=-1, keepdim=True)
        return float((fi * ft).sum())

def load_metrics(device="cpu"):
    print("  Loading ImageReward...")
    ir = _IRMetric(device)
    print("  Loading Aesthetic (ViT-L-14)...")
    aes = _AesMetric(device)
    print("  Loading CLIP (ViT-L-14)...")
    clip = _CLIPMetric(device)
    return {"ir": ir, "aesthetic": aes, "clip": clip}

def score_image(metrics, img, prompt):
    return {
        "ir":        metrics["ir"].score(img, prompt),
        "aesthetic": metrics["aesthetic"].score(img),
        "clip":      metrics["clip"].score(img, prompt),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_uniform_key_steps(total_steps, budget, first_free):
    """Even spacing, always include first_free warm-up steps."""
    ks = set(range(first_free))
    step = max(1, total_steps // budget)
    for s in range(first_free, total_steps, step):
        ks.add(s)
    ks = sorted(ks)[:budget]
    # Pad to budget if needed
    while len(ks) < budget:
        for s in range(total_steps):
            if s not in ks:
                ks.append(s); break
    return sorted(ks[:budget])

def mean_scores(scores_list):
    keys = scores_list[0].keys()
    return {k: float(np.mean([s[k] for s in scores_list])) for k in keys}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts",      type=int,   default=109)
    parser.add_argument("--seeds",        type=int,   default=1)
    parser.add_argument("--budget",       type=int,   default=10,
                        help="Key step budget (out of --steps). 10/30 → ~3× speedup")
    parser.add_argument("--steps",        type=int,   default=30)
    parser.add_argument("--mask_step",    type=int,   default=5)
    parser.add_argument("--cfg_scale",    type=float, default=4.0)
    parser.add_argument("--s_fg",         type=float, default=7.0)
    parser.add_argument("--s_bg",         type=float, default=1.0)
    parser.add_argument("--w_fg",         type=float, default=4.0)
    parser.add_argument("--w_bg",         type=float, default=1.0)
    parser.add_argument("--calib_samples",type=int,   default=5)
    parser.add_argument("--first_free",   type=int,   default=4)
    parser.add_argument("--max_skip",     type=int,   default=8)
    parser.add_argument("--output",       type=str,   default="outputs/drawbench_lumina_eval")
    parser.add_argument("--dtype",        type=str,   default="bf16")
    parser.add_argument("--load_calib",   type=str,   default=None,
                        help="Path prefix to load saved calibration (skip re-calib)")
    parser.add_argument("--drawbench",    type=str,
                        default=str(AESDIT_ROOT / "data" / "drawbench.json"))
    parser.add_argument("--skip_baseline",action="store_true",
                        help="Skip baseline run (use fixed 12.37s reference)")
    args = parser.parse_args()

    out_dir = AESDIT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_prompts = load_drawbench(args.drawbench, args.prompts)
    seeds_list   = list(range(args.seeds))
    total_imgs   = len(eval_prompts) * len(seeds_list)

    print(f"\n{'='*65}")
    print(f"AesPlan DrawBench Lumina Eval")
    print(f"  Budget:  {args.budget}/{args.steps} key steps (~{args.steps/args.budget:.1f}× speedup)")
    print(f"  Prompts: {len(eval_prompts)}  Seeds: {args.seeds}  Total: {total_imgs} imgs/variant")
    print(f"{'='*65}\n")

    # --- Load metrics FIRST (before model, to avoid OOM) ---
    print("Loading metrics (CPU)...")
    metrics = load_metrics("cpu")
    print("Metrics loaded.\n")

    # --- Load model ---
    print("Loading LuminaDiTWrapper...")
    wrapper = LuminaDiTWrapper(dtype=args.dtype)
    device  = wrapper.device

    # --- Calibration ---
    if args.load_calib and Path(args.load_calib + ".json").exists():
        print(f"Loading calibration from {args.load_calib}")
        calib = CalibrationResult.load(args.load_calib)
    else:
        print(f"Running calibration ({args.calib_samples} samples)...")
        calibrator = AesPlanCalibrator(
            wrapper=wrapper,
            budget=args.budget,
            w_fg=args.w_fg,
            w_bg=args.w_bg,
            cfg_scale=args.cfg_scale,
            steps=args.steps,
            mask_step=args.mask_step,
            first_free=args.first_free,
        )
        calib = calibrator.run(
            prompts=CALIB_PROMPTS[:args.calib_samples],
            seeds=[42] * args.calib_samples,
        )
        calib_prefix = str(out_dir / "calibration")
        calib.save(calib_prefix)
        print(f"Calibration saved to {calib_prefix}")

    uniform_ks = build_uniform_key_steps(args.steps, args.budget, calib.first_free)

    print(f"\nAesPlan key steps ({len(calib.key_steps)}): {calib.key_steps}")
    print(f"Uniform  key steps ({len(uniform_ks)}): {uniform_ks}")

    # ---------------------------------------------------------------------------
    # Variants definition
    # ---------------------------------------------------------------------------
    # baseline: key_steps_override = all steps (no skipping, identical to full pipeline)
    all_steps = list(range(args.steps))

    VARIANTS = [
        # (name, description, generate_aesplan kwargs)
        ("baseline",      "Full 30-step inference (no skipping)",
         dict(use_aes_cost=False, use_diff_cfg=False, key_steps_override=all_steps)),

        ("dpcache",       f"DPCache: uniform DP + cache reuse at skip, K={args.budget}",
         dict(use_aes_cost=False, use_diff_cfg=False, key_steps_override=uniform_ks)),

        ("diff_cfg_only", f"Diff-CFG only: uniform DP + diff CFG at skip, K={args.budget}",
         dict(use_aes_cost=False, use_diff_cfg=True,  key_steps_override=uniform_ks)),

        ("aes_cost_only", f"AesMask DP only: AesMask schedule + cache reuse, K={args.budget}",
         dict(use_aes_cost=True,  use_diff_cfg=False)),

        ("aesplan_full",  f"AesPlan full: AesMask DP + diff CFG at skip, K={args.budget}",
         dict(use_aes_cost=True,  use_diff_cfg=True)),
    ]

    if args.skip_baseline:
        VARIANTS = [v for v in VARIANTS if v[0] != "baseline"]

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    all_results = {}

    for vname, vdesc, vkwargs in VARIANTS:
        print(f"\n{'='*55}")
        print(f"Variant: {vname}")
        print(f"  {vdesc}")
        print(f"{'='*55}")

        vdir = out_dir / vname
        vdir.mkdir(exist_ok=True)

        scores_all = []
        times_all  = []

        for pi, prompt in enumerate(eval_prompts):
            for seed in seeds_list:
                idx = pi * len(seeds_list) + seed + 1
                t0 = time.time()
                img = generate_aesplan(
                    wrapper=wrapper,
                    prompt=prompt,
                    seed=seed,
                    calib=calib,
                    s_fg=args.s_fg,
                    s_bg=args.s_bg,
                    **vkwargs,
                )
                elapsed = time.time() - t0
                times_all.append(elapsed)

                s = score_image(metrics, img, prompt)
                scores_all.append(s)
                img.save(vdir / f"p{pi:03d}_s{seed}.png")

                print(f"  [{idx:3d}/{total_imgs}] IR={s['ir']:.3f} "
                      f"Aes={s['aesthetic']:.3f} CLIP={s['clip']:.4f} "
                      f"t={elapsed:.1f}s | {prompt[:50]}", flush=True)

        ms = mean_scores(scores_all)
        mt = float(np.mean(times_all))
        n_ks = len(vkwargs.get("key_steps_override", calib.key_steps))

        all_results[vname] = {
            "desc":       vdesc,
            "n_key_steps": n_ks,
            "mean_time":  mt,
            "mean_ir":    ms["ir"],
            "mean_aesthetic": ms["aesthetic"],
            "mean_clip":  ms["clip"],
            "std_ir":     float(np.std([s["ir"] for s in scores_all])),
            "scores":     scores_all,
            "times":      times_all,
        }

        print(f"\n  [{vname}] IR={ms['ir']:.3f}  Aes={ms['aesthetic']:.3f}  "
              f"CLIP={ms['clip']:.4f}  t={mt:.1f}s/img", flush=True)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY — AesPlan DrawBench Lumina")
    print(f"{'='*70}")

    # Use baseline time if available, else 12.37s (from P0 locked results)
    if "baseline" in all_results:
        base_t = all_results["baseline"]["mean_time"]
    else:
        base_t = 12.37

    print(f"{'Variant':<18} {'Speedup':>8} {'IR':>7} {'Aes':>7} {'CLIP':>7} {'t/img':>7}")
    print("-" * 60)
    for vname, r in all_results.items():
        spd = base_t / r["mean_time"]
        print(f"{vname:<18} {spd:>7.2f}×  {r['mean_ir']:>7.3f}  "
              f"{r['mean_aesthetic']:>7.3f}  {r['mean_clip']:>7.4f}  {r['mean_time']:>6.1f}s")

    if "dpcache" in all_results and "aesplan_full" in all_results:
        d = all_results["dpcache"]
        a = all_results["aesplan_full"]
        print(f"\nAesPlan vs DPCache (at same K={args.budget} budget):")
        print(f"  IR:   {a['mean_ir']:.3f} vs {d['mean_ir']:.3f}  "
              f"(delta={a['mean_ir']-d['mean_ir']:+.3f})")
        print(f"  Aes:  {a['mean_aesthetic']:.3f} vs {d['mean_aesthetic']:.3f}  "
              f"(delta={a['mean_aesthetic']-d['mean_aesthetic']:+.3f})")
        print(f"  CLIP: {a['mean_clip']:.4f} vs {d['mean_clip']:.4f}  "
              f"(delta={a['mean_clip']-d['mean_clip']:+.4f})")

    if "diff_cfg_only" in all_results and "dpcache" in all_results:
        d = all_results["dpcache"]
        c = all_results["diff_cfg_only"]
        print(f"\nDiff-CFG contribution (same uniform schedule, diff CFG vs cache reuse):")
        print(f"  IR delta: {c['mean_ir']-d['mean_ir']:+.3f}  "
              f"({c['mean_ir']:.3f} vs {d['mean_ir']:.3f})")

    # Save
    summary = {
        "config": vars(args),
        "n_prompts": len(eval_prompts),
        "baseline_time_ref": base_t,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "scores"}
                    for k, v in all_results.items()},
        "scores_per_variant": {k: v["scores"] for k, v in all_results.items()},
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
