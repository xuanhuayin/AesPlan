#!/usr/bin/env python3
"""
scripts/run_aesplan_sd3_eval.py
=================================
AesPlan evaluation on SD3: compare 4 variants + AccelAes SD3 reference.

Variants:
  1. uniform_dp    — uniform DP schedule, standard CFG (DPCache-style baseline)
  2. aes_cost_only — AesMask-weighted DP schedule, standard CFG
  3. diff_cfg_only — uniform schedule, differential CFG at skip steps
  4. aesplan_full  — AesMask-weighted schedule + differential CFG (full method)
  5. accelaes_ref  — AccelAes fskip2 on SD3 (existing baseline)
  6. spectrum_ref  — Spectrum-style Chebyshev skip (if installed) [optional]

Metrics: ImageReward, HPSv2, Aesthetic Score, CLIP Score, time/img.

Usage:
  python scripts/run_aesplan_sd3_eval.py [--prompts 20] [--seeds 2] [--budget 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ACCELAES_ROOT = Path("/home/runkai/xuanhua/AccelAes")
AESDIT_ROOT = Path("/home/runkai/xuanhua/AesDiT")
sys.path.insert(0, str(ACCELAES_ROOT))
sys.path.insert(0, str(ACCELAES_ROOT / "src"))
sys.path.insert(0, str(AESDIT_ROOT / "src"))

import numpy as np
import torch

from src.models.sd3_wrapper import SD3DiTWrapper
from aesplan.calibration import CalibrationResult
from aesplan.calibration_sd3 import AesPlanCalibratorSD3
from aesplan.dp_solver import solve_dp
from aesplan.inference_sd3 import generate_aesplan_sd3
from aesplan.dpcache_baseline import calibrate_dpcache_sd3, generate_dpcache_sd3


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

EVAL_PROMPTS_20 = [
    "A peacock displaying its intricate iridescent plumage, photorealistic",
    "A serene Japanese garden with cherry blossoms reflected in a koi pond",
    "Portrait of an elderly craftsman with deeply weathered hands, studio lighting",
    "A fantasy dragon resting on a mountain peak, detailed scales, golden hour",
    "A vibrant street food market in Bangkok at night, neon lights",
    "Close-up of a hummingbird in flight, droplets on feathers, ultra sharp",
    "An astronaut floating in space, Earth reflected in the visor",
    "A watercolor painting of rolling Tuscan hills at sunset",
    "Intricate mandala pattern in gold and emerald, symmetrical, high detail",
    "A wolf running through snow in a pine forest, motion blur, dramatic",
    "A delicate lace wedding dress on display, soft window light",
    "Portrait of a lion with a flowing mane, golden savannah at dusk",
    "An ancient library with towering bookshelves and dust particles in light",
    "A tropical reef with colorful fish, crystal clear water, sunbeams",
    "A blacksmith forging a sword, sparks flying, dramatic red glow",
    "A child blowing dandelion seeds, shallow depth of field",
    "The Aurora Borealis over a snowy Scandinavian landscape",
    "A steampunk clockwork city, intricate gears and copper pipes",
    "A mother bear and two cubs in an autumn forest, warm tones",
    "A vintage tea set on a wooden table, soft bokeh background",
]


# ---------------------------------------------------------------------------
# Inline metric classes
# ---------------------------------------------------------------------------
class _IRMetric:
    def __init__(self, device):
        import ImageReward as ir_module
        self.model = ir_module.load("ImageReward-v1.0", device=device)
    @torch.no_grad()
    def score(self, img, prompt):
        return float(self.model.score(prompt, img))

class _HPSMetric:
    def __init__(self, device):
        import hpsv2; self.hpsv2 = hpsv2
    @torch.no_grad()
    def score(self, img, prompt):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name); tmp = f.name
        r = self.hpsv2.score(tmp, prompt, hps_version="v2.1")
        os.unlink(tmp)
        return float(r[0]) if isinstance(r, (list, np.ndarray)) else float(r)

class _AesMetric:
    def __init__(self, device):
        import open_clip, torch.nn as nn
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
        self.mlp = nn.Sequential(
            nn.Linear(768,1024), nn.Dropout(0.2), nn.Linear(1024,128), nn.Dropout(0.2),
            nn.Linear(128,64), nn.Dropout(0.1), nn.Linear(64,16), nn.Linear(16,1),
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

def load_metrics(device):
    return {
        "ir":        _IRMetric(device),
        "hps":       _HPSMetric(device),
        "aesthetic": _AesMetric(device),
        "clip":      _CLIPMetric(device),
    }

def score_image(metrics, img, prompt):
    return {
        "ir":        metrics["ir"].score(img, prompt),
        "hps":       metrics["hps"].score(img, prompt),
        "aesthetic": metrics["aesthetic"].score(img),
        "clip":      metrics["clip"].score(img, prompt),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--budget", type=int, default=10,
                        help="Key step budget K (out of 28). Default=10 → ~2.8× target")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--mask_step", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--s_fg", type=float, default=7.0)
    parser.add_argument("--s_bg", type=float, default=2.0)
    parser.add_argument("--w_fg", type=float, default=4.0)
    parser.add_argument("--w_bg", type=float, default=1.0)
    parser.add_argument("--calib_samples", type=int, default=5)
    parser.add_argument("--first_free", type=int, default=6,
                        help="Always-compute warm-up steps. Reduce for high-speedup runs (e.g. 4 for budget=6)")
    parser.add_argument("--max_skip", type=int, default=4,
                        help="Max consecutive skipped steps in DP (default=4)")
    parser.add_argument("--output", type=str, default="outputs/aesplan_sd3_eval")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--load_calib", type=str, default=None)
    parser.add_argument("--skip_accelaes", action="store_true")
    parser.add_argument("--skip_dpcache", action="store_true",
                        help="Skip DPCache baseline (saves time)")
    args = parser.parse_args()

    out_dir = AESDIT_ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_prompts = EVAL_PROMPTS_20[:args.prompts]
    seeds_list = list(range(args.seeds))

    print(f"\n{'='*60}")
    print(f"AesPlan SD3 Evaluation")
    print(f"  Budget: {args.budget}/{args.steps} key steps (~{args.steps/args.budget:.1f}× speedup)")
    print(f"  Prompts: {args.prompts}  Seeds: {args.seeds}")
    print(f"  s_fg={args.s_fg}  s_bg={args.s_bg}  w_fg={args.w_fg}  w_bg={args.w_bg}")
    print(f"{'='*60}\n")

    print("Loading SD3DiTWrapper...")
    wrapper = SD3DiTWrapper(dtype=args.dtype)
    device = wrapper.device

    # --- Calibration ---
    if args.load_calib and Path(args.load_calib).exists():
        print(f"Loading calibration from {args.load_calib}")
        calib = CalibrationResult.load(args.load_calib)
    else:
        print(f"Running SD3 calibration ({args.calib_samples} samples)...")
        calibrator = AesPlanCalibratorSD3(
            wrapper=wrapper,
            budget=args.budget,
            w_fg=args.w_fg,
            w_bg=args.w_bg,
            cfg_scale=args.cfg_scale,
            steps=args.steps,
            mask_step=args.mask_step,
            first_free=args.first_free,
            max_skip=args.max_skip,
        )
        calib = calibrator.run(
            prompts=CALIB_PROMPTS[:args.calib_samples],
            seeds=[42] * args.calib_samples,
        )
        calib.save(out_dir / "calibration")

    # Build uniform key steps (same budget)
    first_free = calib.first_free
    budget = calib.budget
    step_size = max(1, args.steps // budget)
    uniform_key_steps = sorted(
        set(range(first_free)) | set(range(first_free, args.steps, step_size))
    )[:budget]
    while len(uniform_key_steps) < budget:
        for s in range(args.steps):
            if s not in uniform_key_steps:
                uniform_key_steps.append(s); break
    uniform_key_steps = sorted(uniform_key_steps[:budget])

    print(f"\nAesPlan key steps ({len(calib.key_steps)}): {calib.key_steps}")
    print(f"Uniform key steps ({len(uniform_key_steps)}): {uniform_key_steps}")

    # --- Load metrics ---
    print("\nLoading metrics...")
    metrics = load_metrics(device=str(device))

    # --- Variants ---
    variants = {
        "uniform_dp": dict(
            use_aes_cost=False, use_diff_cfg=False,
            key_steps_override=uniform_key_steps,
            desc=f"Uniform DP (standard CFG), K={budget}",
        ),
        "aes_cost_only": dict(
            use_aes_cost=True, use_diff_cfg=False,
            desc=f"AesMask cost only (no diff-CFG), K={budget}",
        ),
        "diff_cfg_only": dict(
            use_aes_cost=False, use_diff_cfg=True,
            key_steps_override=uniform_key_steps,
            desc=f"Diff-CFG only (uniform schedule), K={budget}",
        ),
        "aesplan_full": dict(
            use_aes_cost=True, use_diff_cfg=True,
            desc=f"AesPlan full (aes cost + diff-CFG), K={budget}",
        ),
    }

    results = {}
    total_imgs = args.prompts * args.seeds

    for var_name, var_kwargs in variants.items():
        print(f"\n{'─'*50}")
        print(f"Variant: {var_name} — {var_kwargs['desc']}")
        print(f"{'─'*50}")

        var_dir = out_dir / var_name
        var_dir.mkdir(exist_ok=True)

        scores, times = [], []
        for pi, prompt in enumerate(eval_prompts):
            for seed in seeds_list:
                t0 = time.time()
                img = generate_aesplan_sd3(
                    wrapper=wrapper,
                    prompt=prompt,
                    seed=seed,
                    calib=calib,
                    s_fg=args.s_fg,
                    s_bg=args.s_bg,
                    **{k: v for k, v in var_kwargs.items() if k != "desc"},
                )
                elapsed = time.time() - t0
                times.append(elapsed)
                s = score_image(metrics, img, prompt)
                scores.append(s)
                img.save(var_dir / f"p{pi:03d}_s{seed}.png")
                idx = pi * args.seeds + seed + 1
                print(f"  [{idx:3d}/{total_imgs}] "
                      f"IR={s['ir']:.3f} HPS={s['hps']:.4f} "
                      f"Aes={s['aesthetic']:.3f} t={elapsed:.1f}s")

        results[var_name] = {
            "desc": var_kwargs["desc"],
            "mean_time": float(np.mean(times)),
            "mean_ir": float(np.mean([s["ir"] for s in scores])),
            "mean_hps": float(np.mean([s["hps"] for s in scores])),
            "mean_aesthetic": float(np.mean([s["aesthetic"] for s in scores])),
            "mean_clip": float(np.mean([s["clip"] for s in scores])),
            "std_ir": float(np.std([s["ir"] for s in scores])),
            "n_images": len(scores),
        }
        r = results[var_name]
        print(f"\n  Mean: IR={r['mean_ir']:.3f}  HPS={r['mean_hps']:.4f}  "
              f"Aes={r['mean_aesthetic']:.3f}  t={r['mean_time']:.1f}s/img")

    # --- DPCache baseline (3D PACT, content-agnostic) ---
    if not args.skip_dpcache:
        print(f"\n{'─'*50}")
        print("Baseline: DPCache (3D PACT, content-agnostic, no spatial CFG)")
        print(f"{'─'*50}")
        dpc_dir = out_dir / "dpcache"
        dpc_dir.mkdir(exist_ok=True)

        print(f"  Calibrating DPCache ({args.calib_samples} samples)...")
        dpc_key_steps = calibrate_dpcache_sd3(
            wrapper=wrapper,
            prompts=CALIB_PROMPTS[:args.calib_samples],
            seeds=[42] * args.calib_samples,
            cfg_scale=args.cfg_scale,
            steps=args.steps,
            mask_step=args.mask_step,
            budget=args.budget,
            first_free=args.first_free,
            max_skip=args.max_skip,
        )
        print(f"  DPCache key_steps: {dpc_key_steps}")

        scores, times = [], []
        for pi, prompt in enumerate(eval_prompts):
            for seed in seeds_list:
                t0 = time.time()
                img = generate_dpcache_sd3(
                    wrapper=wrapper,
                    prompt=prompt,
                    seed=seed,
                    key_steps=dpc_key_steps,
                    cfg_scale=args.cfg_scale,
                    steps=args.steps,
                )
                elapsed = time.time() - t0
                times.append(elapsed)
                s = score_image(metrics, img, prompt)
                scores.append(s)
                img.save(dpc_dir / f"p{pi:03d}_s{seed}.png")
                idx = pi * args.seeds + seed + 1
                print(f"  [{idx:3d}/{total_imgs}] "
                      f"IR={s['ir']:.3f} HPS={s['hps']:.4f} "
                      f"Aes={s['aesthetic']:.3f} t={elapsed:.1f}s")

        results["dpcache"] = {
            "desc": f"DPCache 3D-PACT content-agnostic, K={budget}",
            "key_steps": dpc_key_steps,
            "mean_time": float(np.mean(times)),
            "mean_ir": float(np.mean([s["ir"] for s in scores])),
            "mean_hps": float(np.mean([s["hps"] for s in scores])),
            "mean_aesthetic": float(np.mean([s["aesthetic"] for s in scores])),
            "mean_clip": float(np.mean([s["clip"] for s in scores])),
            "std_ir": float(np.std([s["ir"] for s in scores])),
            "n_images": len(scores),
        }
        r = results["dpcache"]
        print(f"\n  DPCache: IR={r['mean_ir']:.3f}  HPS={r['mean_hps']:.4f}  "
              f"Aes={r['mean_aesthetic']:.3f}  t={r['mean_time']:.1f}s/img")

    # --- AccelAes reference (SD3 fskip2) ---
    if not args.skip_accelaes:
        print(f"\n{'─'*50}")
        print("Reference: AccelAes fskip2 on SD3")
        print(f"{'─'*50}")
        accel_dir = out_dir / "accelaes_ref"
        accel_dir.mkdir(exist_ok=True)
        scores, times = [], []
        for pi, prompt in enumerate(eval_prompts):
            for seed in seeds_list:
                t0 = time.time()
                img = wrapper.generate_accelerated(
                    prompt=prompt, seed=seed,
                    mask_type="cfg_magnitude",
                    skip_ratio=0.50,
                    s_fg=args.s_fg, s_bg=args.s_bg,
                    mask_step=args.mask_step,
                    full_skip_interval=2,
                    sparse_attn=True,
                    cfg_scale=args.cfg_scale,
                    steps=args.steps,
                )
                elapsed = time.time() - t0
                times.append(elapsed)
                s = score_image(metrics, img, prompt)
                scores.append(s)
                img.save(accel_dir / f"p{pi:03d}_s{seed}.png")

        results["accelaes_ref"] = {
            "desc": "AccelAes fskip2 SD3 (1.50× reference)",
            "mean_time": float(np.mean(times)),
            "mean_ir": float(np.mean([s["ir"] for s in scores])),
            "mean_hps": float(np.mean([s["hps"] for s in scores])),
            "mean_aesthetic": float(np.mean([s["aesthetic"] for s in scores])),
            "mean_clip": float(np.mean([s["clip"] for s in scores])),
            "std_ir": float(np.std([s["ir"] for s in scores])),
            "n_images": len(scores),
        }

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    baseline_time = 4.0  # SD3 dense ~4s/img
    for var_name, r in results.items():
        speedup = baseline_time / r["mean_time"]
        print(f"{var_name:20s}  {speedup:.2f}×  IR={r['mean_ir']:.3f}  "
              f"HPS={r['mean_hps']:.4f}  Aes={r['mean_aesthetic']:.3f}")

    out_json = out_dir / "summary.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
