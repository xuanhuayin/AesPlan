#!/usr/bin/env python3
"""
P2-7: Full Spatial Preference Finetuning (SD3)
==============================================

This script is a standalone training entrypoint for discrepancy-weighted
spatial preference finetuning. It does NOT modify any existing AesPlan code.

Data format (JSONL):
  {"prompt": "...", "chosen": "/abs/or/rel/path/chosen.png", "rejected": "..."}

Core design:
  1) Build discrepancy mask M from frozen reference model:
       M <- soft_topk(|eps_cond - eps_uncond|)
  2) Compute spatially weighted denoising losses for chosen/rejected:
       Lc = wMSE(pred_c, target; M),  Lr = wMSE(pred_r, target; M)
  3) Preference objective (pairwise logistic):
       L_pref = -log sigmoid(beta * (score_c - score_r))
       score = -wMSE(...)
  4) Add chosen-anchor denoising term for stability:
       L = L_pref + lambda_anchor * Lc

Usage example:
  python scripts/run_spatial_preference_finetune_sd3.py \
    --pairs data/prefs/train.jsonl \
    --output outputs/spatial_pref_ft_sd3 \
    --epochs 1 --batch_size 1 --lr 1e-5
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


ACCELAES_ROOT = Path(
    os.environ.get(
        "ACCELAES_ROOT",
        str(Path(__file__).resolve().parent.parent.parent / "AccelAes"),
    )
)
sys.path.insert(0, str(ACCELAES_ROOT))
sys.path.insert(0, str(ACCELAES_ROOT / "src"))

from src.models.sd3_wrapper import SD3DiTWrapper  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dtype_from_str(name: str) -> torch.dtype:
    table = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name.lower() not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name.lower()]


def _load_image_tensor(path: str | Path, size: int) -> torch.Tensor:
    """Load image, resize to square, normalize to [-1, 1], return CHW."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class PreferencePairDataset(Dataset):
    """JSONL preference pairs: prompt + chosen/rejected image paths."""

    def __init__(self, jsonl_path: str | Path, image_size: int = 1024):
        self.jsonl_path = Path(jsonl_path)
        self.image_size = image_size
        self.items: List[dict] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for key in ("prompt", "chosen", "rejected"):
                    if key not in obj:
                        raise ValueError(f"Missing '{key}' in dataset item: {obj}")
                self.items.append(obj)
        if not self.items:
            raise ValueError(f"No valid rows in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.items[idx]
        base = self.jsonl_path.parent
        chosen_path = Path(row["chosen"])
        rejected_path = Path(row["rejected"])
        if not chosen_path.is_absolute():
            chosen_path = (base / chosen_path).resolve()
        if not rejected_path.is_absolute():
            rejected_path = (base / rejected_path).resolve()

        chosen = _load_image_tensor(chosen_path, self.image_size)
        rejected = _load_image_tensor(rejected_path, self.image_size)
        return {
            "prompt": row["prompt"],
            "chosen": chosen,
            "rejected": rejected,
        }


def _collate(batch: List[dict]) -> dict:
    prompts = [x["prompt"] for x in batch]
    chosen = torch.stack([x["chosen"] for x in batch], dim=0)
    rejected = torch.stack([x["rejected"] for x in batch], dim=0)
    return {"prompts": prompts, "chosen": chosen, "rejected": rejected}


@dataclass
class TextCond:
    cond_embed: torch.Tensor
    uncond_embed: torch.Tensor
    cond_pooled: torch.Tensor
    uncond_pooled: torch.Tensor


class PromptEmbedCache:
    """Small in-memory cache to avoid re-encoding repeated prompts."""

    def __init__(self):
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def get(
        self,
        pipe,
        prompt: str,
        device: torch.device,
    ) -> TextCond:
        if prompt not in self.cache:
            out = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                negative_prompt="",
                do_classifier_free_guidance=True,
                num_images_per_prompt=1,
                device=device,
            )
            # SD3 returns: prompt, negative, pooled, negative_pooled
            p, n, pp, npool = out
            self.cache[prompt] = (
                p.detach().cpu(),
                n.detach().cpu(),
                pp.detach().cpu(),
                npool.detach().cpu(),
            )
        p, n, pp, npool = self.cache[prompt]
        return TextCond(
            cond_embed=p.to(device=device),
            uncond_embed=n.to(device=device),
            cond_pooled=pp.to(device=device),
            uncond_pooled=npool.to(device=device),
        )


def _encode_pixels_to_latents(vae, pixels: torch.Tensor) -> torch.Tensor:
    """
    Encode image pixels ([-1,1]) to SD3 latent space.
    SD3 decode path is:
      latents = (latents / scaling_factor) + shift_factor
    so encode inverse is:
      latents = (z - shift_factor) * scaling_factor
    """
    posterior = vae.encode(pixels).latent_dist
    z = posterior.sample()
    return (z - vae.config.shift_factor) * vae.config.scaling_factor


def _add_noise(
    scheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    # Preferred branch for diffusers schedulers.
    if hasattr(scheduler, "add_noise"):
        return scheduler.add_noise(latents, noise, timesteps)
    # Fallback if scheduler lacks add_noise.
    t = timesteps.float().view(-1, 1, 1, 1)
    t = t / max(1, scheduler.config.num_train_timesteps - 1)
    return latents + t * noise


def _target_from_scheduler(
    scheduler,
    latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if pred_type == "epsilon":
        return noise
    if pred_type == "v_prediction" and hasattr(scheduler, "get_velocity"):
        return scheduler.get_velocity(latents, noise, timesteps)
    # Flow-matching/custom cases: fallback to noise target.
    return noise


def _sd3_forward(
    transformer: nn.Module,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
) -> torch.Tensor:
    return transformer(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        return_dict=False,
    )[0]


def _compute_discrepancy_mask(
    ref_transformer: nn.Module,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    cond_embed: torch.Tensor,
    uncond_embed: torch.Tensor,
    cond_pooled: torch.Tensor,
    uncond_pooled: torch.Tensor,
    top_ratio: float,
    sharpness: float,
    blur_sigma: float = 0.0,
) -> torch.Tensor:
    """
    Build soft spatial mask from |eps_cond - eps_uncond| at the training step.
    Returns shape (B,1,H,W), values in [0,1].
    """
    b = noisy_latents.shape[0]
    latents_cfg = torch.cat([noisy_latents, noisy_latents], dim=0)
    t_cfg = timesteps.repeat(2)
    embeds = torch.cat([uncond_embed, cond_embed], dim=0)  # SD3 order: [uncond, cond]
    pooled = torch.cat([uncond_pooled, cond_pooled], dim=0)

    with torch.no_grad():
        pred = _sd3_forward(ref_transformer, latents_cfg, t_cfg, embeds, pooled)
        eps_uncond, eps_cond = pred.chunk(2, dim=0)
        mag = (eps_cond - eps_uncond).abs().mean(dim=1, keepdim=True)  # (B,1,H,W)

        if blur_sigma > 0:
            k = max(3, int(2 * round(2 * blur_sigma) + 1))
            mag = F.avg_pool2d(mag, kernel_size=k, stride=1, padding=k // 2)

        x = mag.view(b, -1)
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-8)
        tau = torch.quantile(x, q=(1.0 - top_ratio), dim=1, keepdim=True)
        soft = torch.sigmoid((x - tau) * sharpness)
        return soft.view_as(mag)


def _weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
) -> torch.Tensor:
    """
    Spatially weighted MSE reduced to per-sample scalar.
    pred/target: (B,C,H,W), weight_map: (B,1,H,W)
    """
    err = (pred - target).pow(2).mean(dim=1, keepdim=True)  # (B,1,H,W)
    num = (err * weight_map).flatten(1).sum(dim=1)
    den = weight_map.flatten(1).sum(dim=1).clamp(min=1e-6)
    return num / den


def _save_checkpoint(
    out_dir: Path,
    step: int,
    transformer: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pt"
    torch.save(
        {
            "step": step,
            "transformer": transformer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": vars(args),
        },
        path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, required=True, help="Path to JSONL preference pairs")
    parser.add_argument("--output", type=str, default="outputs/spatial_pref_ft_sd3")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--beta", type=float, default=1.0, help="Preference logistic temperature")
    parser.add_argument("--lambda_anchor", type=float, default=0.5, help="Chosen denoise anchor weight")
    parser.add_argument("--w_fg", type=float, default=4.0)
    parser.add_argument("--w_bg", type=float, default=1.0)
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Top ratio for discrepancy mask")
    parser.add_argument("--mask_sharpness", type=float, default=8.0)
    parser.add_argument("--mask_blur_sigma", type=float, default=0.0)

    parser.add_argument("--t_min", type=int, default=20)
    parser.add_argument("--t_max", type=int, default=980)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument(
        "--no_frozen_ref",
        action="store_true",
        help="Use current model itself for mask (lower VRAM, weaker stability).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)
    train_dtype = _dtype_from_str(args.dtype)

    print("[Init] Loading SD3 wrapper...")
    wrapper = SD3DiTWrapper(dtype=args.dtype)
    device = wrapper.device
    pipe = wrapper.pipe
    transformer = wrapper.transformer
    vae = wrapper.vae
    scheduler = pipe.scheduler

    transformer.train()
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    if args.no_frozen_ref:
        ref_transformer = transformer
        print("[Init] Reference model: current transformer (no frozen copy).")
    else:
        print("[Init] Creating frozen reference transformer copy...")
        ref_transformer = copy.deepcopy(transformer).to(device)
        ref_transformer.eval()
        for p in ref_transformer.parameters():
            p.requires_grad_(False)

    dataset = PreferencePairDataset(args.pairs, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=True,
    )
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"[Data] pairs={len(dataset)} batch={args.batch_size} epochs={args.epochs} total_steps={total_steps}")

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    use_amp = (device.type == "cuda") and (train_dtype in (torch.float16, torch.bfloat16))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and train_dtype == torch.float16))

    prompt_cache = PromptEmbedCache()
    global_step = 0

    print("[Train] Starting...")
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(loader):
            prompts: List[str] = batch["prompts"]
            chosen = batch["chosen"].to(device=device, dtype=vae.dtype, non_blocking=True)
            rejected = batch["rejected"].to(device=device, dtype=vae.dtype, non_blocking=True)
            bsz = chosen.shape[0]

            with torch.no_grad():
                chosen_lat = _encode_pixels_to_latents(vae, chosen).to(dtype=transformer.dtype)
                rejected_lat = _encode_pixels_to_latents(vae, rejected).to(dtype=transformer.dtype)

                noise_c = torch.randn_like(chosen_lat)
                noise_r = torch.randn_like(rejected_lat)

                t_low = max(0, args.t_min)
                t_high = min(args.t_max, scheduler.config.num_train_timesteps - 1)
                timesteps = torch.randint(
                    low=t_low,
                    high=t_high + 1,
                    size=(bsz,),
                    device=device,
                    dtype=torch.long,
                )

                noisy_c = _add_noise(scheduler, chosen_lat, noise_c, timesteps)
                noisy_r = _add_noise(scheduler, rejected_lat, noise_r, timesteps)
                target_c = _target_from_scheduler(scheduler, chosen_lat, noise_c, timesteps).to(transformer.dtype)
                target_r = _target_from_scheduler(scheduler, rejected_lat, noise_r, timesteps).to(transformer.dtype)

                cond_embeds, uncond_embeds = [], []
                cond_pooleds, uncond_pooleds = [], []
                for ptxt in prompts:
                    tc = prompt_cache.get(pipe, ptxt, device)
                    cond_embeds.append(tc.cond_embed)
                    uncond_embeds.append(tc.uncond_embed)
                    cond_pooleds.append(tc.cond_pooled)
                    uncond_pooleds.append(tc.uncond_pooled)

                cond_embed = torch.cat(cond_embeds, dim=0).to(dtype=transformer.dtype)
                uncond_embed = torch.cat(uncond_embeds, dim=0).to(dtype=transformer.dtype)
                cond_pooled = torch.cat(cond_pooleds, dim=0).to(dtype=transformer.dtype)
                uncond_pooled = torch.cat(uncond_pooleds, dim=0).to(dtype=transformer.dtype)

                mask_soft = _compute_discrepancy_mask(
                    ref_transformer=ref_transformer,
                    noisy_latents=noisy_c,
                    timesteps=timesteps,
                    cond_embed=cond_embed,
                    uncond_embed=uncond_embed,
                    cond_pooled=cond_pooled,
                    uncond_pooled=uncond_pooled,
                    top_ratio=args.mask_ratio,
                    sharpness=args.mask_sharpness,
                    blur_sigma=args.mask_blur_sigma,
                )
                weight_map = args.w_bg + (args.w_fg - args.w_bg) * mask_soft

            with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_amp):
                pred_c = _sd3_forward(
                    transformer=transformer,
                    noisy_latents=noisy_c,
                    timesteps=timesteps,
                    encoder_hidden_states=cond_embed,
                    pooled_projections=cond_pooled,
                )
                pred_r = _sd3_forward(
                    transformer=transformer,
                    noisy_latents=noisy_r,
                    timesteps=timesteps,
                    encoder_hidden_states=cond_embed,
                    pooled_projections=cond_pooled,
                )

                mse_c = _weighted_mse(pred_c.float(), target_c.float(), weight_map.float())
                mse_r = _weighted_mse(pred_r.float(), target_r.float(), weight_map.float())

                score_c = -mse_c
                score_r = -mse_r
                pref_margin = args.beta * (score_c - score_r)
                loss_pref = -F.logsigmoid(pref_margin).mean()
                loss_anchor = mse_c.mean()
                loss = loss_pref + args.lambda_anchor * loss_anchor

            loss = loss / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % args.log_every == 0:
                with torch.no_grad():
                    fg_ratio = float((mask_soft > 0.5).float().mean().item())
                    msg = (
                        f"[Train] step={global_step}/{total_steps} "
                        f"loss={loss.item() * args.grad_accum:.4f} "
                        f"pref={loss_pref.item():.4f} anchor={loss_anchor.item():.4f} "
                        f"mse_c={mse_c.mean().item():.4f} mse_r={mse_r.mean().item():.4f} "
                        f"fg>0.5={fg_ratio:.3f}"
                    )
                    print(msg)

            if global_step % args.save_every == 0:
                path = _save_checkpoint(out_dir, global_step, transformer, optimizer, scaler, args)
                print(f"[CKPT] Saved {path}")

    final_path = _save_checkpoint(out_dir, global_step, transformer, optimizer, scaler, args)
    print(f"[Done] Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
