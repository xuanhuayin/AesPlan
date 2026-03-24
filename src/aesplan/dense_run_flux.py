"""
src/aesplan/dense_run_flux.py
==============================
FLUX dense run: captures noise_pred at every step + guidance_diff mask
at mask_step (one extra forward with g=1.0).

FLUX specifics:
  - Single-pass guidance distillation (g=3.5 standard)
  - No double-pass CFG, so cfg_mag requires extra forward(g=1.0) at mask_step
  - guidance_diff = |pred_g3.5 - pred_g1.0| at mask_step
  - Skip steps: reuse cached noise_pred (no differential CFG, not worth 2× per key step)
  - latents are packed: (1, H/2*W/2, 16*4) — use unpack only for decode
"""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

import os as _os
ACCELAES_ROOT = Path(_os.environ.get("ACCELAES_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent / "AccelAes")))
sys.path.insert(0, str(ACCELAES_ROOT))
sys.path.insert(0, str(ACCELAES_ROOT / "src"))

from src.sparse.boundary_ops import gaussian_blur


def build_guidance_diff_mask_flux(
    noise_g35: torch.Tensor,   # (1, seq_len, 64) packed latents
    noise_g10: torch.Tensor,   # same shape
    latent_h: int,
    latent_w: int,
    ratio: float = 0.5,
    blur_sigma: float = 2.0,
    sharpness: float = 8.0,
) -> torch.Tensor:
    """Build soft FG mask from guidance_diff = |noise_g35 - noise_g10| at mask_step.

    Uses the same SoftCFG pipeline as Lumina:
      1. |noise_g35 - noise_g10|.mean(channels)  → (tok_h, tok_w) heatmap
      2. Gaussian blur (sigma=blur_sigma)
      3. Normalize [0, 1]
      4. Soft sigmoid around (1-ratio) quantile

    Returns (1, 1, latent_h, latent_w) soft float mask in [0, 1].
    """
    diff = (noise_g35 - noise_g10).abs()       # (1, seq_len, 64)
    diff_mean = diff.mean(dim=-1)               # (1, seq_len)

    tok_h = latent_h // 2
    tok_w = latent_w // 2
    diff_map = diff_mean.squeeze(0).view(tok_h, tok_w).float()  # (tok_h, tok_w)

    # Gaussian blur to smooth token-space noise
    if blur_sigma > 0:
        diff_map = gaussian_blur(diff_map, sigma=blur_sigma)

    # Normalize [0, 1]
    diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)

    # Soft sigmoid threshold
    tau = diff_map.flatten().quantile(1.0 - ratio)
    soft = torch.sigmoid((diff_map - tau) * sharpness)  # (tok_h, tok_w)

    # Upsample to latent size (1, 1, latent_h, latent_w)
    mask = soft.unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(mask, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
    return mask.float()


def run_dense_and_capture_flux(
    wrapper,
    prompt: str,
    seed: int,
    guidance_scale: float = 3.5,
    steps: int = 28,
    mask_step: int = 5,
    skip_ratio: float = 0.5,
    height: int = 1024,
    width: int = 1024,
) -> Dict:
    """Dense FLUX generation, capturing noise_pred at every step.

    At mask_step: does one extra forward with guidance_scale=1.0 to get
    guidance_diff mask (equivalent of cfg_mag on FLUX).

    Returns dict:
        noise_preds: list[T] of (1, seq_len, 64) cpu float tensors
        guidance_diff_mask: (1, 1, latent_h, latent_w) float tensor
        latent_h, latent_w: int
    """
    prompt_embeds, pooled_prompt_embeds, text_ids = wrapper._encode_prompt(prompt)
    wrapper._ensure_transformer_on_gpu()

    latent_h = height // wrapper.vae_scale_factor
    latent_w = width  // wrapper.vae_scale_factor
    latents, img_ids = wrapper._prepare_latents(height, width, seed)
    timesteps = wrapper._setup_scheduler(steps, height, width)

    txt_ids = torch.zeros(
        prompt_embeds.shape[1], 3,
        device=wrapper.device, dtype=wrapper.dtype,
    )

    noise_preds: List[torch.Tensor] = []
    guidance_diff_mask: Optional[torch.Tensor] = None

    for i, t in enumerate(timesteps):
        noise_pred = wrapper._transformer_step(
            latents, t, prompt_embeds, pooled_prompt_embeds,
            txt_ids, img_ids, guidance_scale,
        )
        noise_preds.append(noise_pred.cpu().float())

        # At mask_step: extra forward with g=1.0 for guidance_diff mask
        if i == mask_step:
            noise_pred_g1 = wrapper._transformer_step(
                latents, t, prompt_embeds, pooled_prompt_embeds,
                txt_ids, img_ids, 1.0,
            )
            guidance_diff_mask = build_guidance_diff_mask_flux(
                noise_pred.cpu().float(),
                noise_pred_g1.cpu().float(),
                latent_h, latent_w, skip_ratio,
            )

        latents = wrapper.scheduler.step(
            noise_pred, t, latents, return_dict=False
        )[0]

    if guidance_diff_mask is None:
        # fallback: uniform mask (shouldn't happen if mask_step < steps)
        guidance_diff_mask = torch.ones(
            1, 1, latent_h, latent_w, dtype=torch.float32
        )

    return {
        "noise_preds": noise_preds,
        "guidance_diff_mask": guidance_diff_mask,
        "latent_h": latent_h,
        "latent_w": latent_w,
    }
