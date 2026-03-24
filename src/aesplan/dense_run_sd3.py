"""
src/aesplan/dense_run_sd3.py
==============================
Dense generation with eps_cond / eps_uncond capture for SD3.

SD3 differences from Lumina:
  - Batch order: [uncond, cond]
  - Transformer args: hidden_states, timestep, encoder_hidden_states, pooled_projections
  - Decode: (latents / scaling_factor) + shift_factor
  - No rotary embedding, no cross_attention_kwargs

FG mask: SoftCFG magnitude |eps_cond - eps_uncond| at mask_step.
SD3 uses standard double-pass CFG, so cond/uncond are free at every step.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

import os as _os
ACCELAES_ROOT = Path(_os.environ.get("ACCELAES_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent / "AccelAes")))
sys.path.insert(0, str(ACCELAES_ROOT))
sys.path.insert(0, str(ACCELAES_ROOT / "src"))

from src.sparse.sdit_mask_builder import CFGMagnitudeMaskBuilder


def run_dense_and_capture_sd3(
    wrapper,
    prompt: str,
    seed: int,
    cfg_scale: float = 7.0,
    steps: int = 28,
    mask_step: int = 5,
    skip_ratio: float = 0.5,
    height: int = 1024,
    width: int = 1024,
) -> Dict:
    """Run dense SD3 generation capturing eps_cond/eps_uncond at every step.

    Returns:
        eps_cond:  list[T] of (1, 16, latent_h, latent_w) cpu float tensors
        eps_uncond: list[T] of (1, 16, latent_h, latent_w) cpu float tensors
        fg_mask:   (1, 1, latent_h, latent_w) cpu float tensor, 1=FG (soft)
    """
    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    # ---- 1. Encode prompt ----
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt="",
        do_classifier_free_guidance=True,
        num_images_per_prompt=1,
        device=device,
    )

    # SD3 batch order: [uncond, cond]
    prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # ---- 2. Prepare latents ----
    latent_h = height // wrapper.vae_scale_factor   # 128
    latent_w = width  // wrapper.vae_scale_factor   # 128
    latent_channels = wrapper.transformer.config.in_channels  # 16

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, latent_channels, latent_h, latent_w),
        generator=generator, device=device, dtype=dtype,
    )

    # ---- 3. Scheduler ----
    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps

    mask_builder = CFGMagnitudeMaskBuilder(ratio=skip_ratio)

    # ---- 4. Denoising loop ----
    eps_cond_list: List[torch.Tensor] = []
    eps_uncond_list: List[torch.Tensor] = []
    fg_mask_spatial: Optional[torch.Tensor] = None

    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2, dim=0)

        with torch.no_grad():
            noise_pred = wrapper.transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(latent_model_input.shape[0]),
                encoder_hidden_states=prompt_embeds_cfg,
                pooled_projections=pooled_cfg,
                return_dict=False,
            )[0]

        # SD3 batch order: [uncond, cond]
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

        # Build SoftCFG mask at mask_step
        if i == mask_step:
            fg_mask_spatial = mask_builder.build_mask(
                noise_pred_cond, noise_pred_uncond
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, latent_h, latent_w)

        eps_cond_list.append(noise_pred_cond.cpu().float())
        eps_uncond_list.append(noise_pred_uncond.cpu().float())

        # Standard CFG step
        guided = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        latents = pipe.scheduler.step(guided, t, latents, return_dict=False)[0]

    if fg_mask_spatial is None:
        raise RuntimeError("fg_mask_spatial was never set — mask_step out of range?")

    return {
        "eps_cond":   eps_cond_list,
        "eps_uncond": eps_uncond_list,
        "fg_mask":    fg_mask_spatial.cpu().float(),
    }
