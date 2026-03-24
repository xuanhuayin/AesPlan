"""
src/aesplan/inference_sd3.py
==============================
AesPlan inference for SD3: DP-planned key steps + differential CFG at skip steps
+ Taylor-2 extrapolation on combined eps.

Key design (same as Lumina):
  - Key steps:  standard uniform CFG (s=cfg_scale), update Taylor cache on combined eps
  - Skip steps: Taylor-2 predict combined eps, then optionally apply spatial correction:
                  eps_half = combined_hat + (s_map - cfg_scale) * (eps_cond_cached - eps_uncond_cached)
  - FG mask:    SoftCFG magnitude at mask_step (no hooks needed)

SD3 specifics:
  - Batch order: [uncond, cond]
  - Decode: (latents / scaling_factor) + shift_factor
  - No rotary embedding, no cross_attention_kwargs
"""
from __future__ import annotations

from typing import List, Optional, Set

import torch

from .calibration import CalibrationResult
from .inference import _update_taylor_cache, _taylor_predict

import os as _os
ACCELAES_ROOT_STR = _os.environ.get("ACCELAES_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent / "AccelAes"))
import sys
sys.path.insert(0, ACCELAES_ROOT_STR)
sys.path.insert(0, ACCELAES_ROOT_STR + "/src")

from src.sparse.sdit_mask_builder import CFGMagnitudeMaskBuilder


def generate_aesplan_sd3(
    wrapper,
    prompt: str,
    seed: int,
    calib: CalibrationResult,
    s_fg: float = 7.0,
    s_bg: float = 2.0,
    height: int = 1024,
    width: int = 1024,
    taylor_order: int = 2,
    # Ablation flags
    use_aes_cost: bool = True,
    use_diff_cfg: bool = True,
    key_steps_override: Optional[List[int]] = None,
) -> "PIL.Image":
    """
    AesPlan inference on SD3.

    Ablation modes:
      use_aes_cost=True,  use_diff_cfg=True  → full AesPlan
      use_aes_cost=False, use_diff_cfg=False → DPCache-style (uniform cost DP)
      use_aes_cost=True,  use_diff_cfg=False → cost-only (no differential CFG)
      use_aes_cost=False, use_diff_cfg=True  → diff-CFG-only (uniform schedule)

    NOTE: differential CFG is applied ONLY at skip steps.
          Key steps always use standard uniform CFG (cfg_scale) to preserve quality.
    """
    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    T = calib.total_steps
    mask_step = calib.mask_step

    # --- Determine key steps ---
    if key_steps_override is not None:
        key_set: Set[int] = set(key_steps_override)
    elif use_aes_cost:
        key_set = set(calib.key_steps)
    else:
        budget = calib.budget
        first_free = calib.first_free
        step = max(1, T // budget)
        key_set = set(range(0, T, step)[:budget])
        for i in range(first_free):
            key_set.add(i)

    key_set.add(mask_step)

    # --- Encode prompt ---
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

    # --- Prepare latents ---
    latent_h = height // wrapper.vae_scale_factor
    latent_w = width  // wrapper.vae_scale_factor
    latent_channels = wrapper.transformer.config.in_channels

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, latent_channels, latent_h, latent_w),
        generator=generator, device=device, dtype=dtype,
    )

    pipe.scheduler.set_timesteps(T, device=device)
    timesteps = pipe.scheduler.timesteps

    mask_builder = CFGMagnitudeMaskBuilder(ratio=calib.skip_ratio)

    # --- Cache ---
    cache_eps_cond:    Optional[torch.Tensor] = None  # for spatial diff CFG correction
    cache_eps_uncond:  Optional[torch.Tensor] = None  # for spatial diff CFG correction
    cache_eps_combined: Optional[torch.Tensor] = None
    s_map: Optional[torch.Tensor] = None  # (1, 1, latent_h, latent_w)

    # Taylor cache on combined eps (stable: combined cancels correlated noise)
    taylor_cache_combined: dict = {}

    for i, t in enumerate(timesteps):
        is_key = (i in key_set)

        # --- Skip step: Taylor-1 on combined eps + optional spatial correction ---
        if not is_key and cache_eps_combined is not None:
            if taylor_order > 0 and "last_key" in taylor_cache_combined:
                combined_hat = _taylor_predict(taylor_cache_combined, i, taylor_order)
            else:
                combined_hat = cache_eps_combined  # order-0: plain cache reuse

            if use_diff_cfg and s_map is not None and cache_eps_cond is not None:
                diff = cache_eps_cond - cache_eps_uncond
                guided = combined_hat + (s_map - calib.cfg_scale) * diff
            else:
                guided = combined_hat

            latents = pipe.scheduler.step(guided, t, latents, return_dict=False)[0]
            continue

        # --- Key step: full forward pass ---
        latent_model_input = torch.cat([latents] * 2, dim=0)

        with torch.no_grad():
            noise_pred = wrapper.transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(latent_model_input.shape[0]),
                encoder_hidden_states=prompt_embeds_cfg,
                pooled_projections=pooled_cfg,
                return_dict=False,
            )[0]

        # SD3: [uncond, cond]
        eps_uncond_t, eps_cond_t = noise_pred.chunk(2)

        # Build SoftCFG mask at mask_step
        if i == mask_step and s_map is None:
            fg_mask = mask_builder.build_mask(
                eps_cond_t, eps_uncond_t
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, latent_h, latent_w)
            s_map = (s_bg + fg_mask * (s_fg - s_bg)).to(device=device, dtype=dtype)

        # Update caches
        cache_eps_cond   = eps_cond_t
        cache_eps_uncond = eps_uncond_t

        # Key steps: always STANDARD uniform CFG (spatial diff CFG only at skip steps)
        guided = eps_uncond_t + calib.cfg_scale * (eps_cond_t - eps_uncond_t)
        cache_eps_combined = guided.clone()

        # Update Taylor cache on combined eps
        taylor_cache_combined = _update_taylor_cache(taylor_cache_combined, cache_eps_combined, i, taylor_order)

        latents = pipe.scheduler.step(guided, t, latents, return_dict=False)[0]

    # --- Decode ---
    latents = latents.detach()
    latents = (latents / wrapper.vae.config.scaling_factor) + wrapper.vae.config.shift_factor
    with torch.no_grad():
        image = wrapper.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image.detach(), output_type="pil")[0]
