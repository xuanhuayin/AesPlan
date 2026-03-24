"""
src/aesplan/inference.py
========================
AesPlan inference: run Lumina with DP-planned key steps + differential CFG
+ Taylor-2 extrapolation on combined eps at skip steps.

Key design:
  1. Key steps: run full CFG forward, update Taylor cache on combined eps
  2. Skip steps: Taylor-1 predict combined eps, then optionally apply spatial
     differential CFG correction:
       eps_half = combined_hat + (s_map - cfg_scale) * (eps_cond_cached - eps_uncond_cached)
     When use_diff_cfg=False: eps_half = combined_hat  (consistent with DPCache)

This module exports generate_aesplan() which takes a LuminaDiTWrapper and
a CalibrationResult, and runs inference for a given prompt/seed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Set

import torch
import torch.nn.functional as F

import os as _os
ACCELAES_ROOT = Path(_os.environ.get("ACCELAES_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent / "AccelAes")))
sys.path.insert(0, str(ACCELAES_ROOT))
sys.path.insert(0, str(ACCELAES_ROOT / "src"))

from src.models.dit_wrapper import get_2d_rotary_pos_embed_lumina
from src.sparse.sdit_mask_builder import CFGMagnitudeMaskBuilder

from .calibration import CalibrationResult


# ---------------------------------------------------------------------------
# Taylor cache helpers (ported from inference_flux.py)
# ---------------------------------------------------------------------------

def _update_taylor_cache(
    taylor_cache: dict,
    noise_pred: torch.Tensor,
    step_idx: int,
    taylor_order: int,
) -> dict:
    """Update Taylor derivative cache at a key step."""
    new_cache: dict = {"last_key": step_idx, 0: noise_pred.clone()}

    if "last_key" not in taylor_cache or taylor_order < 1:
        return new_cache

    dist = step_idx - taylor_cache["last_key"]
    if dist <= 0:
        return new_cache

    d1 = (noise_pred - taylor_cache[0]) / dist
    new_cache[1] = d1

    if taylor_order >= 2 and 1 in taylor_cache:
        d2 = (d1 - taylor_cache[1]) / dist
        new_cache[2] = d2

    return new_cache


def _taylor_predict(
    taylor_cache: dict,
    step_idx: int,
    taylor_order: int,
) -> torch.Tensor:
    """Predict noise_pred at a skip step via Taylor extrapolation.

    pred ≈ f0 + f1*d + f2/2*d^2  (computed in fp32 to avoid overflow)
    """
    d = float(step_idx - taylor_cache["last_key"])
    orig_dtype = taylor_cache[0].dtype

    pred = taylor_cache[0].float()

    if taylor_order >= 1 and 1 in taylor_cache:
        pred = pred + taylor_cache[1].float() * d

    if taylor_order >= 2 and 2 in taylor_cache:
        pred = pred + taylor_cache[2].float() * (d * d / 2.0)

    return pred.to(orig_dtype)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def generate_aesplan(
    wrapper,
    prompt: str,
    seed: int,
    calib: CalibrationResult,
    s_fg: float = 7.0,
    s_bg: float = 1.0,
    height: int = 1024,
    width: int = 1024,
    taylor_order: int = 2,
    # Ablation flags
    use_aes_cost: bool = True,    # if False → uniform DP cost (DPCache baseline)
    use_diff_cfg: bool = True,    # if False → Taylor on combined only (no spatial correction)
    # Optionally override key steps (e.g. for uniform baseline)
    key_steps_override: Optional[List[int]] = None,
) -> "PIL.Image":
    """
    AesPlan inference.

    At skip steps:
      - Taylor-1 predict combined_hat from cached combined eps trajectory
      - If use_diff_cfg: add spatial correction using cached (eps_cond - eps_uncond)
          eps_half = combined_hat + (s_map - cfg_scale) * (eps_cond_cached - eps_uncond_cached)
      - If not use_diff_cfg: eps_half = combined_hat  (pure DPCache-style)

    Ablation modes:
      use_aes_cost=True,  use_diff_cfg=True  → full AesPlan
      use_aes_cost=False, use_diff_cfg=False → DPCache-style (uniform DP + Taylor on combined)
      use_aes_cost=True,  use_diff_cfg=False → cost-only (AesMask DP + Taylor on combined)
      use_aes_cost=False, use_diff_cfg=True  → diff-CFG-only (uniform DP + spatial correction)

    Args:
        wrapper: LuminaDiTWrapper
        prompt: text prompt
        seed: generation seed
        calib: CalibrationResult from AesPlanCalibrator
        s_fg: foreground CFG scale
        s_bg: background CFG scale
        taylor_order: 0=cache reuse, 1=linear, 2=quadratic (default, stable on combined eps)
        use_aes_cost: use AesMask-weighted planned schedule
        use_diff_cfg: apply differential CFG spatial correction at skip steps
        key_steps_override: explicit list of key steps (for baseline comparison)
    """
    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    T = calib.total_steps
    mask_step = calib.mask_step

    # Determine key step set
    if key_steps_override is not None:
        key_set: Set[int] = set(key_steps_override)
    elif use_aes_cost:
        key_set = set(calib.key_steps)
    else:
        # Uniform spacing: same budget as AesPlan
        budget = calib.budget
        step = max(1, T // budget)
        key_set = set(range(0, T, step)[:budget])
        # Always include first_free steps
        for i in range(calib.first_free):
            key_set.add(i)

    # --- Encode prompt ---
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=True,
        negative_prompt="",
        num_images_per_prompt=1,
        device=device,
    )

    prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds], dim=0)
    prompt_attention_mask = torch.cat(
        [prompt_attention_mask, negative_prompt_attention_mask], dim=0
    )

    cross_attention_kwargs = {}
    default_sample_size = getattr(pipe, "default_sample_size", 128)
    cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

    scaling_factor = math.sqrt(width * height / wrapper.default_image_size ** 2)

    latent_h = height // wrapper.vae_scale_factor
    latent_w = width // wrapper.vae_scale_factor
    latent_channels = wrapper.transformer.config.in_channels
    shape = (1, latent_channels, latent_h, latent_w)

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    pipe.scheduler.set_timesteps(T, device=device)
    timesteps = pipe.scheduler.timesteps

    head_dim = wrapper.transformer.head_dim

    # --- SoftCFG mask builder (no hooks needed) ---
    mask_builder = CFGMagnitudeMaskBuilder(ratio=calib.skip_ratio)

    # --- Cache ---
    cache_eps_cond: Optional[torch.Tensor] = None    # (1, 3, H, W) for spatial diff CFG
    cache_eps_uncond: Optional[torch.Tensor] = None  # (1, 3, H, W) for spatial diff CFG
    cache_eps_combined: Optional[torch.Tensor] = None  # (1, 3, H, W) combined eps
    cache_noise_pred_rest: Optional[torch.Tensor] = None  # (1, 1, H, W) 4th channel
    fg_mask_spatial: Optional[torch.Tensor] = None  # (1, 1, latent_h, latent_w)
    s_map: Optional[torch.Tensor] = None

    # Taylor cache on combined eps (stable: combined cancels correlated noise)
    taylor_cache_combined: dict = {}

    for i, t in enumerate(timesteps):
        is_key = (i in key_set)

        # --- Skip step: Taylor-1 on combined eps + optional spatial correction ---
        if not is_key and cache_eps_combined is not None:
            # Predict combined eps via Taylor extrapolation
            if taylor_order > 0 and "last_key" in taylor_cache_combined:
                combined_hat = _taylor_predict(taylor_cache_combined, i, taylor_order)
            else:
                combined_hat = cache_eps_combined  # order-0: plain cache reuse

            if use_diff_cfg and s_map is not None and cache_eps_cond is not None:
                # Spatial correction using cached cond/uncond structure:
                #   eps_half = combined_hat + (s_map - cfg_scale) * (eps_cond - eps_uncond)
                # When s_map = cfg_scale (uniform), correction vanishes → pure Taylor
                diff = cache_eps_cond - cache_eps_uncond
                eps_half = combined_hat + (s_map - calib.cfg_scale) * diff
            else:
                eps_half = combined_hat

            # Scheduler needs full latent channels: cat eps (3ch) + rest (1ch)
            if cache_noise_pred_rest is not None:
                noise_pred_step = -torch.cat([eps_half, cache_noise_pred_rest], dim=1)
            else:
                noise_pred_step = -torch.cat(
                    [eps_half, torch.zeros_like(eps_half[:, :1])], dim=1
                )
            latents = pipe.scheduler.step(noise_pred_step, t, latents, return_dict=False)[0]
            continue

        # --- Key step: run transformer ---
        latent_model_input = torch.cat([latents] * 2, dim=0)

        current_timestep = t
        if not torch.is_tensor(current_timestep):
            current_timestep = torch.tensor([current_timestep], dtype=torch.float64, device=device)
        elif len(current_timestep.shape) == 0:
            current_timestep = current_timestep[None].to(device)
        current_timestep = current_timestep.expand(latent_model_input.shape[0])
        current_timestep = 1 - current_timestep / pipe.scheduler.config.num_train_timesteps

        if current_timestep[0] < 1.0:
            linear_factor = scaling_factor
            ntk_factor = 1.0
        else:
            linear_factor = 1.0
            ntk_factor = scaling_factor

        image_rotary_emb = get_2d_rotary_pos_embed_lumina(
            head_dim, 384, 384,
            linear_factor=linear_factor,
            ntk_factor=ntk_factor,
        )

        with torch.no_grad():
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_mask=prompt_attention_mask,
                image_rotary_emb=image_rotary_emb,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

        # Split and separate cond/uncond
        noise_pred_split = noise_pred.chunk(2, dim=1)[0]   # (2, 3, H, W)
        eps = noise_pred_split[:, :3]                       # (2, 3, H, W)
        noise_pred_rest = noise_pred_split[:, 3:]

        eps_cond_t, eps_uncond_t = torch.split(eps, len(eps) // 2, dim=0)
        # shapes: (1, 3, latent_h, latent_w)

        # Build SoftCFG mask at mask_step
        if i == mask_step:
            fg_mask_spatial = mask_builder.build_mask(
                eps_cond_t, eps_uncond_t
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, latent_h, latent_w)
            s_map = s_bg + fg_mask_spatial * (s_fg - s_bg)

        # Update caches
        cache_eps_cond = eps_cond_t.clone()
        cache_eps_uncond = eps_uncond_t.clone()
        cache_noise_pred_rest = noise_pred_rest[0:1].clone()  # (1, 1, H, W)

        # Key steps: always STANDARD uniform CFG (no spatial differentiation at key steps)
        # Differential CFG is applied ONLY at skip steps (zero extra NFE overhead)
        eps_half = eps_uncond_t + calib.cfg_scale * (eps_cond_t - eps_uncond_t)
        cache_eps_combined = eps_half.clone()

        # Update Taylor cache on combined eps (stable extrapolation target)
        taylor_cache_combined = _update_taylor_cache(taylor_cache_combined, cache_eps_combined, i, taylor_order)

        # Reconstruct noise_pred for scheduler
        eps_full = torch.cat([eps_half, eps_half], dim=0)
        noise_pred_out = torch.cat([eps_full, noise_pred_rest], dim=1)
        noise_pred_out, _ = noise_pred_out.chunk(2, dim=0)
        noise_pred_out = -noise_pred_out

        latents = pipe.scheduler.step(noise_pred_out, t, latents, return_dict=False)[0]

    # Decode
    latents = latents.detach() / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image.detach(), output_type="pil")[0]
    return image
