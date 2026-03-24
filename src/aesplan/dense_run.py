"""
src/aesplan/dense_run.py
=========================
Dense generation with eps_cond / eps_uncond capture at every step.
Used by both E0 validation and AesPlan calibration.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

import os as _os
ACCELAES_ROOT = Path(_os.environ.get("ACCELAES_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent / "AccelAes")))
sys.path.insert(0, str(ACCELAES_ROOT))
sys.path.insert(0, str(ACCELAES_ROOT / "src"))

from src.models.dit_wrapper import get_2d_rotary_pos_embed_lumina
from src.sparse.sdit_mask_builder import CFGMagnitudeMaskBuilder


def run_dense_and_capture(
    wrapper,
    prompt: str,
    seed: int,
    cfg_scale: float = 4.0,
    steps: int = 30,
    mask_step: int = 5,
    skip_ratio: float = 0.5,
    height: int = 1024,
    width: int = 1024,
) -> Dict:
    """Run dense Lumina generation capturing eps_cond/eps_uncond at every step.

    Returns:
        eps_cond:  list[T] of (1, 3, H, W) cpu float tensors
        eps_uncond: list[T] of (1, 3, H, W) cpu float tensors
        fg_mask:   (1, 1, H, W) cpu float tensor in [0,1], 1=FG
    """
    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

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

    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps

    mask_builder = CFGMagnitudeMaskBuilder(ratio=skip_ratio)
    head_dim = wrapper.transformer.head_dim

    eps_cond_list: List[torch.Tensor] = []
    eps_uncond_list: List[torch.Tensor] = []
    fg_mask_spatial: Optional[torch.Tensor] = None

    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2, dim=0)

        current_timestep = t
        if not torch.is_tensor(current_timestep):
            current_timestep = torch.tensor([current_timestep], dtype=torch.float64, device=device)
        elif len(current_timestep.shape) == 0:
            current_timestep = current_timestep[None].to(device)
        current_timestep = current_timestep.expand(latent_model_input.shape[0])
        current_timestep = 1 - current_timestep / pipe.scheduler.config.num_train_timesteps

        linear_factor = scaling_factor if current_timestep[0] < 1.0 else 1.0
        ntk_factor = 1.0 if current_timestep[0] < 1.0 else scaling_factor

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

        noise_pred_split = noise_pred.chunk(2, dim=1)[0]
        noise_pred_eps = noise_pred_split[:, :3]
        eps_cond_t, eps_uncond_t = torch.split(noise_pred_eps, len(noise_pred_eps) // 2, dim=0)

        # Build SoftCFG mask at mask_step using cond/uncond eps
        if i == mask_step:
            fg_mask_spatial = mask_builder.build_mask(
                eps_cond_t, eps_uncond_t
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        eps_cond_list.append(eps_cond_t.cpu().float())
        eps_uncond_list.append(eps_uncond_t.cpu().float())

        eps_combined = eps_uncond_t + cfg_scale * (eps_cond_t - eps_uncond_t)
        eps_full = torch.cat([eps_combined, eps_combined], dim=0)
        noise_pred_out = torch.cat([eps_full, noise_pred_split[:, 3:]], dim=1)
        noise_pred_out, _ = noise_pred_out.chunk(2, dim=0)
        noise_pred_out = -noise_pred_out
        latents = pipe.scheduler.step(noise_pred_out, t, latents, return_dict=False)[0]

    if fg_mask_spatial is None:
        raise RuntimeError("fg_mask_spatial was never set — mask_step out of range?")

    return {
        "eps_cond": eps_cond_list,
        "eps_uncond": eps_uncond_list,
        "fg_mask": fg_mask_spatial.cpu().float(),
    }
