"""
src/aesplan/dual_track.py
==========================
Dual-Track Spatial Scheduling (AeSched) for SD3.

Core idea: FG and BG have different temporal sensitivities (ratio ρ = FG/BG ≈ 1.555).
Instead of one unified schedule (which degenerates to DPCache), we maintain two
independent refresh tracks:

  FG track: K_fg key steps, optimized to minimize FG skip cost only
  BG track: K_bg key steps, optimized to minimize BG skip cost only

Budget allocation based on optimal control (Lagrangian):
  τ_fg / τ_bg = 1/√ρ  →  K_fg / K_bg = √ρ

At inference:
  - full steps (FG ∩ BG): full forward, update both caches, standard output
  - fg_only steps (FG \ BG): full forward, output = M * ε_new + (1-M) * cache_bg
  - bg_only steps (BG \ FG): full forward, output = M * cache_fg + (1-M) * ε_new
  - skip steps (neither):    no forward,  output = M * cache_fg + (1-M) * cache_bg

This DIRECTLY exploits the FG/BG ratio:
  - Same total forward passes K as baseline
  - But FG is refreshed more often (K_fg > K_bg)
  - Schedule allocation ratio = √ρ (theoretically optimal)

Unlike DP-based methods (DPCache, our previous AesMask DP), the dual-track schedule
is NOT dominated by temporal structure alone — FG and BG DP solve different cost
functions and produce genuinely different schedules.
"""
from __future__ import annotations

import sys
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Dict

import numpy as np
import torch
import torch.nn.functional as F

import os as _os
ACCELAES_ROOT = Path(_os.environ.get("ACCELAES_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent / "AccelAes")))
sys.path.insert(0, str(ACCELAES_ROOT))
sys.path.insert(0, str(ACCELAES_ROOT / "src"))

from .dp_solver import solve_dp
from .dense_run_sd3 import run_dense_and_capture_sd3
from .calibration_sd3 import build_cfg_mag_mask_sd3


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DualTrackResult:
    """Result of dual-track calibration."""
    fg_schedule:    List[int]       # all steps that refresh FG cache (sorted)
    bg_schedule:    List[int]       # all steps that refresh BG cache (sorted)
    full_steps:     List[int]       # FG ∩ BG  (refresh both)
    fg_only_steps:  List[int]       # FG \ BG  (refresh FG, use cached BG)
    bg_only_steps:  List[int]       # BG \ FG  (refresh BG, use cached FG)
    fg_mask:        Optional[torch.Tensor]  # (1,1,H,W) float, cpu
    rho:            float           # measured FG/BG ratio
    K_fg:           int             # total FG key steps
    K_bg:           int             # total BG key steps
    K_total:        int             # total forward passes = |FG ∪ BG|
    total_steps:    int             # T = 28
    cfg_scale:      float
    mask_step:      int
    skip_ratio:     float
    first_free:     int
    fg_bg_ratios:   List[float]     # per-sample ratios from calibration

    def summary(self) -> str:
        lines = [
            f"DualTrack: T={self.total_steps}, K_fg={self.K_fg}, K_bg={self.K_bg}, "
            f"K_total={self.K_total} ({self.total_steps/self.K_total:.2f}× target)",
            f"  ρ={self.rho:.3f}  √ρ={math.sqrt(self.rho):.3f}",
            f"  FG schedule: {self.fg_schedule}",
            f"  BG schedule: {self.bg_schedule}",
            f"  full_steps:  {self.full_steps}",
            f"  fg_only:     {self.fg_only_steps}",
            f"  bg_only:     {self.bg_only_steps}",
        ]
        return "\n".join(lines)

    def save(self, path: str):
        import json
        d = {
            "fg_schedule": self.fg_schedule,
            "bg_schedule": self.bg_schedule,
            "full_steps": self.full_steps,
            "fg_only_steps": self.fg_only_steps,
            "bg_only_steps": self.bg_only_steps,
            "rho": self.rho,
            "K_fg": self.K_fg,
            "K_bg": self.K_bg,
            "K_total": self.K_total,
            "total_steps": self.total_steps,
            "cfg_scale": self.cfg_scale,
            "mask_step": self.mask_step,
            "skip_ratio": self.skip_ratio,
            "first_free": self.first_free,
            "fg_bg_ratios": self.fg_bg_ratios,
        }
        with open(path, "w") as f:
            import json; json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DualTrackResult":
        import json
        with open(path) as f:
            d = json.load(f)
        return cls(
            fg_schedule=d["fg_schedule"],
            bg_schedule=d["bg_schedule"],
            full_steps=d["full_steps"],
            fg_only_steps=d["fg_only_steps"],
            bg_only_steps=d["bg_only_steps"],
            fg_mask=None,   # not saved (too large); recompute at inference if needed
            rho=d["rho"],
            K_fg=d["K_fg"],
            K_bg=d["K_bg"],
            K_total=d["K_total"],
            total_steps=d["total_steps"],
            cfg_scale=d["cfg_scale"],
            mask_step=d["mask_step"],
            skip_ratio=d["skip_ratio"],
            first_free=d["first_free"],
            fg_bg_ratios=d["fg_bg_ratios"],
        )


# ---------------------------------------------------------------------------
# Cost builders (FG-only and BG-only)
# ---------------------------------------------------------------------------

def build_fg_cost_table(
    eps_cond: list,
    eps_uncond: list,
    fg_mask: torch.Tensor,
    cfg_scale: float = 7.0,
    mask_step: int = 5,
) -> np.ndarray:
    """FG-only cost table: cost[j,k] = mean(M * |eps(k) - eps(j)|)."""
    T = len(eps_cond)
    H, W = eps_cond[0].shape[-2], eps_cond[0].shape[-1]
    mask = F.interpolate(fg_mask, size=(H, W), mode="bilinear", align_corners=False)
    mask = mask.squeeze(0).squeeze(0)  # (H, W)

    eps_combined = []
    for t in range(T):
        ec = eps_cond[t].squeeze(0)
        eu = eps_uncond[t].squeeze(0)
        eps_combined.append(eu + cfg_scale * (ec - eu))

    cost = np.zeros((T, T), dtype=np.float32)
    for j in range(T):
        for k in range(j + 1, T):
            diff = (eps_combined[k] - eps_combined[j]).abs().mean(dim=0)  # (H,W)
            if k < mask_step:
                cost[j, k] = diff.mean().item()
            else:
                cost[j, k] = (mask * diff).sum().item() / mask.sum().clamp(min=1e-6).item()
    return cost


def build_bg_cost_table(
    eps_cond: list,
    eps_uncond: list,
    fg_mask: torch.Tensor,
    cfg_scale: float = 7.0,
    mask_step: int = 5,
) -> np.ndarray:
    """BG-only cost table: cost[j,k] = mean((1-M) * |eps(k) - eps(j)|)."""
    T = len(eps_cond)
    H, W = eps_cond[0].shape[-2], eps_cond[0].shape[-1]
    mask = F.interpolate(fg_mask, size=(H, W), mode="bilinear", align_corners=False)
    mask = mask.squeeze(0).squeeze(0)
    bg_mask = 1.0 - mask

    eps_combined = []
    for t in range(T):
        ec = eps_cond[t].squeeze(0)
        eu = eps_uncond[t].squeeze(0)
        eps_combined.append(eu + cfg_scale * (ec - eu))

    cost = np.zeros((T, T), dtype=np.float32)
    for j in range(T):
        for k in range(j + 1, T):
            diff = (eps_combined[k] - eps_combined[j]).abs().mean(dim=0)
            if k < mask_step:
                cost[j, k] = diff.mean().item()
            else:
                cost[j, k] = (bg_mask * diff).sum().item() / bg_mask.sum().clamp(min=1e-6).item()
    return cost


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------

def allocate_budget(
    K: int,
    first_free: int,
    rho: float,
) -> tuple[int, int]:
    """Compute K_fg_plan, K_bg_plan for dual-track.

    Optimal budget split (Lagrangian derivation):
      K_fg / K_bg = √ρ
    where K_fg and K_bg are the TOTAL refresh counts (including first_free).

    Solving:
      (first_free + K_fg_plan) / (first_free + K_bg_plan) = √ρ
      K_fg_plan + K_bg_plan = K - first_free

    Returns (K_fg_plan, K_bg_plan).
    """
    K_plan = K - first_free
    if K_plan <= 0:
        return 0, 0

    sqrt_rho = math.sqrt(rho)
    # Solve linear system:
    #   K_fg_plan - sqrt_rho * K_bg_plan = first_free * (sqrt_rho - 1)
    #   K_fg_plan + K_bg_plan = K_plan
    rhs = first_free * (sqrt_rho - 1)
    K_fg_plan = round((rhs + sqrt_rho * K_plan) / (1.0 + sqrt_rho))
    K_bg_plan = K_plan - K_fg_plan

    # Ensure at least 1 each
    K_fg_plan = max(1, K_fg_plan)
    K_bg_plan = max(1, K_bg_plan)

    # If rounding pushed total over K_plan, trim FG (FG gets priority)
    if K_fg_plan + K_bg_plan > K_plan:
        K_bg_plan = K_plan - K_fg_plan
        if K_bg_plan < 1:
            K_bg_plan = 1
            K_fg_plan = K_plan - 1

    return K_fg_plan, K_bg_plan


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class DualTrackCalibrator:
    """Calibrate dual-track schedules for SD3.

    Args:
        wrapper:    SD3DiTWrapper
        budget:     total forward passes K (same as single-track budget)
        cfg_scale:  guidance scale
        steps:      total denoising steps T
        mask_step:  step to compute cfg_mag mask
        skip_ratio: FG pixel fraction
        first_free: always-compute warm-up steps (in both tracks)
        max_skip_fg: max consecutive skips for FG track (smaller → more conservative)
        max_skip_bg: max consecutive skips for BG track (larger → more aggressive)
        rho_override: use this rho instead of measured (None = auto)
    """

    def __init__(
        self,
        wrapper,
        budget: int = 10,
        cfg_scale: float = 7.0,
        steps: int = 28,
        mask_step: int = 3,
        skip_ratio: float = 0.5,
        first_free: int = 4,   # KEY: must be small enough that K_plan = budget - first_free
        max_skip_fg: int = 5,  # ≥ ceil(T_plan/K_fg_plan) - 1 to avoid DP fallback
        max_skip_bg: int = 12, # BG is lenient; large max_skip lets BG DP use cost freely
        rho_override: Optional[float] = None,
    ):
        self.wrapper = wrapper
        self.budget = budget
        self.cfg_scale = cfg_scale
        self.steps = steps
        self.mask_step = mask_step
        self.skip_ratio = skip_ratio
        self.first_free = first_free
        self.max_skip_fg = max_skip_fg
        self.max_skip_bg = max_skip_bg
        self.rho_override = rho_override

    def run(
        self,
        prompts: List[str],
        seeds: Optional[List[int]] = None,
    ) -> DualTrackResult:
        if seeds is None:
            seeds = [42] * len(prompts)

        fg_cost_list, bg_cost_list = [], []
        fg_masks, fg_bg_ratios = [], []

        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            print(f"[DualTrack Calib] Sample {i+1}/{len(prompts)}: {prompt[:50]}...")
            data = run_dense_and_capture_sd3(
                wrapper=self.wrapper,
                prompt=prompt,
                seed=seed,
                cfg_scale=self.cfg_scale,
                steps=self.steps,
                mask_step=self.mask_step,
                skip_ratio=self.skip_ratio,
            )

            cfg_mask = build_cfg_mag_mask_sd3(
                data["eps_cond"], data["eps_uncond"],
                self.mask_step, self.skip_ratio,
            )
            fg_masks.append(cfg_mask)

            fg_c = build_fg_cost_table(
                data["eps_cond"], data["eps_uncond"],
                cfg_mask, self.cfg_scale, self.mask_step,
            )
            bg_c = build_bg_cost_table(
                data["eps_cond"], data["eps_uncond"],
                cfg_mask, self.cfg_scale, self.mask_step,
            )
            fg_cost_list.append(fg_c)
            bg_cost_list.append(bg_c)

            # Measure FG/BG ratio for budget allocation
            rho = self._fg_bg_ratio(data, cfg_mask)
            fg_bg_ratios.append(rho)
            print(f"  ρ (FG/BG ratio): {rho:.3f}")

        mean_rho = float(np.mean(fg_bg_ratios))
        rho = self.rho_override if self.rho_override is not None else mean_rho

        fg_cost_avg = np.mean(fg_cost_list, axis=0)
        bg_cost_avg = np.mean(bg_cost_list, axis=0)
        avg_fg_mask = torch.stack(fg_masks).mean(dim=0)  # (1,1,H,W)

        # Budget allocation
        K_fg_plan, K_bg_plan = allocate_budget(self.budget, self.first_free, rho)
        K_fg = self.first_free + K_fg_plan
        K_bg = self.first_free + K_bg_plan

        print(f"[DualTrack Calib] ρ={rho:.3f}, √ρ={math.sqrt(rho):.3f}")
        print(f"[DualTrack Calib] Budget K={self.budget}: K_fg={K_fg} (plan +{K_fg_plan}), "
              f"K_bg={K_bg} (plan +{K_bg_plan})")

        # Solve FG schedule (FG-only cost)
        fg_schedule = solve_dp(
            fg_cost_avg,
            budget=K_fg,
            first_free=self.first_free,
            max_skip=self.max_skip_fg,
        )

        # Solve BG schedule (BG-only cost)
        bg_schedule = solve_dp(
            bg_cost_avg,
            budget=K_bg,
            first_free=self.first_free,
            max_skip=self.max_skip_bg,
        )

        # Classify steps
        fg_set = set(fg_schedule)
        bg_set = set(bg_schedule)
        full_steps   = sorted(fg_set & bg_set)
        fg_only_steps = sorted(fg_set - bg_set)
        bg_only_steps = sorted(bg_set - fg_set)
        K_total = len(fg_set | bg_set)

        result = DualTrackResult(
            fg_schedule=sorted(fg_schedule),
            bg_schedule=sorted(bg_schedule),
            full_steps=full_steps,
            fg_only_steps=fg_only_steps,
            bg_only_steps=bg_only_steps,
            fg_mask=avg_fg_mask.cpu(),
            rho=rho,
            K_fg=K_fg,
            K_bg=K_bg,
            K_total=K_total,
            total_steps=self.steps,
            cfg_scale=self.cfg_scale,
            mask_step=self.mask_step,
            skip_ratio=self.skip_ratio,
            first_free=self.first_free,
            fg_bg_ratios=fg_bg_ratios,
        )

        print(result.summary())
        return result

    def _fg_bg_ratio(self, data: dict, cfg_mask: torch.Tensor, skip_d: int = 2) -> float:
        eps_cond = data["eps_cond"]
        eps_uncond = data["eps_uncond"]
        T = len(eps_cond)
        H, W = eps_cond[0].shape[-2], eps_cond[0].shape[-1]

        mask = F.interpolate(cfg_mask, size=(H, W), mode="bilinear", align_corners=False)
        mask = mask.squeeze(0).squeeze(0)

        fg_costs, bg_costs = [], []
        for t_idx in range(self.mask_step + skip_d, T):
            ec_t = eps_cond[t_idx].squeeze(0)
            eu_t = eps_uncond[t_idx].squeeze(0)
            ec_j = eps_cond[t_idx - skip_d].squeeze(0)
            eu_j = eps_uncond[t_idx - skip_d].squeeze(0)
            eps_d = eu_t + self.cfg_scale * (ec_t - eu_t)
            eps_c = eu_j + self.cfg_scale * (ec_j - eu_j)
            diff = (eps_d - eps_c).abs().mean(dim=0)
            fg_costs.append((mask * diff).sum() / mask.sum().clamp(min=1e-6))
            bg_costs.append(((1 - mask) * diff).sum() / (1 - mask).sum().clamp(min=1e-6))

        fg_mean = float(torch.stack(fg_costs).mean())
        bg_mean = float(torch.stack(bg_costs).mean())
        return fg_mean / bg_mean if bg_mean > 1e-8 else 1.0


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_dual_track_sd3(
    wrapper,
    prompt: str,
    seed: int,
    dt: DualTrackResult,
    height: int = 1024,
    width: int = 1024,
) -> "PIL.Image":
    """Dual-track inference on SD3.

    Step types and their behavior:
      full_steps   (FG ∩ BG): full forward → standard CFG → update both caches
      fg_only_steps (FG \ BG): full forward → M * eps_new + (1-M) * cache_bg
      bg_only_steps (BG \ FG): full forward → M * cache_fg + (1-M) * eps_new
      skip_steps    (neither): no forward  → M * cache_fg + (1-M) * cache_bg
    """
    pipe    = wrapper.pipe
    device  = wrapper.device
    dtype   = wrapper.dtype
    T       = dt.total_steps

    full_set    = set(dt.full_steps)
    fg_only_set = set(dt.fg_only_steps)
    bg_only_set = set(dt.bg_only_steps)

    # --- Encode prompt ---
    (
        prompt_embeds, negative_prompt_embeds,
        pooled_prompt_embeds, negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        negative_prompt="",
        do_classifier_free_guidance=True,
        num_images_per_prompt=1, device=device,
    )
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

    # --- FG mask on device ---
    if dt.fg_mask is not None:
        s_mask = F.interpolate(
            dt.fg_mask, size=(latent_h, latent_w),
            mode="bilinear", align_corners=False,
        ).to(device=device, dtype=dtype)   # (1,1,H,W), values in {0,1}
    else:
        # Recompute mask at mask_step (will be set at that step)
        s_mask = None

    # --- Caches ---
    cache_fg: Optional[torch.Tensor] = None   # guided eps from last FG refresh
    cache_bg: Optional[torch.Tensor] = None   # guided eps from last BG refresh

    for i, t in enumerate(timesteps):
        if i in full_set or i in fg_only_set or i in bg_only_set:
            # --- Run transformer ---
            latent_model_input = torch.cat([latents] * 2, dim=0)
            with torch.no_grad():
                noise_pred = wrapper.transformer(
                    hidden_states=latent_model_input,
                    timestep=t.expand(latent_model_input.shape[0]),
                    encoder_hidden_states=prompt_embeds_cfg,
                    pooled_projections=pooled_cfg,
                    return_dict=False,
                )[0]
            eps_uncond, eps_cond = noise_pred.chunk(2)

            # Build mask at mask_step if not yet available
            if i == dt.mask_step and s_mask is None:
                from .calibration_sd3 import build_cfg_mag_mask_sd3
                ec_wrap = [None] * (dt.mask_step + 1)
                eu_wrap = [None] * (dt.mask_step + 1)
                ec_wrap[dt.mask_step] = eps_cond.cpu().float()
                eu_wrap[dt.mask_step] = eps_uncond.cpu().float()
                s_mask = F.interpolate(
                    build_cfg_mag_mask_sd3(ec_wrap, eu_wrap, dt.mask_step, dt.skip_ratio),
                    size=(latent_h, latent_w), mode="bilinear", align_corners=False,
                ).to(device=device, dtype=dtype)

            guided_new = eps_uncond + dt.cfg_scale * (eps_cond - eps_uncond)

            if i in full_set:
                # Both tracks refreshed
                cache_fg = guided_new.clone()
                cache_bg = guided_new.clone()
                guided = guided_new

            elif i in fg_only_set:
                # FG refreshed, BG uses cache
                cache_fg = guided_new.clone()
                if cache_bg is not None and s_mask is not None:
                    guided = s_mask * guided_new + (1 - s_mask) * cache_bg
                else:
                    guided = guided_new  # fallback before first BG refresh

            else:  # bg_only
                # BG refreshed, FG uses cache
                cache_bg = guided_new.clone()
                if cache_fg is not None and s_mask is not None:
                    guided = s_mask * cache_fg + (1 - s_mask) * guided_new
                else:
                    guided = guided_new  # fallback before first FG refresh

        else:
            # --- Skip step ---
            if cache_fg is not None and cache_bg is not None and s_mask is not None:
                guided = s_mask * cache_fg + (1 - s_mask) * cache_bg
            elif cache_fg is not None:
                guided = cache_fg
            else:
                continue  # no cache yet (shouldn't happen with first_free > 0)

        latents = pipe.scheduler.step(guided, t, latents, return_dict=False)[0]

    # --- Decode ---
    latents = (latents / wrapper.vae.config.scaling_factor) + wrapper.vae.config.shift_factor
    with torch.no_grad():
        image = wrapper.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]
