"""
src/aesplan/calibration_sd3.py
================================
AesPlan calibration for SD3: uses cfg_mag mask (ratio=1.54 on SD3)
instead of joint_attn mask (ratio=1.25).

cfg_mag = |eps_cond - eps_uncond| at mask_step — zero extra overhead
since eps are already captured during dense run.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

import os as _os
ACCELAES_ROOT = Path(_os.environ.get("ACCELAES_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent / "AccelAes")))
sys.path.insert(0, str(ACCELAES_ROOT))
sys.path.insert(0, str(ACCELAES_ROOT / "src"))

from .dp_solver import build_cost_table, solve_dp
from .calibration import CalibrationResult
from .dense_run_sd3 import run_dense_and_capture_sd3


def build_cfg_mag_mask_sd3(
    eps_cond: list,
    eps_uncond: list,
    mask_step: int,
    ratio: float = 0.5,
) -> torch.Tensor:
    """Build cfg_mag FG mask from SD3 dense run eps.

    Returns (1, 1, latent_h, latent_w) float mask.
    """
    ec = eps_cond[mask_step].squeeze(0)   # (16, H, W)
    eu = eps_uncond[mask_step].squeeze(0) # (16, H, W)
    magnitude = (ec - eu).abs().mean(dim=0)  # (H, W)
    threshold = magnitude.flatten().quantile(1.0 - ratio)
    mask = (magnitude >= threshold).float()
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)


class AesPlanCalibratorSD3:
    """Calibrate AesPlan for SD3: dense runs → cfg_mag mask → DP schedule.

    Args:
        wrapper:    SD3DiTWrapper
        budget:     key step count K (e.g. 10 out of 28)
        w_fg, w_bg: FG/BG cost weights
        cfg_scale:  SD3 guidance scale
        steps:      total denoising steps
        mask_step:  step to compute cfg_mag mask (always a key step)
        skip_ratio: fraction of pixels as FG (top-ratio by cfg_mag)
        first_free: always-compute warm-up steps (not in budget)
    """

    def __init__(
        self,
        wrapper,
        budget: int = 10,
        w_fg: float = 4.0,
        w_bg: float = 1.0,
        cfg_scale: float = 7.0,
        steps: int = 28,
        mask_step: int = 5,
        skip_ratio: float = 0.5,
        first_free: int = 6,
        max_skip: int = 4,
    ):
        self.wrapper = wrapper
        self.budget = budget
        self.w_fg = w_fg
        self.w_bg = w_bg
        self.cfg_scale = cfg_scale
        self.steps = steps
        self.mask_step = mask_step
        self.skip_ratio = skip_ratio
        self.first_free = first_free
        self.max_skip = max_skip

    def run(
        self,
        prompts: List[str],
        seeds: Optional[List[int]] = None,
    ) -> CalibrationResult:
        if seeds is None:
            seeds = [42] * len(prompts)

        cost_tables = []
        fg_bg_ratios = []

        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            print(f"[SD3 Calibration] Sample {i+1}/{len(prompts)}: {prompt[:50]}...")
            data = run_dense_and_capture_sd3(
                wrapper=self.wrapper,
                prompt=prompt,
                seed=seed,
                cfg_scale=self.cfg_scale,
                steps=self.steps,
                mask_step=self.mask_step,
                skip_ratio=self.skip_ratio,
            )

            # Use cfg_mag mask (better ratio than joint_attn)
            cfg_mask = build_cfg_mag_mask_sd3(
                data["eps_cond"], data["eps_uncond"], self.mask_step, self.skip_ratio
            )

            c = build_cost_table(
                eps_cond=data["eps_cond"],
                eps_uncond=data["eps_uncond"],
                fg_mask=cfg_mask,
                cfg_scale=self.cfg_scale,
                w_fg=self.w_fg,
                w_bg=self.w_bg,
                mask_step=self.mask_step,
            )
            cost_tables.append(c)

            ratio = self._fg_bg_ratio(data)
            fg_bg_ratios.append(ratio)
            print(f"  FG/BG ratio (d=2, cfg_mag): {ratio:.3f}")

        cost_avg = np.mean(cost_tables, axis=0)
        key_steps = solve_dp(cost_avg, budget=self.budget,
                             first_free=self.first_free, max_skip=self.max_skip)
        print(f"[SD3 Calibration] Budget={self.budget}/{self.steps}, key_steps={key_steps}")
        print(f"[SD3 Calibration] Speedup target: {self.steps/len(key_steps):.2f}×")
        print(f"[SD3 Calibration] Mean FG/BG ratio: {np.mean(fg_bg_ratios):.3f}")

        return CalibrationResult(
            key_steps=key_steps,
            cost_table=cost_avg,
            budget=self.budget,
            total_steps=self.steps,
            w_fg=self.w_fg,
            w_bg=self.w_bg,
            cfg_scale=self.cfg_scale,
            mask_step=self.mask_step,
            skip_ratio=self.skip_ratio,
            first_free=self.first_free,
            fg_bg_ratios=fg_bg_ratios,
        )

    def _fg_bg_ratio(self, data: dict, skip_d: int = 2) -> float:
        eps_cond = data["eps_cond"]
        eps_uncond = data["eps_uncond"]
        T = len(eps_cond)
        H, W = eps_cond[0].shape[-2], eps_cond[0].shape[-1]

        cfg_mask = build_cfg_mag_mask_sd3(eps_cond, eps_uncond, self.mask_step, self.skip_ratio)
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
