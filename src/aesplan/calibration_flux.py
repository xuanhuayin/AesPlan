"""
src/aesplan/calibration_flux.py
================================
AesPlan calibration for FLUX: guidance_diff mask + AesMask-weighted DP.

Key difference from SD3:
  - FLUX is single-pass (guidance distillation), so no differential CFG at skip steps.
  - At mask_step: one extra forward(g=1.0) to get guidance_diff → FG mask.
  - Skip steps: reuse last cached noise_pred (same as DPCache, but with semantic schedule).
  - This enables a direct comparison: semantic-aware DP vs content-agnostic DP (DPCache).
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

from .dp_solver import build_cost_table_flux, solve_dp
from .calibration import CalibrationResult
from .dense_run_flux import run_dense_and_capture_flux


class AesPlanCalibratorFlux:
    """Calibrate AesPlan for FLUX: dense runs → guidance_diff mask → DP schedule.

    Args:
        wrapper:    FluxDiTWrapper
        budget:     key step count K (e.g. 6 out of 28)
        w_fg, w_bg: FG/BG cost weights
        guidance_scale: FLUX guidance scale
        steps:      total denoising steps
        mask_step:  step to compute guidance_diff mask
        skip_ratio: fraction of pixels as FG (top-ratio by guidance_diff)
        first_free: always-compute warm-up steps (not in budget)
        max_skip:   max consecutive skips
    """

    def __init__(
        self,
        wrapper,
        budget: int = 6,
        w_fg: float = 4.0,
        w_bg: float = 1.0,
        guidance_scale: float = 3.5,
        steps: int = 28,
        mask_step: int = 3,
        skip_ratio: float = 0.5,
        first_free: int = 4,
        max_skip: int = 6,
    ):
        self.wrapper = wrapper
        self.budget = budget
        self.w_fg = w_fg
        self.w_bg = w_bg
        self.guidance_scale = guidance_scale
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
            print(f"[FLUX Calibration] Sample {i+1}/{len(prompts)}: {prompt[:50]}...")
            data = run_dense_and_capture_flux(
                wrapper=self.wrapper,
                prompt=prompt,
                seed=seed,
                guidance_scale=self.guidance_scale,
                steps=self.steps,
                mask_step=self.mask_step,
                skip_ratio=self.skip_ratio,
            )

            fg_mask = data["guidance_diff_mask"]  # (1, 1, latent_h, latent_w)
            latent_h = data["latent_h"]
            latent_w = data["latent_w"]

            c = build_cost_table_flux(
                noise_preds=data["noise_preds"],
                fg_mask=fg_mask,
                latent_h=latent_h,
                latent_w=latent_w,
                w_fg=self.w_fg,
                w_bg=self.w_bg,
                mask_step=self.mask_step,
            )
            cost_tables.append(c)

            ratio = self._fg_bg_ratio(data)
            fg_bg_ratios.append(ratio)
            print(f"  FG/BG ratio (d=2, guidance_diff): {ratio:.3f}")

        cost_avg = np.mean(cost_tables, axis=0)
        key_steps = solve_dp(
            cost_avg,
            budget=self.budget,
            first_free=self.first_free,
            max_skip=self.max_skip,
        )
        print(f"[FLUX Calibration] Budget={self.budget}/{self.steps}, key_steps={key_steps}")
        print(f"[FLUX Calibration] Speedup target: {self.steps/len(key_steps):.2f}×")
        print(f"[FLUX Calibration] Mean FG/BG ratio: {np.mean(fg_bg_ratios):.3f}")

        return CalibrationResult(
            key_steps=key_steps,
            cost_table=cost_avg,
            budget=self.budget,
            total_steps=self.steps,
            w_fg=self.w_fg,
            w_bg=self.w_bg,
            cfg_scale=self.guidance_scale,
            mask_step=self.mask_step,
            skip_ratio=self.skip_ratio,
            first_free=self.first_free,
            fg_bg_ratios=fg_bg_ratios,
        )

    def _fg_bg_ratio(self, data: dict, skip_d: int = 2) -> float:
        """Compute FG/BG skip cost ratio from noise_preds."""
        noise_preds = data["noise_preds"]
        fg_mask = data["guidance_diff_mask"]  # (1, 1, latent_h, latent_w)
        latent_h = data["latent_h"]
        latent_w = data["latent_w"]
        T = len(noise_preds)

        tok_h = latent_h // 2
        tok_w = latent_w // 2

        # Downsample mask to token grid for cost comparison
        mask_tok = F.interpolate(fg_mask, size=(tok_h, tok_w),
                                 mode="bilinear", align_corners=False)
        mask_tok = mask_tok.squeeze(0).squeeze(0)  # (tok_h, tok_w)

        fg_costs, bg_costs = [], []
        for t_idx in range(self.mask_step + skip_d, T):
            pred_t = noise_preds[t_idx].squeeze(0)   # (seq_len, 64)
            pred_j = noise_preds[t_idx - skip_d].squeeze(0)
            diff = (pred_t - pred_j).abs().mean(dim=-1)  # (seq_len,)
            diff_map = diff.view(tok_h, tok_w)            # (tok_h, tok_w)

            fg_costs.append((mask_tok * diff_map).sum() /
                            mask_tok.sum().clamp(min=1e-6))
            bg_costs.append(((1 - mask_tok) * diff_map).sum() /
                            (1 - mask_tok).sum().clamp(min=1e-6))

        fg_mean = float(torch.stack(fg_costs).mean())
        bg_mean = float(torch.stack(bg_costs).mean())
        return fg_mean / bg_mean if bg_mean > 1e-8 else 1.0
