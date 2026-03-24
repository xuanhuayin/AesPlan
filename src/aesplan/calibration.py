"""
src/aesplan/calibration.py
===========================
AesPlan calibration: build AesMask-weighted cost table from N dense runs,
average across samples, run DP to get optimal key step schedule.

Usage:
    calibrator = AesPlanCalibrator(wrapper, budget=12, w_fg=4.0, w_bg=1.0)
    result = calibrator.run(prompts, seeds)
    # result.key_steps: list of key step indices
    # result.cost_table: (T, T) averaged cost
    # result.fg_mask_params: mask builder config for inference

Saves result to a JSON file for reuse across inference runs.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

import os as _os
ACCELAES_ROOT = Path(_os.environ.get("ACCELAES_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent / "AccelAes")))
sys.path.insert(0, str(ACCELAES_ROOT))
sys.path.insert(0, str(ACCELAES_ROOT / "src"))

from .dp_solver import build_cost_table, solve_dp
from .dense_run import run_dense_and_capture


@dataclass
class CalibrationResult:
    key_steps: List[int]
    cost_table: np.ndarray          # (T, T) averaged over samples
    budget: int
    total_steps: int
    w_fg: float
    w_bg: float
    cfg_scale: float
    mask_step: int
    skip_ratio: float
    first_free: int = 6             # warm-up steps always computed
    # Per-sample FG/BG cost ratios for diagnostics
    fg_bg_ratios: List[float] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path.with_suffix(".cost.npy")), self.cost_table)
        meta = {
            "key_steps": self.key_steps,
            "budget": self.budget,
            "total_steps": self.total_steps,
            "w_fg": self.w_fg,
            "w_bg": self.w_bg,
            "cfg_scale": self.cfg_scale,
            "mask_step": self.mask_step,
            "skip_ratio": self.skip_ratio,
            "first_free": self.first_free,
            "fg_bg_ratios": self.fg_bg_ratios,
            "cost_table_file": str(path.with_suffix(".cost.npy")),
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[Calibration] Saved to {path.with_suffix('.json')}")

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationResult":
        path = Path(path)
        with open(path.with_suffix(".json")) as f:
            meta = json.load(f)
        cost = np.load(meta["cost_table_file"])
        return cls(
            key_steps=meta["key_steps"],
            cost_table=cost,
            budget=meta["budget"],
            total_steps=meta["total_steps"],
            w_fg=meta["w_fg"],
            w_bg=meta["w_bg"],
            cfg_scale=meta["cfg_scale"],
            mask_step=meta["mask_step"],
            skip_ratio=meta["skip_ratio"],
            first_free=meta.get("first_free", 6),
            fg_bg_ratios=meta.get("fg_bg_ratios", []),
        )


class AesPlanCalibrator:
    """Run dense generation on calibration samples, build cost table, solve DP.

    Args:
        wrapper: LuminaDiTWrapper instance
        budget: number of key steps K (e.g. 10 out of 30 = 3× speedup target)
        w_fg: FG cost weight (default 4.0)
        w_bg: BG cost weight (default 1.0)
        cfg_scale: uniform CFG for cost computation
        steps: total denoising steps
        mask_step: step to extract AesMask
        skip_ratio: FG ratio for AesMask (0.5 = top 50% as FG)
        first_free: always-compute warm-up steps (not counted in budget)
    """

    def __init__(
        self,
        wrapper,
        budget: int = 10,
        w_fg: float = 4.0,
        w_bg: float = 1.0,
        cfg_scale: float = 4.0,
        steps: int = 30,
        mask_step: int = 5,
        skip_ratio: float = 0.5,
        first_free: int = 6,   # steps 0-5 always computed (mask not yet built)
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

    def run(
        self,
        prompts: List[str],
        seeds: Optional[List[int]] = None,
    ) -> CalibrationResult:
        """Run calibration on given prompts, return schedule + cost table."""
        if seeds is None:
            seeds = [42] * len(prompts)

        cost_tables = []
        fg_bg_ratios = []

        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            print(f"[Calibration] Sample {i+1}/{len(prompts)}: {prompt[:50]}...")
            data = run_dense_and_capture(
                wrapper=self.wrapper,
                prompt=prompt,
                seed=seed,
                cfg_scale=self.cfg_scale,
                steps=self.steps,
                mask_step=self.mask_step,
                skip_ratio=self.skip_ratio,
            )

            c = build_cost_table(
                eps_cond=data["eps_cond"],
                eps_uncond=data["eps_uncond"],
                fg_mask=data["fg_mask"],
                cfg_scale=self.cfg_scale,
                w_fg=self.w_fg,
                w_bg=self.w_bg,
                mask_step=self.mask_step,
            )
            cost_tables.append(c)

            # Compute FG/BG ratio for d=2 as diagnostic
            ratio = self._compute_fg_bg_ratio(data, skip_d=2)
            fg_bg_ratios.append(ratio)
            print(f"  FG/BG ratio (d=2): {ratio:.3f}")

        # Average cost table across samples
        cost_avg = np.mean(cost_tables, axis=0)

        # Solve DP
        key_steps = solve_dp(cost_avg, budget=self.budget, first_free=self.first_free)
        print(f"[Calibration] Budget={self.budget}/{self.steps}, "
              f"key_steps={key_steps} ({len(key_steps)} steps)")
        print(f"[Calibration] Expected speedup: {self.steps/len(key_steps):.2f}×")
        print(f"[Calibration] Mean FG/BG ratio: {np.mean(fg_bg_ratios):.3f}")

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
            fg_bg_ratios=fg_bg_ratios,
        )

    def _compute_fg_bg_ratio(self, data: dict, skip_d: int = 2) -> float:
        """Quick diagnostic: FG/BG cost ratio for a given skip interval."""
        import torch
        import torch.nn.functional as F

        eps_cond = data["eps_cond"]
        eps_uncond = data["eps_uncond"]
        fg_mask = data["fg_mask"]
        T = len(eps_cond)
        H, W = eps_cond[0].shape[-2], eps_cond[0].shape[-1]

        mask = F.interpolate(fg_mask, size=(H, W), mode="bilinear", align_corners=False)
        mask = mask.squeeze(0).squeeze(0)

        fg_costs, bg_costs = [], []
        for t in range(self.mask_step + skip_d, T):
            ec_t = eps_cond[t].squeeze(0)
            eu_t = eps_uncond[t].squeeze(0)
            ec_j = eps_cond[t - skip_d].squeeze(0)
            eu_j = eps_uncond[t - skip_d].squeeze(0)

            eps_d = eu_t + self.cfg_scale * (ec_t - eu_t)
            eps_c = eu_j + self.cfg_scale * (ec_j - eu_j)
            diff = (eps_d - eps_c).abs().mean(dim=0)

            fg_costs.append((mask * diff).sum() / mask.sum().clamp(min=1e-6))
            bg_costs.append(((1 - mask) * diff).sum() / (1 - mask).sum().clamp(min=1e-6))

        fg_mean = float(torch.stack(fg_costs).mean())
        bg_mean = float(torch.stack(bg_costs).mean())
        return fg_mean / bg_mean if bg_mean > 1e-8 else 1.0
