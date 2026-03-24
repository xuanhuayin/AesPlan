"""
src/aesplan/dp_solver.py
========================
Dynamic Programming scheduler for AesPlan.

Given a 2D cost table cost[j, k] = cost of reusing step-j's cached prediction
for step k (j < k), find the optimal set of K "key steps" that minimises total
skip cost within budget K.

The DP formulation:
  - State: (current step t, keys used b, last key step lk)
  - Transition: either compute step t (key step, cost=0) or skip it (cost=cost[lk, t])
  - Objective: minimise total skip cost, with exactly K key steps total

We use a 3D DP:
  dp[t][b][lk] = min total cost to handle planning steps 0..t,
                 having used b key steps, with last key at original index lk.

This correctly handles path-dependent feasibility: the 2D formulation (storing
only one last_key per (t,b) state) fails when a cheaper path has an earlier
last_key that blocks future skip steps under the max_skip constraint.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple


def solve_dp(
    cost: np.ndarray,
    budget: int,
    first_free: int = 0,
    max_skip: int = 4,
) -> List[int]:
    """Find optimal key step schedule via DP.

    Uses a 3D DP (t, keys_used, last_key) to correctly handle the max_skip
    constraint without path-dependency issues.

    Args:
        cost: (T, T) float array. cost[j, k] = skip cost when reusing step j
              to approximate step k (j < k). cost[j, k] = 0 if j >= k.
        budget: number of key steps K to select.
        first_free: steps 0..first_free-1 are always key steps (warm-up).
                    These don't count against the budget.
        max_skip: maximum gap from last key step to any skip step.
                  I.e. orig(skip_step) - orig(last_key) <= max_skip.
                  Default=4. Use a large value (e.g. T) to remove constraint.

    Returns:
        key_steps: sorted list of selected step indices (length = budget).
                   Steps 0..first_free-1 are always included.
    """
    T = cost.shape[0]
    assert budget <= T, f"budget {budget} > T {T}"

    forced = list(range(first_free))
    T_plan = T - first_free
    K_plan = budget - first_free

    if K_plan <= 0:
        return forced

    INF = float("inf")
    T_p = T_plan
    K_p = K_plan

    def orig(t_plan: int) -> int:
        return t_plan + first_free

    def skip_cost_fn(lk: int, k: int) -> float:
        if lk < 0:
            return INF
        return float(cost[lk, k])

    # 3D DP: dp[t, b, lk] = min cost to handle planning steps 0..t
    #        with b key steps used and last key at original index lk.
    # t in [0, T_p-1], b in [0, K_p], lk in [0, T-1]
    dp = np.full((T_p, K_p + 1, T), INF, dtype=np.float64)

    # Base case: t=0 (planning step 0 = original step first_free)
    o0 = orig(0)
    # Option A: key at t=0
    dp[0, 1, o0] = 0.0
    # Option B: skip t=0 using last forced step (only if first_free > 0)
    if first_free > 0:
        lk_prev = first_free - 1
        if (o0 - lk_prev) <= max_skip:
            c = skip_cost_fn(lk_prev, o0)
            if c < INF:
                dp[0, 0, lk_prev] = c

    # Fill DP
    for t in range(1, T_p):
        o = orig(t)
        for b in range(K_p + 1):
            # Option A: key at t (new last_key = o).
            # Cost = min over all prev_lk of dp[t-1, b-1, prev_lk].
            if b >= 1:
                best_prev = float(dp[t - 1, b - 1].min())
                if best_prev < INF and best_prev < dp[t, b, o]:
                    dp[t, b, o] = best_prev

            # Option B: skip at t, reusing last_key=lk (unchanged).
            # For each feasible lk, propagate dp[t-1, b, lk] → dp[t, b, lk].
            for lk in range(T):
                prev = dp[t - 1, b, lk]
                if prev < INF and (o - lk) <= max_skip:
                    c_b = prev + skip_cost_fn(lk, o)
                    if c_b < dp[t, b, lk]:
                        dp[t, b, lk] = c_b

    # Best final state: (T_p-1, K_p, any lk)
    final_row = dp[T_p - 1, K_p]
    if final_row.min() == INF:
        import warnings
        min_budget_needed = -(-T_plan // (max_skip + 1))
        warnings.warn(
            f"[solve_dp] DP infeasible → FALLBACK to uniform spacing! "
            f"Cost function is IGNORED. "
            f"Minimum budget needed: {min_budget_needed + first_free} "
            f"(current: budget={budget}, first_free={first_free}, "
            f"max_skip={max_skip}, T_plan={T_plan}, K_plan={K_plan}).",
            stacklevel=2,
        )
        step = max(1, T_p // K_p)
        planned = sorted(set(range(0, T_p, step)[:K_p]))
        return sorted(forced + [orig(t) for t in planned])

    # Backtrack.
    # Invariant: if cur_lk == orig(t), step t was a key step.
    # (A skip step at t keeps lk from the previous state unchanged;
    #  a key step at t sets lk = orig(t), which is unique among all t.)
    planned_keys = []
    cur_lk = int(np.argmin(final_row))
    b = K_p

    for t in range(T_p - 1, -1, -1):
        if cur_lk == orig(t):
            # Step t was a key step
            planned_keys.append(orig(t))
            b -= 1
            if b >= 0 and t > 0:
                # Previous state: (t-1, b, prev_lk) — find prev_lk
                cur_lk = int(np.argmin(dp[t - 1, b]))
            else:
                break
        # else: step t was a skip, cur_lk unchanged

    planned_keys.reverse()
    return sorted(forced + planned_keys)


def build_cost_table(
    eps_cond: list,
    eps_uncond: list,
    fg_mask: "torch.Tensor",
    cfg_scale: float = 4.0,
    w_fg: float = 4.0,
    w_bg: float = 1.0,
    mask_step: int = 5,
) -> np.ndarray:
    """Build the 2D AesMask-weighted cost table from one dense run.

    Args:
        eps_cond:  list[T] of (1, 3, H, W) cpu float tensors
        eps_uncond: list[T] of (1, 3, H, W) cpu float tensors
        fg_mask:  (1, 1, H, W) cpu float tensor in [0, 1]
        cfg_scale: uniform guidance scale for combining ε
        w_fg, w_bg: FG/BG loss weights
        mask_step: steps before this are computed without AesMask

    Returns:
        cost: (T, T) float32 ndarray. cost[j, k] = AesMask-weighted L1 error
              when reusing step j's prediction for step k.
              Upper triangle is 0 (j >= k makes no sense).
    """
    import torch
    import torch.nn.functional as F

    T = len(eps_cond)
    H, W = eps_cond[0].shape[-2], eps_cond[0].shape[-1]

    mask = F.interpolate(fg_mask, size=(H, W), mode="bilinear", align_corners=False)
    mask = mask.squeeze(0).squeeze(0)  # (H, W)

    # Pre-compute combined ε
    eps_combined = []
    for t in range(T):
        ec = eps_cond[t].squeeze(0)
        eu = eps_uncond[t].squeeze(0)
        eps_combined.append(eu + cfg_scale * (ec - eu))  # (3, H, W)

    cost = np.zeros((T, T), dtype=np.float32)

    for j in range(T):
        for k in range(j + 1, T):
            if k < mask_step:
                # No mask available yet, use uniform L1
                diff = (eps_combined[k] - eps_combined[j]).abs().mean().item()
                cost[j, k] = diff
            else:
                diff = (eps_combined[k] - eps_combined[j]).abs()  # (3, H, W)
                diff_m = diff.mean(dim=0)  # (H, W)
                fg_c = (mask * diff_m).sum() / mask.sum().clamp(min=1e-6)
                bg_c = ((1 - mask) * diff_m).sum() / (1 - mask).sum().clamp(min=1e-6)
                cost[j, k] = (w_fg * fg_c + w_bg * bg_c).item()

    return cost


def build_cost_table_flux(
    noise_preds: list,
    fg_mask: "torch.Tensor",
    latent_h: int,
    latent_w: int,
    w_fg: float = 4.0,
    w_bg: float = 1.0,
    mask_step: int = 3,
) -> np.ndarray:
    """Build AesMask-weighted cost table for FLUX.

    FLUX noise_preds are packed: (1, seq_len, 64) where seq_len=(latent_h//2)*(latent_w//2).

    Args:
        noise_preds: list[T] of (1, seq_len, 64) cpu float tensors
        fg_mask:     (1, 1, latent_h, latent_w) cpu float tensor
        latent_h, latent_w: latent spatial dimensions (before FLUX 2x2 pack)
        w_fg, w_bg:  FG/BG cost weights
        mask_step:   steps before this use uniform cost (mask not yet computed)

    Returns:
        cost: (T, T) float32 ndarray
    """
    import torch
    import torch.nn.functional as F

    T = len(noise_preds)
    tok_h = latent_h // 2
    tok_w = latent_w // 2

    # Downsample mask to token grid
    mask_tok = F.interpolate(fg_mask, size=(tok_h, tok_w),
                             mode="bilinear", align_corners=False)
    mask_tok = mask_tok.squeeze(0).squeeze(0)  # (tok_h, tok_w)

    cost = np.zeros((T, T), dtype=np.float32)

    for j in range(T):
        for k in range(j + 1, T):
            pred_j = noise_preds[j].squeeze(0)   # (seq_len, 64)
            pred_k = noise_preds[k].squeeze(0)
            diff_per_tok = (pred_k - pred_j).abs().mean(dim=-1)  # (seq_len,)
            diff_map = diff_per_tok.view(tok_h, tok_w)            # (tok_h, tok_w)

            if k < mask_step:
                cost[j, k] = float(diff_map.mean())
            else:
                fg_c = (mask_tok * diff_map).sum() / mask_tok.sum().clamp(min=1e-6)
                bg_c = ((1 - mask_tok) * diff_map).sum() / (1 - mask_tok).sum().clamp(min=1e-6)
                cost[j, k] = float(w_fg * fg_c + w_bg * bg_c)

    return cost
