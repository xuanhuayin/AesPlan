"""
src/aesplan/inference_flux.py
==============================
AesPlan inference for FLUX.

Key steps: full single-pass forward (g=3.5), update Taylor cache.
Skip steps: Taylor order-2 extrapolation (same as DPCache) or simple cache reuse.

taylor_order:
  0 = simple cache reuse (old default, for ablation)
  1 = linear extrapolation
  2 = quadratic Taylor (matches DPCache's Taylor-DP, recommended)

This enables a clean comparison:
  - Our method: AesMask-weighted DP schedule (semantic-aware) + Taylor-2
  - DPCache:    content-agnostic PACT schedule + Taylor-2
"""
from __future__ import annotations

from typing import List, Optional, Set

import torch

from .calibration import CalibrationResult


# ---------------------------------------------------------------------------
# Taylor cache helpers
# ---------------------------------------------------------------------------

def _update_taylor_cache(
    taylor_cache: dict,
    noise_pred: torch.Tensor,
    step_idx: int,
    taylor_order: int,
) -> dict:
    """Update Taylor derivative cache at a key step.

    Computes finite-difference derivatives matching DPCache's
    derivative_approximation() approach.

    Args:
        taylor_cache: current cache (may be empty on first call)
        noise_pred:   noise prediction at the current key step
        step_idx:     integer step index (0-based)
        taylor_order: max derivative order (0=cache reuse, 1=linear, 2=quadratic)

    Returns:
        new_cache dict with keys 0, 1, 2 (derivatives) and 'last_key' (step index).
    """
    new_cache: dict = {"last_key": step_idx, 0: noise_pred.clone()}

    if "last_key" not in taylor_cache or taylor_order < 1:
        return new_cache

    dist = step_idx - taylor_cache["last_key"]
    if dist <= 0:
        return new_cache

    # 1st-order finite difference: d1 = (f_current - f_prev) / dist
    d1 = (noise_pred - taylor_cache[0]) / dist
    new_cache[1] = d1

    # 2nd-order: d2 = (d1_current - d1_prev) / dist
    if taylor_order >= 2 and 1 in taylor_cache:
        d2 = (d1 - taylor_cache[1]) / dist
        new_cache[2] = d2

    return new_cache


def _taylor_predict(
    taylor_cache: dict,
    step_idx: int,
    taylor_order: int,
) -> torch.Tensor:
    """Predict noise_pred at skip step using Taylor extrapolation.

    Taylor formula (expanded at last key step):
      pred ≈ f0 + f1*d + f2/2*d^2
    where d = step_idx - last_key_step, fi = i-th derivative.

    Computes in fp32 to avoid fp16 overflow (same as DPCache clip_fp16).
    """
    d = float(step_idx - taylor_cache["last_key"])
    orig_dtype = taylor_cache[0].dtype

    # Compute in fp32 to avoid overflow
    pred = taylor_cache[0].float()

    if taylor_order >= 1 and 1 in taylor_cache:
        pred = pred + taylor_cache[1].float() * d

    if taylor_order >= 2 and 2 in taylor_cache:
        pred = pred + taylor_cache[2].float() * (d * d / 2.0)

    return pred.to(orig_dtype)


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def generate_aesplan_flux(
    wrapper,
    prompt: str,
    seed: int,
    calib: CalibrationResult,
    height: int = 1024,
    width: int = 1024,
    # Ablation: use_aes_cost=False → uniform spacing (DPCache-equivalent)
    use_aes_cost: bool = True,
    key_steps_override: Optional[List[int]] = None,
    # Taylor extrapolation order: 0=cache reuse, 1=linear, 2=quadratic (DPCache default)
    taylor_order: int = 2,
) -> "PIL.Image":
    """AesPlan inference on FLUX with Taylor extrapolation at skip steps.

    Ablation modes:
      use_aes_cost=True  → AesMask-weighted DP schedule (our method)
      use_aes_cost=False → uniform spacing with same budget (≈DPCache)

    taylor_order:
      0 → simple cache reuse (no extrapolation)
      2 → quadratic Taylor (matches DPCache Taylor-DP)
    """
    guidance_scale = calib.cfg_scale
    T = calib.total_steps

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

    # mask_step must be a key step (we did an extra forward there during calib)
    key_set.add(calib.mask_step)

    # --- Setup ---
    prompt_embeds, pooled_prompt_embeds, text_ids = wrapper._encode_prompt(prompt)
    wrapper._ensure_transformer_on_gpu()

    latents, img_ids = wrapper._prepare_latents(height, width, seed)
    timesteps = wrapper._setup_scheduler(T, height, width)

    txt_ids = torch.zeros(
        prompt_embeds.shape[1], 3,
        device=wrapper.device, dtype=wrapper.dtype,
    )

    # Taylor cache: {0: noise_pred, 1: d1, 2: d2, 'last_key': step_idx}
    taylor_cache: dict = {}

    for i, t in enumerate(timesteps):
        is_key = (i in key_set)

        if not is_key and taylor_cache:
            # Skip step: Taylor extrapolation or simple cache reuse
            if taylor_order == 0:
                noise_pred = taylor_cache[0]
            else:
                noise_pred = _taylor_predict(taylor_cache, i, taylor_order)

            latents = wrapper.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]
            continue

        # Key step: full forward
        noise_pred = wrapper._transformer_step(
            latents, t, prompt_embeds, pooled_prompt_embeds,
            txt_ids, img_ids, guidance_scale,
        )

        # Update Taylor cache
        taylor_cache = _update_taylor_cache(taylor_cache, noise_pred, i, taylor_order)

        latents = wrapper.scheduler.step(
            noise_pred, t, latents, return_dict=False
        )[0]

    return wrapper._decode(latents, height, width)


def calibrate_dpcache_flux(
    wrapper,
    prompts: list,
    seeds: list = None,
    guidance_scale: float = 3.5,
    steps: int = 28,
    budget: int = 6,
    first_free: int = 4,
    max_skip: int = 6,
) -> List[int]:
    """DPCache-style calibration for FLUX (content-agnostic, global mean cost).

    Uses uniform L1 cost across all spatial tokens — no FG/BG distinction.
    Serves as the direct baseline for our AesMask-weighted method.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from aesplan.dense_run_flux import run_dense_and_capture_flux
    from aesplan.dp_solver import solve_dp
    import numpy as np

    if seeds is None:
        seeds = [42] * len(prompts)

    cost_tables = []
    for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
        print(f"[FLUX DPCache calib] {i+1}/{len(prompts)}: {prompt[:50]}...")
        data = run_dense_and_capture_flux(
            wrapper=wrapper, prompt=prompt, seed=seed,
            guidance_scale=guidance_scale, steps=steps,
        )
        noise_preds = data["noise_preds"]
        T = len(noise_preds)

        # Content-agnostic: global mean L1 per step pair
        cost = np.zeros((T, T), dtype=np.float32)
        for j in range(T):
            for k in range(j + 1, T):
                pred_j = noise_preds[j].squeeze(0)
                pred_k = noise_preds[k].squeeze(0)
                cost[j, k] = float((pred_k - pred_j).abs().mean())
        cost_tables.append(cost)

    cost_avg = np.mean(cost_tables, axis=0)
    key_steps = solve_dp(cost_avg, budget=budget,
                         first_free=first_free, max_skip=max_skip)
    print(f"[FLUX DPCache calib] key_steps={key_steps}")
    return key_steps
