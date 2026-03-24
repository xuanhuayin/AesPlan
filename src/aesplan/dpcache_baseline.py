"""
src/aesplan/dpcache_baseline.py
=================================
DPCache-style baseline on SD3 and Lumina: global content-agnostic DP scheduling.

Faithful reproduction of DPCache core:
  - PACT (Path-Aware Cost Tensor): C[i,j,k] = accumulated L1 eps distance
    when last key step was i, cache from j, predicting step k.
    Captures multi-step error accumulation (path-dependent).
  - 3D DP solver: finds optimal K key steps given path-dependent cost.
  - No spatial awareness, no differential CFG.
  - Skip steps: Taylor-2 extrapolation of combined eps (order=2, consistent with
    official DPCache paper: mode="Taylor-DP", order=2 default).

NOTE: Official DPCache applies Taylor-2 on per-block transformer hidden states.
      Our implementation applies Taylor-2 on final combined eps (noise-prediction
      level) — simpler, comparable spirit, no block-level hooks needed.

This serves as a direct baseline comparison for our AesMask-weighted method.

Reference: DPCache (CVPR 2026, arXiv 2602.22654)
"""
from __future__ import annotations

import numpy as np
from typing import List


def build_pact_tensor(
    eps_cond: list,
    eps_uncond: list,
    cfg_scale: float = 7.0,
    mask_step: int = 0,
) -> np.ndarray:
    """Build 3D Path-Aware Cost Tensor C[i,j,k].

    C[i,j,k] = L1 distance between combined eps at step k and combined eps
    at step j, weighted by distance since last key step i.

    For steps before mask_step, all steps are "free" (cost=0, always computed).

    Args:
        eps_cond:   list[T] of (1, C, H, W) cpu float tensors
        eps_uncond: list[T] of (1, C, H, W) cpu float tensors
        cfg_scale:  guidance scale for combining eps
        mask_step:  steps 0..mask_step are always computed (first_free)

    Returns:
        pact: (T, T, T) float32. pact[i,j,k] = cost of reusing step-j
              prediction at step k, given last key step was i. 0 if invalid.
    """
    import torch

    T = len(eps_cond)

    # Pre-compute combined eps (content-agnostic, global)
    eps_combined = []
    for t in range(T):
        ec = eps_cond[t].squeeze(0)
        eu = eps_uncond[t].squeeze(0)
        eps_combined.append((eu + cfg_scale * (ec - eu)).mean().item())  # scalar mean

    # For spatial version, we use per-pixel:
    eps_maps = []
    for t in range(T):
        ec = eps_cond[t].squeeze(0)
        eu = eps_uncond[t].squeeze(0)
        eps_maps.append(eu + cfg_scale * (ec - eu))  # (C, H, W)

    pact = np.zeros((T, T, T), dtype=np.float32)

    for i in range(T):        # last key step
        for j in range(i, T): # cache from
            for k in range(j + 1, T):  # predict at
                if k <= mask_step:
                    continue
                # Cost: L1 between k and j (direct skip error)
                # Path-awareness: penalize longer gaps from last key i
                direct_cost = (eps_maps[k] - eps_maps[j]).abs().mean().item()
                # Accumulation factor: longer since last key → higher uncertainty
                gap_from_key = j - i + 1  # steps since we last computed
                pact[i, j, k] = direct_cost * (1.0 + 0.1 * gap_from_key)

    return pact


def solve_dp_3d(
    pact: np.ndarray,
    budget: int,
    first_free: int = 6,
    max_skip: int = 4,
) -> List[int]:
    """3D path-aware DP solver.

    Finds the optimal set of K key steps minimising total path-aware
    skip cost, given the PACT tensor.

    Args:
        pact: (T, T, T) float32 cost tensor
        budget: number of key steps K
        first_free: always-compute warm-up steps
        max_skip: max consecutive skips

    Returns:
        key_steps: sorted list of step indices (length = budget)
    """
    T = pact.shape[0]
    INF = float("inf")

    forced = list(range(first_free))
    T_plan = T - first_free
    K_plan = budget - first_free

    if K_plan <= 0:
        return forced

    def orig(t_plan: int) -> int:
        return t_plan + first_free

    # State: dp[t][b] = (min_cost, last_key_orig, prev_key_orig)
    # last_key = j in pact (most recent key step)
    # prev_key = i in pact (key step before last_key, for path-awareness)

    best_cost = np.full((T_plan, K_plan + 1), INF)
    last_key_arr = np.full((T_plan, K_plan + 1), -1, dtype=int)
    prev_key_arr = np.full((T_plan, K_plan + 1), -1, dtype=int)
    is_key_arr = np.zeros((T_plan, K_plan + 1), dtype=bool)

    def skip_cost_3d(i_orig: int, j_orig: int, k_orig: int) -> float:
        if i_orig < 0 or j_orig < 0:
            return INF
        i_c = min(i_orig, T - 1)
        j_c = min(j_orig, T - 1)
        k_c = min(k_orig, T - 1)
        if i_c <= j_c < k_c:
            return float(pact[i_c, j_c, k_c])
        return INF

    # Base: t=0
    # Option A: make it key (b=1)
    best_cost[0][1] = 0.0
    last_key_arr[0][1] = orig(0)
    prev_key_arr[0][1] = first_free - 1 if first_free > 0 else -1
    is_key_arr[0][1] = True
    # Option B: skip (b=0, use forced[-1] as last key)
    if first_free > 0:
        i_prev = first_free - 2 if first_free >= 2 else -1
        j_prev = first_free - 1
        k_cur = orig(0)
        if k_cur - j_prev <= max_skip:
            c = skip_cost_3d(i_prev, j_prev, k_cur)
            if c < INF:
                best_cost[0][0] = c
                last_key_arr[0][0] = j_prev
                prev_key_arr[0][0] = i_prev
                is_key_arr[0][0] = False

    # Fill DP
    for t in range(1, T_plan):
        o = orig(t)
        for b in range(K_plan + 1):
            # Option A: step t is key
            if b >= 1 and best_cost[t - 1][b - 1] < INF:
                c_a = best_cost[t - 1][b - 1]
                if c_a < best_cost[t][b]:
                    best_cost[t][b] = c_a
                    last_key_arr[t][b] = o
                    prev_key_arr[t][b] = last_key_arr[t - 1][b - 1]
                    is_key_arr[t][b] = True

            # Option B: skip step t
            lk = last_key_arr[t - 1][b]
            pk = prev_key_arr[t - 1][b]
            if lk >= 0 and best_cost[t - 1][b] < INF:
                if (o - lk) <= max_skip:
                    c_b = best_cost[t - 1][b] + skip_cost_3d(pk if pk >= 0 else lk, lk, o)
                    if c_b < best_cost[t][b]:
                        best_cost[t][b] = c_b
                        last_key_arr[t][b] = lk
                        prev_key_arr[t][b] = pk
                        is_key_arr[t][b] = False

    # Backtrack
    if best_cost[T_plan - 1][K_plan] == INF:
        step = max(1, T_plan // K_plan)
        planned = sorted(set(range(0, T_plan, step)[:K_plan]))
        return sorted(forced + [orig(t) for t in planned])

    planned_keys = []
    b = K_plan
    for t in range(T_plan - 1, -1, -1):
        if is_key_arr[t][b]:
            planned_keys.append(orig(t))
            b -= 1

    planned_keys.reverse()
    return sorted(forced + planned_keys)


def calibrate_dpcache_sd3(
    wrapper,
    prompts: list,
    seeds: list = None,
    cfg_scale: float = 7.0,
    steps: int = 28,
    mask_step: int = 5,
    budget: int = 10,
    first_free: int = 6,
    max_skip: int = 4,
) -> List[int]:
    """Run DPCache calibration on SD3, return key_steps.

    Uses global content-agnostic cost (no spatial awareness).
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from aesplan.dense_run_sd3 import run_dense_and_capture_sd3

    if seeds is None:
        seeds = [42] * len(prompts)

    pact_list = []
    for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
        print(f"[DPCache calib] {i+1}/{len(prompts)}: {prompt[:50]}...")
        data = run_dense_and_capture_sd3(
            wrapper=wrapper, prompt=prompt, seed=seed,
            cfg_scale=cfg_scale, steps=steps, mask_step=mask_step,
        )
        pact = build_pact_tensor(
            data["eps_cond"], data["eps_uncond"],
            cfg_scale=cfg_scale, mask_step=mask_step,
        )
        pact_list.append(pact)

    pact_avg = np.mean(pact_list, axis=0)
    key_steps = solve_dp_3d(pact_avg, budget=budget,
                            first_free=first_free, max_skip=max_skip)
    print(f"[DPCache calib] key_steps={key_steps}")
    return key_steps


def generate_dpcache_sd3(
    wrapper,
    prompt: str,
    seed: int,
    key_steps: list,
    cfg_scale: float = 7.0,
    steps: int = 28,
    height: int = 1024,
    width: int = 1024,
    taylor_order: int = 2,
) -> "PIL.Image":
    """DPCache-style inference on SD3: Taylor-2 on combined eps at skip steps."""
    import torch

    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

    # Encode prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        negative_prompt="",
        do_classifier_free_guidance=True,
        num_images_per_prompt=1, device=device,
    )
    prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    latent_h = height // wrapper.vae_scale_factor
    latent_w = width  // wrapper.vae_scale_factor
    latent_channels = wrapper.transformer.config.in_channels

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, latent_channels, latent_h, latent_w),
        generator=generator, device=device, dtype=dtype,
    )

    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    key_set = set(key_steps)

    # Taylor cache on combined eps (order-2)
    tc_f0 = None   # f0: last combined eps
    tc_f1 = None   # f1: first derivative
    tc_f2 = None   # f2: second derivative
    tc_last_key = None

    for i, t in enumerate(timesteps):
        if i not in key_set and tc_f0 is not None:
            d = float(i - tc_last_key)
            eps_hat = tc_f0.clone()
            if taylor_order >= 1 and tc_f1 is not None:
                eps_hat = eps_hat + tc_f1 * d
            if taylor_order >= 2 and tc_f2 is not None:
                eps_hat = eps_hat + tc_f2 * (d * d / 2.0)
            latents = pipe.scheduler.step(eps_hat, t, latents, return_dict=False)[0]
            continue

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
        guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        # Update Taylor derivatives
        if tc_f0 is not None and tc_last_key is not None:
            dist = float(i - tc_last_key)
            if dist > 0:
                new_f1 = (guided - tc_f0) / dist
                if taylor_order >= 2 and tc_f1 is not None:
                    tc_f2 = (new_f1 - tc_f1) / dist
                tc_f1 = new_f1
        else:
            tc_f1 = None
            tc_f2 = None

        tc_f0 = guided.clone()
        tc_last_key = i
        latents = pipe.scheduler.step(guided, t, latents, return_dict=False)[0]

    latents = (latents / wrapper.vae.config.scaling_factor) + wrapper.vae.config.shift_factor
    with torch.no_grad():
        image = wrapper.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


# ---------------------------------------------------------------------------
# Lumina DPCache: calibration + Taylor-1 inference
# ---------------------------------------------------------------------------

def calibrate_dpcache_lumina(
    wrapper,
    prompts: list,
    seeds: list = None,
    cfg_scale: float = 4.0,
    steps: int = 30,
    mask_step: int = 5,
    budget: int = 10,
    first_free: int = 6,
    max_skip: int = 8,
) -> List[int]:
    """Run DPCache calibration on Lumina, return optimal key_steps.

    Uses global content-agnostic PACT cost (no spatial awareness).
    Same algorithm as calibrate_dpcache_sd3 but with Lumina dense capture.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from aesplan.dense_run import run_dense_and_capture

    if seeds is None:
        seeds = [42] * len(prompts)

    pact_list = []
    for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
        print(f"[DPCache Lumina calib] {i+1}/{len(prompts)}: {prompt[:50]}...")
        data = run_dense_and_capture(
            wrapper=wrapper, prompt=prompt, seed=seed,
            cfg_scale=cfg_scale, steps=steps, mask_step=mask_step,
        )
        pact = build_pact_tensor(
            data["eps_cond"], data["eps_uncond"],
            cfg_scale=cfg_scale, mask_step=mask_step,
        )
        pact_list.append(pact)

    pact_avg = np.mean(pact_list, axis=0)
    key_steps = solve_dp_3d(pact_avg, budget=budget,
                            first_free=first_free, max_skip=max_skip)
    print(f"[DPCache Lumina calib] key_steps={key_steps}")
    return key_steps


def generate_dpcache_lumina(
    wrapper,
    prompt: str,
    seed: int,
    key_steps: list,
    cfg_scale: float = 4.0,
    steps: int = 30,
    height: int = 1024,
    width: int = 1024,
    taylor_order: int = 2,
) -> "PIL.Image":
    """DPCache-style inference on Lumina.

    At key steps: full transformer forward, update Taylor cache.
    At skip steps: Taylor-2 extrapolation of combined eps.
    No spatial differentiation (content-agnostic baseline).

    taylor_order=2: quadratic Taylor (default, matches DPCache paper order=2)
    taylor_order=1: linear Taylor (ablation only)
    taylor_order=0: plain cache reuse
    """
    import math
    import torch
    import sys
    from pathlib import Path
    import os as _os
    _accelaes = Path(_os.environ.get("ACCELAES_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent / "AccelAes")))
    sys.path.insert(0, str(_accelaes / "src"))
    from src.models.dit_wrapper import get_2d_rotary_pos_embed_lumina

    pipe = wrapper.pipe
    device = wrapper.device
    dtype = wrapper.dtype

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
    latent_w = width  // wrapper.vae_scale_factor
    latent_channels = wrapper.transformer.config.in_channels
    shape = (1, latent_channels, latent_h, latent_w)

    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    key_set = set(key_steps)
    head_dim = wrapper.transformer.head_dim

    # Taylor cache for combined eps (content-agnostic, order-2)
    tc_f0 = None   # last combined eps (1, 3, H, W)
    tc_f1 = None   # first derivative
    tc_f2 = None   # second derivative
    tc_last_key = None  # step index of last key

    cache_rest = None   # (1, 1, H, W) 4th channel

    for i, t in enumerate(timesteps):
        is_key = (i in key_set)

        # --- Skip step: Taylor-2 extrapolation ---
        if not is_key and tc_f0 is not None:
            d = float(i - tc_last_key)
            eps_hat = tc_f0.float().clone()
            if taylor_order >= 1 and tc_f1 is not None:
                eps_hat = eps_hat + tc_f1.float() * d
            if taylor_order >= 2 and tc_f2 is not None:
                eps_hat = eps_hat + tc_f2.float() * (d * d / 2.0)
            eps_hat = eps_hat.to(tc_f0.dtype)

            if cache_rest is not None:
                noise_pred_step = -torch.cat([eps_hat, cache_rest], dim=1)
            else:
                noise_pred_step = -torch.cat(
                    [eps_hat, torch.zeros_like(eps_hat[:, :1])], dim=1
                )
            latents = pipe.scheduler.step(noise_pred_step, t, latents, return_dict=False)[0]
            continue

        # --- Key step: full forward ---
        latent_model_input = torch.cat([latents] * 2, dim=0)

        current_timestep = t
        if not torch.is_tensor(current_timestep):
            current_timestep = torch.tensor([current_timestep], dtype=torch.float64, device=device)
        elif len(current_timestep.shape) == 0:
            current_timestep = current_timestep[None].to(device)
        current_timestep = current_timestep.expand(latent_model_input.shape[0])
        current_timestep = 1 - current_timestep / pipe.scheduler.config.num_train_timesteps

        linear_factor = scaling_factor if current_timestep[0] < 1.0 else 1.0
        ntk_factor    = 1.0 if current_timestep[0] < 1.0 else scaling_factor

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

        noise_pred_split = noise_pred.chunk(2, dim=1)[0]   # (2, 4, H, W)
        eps = noise_pred_split[:, :3]                       # (2, 3, H, W)
        noise_pred_rest = noise_pred_split[:, 3:]

        eps_cond_t, eps_uncond_t = torch.split(eps, len(eps) // 2, dim=0)
        guided = eps_uncond_t + cfg_scale * (eps_cond_t - eps_uncond_t)  # (1, 3, H, W)

        # Update Taylor cache (order-2: f0, f1, f2)
        if tc_f0 is not None and tc_last_key is not None:
            dist = float(i - tc_last_key)
            if dist > 0:
                new_f1 = (guided - tc_f0) / dist
                if taylor_order >= 2 and tc_f1 is not None:
                    tc_f2 = (new_f1 - tc_f1) / dist
                tc_f1 = new_f1
        else:
            tc_f1 = None
            tc_f2 = None

        tc_f0 = guided.clone()
        tc_last_key = i
        cache_rest = noise_pred_rest[0:1].clone()

        # Reconstruct full noise_pred for scheduler
        eps_full = torch.cat([guided, guided], dim=0)
        noise_pred_out = torch.cat([eps_full, noise_pred_rest], dim=1)
        noise_pred_out, _ = noise_pred_out.chunk(2, dim=0)
        noise_pred_out = -noise_pred_out

        latents = pipe.scheduler.step(noise_pred_out, t, latents, return_dict=False)[0]

    # Decode (Lumina: no shift_factor)
    latents = latents.detach() / pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image.detach(), output_type="pil")[0]
