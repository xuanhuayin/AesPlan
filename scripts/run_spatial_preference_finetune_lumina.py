#!/usr/bin/env python3
"""
P2-7: Full Spatial Preference Finetuning (Lumina)
=================================================

Standalone training script for discrepancy-weighted spatial preference finetuning
on Lumina. Existing project code is untouched.

Data format (JSONL):
  {"prompt": "...", "chosen": "path/to/chosen.png", "rejected": "path/to/rejected.png"}
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


ACCELAES_ROOT = Path(
    os.environ.get(
        "ACCELAES_ROOT",
        str(Path(__file__).resolve().parent.parent.parent / "AccelAes"),
    )
)
sys.path.insert(0, str(ACCELAES_ROOT))
sys.path.insert(0, str(ACCELAES_ROOT / "src"))

from src.models.dit_wrapper import LuminaDiTWrapper, get_2d_rotary_pos_embed_lumina  # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dtype_from_str(name: str) -> torch.dtype:
    table = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name.lower() not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name.lower()]


def _load_image_tensor(path: str | Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), resample=Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class PreferencePairDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, image_size: int = 1024):
        self.jsonl_path = Path(jsonl_path)
        self.image_size = image_size
        self.items: List[dict] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for key in ("prompt", "chosen", "rejected"):
                    if key not in obj:
                        raise ValueError(f"Missing '{key}' in dataset item: {obj}")
                self.items.append(obj)
        if not self.items:
            raise ValueError(f"No valid rows in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.items[idx]
        base = self.jsonl_path.parent
        chosen_path = Path(row["chosen"])
        rejected_path = Path(row["rejected"])
        if not chosen_path.is_absolute():
            chosen_path = (base / chosen_path).resolve()
        if not rejected_path.is_absolute():
            rejected_path = (base / rejected_path).resolve()
        return {
            "prompt": row["prompt"],
            "chosen": _load_image_tensor(chosen_path, self.image_size),
            "rejected": _load_image_tensor(rejected_path, self.image_size),
        }


def _collate(batch: List[dict]) -> dict:
    return {
        "prompts": [x["prompt"] for x in batch],
        "chosen": torch.stack([x["chosen"] for x in batch], dim=0),
        "rejected": torch.stack([x["rejected"] for x in batch], dim=0),
    }


@dataclass
class TextCond:
    cond_embed: torch.Tensor
    uncond_embed: torch.Tensor
    cond_mask: torch.Tensor
    uncond_mask: torch.Tensor


class PromptEmbedCache:
    def __init__(self):
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def get(self, pipe, prompt: str, device: torch.device) -> TextCond:
        if prompt not in self.cache:
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
            self.cache[prompt] = (
                prompt_embeds.detach().cpu(),
                negative_prompt_embeds.detach().cpu(),
                prompt_attention_mask.detach().cpu(),
                negative_prompt_attention_mask.detach().cpu(),
            )
        pe, ne, pm, nm = self.cache[prompt]
        return TextCond(
            cond_embed=pe.to(device=device),
            uncond_embed=ne.to(device=device),
            cond_mask=pm.to(device=device),
            uncond_mask=nm.to(device=device),
        )


def _encode_pixels_to_latents(vae, pixels: torch.Tensor) -> torch.Tensor:
    posterior = vae.encode(pixels).latent_dist
    z = posterior.sample()
    return z * vae.config.scaling_factor


def _add_noise(scheduler, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    if hasattr(scheduler, "add_noise"):
        return scheduler.add_noise(latents, noise, timesteps)
    t = timesteps.float().view(-1, 1, 1, 1)
    t = t / max(1, scheduler.config.num_train_timesteps - 1)
    return latents + t * noise


def _target_from_scheduler(scheduler, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if pred_type == "epsilon":
        return noise
    if pred_type == "v_prediction" and hasattr(scheduler, "get_velocity"):
        return scheduler.get_velocity(latents, noise, timesteps)
    return noise


def _lumina_forward_eps(
    transformer: nn.Module,
    pipe,
    wrapper,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    embeds: torch.Tensor,
    embed_masks: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Returns eps channels only: shape (B, 3, H, W).
    """
    cross_attention_kwargs = {}
    default_sample_size = getattr(pipe, "default_sample_size", 128)
    cross_attention_kwargs["base_sequence_length"] = (default_sample_size // 2) ** 2

    scaling_factor = math.sqrt(width * height / wrapper.default_image_size**2)
    t = timesteps
    if not torch.is_tensor(t):
        t = torch.tensor([t], dtype=torch.float64, device=latents.device)
    if len(t.shape) == 0:
        t = t[None].to(latents.device)
    t = t.to(latents.device).expand(latents.shape[0])
    t_norm = 1 - t / pipe.scheduler.config.num_train_timesteps

    linear_factor = scaling_factor if t_norm[0] < 1.0 else 1.0
    ntk_factor = 1.0 if t_norm[0] < 1.0 else scaling_factor
    image_rotary_emb = get_2d_rotary_pos_embed_lumina(
        wrapper.transformer.head_dim,
        384,
        384,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
    )

    pred = transformer(
        hidden_states=latents,
        timestep=t_norm,
        encoder_hidden_states=embeds,
        encoder_mask=embed_masks,
        image_rotary_emb=image_rotary_emb,
        cross_attention_kwargs=cross_attention_kwargs,
        return_dict=False,
    )[0]
    pred_split = pred.chunk(2, dim=1)[0]
    return pred_split[:, :3]


def _compute_discrepancy_mask(
    ref_transformer: nn.Module,
    pipe,
    wrapper,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    cond_embed: torch.Tensor,
    uncond_embed: torch.Tensor,
    cond_mask: torch.Tensor,
    uncond_mask: torch.Tensor,
    top_ratio: float,
    sharpness: float,
    height: int,
    width: int,
) -> torch.Tensor:
    b = noisy_latents.shape[0]
    lat_cfg = torch.cat([noisy_latents, noisy_latents], dim=0)
    t_cfg = timesteps.repeat(2)
    embeds = torch.cat([cond_embed, uncond_embed], dim=0)
    emasks = torch.cat([cond_mask, uncond_mask], dim=0)

    with torch.no_grad():
        eps = _lumina_forward_eps(
            transformer=ref_transformer,
            pipe=pipe,
            wrapper=wrapper,
            latents=lat_cfg,
            timesteps=t_cfg,
            embeds=embeds,
            embed_masks=emasks,
            height=height,
            width=width,
        )
        eps_cond, eps_uncond = torch.split(eps, b, dim=0)
        mag = (eps_cond - eps_uncond).abs().mean(dim=1, keepdim=True)
        x = mag.flatten(1)
        x = (x - x.min(dim=1, keepdim=True)[0]) / (
            x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0] + 1e-8
        )
        tau = torch.quantile(x, q=(1.0 - top_ratio), dim=1, keepdim=True)
        soft = torch.sigmoid((x - tau) * sharpness)
        return soft.view_as(mag)


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    err = (pred - target).pow(2).mean(dim=1, keepdim=True)
    num = (err * weight_map).flatten(1).sum(dim=1)
    den = weight_map.flatten(1).sum(dim=1).clamp(min=1e-6)
    return num / den


def _save_checkpoint(
    out_dir: Path,
    step: int,
    transformer: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pt"
    torch.save(
        {
            "step": step,
            "transformer": transformer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": vars(args),
        },
        path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/spatial_pref_ft_lumina")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lambda_anchor", type=float, default=0.5)
    parser.add_argument("--w_fg", type=float, default=4.0)
    parser.add_argument("--w_bg", type=float, default=1.0)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--mask_sharpness", type=float, default=8.0)

    parser.add_argument("--t_min", type=int, default=20)
    parser.add_argument("--t_max", type=int, default=980)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--no_frozen_ref", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)

    train_dtype = _dtype_from_str(args.dtype)
    print("[Init] Loading Lumina wrapper...")
    wrapper = LuminaDiTWrapper(dtype=args.dtype)
    device = wrapper.device
    pipe = wrapper.pipe
    transformer = wrapper.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler
    transformer.train()
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    if args.no_frozen_ref:
        ref_transformer = transformer
        print("[Init] Reference model: current transformer.")
    else:
        print("[Init] Building frozen reference transformer copy...")
        ref_transformer = copy.deepcopy(transformer).to(device)
        ref_transformer.eval()
        for p in ref_transformer.parameters():
            p.requires_grad_(False)

    dataset = PreferencePairDataset(args.pairs, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=True,
    )
    steps_per_epoch = math.ceil(len(dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"[Data] pairs={len(dataset)} batch={args.batch_size} epochs={args.epochs} total_steps={total_steps}")

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    use_amp = (device.type == "cuda") and (train_dtype in (torch.float16, torch.bfloat16))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and train_dtype == torch.float16))
    prompt_cache = PromptEmbedCache()

    global_step = 0
    print("[Train] Starting...")
    for _epoch in range(args.epochs):
        for batch in loader:
            prompts: List[str] = batch["prompts"]
            chosen = batch["chosen"].to(device=device, dtype=vae.dtype, non_blocking=True)
            rejected = batch["rejected"].to(device=device, dtype=vae.dtype, non_blocking=True)
            bsz = chosen.shape[0]

            with torch.no_grad():
                chosen_lat = _encode_pixels_to_latents(vae, chosen).to(dtype=transformer.dtype)
                rejected_lat = _encode_pixels_to_latents(vae, rejected).to(dtype=transformer.dtype)

                noise_c = torch.randn_like(chosen_lat)
                noise_r = torch.randn_like(rejected_lat)

                t_low = max(0, args.t_min)
                t_high = min(args.t_max, scheduler.config.num_train_timesteps - 1)
                timesteps = torch.randint(
                    low=t_low, high=t_high + 1, size=(bsz,), device=device, dtype=torch.long
                )
                noisy_c = _add_noise(scheduler, chosen_lat, noise_c, timesteps)
                noisy_r = _add_noise(scheduler, rejected_lat, noise_r, timesteps)
                target_c = _target_from_scheduler(scheduler, chosen_lat, noise_c, timesteps).to(transformer.dtype)
                target_r = _target_from_scheduler(scheduler, rejected_lat, noise_r, timesteps).to(transformer.dtype)

                cond_e, uncond_e, cond_m, uncond_m = [], [], [], []
                for ptxt in prompts:
                    tc = prompt_cache.get(pipe, ptxt, device)
                    cond_e.append(tc.cond_embed)
                    uncond_e.append(tc.uncond_embed)
                    cond_m.append(tc.cond_mask)
                    uncond_m.append(tc.uncond_mask)
                cond_embed = torch.cat(cond_e, dim=0).to(dtype=transformer.dtype)
                uncond_embed = torch.cat(uncond_e, dim=0).to(dtype=transformer.dtype)
                cond_mask = torch.cat(cond_m, dim=0).to(dtype=transformer.dtype)
                uncond_mask = torch.cat(uncond_m, dim=0).to(dtype=transformer.dtype)

                mask_soft = _compute_discrepancy_mask(
                    ref_transformer=ref_transformer,
                    pipe=pipe,
                    wrapper=wrapper,
                    noisy_latents=noisy_c,
                    timesteps=timesteps,
                    cond_embed=cond_embed,
                    uncond_embed=uncond_embed,
                    cond_mask=cond_mask,
                    uncond_mask=uncond_mask,
                    top_ratio=args.mask_ratio,
                    sharpness=args.mask_sharpness,
                    height=args.image_size,
                    width=args.image_size,
                )
                weight_map = args.w_bg + (args.w_fg - args.w_bg) * mask_soft

            with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_amp):
                pred_c = _lumina_forward_eps(
                    transformer, pipe, wrapper, noisy_c, timesteps, cond_embed, cond_mask, args.image_size, args.image_size
                )
                pred_r = _lumina_forward_eps(
                    transformer, pipe, wrapper, noisy_r, timesteps, cond_embed, cond_mask, args.image_size, args.image_size
                )
                mse_c = _weighted_mse(pred_c.float(), target_c[:, :3].float(), weight_map.float())
                mse_r = _weighted_mse(pred_r.float(), target_r[:, :3].float(), weight_map.float())
                score_c = -mse_c
                score_r = -mse_r
                loss_pref = -F.logsigmoid(args.beta * (score_c - score_r)).mean()
                loss_anchor = mse_c.mean()
                loss = loss_pref + args.lambda_anchor * loss_anchor

            loss = loss / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % args.log_every == 0:
                fg_ratio = float((mask_soft > 0.5).float().mean().item())
                print(
                    f"[Train] step={global_step}/{total_steps} "
                    f"loss={loss.item() * args.grad_accum:.4f} "
                    f"pref={loss_pref.item():.4f} anchor={loss_anchor.item():.4f} "
                    f"mse_c={mse_c.mean().item():.4f} mse_r={mse_r.mean().item():.4f} "
                    f"fg>0.5={fg_ratio:.3f}"
                )
            if global_step % args.save_every == 0:
                path = _save_checkpoint(out_dir, global_step, transformer, optimizer, scaler, args)
                print(f"[CKPT] Saved {path}")

    final = _save_checkpoint(out_dir, global_step, transformer, optimizer, scaler, args)
    print(f"[Done] Final checkpoint: {final}")


if __name__ == "__main__":
    main()
