#!/usr/bin/env python3
"""
P2-7: Full Spatial Preference Finetuning (FLUX)
===============================================

Standalone script for discrepancy/guidance-diff weighted preference finetuning
on FLUX. Existing repository files remain unchanged.

Supported data modes:
1) Image pairs JSONL:
   {"prompt":"...","chosen":".../chosen.png","rejected":".../rejected.png"}
2) Packed latent pairs JSONL (recommended fallback):
   {"prompt":"...","chosen_latent":".../chosen.pt","rejected_latent":".../rejected.pt"}
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

from src.models.flux_wrapper import FluxDiTWrapper  # noqa: E402


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


class FluxPreferenceDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, image_size: int = 1024):
        self.jsonl_path = Path(jsonl_path)
        self.image_size = image_size
        self.items: List[dict] = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "prompt" not in row:
                    raise ValueError(f"Missing prompt in row: {row}")
                has_img = ("chosen" in row and "rejected" in row)
                has_lat = ("chosen_latent" in row and "rejected_latent" in row)
                if not has_img and not has_lat:
                    raise ValueError("Each row must contain image pair or latent pair.")
                self.items.append(row)
        if not self.items:
            raise ValueError(f"No valid rows in {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        row = self.items[idx]
        base = self.jsonl_path.parent
        out = {"prompt": row["prompt"]}
        if "chosen_latent" in row:
            c = Path(row["chosen_latent"])
            r = Path(row["rejected_latent"])
            if not c.is_absolute():
                c = (base / c).resolve()
            if not r.is_absolute():
                r = (base / r).resolve()
            out["chosen_latent"] = torch.load(c, map_location="cpu")
            out["rejected_latent"] = torch.load(r, map_location="cpu")
            out["mode"] = "latent"
            return out

        c = Path(row["chosen"])
        r = Path(row["rejected"])
        if not c.is_absolute():
            c = (base / c).resolve()
        if not r.is_absolute():
            r = (base / r).resolve()
        out["chosen"] = _load_image_tensor(c, self.image_size)
        out["rejected"] = _load_image_tensor(r, self.image_size)
        out["mode"] = "image"
        return out


def _collate(batch: List[dict]) -> dict:
    prompts = [x["prompt"] for x in batch]
    modes = [x["mode"] for x in batch]
    if len(set(modes)) != 1:
        raise ValueError("Mixed image/latent modes in one batch is not supported.")
    mode = modes[0]
    out = {"prompts": prompts, "mode": mode}
    if mode == "latent":
        out["chosen_latent"] = torch.stack([x["chosen_latent"] for x in batch], dim=0)
        out["rejected_latent"] = torch.stack([x["rejected_latent"] for x in batch], dim=0)
    else:
        out["chosen"] = torch.stack([x["chosen"] for x in batch], dim=0)
        out["rejected"] = torch.stack([x["rejected"] for x in batch], dim=0)
    return out


@dataclass
class FluxTextCond:
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    txt_ids: torch.Tensor


class PromptEmbedCache:
    def __init__(self):
        self.cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, wrapper, prompt: str, device: torch.device, dtype: torch.dtype) -> FluxTextCond:
        if prompt not in self.cache:
            p, pp, _ = wrapper._encode_prompt(prompt)
            self.cache[prompt] = (p.detach().cpu(), pp.detach().cpu())
        p, pp = self.cache[prompt]
        p = p.to(device=device, dtype=dtype)
        pp = pp.to(device=device, dtype=dtype)
        txt_ids = torch.zeros(p.shape[1], 3, device=device, dtype=dtype)
        return FluxTextCond(prompt_embeds=p, pooled_prompt_embeds=pp, txt_ids=txt_ids)


def _encode_images_to_flux_latents(wrapper, pixels: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode images to FLUX packed latents if wrapper exposes required helpers.
    Returns (packed_latents, img_ids).
    """
    if hasattr(wrapper, "_encode_images_to_latents"):
        return wrapper._encode_images_to_latents(pixels, height=height, width=width)

    if hasattr(wrapper, "_encode_image_to_latents"):
        packed, img_ids = [], []
        for i in range(pixels.shape[0]):
            p, ids = wrapper._encode_image_to_latents(pixels[i : i + 1], height=height, width=width)
            packed.append(p)
            img_ids.append(ids)
        return torch.cat(packed, dim=0), img_ids[0]

    raise RuntimeError(
        "FLUX wrapper does not expose image->latent encode helpers. "
        "Use latent-mode dataset rows with chosen_latent/rejected_latent .pt files."
    )


def _add_noise(scheduler, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    if hasattr(scheduler, "add_noise"):
        return scheduler.add_noise(latents, noise, timesteps)
    t = timesteps.float().view(-1, 1, 1)
    t = t / max(1, scheduler.config.num_train_timesteps - 1)
    return latents + t * noise


def _target_from_scheduler(scheduler, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    pred_type = getattr(scheduler.config, "prediction_type", "epsilon")
    if pred_type == "epsilon":
        return noise
    return noise


def _flux_forward(
    wrapper,
    transformer: nn.Module,
    latents: torch.Tensor,
    timestep,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    txt_ids: torch.Tensor,
    img_ids,
    guidance_scale: float,
) -> torch.Tensor:
    """
    Wrapper-agnostic forward on FLUX.
    """
    if hasattr(wrapper, "_transformer_step"):
        return wrapper._transformer_step(
            latents,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
            txt_ids,
            img_ids,
            guidance_scale,
        )

    return transformer(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        txt_ids=txt_ids,
        img_ids=img_ids,
        guidance=guidance_scale,
        return_dict=False,
    )[0]


def _compute_guidance_diff_mask(
    wrapper,
    ref_transformer: nn.Module,
    latents: torch.Tensor,
    timestep,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    txt_ids: torch.Tensor,
    img_ids,
    latent_h: int,
    latent_w: int,
    top_ratio: float,
    sharpness: float,
) -> torch.Tensor:
    with torch.no_grad():
        p35 = _flux_forward(
            wrapper, ref_transformer, latents, timestep, prompt_embeds, pooled_prompt_embeds, txt_ids, img_ids, 3.5
        )
        p10 = _flux_forward(
            wrapper, ref_transformer, latents, timestep, prompt_embeds, pooled_prompt_embeds, txt_ids, img_ids, 1.0
        )
        diff = (p35 - p10).abs().mean(dim=-1)  # (B, seq_len)
        tok_h = latent_h // 2
        tok_w = latent_w // 2
        diff_map = diff.view(diff.shape[0], 1, tok_h, tok_w)
        x = diff_map.flatten(1)
        x = (x - x.min(dim=1, keepdim=True)[0]) / (
            x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0] + 1e-8
        )
        tau = torch.quantile(x, q=(1.0 - top_ratio), dim=1, keepdim=True)
        soft = torch.sigmoid((x - tau) * sharpness).view_as(diff_map)
        return F.interpolate(soft, size=(latent_h, latent_w), mode="bilinear", align_corners=False)


def _weighted_mse_token(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_map: torch.Tensor,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    """
    pred/target: (B, seq_len, C), weight_map: (B,1,latent_h,latent_w)
    """
    tok_h = latent_h // 2
    tok_w = latent_w // 2
    w_tok = F.interpolate(weight_map, size=(tok_h, tok_w), mode="bilinear", align_corners=False)
    w_tok = w_tok.flatten(2)  # (B,1,seq)

    err = (pred - target).pow(2).mean(dim=-1, keepdim=True).transpose(1, 2)  # (B,1,seq)
    num = (err * w_tok).sum(dim=-1).squeeze(1)
    den = w_tok.sum(dim=-1).squeeze(1).clamp(min=1e-6)
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
    parser.add_argument("--output", type=str, default="outputs/spatial_pref_ft_flux")
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
    parser.add_argument("--guidance_scale", type=float, default=3.5)

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

    print("[Init] Loading FLUX wrapper...")
    wrapper = FluxDiTWrapper(dtype=args.dtype)
    device = wrapper.device
    transformer = wrapper.transformer
    scheduler = wrapper.scheduler
    transformer.train()

    if args.no_frozen_ref:
        ref_transformer = transformer
        print("[Init] Reference model: current transformer.")
    else:
        print("[Init] Building frozen reference transformer copy...")
        ref_transformer = copy.deepcopy(transformer).to(device)
        ref_transformer.eval()
        for p in ref_transformer.parameters():
            p.requires_grad_(False)

    dataset = FluxPreferenceDataset(args.pairs, image_size=args.image_size)
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
            mode = batch["mode"]
            bsz = len(prompts)

            with torch.no_grad():
                latent_h = args.image_size // wrapper.vae_scale_factor
                latent_w = args.image_size // wrapper.vae_scale_factor
                lat_seed, img_ids = wrapper._prepare_latents(args.image_size, args.image_size, seed=args.seed)
                seq_len = lat_seed.shape[1]
                ch = lat_seed.shape[2]
                img_ids = img_ids

                if mode == "latent":
                    chosen_lat = batch["chosen_latent"].to(device=device, dtype=wrapper.dtype, non_blocking=True)
                    rejected_lat = batch["rejected_latent"].to(device=device, dtype=wrapper.dtype, non_blocking=True)
                    if chosen_lat.dim() == 2:
                        chosen_lat = chosen_lat.unsqueeze(0)
                        rejected_lat = rejected_lat.unsqueeze(0)
                else:
                    chosen_img = batch["chosen"].to(device=device, dtype=wrapper.dtype, non_blocking=True)
                    rejected_img = batch["rejected"].to(device=device, dtype=wrapper.dtype, non_blocking=True)
                    chosen_lat, _ = _encode_images_to_flux_latents(wrapper, chosen_img, args.image_size, args.image_size)
                    rejected_lat, _ = _encode_images_to_flux_latents(wrapper, rejected_img, args.image_size, args.image_size)

                if chosen_lat.shape[1:] != (seq_len, ch):
                    raise ValueError(
                        f"Latent shape mismatch. Expected (*,{seq_len},{ch}), got {tuple(chosen_lat.shape)}"
                    )

                noise_c = torch.randn_like(chosen_lat)
                noise_r = torch.randn_like(rejected_lat)

                t_low = max(0, args.t_min)
                t_high = min(args.t_max, scheduler.config.num_train_timesteps - 1)
                t_idx = torch.randint(low=t_low, high=t_high + 1, size=(bsz,), device=device, dtype=torch.long)
                noisy_c = _add_noise(scheduler, chosen_lat, noise_c, t_idx)
                noisy_r = _add_noise(scheduler, rejected_lat, noise_r, t_idx)
                target_c = _target_from_scheduler(scheduler, chosen_lat, noise_c, t_idx).to(wrapper.dtype)
                target_r = _target_from_scheduler(scheduler, rejected_lat, noise_r, t_idx).to(wrapper.dtype)

                p_emb, pp_emb, txt_ids = [], [], []
                for ptxt in prompts:
                    tc = prompt_cache.get(wrapper, ptxt, device, wrapper.dtype)
                    p_emb.append(tc.prompt_embeds)
                    pp_emb.append(tc.pooled_prompt_embeds)
                    txt_ids.append(tc.txt_ids)
                prompt_embeds = torch.cat(p_emb, dim=0)
                pooled_prompt_embeds = torch.cat(pp_emb, dim=0)
                txt_ids = txt_ids[0]

                t_for_model = t_idx[0] if bsz == 1 else t_idx
                mask_soft = _compute_guidance_diff_mask(
                    wrapper=wrapper,
                    ref_transformer=ref_transformer,
                    latents=noisy_c,
                    timestep=t_for_model,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    top_ratio=args.mask_ratio,
                    sharpness=args.mask_sharpness,
                )
                weight_map = args.w_bg + (args.w_fg - args.w_bg) * mask_soft

            with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_amp):
                pred_c = _flux_forward(
                    wrapper, transformer, noisy_c, t_for_model, prompt_embeds, pooled_prompt_embeds, txt_ids, img_ids, args.guidance_scale
                )
                pred_r = _flux_forward(
                    wrapper, transformer, noisy_r, t_for_model, prompt_embeds, pooled_prompt_embeds, txt_ids, img_ids, args.guidance_scale
                )
                mse_c = _weighted_mse_token(pred_c.float(), target_c.float(), weight_map.float(), latent_h, latent_w)
                mse_r = _weighted_mse_token(pred_r.float(), target_r.float(), weight_map.float(), latent_h, latent_w)
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
