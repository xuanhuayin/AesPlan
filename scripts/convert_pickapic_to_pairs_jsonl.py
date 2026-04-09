#!/usr/bin/env python3
"""
Convert PickaPic-like metadata to AesPlan preference-pairs JSONL.

Output format (one JSON per line):
{"prompt":"...","chosen":"/abs/path/win.png","rejected":"/abs/path/lose.png"}

Supported metadata files:
- .jsonl / .json / .csv / .tsv / .parquet

The converter tries to auto-detect common PickaPic schemas:
- explicit chosen/rejected image path columns
- image_0/image_1 + winner/label/preference columns

Example:
python scripts/convert_pickapic_to_pairs_jsonl.py \
  --data_root /root/autodl-tmp/pickapic_v1/data \
  --output /root/AesPlan/data/pickapic/train_pairs.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROMPT_KEYS = [
    "prompt",
    "caption",
    "text",
    "instruction",
]

CHOSEN_KEYS = [
    "chosen",
    "chosen_image",
    "image_chosen",
    "preferred_image",
    "winner_image",
]

REJECTED_KEYS = [
    "rejected",
    "rejected_image",
    "image_rejected",
    "loser_image",
    "non_preferred_image",
]

IMG0_KEYS = ["image_0", "img0", "left_image", "image_left", "image_a", "a_image"]
IMG1_KEYS = ["image_1", "img1", "right_image", "image_right", "image_b", "b_image"]

WINNER_KEYS = ["winner", "choice", "preference", "label", "pick", "selected"]


def _pick_key(keys: List[str], row: dict) -> Optional[str]:
    row_keys = set(row.keys())
    for k in keys:
        if k in row_keys:
            return k
    # case-insensitive fallback
    lower_map = {k.lower(): k for k in row.keys()}
    for k in keys:
        if k.lower() in lower_map:
            return lower_map[k.lower()]
    return None


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_json(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
    elif isinstance(obj, dict):
        # if nested list field exists, prefer first list-of-dicts
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                for x in v:
                    yield x
                return
        # otherwise treat dict as one row
        yield obj


def _iter_csv_like(path: Path, delimiter: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield dict(row)


def _iter_parquet(path: Path) -> Iterable[dict]:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("Reading parquet requires pandas+pyarrow installed.") from e
    df = pd.read_parquet(path)
    for row in df.to_dict(orient="records"):
        yield dict(row)


def _iter_rows(path: Path) -> Iterable[dict]:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return _iter_jsonl(path)
    if suf == ".json":
        return _iter_json(path)
    if suf == ".csv":
        return _iter_csv_like(path, ",")
    if suf == ".tsv":
        return _iter_csv_like(path, "\t")
    if suf == ".parquet":
        return _iter_parquet(path)
    raise ValueError(f"Unsupported metadata file: {path}")


def _resolve_img_path(data_root: Path, v: str) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return str(p) if p.exists() else None
    # common relative conventions
    cands = [
        data_root / s,
        data_root / "images" / s,
        data_root / "imgs" / s,
    ]
    for c in cands:
        if c.exists():
            return str(c.resolve())
    return None


def _winner_to_index(v) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"0", "left", "a", "img0", "image_0", "first", "chosen_0", "win0"}:
        return 0
    if s in {"1", "right", "b", "img1", "image_1", "second", "chosen_1", "win1"}:
        return 1
    # numeric fallback
    try:
        n = int(float(s))
        if n in (0, 1):
            return n
    except Exception:
        pass
    return None


def _row_to_pair(row: dict, data_root: Path) -> Optional[Tuple[str, str, str]]:
    prompt_k = _pick_key(PROMPT_KEYS, row)
    if prompt_k is None:
        return None
    prompt = str(row.get(prompt_k, "")).strip()
    if not prompt:
        return None

    # direct chosen/rejected paths
    ck = _pick_key(CHOSEN_KEYS, row)
    rk = _pick_key(REJECTED_KEYS, row)
    if ck and rk:
        c = _resolve_img_path(data_root, row.get(ck))
        r = _resolve_img_path(data_root, row.get(rk))
        if c and r and c != r:
            return (prompt, c, r)

    # image_0/image_1 + winner
    i0k = _pick_key(IMG0_KEYS, row)
    i1k = _pick_key(IMG1_KEYS, row)
    wk = _pick_key(WINNER_KEYS, row)
    if i0k and i1k and wk:
        i0 = _resolve_img_path(data_root, row.get(i0k))
        i1 = _resolve_img_path(data_root, row.get(i1k))
        w = _winner_to_index(row.get(wk))
        if i0 and i1 and w is not None and i0 != i1:
            if w == 0:
                return (prompt, i0, i1)
            return (prompt, i1, i0)

    return None


def _discover_meta_files(data_root: Path) -> List[Path]:
    exts = {".jsonl", ".json", ".csv", ".tsv", ".parquet"}
    files = [p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    # prefer likely annotation files
    files.sort(key=lambda p: ("meta" not in p.name.lower() and "pair" not in p.name.lower(), len(str(p))))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--meta", type=str, default=None, help="Optional explicit metadata file path.")
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all.")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    if args.meta:
        meta_files = [Path(args.meta).resolve()]
    else:
        meta_files = _discover_meta_files(data_root)
    if not meta_files:
        raise RuntimeError(
            f"No metadata files found under {data_root}. "
            "Use --meta to point to a JSON/JSONL/CSV/TSV/Parquet file."
        )

    written = 0
    seen = set()
    used_meta = None
    with open(out_path, "w", encoding="utf-8") as fw:
        for meta in meta_files:
            try:
                rows = _iter_rows(meta)
            except Exception:
                continue
            local_count = 0
            for row in rows:
                pair = _row_to_pair(row, data_root)
                if pair is None:
                    continue
                prompt, chosen, rejected = pair
                key = (prompt, chosen, rejected)
                if key in seen:
                    continue
                seen.add(key)
                fw.write(
                    json.dumps(
                        {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
                local_count += 1
                if args.max_samples > 0 and written >= args.max_samples:
                    used_meta = meta
                    break
            if local_count > 0:
                used_meta = meta
            if args.max_samples > 0 and written >= args.max_samples:
                break

    if written == 0:
        raise RuntimeError(
            "No valid pairs converted. "
            "Likely column names differ from defaults. "
            "Please rerun with --meta <your_metadata_file> and share one sample row."
        )

    print(f"[Done] Wrote {written} pairs to: {out_path}")
    if used_meta is not None:
        print(f"[Info] Source metadata: {used_meta}")


if __name__ == "__main__":
    main()
