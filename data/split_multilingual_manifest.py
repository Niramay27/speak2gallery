#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split multilingual COCO-TTS manifest into train / val sets,
keeping all language rows for a given image/caption pair together.

Usage:
    python data/split_multilingual_manifest.py --out_dir /path/to/out_dir
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Directory containing coco_train_tts_multilingual.jsonl")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir    = Path(args.out_dir)
    input_jsonl = out_dir / "coco_train_tts_multilingual.jsonl"
    train_jsonl = out_dir / "coco_train_tts_multilingual_train.jsonl"
    val_jsonl   = out_dir / "coco_train_tts_multilingual_val.jsonl"

    random.seed(args.seed)

    groups = defaultdict(list)
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["image_path"], row["english_caption"])
            groups[key].append(row)

    keys = list(groups.keys())
    random.shuffle(keys)

    n_val     = int(len(keys) * args.val_ratio)
    val_keys  = set(keys[:n_val])

    train_rows, val_rows = [], []
    for k, rows in groups.items():
        if k in val_keys:
            val_rows.extend(rows)
        else:
            train_rows.extend(rows)

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_jsonl, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Train rows   : {len(train_rows)}")
    print(f"Val rows     : {len(val_rows)}")
    print(f"Train groups : {len(keys) - n_val}")
    print(f"Val groups   : {n_val}")
    print(f"Saved : {train_jsonl}")
    print(f"Saved : {val_jsonl}")


if __name__ == "__main__":
    main()