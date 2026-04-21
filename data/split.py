#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split English COCO-TTS manifest into train / val sets.

Usage:
    python data/split.py --out_dir /path/to/out_dir
"""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Directory containing coco_train_tts.jsonl (output written here too)")
    ap.add_argument("--val_ratio", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    inp       = out_dir / "coco_train_tts.jsonl"
    train_out = out_dir / "coco_train_tts_train.jsonl"
    val_out   = out_dir / "coco_train_tts_val.jsonl"

    random.seed(args.seed)

    with open(inp, "r", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]

    random.shuffle(rows)
    n_val = int(args.val_ratio * len(rows))

    val_rows   = rows[:n_val]
    train_rows = rows[n_val:]

    for path, split_rows in [(train_out, train_rows), (val_out, val_rows)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in split_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"train : {len(train_rows)}")
    print(f"val   : {len(val_rows)}")
    print(f"Saved : {train_out}")
    print(f"Saved : {val_out}")


if __name__ == "__main__":
    main()