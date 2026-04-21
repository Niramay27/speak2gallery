#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_tag_index.py
==================
Builds a JSON tag index from the multilingual COCO manifest:

    { "/path/to/img.jpg": ["dog", "surfboard", "beach", ...], ... }

The tags are extracted from the English captions using the same
keyword extractor in query_processor.py (no ML required — fast).

Optionally, if spaCy is available, uses NOUN extraction for richer tags.

Usage
-----
  python build_tag_index.py \
    --manifest coco_train_tts_multilingual_train.jsonl \
    --output   coco_tags.json

The resulting JSON can be passed to HybridRetriever via --tag_index.
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from query_processor import QueryProcessor, COCO_OBJECTS, COMMON_ATTRIBUTES


def build_tag_index(manifest_path: str, output_path: str, use_spacy: bool = True):
    """
    Read the manifest and extract per-image tags from English captions.

    For each image, all English captions across languages are combined
    so the tag set is as rich as possible.
    """
    # image_path → set of tags
    tag_map = defaultdict(set)

    # load QueryProcessor for keyword extraction (no translation needed)
    qp = QueryProcessor(translate=False, use_spacy=use_spacy)

    print(f"[build_tag_index] Reading manifest: {manifest_path}")
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"[build_tag_index] Processing {len(rows)} rows…")
    for row in tqdm(rows, desc="Extracting tags"):
        img_path = row["image_path"]
        caption  = row.get("english_caption", row.get("caption", ""))
        if not caption:
            continue

        qc = qp.process(caption, src_lang="en")

        # combine objects + attributes into tags
        for kw in qc.objects + qc.attributes:
            kw = kw.lower().strip()
            if len(kw) >= 3:
                tag_map[img_path].add(kw)

        # also add raw COCO-object matches from caption tokens
        for tok in caption.lower().split():
            if tok in COCO_OBJECTS:
                tag_map[img_path].add(tok)

    # convert sets to sorted lists for JSON serialisation
    tag_dict = {path: sorted(tags) for path, tags in tag_map.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tag_dict, f, ensure_ascii=False, indent=2)

    n_images = len(tag_dict)
    n_tags   = sum(len(v) for v in tag_dict.values())
    print(f"\n[build_tag_index] Done.")
    print(f"  Images indexed  : {n_images}")
    print(f"  Total tag pairs : {n_tags}")
    print(f"  Avg tags/image  : {n_tags / max(n_images, 1):.1f}")
    print(f"  Saved to        : {output_path}")


def main():
    p = argparse.ArgumentParser(description="Build per-image tag index from COCO manifest")
    p.add_argument("--manifest", required=True,
                   help="Path to coco_train_tts_multilingual_train.jsonl")
    p.add_argument("--output",   default="coco_tags.json",
                   help="Output JSON file path")
    p.add_argument("--no_spacy", action="store_true",
                   help="Disable spaCy (use keyword matching only)")
    args = p.parse_args()

    build_tag_index(
        manifest_path = args.manifest,
        output_path   = args.output,
        use_spacy     = not args.no_spacy,
    )


if __name__ == "__main__":
    main()
