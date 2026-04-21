#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a CLIP image embedding index from a multilingual COCO-TTS manifest.

Usage:
    python retrieval/build_image_index.py \
        --manifest /path/to/coco_train_tts_multilingual_train.jsonl \
        --output   /path/to/image_index.pt
"""

import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
from tqdm import tqdm


class ImageOnlyDataset(Dataset):
    def __init__(self, manifest_jsonl, clip_preprocess):
        self.clip_preprocess = clip_preprocess

        seen = set()
        self.image_paths = []

        with open(manifest_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                path = row["image_path"]
                if path not in seen:
                    seen.add(path)
                    self.image_paths.append(path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.clip_preprocess(image)
        return {"image": image, "image_path": path}


def collate_fn(batch):
    images = torch.stack([x["image"] for x in batch], dim=0)
    image_paths = [x["image_path"] for x in batch]
    return {"image": images, "image_path": image_paths}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True,
                    help="Path to coco_train_tts_multilingual_train.jsonl")
    ap.add_argument("--output", type=str, required=True,
                    help="Output path for image_index.pt")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()

    dataset = ImageOnlyDataset(args.manifest, clip_preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    all_embs  = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Indexing images"):
            images = batch["image"].to(device, non_blocking=True)
            emb = clip_model.encode_image(images)
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
            all_paths.extend(batch["image_path"])

    all_embs = torch.cat(all_embs, dim=0)

    torch.save({"image_paths": all_paths, "embeddings": all_embs}, args.output)

    print(f"Saved image index to : {args.output}")
    print(f"Total indexed images : {len(all_paths)}")
    print(f"Embedding shape      : {tuple(all_embs.shape)}")


if __name__ == "__main__":
    main()