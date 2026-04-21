#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the CNN multilingual audio encoder.

Usage:
    python evaluation/eval_retrieval_multilingual_cnn.py \
        --manifest    /path/to/coco_train_tts_multilingual_val.jsonl \
        --checkpoint  /path/to/audio_encoder_cnn_multilingual.pt
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_coco_tts_multilingual import CocoTtsMultilingual, collate_coco_tts_multilingual
from models.models_cnn import AudioEncoder


@torch.no_grad()
def compute_embeddings(audio_model, clip_model, loader, device):
    audio_embeds = []
    image_embeds = []

    for batch in tqdm(loader, desc="Encoding"):
        images = batch["image"].to(device, non_blocking=True)
        mels   = batch["mel"].to(device, non_blocking=True)

        img_emb = clip_model.encode_image(images)
        img_emb = F.normalize(img_emb, dim=-1)

        aud_emb = audio_model(mels)
        aud_emb = F.normalize(aud_emb, dim=-1)

        image_embeds.append(img_emb.cpu())
        audio_embeds.append(aud_emb.cpu())

    return torch.cat(audio_embeds, dim=0), torch.cat(image_embeds, dim=0)


def ranks_from_similarity(similarity):
    sorted_indices = torch.argsort(similarity, dim=1, descending=True)
    ranks = [(sorted_indices[i] == i).nonzero(as_tuple=False).item() + 1
             for i in range(similarity.size(0))]
    return torch.tensor(ranks)


def metrics_from_ranks(ranks):
    return {
        "R@1":        (ranks <= 1).float().mean().item(),
        "R@5":        (ranks <= 5).float().mean().item(),
        "R@10":       (ranks <= 10).float().mean().item(),
        "MRR":        (1.0 / ranks.float()).mean().item(),
        "MedianRank": ranks.median().item(),
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    print(f"R@1        : {metrics['R@1'] * 100:.2f}")
    print(f"R@5        : {metrics['R@5'] * 100:.2f}")
    print(f"R@10       : {metrics['R@10'] * 100:.2f}")
    print(f"MRR        : {metrics['MRR']:.4f}")
    print(f"Median Rank: {metrics['MedianRank']}")


def evaluate_subset(manifest_path, checkpoint_path, lang_filter=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    ds = CocoTtsMultilingual(
        manifest_jsonl=manifest_path,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
    )
    if lang_filter is not None:
        ds.rows = [r for r in ds.rows if r["lang"] == lang_filter]

    loader = DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=4,
        pin_memory=True, collate_fn=collate_coco_tts_multilingual,
    )

    audio_model = AudioEncoder(n_mels=80, embed_dim=512).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    audio_model.load_state_dict(state)
    print(f"\nLoaded CNN checkpoint: {checkpoint_path}")
    audio_model.eval()

    audio_embeds, image_embeds = compute_embeddings(audio_model, clip_model, loader, device)

    sim_a2i = audio_embeds @ image_embeds.T
    sim_i2a = image_embeds @ audio_embeds.T

    label = "ALL" if lang_filter is None else lang_filter.upper()
    print_metrics(f"CNN {label} Audio -> Image Retrieval", metrics_from_ranks(ranks_from_similarity(sim_a2i)))
    print_metrics(f"CNN {label} Image -> Audio Retrieval", metrics_from_ranks(ranks_from_similarity(sim_i2a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",   type=str, required=True,
                    help="Path to coco_train_tts_multilingual_val.jsonl")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to audio_encoder_cnn_multilingual.pt")
    args = ap.parse_args()

    for lang in ("en", "hi", "bn", None):
        evaluate_subset(args.manifest, args.checkpoint, lang_filter=lang)


if __name__ == "__main__":
    main()