#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the Whisper audio encoder (multilingual) — BEST MODEL.

Usage:
    python training/train_bridge_multilingual_whisper.py \
        --manifest /path/to/coco_train_tts_multilingual_train.jsonl \
        --output   /path/to/audio_encoder_multilingual.pt
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm
from pathlib import Path
import sys

# Allow imports from project root when called from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_coco_tts_multilingual import CocoTtsMultilingual, collate_coco_tts_multilingual
from models.models_whisper import WhisperAudioEncoder


def contrastive_loss(a, b, logit_scale):
    logits = logit_scale.exp() * (a @ b.T)
    labels = torch.arange(a.size(0), device=a.device)
    loss_a2b = F.cross_entropy(logits, labels)
    loss_b2a = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_a2b + loss_b2a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True,
                    help="Path to coco_train_tts_multilingual_train.jsonl")
    ap.add_argument("--output", type=str, required=True,
                    help="Output path for trained checkpoint .pt")
    ap.add_argument("--whisper_name", type=str, default="openai/whisper-small")
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"Model      : {args.whisper_name}")
    print(f"Manifest   : {args.manifest}")
    print(f"Output     : {args.output}")
    print(f"Epochs     : {args.epochs}")

    # ── CLIP (frozen) ──
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # ── Dataset / DataLoader ──
    train_ds = CocoTtsMultilingual(
        manifest_jsonl=args.manifest,
        clip_preprocess=clip_preprocess,
        clip_tokenizer=clip_tokenizer,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_coco_tts_multilingual,
    )

    # ── Audio encoder ──
    audio_model = WhisperAudioEncoder(
        whisper_name=args.whisper_name,
        embed_dim=args.embed_dim,
        freeze_encoder=True,
    ).to(device)

    # Only train the projection head (Whisper encoder is frozen)
    trainable_params = [p for p in audio_model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    logit_scale = torch.nn.Parameter(torch.tensor(2.6592, device=device))

    # ── Training loop ──
    for epoch in range(args.epochs):
        audio_model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            mel         = batch["mel"].to(device)
            text_tokens = batch["text_tokens"].to(device)

            with torch.no_grad():
                text_emb = clip_model.encode_text(text_tokens)
                text_emb = F.normalize(text_emb, dim=-1)

            audio_emb = audio_model(mel)
            # audio_emb already normalized inside WhisperAudioEncoder

            loss = contrastive_loss(audio_emb, text_emb, logit_scale)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss = {avg_loss:.4f}")

    # ── Save ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(audio_model.state_dict(), out_path)
    print(f"\nSaved Whisper model to: {out_path}")


if __name__ == "__main__":
    main()