#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieve_from_audio.py  — hybrid pipeline
==========================================
  Audio → mel → WhisperAudioEncoder → audio_emb  (semantic path)
  ASR text → QueryProcessor → keywords            (tag-filter path)
  Hybrid fusion → top-K image paths

Usage:
    python retrieval/retrieve_from_audio.py \
        --audio     query.wav \
        --index     image_index.pt \
        --checkpoint audio_encoder_multilingual.pt \
        [--tag_index coco_tags.json] \
        [--top_k 9] \
        [--alpha 0.7]
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import torchaudio

# Allow imports from project root when called from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.models_whisper import WhisperAudioEncoder
from retrieval.query_processor import QueryProcessor
from retrieval.hybrid_retrieval import HybridRetriever


# ─── Audio helpers ────────────────────────────────────────────────────────────

def load_and_preprocess_audio(audio_path: str, sample_rate: int = 16000,
                               n_mels: int = 80) -> torch.Tensor:
    """Load audio file → normalised mel spectrogram [1, n_mels, T]."""
    wav, sr = torchaudio.load(audio_path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=n_mels)
    db  = torchaudio.transforms.AmplitudeToDB(stype="power")
    mel = db(melspec(wav))
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel   # [1, n_mels, T']


def transcribe_audio(audio_path: str, model_size: str = "small") -> dict:
    try:
        import whisper
        model  = whisper.load_model(model_size)
        result = model.transcribe(audio_path, fp16=False)
        return {"text": result["text"].strip(), "language": result.get("language", "en")}
    except ImportError:
        print("[ASR] openai-whisper not installed. Install: pip install openai-whisper")
        return {"text": "", "language": "en"}
    except Exception as e:
        print(f"[ASR] Transcription error: {e}")
        return {"text": "", "language": "en"}


# ─── Main retrieval function ─────────────────────────────────────────────────

@torch.no_grad()
def retrieve_from_audio(
    audio_path: str,
    image_index_path: str,
    checkpoint_path: str,
    tag_index_path: str = None,
    top_k: int = 9,
    alpha: float = 0.7,
    whisper_name: str = "openai/whisper-small",
    embed_dim: int = 512,
    n_mels: int = 80,
    translate: bool = False,
    whisper_asr_size: str = "small",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[retrieve_from_audio] Device: {device}\n")

    print(f"[Step 1/5] ASR transcription of: {audio_path}")
    asr_result    = transcribe_audio(audio_path, model_size=whisper_asr_size)
    transcribed   = asr_result["text"]
    detected_lang = asr_result.get("language", "en")
    print(f"  → '{transcribed}'  (lang={detected_lang})")

    print("\n[Step 2/5] Normalising query & extracting components…")
    qp = QueryProcessor(translate=translate, use_spacy=True)
    qc = qp.process(transcribed, src_lang=detected_lang)
    print(f"  normalized : '{qc.normalized_text}'")
    print(f"  objects    : {qc.objects}")
    print(f"  attributes : {qc.attributes}")
    print(f"  relations  : {qc.relations}")
    print(f"  keywords   : {qc.all_keywords}")

    print("\n[Step 3/5] Computing audio embedding…")
    mel = load_and_preprocess_audio(audio_path, n_mels=n_mels).to(device)

    audio_model = WhisperAudioEncoder(
        whisper_name=whisper_name, embed_dim=embed_dim, freeze_encoder=True
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    audio_model.load_state_dict(state)
    audio_model.eval()

    audio_emb = F.normalize(audio_model(mel), dim=-1)
    print(f"  Audio embedding shape: {tuple(audio_emb.shape)}")

    print(f"\n[Step 4/5] Hybrid retrieval  (alpha={alpha})…")
    retriever = HybridRetriever(
        image_index_path=image_index_path,
        tag_index_path=tag_index_path,
        alpha=alpha,
        device=device,
    )
    results = retriever.retrieve_from_audio_embedding(audio_emb.cpu(), qc, top_k=top_k)

    print(f"\n[Step 5/5] Top-{top_k} results:")
    print(f"  {'Rank':>4}  {'Semantic':>8}  {'Tag':>6}  {'Final':>7}  Path")
    print("  " + "-" * 72)
    for r in results:
        path_short = r["image_path"][-60:] if len(r["image_path"]) > 60 else r["image_path"]
        print(f"  #{r['rank']:>3}  {r['semantic_score']:>8.4f}  "
              f"{r['tag_score']:>6.4f}  {r['final_score']:>7.4f}  …{path_short}")

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Hybrid audio-to-image retrieval (semantic + tag filtering)")
    p.add_argument("--audio",        required=True)
    p.add_argument("--index",        required=True)
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--tag_index",    default=None)
    p.add_argument("--top_k",        type=int,   default=9)
    p.add_argument("--alpha",        type=float, default=0.7)
    p.add_argument("--whisper_name", default="openai/whisper-small")
    p.add_argument("--whisper_asr",  default="small")
    p.add_argument("--embed_dim",    type=int,   default=512)
    p.add_argument("--translate",    action="store_true")
    args = p.parse_args()

    retrieve_from_audio(
        audio_path       = args.audio,
        image_index_path = args.index,
        checkpoint_path  = args.checkpoint,
        tag_index_path   = args.tag_index,
        top_k            = args.top_k,
        alpha            = args.alpha,
        whisper_name     = args.whisper_name,
        embed_dim        = args.embed_dim,
        translate        = args.translate,
        whisper_asr_size = args.whisper_asr,
    )


if __name__ == "__main__":
    main()