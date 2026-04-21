#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a multilingual COCO-TTS JSONL manifest from an existing English manifest.

Input JSONL format (existing):
{"image_path": "...", "caption": "...", "audio_path": "..."}

Output JSONL format (multilingual, one row per language):
{"image_path": "...", "lang": "en", "caption": "...", "english_caption": "...", "audio_path": "..."}
{"image_path": "...", "lang": "hi", "caption": "...", "english_caption": "...", "audio_path": "..."}
{"image_path": "...", "lang": "bn", "caption": "...", "english_caption": "...", "audio_path": "..."}

Features:
- Reads your existing English manifest
- Keeps English rows
- Translates captions to Hindi and Bengali using NLLB
- Generates Hindi/Bengali TTS audio if requested
- Resume-safe: skips audio files that already exist
- Retry logic for flaky TTS commands

python make_multilingual_coco_tts_manifest.py \
  --input_manifest_jsonl /home/other/Niramay/DL/coco_train_tts.jsonl \
  --out_manifest_jsonl /home/other/Niramay/DL/coco_train_tts_multilingual.jsonl \
  --out_audio_dir /home/other/Niramay/DL/tts_wavs_multilingual \
  --include_english \
  --limit 5000 \
  --tts_cmd_hi 'edge-tts --text "{text}" --voice hi-IN-SwaraNeural --write-media "{out_wav}"' \
  --tts_cmd_bn 'edge-tts --text "{text}" --voice bn-IN-TanishaaNeural --write-media "{out_wav}"' \
  --skip_missing_images
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# -----------------------------
# Helpers
# -----------------------------

def short_hash(text: str, n: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def render_tts_cmd(template: str, text: str, out_wav: str) -> str:
    safe_text = text.replace('"', '\\"')
    return template.format(text=safe_text, out_wav=out_wav)


def maybe_generate_audio(
    tts_cmd_template: str | None,
    caption: str,
    out_wav: str,
    max_retries: int = 5,
    retry_sleep: float = 5.0,
    continue_on_error: bool = True,
    success_sleep: float = 0.5,
):
    if tts_cmd_template is None:
        return False

    cmd = render_tts_cmd(tts_cmd_template, caption, out_wav)

    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(cmd, shell=True, check=True)
            time.sleep(success_sleep)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[TTS ERROR] attempt {attempt}/{max_retries} failed")
            print(f"Command: {cmd}")
            print(f"Exit code: {e.returncode}")

            if attempt < max_retries:
                print(f"Sleeping {retry_sleep} seconds before retry...\n")
                time.sleep(retry_sleep)
            else:
                print("[TTS ERROR] Max retries reached.")
                if continue_on_error:
                    print(f"[TTS SKIP] Skipping caption: {caption}\n")
                    return False
                raise


# -----------------------------
# Translator using NLLB
# -----------------------------

class NLLBTranslator:
    """
    Uses facebook/nllb-200-distilled-600M for translation.

    Language codes:
      English: eng_Latn
      Hindi:   hin_Deva
      Bengali: ben_Beng
    """
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", device=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def translate(self, text: str, src_lang: str, tgt_lang: str, max_new_tokens: int = 128) -> str:
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
        )
        out = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return out.strip()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_manifest_jsonl", type=str, required=True,
                    help="Existing English JSONL manifest.")
    ap.add_argument("--out_manifest_jsonl", type=str, required=True,
                    help="Output multilingual JSONL manifest.")
    ap.add_argument("--out_audio_dir", type=str, required=True,
                    help="Directory to store generated Hindi/Bengali audio.")

    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--include_english", action="store_true",
                    help="Include English rows from input manifest in output.")
    ap.add_argument("--skip_missing_images", action="store_true")

    ap.add_argument("--skip_translation", action="store_true",
                    help="If set, does not translate. Useful for debugging.")
    ap.add_argument("--skip_audio_generation", action="store_true",
                    help="If set, never run TTS even if TTS commands are provided.")

    ap.add_argument("--translate_model", type=str,
                    default="facebook/nllb-200-distilled-600M")

    # Separate TTS commands per language
    ap.add_argument("--tts_cmd_hi", type=str, default=None,
                    help='Shell template with {text} and {out_wav} for Hindi TTS.')
    ap.add_argument("--tts_cmd_bn", type=str, default=None,
                    help='Shell template with {text} and {out_wav} for Bengali TTS.')

    # Retry params
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--retry_sleep", type=float, default=5.0)
    ap.add_argument("--success_sleep", type=float, default=0.5)

    args = ap.parse_args()

    out_audio_dir = Path(args.out_audio_dir)
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    hi_audio_dir = out_audio_dir / "hi"
    bn_audio_dir = out_audio_dir / "bn"
    hi_audio_dir.mkdir(parents=True, exist_ok=True)
    bn_audio_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input_manifest_jsonl)

    translator = None
    if not args.skip_translation:
        print(f"Loading translator model: {args.translate_model}")
        translator = NLLBTranslator(model_name=args.translate_model)

    out_manifest = Path(args.out_manifest_jsonl)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_hi_audio = 0
    n_bn_audio = 0
    n_skipped_img = 0

    with open(out_manifest, "w", encoding="utf-8") as w:
        for idx, row in enumerate(rows):
            image_path = Path(row["image_path"])
            english_caption = row["caption"].strip()
            english_audio_path = row["audio_path"]

            if not image_path.exists():
                if args.skip_missing_images:
                    n_skipped_img += 1
                    continue
                raise FileNotFoundError(f"Missing image: {image_path}")

            # Stable uid from original English caption
            uid = f"{image_path.stem}_{short_hash(english_caption)}"

            # 1) English row
            if args.include_english:
                en_row = {
                    "image_path": str(image_path),
                    "lang": "en",
                    "caption": english_caption,
                    "english_caption": english_caption,
                    "audio_path": english_audio_path,
                }
                w.write(json.dumps(en_row, ensure_ascii=False) + "\n")
                n_written += 1

            # 2) Hindi row
            if not args.skip_translation:
                hi_caption = translator.translate(
                    english_caption,
                    src_lang="eng_Latn",
                    tgt_lang="hin_Deva"
                )
            else:
                hi_caption = english_caption

            hi_audio_path = hi_audio_dir / f"{uid}_hi.wav"

            if (not args.skip_audio_generation) and args.tts_cmd_hi is not None and (not hi_audio_path.exists()):
                generated = maybe_generate_audio(
                    args.tts_cmd_hi,
                    hi_caption,
                    str(hi_audio_path),
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    success_sleep=args.success_sleep,
                    continue_on_error=True,
                )
                if generated:
                    n_hi_audio += 1

            hi_row = {
                "image_path": str(image_path),
                "lang": "hi",
                "caption": hi_caption,
                "english_caption": english_caption,
                "audio_path": str(hi_audio_path),
            }
            w.write(json.dumps(hi_row, ensure_ascii=False) + "\n")
            n_written += 1

            # 3) Bengali row
            if not args.skip_translation:
                bn_caption = translator.translate(
                    english_caption,
                    src_lang="eng_Latn",
                    tgt_lang="ben_Beng"
                )
            else:
                bn_caption = english_caption

            bn_audio_path = bn_audio_dir / f"{uid}_bn.wav"

            if (not args.skip_audio_generation) and args.tts_cmd_bn is not None and (not bn_audio_path.exists()):
                generated = maybe_generate_audio(
                    args.tts_cmd_bn,
                    bn_caption,
                    str(bn_audio_path),
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    success_sleep=args.success_sleep,
                    continue_on_error=True,
                )
                if generated:
                    n_bn_audio += 1

            bn_row = {
                "image_path": str(image_path),
                "lang": "bn",
                "caption": bn_caption,
                "english_caption": english_caption,
                "audio_path": str(bn_audio_path),
            }
            w.write(json.dumps(bn_row, ensure_ascii=False) + "\n")
            n_written += 1

            if args.limit and (idx + 1) >= args.limit:
                break

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} source rows...")

    print(f"\n✅ Wrote multilingual manifest: {out_manifest}")
    print(f"   Rows written: {n_written}")
    print(f"   Hindi audio generated this run: {n_hi_audio}")
    print(f"   Bengali audio generated this run: {n_bn_audio}")
    if n_skipped_img:
        print(f"   Skipped missing images: {n_skipped_img}")


if __name__ == "__main__":
    main()