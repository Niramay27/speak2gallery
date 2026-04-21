#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a COCO-TTS JSONL manifest:

Each line:
{"image_path": ".../000000123.jpg", "caption": "...", "audio_path": ".../tts/123_<hash>.wav"}

Inputs:
- coco_images_dir: COCO train2017/val2017 folder containing images
- coco_captions_json: COCO captions annotations json (captions_train2017.json etc.)
- out_audio_dir: where audio wavs are (or will be generated)
- out_manifest_jsonl: output jsonl file path

Optional:
- --tts_cmd: template command to generate audio. Must contain {text} and {out_wav}
  Example (Piper):
    --tts_cmd 'echo "{text}" | piper --model en_US-lessac-medium --output_file "{out_wav}"'

If --tts_cmd is omitted, the script will ONLY write the manifest and will not generate audio.

python /home/other/Niramay/DL/make_coco_tts_manifest.py \
  --coco_images_dir /home/other/coco_dataset/train2017 \
  --coco_captions_json /home/other/coco_dataset/annotations/captions_train2017.json \
  --out_audio_dir /home/other/Niramay/DL/tts_wavs \
  --out_manifest_jsonl /home/other/Niramay/DL/coco_train_tts.jsonl \
  --limit 5000 \
  --tts_cmd 'edge-tts --text "{text}" --write-media "{out_wav}"' \
  --skip_missing_images
  
python scripts/make_coco_tts_manifest.py \
  --coco_images_dir /path/to/coco/train2017 \
  --coco_captions_json /path/to/annotations/captions_train2017.json \
  --out_audio_dir /path/to/tts_wavs \
  --out_manifest_jsonl data/manifests/coco_train_tts.jsonl \
  --limit 5000 \
  --tts_cmd 'echo "{text}" | piper --model en_US-lessac-medium --output_file "{out_wav}"' \
  --skip_missing_images
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def load_coco(coco_captions_json: str):
    with open(coco_captions_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # image_id -> file_name
    img_map = {img["id"]: img["file_name"] for img in coco["images"]}
    anns = coco["annotations"]
    return img_map, anns


def render_tts_cmd(template: str, text: str, out_wav: str) -> str:
    # Escape quotes minimally for shell templates.
    safe_text = text.replace('"', '\\"')
    return template.format(text=safe_text, out_wav=out_wav)


def maybe_generate_audio(tts_cmd_template: str | None, caption: str, out_wav: str):
    if tts_cmd_template is None:
        return  # do nothing

    cmd = render_tts_cmd(tts_cmd_template, caption, out_wav)
    # Use shell=True because user passes a shell pipeline template (e.g., echo | piper)
    subprocess.run(cmd, shell=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_images_dir", type=str, required=True)
    ap.add_argument("--coco_captions_json", type=str, required=True)
    ap.add_argument("--out_audio_dir", type=str, required=True)
    ap.add_argument("--out_manifest_jsonl", type=str, required=True)

    ap.add_argument("--split", type=str, default="", help="Optional label like train2017/val2017")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--skip_missing_images", action="store_true")
    ap.add_argument("--skip_audio_generation", action="store_true",
                    help="If set, never run TTS even if --tts_cmd is provided.")
    ap.add_argument("--tts_cmd", type=str, default=None,
                    help='Optional shell template with {text} and {out_wav} placeholders.')

    args = ap.parse_args()

    coco_images_dir = Path(args.coco_images_dir)
    out_audio_dir = Path(args.out_audio_dir)
    out_audio_dir.mkdir(parents=True, exist_ok=True)

    img_map, anns = load_coco(args.coco_captions_json)

    out_manifest = Path(args.out_manifest_jsonl)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped_img = 0
    n_audio_gen = 0

    with open(out_manifest, "w", encoding="utf-8") as w:
        for ann in anns:
            image_id = ann["image_id"]
            caption = ann["caption"].strip()

            file_name = img_map.get(image_id)
            if file_name is None:
                continue

            image_path = coco_images_dir / file_name
            if not image_path.exists():
                if args.skip_missing_images:
                    n_skipped_img += 1
                    continue
                else:
                    raise FileNotFoundError(f"Missing image: {image_path}")

            # Stable-ish audio filename per (image_id, caption)
            uid = f"{image_id}_{short_hash(caption)}"
            audio_path = out_audio_dir / f"{uid}.wav"

            # Generate audio if requested and needed
            if (not args.skip_audio_generation) and args.tts_cmd is not None and (not audio_path.exists()):
                maybe_generate_audio(args.tts_cmd, caption, str(audio_path))
                n_audio_gen += 1

            row = {
                "image_path": str(image_path),
                "caption": caption,
                "audio_path": str(audio_path),
            }
            if args.split:
                row["split"] = args.split

            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

            if args.limit and n_written >= args.limit:
                break

    print(f"✅ Wrote: {out_manifest}")
    print(f"   Lines: {n_written}")
    if n_skipped_img:
        print(f"   Skipped missing images: {n_skipped_img}")
    if args.tts_cmd is not None:
        print(f"   Audio generated this run: {n_audio_gen}")
        print(f"   Audio dir: {out_audio_dir}")


if __name__ == "__main__":
    main()