#!/usr/bin/env bash
# =============================================================================
# 01_build_dataset.sh
# Stage 1: Build the multilingual COCO-TTS dataset
#
# Usage:
#   chmod +x scripts/01_build_dataset.sh
#   ./scripts/01_build_dataset.sh
#
# Edit the CONFIG section below (COCO paths and LIMIT) before running.
# OUT_DIR is set automatically to the project root (where this repo lives).
# =============================================================================

set -euo pipefail

# ── Resolve project root (directory containing this scripts/ folder) ──────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# =============================================================================
# CONFIG — edit COCO paths and LIMIT before running
# =============================================================================
# Path to your COCO train2017 images directory
COCO_IMAGES="/home/other/coco_dataset/train2017"
# Path to the COCO captions annotation JSON
COCO_CAPTIONS="/home/other/coco_dataset/annotations/captions_train2017.json"

# Number of COCO captions to use (0 = all ~591k)
LIMIT=10

# TTS voices (edge-tts must be installed: pip install edge-tts)
TTS_CMD_EN='edge-tts --text "{text}" --write-media "{out_wav}"'
TTS_CMD_HI='edge-tts --text "{text}" --voice hi-IN-SwaraNeural --write-media "{out_wav}"'
TTS_CMD_BN='edge-tts --text "{text}" --voice bn-IN-TanishaaNeural --write-media "{out_wav}"'
# =============================================================================

# All output lives inside the project root
OUT_DIR="$PROJECT_DIR"

echo "============================================================"
echo "  Stage 1: Building Multilingual COCO-TTS Dataset"
echo "============================================================"
echo "  Project dir : $PROJECT_DIR"
echo "  COCO images : $COCO_IMAGES"
echo "  COCO caps   : $COCO_CAPTIONS"
echo "  Output dir  : $OUT_DIR"
echo "  Limit       : $LIMIT captions"
echo "============================================================"

cd "$PROJECT_DIR"

# ── Step 1: English manifest + TTS ──────────────────────────────────────────
echo ""
echo "[1/4] Generating English TTS manifest…"
python data/make_coco_tts_manifest.py \
  --coco_images_dir   "$COCO_IMAGES" \
  --coco_captions_json "$COCO_CAPTIONS" \
  --out_audio_dir     "$OUT_DIR/tts_wavs" \
  --out_manifest_jsonl "$OUT_DIR/coco_train_tts.jsonl" \
  --limit             "$LIMIT" \
  --tts_cmd           "$TTS_CMD_EN" \
  --skip_missing_images
echo "  ✓ English manifest + audio done."

# ── Step 2: Multilingual manifest (Hindi + Bengali) ──────────────────────────
echo ""
echo "[2/4] Generating multilingual manifest (Hindi + Bengali)…"
python data/make_multilingual_coco_tts_manifest.py \
  --input_manifest_jsonl "$OUT_DIR/coco_train_tts.jsonl" \
  --out_manifest_jsonl   "$OUT_DIR/coco_train_tts_multilingual.jsonl" \
  --out_audio_dir        "$OUT_DIR/tts_wavs_multilingual" \
  --include_english \
  --limit                "$LIMIT" \
  --tts_cmd_hi           "$TTS_CMD_HI" \
  --tts_cmd_bn           "$TTS_CMD_BN" \
  --skip_missing_images
echo "  ✓ Multilingual manifest + audio done."

# ── Step 3: Split English dataset ────────────────────────────────────────────
echo ""
echo "[3/4] Splitting English manifest into train/val…"
python data/split.py --out_dir "$OUT_DIR"
echo "  ✓ English split done."

# ── Step 4: Split multilingual dataset ───────────────────────────────────────
echo ""
echo "[4/4] Splitting multilingual manifest into train/val…"
python data/split_multilingual_manifest.py --out_dir "$OUT_DIR"
echo "  ✓ Multilingual split done."

echo ""
echo "============================================================"
echo "  Stage 1 COMPLETE"
echo "============================================================"
echo "  Output files in: $OUT_DIR"
echo "    coco_train_tts.jsonl"
echo "    coco_train_tts_multilingual.jsonl"
echo "    coco_train_tts_train.jsonl / _val.jsonl"
echo "    coco_train_tts_multilingual_train.jsonl / _val.jsonl"
echo "    tts_wavs/                   (English audio)"
echo "    tts_wavs_multilingual/hi/   (Hindi audio)"
echo "    tts_wavs_multilingual/bn/   (Bengali audio)"
echo "============================================================"