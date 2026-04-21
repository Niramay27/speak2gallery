#!/usr/bin/env bash
# =============================================================================
# 02_train.sh
# Stage 2: Train the audio encoder
#
# Usage:
#   MODEL=whisper ./scripts/02_train.sh      # default — best model
#   MODEL=cnn     ./scripts/02_train.sh      # CNN encoder
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Manifests and output live in the project root
TRAIN_MANIFEST="$PROJECT_DIR/coco_train_tts_multilingual_train.jsonl"
MODEL="${MODEL:-whisper}"   # whisper | cnn

echo "============================================================"
echo "  Stage 2: Training Audio Encoder"
echo "  Model      : $MODEL"
echo "  Project dir: $PROJECT_DIR"
echo "  Manifest   : $TRAIN_MANIFEST"
echo "============================================================"

cd "$PROJECT_DIR"

if [ "$MODEL" = "whisper" ]; then
    echo ""
    echo "[1/1] Training Whisper (small) + projection head — multilingual…"
    python training/train_bridge_multilingual_whisper.py \
        --manifest "$TRAIN_MANIFEST" \
        --output   "$PROJECT_DIR/audio_encoder_multilingual.pt"
    echo ""
    echo "  ✓ Saved: $PROJECT_DIR/audio_encoder_multilingual.pt"

elif [ "$MODEL" = "cnn" ]; then
    echo ""
    echo "[1/1] Training CNN encoder — multilingual…"
    python training/train_bridge_multilingual_cnn.py \
        --manifest "$TRAIN_MANIFEST" \
        --output   "$PROJECT_DIR/audio_encoder_cnn_multilingual.pt"
    echo ""
    echo "  ✓ Saved: $PROJECT_DIR/audio_encoder_cnn_multilingual.pt"

else
    echo "ERROR: Unknown MODEL='$MODEL'. Use 'whisper' or 'cnn'."
    exit 1
fi

echo ""
echo "============================================================"
echo "  Stage 2 COMPLETE"
echo "============================================================"