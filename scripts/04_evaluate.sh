#!/usr/bin/env bash
# =============================================================================
# 04_evaluate.sh
# Stage 4: Run full evaluation on the multilingual validation set
#
# Usage:
#   MODEL=whisper ./scripts/04_evaluate.sh    # default
#   MODEL=cnn     ./scripts/04_evaluate.sh
#   MODEL=base    ./scripts/04_evaluate.sh    # random baseline
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

VAL_MANIFEST="$PROJECT_DIR/coco_train_tts_multilingual_val.jsonl"
MODEL="${MODEL:-whisper}"

echo "============================================================"
echo "  Stage 4: Evaluation"
echo "  Model      : $MODEL"
echo "  Project dir: $PROJECT_DIR"
echo "  Val set    : $VAL_MANIFEST"
echo "============================================================"

cd "$PROJECT_DIR"

if [ "$MODEL" = "whisper" ]; then
    echo ""
    echo "Running Whisper multilingual evaluation…"
    python evaluation/eval_retrieval_multilingual_whisper.py \
        --manifest   "$VAL_MANIFEST" \
        --checkpoint "$PROJECT_DIR/audio_encoder_multilingual.pt"

elif [ "$MODEL" = "cnn" ]; then
    echo ""
    echo "Running CNN multilingual evaluation…"
    python evaluation/eval_retrieval_multilingual_cnn.py \
        --manifest   "$VAL_MANIFEST" \
        --checkpoint "$PROJECT_DIR/audio_encoder_cnn_multilingual.pt"

elif [ "$MODEL" = "base" ]; then
    echo ""
    echo "Running random baseline evaluation…"
    python evaluation/eval_retrieval_multilingual_base.py \
        --manifest "$VAL_MANIFEST"

else
    echo "ERROR: Unknown MODEL='$MODEL'. Use whisper, cnn, or base."
    exit 1
fi

echo ""
echo "============================================================"
echo "  Stage 4 COMPLETE"
echo "============================================================"