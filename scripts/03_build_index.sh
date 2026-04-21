#!/usr/bin/env bash
# =============================================================================
# 03_build_index.sh
# Stage 3: Build CLIP image index and keyword tag index
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TRAIN_MANIFEST="$PROJECT_DIR/coco_train_tts_multilingual_train.jsonl"
IMAGE_INDEX="$PROJECT_DIR/image_index.pt"
TAG_INDEX="$PROJECT_DIR/coco_tags.json"

echo "============================================================"
echo "  Stage 3: Building Retrieval Indexes"
echo "  Project dir: $PROJECT_DIR"
echo "============================================================"

cd "$PROJECT_DIR"

# ── Step 1: CLIP image index ─────────────────────────────────────────────────
echo ""
echo "[1/2] Building CLIP image embedding index…"
python retrieval/build_image_index.py \
  --manifest "$TRAIN_MANIFEST" \
  --output   "$IMAGE_INDEX"
echo "  ✓ Saved: $IMAGE_INDEX"

# ── Step 2: Tag index ─────────────────────────────────────────────────────────
echo ""
echo "[2/2] Building keyword tag index…"
python retrieval/build_tag_index.py \
  --manifest "$TRAIN_MANIFEST" \
  --output   "$TAG_INDEX"
echo "  ✓ Saved: $TAG_INDEX"

echo ""
echo "============================================================"
echo "  Stage 3 COMPLETE"
echo "  Image index : $IMAGE_INDEX"
echo "  Tag index   : $TAG_INDEX"
echo "============================================================"