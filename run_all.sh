#!/usr/bin/env bash
# =============================================================================
# run_all.sh
# End-to-end pipeline: dataset → train → index → eval → app
#
# Usage:
#   chmod +x run_all.sh scripts/*.sh
#   ./run_all.sh
#
# To skip completed stages, comment them out below.
# =============================================================================

set -euo pipefail

# Always run from the project root regardless of where the script is called from
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Speak2Gallery — Full Pipeline Runner                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Project dir: $PROJECT_DIR"
echo ""

# ── Stage 1: Build dataset ───────────────────────────────────────────────────
echo "► Stage 1/5: Building multilingual dataset…"
bash scripts/01_build_dataset.sh
echo ""

# ── Stage 2: Train best model (Whisper) ─────────────────────────────────────
echo "► Stage 2/5: Training Whisper audio encoder…"
MODEL=whisper bash scripts/02_train.sh
echo ""

# ── Stage 3: Build indexes ───────────────────────────────────────────────────
echo "► Stage 3/5: Building retrieval indexes…"
bash scripts/03_build_index.sh
echo ""

# ── Stage 4: Evaluate ────────────────────────────────────────────────────────
echo "► Stage 4/5: Running evaluation…"
MODEL=whisper bash scripts/04_evaluate.sh
echo ""

# ── Stage 5: Launch app ──────────────────────────────────────────────────────
echo "► Stage 5/5: Launching web application…"
bash scripts/05_run_app.sh