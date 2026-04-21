#!/usr/bin/env bash
# =============================================================================
# 05_run_app.sh
# Stage 5: Launch the Speak2Gallery web application
#
# Usage:
#   ./scripts/05_run_app.sh                         # HTTP on localhost:5000
#   SSL_ADHOC=1 APP_PUBLIC_HOST=192.168.1.5 \
#     ./scripts/05_run_app.sh                        # HTTPS (self-signed)
#
# Microphone access requires HTTPS when connecting from any non-localhost browser.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Export settings ───────────────────────────────────────────────────────────
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-5000}"
export APP_PUBLIC_HOST="${APP_PUBLIC_HOST:-localhost}"
export SSL_ADHOC="${SSL_ADHOC:-0}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-}"
export SSL_KEY_FILE="${SSL_KEY_FILE:-}"
export FORCE_HTTPS="${FORCE_HTTPS:-0}"

# Let app.py know where the project root is (models, indexes, etc.)
export PROJECT_DIR="$PROJECT_DIR"

SCHEME="http"
if [ "$SSL_ADHOC" = "1" ] || [ -n "$SSL_CERT_FILE" ]; then
    SCHEME="https"
fi

echo "============================================================"
echo "  Speak2Gallery Web Application"
echo "============================================================"
echo "  Project dir : $PROJECT_DIR"
echo "  URL         : $SCHEME://$APP_PUBLIC_HOST:$PORT"
echo "  SSL         : $SCHEME"
echo "============================================================"
echo ""
echo "  Mic access requires HTTPS when connecting from remote devices."
echo "  For self-signed HTTPS: SSL_ADHOC=1 APP_PUBLIC_HOST=<ip> $0"
echo ""
echo "  Press Ctrl+C to stop."
echo "============================================================"

# Run app.py from the project root so all relative imports work correctly
cd "$PROJECT_DIR"
SSL_ADHOC=1 APP_PUBLIC_HOST=YOUR_IP python app/app.py