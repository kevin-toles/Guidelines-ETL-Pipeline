#!/bin/bash
#
# run_mining_pipeline.command — macOS double-click launcher
#
# Opens Terminal, runs the full mining pipeline with resume support.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "══════════════════════════════════════════════════════════════════"
echo "  Guideline Mining Pipeline — Full Run (with resume)"
echo "  Started: $(date)"
echo "══════════════════════════════════════════════════════════════════"
echo ""

bash full_mining_pipeline.sh --resume

EXIT_CODE=$?
echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ Pipeline completed successfully at $(date)"
else
    echo "❌ Pipeline failed (exit $EXIT_CODE) at $(date)"
    echo "   Fix the issue and re-run — it will resume from the failed step."
fi

echo ""
echo "Press Enter to close this window..."
read
