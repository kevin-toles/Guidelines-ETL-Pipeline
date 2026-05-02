#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# StackExchange ETL Pipeline — Shell Wrapper
# Launches a new Terminal.app window to show live progress.
# Logs full output to ./output/pipeline.log for post-hoc review.
#
# Usage:
#   # First run (skips stackoverflow.com >20GB, processes 110 sites)
#   bash scripts/etl/run_stackexchange_etl.sh
#
#   # Resume from where you left off
#   bash scripts/etl/run_stackexchange_etl.sh --resume
#
#   # Dry-run (preview stats without writing)
#   bash scripts/etl/run_stackexchange_etl.sh --dry-run
#
#   # Skip large archives >5 GB
#   bash scripts/etl/run_stackexchange_etl.sh --skip-larger-than 5 --resume
# ═══════════════════════════════════════════════════════════════════

# NOTE: no set -e because the Python pipeline handles errors internally
# and continues to the next site on failure
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARCHIVE_DIR="/Volumes/USB321FD/Guidelines ETL Data/stackexchange"
OUTPUT_DIR="$PROJECT_DIR/output"
LOG_FILE="$OUTPUT_DIR/pipeline.log"
STAGES="1-5"
DRY_RUN=""
RESUME=""
SKIP_LARGER="20"
SITES=""

# Parse wrapper args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)   RESUME="--resume"; shift ;;
        --dry-run)  DRY_RUN="--dry-run"; STAGES="1-4"; shift ;;
        --sites)    SITES="--sites $2"; shift 2 ;;
        --skip-larger-than)
            SKIP_LARGER="$2"; shift 2 ;;
        --stages)
            STAGES="$2"; shift 2 ;;
        *)  echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Build the Python command (unbuffered for live output)
CMD="PYTHONUNBUFFERED=1 python3 \"$PROJECT_DIR/scripts/etl/process_stackexchange.py\""
CMD="$CMD --archive-dir \"$ARCHIVE_DIR\""
CMD="$CMD --output-dir \"$OUTPUT_DIR\""
CMD="$CMD --stages $STAGES"
CMD="$CMD --skip-larger-than $SKIP_LARGER"
[[ -n "$SITES" ]] && CMD="$CMD $SITES"
[[ -n "$RESUME" ]] && CMD="$CMD $RESUME"
[[ -n "$DRY_RUN" ]] && CMD="$CMD $DRY_RUN"

mkdir -p "$OUTPUT_DIR"

echo "================================================"
echo "  StackExchange ETL Pipeline"
echo "================================================"
echo "  Project:  $PROJECT_DIR"
echo "  Archives: $ARCHIVE_DIR"
echo "  Output:   $OUTPUT_DIR"
echo "  Log:      $LOG_FILE"
echo "  Stages:   $STAGES"
echo "  Resume:   ${RESUME:-no}"
echo "  Dry-run:  ${DRY_RUN:-no}"
echo "  Skip >    ${SKIP_LARGER} GB"
echo "================================================"
echo ""
echo "Opening new Terminal window for live progress..."

# Save the full command to a temp script (avoids quote escaping hell with osascript)
RUN_SCRIPT="/tmp/run_etl_pipeline.sh"
cat > "$RUN_SCRIPT" << RUNEOF
#!/bin/bash
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           StackExchange ETL Pipeline — Live Output            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Archives: $ARCHIVE_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Log:      $LOG_FILE"
echo ""
$CMD 2>&1 | tee "$LOG_FILE"
EXIT_CODE=\${PIPESTATUS[0]}
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Pipeline finished with exit code \$EXIT_CODE"
echo "  Full log: $LOG_FILE"
echo "══════════════════════════════════════════════════════════════════"
read -p "Press Enter to close this window..."
exit \$EXIT_CODE
RUNEOF

chmod +x "$RUN_SCRIPT"

# Launch in a new Terminal.app window
osascript -e "tell app \"Terminal\" to do script \"$RUN_SCRIPT\""

echo ""
echo "Pipeline launched in new Terminal window."
echo "Monitor progress: tail -f $LOG_FILE"
echo "Resume if interrupted: bash scripts/etl/run_stackexchange_etl.sh --resume"
