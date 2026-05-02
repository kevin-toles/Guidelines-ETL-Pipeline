#!/bin/bash
#
# full_collections_pipeline.sh — Guideline Mining Pipeline for Textbook Collections
#
# Processes the platform textbook/research-paper JSON collections
# (software-engineering, ai-safety) through the same A1→A7 → B1→B5
# mining pipeline used for StackExchange data.
#
# PHASES:
#   0. Pre-flight — checks collections dirs, Python deps, scripts
#   1. CONVERT — Convert collections JSON → A1-compatible corpus files
#                 (replaces mine_text_preprocess.py for this data source)
#   2. A2  — Topic Clustering (K-Means, DBSCAN)
#   3. A3  — Topic Modeling (LDA)
#   4. A5  — Anomaly Detection (Isolation Forest, SVM, Z-score)
#   5. A6  — Code Pattern Mining
#   6. A7  — Quality Modeling
#   7. A4  — Association Rules
#   8. B1  — Interpret Clusters
#   9. B2  — Coverage Mapping (against existing guidelines)
#   10. B3 — Enrichment Scoring
#   11. B4 — Gap Quantification
#   12. B5 — Viability Assessment (GO/NO-GO)
#
# Usage:
#   ./scripts/full_collections_pipeline.sh                    # Full run with resume
#   ./scripts/full_collections_pipeline.sh --dry-run           # Validate chain only
#   ./scripts/full_collections_pipeline.sh --force             # Re-run everything
#   ./scripts/full_collections_pipeline.sh --resume            # Skip completed steps
#   ./scripts/full_collections_pipeline.sh --only-safety      # Process only ai-safety
#   ./scripts/full_collections_pipeline.sh --only-se          # Process only software-engineering
#
# Input:  /Users/kevintoles/POC/ai-platform-data/collections/{software-engineering,ai-safety}/raw/
# Output: /Volumes/USB321FD/Guidelines ETL Data/data/collections/mining_output/
# Logs:   /tmp/collections_pipeline_<YYYYMMDD_HHMMSS>/
#
# Requirements:
#   - Python 3.10+ with: scikit-learn, sentence-transformers, numpy,
#     scipy, nltk, gensim, mlxtend, pygments
#   - Platform collection JSON files at the input path above
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Project Paths ───────────────────────────────────────────────────────────
USB="/Volumes/USB321FD/Guidelines ETL Data"
COLLECTIONS_INPUT="/Users/kevintoles/POC/ai-platform-data/collections"
MINING_OUT="$USB/output/collections_mining"
PIPELINE_DIR="$USB/scripts/mining_pipeline"
PYTHON="python3"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ── Options ─────────────────────────────────────────────────────────────────
DRY_RUN=false
FORCE=false
RESUME=true
ONLY_SE=false
ONLY_SAFETY=false
TARGET_COLLECTIONS="software-engineering ai-safety"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)          DRY_RUN=true; RESUME=false;         shift ;;
        --force)            FORCE=true; RESUME=false;           shift ;;
        --resume)           RESUME=true;                        shift ;;
        --only-se)          ONLY_SE=true;                       shift ;;
        --only-safety)      ONLY_SAFETY=true;                   shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$ONLY_SE" == "true" && "$ONLY_SAFETY" == "true" ]]; then
    echo -e "${RED}Error: --only-se and --only-safety are mutually exclusive${NC}"
    exit 1
fi
if [[ "$ONLY_SE" == "true" ]]; then
    TARGET_COLLECTIONS="software-engineering"
fi
if [[ "$ONLY_SAFETY" == "true" ]]; then
    TARGET_COLLECTIONS="ai-safety"
fi

# ── Log Setup ───────────────────────────────────────────────────────────────
LOGDATE=$(date +%Y%m%d_%H%M%S)
LOGDIR="/tmp/collections_pipeline_${LOGDATE}"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/pipeline.log"
exec > >(tee -a "$LOGFILE") 2>&1

# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

phase_header() {
    local label="$1"; shift
    echo ""
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $label${NC}"
    echo -e "${BLUE}  Started: $(date)${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
}

phase_done() {
    local label="$1"; shift
    local exit_code="$1"; shift
    if [[ "$exit_code" -eq 0 ]]; then
        echo -e "  ${GREEN}✓ $label complete ($(date))${NC}"
    else
        echo -e "  ${RED}✗ $label FAILED (exit $exit_code) — ($(date))${NC}"
    fi
}

files_exist() {
    for f in "$@"; do
        if [[ ! -f "$f" ]]; then
            return 1
        fi
    done
    return 0
}

require_inputs() {
    local step="$1"; shift
    local missing=false
    for f in "$@"; do
        if [[ ! -f "$f" ]]; then
            echo -e "  ${RED}✗ Missing input: $f${NC}"
            missing=true
        fi
    done
    if [[ "$missing" == "true" ]]; then
        echo -e "  ${RED}✗ $step: required inputs missing — aborting.${NC}"
        return 1
    fi
    return 0
}

run_step() {
    local step="$1"; shift
    local label="$1"; shift
    local script_name="$1"; shift
    local -a output_files=("${@}")

    # ── Checkpoint ──
    if [[ "$RESUME" == "true" ]]; then
        local all_exist=true
        for f in "${output_files[@]}"; do
            if [[ ! -f "$f" ]]; then
                all_exist=false
                break
            fi
        done
        if [[ "$all_exist" == "true" ]]; then
            echo -e "  ${GREEN}✓ $label — checkpoint found, skipping${NC}"
            return 0
        fi
    fi

    local script_path="$PIPELINE_DIR/$script_name"

    # ── Dry-run ──
    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "  ${CYAN}⏳ $label — would run:${NC}"
        echo "       python3 $script_name $SE_DATA $MINING_OUT"
        echo -e "  ${CYAN}     Outputs:${NC}"
        for f in "${output_files[@]}"; do
            local rel="${f#$MINING_OUT/}"
            if [[ -f "$f" ]]; then
                local size
                size=$(du -h "$f" | cut -f1)
                echo -e "       ${GREEN}✓ $rel ($size, exists)${NC}"
            else
                echo -e "       ${YELLOW}⏳ $rel (will be created)${NC}"
            fi
        done
        return 0
    fi

    # ── Force cleanup ──
    if [[ "$FORCE" == "true" ]]; then
        for f in "${output_files[@]}"; do
            if [[ -f "$f" ]]; then
                rm -f "$f"
                echo -e "  ${YELLOW}⚠ Removed (--force): ${f#$MINING_OUT/}${NC}"
            fi
        done
    fi

    # ── Run ──
    echo -e "  ${CYAN}▶ Running $label...${NC}"
    mkdir -p "$MINING_OUT"
    cd "$USB"

    # Special handling for the converter step (takes different args)
    if [[ "$script_name" == "convert_collections_to_corpus.py" ]]; then
        /usr/bin/time $PYTHON "$script_path" "$COLLECTIONS_INPUT" "$MINING_OUT" 2>&1
    else
        /usr/bin/time $PYTHON "$script_path" "" "$MINING_OUT" 2>&1
    fi
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        echo -e "  ${RED}✗ $label failed (exit $exit_code)${NC}"
        echo -e "  ${RED}  Check logs: $LOGFILE${NC}"
        return $exit_code
    fi

    for f in "${output_files[@]}"; do
        if [[ ! -f "$f" ]]; then
            echo -e "  ${RED}✗ $label completed but output missing: ${f#$MINING_OUT/}${NC}"
            return 1
        fi
        local size
        size=$(du -h "$f" | cut -f1)
        echo -e "  ${GREEN}✓ ${f#$MINING_OUT/} ($size)${NC}"
    done

    return 0
}

# ═════════════════════════════════════════════════════════════════════════════
# PHASE 0: PRE-FLIGHT CHECKS
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   COLLECTIONS MINING — Full Pipeline                            ║${NC}"
echo -e "${BLUE}║   Collections: ${TARGET_COLLECTIONS}${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo " Log dir : $LOGDIR"
echo " Log file: $LOGFILE"
echo ""

phase_header "Phase 0: Pre-flight Checks"

ALL_CHECKS_PASSED=true

# Python
if command -v $PYTHON &>/dev/null; then
    echo -e "  ${GREEN}✓ Python: $($PYTHON --version)${NC}"
else
    echo -e "  ${RED}✗ Python not found${NC}"; ALL_CHECKS_PASSED=false
fi

# Python deps
for pkg in sklearn numpy scipy nltk gensim mlxtend pygments; do
    if $PYTHON -c "import $pkg" 2>/dev/null; then
        :  # ok
    else
        case $pkg in
            sklearn) echo -e "  ${RED}✗ Missing: scikit-learn${NC}"; ALL_CHECKS_PASSED=false ;;
            *)       echo -e "  ${RED}✗ Missing: $pkg${NC}"; ALL_CHECKS_PASSED=false ;;
        esac
    fi
done
echo -e "  ${GREEN}✓ Python dependencies${NC}"

# sentence-transformers (optional)
$PYTHON -c "import sentence_transformers" 2>/dev/null && \
    echo -e "  ${GREEN}✓ sentence-transformers available${NC}" || \
    echo -e "  ${YELLOW}⚠ sentence-transformers not installed — using TF-IDF fallback${NC}"

# Scripts
MISSING_SCRIPTS=false
for s in convert_collections_to_corpus.py mine_topic_clustering.py mine_topic_modeling.py \
         mine_association_rules.py mine_anomaly_detection.py mine_code_patterns.py \
         mine_quality_modeling.py analyze_interpret_clusters.py analyze_coverage_mapping.py \
         analyze_enrichment_scoring.py analyze_gap_quantification.py analyze_viability.py; do
    if [[ ! -f "$PIPELINE_DIR/$s" ]]; then
        echo -e "  ${RED}✗ Missing script: $s${NC}"
        MISSING_SCRIPTS=true
        ALL_CHECKS_PASSED=false
    fi
done
[[ "$MISSING_SCRIPTS" == "false" ]] && echo -e "  ${GREEN}✓ All 12 pipeline scripts found (incl. converter)${NC}"

# Collections JSON files
COLLECTION_OK=true
for coll in $TARGET_COLLECTIONS; do
    COLL_DIR="$COLLECTIONS_INPUT/$coll/raw"
    if [[ -d "$COLL_DIR" ]]; then
        JSON_COUNT=$(find "$COLL_DIR" -name '*.json' ! -name '._*' 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$JSON_COUNT" -gt 0 ]]; then
            echo -e "  ${GREEN}✓ $coll: $JSON_COUNT JSON files${NC}"
        else
            echo -e "  ${RED}✗ $coll: 0 JSON files found${NC}"
            COLLECTION_OK=false
        fi
    else
        echo -e "  ${RED}✗ Collection dir not found: $COLL_DIR${NC}"
        COLLECTION_OK=false
    fi
done
[[ "$COLLECTION_OK" == "false" ]] && ALL_CHECKS_PASSED=false

# ── Summary ──
echo ""
if [[ "$ALL_CHECKS_PASSED" == "true" ]]; then
    echo -e "${GREEN}✅ Phase 0: All pre-flight checks passed.${NC}"
else
    echo -e "${RED}❌ Phase 0: Some checks failed — aborting.${NC}"
    exit 1
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo -e "${YELLOW}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  DRY RUN SUMMARY${NC}"
    echo -e "${YELLOW}══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Pre-flight: ✅ PASS"
    echo "  Collections: $TARGET_COLLECTIONS"
    echo ""
    echo "  Pipeline order:"
    echo "    Step 0: CONVERT (convert_collections_to_corpus.py)"
    echo "    Step 1: A2 — Topic Clustering"
    echo "    Step 2: A3 — Topic Modeling"
    echo "    Step 3: A5 — Anomaly Detection"
    echo "    Step 4: A6 — Code Pattern Mining"
    echo "    Step 5: A7 — Quality Modeling"
    echo "    Step 6: A4 — Association Rules"
    echo "    Step 7: B1 — Interpret Clusters"
    echo "    Step 8: B2 — Coverage Mapping"
    echo "    Step 9: B3 — Enrichment Scoring"
    echo "    Step 10: B4 — Gap Quantification"
    echo "    Step 11: B5 — Viability Assessment"
    echo ""
    echo "  To execute: ./scripts/full_collections_pipeline.sh"
    echo "  To execute (resume): ./scripts/full_collections_pipeline.sh --resume"
    exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

OVERALL_EXIT=0

# ── STEP 0: CONVERT (Collections JSON → A1 corpus) ─────────────────────────
phase_header "CONVERT: Collections JSON → A1 Corpus"
run_step "CONVERT" "Collections Conversion" "convert_collections_to_corpus.py" \
    "$MINING_OUT/A1/text_corpus.jsonl" \
    "$MINING_OUT/A1/code_corpus.jsonl" \
    "$MINING_OUT/A1/tfidf_matrix.npz" \
    "$MINING_OUT/A1/embeddings.npy" \
    "$MINING_OUT/A1/feature_names.npy" \
    "$MINING_OUT/A1/stats.json" \
|| { OVERALL_EXIT=$?; }
phase_done "CONVERT" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ── STEP 1: A2 — Topic Clustering ──────────────────────────────────────────
phase_header "A2: Topic Clustering (K-Means, DBSCAN)"
require_inputs "A2" \
    "$MINING_OUT/A1/tfidf_matrix.npz" \
    "$MINING_OUT/A1/embeddings.npy" \
    "$MINING_OUT/A1/text_corpus.jsonl" \
&& run_step "A2" "Topic Clustering" "mine_topic_clustering.py" \
    "$MINING_OUT/A2/cluster_labels.jsonl" \
    "$MINING_OUT/A2/cluster_stats.json" \
|| { OVERALL_EXIT=$?; }
phase_done "A2" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ── STEP 2: A3 — Topic Modeling ──────────────────────────────────────────
phase_header "A3: Topic Modeling (LDA)"
require_inputs "A3" \
    "$MINING_OUT/A1/text_corpus.jsonl" \
&& run_step "A3" "Topic Modeling" "mine_topic_modeling.py" \
    "$MINING_OUT/A3/topic_model.json" \
    "$MINING_OUT/A3/doc_topics.jsonl" \
    "$MINING_OUT/A3/stats.json" \
|| { OVERALL_EXIT=$?; }
phase_done "A3" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ── STEP 3: A5 — Anomaly Detection ─────────────────────────────────────────
phase_header "A5: Anomaly Detection (Isolation Forest, SVM, Z-score)"
require_inputs "A5" \
    "$MINING_OUT/A1/embeddings.npy" \
    "$MINING_OUT/A1/text_corpus.jsonl" \
    "$MINING_OUT/A2/cluster_stats.json" \
&& run_step "A5" "Anomaly Detection" "mine_anomaly_detection.py" \
    "$MINING_OUT/A5/anomaly_scores.jsonl" \
    "$MINING_OUT/A5/anomaly_stats.json" \
|| { OVERALL_EXIT=$?; }
phase_done "A5" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ── STEP 4: A6 — Code Pattern Mining ──────────────────────────────────────
phase_header "A6: Code Pattern Mining"
require_inputs "A6" \
    "$MINING_OUT/A1/code_corpus.jsonl" \
    "$MINING_OUT/A3/doc_topics.jsonl" \
    "$MINING_OUT/A2/cluster_labels.jsonl" \
&& run_step "A6" "Code Pattern Mining" "mine_code_patterns.py" \
    "$MINING_OUT/A6/code_stats.json" \
    "$MINING_OUT/A6/code_patterns.json" \
|| { OVERALL_EXIT=$?; }
phase_done "A6" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ── STEP 5: A7 — Quality Modeling ─────────────────────────────────────────
phase_header "A7: Quality Modeling (composite quality scores)"
require_inputs "A7" \
    "$MINING_OUT/A1/text_corpus.jsonl" \
&& run_step "A7" "Quality Modeling" "mine_quality_modeling.py" \
    "$MINING_OUT/A7/quality_scores.jsonl" \
    "$MINING_OUT/A7/quality_model.json" \
|| { OVERALL_EXIT=$?; }
phase_done "A7" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ── STEP 6: A4 — Association Rules ────────────────────────────────────────
phase_header "A4: Association Rules (Apriori)"
require_inputs "A4" \
    "$MINING_OUT/A1/text_corpus.jsonl" \
    "$MINING_OUT/A2/cluster_labels.jsonl" \
    "$MINING_OUT/A3/doc_topics.jsonl" \
&& run_step "A4" "Association Rules" "mine_association_rules.py" \
    "$MINING_OUT/A4/association_rules.json" \
|| { OVERALL_EXIT=$?; }
phase_done "A4" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ── STEP 7: B1 — Interpret Clusters ───────────────────────────────────────
phase_header "B1: Interpret Clusters (human-readable labels)"
require_inputs "B1" \
    "$MINING_OUT/A2/cluster_labels.jsonl" \
    "$MINING_OUT/A2/cluster_stats.json" \
    "$MINING_OUT/A1/text_corpus.jsonl" \
    "$MINING_OUT/A3/topic_model.json" \
&& run_step "B1" "Interpret Clusters" "analyze_interpret_clusters.py" \
    "$MINING_OUT/B1/cluster_interpretation.json" \
|| { OVERALL_EXIT=$?; }
phase_done "B1" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ── STEP 8: B2 — Coverage Mapping ─────────────────────────────────────────
phase_header "B2: Coverage Mapping (topics → guideline taxonomy)"
require_inputs "B2" \
    "$MINING_OUT/B1/cluster_interpretation.json" \
    "$MINING_OUT/A3/topic_model.json" \
    "$MINING_OUT/A4/association_rules.json" \
&& run_step "B2" "Coverage Mapping" "analyze_coverage_mapping.py" \
    "$MINING_OUT/B2/coverage_map.json" \
    "$MINING_OUT/B2/coverage_gaps.json" \
|| { OVERALL_EXIT=$?; }
phase_done "B2" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ── STEP 9: B3 — Enrichment Scoring ───────────────────────────────────────
phase_header "B3: Enrichment Scoring (ranked recommendations)"
require_inputs "B3" \
    "$MINING_OUT/B2/coverage_map.json" \
    "$MINING_OUT/A7/quality_scores.jsonl" \
    "$MINING_OUT/A6/code_stats.json" \
&& run_step "B3" "Enrichment Scoring" "analyze_enrichment_scoring.py" \
    "$MINING_OUT/B3/enrichment_scores.json" \
|| { OVERALL_EXIT=$?; }
phase_done "B3" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ── STEP 10: B4 — Gap Quantification ──────────────────────────────────────
phase_header "B4: Gap Quantification (actionable gaps)"
require_inputs "B4" \
    "$MINING_OUT/B2/coverage_map.json" \
    "$MINING_OUT/B2/coverage_gaps.json" \
    "$MINING_OUT/B3/enrichment_scores.json" \
    "$MINING_OUT/A5/anomaly_stats.json" \
&& run_step "B4" "Gap Quantification" "analyze_gap_quantification.py" \
    "$MINING_OUT/B4/gap_quantification.json" \
|| { OVERALL_EXIT=$?; }
phase_done "B4" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ── STEP 11: B5 — Viability Assessment (GO/NO-GO) ─────────────────────────
phase_header "B5: Viability Assessment (GO/NO-GO verdict)"
require_inputs "B5" \
    "$MINING_OUT/A1/stats.json" \
    "$MINING_OUT/A2/cluster_stats.json" \
    "$MINING_OUT/A3/stats.json" \
    "$MINING_OUT/A4/association_rules.json" \
    "$MINING_OUT/A5/anomaly_stats.json" \
    "$MINING_OUT/A6/code_stats.json" \
    "$MINING_OUT/A7/quality_model.json" \
    "$MINING_OUT/B1/cluster_interpretation.json" \
    "$MINING_OUT/B2/coverage_map.json" \
    "$MINING_OUT/B3/enrichment_scores.json" \
    "$MINING_OUT/B4/gap_quantification.json" \
&& run_step "B5" "Viability Assessment" "analyze_viability.py" \
    "$MINING_OUT/B5/viability_report.json" \
|| { OVERALL_EXIT=$?; }
phase_done "B5" $OVERALL_EXIT

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  COLLECTIONS PIPELINE SUMMARY${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
echo ""

if [[ "$OVERALL_EXIT" -eq 0 ]]; then
    echo -e "${GREEN}✅  Collections pipeline completed successfully.${NC}"
    echo ""
    echo "  Output directory: $MINING_OUT"
    echo "  Log:              $LOGFILE"
    echo ""
    echo "  ── Generated Files ──"
    find "$MINING_OUT" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.npz' -o -name '*.npy' \) -exec ls -lh {} \; 2>/dev/null | \
        awk '{printf "  %-55s %s\n", $NF, $5}'
    echo ""
    echo -e "  ${GREEN}✓${NC} Next step: Check $MINING_OUT/B5/viability_report.json for GO/NO-GO."
else
    echo -e "${RED}❌  Collections pipeline aborted at exit code $OVERALL_EXIT.${NC}"
    echo ""
    echo "  Fix the issue, then resume:"
    echo "    ./scripts/full_collections_pipeline.sh --resume"
    echo ""
    echo "  Log file: $LOGFILE"
fi

echo ""
exit $OVERALL_EXIT
