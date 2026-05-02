#!/bin/bash
#
# full_mining_pipeline.sh — Deterministic Guideline Mining Pipeline
#
# Runs all 12 steps (A1–A7, B1–B5) in dependency order with checkpointing.
# Opens in a fresh Terminal window — never ties up the VS Code chat.
#
# PHASES:
#   0. Pre-flight checks — data sources, Python deps, scripts
#   1. A1  — Text Preprocessing (HTML clean, TF-IDF, SBERT embeddings)
#   2. A2  — Topic Clustering (K-Means, DBSCAN)
#   3. A3  — Topic Modeling (LDA)
#   4. A5  — Anomaly Detection (Isolation Forest, SVM, Z-score)
#   5. A6  — Code Pattern Mining (language distribution, patterns)
#   6. A7  — Quality Modeling (composite quality scores)
#   7. A4  — Association Rules (Apriori — needs A2+A3)
#   8. B1  — Interpret Clusters (human-readable labels — needs A2+A3+A6)
#   9. B2  — Coverage Mapping (SE→guideline taxonomy — needs B1+A3+A4)
#   10. B3 — Enrichment Scoring (ranked recommendations — needs B2+A7+A6)
#   11. B4 — Gap Quantification (actionable gaps — needs B2+B3+A5)
#   12. B5 — Viability Assessment (GO/NO-GO verdict — needs ALL)
#
# Usage:
#   ./scripts/full_mining_pipeline.sh                     # Full run with resume
#   ./scripts/full_mining_pipeline.sh --dry-run            # Validate chain only
#   ./scripts/full_mining_pipeline.sh --force              # Re-run everything
#   ./scripts/full_mining_pipeline.sh --resume             # Skip completed steps
#   ./scripts/full_mining_pipeline.sh --skip-a1            # Skip A1 (use existing)
#   ./scripts/full_mining_pipeline.sh --only-viability     # Only run B5
#
# Logs: /tmp/mining_pipeline_<YYYYMMDD_HHMMSS>/
#
# Requirements:
#   - Python 3.10+ with: scikit-learn, sentence-transformers, numpy,
#     scipy, beautifulsoup4, nltk, gensim, mlxtend, pygments
#   - 315 StackExchange JSONL files in data/stackexchange/{primary,hacks,supplemental}/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Project Paths ───────────────────────────────────────────────────────────
USB="/Volumes/USB321FD/Guidelines ETL Data"
SE_DATA="$USB/data/stackexchange"
MINING_OUT="$USB/output/mining_output"
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
RESUME=true   # default: skip completed steps
SKIP_A1=false
ONLY_VIABILITY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)          DRY_RUN=true; RESUME=false;      shift ;;
        --force)            FORCE=true; RESUME=false;        shift ;;
        --resume)           RESUME=true;                     shift ;;
        --skip-a1)          SKIP_A1=true;                    shift ;;
        --only-viability)   ONLY_VIABILITY=true;             shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Log Setup ───────────────────────────────────────────────────────────────
LOGDATE=$(date +%Y%m%d_%H%M%S)
LOGDIR="/tmp/mining_pipeline_${LOGDATE}"
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

# Check if a set of files all exist
files_exist() {
    for f in "$@"; do
        if [[ ! -f "$f" ]]; then
            return 1
        fi
    done
    return 0
}

# Check required input files for a step
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

# Run a mining step with checkpointing
run_step() {
    local step="$1"; shift           # e.g. "A1"
    local label="$1"; shift          # e.g. "Text Preprocessing"
    local script_name="$1"; shift     # e.g. "mine_text_preprocess.py"
    local -a output_files=("${@}")   # output files to check for checkpoint

    # ── Checkpoint: skip if all outputs exist and --resume ──
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
        echo "       python3 $script_path $SE_DATA $MINING_OUT"
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

    # ── Force: remove existing outputs so re-run is clean ──
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
    /usr/bin/time $PYTHON "$script_path" "$SE_DATA" "$MINING_OUT" 2>&1
    local exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        echo -e "  ${RED}✗ $label failed (exit $exit_code)${NC}"
        echo -e "  ${RED}  Check logs: $LOGFILE${NC}"
        return $exit_code
    fi

    # Verify outputs were created
    for f in "${output_files[@]}"; do
        if [[ ! -f "$f" ]]; then
            echo -e "  ${RED}✗ $label completed but output missing: ${f#$MINING_OUT/}${NC}"
            return 1
        fi
    done

    local total_size=0
    for f in "${output_files[@]}"; do
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
echo -e "${BLUE}║       GUIDELINE MINING — FULL PIPELINE                           ║${NC}"
echo -e "${BLUE}║       A1→A7 → B1→B5 → GO/NO-GO Verdict                          ║${NC}"
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

# Python dependencies (check key packages)
for pkg in sklearn numpy scipy bs4 nltk gensim mlxtend pygments; do
    if $PYTHON -c "import $pkg" 2>/dev/null; then
        :  # ok
    else
        # Try full package names
        case $pkg in
            sklearn) $PYTHON -c "import sklearn" 2>/dev/null || { echo -e "  ${RED}✗ Missing: scikit-learn${NC}"; ALL_CHECKS_PASSED=false; } ;;
            bs4)    $PYTHON -c "import bs4" 2>/dev/null || { echo -e "  ${RED}✗ Missing: beautifulsoup4${NC}"; ALL_CHECKS_PASSED=false; } ;;
            *)      echo -e "  ${RED}✗ Missing: $pkg${NC}"; ALL_CHECKS_PASSED=false ;;
        esac
    fi
done
echo -e "  ${GREEN}✓ Python dependencies${NC}"

# SBERT (optional but recommended for A1 + A5)
$PYTHON -c "import sentence_transformers" 2>/dev/null && \
    echo -e "  ${GREEN}✓ sentence-transformers available (SBERT embeddings enabled)${NC}" || \
    echo -e "  ${YELLOW}⚠ sentence-transformers not installed — A1 will skip embeddings, A5 will be limited${NC}"

# Scripts
MISSING_SCRIPTS=false
for s in mine_text_preprocess.py mine_topic_clustering.py mine_topic_modeling.py \
         mine_association_rules.py mine_anomaly_detection.py mine_code_patterns.py \
         mine_quality_modeling.py analyze_interpret_clusters.py analyze_coverage_mapping.py \
         analyze_enrichment_scoring.py analyze_gap_quantification.py analyze_viability.py; do
    if [[ ! -f "$PIPELINE_DIR/$s" ]]; then
        echo -e "  ${RED}✗ Missing script: $s${NC}"
        MISSING_SCRIPTS=true
        ALL_CHECKS_PASSED=false
    fi
done
[[ "$MISSING_SCRIPTS" == "false" ]] && echo -e "  ${GREEN}✓ All 12 pipeline scripts found${NC}"

# Source data: StackExchange JSONL files
echo -n "  StackExchange JSONL files: "
JSONL_COUNT=$(find "$SE_DATA/primary" "$SE_DATA/hacks" "$SE_DATA/supplemental" -name '*.jsonl' ! -name '._*' 2>/dev/null | wc -l)
if [[ "$JSONL_COUNT" -gt 0 ]]; then
    echo -e "${GREEN}$JSONL_COUNT files ✓${NC}"
else
    echo -e "${RED}0 files — data/stackexchange/{primary,hacks,supplemental} is empty${NC}"
    ALL_CHECKS_PASSED=false
fi

# Guideline taxonomy for B2 coverage mapping
TAXONOMY_DIR="$USB/guidelines/docs"
if [[ -d "$TAXONOMY_DIR" ]]; then
    echo -e "  ${GREEN}✓ Guideline taxonomy: $TAXONOMY_DIR${NC}"
else
    echo -e "  ${YELLOW}⚠ No guideline taxonomy dir — B2 will use fallback${NC}"
fi

# ── Summary ──
echo ""
if [[ "$ALL_CHECKS_PASSED" == "true" ]]; then
    echo -e "${GREEN}✅ Phase 0: All pre-flight checks passed.${NC}"
else
    echo -e "${RED}❌ Phase 0: Some checks failed — aborting.${NC}"
    exit 1
fi

# ── Exit if dry-run (pre-flight validates the chain) ──
if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo -e "${YELLOW}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  DRY RUN SUMMARY${NC}"
    echo -e "${YELLOW}══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Pre-flight: ✅ PASS"
    echo "  All source data and scripts verified."
    echo ""
    echo "  Pipeline dependency order:"
    echo "    Wave 0: A1 (no deps)"
    echo "    Wave 1: A2, A3, A5, A6, A7 (all need A1)"
    echo "    Wave 2: A4 (needs A2, A3)"
    echo "    Wave 3: B1 (needs A2, A3, A6)"
    echo "    Wave 4: B2 (needs B1, A3, A4)"
    echo "    Wave 5: B3 (needs B2, A7, A6)"
    echo "    Wave 6: B4 (needs B2, B3, A5)"
    echo "    Wave 7: B5 (needs ALL)"
    echo ""
    echo "  To execute: ./scripts/full_mining_pipeline.sh"
    echo "  To execute (resume): ./scripts/full_mining_pipeline.sh --resume"
    echo "  To execute (force re-run): ./scripts/full_mining_pipeline.sh --force"
    echo ""
    exit 0
fi

# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

OVERALL_EXIT=0

# ── Only Viability mode ──
if [[ "$ONLY_VIABILITY" == "true" ]]; then
    echo ""
    echo -e "${YELLOW}⚠ --only-viability: Skipping to B5 only${NC}"
    echo ""

    # Verify all required B5 inputs exist
    phase_header "B5: Viability Assessment"
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
    || { echo -e "${RED}✗ B5 failed or missing dependencies${NC}"; OVERALL_EXIT=1; }
    phase_done "B5" $OVERALL_EXIT

    echo ""
    echo -e "══════════════════════════════════════════════════════════════════"
    if [[ "$OVERALL_EXIT" -eq 0 ]]; then
        echo -e "${GREEN}✅ Pipeline complete.${NC}"
        echo "  Final verdict: $MINING_OUT/B5/viability_report.json"
    else
        echo -e "${RED}❌ Pipeline aborted.${NC}"
        echo "  Log: $LOGFILE"
    fi
    exit $OVERALL_EXIT
fi

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  WAVE 0: A1 — Text Preprocessing                                          ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if [[ "$SKIP_A1" == "false" ]]; then
    phase_header "A1: Text Preprocessing (HTML clean, TF-IDF, SBERT embeddings)"
    run_step "A1" "Text Preprocessing" "mine_text_preprocess.py" \
        "$MINING_OUT/A1/text_corpus.jsonl" \
        "$MINING_OUT/A1/code_corpus.jsonl" \
        "$MINING_OUT/A1/tfidf_matrix.npz" \
        "$MINING_OUT/A1/feature_names.npy" \
        "$MINING_OUT/A1/embeddings.npy" \
        "$MINING_OUT/A1/stats.json" \
    || { OVERALL_EXIT=$?; }
    phase_done "A1" $OVERALL_EXIT
    [[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT
else
    echo -e "  ${YELLOW}⚠ Skipping A1 (--skip-a1)${NC}"
fi

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  WAVE 1: A2, A3, A5, A6, A7 — Parallelizable (all depend on A1 only)     ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# These run sequentially in this script — each checks its own inputs.
# For true parallelism, launch manually after A1 completes.

# ── A2: Topic Clustering ────────────────────────────────────────────────────
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

# ── A3: Topic Modeling ──────────────────────────────────────────────────────
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

# ── A5: Anomaly Detection ───────────────────────────────────────────────────
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

# ── A6: Code Pattern Mining ─────────────────────────────────────────────────
phase_header "A6: Code Pattern Mining (language distribution, code patterns)"
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

# ── A7: Quality Modeling ────────────────────────────────────────────────────
phase_header "A7: Quality Modeling (composite quality scores)"
require_inputs "A7" \
    "$MINING_OUT/A1/text_corpus.jsonl" \
&& run_step "A7" "Quality Modeling" "mine_quality_modeling.py" \
    "$MINING_OUT/A7/quality_scores.jsonl" \
    "$MINING_OUT/A7/quality_model.json" \
|| { OVERALL_EXIT=$?; }
phase_done "A7" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  WAVE 2: A4 — Association Rules (needs A2, A3)                            ║
# ╚════════════════════════════════════════════════════════════════════════════╝

phase_header "A4: Association Rules (Apriori — tag/topic/site patterns)"
require_inputs "A4" \
    "$MINING_OUT/A1/text_corpus.jsonl" \
    "$MINING_OUT/A2/cluster_labels.jsonl" \
    "$MINING_OUT/A3/doc_topics.jsonl" \
&& run_step "A4" "Association Rules" "mine_association_rules.py" \
    "$MINING_OUT/A4/association_rules.json" \
|| { OVERALL_EXIT=$?; }
phase_done "A4" $OVERALL_EXIT
[[ $OVERALL_EXIT -ne 0 ]] && exit $OVERALL_EXIT

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  WAVE 3: B1 — Interpret Clusters (needs A2, A3, A6)                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

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

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  WAVE 4: B2 — Coverage Mapping (needs B1, A3, A4)                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

phase_header "B2: Coverage Mapping (SE topics → guideline taxonomy)"
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

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  WAVE 5: B3 — Enrichment Scoring (needs B2, A7, A6)                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

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

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  WAVE 6: B4 — Gap Quantification (needs B2, B3, A5)                      ║
# ╚════════════════════════════════════════════════════════════════════════════╝

phase_header "B4: Gap Quantification (actionable gap items)"
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

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  WAVE 7: B5 — Viability Assessment (needs ALL) — GO/NO-GO Verdict        ║
# ╚════════════════════════════════════════════════════════════════════════════╝

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
echo -e "${BLUE}  PIPELINE SUMMARY${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
echo ""

if [[ "$OVERALL_EXIT" -eq 0 ]]; then
    echo -e "${GREEN}✅  Pipeline completed successfully.${NC}"
    echo ""
    echo "  Output directory: $MINING_OUT"
    echo "  Log:              $LOGFILE"
    echo ""
    echo "  ── Generated Files ──"
    find "$MINING_OUT" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.npz' -o -name '*.npy' \) -exec ls -lh {} \; 2>/dev/null | \
        awk '{printf "  %-55s %s\n", $NF, $5}'
    echo ""
    echo -e "  ${GREEN}✓${NC} Next step: Check $MINING_OUT/B5/viability_report.json for the GO/NO-GO decision."
else
    echo -e "${RED}❌  Pipeline aborted at exit code $OVERALL_EXIT.${NC}"
    echo ""
    echo "  Check the log above for the failing step."
    echo "  Fix the issue, then resume:"
    echo "    ./scripts/full_mining_pipeline.sh --resume"
    echo ""
    echo "  Log file: $LOGFILE"
fi

echo ""
exit $OVERALL_EXIT
