#!/bin/bash
# Shared shell helpers for pipeline scripts.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

phase_header() {
    local label="$1"; shift
    echo ""
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $label${NC}"
    echo -e "${BLUE}  Started: $(date)${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════════════════${NC}"
}

# require_python_module <module> — pre-flight dependency check (ETDI BEH-02.03). Prints ✓/✗ Missing and returns 1
# when $PYTHON cannot import the module; callers fold the result into ALL_CHECKS_PASSED.
require_python_module() {
    local module="$1"
    if $PYTHON -c "import $module" 2>/dev/null; then
        echo -e "  ${GREEN}✓ $module available${NC}"
    else
        echo -e "  ${RED}✗ Missing: $module${NC}"
        return 1
    fi
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
    # RUN_STEP_INPUT is set by the sourcing script: $SE_DATA for mining, "" for collections.
    local input_arg="${RUN_STEP_INPUT-}"
    if [[ "$script_name" == "convert_collections_to_corpus.py" ]]; then
        input_arg="$COLLECTIONS_INPUT"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "  ${CYAN}⏳ $label — would run:${NC}"
        echo "       python3 $script_path $input_arg $MINING_OUT"
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

    if [[ "$FORCE" == "true" ]]; then
        for f in "${output_files[@]}"; do
            if [[ -f "$f" ]]; then
                rm -f "$f"
                echo -e "  ${YELLOW}⚠ Removed (--force): ${f#$MINING_OUT/}${NC}"
            fi
        done
    fi

    echo -e "  ${CYAN}▶ Running $label...${NC}"
    mkdir -p "$MINING_OUT"
    cd "$USB"

    /usr/bin/time $PYTHON "$script_path" "$input_arg" "$MINING_OUT" 2>&1
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
