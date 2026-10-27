"""GETL shell pipelines against ETDI BEH-02.03 / SC-02.03.01.A / SC-02.03.02.A / GHK-02.03.01–02.A. ETDI-1.0.80/.81/.82.

Step 1: both scripts gain `run_step "C5b" "Graph Traversal" "c5b_graph_traversal.py"` after C5 plus the fastembed check;
step 3: a C5b failure is non-fatal. GHK-02.03.02.A is exercised by running the real script in an environment without
fastembed: Phase 0 must report it and abort before A1.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent
PIPELINES = [SCRIPTS / "full_mining_pipeline.sh", SCRIPTS / "full_collections_pipeline.sh"]


@pytest.mark.parametrize("script", PIPELINES, ids=lambda p: p.name)
def test_exactly_one_c5b_step_after_c5_running_the_design_script(script: Path) -> None:   # .80 GHK-02.03.01.A
    text = script.read_text()
    c5b = [m.start() for m in re.finditer(r'run_step "C5b"', text)]
    assert len(c5b) == 1
    assert 0 <= text.find('run_step "C5" ') < c5b[0]
    assert re.search(r'run_step "C5b" "Graph Traversal" "c5b_graph_traversal\.py"', text)


@pytest.mark.parametrize("script", PIPELINES, ids=lambda p: p.name)
def test_c5b_failure_is_non_fatal(script: Path) -> None:   # BEH-02.03 step 3
    text = script.read_text()
    step = text[text.find('run_step "C5b"'):]
    handler = step[:step.find("phase_done")]
    assert "OVERALL_EXIT" not in handler, "a failed C5b must not fail the pipeline"
    assert "|| " in handler


def test_missing_fastembed_aborts_before_a1() -> None:   # .82 GHK-02.03.02.A
    pytest.importorskip  # fastembed is absent in this environment (checked by the assertion below)
    out = subprocess.run(["bash", str(PIPELINES[0])], capture_output=True, text=True, timeout=120)
    combined = out.stdout + out.stderr
    assert "✗ Missing: fastembed" in combined
    assert out.returncode != 0
    assert "A1: Text Preprocessing" not in combined
