"""RED tests for full_collections_pipeline shell helper consolidation. ETDI-1.0 WBS."""

from __future__ import annotations

from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent


# -- ETDI-1.0.88 --


class TestSharedHelpersExistence:
    """REFACTOR RED: Consolidate duplicated shell logic into _shared_helpers.sh. ETDI-1.0.88."""

    def test_shared_helpers_sh_exists(self) -> None:
        helpers_path = SCRIPTS_DIR / "_shared_helpers.sh"
        assert helpers_path.exists(), (
            f"_shared_helpers.sh must exist at {helpers_path} after consolidation"
        )

    def test_full_collections_pipeline_sources_shared_helpers(self) -> None:
        pipeline = SCRIPTS_DIR / "full_collections_pipeline.sh"
        content = pipeline.read_text()
        assert "source" in content and "_shared_helpers.sh" in content, (
            "full_collections_pipeline.sh must source _shared_helpers.sh"
        )


class TestDependencyCheckConsolidated:
    """ETDI-1.0.88: the fastembed pre-A1 check lives once in _shared_helpers.sh; both pipelines call it."""

    def test_helper_defines_the_check_and_both_pipelines_call_it(self) -> None:
        helpers = (SCRIPTS_DIR / "_shared_helpers.sh").read_text()
        assert "require_python_module()" in helpers
        for name in ("full_mining_pipeline.sh", "full_collections_pipeline.sh"):
            text = (SCRIPTS_DIR / name).read_text()
            assert "require_python_module fastembed" in text
            assert 'import fastembed' not in text, f"{name} still carries its own copy of the check"

    def test_helper_reports_a_missing_module_and_fails(self) -> None:
        import subprocess
        out = subprocess.run(["bash", "-c", f'source "{SCRIPTS_DIR / "_shared_helpers.sh"}"; PYTHON=python3; '
                              'require_python_module no_such_module_xyz'], capture_output=True, text=True)
        assert out.returncode != 0 and "✗ Missing: no_such_module_xyz" in out.stdout
