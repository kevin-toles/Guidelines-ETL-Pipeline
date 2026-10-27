"""RED test for requirements.txt — ETDI-1.0.347.

This test MUST fail: requirements.txt does not exist or is missing expected packages.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"


class TestRequirements:
    """Verify requirements.txt declares all needed dependencies."""

    # ETDI-1.0.347
    def test_requirements_txt_declares_dependencies(self) -> None:
        assert REQUIREMENTS_PATH.exists(), (
            f"requirements.txt not found at {REQUIREMENTS_PATH}"
        )

        content = REQUIREMENTS_PATH.read_text(encoding="utf-8").lower()
        expected_packages = [
            "scikit-learn",
            "sentence-transformers",
            "numpy",
            "scipy",
            "beautifulsoup4",
            "nltk",
            "gensim",
            "fastembed",
            "neo4j",
            "qdrant-client",
        ]

        missing = [pkg for pkg in expected_packages if pkg not in content]
        assert not missing, (
            f"requirements.txt is missing packages: {missing}"
        )
