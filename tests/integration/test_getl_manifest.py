"""RED tests for GETL manifest integrity — ETDI-1.0.76, ETDI-1.0.78.

These tests MUST fail: the GETL manifest file does not exist yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = REPO_ROOT / "getl_manifest.json"


class TestGETLManifest:
    """Integration tests for the GETL pipeline manifest."""

    # ETDI-1.0.76
    def test_every_manifest_entry_resolves(self) -> None:
        assert MANIFEST_PATH.exists(), (
            f"GETL manifest not found at {MANIFEST_PATH} — create it first"
        )
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        missing = []
        for entry in manifest.get("entries", []):
            entry_path = REPO_ROOT / entry["path"]
            if not entry_path.exists():
                missing.append(entry["path"])

        assert not missing, (
            f"{len(missing)} manifest entries do not resolve on disk:\n"
            + "\n".join(f"  - {p}" for p in missing)
        )

    # ETDI-1.0.78
    def test_manifest_excludes_retired_paths(self) -> None:
        assert MANIFEST_PATH.exists(), (
            f"GETL manifest not found at {MANIFEST_PATH} — create it first"
        )
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        retired_patterns = ["src/resolvers/", "a1_analyzer.py"]
        violations = []
        for entry in manifest.get("entries", []):
            for pattern in retired_patterns:
                if pattern in entry["path"]:
                    violations.append((entry["path"], pattern))

        assert not violations, (
            f"Manifest references retired paths:\n"
            + "\n".join(f"  - {path} (matches '{pat}')" for path, pat in violations)
        )
