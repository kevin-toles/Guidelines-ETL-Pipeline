"""Phase C startup: the model-identity alignment gate before any Qdrant-facing resolver (ETDI BEH-02.01 step 3)."""
from __future__ import annotations

import os

from src.embedders.dimension_alignment_gate import AlignmentVerdict, DimensionAlignmentGate

#: C3 `chapters`, C4 `code_chunks` and its secondary source `code_textbook_bridge` (OQ#16).
PHASE_C_COLLECTIONS = ("chapters", "code_chunks", "code_textbook_bridge")


class PhaseCAborted(RuntimeError):
    """A Phase C collection failed the alignment gate; no resolver may run."""


class PhaseCOrchestrator:
    def __init__(self, embedder, embedder_model: str, qdrant_client) -> None:
        self._embedder = embedder
        self._embedder_model = embedder_model
        self._qdrant = qdrant_client

    def startup_gate(self) -> dict[str, AlignmentVerdict]:
        """Validate every Phase C collection (`${ALIGNMENT_SAMPLE_COUNT}`, `${ALIGNMENT_TAG_MODE}`); abort on any failure."""
        sample_count = int(os.environ.get("ALIGNMENT_SAMPLE_COUNT", "8"))
        tag_mode = os.environ.get("ALIGNMENT_TAG_MODE", "expand")
        verdicts = {c: DimensionAlignmentGate.validate(self._embedder.dim, self._embedder_model, self._qdrant, c,
                                                       sample_count=sample_count, tag_mode=tag_mode)
                    for c in PHASE_C_COLLECTIONS}
        failed = [v.message for v in verdicts.values() if not v.is_passed]
        if failed:
            raise PhaseCAborted("; ".join(failed))
        return verdicts
