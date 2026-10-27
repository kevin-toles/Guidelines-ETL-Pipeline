"""DimensionAlignmentGate against ETDI §5.1.9.11 / BEH-02.01 step 3 / BEH-08.05 / GHK-02.01.02.A, GHK-02.01.05.A.
ETDI-1.0.68/.69/.70/.71/.86. Collections are the qdrant-client in-memory engine holding the real V005 payloads.
"""
from __future__ import annotations

import logging

from src.embedders.dimension_alignment_gate import DimensionAlignmentGate
from src.embedders.qdrant_test_support import real_payloads, retagged, seeded

V2 = "nomic-ai/nomic-embed-text-v2-moe"
V15 = "nomic-ai/nomic-embed-text-v1.5"
CHAPTERS = real_payloads("v005_chapters_tiers.json")


def test_matching_dimension_and_model_pass() -> None:
    q = seeded({"chapters": (768, CHAPTERS)})
    v = DimensionAlignmentGate.validate(768, V2, q, "chapters")
    assert (v.is_passed, v.is_model_matched, v.query_dim, v.storage_dim, v.storage_model) == (True, True, 768, 768, V2)


def test_dimension_mismatch_fails_naming_the_collection(caplog) -> None:   # .68/.69 GHK-02.01.02.A
    q = seeded({"chapters": (768, CHAPTERS)})
    with caplog.at_level(logging.ERROR):
        v = DimensionAlignmentGate.validate(384, "sentence-transformers/all-MiniLM-L6-v2", q, "chapters")
    assert v.is_passed is False
    assert "384 != 768 (chapters)" in caplog.text


def test_same_dimension_other_model_fails(caplog) -> None:   # .70/.71 GHK-02.01.05.A
    q = seeded({"chapters": (768, retagged(CHAPTERS, V15))})
    with caplog.at_level(logging.ERROR):
        v = DimensionAlignmentGate.validate(768, V2, q, "chapters")
    assert (v.is_passed, v.is_model_matched, v.storage_model) == (False, False, V15)
    assert "v1.5 != " in caplog.text


def test_untagged_passes_unmatched_in_expand_and_fails_in_contract(caplog) -> None:   # BEH-08.05 parallel change
    q = seeded({"code_textbook_bridge": (768, retagged(CHAPTERS, None))})
    expand = DimensionAlignmentGate.validate(768, V2, q, "code_textbook_bridge", tag_mode="expand")
    contract = DimensionAlignmentGate.validate(768, V2, q, "code_textbook_bridge", tag_mode="contract")
    assert (expand.is_passed, expand.is_model_matched, expand.storage_model, expand.tag_mode) == (True, False, "<untagged>", "expand")
    assert (contract.is_passed, contract.tag_mode) == (False, "contract")


def test_missing_collection_fails_naming_it() -> None:   # SCE-02.01.03 (stale enriched_chapters)
    v = DimensionAlignmentGate.validate(768, V2, seeded({}), "enriched_chapters")
    assert v.is_passed is False and "enriched_chapters" in v.message


def test_only_sample_count_points_are_read() -> None:
    q = seeded({"chapters": (768, CHAPTERS)})
    assert DimensionAlignmentGate.validate(768, V2, q, "chapters", sample_count=1).is_passed
