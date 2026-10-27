"""C3 TextbookBridgeResolver against ETDI BEH-02.01 steps 1–2 / SC-02.01.03.A / GHK-02.01.03.A (+ GETL-2.0 §5.1.9.14
contract). ETDI-1.0.72/.73. Qdrant = in-memory engine holding the real V005 points (chapter + passage +
paragraph_group of one book); the query vector comes from the Nomic adapter.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.embedders.qdrant_test_support import RecordingAdapter, real_payloads, seeded
from src.transformation.textbook_bridge_resolver import TextbookBridgeResolver

GUIDELINE = {"title": "Explaining classifier predictions", "summary": "Use local surrogate models to explain a prediction."}


def _resolver():
    q = seeded({"chapters": (768, real_payloads("v005_chapters_tiers.json"))})
    return TextbookBridgeResolver(q, RecordingAdapter([0.1] * 768)), q


def test_query_targets_chapters_at_chapter_granularity_with_the_nomic_query_vector() -> None:   # .72/.73
    resolver, _ = _resolver()
    hits = resolver._query_qdrant("Explaining classifier predictions. Use local surrogate models.")
    assert resolver._adapter.calls == [("Explaining classifier predictions. Use local surrogate models.", True)]
    assert hits and {h["payload"]["granularity"] for h in hits} == {"chapter"}, "passage/paragraph_group excluded"


def test_collection_is_chapters_never_enriched_chapters() -> None:
    resolver, _ = _resolver()
    assert resolver.collection == "chapters"


def test_resolve_embeds_title_and_summary_and_stores_no_pointer_only_evidence() -> None:
    resolver, _ = _resolver()
    out = resolver.resolve(dict(GUIDELINE))
    assert resolver._adapter.calls[0] == (f"{GUIDELINE['title']}. {GUIDELINE['summary']}", True)
    # V005 chapter payloads carry no chapter text (findings list): no textbook_excerpt may be invented from an id.
    assert all(b.get("content") for b in out.get("evidence_blocks", []))


def test_the_excerpt_is_the_v005_chapter_content() -> None:
    """ETDI BEH-08.02 step 2 (amended): V005 chapter points carry their embedded text as `content`, so C3 resolves a
    real textbook_excerpt (GETL-2.0 C3: full chapter text from the payload). Fixture regenerated 2026-10-19 by the
    fixed HTC seeder."""
    resolver, _ = _resolver()
    out = resolver.resolve(dict(GUIDELINE))
    [block] = out["evidence_blocks"]
    fixture = json.loads((Path(__file__).resolve().parents[1] / "embedders" / "fixtures" / "v005_chapters_tiers.json").read_text())
    chapter = next(p["payload"] for p in fixture["points"] if p["payload"]["granularity"] == "chapter")
    assert block["type"] == "textbook_excerpt" and block["content"] == chapter["content"]
    assert out["_evidence_ids"]["textbook"] == [{"chapter_id": chapter["chapter_id"], "book": chapter["text_id"]}]
