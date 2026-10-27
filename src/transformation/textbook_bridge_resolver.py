"""C3 TextbookBridgeResolver — textbook evidence from `chapters` (GETL-2.0 §5.1.9.14; ETDI BEH-02.01 steps 1–2).

Queries are embedded by ProfileEmbeddingAdapter (COS, the active profile's dimension) and searched on V005's `chapters` collection with a
`granularity == chapter` filter (not `enriched_chapters`). Evidence carries resolved text only — a hit whose payload
has no chapter text yields no block (never a pointer-only block).
"""
from __future__ import annotations

from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

TEXT_FIELDS = ("content", "text")


class TextbookBridgeResolver:
    def __init__(self, qdrant_client: Any, adapter: Any, similarity_threshold: float = 0.55,
                 collection: str = "chapters") -> None:
        self._qdrant = qdrant_client
        self._adapter = adapter
        self._threshold = similarity_threshold
        self.collection = collection

    def resolve(self, guideline: dict) -> dict:
        """Embed title + summary, search chapters, add a textbook_excerpt per hit that carries chapter text."""
        hits = self._query_qdrant(f"{guideline.get('title', '')}. {guideline.get('summary', '')}")
        for hit in hits:
            text = self._resolve_chapter_text(hit)
            if not text:
                continue
            guideline.setdefault("evidence_blocks", []).append(
                {"type": "textbook_excerpt", "source": hit["payload"].get("text_id"), "content": text,
                 "relevance": hit["score"]})
            guideline.setdefault("_evidence_ids", {}).setdefault("textbook", []).append(
                {"chapter_id": hit["payload"].get("chapter_id"), "book": hit["payload"].get("text_id")})
        return guideline

    def _query_qdrant(self, query_text: str, limit: int = 5) -> list[dict]:
        """Nomic query vector → `chapters` at granularity `chapter`, hits above the similarity threshold."""
        vector = self._adapter.embed(query_text, is_query=True)
        response = self._qdrant.query_points(
            self.collection, query=[float(x) for x in vector], limit=limit, with_payload=True,
            query_filter=Filter(must=[FieldCondition(key="granularity", match=MatchValue(value="chapter"))]),
            score_threshold=self._threshold)
        return [{"id": str(p.id), "score": p.score, "payload": p.payload or {}} for p in response.points]

    def _resolve_chapter_text(self, qdrant_hit: dict) -> str:
        """Full chapter text from the payload, or '' when the point carries none."""
        payload = qdrant_hit["payload"]
        return next((payload[f] for f in TEXT_FIELDS if payload.get(f)), "")
