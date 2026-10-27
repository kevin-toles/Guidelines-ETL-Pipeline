"""C4 CodeBridgeResolver — code evidence from `code_chunks` (GETL-2.0 §5.1.9.15; ETDI BEH-02.01 steps 1/5).

Queries are embedded by ProfileEmbeddingAdapter and searched on `code_chunks` at `granularity == code_unit` (module and
component tier points carry no code). Each hit becomes a `code_reference` whose `lines` and `signal_kind` come from
the unit's first `statement_signals` entry — the failure-handling path SAS recorded — with the unit span kept as
`unit_lines`; a unit without signals cites its whole span.
"""
from __future__ import annotations

from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue


class CodeBridgeResolver:
    def __init__(self, qdrant_client: Any, adapter: Any, similarity_threshold: float = 0.55,
                 collection: str = "code_chunks") -> None:
        self._qdrant = qdrant_client
        self._adapter = adapter
        self._threshold = similarity_threshold
        self.collection = collection

    def resolve(self, guideline: dict) -> dict:
        """Embed title + summary, search code units, add a code_reference per hit."""
        for hit in self._query_qdrant(f"{guideline.get('title', '')}. {guideline.get('summary', '')}"):
            guideline.setdefault("evidence_blocks", []).append(self._code_block(hit))
            p = hit["payload"]
            guideline.setdefault("_evidence_ids", {}).setdefault("code", []).append(
                {"chunk_id": p.get("chunk_id"), "repo": p.get("repo_id"), "file": p.get("file_path"),
                 "lines": [p.get("start_line"), p.get("end_line")]})
        return guideline

    def _query_qdrant(self, query_text: str, limit: int = 5) -> list[dict]:
        vector = self._adapter.embed(query_text, is_query=True)
        response = self._qdrant.query_points(
            self.collection, query=[float(x) for x in vector], limit=limit, with_payload=True,
            query_filter=Filter(must=[FieldCondition(key="granularity", match=MatchValue(value="code_unit"))]),
            score_threshold=self._threshold)
        return [{"id": str(p.id), "score": p.score, "payload": p.payload or {}} for p in response.points]

    def _code_block(self, hit: dict) -> dict:
        code, language = self._resolve_code_text(hit)
        p = hit["payload"]
        unit_lines = [p.get("start_line"), p.get("end_line")]
        signals = p.get("statement_signals") or []
        first = signals[0] if signals else None
        return {"type": "code_reference", "source": f"{p.get('repo_id')}/{p.get('file_path')}", "content": code,
                "language": language, "relevance": hit.get("score"),
                "lines": [first["start_line"], first["end_line"]] if first else unit_lines,
                "signal_kind": first["kind"] if first else None, "unit_lines": unit_lines}

    def _resolve_code_text(self, qdrant_hit: dict) -> tuple[str, str]:
        """(code, language) from the payload."""
        p = qdrant_hit["payload"]
        return p.get("content", ""), p.get("language", "")
