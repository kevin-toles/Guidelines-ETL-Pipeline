"""C4 CodeBridgeResolver against ETDI BEH-02.01 steps 1/5 / SCE-02.01.04 / GHK-02.01.04.A (+ GETL-2.0 §5.1.9.15).
ETDI-1.0.74/.75. Qdrant = in-memory engine holding the real code_chunks points HTC V013 seeds for mcp-gateway
src/security/authn.py (16 units + module + component, with content and statement_signals). Only validate_token's
vector matches the query, so it is the hit. Real source: its exception_handler spans 178–198 (the design's 178–186
came from an older authn.py — REMEDIATION-LOG U19).
"""
from __future__ import annotations

from src.embedders.qdrant_test_support import RecordingAdapter, basis, real_payloads, seeded
from src.transformation.code_bridge_resolver import CodeBridgeResolver

GUIDELINE = {"title": "Validate JWTs", "summary": "Reject expired or badly signed tokens."}


def _resolver():
    q = seeded({"code_chunks": (768, real_payloads("v013_code_chunks_authn.json"))},
               vector=lambda p, size: basis(0 if p.get("qualified_name") == "validate_token" else 1, size))
    return CodeBridgeResolver(q, RecordingAdapter(basis(0))), q


def test_validate_token_hit_becomes_a_line_precise_code_reference() -> None:   # .74/.75 GHK-02.01.04.A
    resolver, _ = _resolver()
    out = resolver.resolve(dict(GUIDELINE))
    [block] = out["evidence_blocks"]
    assert block["type"] == "code_reference" and block["language"] == "python"
    assert block["content"].startswith("async def validate_token(")
    assert (block["lines"], block["signal_kind"], block["unit_lines"]) == ([178, 198], "exception_handler", [156, 208])
    assert out["_evidence_ids"]["code"] == [{"chunk_id": "mcp-gateway:src/security/authn.py:validate_token:156",
                                             "repo": "mcp-gateway", "file": "src/security/authn.py", "lines": [156, 208]}]


def test_query_is_the_nomic_vector_at_code_unit_granularity() -> None:
    resolver, _ = _resolver()
    hits = resolver._query_qdrant("Validate JWTs. Reject expired or badly signed tokens.")
    assert resolver._adapter.calls == [("Validate JWTs. Reject expired or badly signed tokens.", True)]
    assert {h["payload"]["granularity"] for h in hits} == {"code_unit"}


def test_a_unit_without_signals_cites_its_whole_span() -> None:
    resolver, _ = _resolver()
    payload = {"content": "def f(): pass", "language": "python", "start_line": 3, "end_line": 4, "statement_signals": []}
    assert resolver._code_block({"payload": payload, "score": 0.9})["lines"] == [3, 4]
