"""Phase C startup gate against ETDI BEH-02.01 step 3 / GHK-02.01.01.A / SCE-02.01.02. ETDI-1.0.66/.67.

The gate runs for chapters, code_chunks and code_textbook_bridge before any resolver; the query side is the
ProfileEmbeddingAdapter's reported dim + model; collections hold the real V005/V013 payloads (in-memory Qdrant).
"""
from __future__ import annotations

import pytest

from src.embedders.qdrant_test_support import real_payloads, retagged, seeded
from src.transformation.phase_c_orchestrator import PhaseCAborted, PhaseCOrchestrator

V2 = "nomic-ai/nomic-embed-text-v2-moe"


class _Embedder:
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim


def _qdrant(chapters_dim: int = 768):
    return seeded({"chapters": (chapters_dim, real_payloads("v005_chapters_tiers.json")),
                   "code_chunks": (768, real_payloads("v013_code_chunks_authn.json")),
                   "code_textbook_bridge": (768, retagged(real_payloads("v005_chapters_tiers.json"), None))})


def test_startup_gate_passes_chapters_and_code_chunks_at_768() -> None:   # .66 GHK-02.01.01.A
    verdicts = PhaseCOrchestrator(_Embedder(), V2, _qdrant()).startup_gate()
    assert set(verdicts) == {"chapters", "code_chunks", "code_textbook_bridge"}
    assert all(verdicts[c].is_passed and verdicts[c].query_dim == 768 for c in ("chapters", "code_chunks"))
    assert verdicts["code_textbook_bridge"].is_model_matched is False   # pre-cycle points, expand mode


def test_any_failed_collection_aborts_phase_c_before_resolvers() -> None:   # .67 SCE-02.01.02
    with pytest.raises(PhaseCAborted, match="chapters"):
        PhaseCOrchestrator(_Embedder(384), "sentence-transformers/all-MiniLM-L6-v2", _qdrant()).startup_gate()


# ── ETDI-1.0.89 — Phase C end-to-end: real adapter (COS stand-in) → startup gate → C3/C4 on real V005/V013 points ──
def test_phase_c_gate_then_resolvers_through_the_real_adapter(monkeypatch) -> None:
    import json as _json

    import httpx

    from src.embedders.profile_embedding_adapter import ProfileEmbeddingAdapter
    from src.embedders.qdrant_test_support import basis
    from src.transformation.code_bridge_resolver import CodeBridgeResolver
    from src.transformation.textbook_bridge_resolver import TextbookBridgeResolver

    calls = []

    def cos(request: httpx.Request) -> httpx.Response:   # COS ProfileEmbedResponse fields (nomic-v2 profile); priority + generic task required
        body = _json.loads(request.content)
        calls.append(body)
        if request.url.path != "/api/v1/embed/profile" or body.get("priority") != "live" or body.get("task") not in ("query", "document"):
            return httpx.Response(422, json={"detail": "invalid"})
        return httpx.Response(200, json={"embedding": basis(0), "model": V2, "dim": 768, "max_tokens": 512, "task": body["task"]})

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(cos)}))
    q = seeded({"chapters": (768, real_payloads("v005_chapters_tiers.json")),
                "code_chunks": (768, real_payloads("v013_code_chunks_authn.json")),
                "code_textbook_bridge": (768, retagged(real_payloads("v005_chapters_tiers.json"), None))},
               vector=lambda p, size: basis(0 if p.get("qualified_name") in ("validate_token", None) else 1, size))
    adapter = ProfileEmbeddingAdapter("http://localhost:8083/api/v1/embed/profile")

    verdicts = PhaseCOrchestrator(adapter, V2, q).startup_gate()
    assert verdicts["chapters"].query_dim == 768 and verdicts["code_chunks"].is_passed

    guideline = {"title": "Validate JWTs", "summary": "Reject expired or badly signed tokens."}
    TextbookBridgeResolver(q, adapter).resolve(guideline)
    out = CodeBridgeResolver(q, adapter).resolve(guideline)

    assert all(c["task"] == "query" and c["priority"] == "live" for c in calls)
    code = [b for b in out["evidence_blocks"] if b["type"] == "code_reference"]
    assert code and code[0]["lines"] == [178, 198] and code[0]["signal_kind"] == "exception_handler"
