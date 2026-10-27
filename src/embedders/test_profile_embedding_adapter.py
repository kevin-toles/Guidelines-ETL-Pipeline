"""ProfileEmbeddingAdapter against ETDI-1.0 v0.18 §5.1.9.7 (COMP-01.06), BEH-02.01 step 1, BEH-08.06 step 1, BEH-08.07.
ETDI-1.0.685/.686, reopened .64/.65/.84.

The COS stand-in answers only POST /api/v1/embed/profile[/batch] with COS's ProfileEmbedResponse /
ProfileBatchEmbedResponse fields ({embedding|embeddings, model, dim, max_tokens, task} — COS profile_embed_response.py,
profile_batch_embed_response.py, embed.py /embed/profile) and, like the real COS, requires `priority` (GETL C3/C4 are
search-time: `live`) plus a generic task (query | document). Anything else → 404/422; unavailable → 503 {detail}.
"""
from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from src.embedders.profile_embedding_adapter import ProfileEmbeddingAdapter

JINA = ("jinaai/jina-embeddings-v3", 1024, 8192)
NOMIC = ("nomic-ai/nomic-embed-text-v2-moe", 768, 512)
ENDPOINT = "http://localhost:8083/api/v1/embed/profile"


class _Cos:
    def __init__(self, profile=JINA, status: int = 200, vector_len: int | None = None) -> None:
        (self.model, self.dim, self.max_tokens), self.status = profile, status
        self.vector_len, self.requests = vector_len or self.dim, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path not in ("/api/v1/embed/profile", "/api/v1/embed/profile/batch"):
            return httpx.Response(404, json={"detail": "Not Found"})
        if body.get("priority") not in ("live", "batch") or body.get("task") not in ("query", "document"):
            return httpx.Response(422, json={"detail": "invalid request"})
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "embedding profile jina-v3 unavailable — einops"})
        vec = lambda t: [((len(t) + i) % 7 - 3) / 10 for i in range(self.vector_len)]  # noqa: E731
        meta = {"model": self.model, "dim": self.dim, "max_tokens": self.max_tokens, "task": body["task"]}
        if request.url.path.endswith("/batch"):
            return httpx.Response(200, json={"embeddings": [vec(t) for t in body["texts"]], **meta})
        return httpx.Response(200, json={"embedding": vec(body["text"]), **meta})


@pytest.fixture
def cos(monkeypatch):
    def install(stand_in: _Cos) -> _Cos:
        real = httpx.Client
        monkeypatch.setattr(httpx, "Client", lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(stand_in)}))
        return stand_in
    return install


def test_embed_posts_a_live_query_and_returns_the_profile_vector(cos) -> None:   # .685/.686, .64/.65
    c = cos(_Cos())
    vec = ProfileEmbeddingAdapter(ENDPOINT).embed("circuit breaker")
    assert c.requests == [("/api/v1/embed/profile", {"text": "circuit breaker", "task": "query", "priority": "live"})]
    assert vec.shape == (1024,) and np.any(vec != 0)


def test_document_side_uses_the_document_task(cos) -> None:
    c = cos(_Cos())
    ProfileEmbeddingAdapter(ENDPOINT).embed("chapter text", is_query=False)
    assert c.requests[0][1]["task"] == "document"


def test_embed_batch_uses_the_batch_route(cos) -> None:
    c = cos(_Cos(NOMIC))
    arr = ProfileEmbeddingAdapter(ENDPOINT).embed_batch(["a", "bb", "ccc"])
    assert c.requests == [("/api/v1/embed/profile/batch", {"texts": ["a", "bb", "ccc"], "task": "query", "priority": "live"})]
    assert arr.shape == (3, 768)


@pytest.mark.parametrize("profile", [JINA, NOMIC])
def test_dim_plus_model_are_what_the_endpoint_reports(cos, profile) -> None:
    cos(_Cos(profile))
    adapter = ProfileEmbeddingAdapter(ENDPOINT)
    assert (adapter.model, adapter.dim) == profile[:2]


def test_cos_failure_raises_after_max_retries_naming_the_endpoint(cos) -> None:
    cos(_Cos(status=503))
    with pytest.raises(RuntimeError, match="api/v1/embed/profile"):
        ProfileEmbeddingAdapter(ENDPOINT, max_retries=3, backoff_s=0).embed("x")


def test_a_vector_unlike_the_reported_dimension_is_refused(cos) -> None:   # Anti-Responsibility
    cos(_Cos(vector_len=384))
    with pytest.raises(ValueError, match="1024"):
        ProfileEmbeddingAdapter(ENDPOINT).embed("x")


def test_a_dimension_change_mid_session_is_refused(cos) -> None:
    adapter = ProfileEmbeddingAdapter(ENDPOINT)
    c = cos(_Cos(JINA)); adapter.embed("x")
    (c.model, c.dim, c.max_tokens), c.vector_len = NOMIC, NOMIC[1]      # COS restarted under another profile
    with pytest.raises(ValueError, match="changed"):
        adapter.embed("y")


def test_endpoint_comes_from_cos_env() -> None:
    assert ProfileEmbeddingAdapter.from_env({}).endpoint == ENDPOINT
    env = {"COS_HOST": "cos", "COS_PORT": "9000", "COS_EMBED_PATH": "/x/profile"}
    assert ProfileEmbeddingAdapter.from_env(env).endpoint == "http://cos:9000/x/profile"
