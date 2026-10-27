"""Test support: real ETDI seeded point payloads loaded into the qdrant-client in-memory engine.

fixtures/v005_chapters_tiers.json — the points HTC V005 seed_classification_points upserts (chapter/passage/
paragraph_group, embedding_model nomic-ai/nomic-embed-text-v2-moe); fixtures/v013_code_chunks_authn.json — the
code_chunks points HTC V013 seeds for mcp-gateway authn.py. Vectors are filler: the alignment gate reads only the
collection vector config and the payload `embedding_model` tag.
"""
from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

FIX = Path(__file__).parent / "fixtures"


def real_payloads(name: str) -> list[tuple[str, dict]]:
    return [(p["id"], p["payload"]) for p in json.loads((FIX / name).read_text())["points"]]


def seeded(collections: dict[str, tuple[int, list[tuple[str, dict]]]], vector=None) -> QdrantClient:
    """{collection: (vector size, [(id, payload)])} → in-memory Qdrant; `vector(payload, size)` picks each vector."""
    q = QdrantClient(":memory:")
    for name, (size, points) in collections.items():
        q.create_collection(name, vectors_config=VectorParams(size=size, distance=Distance.COSINE))
        q.upsert(name, [PointStruct(id=pid, vector=(vector(payload, size) if vector else [0.1] * size), payload=payload)
                        for pid, payload in points])
    return q


def basis(i: int, size: int = 768) -> list[float]:
    v = [0.0] * size
    v[i] = 1.0
    return v


class RecordingAdapter:
    """Query side: answers like ProfileEmbeddingAdapter.embed (contract-tested in test_profile_embedding_adapter.py)."""

    def __init__(self, vector: list[float]) -> None:
        self.vector, self.calls = vector, []

    def embed(self, text: str, is_query: bool = True):
        import numpy as np
        self.calls.append((text, is_query))
        return np.asarray(self.vector, dtype=np.float32)


def retagged(points: list[tuple[str, dict]], model: str | None) -> list[tuple[str, dict]]:
    """Same points with `embedding_model` set to `model` (None → field removed: a pre-cycle untagged point)."""
    out = []
    for pid, payload in points:
        p = {k: v for k, v in payload.items() if k != "embedding_model"}
        if model is not None:
            p["embedding_model"] = model
        out.append((pid, p))
    return out
