"""ProfileEmbeddingAdapter — GETL query embedding through the canonical COS endpoint with the active embedding profile
(ETDI-1.0 v0.18 §5.1.9.7, COMP-01.06, BEH-02.01, BEH-08.07).

Thin HTTP client: no in-process model, never a MiniLM fallback. The dimension and model are whatever COS reports for the
active profile; a vector of another length, or a dimension that changes mid-session, is refused. GETL C3/C4 resolve
evidence at search time, so every request carries `priority: live` (BEH-08.06 step 1).
"""
from __future__ import annotations

import time
from collections.abc import Mapping

import httpx
import numpy as np


class ProfileEmbeddingAdapter:
    """Vectors from COS `${COS_HOST}:${COS_PORT}${COS_EMBED_PATH}` (`/api/v1/embed/profile`)."""

    def __init__(self, endpoint: str, timeout_s: float = 30.0, max_retries: int = 3, *, backoff_s: float = 0.5) -> None:
        self.endpoint = endpoint
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._dim: int | None = None
        self._model: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> ProfileEmbeddingAdapter:
        """Endpoint from §7 `COS_HOST` (`localhost`), `COS_PORT` (`8083`), `COS_EMBED_PATH` (`/api/v1/embed/profile`)."""
        host = env.get("COS_HOST", "localhost")
        port = env.get("COS_PORT", "8083")
        path = env.get("COS_EMBED_PATH", "/api/v1/embed/profile")
        return cls(f"http://{host}:{port}{path}")

    def embed(self, text: str, is_query: bool = True) -> np.ndarray:
        """POST {text, task, priority: live}; the response's `embedding`."""
        data = self._post(self.endpoint, {"text": text, "task": _task(is_query), "priority": "live"})
        return self._vector(data["embedding"], data)

    def embed_batch(self, texts: list[str], is_query: bool = True) -> np.ndarray:
        """POST {texts, task, priority: live} to the /batch route; (N, dim)."""
        data = self._post(f"{self.endpoint}/batch", {"texts": texts, "task": _task(is_query), "priority": "live"})
        return np.stack([self._vector(v, data) for v in data["embeddings"]])

    @property
    def dim(self) -> int:
        """The `dim` the endpoint reports (probed once); `DimensionAlignmentGate` compares it to the collection."""
        if self._dim is None:
            self.embed("dimension probe")
        return self._dim  # type: ignore[return-value]

    @property
    def model(self) -> str:
        """The model id the endpoint reports for the active profile."""
        if self._model is None:
            self.embed("model probe")
        return self._model  # type: ignore[return-value]

    def _post(self, url: str, payload: dict) -> dict:
        last = ""
        for attempt in range(self._max_retries):
            try:
                with httpx.Client(timeout=self._timeout_s) as client:
                    response = client.post(url, json=payload)
                if response.status_code == 200:
                    return response.json()
                last = f"HTTP {response.status_code}: {response.text[:200]}"
            except httpx.HTTPError as e:
                last = str(e)
            if attempt + 1 < self._max_retries:
                time.sleep(self._backoff_s * 2**attempt)
        raise RuntimeError(f"COS embedding endpoint {url} failed after {self._max_retries} attempts: {last}")

    def _vector(self, values: list[float], data: dict) -> np.ndarray:
        vec = np.asarray(values, dtype=np.float32)
        dim = int(data["dim"])
        if vec.shape != (dim,):
            raise ValueError(f"COS returned a {vec.shape[0]}-dim vector; the profile reports {dim}")
        if self._dim is not None and self._dim != dim:
            raise ValueError(f"COS dimension changed mid-session ({self._dim} → {dim}); the profile was switched — reseed")
        self._dim, self._model = dim, data["model"]
        return vec


def _task(is_query: bool) -> str:
    return "query" if is_query else "document"
