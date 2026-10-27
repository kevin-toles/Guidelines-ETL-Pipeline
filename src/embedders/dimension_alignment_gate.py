"""DimensionAlignmentGate — query embedder dimension + model identity vs a Qdrant collection (ETDI §5.1.9.11).

Reads the collection's vector-config size and the `embedding_model` payload tag of `sample_count` scrolled points.
A differing dimension or tag fails in every mode; an untagged sample fails only under `ALIGNMENT_TAG_MODE=contract`
— under `expand` it passes unmatched with a warning and `alignment_untagged_total` (BEH-08.05 parallel change).
Every failure is logged as `QueryX != StorageY (collection)` and Phase C aborts on it. Never modifies Qdrant.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

UNTAGGED = "<untagged>"

#: alignment_untagged_total{collection} — in-process count (GETL has no metrics exporter; BEH-09.05).
ALIGNMENT_UNTAGGED_TOTAL: Counter[str] = Counter()


@dataclass(frozen=True)
class AlignmentVerdict:
    is_passed: bool
    is_model_matched: bool
    query_dim: int
    storage_dim: int
    query_model: str
    storage_model: str
    tag_mode: str
    collection_name: str
    message: str


def _verdict(collection: str, query_dim: int, storage_dim: int, query_model: str, storage_model: str,
             tag_mode: str, is_passed: bool, message: str) -> AlignmentVerdict:
    """The one constructor every branch uses; failures are logged here."""
    if not is_passed:
        logger.error("alignment gate failed: %s", message)
    return AlignmentVerdict(is_passed=is_passed, is_model_matched=storage_model == query_model, query_dim=query_dim,
                            storage_dim=storage_dim, query_model=query_model, storage_model=storage_model,
                            tag_mode=tag_mode, collection_name=collection, message=message)


def _sampled_model(qdrant_client, collection: str, sample_count: int) -> str:
    """The single tag of the sample; a differing pair returns the first non-query tag; any untagged → UNTAGGED."""
    points, _ = qdrant_client.scroll(collection, limit=sample_count, with_payload=["embedding_model"],
                                     with_vectors=False)
    tags = [(p.payload or {}).get("embedding_model") for p in points]
    if not tags or any(t is None for t in tags):
        return UNTAGGED
    distinct = sorted(set(tags))
    return distinct[0] if len(distinct) == 1 else ",".join(distinct)


class DimensionAlignmentGate:
    @staticmethod
    def validate(embedder_dim: int, embedder_model: str, qdrant_client, collection_name: str,
                 sample_count: int = 8, tag_mode: str = "expand") -> AlignmentVerdict:
        """Dimension AND model identity; see module docstring for the expand/contract rule."""
        if not qdrant_client.collection_exists(collection_name):
            return _verdict(collection_name, embedder_dim, 0, embedder_model, UNTAGGED, tag_mode, False,
                            f"collection not found ({collection_name})")
        storage_dim = qdrant_client.get_collection(collection_name).config.params.vectors.size
        if storage_dim != embedder_dim:
            return _verdict(collection_name, embedder_dim, storage_dim, embedder_model, UNTAGGED, tag_mode, False,
                            f"QueryDimension={embedder_dim} != StorageDimension={storage_dim} "
                            f"— {embedder_dim} != {storage_dim} ({collection_name})")
        storage_model = _sampled_model(qdrant_client, collection_name, sample_count)
        if storage_model == embedder_model:
            return _verdict(collection_name, embedder_dim, storage_dim, embedder_model, storage_model, tag_mode, True,
                            f"aligned ({collection_name})")
        if storage_model != UNTAGGED:
            return _verdict(collection_name, embedder_dim, storage_dim, embedder_model, storage_model, tag_mode, False,
                            f"StorageModel={storage_model} != QueryModel={embedder_model} ({collection_name})")
        if tag_mode == "contract":
            return _verdict(collection_name, embedder_dim, storage_dim, embedder_model, storage_model, tag_mode, False,
                            f"untagged embedding_model in contract mode ({collection_name})")
        ALIGNMENT_UNTAGGED_TOTAL[collection_name] += 1
        logger.warning("expand mode: untagged embedding_model points in %s — proceeding", collection_name)
        return _verdict(collection_name, embedder_dim, storage_dim, embedder_model, storage_model, tag_mode, True,
                        f"untagged, tolerated in expand mode ({collection_name})")
