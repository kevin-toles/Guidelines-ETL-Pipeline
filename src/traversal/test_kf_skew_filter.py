"""KFSkewFilter against ETDI §5.1.9.10 / BEH-03.02 / GHK-03.02.01–02.A. ETDI-1.0.98–.101/.114.
Rows are keyed by CYPHER_KF_COUNTS' RETURN aliases (asserted); unavailability uses the real driver."""
from __future__ import annotations

import logging
import socket

from src.traversal.kf_skew_filter import CYPHER_KF_COUNTS, KFSkewFilter, skewed_concepts


def _refused_uri() -> str:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return f"bolt://127.0.0.1:{port}"


def test_return_aliases_match_the_row_keys() -> None:   # .114 shape kept in sync with the query
    assert "AS concept_id, c.name AS concept_name, kf, n, subdomains" in " ".join(CYPHER_KF_COUNTS.split())


def test_45_implementation_vs_3_failure_mode_chapters_is_flagged() -> None:   # .98/.99 GHK-03.02.01.A
    rows = [{"concept_id": "circuit-breaker", "concept_name": "circuit breaker", "kf": "implementation-ref-spec", "n": 45,
             "subdomains": ["software-architecture"]},
            {"concept_id": "circuit-breaker", "concept_name": "circuit breaker", "kf": "failure-mode-limitation", "n": 3,
             "subdomains": ["software-architecture"]},
            {"concept_id": "bulkhead", "concept_name": "bulkhead", "kf": "implementation-ref-spec", "n": 4, "subdomains": []},
            {"concept_id": "bulkhead", "concept_name": "bulkhead", "kf": "failure-mode-limitation", "n": 2, "subdomains": []}]
    [flag] = skewed_concepts(rows, 3.0)
    assert (flag.concept_id, flag.implementation_count, flag.failure_mode_count, flag.ratio, flag.subdomain) == (
        "circuit-breaker", 45, 3, 15.0, "software-architecture")


def test_ratio_compares_implementation_to_failure_mode_only() -> None:   # not top/second of any KFs
    rows = [{"concept_id": "x", "kf": "role-definition", "n": 50, "subdomains": []},
            {"concept_id": "x", "kf": "implementation-ref-spec", "n": 5, "subdomains": []},
            {"concept_id": "x", "kf": "failure-mode-limitation", "n": 5, "subdomains": []}]
    assert skewed_concepts(rows, 3.0) == []


def test_neo4j_refusing_connections_returns_empty(caplog) -> None:   # .100/.101 GHK-03.02.02.A
    with caplog.at_level(logging.WARNING):
        assert KFSkewFilter(_refused_uri(), ("neo4j", "x")).filter() == []
    assert "Neo4j unavailable" in caplog.text
