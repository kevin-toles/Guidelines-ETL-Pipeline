"""CrossSubdomainTraverser against ETDI §5.1.9.8 / §5.1.10.2 / BEH-03.01 / BEH-03.04 step 3.
ETDI-1.0.90–.97/.110–.113/.116.

- Unavailability uses the real neo4j driver against a refused port (real ServiceUnavailable — .112 PERRY).
- Rows fed to the cluster builders are keyed exactly by the Cypher RETURN aliases (asserted below), as the driver's
  record.data() yields them. Graph-level GHK scenarios need a seeded Neo4j (question list) — not run here.
"""
from __future__ import annotations

import logging
import socket

import pytest
from neo4j.exceptions import ServiceUnavailable

from src.traversal.cross_subdomain_traverser import (
    CYPHER_CONVERGENCE, CYPHER_DEPENDENCY, PRECONDITIONS, CrossSubdomainTraverser, TraversalPreconditionError,
    attribution, convergence_clusters, dependency_clusters)


def _refused_uri() -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"bolt://127.0.0.1:{port}"


class _Graph(CrossSubdomainTraverser):
    """Answers each query with rows keyed as the driver would return them for that Cypher."""

    def __init__(self, answers: dict[str, list[dict]]) -> None:
        super().__init__(_refused_uri(), ("neo4j", "x"))
        self.answers, self.queries = answers, []

    def _run(self, cypher: str, **params) -> list[dict]:
        self.queries.append((cypher, params))
        return self.answers.get(cypher, [])


SEEDED = {c: [{"n": 1}] for _, c in PRECONDITIONS}


def test_return_aliases_match_the_row_keys_the_builders_read() -> None:
    assert "AS concept, subdomains, bridging, tier_adjacent" in " ".join(CYPHER_CONVERGENCE.split())
    assert "AS source, c2.concept_id AS target, subdomains, bridging, edge_ids" in " ".join(CYPHER_DEPENDENCY.split())


def test_convergence_cluster_spans_two_subdomains_via_bridging_chapters() -> None:   # .90/.91 GHK-03.01.01.A shape
    rows = [{"concept": "circuit-breaker", "subdomains": ["chaos-engineering", "software-architecture"],
             "bridging": [["ddia-ch8", "implementation-ref-spec"], ["chaos-ch2", "failure-mode-limitation"]],
             "tier_adjacent": 1}]
    g = _Graph({**SEEDED, CYPHER_CONVERGENCE: rows})
    [cluster] = g.discover(mode="convergence")
    assert (cluster.concepts, cluster.subdomains, cluster.discovery_mode) == (
        ["circuit-breaker"], ["chaos-engineering", "software-architecture"], "convergence")
    assert (cluster.bridging_chapter_ids, cluster.support, cluster.tier_adjacency_weight) == (["ddia-ch8", "chaos-ch2"], 2, 1.0)
    assert cluster.kf_distribution == {"implementation-ref-spec": 1, "failure-mode-limitation": 1}


def test_a_single_subdomain_graph_yields_no_cluster_without_error() -> None:   # .92/.93 GHK-03.01.02.A
    rows = [{"concept": "c", "subdomains": ["software-architecture"], "bridging": [["a", None]], "tier_adjacent": 0}]
    assert convergence_clusters(rows, 2) == []
    assert _Graph(SEEDED).discover() == []


def test_neo4j_refusing_connections_returns_empty_with_a_warning(caplog) -> None:   # .94/.95/.112 GHK-03.01.03.A
    t = CrossSubdomainTraverser(_refused_uri(), ("neo4j", "x"), query_timeout_s=2)
    with caplog.at_level(logging.WARNING):
        assert t.discover() == [] and t.discover_for_concepts(["circuit-breaker"]) == []
    assert "Neo4j unavailable" in caplog.text
    with pytest.raises(ServiceUnavailable):
        t.assert_preconditions()   # the real driver's exception type


def test_dependency_mode_keeps_edge_direction() -> None:   # .96/.97 GHK-03.01.04.A shape
    rows = [{"source": "consensus", "target": "distributed-lock", "subdomains": ["distributed-coordination", "consensus-algorithms"],
             "bridging": [["raft-ch3", "implementation-ref-spec"]], "edge_ids": ["5:abc:17"]}]
    g = _Graph({**SEEDED, CYPHER_DEPENDENCY: rows})
    [cluster] = g.discover(mode="dependency")
    assert (cluster.discovery_mode, cluster.subdomains) == ("dependency", ["consensus-algorithms", "distributed-coordination"])
    assert (cluster.concepts, cluster.prerequisite_edge_ids, cluster.support) == (
        ["consensus", "distributed-lock"], ["5:abc:17"], 1)


def test_missing_classified_as_raises_naming_it() -> None:   # .110/.111 GHK-03.04.03.A
    g = _Graph({**SEEDED, PRECONDITIONS[1][1]: [{"n": 0}]})
    with pytest.raises(TraversalPreconditionError, match="CLASSIFIED_AS"):
        g.discover()


def test_discover_for_concepts_anchors_both_modes_on_concept_id() -> None:
    g = _Graph(SEEDED)
    g.discover_for_concepts(["circuit-breaker", "chaos-engineering"])
    anchored = [p for c, p in g.queries if c in (CYPHER_CONVERGENCE, CYPHER_DEPENDENCY)]
    assert anchored == [{"concept_ids": ["circuit-breaker", "chaos-engineering"]}] * 2
    assert "c.concept_id IN $concept_ids" in CYPHER_CONVERGENCE


def test_both_modes_share_one_attribution_fragment() -> None:   # .113
    for c, ch, s in (("c:Concept", "ch1", "s1"), ("c1", "ch1", "s1"), ("c2", "ch2", "s2")):
        assert attribution(c, ch, s) in CYPHER_CONVERGENCE + CYPHER_DEPENDENCY


def test_preconditions_are_one_table() -> None:   # .116
    assert [name for name, _ in PRECONDITIONS] == ["COVERS", "CLASSIFIED_AS", "primary_kf"]
