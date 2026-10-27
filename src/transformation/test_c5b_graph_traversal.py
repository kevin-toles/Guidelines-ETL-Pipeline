"""GETLGraphTraversalStep against ETDI §5.1.9.14 / BEH-03.03 / GHK-03.03.01.A. ETDI-1.0.102/.103/.115."""
from __future__ import annotations

import inspect

from src.transformation.c5b_graph_traversal import GETLGraphTraversalStep, GraphTraversalOutcome
from src.traversal.cross_subdomain_traverser import CrossSubdomainCluster, CrossSubdomainTraverser


class _Traverser:
    """Same discover_for_concepts signature/return type as COMP-01.07 (checked below)."""

    def __init__(self) -> None:
        self.calls = []

    def discover_for_concepts(self, concept_ids: list[str], min_subdomains: int = 2) -> list[CrossSubdomainCluster]:
        self.calls.append(concept_ids)
        return [CrossSubdomainCluster(["circuit-breaker"], ["chaos-engineering", "software-architecture"],
                                      "convergence", ["ddia-ch8", "chaos-ch2"], support=2)]


def test_stand_in_matches_the_real_traverser_interface() -> None:   # .115 PERRY
    assert inspect.signature(_Traverser.discover_for_concepts) == inspect.signature(
        CrossSubdomainTraverser.discover_for_concepts)


def test_c5_concept_ids_reach_the_traverser_and_scenarios_reach_c6() -> None:   # .102/.103 GHK-03.03.01.A
    t = _Traverser()
    out = GETLGraphTraversalStep(t).traverse_for_scenarios(["circuit-breaker", "chaos-engineering"])
    assert t.calls == [["circuit-breaker", "chaos-engineering"]]
    assert (out.concept_ids_received, out.scenarios_produced) == (2, 1)
    assert isinstance(out, GraphTraversalOutcome) and out.scenarios[0].discovery_mode == "convergence"


def test_no_ids_no_scenario_and_nothing_invented() -> None:
    out = GETLGraphTraversalStep(_Traverser()).traverse_for_scenarios([])
    assert (out.concept_ids_received, out.scenarios_produced, out.scenarios) == (0, 0, [])
