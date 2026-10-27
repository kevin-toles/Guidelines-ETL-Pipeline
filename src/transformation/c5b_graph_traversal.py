"""C5b GETLGraphTraversalStep — C5 concept ids → cross-subdomain scenarios for C6 (ETDI §5.1.9.14, BEH-03.03).

Performs no traversal itself and never invents a scenario: it passes the ids to
`CrossSubdomainTraverser.discover_for_concepts` and reports what came back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphTraversalOutcome:
    concept_ids_received: int
    scenarios_produced: int
    scenarios: list = field(default_factory=list)  # CrossSubdomainCluster


class GETLGraphTraversalStep:
    def __init__(self, traverser: Any) -> None:
        self._traverser = traverser

    def traverse_for_scenarios(self, concept_ids: list[str]) -> GraphTraversalOutcome:
        scenarios = self._traverser.discover_for_concepts(list(concept_ids)) if concept_ids else []
        return GraphTraversalOutcome(len(concept_ids), len(scenarios), list(scenarios))
