"""CrossSubdomainTraverser — concept clusters whose chapter-attributed subdomains differ (ETDI §5.1.9.8, §5.1.10.2).

Subdomain attribution is always Chapter-mediated (BEH-03.01 step 1): a concept's subdomains are those of the chapters
that `COVERS` it and that V012 `CLASSIFIED_AS` a `TaxonomySubdomain`. Two modes: convergence (one concept covered from
two subdomains) and dependency (a V010 `PREREQUISITE_OF` edge whose endpoints' subdomains differ). Read-only; Neo4j
unavailability returns [] with a warning; missing COVERS/CLASSIFIED_AS/primary_kf raises TraversalPreconditionError.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from neo4j import GraphDatabase, Query, READ_ACCESS
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

logger = logging.getLogger(__name__)


def attribution(concept: str, chapter: str, subdomain: str) -> str:
    """The shared COVERS→CLASSIFIED_AS subdomain-attribution pattern."""
    return f"({concept})<-[:COVERS]-({chapter}:Chapter)-[:CLASSIFIED_AS]->({subdomain}:TaxonomySubdomain)"


CYPHER_CONVERGENCE = f"""
MATCH {attribution("c:Concept", "ch1", "s1")}
MATCH {attribution("c", "ch2", "s2")}
WHERE s1.slug < s2.slug AND ($concept_ids IS NULL OR c.concept_id IN $concept_ids)
OPTIONAL MATCH (b1:Book)-[:HAS_CHAPTER]->(ch1), (b2:Book)-[:HAS_CHAPTER]->(ch2), (b1)-[:PERPENDICULAR]-(b2)
WITH c, collect(DISTINCT s1.slug) + collect(DISTINCT s2.slug) AS subdomains,
     collect(DISTINCT [ch1.chapter_id, ch1.primary_kf]) + collect(DISTINCT [ch2.chapter_id, ch2.primary_kf]) AS bridging,
     count(b1) AS tier_adjacent
RETURN c.concept_id AS concept, subdomains, bridging, tier_adjacent
"""

CYPHER_DEPENDENCY = f"""
MATCH (c1:Concept)-[e:PREREQUISITE_OF]->(c2:Concept)
WHERE $concept_ids IS NULL OR c1.concept_id IN $concept_ids OR c2.concept_id IN $concept_ids
MATCH {attribution("c1", "ch1", "s1")}
MATCH {attribution("c2", "ch2", "s2")}
WHERE s1 <> s2
WITH c1, c2, collect(DISTINCT s1.slug) + collect(DISTINCT s2.slug) AS subdomains,
     collect(DISTINCT [ch1.chapter_id, ch1.primary_kf]) + collect(DISTINCT [ch2.chapter_id, ch2.primary_kf]) AS bridging,
     collect(DISTINCT elementId(e)) AS edge_ids
RETURN c1.concept_id AS source, c2.concept_id AS target, subdomains, bridging, edge_ids
"""

#: (structure, count query) — each must be ≥ 1 before discovery (BEH-03.04 step 3).
PRECONDITIONS: list[tuple[str, str]] = [
    ("COVERS", "MATCH (:Chapter)-[r:COVERS]->(:Concept) RETURN count(r) AS n"),
    ("CLASSIFIED_AS", "MATCH (:Chapter)-[r:CLASSIFIED_AS]->(:TaxonomySubdomain) RETURN count(r) AS n"),
    ("primary_kf", "MATCH (ch:Chapter) WHERE ch.primary_kf IS NOT NULL RETURN count(ch) AS n"),
]

_UNAVAILABLE = (ServiceUnavailable, SessionExpired, OSError)


class TraversalPreconditionError(RuntimeError):
    """A structure the traversal depends on is absent from the graph."""


@dataclass(frozen=True)
class CrossSubdomainCluster:
    concepts: list[str]
    subdomains: list[str]
    discovery_mode: str                       # convergence | dependency
    bridging_chapter_ids: list[str]
    prerequisite_edge_ids: list[str] = field(default_factory=list)
    support: int = 0
    tier_adjacency_weight: float = 0.0
    kf_distribution: dict[str, int] = field(default_factory=dict)


def _dedupe(values) -> list:
    return list(dict.fromkeys(v for v in values if v is not None))


def _chapters(bridging) -> tuple[list[str], dict[str, int]]:
    pairs = _dedupe(tuple(p) for p in bridging)
    chapter_ids = _dedupe(p[0] for p in pairs)
    return chapter_ids, dict(Counter(p[1] for p in pairs if p[1] is not None))


def convergence_clusters(rows: list[dict], min_subdomains: int) -> list[CrossSubdomainCluster]:
    """Rows of CYPHER_CONVERGENCE (concept, subdomains, bridging[[chapter_id, primary_kf]], tier_adjacent)."""
    out = []
    for r in rows:
        subdomains = sorted(_dedupe(r["subdomains"]))
        if len(subdomains) < min_subdomains:
            continue
        chapter_ids, kfs = _chapters(r["bridging"])
        out.append(CrossSubdomainCluster([r["concept"]], subdomains, "convergence", chapter_ids,
                                         support=len(chapter_ids), tier_adjacency_weight=float(r["tier_adjacent"]),
                                         kf_distribution=kfs))
    return out


def dependency_clusters(rows: list[dict], min_subdomains: int) -> list[CrossSubdomainCluster]:
    """Rows of CYPHER_DEPENDENCY; `concepts` keeps the edge direction [source, target]."""
    out = []
    for r in rows:
        subdomains = sorted(_dedupe(r["subdomains"]))
        if len(subdomains) < min_subdomains:
            continue
        chapter_ids, kfs = _chapters(r["bridging"])
        edges = _dedupe(r["edge_ids"])
        out.append(CrossSubdomainCluster([r["source"], r["target"]], subdomains, "dependency", chapter_ids,
                                         prerequisite_edge_ids=edges, support=len(edges), kf_distribution=kfs))
    return out


class CrossSubdomainTraverser:
    def __init__(self, neo4j_uri: str, neo4j_auth: tuple[str, str], query_timeout_s: float = 30.0) -> None:
        self._driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth, connection_timeout=query_timeout_s)
        self._timeout_s = query_timeout_s

    def _run(self, cypher: str, **params) -> list[dict]:
        with self._driver.session(default_access_mode=READ_ACCESS) as session:
            return [r.data() for r in session.run(Query(cypher, timeout=self._timeout_s), **params)]

    def assert_preconditions(self) -> None:
        """Raise TraversalPreconditionError naming each absent structure (COVERS, CLASSIFIED_AS, primary_kf)."""
        missing = [name for name, cypher in PRECONDITIONS if (self._run(cypher) or [{"n": 0}])[0]["n"] < 1]
        if missing:
            raise TraversalPreconditionError(f"graph lacks {', '.join(missing)} — run seed_neo4j.py / V012 first")

    def discover(self, min_subdomains: int = 2, max_clusters: int = 100,
                 mode: str = "both") -> list[CrossSubdomainCluster]:
        """mode ∈ {convergence, dependency, both}; [] on Neo4j failure; TraversalPreconditionError propagates."""
        return self._discover(None, min_subdomains, max_clusters, mode)

    def discover_for_concepts(self, concept_ids: list[str],
                              min_subdomains: int = 2) -> list[CrossSubdomainCluster]:
        """Both modes anchored to `concept_ids` (Concept.concept_id)."""
        return self._discover(list(concept_ids), min_subdomains, 100, "both")

    def _discover(self, concept_ids, min_subdomains, max_clusters, mode) -> list[CrossSubdomainCluster]:
        try:
            self.assert_preconditions()
            clusters: list[CrossSubdomainCluster] = []
            if mode in ("convergence", "both"):
                clusters += convergence_clusters(self._run(CYPHER_CONVERGENCE, concept_ids=concept_ids), min_subdomains)
            if mode in ("dependency", "both"):
                dependency = dependency_clusters(self._run(CYPHER_DEPENDENCY, concept_ids=concept_ids), min_subdomains)
                if not dependency:
                    logger.info("dependency mode: no PREREQUISITE_OF edge crosses a subdomain (prerequisite_edge_count=0)")
                clusters += dependency
        except _UNAVAILABLE as e:
            logger.warning("Neo4j unavailable; skipping cross-subdomain traversal: %s", e)
            return []
        except Neo4jError as e:
            logger.error("cross-subdomain traversal query failed: %s", e)
            return []
        clusters.sort(key=lambda c: c.support, reverse=True)
        return clusters[:max_clusters]
