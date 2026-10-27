"""KFSkewFilter — concepts with strong implementation but weak failure-mode coverage (ETDI §5.1.9.10, BEH-03.02).

Per-concept KF counts are derived, not stored: chapters that `COVERS` the concept, grouped by `primary_kf` (V012).
A concept is flagged when implementation-ref-spec : failure-mode-limitation exceeds `skew_threshold`
(`${KF_SKEW_THRESHOLD}`, 3.0); a concept with implementation coverage and no failure-mode coverage has ratio ∞.
Read-only; [] on Neo4j unavailability.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass

from neo4j import GraphDatabase, Query, READ_ACCESS
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired

logger = logging.getLogger(__name__)

IMPLEMENTATION = "implementation-ref-spec"
FAILURE_MODE = "failure-mode-limitation"

CYPHER_KF_COUNTS = """
MATCH (c:Concept)<-[:COVERS]-(ch:Chapter)
WHERE ch.primary_kf IS NOT NULL
OPTIONAL MATCH (ch)-[:CLASSIFIED_AS]->(s:TaxonomySubdomain)
WITH c, ch.primary_kf AS kf, count(DISTINCT ch) AS n, collect(DISTINCT s.slug) AS subdomains
RETURN c.concept_id AS concept_id, c.name AS concept_name, kf, n, subdomains
"""


@dataclass(frozen=True)
class SkewedConcept:
    concept_id: str
    concept_name: str
    subdomain: str
    implementation_count: int
    failure_mode_count: int
    ratio: float


def skewed_concepts(rows: list[dict], threshold: float) -> list[SkewedConcept]:
    """Rows of CYPHER_KF_COUNTS (one per concept × kf) → flagged concepts, highest ratio first."""
    counts: dict[str, Counter] = defaultdict(Counter)
    names: dict[str, str] = {}
    subdomains: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        counts[r["concept_id"]][r["kf"]] += r["n"]
        names[r["concept_id"]] = r.get("concept_name") or r["concept_id"]
        subdomains[r["concept_id"]].update(s for s in r.get("subdomains") or [] if s)
    flagged = []
    for concept, kf in counts.items():
        impl, fm = kf[IMPLEMENTATION], kf[FAILURE_MODE]
        ratio = impl / fm if fm else (float("inf") if impl else 0.0)
        if impl and ratio > threshold:
            top = subdomains[concept].most_common(1)
            flagged.append(SkewedConcept(concept, names[concept], top[0][0] if top else "", impl, fm, ratio))
    return sorted(flagged, key=lambda s: s.ratio, reverse=True)


class KFSkewFilter:
    def __init__(self, neo4j_uri: str, neo4j_auth: tuple[str, str], skew_threshold: float = 3.0) -> None:
        self._driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self._threshold = skew_threshold

    def filter(self) -> list[SkewedConcept]:
        try:
            with self._driver.session(default_access_mode=READ_ACCESS) as session:
                rows = [r.data() for r in session.run(Query(CYPHER_KF_COUNTS))]
        except (ServiceUnavailable, SessionExpired, OSError) as e:
            logger.warning("Neo4j unavailable; KF skew filter skipped: %s", e)
            return []
        except Neo4jError as e:
            logger.error("KF skew query failed: %s", e)
            return []
        flagged = skewed_concepts(rows, self._threshold)
        if not flagged:
            logger.info("no concept exceeds implementation:failure-mode %.1f:1 — KF distribution over %d rows",
                        self._threshold, len(rows))
        return flagged
