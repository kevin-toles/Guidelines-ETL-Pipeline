"""Traverser/skew Cypher vs the schema the real seeders create — ETDI BEH-03.04 steps 1–2, SC-03.04.01.A/.02.A.
ETDI-1.0.106/.107/.108/.109.

The seeded schema is read from the seeder sources themselves (sibling repos under the POC root):
ai-platform-data scripts/seeding/seed_neo4j.py (Book/Chapter/Concept{concept_id}, COVERS, HAS_CHAPTER, PERPENDICULAR),
scripts/seeding/concept_highway/prerequisite_builder.py (V010 PREREQUISITE_OF), HTC V012__seed_neo4j_classification.py
(TaxonomySubdomain{slug}, CLASSIFIED_AS, Chapter.primary_kf) and HTC cre_graph_seeder.py (V014 Code*).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.traversal.cross_subdomain_traverser import CYPHER_CONVERGENCE, CYPHER_DEPENDENCY, PRECONDITIONS
from src.traversal.kf_skew_filter import CYPHER_KF_COUNTS
from src.traversal.neo4j_schema_alignment import SeededSchema

POC = Path(__file__).resolve().parents[3]
SEEDERS = [POC / "ai-platform-data/scripts/seeding/seed_neo4j.py",
           POC / "ai-platform-data/scripts/seeding/concept_highway/prerequisite_builder.py",
           POC / "htc-multidim-classification-service/src/seeding_v3/migrations/V012__seed_neo4j_classification.py",
           POC / "htc-multidim-classification-service/src/seeding_v3/lib/cre_graph_seeder.py"]


@pytest.fixture(scope="module")
def schema() -> SeededSchema:
    missing = [str(p) for p in SEEDERS if not p.exists()]
    if missing:
        pytest.skip(f"seeder sources not checked out: {missing}")
    return SeededSchema.from_sources(SEEDERS)


def test_the_real_seeders_create_the_documented_schema(schema) -> None:   # BEH-03.04 step 1
    assert {"Book", "Chapter", "Concept", "TaxonomySubdomain", "CodeUnit"} <= schema.labels
    assert {"COVERS", "HAS_CHAPTER", "PERPENDICULAR", "CLASSIFIED_AS", "PREREQUISITE_OF"} <= schema.relationships
    assert {"concept_id", "primary_kf", "slug", "chapter_id"} <= schema.properties
    assert "Subdomain" not in schema.labels and "KnowledgeFrame" not in schema.labels


@pytest.mark.parametrize("name,cypher", [("convergence", CYPHER_CONVERGENCE), ("dependency", CYPHER_DEPENDENCY),
                                         ("kf_counts", CYPHER_KF_COUNTS),
                                         *[(f"precondition:{n}", c) for n, c in PRECONDITIONS]])
def test_traversal_cypher_uses_only_seeded_labels_edges_and_properties(schema, name, cypher) -> None:   # .106/.107
    assert schema.violations(cypher) == []


def test_concept_subdomain_and_kf_counts_are_never_read(schema) -> None:   # SC-03.04.01.A
    for cypher in (CYPHER_CONVERGENCE, CYPHER_DEPENDENCY, CYPHER_KF_COUNTS):
        assert ".subdomain" not in cypher and "kf_counts" not in cypher


def test_a_label_the_seeders_never_create_is_reported_by_name(schema) -> None:   # .108/.109 SC-03.04.02.A
    violations = schema.violations("MATCH (c:Concept)-[:COVERS]->(s:Subdomain) RETURN c.subdomain")
    assert {(v.kind, v.name) for v in violations} == {("label", "Subdomain"), ("property", "subdomain")}
