"""Neo4j schema alignment — traversal Cypher vs the schema the seeders actually create (ETDI BEH-03.04).

`SeededSchema.from_sources` reads the seeder source files (seed_neo4j.py, V010 prerequisite_builder.py, V012, V014)
and collects every node label, relationship type and property key their Cypher creates or matches;
`violations(cypher)` names each label, relationship or property a traversal query uses that no seeder provides.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_LABEL = re.compile(r"\(\s*\w*\s*:\s*(\w+)")
_REL = re.compile(r"\[\s*\w*\s*:\s*([A-Z_][A-Z0-9_]*)")
_PROP_MAP = re.compile(r"\{\s*(\w+)\s*:")
_PROP_DOT = re.compile(r"\b[a-z]\w*\.([a-z_]\w*)\b")
_SET_PROP = re.compile(r"\bSET\s+\w+\.(\w+)|,\s*\w+\.(\w+)\s*=")

#: Cypher functions / map keys that look like property access but are not graph properties.
_NOT_PROPERTIES = {"row", "item"}


@dataclass(frozen=True)
class SchemaViolation:
    kind: str     # label | relationship | property
    name: str
    message: str


@dataclass
class SeededSchema:
    labels: set[str] = field(default_factory=set)
    relationships: set[str] = field(default_factory=set)
    properties: set[str] = field(default_factory=set)

    @classmethod
    def from_sources(cls, paths: Iterable[Path]) -> SeededSchema:
        schema = cls()
        for path in paths:
            text = Path(path).read_text(encoding="utf-8")
            schema.labels |= set(_LABEL.findall(text))
            schema.relationships |= set(_REL.findall(text))
            schema.properties |= set(_PROP_MAP.findall(text)) | set(_PROP_DOT.findall(text))
            schema.properties |= {a or b for a, b in _SET_PROP.findall(text)}
        return schema

    def violations(self, cypher: str) -> list[SchemaViolation]:
        found: list[SchemaViolation] = []
        for label in sorted(set(_LABEL.findall(cypher)) - self.labels):
            found.append(SchemaViolation("label", label, f"label {label} is created by no seeder"))
        for rel in sorted(set(_REL.findall(cypher)) - self.relationships):
            found.append(SchemaViolation("relationship", rel, f"relationship {rel} is created by no seeder"))
        used = (set(_PROP_DOT.findall(cypher)) | set(_PROP_MAP.findall(cypher))) - _NOT_PROPERTIES
        for prop in sorted(used - self.properties):
            found.append(SchemaViolation("property", prop, f"property {prop} is set by no seeder"))
        return found
