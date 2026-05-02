#!/usr/bin/env python3
"""
Cross-Source Enrichment Mapper & Strategy Generator — v1.0
Synthesizes ALL analysis results into a comprehensive enrichment strategy.

Inputs (from previous analyzers):
  - completeness_by_source.json      — what % of each field is populated per source
  - source_field_matrix.json         — which fields exist in which sources
  - csv_to_schema_mapping.json       — which CSV columns map to guideline fields
  - cre_enrichment_inventory.json    — CRE enrichment quality assessment
  - enrichment_mapping.json          — StackExchange enrichment potential

Outputs:
  - cross_source_enrichment_map.json — complete field-by-field enrichment plan
  - enrichment_priority_matrix.md    — prioritized action plan
  - data_mining_strategy.md          — comprehensive ETL strategy following best practices

Usage:
  python3 build_cross_source_mapping.py [--output-dir PATH]
"""

import json, os, sys
from collections import defaultdict, Counter
from datetime import datetime

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines/scripts/analysis_tools/output"
)

# ── Guideline v2.0 Schema Reference ───────────────────────────────────────
SCHEMA_V2 = {
    "core_identity": {
        "guideline_id": "Unique identifier (e.g., guideline_02404, cre_folly_001_001)",
        "schema_version": "Schema version (2.0)",
        "title": "Problem/repo name",
        "title_slug": "URL-friendly slug",
        "link": "URL to source problem/repo",
    },
    "situation": {
        "situation.summary": "Concise problem description (AI-synthesized)",
        "situation.difficulty": "Easy/Medium/Hard or N/A",
        "situation.topics": "Array of relevant topic tags",
        "situation.tags.*": "Boolean flags: has_sorted_input, has_duplicates, etc.",
    },
    "approach": {
        "guideline": "The guideline/approach text (AI-generated core)",
        "reasoning": "Why this approach works (AI-generated)",
        "complexity.time": "Time complexity (e.g., O(n))",
        "complexity.space": "Space complexity",
        "complexity.code_hints": "Code implementation hints array",
    },
    "pattern": {
        "pattern.pattern_id": "Pattern identifier slug",
        "pattern.pattern_name": "Human-readable pattern name",
        "pattern.category": "High-level pattern category",
    },
    "code": {
        "solution_code": "Python solution code (CRITICAL GAP)",
        "has_solution_code": "Boolean flag: has solution code",
        "has_hints": "Boolean flag: has hints",
        "hints": "Array of progressive hints",
        "code_analysis.*": "Detection flags (uses_dict, uses_set, loops, etc.)",
    },
    "metadata": {
        "source_problem_id": "Numeric source problem ID",
        "source_dataset": "Source dataset name",
        "acceptance_rate": "Acceptance rate % (LeetCode only)",
        "likes": "Number of likes",
        "dislikes": "Number of dislikes",
        "stats": "Submission statistics",
        "description": "Full HTML problem description",
        "similar_questions": "Related problem references",
        "metadata.*": "Extended metadata (premium, category, companies)",
    },
    "bridges": {
        "bridges.*": "Cross-references: code_repos, code_chunks, textbook_chapters, diagrams",
    },
    "derived": {
        "constraints": "Problem constraints list (AI-derived)",
        "alternatives": "Alternative approaches (AI-derived or from SE)",
    },
}

# ── Enrichment Source Assessment (from analyzer outputs) ───────────────────
ENRICHMENT_SOURCES = {
    "leetcode_csvs": {
        "description": "Raw LeetCode CSV data (10 CSV files)",
        "quality": "high",
        "coverage": "8 CSVs have solution_code, acceptance_rate, likes, etc.",
        "best_for": [
            "solution_code", "acceptance_rate", "likes", "dislikes",
            "description", "situation.difficulty", "situation.topics",
            "link", "title", "hints", "similar_questions", "stats",
        ],
        "limitation": "Only covers LeetCode problems. No algorithmic reasoning.",
    },
    "cre_enrichment": {
        "description": "CRE repo enrichment metadata (500 files, 98 domains)",
        "quality": "mixed — 40% rich/adequate, 60% thin/stub",
        "coverage": "Covers patterns from 505 repos across 92 domains",
        "best_for": [
            "pattern.pattern_id", "pattern.pattern_name", "pattern.category",
            "situation.summary", "guideline", "reasoning",
        ],
        "limitation": "60% are stubs needing re-enrichment. No solution code.",
    },
    "stackexchange": {
        "description": "StackExchange Q&A data (624 JSONL files, ~1.2M records)",
        "quality": "mixed — varies by site and score",
        "coverage": "200+ StackExchange sites, 16 common fields",
        "best_for": [
            "hints", "alternatives", "reasoning", "constraints",
            "similar_questions", "situation.summary",
        ],
        "limitation": "Not algorithmically tagged. Requires NLP to extract patterns.",
    },
    "leetcode_guidelines": {
        "description": "Already-generated LeetCode guideline JSONs (3,982 files)",
        "quality": "high — 94.7% complete for core fields",
        "coverage": "46 pattern categories, 3,982 problems",
        "best_for": [
            "guideline", "reasoning", "complexity.*", "pattern.*",
            "situation.*", "code_analysis.*",
        ],
        "limitation": "Only 27.5% have solution_code. code_analysis flags all false.",
    },
    "aws_architecture": {
        "description": "AWS Well-Architected Framework HTML (4 files)",
        "quality": "high — authoritative patterns",
        "coverage": "Cloud architecture patterns (not algorithmic)",
        "best_for": [
            "pattern.pattern_name (architectural)", "guideline (cloud patterns)",
            "reasoning (architecture decisions)",
        ],
        "limitation": "Architectural only, not algorithmic. Needs HTML parsing.",
    },
}

# ── Build comprehensive enrichment plan ────────────────────────────────────

def load_json(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def build_enrichment_plan():
    """Build field-by-field enrichment plan."""
    completeness = load_json("completeness_by_source.json")
    csv_mapping = load_json("csv_to_schema_mapping.json")
    cre_inv = load_json("cre_enrichment_inventory.json")

    plan = {}

    for category, fields in SCHEMA_V2.items():
        for field_path, description in fields.items():
            if field_path.endswith(".*"):
                continue  # handle wildcards separately

            # Current state: completeness across sources
            current_state = {}
            for src in ["leetcode", "cre-repos", "codewars", "textbooks", "pattern-profiles"]:
                src_data = completeness.get(src, {}).get("field_completeness", {}).get(field_path, {})
                current_state[src] = {
                    "populated": src_data.get("populated", 0),
                    "total": src_data.get("total", 0),
                    "pct": src_data.get("pct", 0),
                }

            # Enrichment sources available
            enrichment = []
            priority = "none"

            # Determine which sources can provide this field
            for src_name, src_info in ENRICHMENT_SOURCES.items():
                if field_path in src_info["best_for"] or any(
                    field_path.startswith(best.rstrip(".*"))
                    for best in src_info["best_for"]
                ):
                    enrichment.append({
                        "source": src_name,
                        "quality": src_info["quality"],
                        "method": "direct_mapping" if field_path in src_info["best_for"] else "derived",
                    })

            # Determine priority
            # HIGH: field is <30% populated in primary target (leetcode or cre-repos)
            # MEDIUM: field is 30-80% populated
            # LOW: field is >80% populated
            leetcode_pct = current_state.get("leetcode", {}).get("pct", 0)
            cre_pct = current_state.get("cre-repos", {}).get("pct", 0)
            worst_pct = min(leetcode_pct, cre_pct) if leetcode_pct > 0 or cre_pct > 0 else 0

            if worst_pct < 30 and enrichment:
                priority = "HIGH"
            elif worst_pct < 80 and enrichment:
                priority = "MEDIUM"
            elif enrichment:
                priority = "LOW"
            else:
                priority = "MONITOR"

            plan[field_path] = {
                "description": description,
                "category": category,
                "current_completeness": current_state,
                "enrichment_sources": enrichment,
                "priority": priority,
                "gap": f"Lowest: {worst_pct}%" if worst_pct < 100 else "Complete",
            }

    return plan


def generate_strategy_md(plan):
    """Generate comprehensive data mining strategy document."""

    lines = []
    lines.append("# Cross-Source ETL & Data Mining Strategy")
    lines.append(f"\n**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Based on:** Deep analysis of 5,805 guidelines, 500 CRE enrichments, 624 StackExchange JSONLs, 10 LeetCode CSVs")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")
    lines.append("| Finding | Metric |")
    lines.append("|---------|--------|")
    lines.append("| **Total guideline files** | 5,805 (3,982 LeetCode + 1,474 CRE + 122 Codewars + 214 Textbooks + 10 Profiles) |")
    lines.append("| **Total unique fields** | 1,477 across all sources |")
    lines.append("| **Biggest gap** | `solution_code` populated in only 27.5% of LeetCode, 0% of CRE, 0% of Codewars |")
    lines.append("| **CRE enrichment quality** | 40% rich/adequate, 60% thin/stub |")
    lines.append("| **8 CSVs have solution_code** | But ETL pipeline not transferring to JSONs |")
    lines.append("| **code_analysis flags** | All `False` — code analysis was never run |")
    lines.append("| **StackExchange data** | 1.2M Q&A records with code in answers, completely unprocessed |")
    lines.append("| **Identity keys** | 51 fields shared across ≥2 sources for entity resolution |")
    lines.append("")

    lines.append("### Priority Enrichment Pipeline")
    lines.append("")
    lines.append("```")
    lines.append("Phase 1 (CRITICAL): Backfill solution_code from CSVs")
    lines.append("  → 8 CSVs have code → 3,647 LeetCode JSONs missing code")
    lines.append("  → Run merge_csv_solution_codes.py")
    lines.append("")
    lines.append("Phase 2 (HIGH): Re-run code_analysis detection")
    lines.append("  → All code_analysis flags are False (detector never ran)")
    lines.append("  → Run code_analysis on solution_code using AST/pattern detection")
    lines.append("")
    lines.append("Phase 3 (HIGH): Re-enrich CRE stubs")
    lines.append("  → 60% (300/500) CRE enrichment files are stubs")
    lines.append("  → Re-run enrichment pipeline with deeper content extraction")
    lines.append("")
    lines.append("Phase 4 (MEDIUM): Process StackExchange for hints + alternatives")
    lines.append("  → Extract code blocks from answers (HTML <code>/<pre> tags)")
    lines.append("  → Map SE tags → patterns using TAG_PATTERN_HINTS")
    lines.append("  → Merge hints, alternatives, constraints into guidelines")
    lines.append("")
    lines.append("Phase 5 (MEDIUM): Fill bridging references")
    lines.append("  → bridges.code_repos: link to CRE repo code")
    lines.append("  → bridges.code_chunks: link to Qdrant code snippets")
    lines.append("  → bridges.textbook_chapters: link enriched chapters")
    lines.append("")
    lines.append("Phase 6 (LOW): Parse AWS architecture HTML")
    lines.append("  → Extract architectural patterns for cloud guidelines")
    lines.append("  → Map to microservices/design-pattern categories")
    lines.append("```")
    lines.append("")

    lines.append("## Field-by-Field Enrichment Plan")
    lines.append("")
    lines.append("### 🔴 HIGH Priority (critical gaps)")
    lines.append("")
    lines.append("| Field | Current Best | Gap | Enrichment Source |")
    lines.append("|-------|-------------|-----|-------------------|")
    for field_path, info in sorted(plan.items()):
        if info["priority"] == "HIGH":
            sources = ", ".join(e["source"] for e in info["enrichment_sources"])
            lines.append(f"| `{field_path}` | {info['gap']} | Critical | {sources} |")

    lines.append("")
    lines.append("### 🟡 MEDIUM Priority (partial gaps)")
    lines.append("")
    lines.append("| Field | Current Best | Gap | Enrichment Source |")
    lines.append("|-------|-------------|-----|-------------------|")
    for field_path, info in sorted(plan.items()):
        if info["priority"] == "MEDIUM":
            sources = ", ".join(e["source"] for e in info["enrichment_sources"])
            lines.append(f"| `{field_path}` | {info['gap']} | Partial | {sources} |")

    lines.append("")
    lines.append("### 🟢 LOW / MONITOR Priority")
    lines.append("")
    lines.append("| Field | Current Best | Status |")
    lines.append("|-------|-------------|--------|")
    for field_path, info in sorted(plan.items()):
        if info["priority"] in ("LOW", "MONITOR"):
            lines.append(f"| `{field_path}` | {info['gap']} | {info['priority']} |")

    lines.append("")
    lines.append("## Data Mining Best Practices Applied")
    lines.append("")
    lines.append("### 1. Entity Resolution Strategy")
    lines.append("- **51 identity keys** shared across ≥2 sources enable cross-source merging")
    lines.append("- Primary keys: `guideline_id` (LeetCode), `source_problem_id` (CSV→LeetCode), `title_slug` (fuzzy match)")
    lines.append("- SE tag→pattern mapping enables algorithmic classification of unlabeled Q&A")
    lines.append("")
    lines.append("### 2. Schema Harmonization")
    lines.append("- Canonical v2.0 schema with 7 categories: core_identity, situation, approach, pattern, code, metadata, bridges")
    lines.append("- Discovered 1,477 unique field paths across sources → consolidated to canonical")
    lines.append("- All sources map to v2.0 via FIELD_SYNONYMS lookup table")
    lines.append("")
    lines.append("### 3. Quality-Driven Enrichment")
    lines.append("- Gold/Silver/Bronze tiers for StackExchange records (by score + code presence)")
    lines.append("- Rich/Adequate/Thin/Stub tiers for CRE enrichment files")
    lines.append("- Only enrich from gold-tier sources; flag thin/stub sources for re-enrichment")
    lines.append("")
    lines.append("### 4. Incremental Pipeline Design")
    lines.append("- Each phase produces independently usable output")
    lines.append("- Phase N output serves as input to Phase N+1")
    lines.append("- Idempotent: re-running any phase produces identical results")
    lines.append("")
    lines.append("### 5. Gap Analysis First, ETL Second")
    lines.append("- Analyzed ALL data BEFORE designing ETL (not the other way around)")
    lines.append("- Completeness matrix identifies exactly which fields need enrichment")
    lines.append("- Cross-source field matrix eliminates redundant enrichment work")
    lines.append("")

    lines.append("## Data Source Inventory")
    lines.append("")
    lines.append("| Source | Files | Records | Quality | Status |")
    lines.append("|--------|-------|---------|---------|--------|")
    lines.append("| LeetCode Guidelines | 3,982 JSON | 3,982 problems | 94.7% core fields | ✅ Good, missing solution_code |")
    lines.append("| CRE Guidelines | 1,474 JSON | 1,474 patterns | 99.9% core fields | ⚠️ Thin skeletons, no solution_code |")
    lines.append("| Codewars Guidelines | 122 JSON | 122 problems | 100% core fields | ⚠️ No solution_code |")
    lines.append("| Textbook Profiles | 214 JSON | 214 patterns | 0% guideline fields | ❌ Different schema (profiles, not guidelines) |")
    lines.append("| Pattern Profiles | 10 JSON | 10 profiles | 0% guideline fields | ❌ Different schema |")
    lines.append("| CRE Enrichment | 500 JSON | 98 domains | 40% rich | ⚠️ 60% stubs |")
    lines.append("| LeetCode CSVs | 10 CSV | ~3,647 problems | High raw data | ⚠️ 8 have solution_code, not merged |")
    lines.append("| StackExchange JSONL | 624 JSONL | ~1.2M Q&A | Varies by site | ❌ Completely unprocessed |")
    lines.append("| AWS Architecture | 4 HTML | — | High (authoritative) | ❌ Unparsed |")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 72)
    print("  CROSS-SOURCE ENRICHMENT MAPPER & STRATEGY GENERATOR")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 72)

    # ── Build enrichment plan ──────────────────────────────────────────────
    plan = build_enrichment_plan()

    # Summary
    priorities = Counter(info["priority"] for info in plan.values())
    print(f"\n📊 Enrichment Priority Distribution:")
    for p in ["HIGH", "MEDIUM", "LOW", "MONITOR"]:
        print(f"  {p}: {priorities.get(p, 0)} fields")

    print(f"\n🔴 HIGH Priority Fields:")
    for field_path, info in sorted(plan.items()):
        if info["priority"] == "HIGH":
            sources = ", ".join(e["source"] for e in info["enrichment_sources"])
            print(f"  {field_path}: {info['gap']} → {sources}")

    # ── Generate strategy document ──────────────────────────────────────────
    strategy_md = generate_strategy_md(plan)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write enrichment plan JSON
    with open(os.path.join(OUTPUT_DIR, "cross_source_enrichment_map.json"), "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "priority_summary": dict(priorities),
            "field_plan": plan,
        }, f, indent=2)

    # Write strategy markdown
    with open(os.path.join(OUTPUT_DIR, "data_mining_strategy.md"), "w") as f:
        f.write(strategy_md)

    print(f"\n  ✅ Wrote: {os.path.join(OUTPUT_DIR, 'cross_source_enrichment_map.json')}")
    print(f"  ✅ Wrote: {os.path.join(OUTPUT_DIR, 'data_mining_strategy.md')}")
    print(f"\n{'=' * 72}")
    print(f"  CROSS-SOURCE ANALYSIS COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
