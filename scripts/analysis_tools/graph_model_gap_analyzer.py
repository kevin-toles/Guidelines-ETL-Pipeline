#!/usr/bin/env python3
"""
Guideline Graph-Model Gap Analyzer & ETL Strategy Generator
===========================================================
Maps the CURRENT guideline data model (schema v2.0) to the TARGET graph model
and identifies exactly what data mining/transformation is needed from each
available data source to bridge the gap.

Target Graph Model (from user specification):
  Guideline → applies_to    → Scenario     (scenario families, problem shapes)
  Guideline → requires      → Constraint   (preconditions, input shape reqs)
  Guideline → disqualifies  → Constraint   (anti-constraints: when NOT to use)
  Guideline → recommends    → Concept      (preferred approach, pattern)
  Guideline → warns_about   → FailureMode  (anti-patterns, common mistakes)
  Guideline → validated_by  → Check        (validation checks, edge case tests)
  Guideline → supported_by  → Artifact     (evidence, citations)
  Guideline → implemented_by → Artifact    (code, reference implementation)
  Guideline → bounded_by    → Artifact     (scope, domain boundaries)

Guideline-level concepts:
  scenario_families, problem_shapes, constraints, preferred_approaches,
  disqualifiers, tradeoffs, anti_patterns, validation_checks
"""

import json, os, sys, re
from collections import defaultdict, Counter
from datetime import datetime

OUTPUT_DIR = "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines/scripts/analysis_tools/output"

GUIDELINES_ROOT = "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines/guidelines"
ETL_ROOT = "/Volumes/USB321FD/Guidelines ETL Data"

# ── Target Graph Model Definition ────────────────────────────────────────
GRAPH_MODEL = {
    "nodes": {
        "Guideline": "A software development guideline derived from analysis of common scenarios",
        "Scenario": "A problem shape / scenario family (e.g., 'sorted interval merging', 'bounded sliding window')",
        "Constraint": "A condition that must be met, or that disqualifies an approach",
        "Concept": "A software engineering concept or pattern (e.g., 'Sliding Window', 'Hash-based Lookup')",
        "FailureMode": "An anti-pattern, common mistake, or failure mode to avoid",
        "Check": "A validation check or edge case test",
        "Artifact": "A concrete artifact: code snippet, textbook chapter, repo reference, diagram",
    },
    "edges": {
        "applies_to":    ("Guideline", "Scenario",    "This guideline applies to this scenario family"),
        "requires":      ("Guideline", "Constraint",  "This guideline requires this precondition"),
        "disqualifies":  ("Guideline", "Constraint",  "This constraint disqualifies this guideline"),
        "recommends":    ("Guideline", "Concept",      "This guideline recommends this pattern/approach"),
        "warns_about":   ("Guideline", "FailureMode",  "This guideline warns about this anti-pattern"),
        "validated_by":  ("Guideline", "Check",        "This guideline is validated by this check"),
        "supported_by":  ("Guideline", "Artifact",     "This guideline is supported by this evidence"),
        "implemented_by": ("Guideline", "Artifact",    "This guideline is implemented by this code"),
        "bounded_by":    ("Guideline", "Artifact",     "This guideline is scoped by this boundary"),
    },
}

# ── CURRENT → TARGET MAPPING ────────────────────────────────────────────
# Maps current schema v2.0 fields → target graph concepts
CURRENT_TO_TARGET = {
    # ─── Direct mappings (current field already serves target concept) ───
    "situation": {
        "target_node": "Scenario",
        "target_edge": "applies_to",
        "quality": "partial",  # Has summary text but NOT scenario_family or problem_shape
        "gap": "Missing: scenario_family enum, problem_shape classification, structured scenario decomposition",
    },
    "situation.tags": {
        "target_node": "Scenario",
        "target_edge": "applies_to",
        "quality": "rich",
        "gap": "Tags are boolean flags (has_sorted_input, has_duplicates) — useful but not structured scenario families",
    },
    "situation.difficulty": {
        "target_node": "Scenario",
        "target_edge": "applies_to",
        "quality": "adequate",
        "gap": "Difficulty alone doesn't define scenario shape",
    },
    "pattern": {
        "target_node": "Concept",
        "target_edge": "recommends",
        "quality": "partial",
        "gap": "Has pattern_id/name/category but NOT detailed concept definition, no relationship graph to other concepts",
    },
    "guideline": {
        "target_node": "Concept",
        "target_edge": "recommends",
        "quality": "rich",  # AI-generated guideline text is good
        "gap": "Guideline text exists but isn't structured as 'preferred approach' with explicit decision logic",
    },
    "reasoning": {
        "target_node": "Concept",
        "target_edge": "recommends",  # Part of recommending why this approach
        "quality": "rich",
        "gap": "Reasoning explains 'why' but doesn't contrast with alternatives as tradeoffs",
    },
    "complexity": {
        "target_node": "Constraint",
        "target_edge": "requires",  # Time/space complexity ARE constraints
        "quality": "partial",
        "gap": "Has time/space but not input constraints, precondition constraints, or disqualifier conditions",
    },
    "constraints": {
        "target_node": "Constraint",
        "target_edge": "requires",
        "quality": "thin",
        "gap": "Generic platitudes ('Handle edge cases') — NOT actionable structured constraints with predicates. Missing: input size bounds, monotonicity reqs, data structure preconditions",
    },
    "alternatives": {
        "target_node": "Concept",  # Alternative approaches
        "target_edge": "recommends",
        "quality": "thin",
        "gap": "Limited to 1-2 alternatives with vague complexity. Missing: tradeoff comparison table, disqualifier conditions, when to prefer each",
    },
    "solution_code": {
        "target_node": "Artifact",
        "target_edge": "implemented_by",
        "quality": "rich",  # When present (27.5%)
        "gap": "Only 27.5% LeetCode have it, 0% CRE. Missing: multiple implementation styles, inline annotation of pattern application",
    },
    "bridges": {
        "target_node": "Artifact",
        "target_edge": "supported_by | bounded_by",  # Links to repos, chapters, diagrams
        "quality": "empty",
        "gap": "ALL bridges fields are empty arrays. Zero artifact connections. This is the single biggest structural gap.",
    },
    "code_analysis": {
        "target_node": "Concept",  # Informs pattern detection
        "target_edge": "recommends",
        "quality": "thin",
        "gap": "Detection flags (uses_dict, uses_set, etc.) are all False. Code analysis never actually ran.",
    },
    "hints": {
        "target_node": "Concept",
        "target_edge": "recommends",  # Progressive hints guide approach selection
        "quality": "partial",
        "gap": "Hints exist for some problems but not linked to pattern stages or decision points",
    },
    "similar_questions": {
        "target_node": "Scenario",
        "target_edge": "applies_to",  # Links to other scenarios in same family
        "quality": "adequate",
        "gap": "Good cross-references but not organized into scenario families or problem shape taxonomy",
    },

    # ─── Missing entirely from current schema ───
    "tradeoffs": {
        "target_node": "Concept",
        "target_edge": "disqualifies",
        "quality": "missing",
        "gap": "NO tradeoff analysis exists. Need: comparison tables, when to prefer X over Y, performance characteristics under different constraints",
    },
    "disqualifiers": {
        "target_node": "Constraint",
        "target_edge": "disqualifies",
        "quality": "missing",
        "gap": "NO disqualifier conditions. Need: 'don't use this when...' conditions (e.g., don't use two-pointer if input unsorted)",
    },
    "anti_patterns": {
        "target_node": "FailureMode",
        "target_edge": "warns_about",
        "quality": "missing",
        "gap": "NO anti-pattern documentation. Need: common mistakes, off-by-one errors, incorrect termination conditions",
    },
    "failure_modes": {
        "target_node": "FailureMode",
        "target_edge": "warns_about",
        "quality": "missing",
        "gap": "NO failure mode analysis. Need: what breaks, edge cases that cause incorrect output, performance degradation scenarios",
    },
    "validation_checks": {
        "target_node": "Check",
        "target_edge": "validated_by",
        "quality": "missing",
        "gap": "NO validation checks. Need: edge case test cases, boundary condition tests, property-based test templates",
    },
    "scenario_family": {
        "target_node": "Scenario",
        "target_edge": "applies_to",
        "quality": "missing",
        "gap": "NO scenario family taxonomy. Need: hierarchical classification (e.g., 'Interval Manipulation → Merging → Insert Non-Overlapping')",
    },
    "problem_shape": {
        "target_node": "Scenario",
        "target_edge": "applies_to",
        "quality": "missing",
        "gap": "NO problem shape decomposition. Need: input shape signature, output shape, transformation type (map/filter/reduce/aggregate/search)",
    },
    "preferred_approach": {
        "target_node": "Concept",
        "target_edge": "recommends",
        "quality": "missing",
        "gap": "Guideline text exists but NOT structured as explicit decision tree with ranked approaches. Need: primary approach, fallback approach, when-each-wins table",
    },
    "decision_graph": {
        "target_node": None,  # Cross-cutting
        "target_edge": None,  # Meta-structure
        "quality": "missing",
        "gap": "NO decision graph linking constraints → approaches → tradeoffs → checks. This is the meta-structure that makes guidelines navigable.",
    },
}


# ── DATA SOURCE ASSESSMENT ──────────────────────────────────────────────
DATA_SOURCES = {
    "leetcode_guidelines": {
        "path": f"{GUIDELINES_ROOT}/*/guideline_*.json",
        "count": "3,982",
        "has_code": "27.5%",
        "strengths": [
            "Rich guideline text (AI-generated)",
            "Good situation.tags (boolean flags for input characteristics)",
            "Complexity data (time/space)",
            "Some have solution_code + solution_description (rich multi-approach analysis)",
            "Hints and similar_questions provide cross-references",
        ],
        "weaknesses": [
            "Code missing from 72.5%",
            "code_analysis flags all False",
            "Bridges ALL empty",
            "Constraints are generic platitudes",
            "No anti-patterns, failure modes, or validation checks",
            "No tradeoff analysis between approaches",
            "Alternatives vague (only 1-2, no comparison)",
        ],
    },
    "cre_guidelines": {
        "path": f"{GUIDELINES_ROOT}/cre-repos/*_guideline_*.json",
        "count": "1,474",
        "has_code": "0%",
        "strengths": [
            "Consistent schema v2.0 structure",
            "Good metadata.source/domain/repo_id linking",
            "Bridges to CRE repos conceptually (though empty in JSON)",
            "Pattern mapping to real-world repos (474 repos)",
        ],
        "weaknesses": [
            "NO solution_code",
            "Generic guideline text (template: 'consider the approach based on design patterns')",
            "Empty constraints, alternatives",
            "Empty bridges",
            "Thin code_analysis (only primary_languages, no detection)",
            "Created from repo structure, not from code analysis",
        ],
    },
    "leetcode_csvs": {
        "path": f"{GUIDELINES_ROOT}/../scripts/*.csv",
        "count": "10 CSVs, ~3,647 problems",
        "strengths": [
            "8 CSVs have solution_code (Python/Java/C++/etc.)",
            "Have acceptance_rate, likes, dislikes, description",
            "Can backfill missing code into guidelines",
        ],
        "weaknesses": [
            "No AI-generated reasoning or guideline text",
            "No pattern classification",
            "Not mapped to guideline IDs",
            "Solution code not extracted into guidelines",
        ],
    },
    "stackexchange_jsonl": {
        "path": f"{ETL_ROOT}/ai-platform-output/",
        "count": "315 JSONL files (105 sites × 3 tiers)",
        "strengths": [
            "Massive Q&A corpus with code in answers",
            "Tags provide domain classification",
            "Answers contain implementation variants and edge-case discussion",
            "Votes/scores provide quality signal",
            "Comments contain corrections and improvements",
        ],
        "weaknesses": [
            "Not algorithmically tagged (SE tags are site-categories, not pattern IDs)",
            "Code is embedded in HTML (needs extraction)",
            "No guideline structure — raw Q&A",
            "Quality varies widely by site and record",
            "No direct mapping to pattern taxonomy",
        ],
    },
    "cre_enrichment": {
        "path": "/Users/kevintoles/POC/ai-platform-data/repos/enriched/",
        "count": "500 files, 98 domains",
        "strengths": [
            "Rich content extraction from 40% (adequate+rich)",
            "Domain classification and concept extraction",
            "Textbook chapter links",
            "Pattern identification from real-world code",
        ],
        "weaknesses": [
            "60% stubs (need re-enrichment)",
            "No guideline-format output",
            "Pattern links exist but not mapped to guideline pattern taxonomy",
        ],
    },
    "pattern_profiles": {
        "path": f"{GUIDELINES_ROOT}/pattern_profiles/",
        "count": "8 profiles",
        "strengths": [
            "Structured pattern definitions (when_to_use, implementation, variants)",
            "Related problems list",
            "Complexity analysis",
        ],
        "weaknesses": [
            "Only 8 patterns covered (grokking subset)",
            "No mapping to 3,982 guideline problems",
            "No decision graph structure",
        ],
    },
    "pattern_index": {
        "path": f"{GUIDELINES_ROOT}/_pattern_index.json",
        "count": "41 patterns",
        "strengths": [
            "Complete pattern taxonomy (41 algorithmic patterns)",
            "Problem counts per pattern",
            "Pattern categories",
        ],
        "weaknesses": [
            "Flat list — no hierarchy, no relationships between patterns",
            "No scenario family decomposition",
        ],
    },
    "aws_architecture": {
        "path": f"{ETL_ROOT}/aws-architecture/",
        "count": "4 HTML files",
        "strengths": [
            "Authoritative cloud architecture patterns",
            "Well-Architected Framework",
        ],
        "weaknesses": [
            "HTML format (needs parsing)",
            "Architectural only — not algorithmic",
            "No mapping to guideline format",
        ],
    },
}


def analyze_guideline_depth(sample_size=30):
    """Analyze actual guideline content depth across LeetCode and CRE sources."""
    import glob

    results = {
        "leetcode": {
            "files": 0,
            "has_solution_code": 0,
            "has_solution_description": 0,
            "constraints_non_empty": 0,
            "constraints_avg_count": 0,
            "alternatives_non_empty": 0,
            "alternatives_avg_count": 0,
            "bridges_non_empty": 0,
            "bridges_fields_populated": Counter(),
            "code_analysis_detected": 0,  # Any detection flag True
            "similar_questions_avg": 0,
            "hints_available": 0,
            "pattern_categories": set(),
            "situation_topics": set(),
            "situation_tags_used": Counter(),
        },
        "cre": {
            "files": 0,
            "has_solution_code": 0,
            "constraints_non_empty": 0,
            "constraints_avg_count": 0,
            "alternatives_non_empty": 0,
            "bridges_non_empty": 0,
            "code_analysis_keys": Counter(),
            "pattern_categories": set(),
            "situation_topics": set(),
        },
    }

    # Sample LeetCode guidelines
    leetcode_dirs = [d for d in glob.glob(f"{GUIDELINES_ROOT}/*/")
                     if os.path.isdir(d) and 'cre-repos' not in d
                     and 'pattern_profiles' not in d and 'patterns' not in d]
    sampled = 0
    for d in leetcode_dirs:
        files = glob.glob(os.path.join(d, "guideline_*.json"))
        for f in files:
            if sampled >= sample_size:
                break
            try:
                with open(f) as fh:
                    g = json.load(fh)
            except:
                continue
            sampled += 1
            r = results["leetcode"]
            r["files"] += 1
            if g.get("solution_code") and isinstance(g["solution_code"], dict) and g["solution_code"]:
                r["has_solution_code"] += 1
            if g.get("solution_description"):
                r["has_solution_description"] += 1
            if g.get("constraints") and len(g["constraints"]) > 0:
                r["constraints_non_empty"] += 1
                r["constraints_avg_count"] += len(g["constraints"])
            if g.get("alternatives") and len(g["alternatives"]) > 0:
                r["alternatives_non_empty"] += 1
                r["alternatives_avg_count"] += len(g["alternatives"])
            bridges = g.get("bridges", {})
            if isinstance(bridges, dict):
                for bk, bv in bridges.items():
                    if bv and len(bv) > 0:
                        r["bridges_fields_populated"][bk] += 1
            elif isinstance(bridges, list) and len(bridges) > 0:
                r["bridges_non_empty"] += 1
            ca = g.get("code_analysis", {})
            detection_flags = ["uses_dict", "uses_set", "uses_list", "uses_deque",
                              "uses_heap", "uses_sorting", "uses_recursion",
                              "has_nested_loops", "has_while_loop", "has_for_loop"]
            if any(ca.get(f) for f in detection_flags):
                r["code_analysis_detected"] += 1
            if g.get("similar_questions"):
                r["similar_questions_avg"] += len(g["similar_questions"])
            if g.get("has_hints"):
                r["hints_available"] += 1
            pat = g.get("pattern", {})
            if pat.get("category"):
                r["pattern_categories"].add(pat["category"])
            sit = g.get("situation", {})
            if sit.get("topics"):
                for t in sit["topics"]:
                    if isinstance(t, str):
                        r["situation_topics"].add(t)
                    elif isinstance(t, dict):
                        r["situation_topics"].add(t.get("name", str(t)))
            if sit.get("tags"):
                for tk, tv in sit["tags"].items():
                    if tv:
                        r["situation_tags_used"][tk] += 1
            if r["files"] > 0:
                r["constraints_avg"] = round(r["constraints_avg_count"] / r["files"], 1)
                r["alternatives_avg"] = round(r["alternatives_avg_count"] / r["files"], 1) if r["files"] else 0
                r["similar_questions_avg_val"] = round(r["similar_questions_avg"] / r["files"], 1)

    # Sample CRE guidelines
    cre_files = glob.glob(f"{GUIDELINES_ROOT}/cre-repos/*_guideline_*.json")
    sampled_cre = 0
    for f in cre_files:
        if sampled_cre >= sample_size:
            break
        try:
            with open(f) as fh:
                g = json.load(fh)
        except:
            continue
        sampled_cre += 1
        r = results["cre"]
        r["files"] += 1
        if g.get("solution_code") and isinstance(g["solution_code"], dict) and g["solution_code"]:
            r["has_solution_code"] += 1
        if g.get("constraints") and len(g.get("constraints", [])) > 0:
            r["constraints_non_empty"] += 1
        if g.get("alternatives") and len(g.get("alternatives", [])) > 0:
            r["alternatives_non_empty"] += 1
        bridges = g.get("bridges", [])
        if isinstance(bridges, list) and len(bridges) > 0:
            r["bridges_non_empty"] += 1
        ca = g.get("code_analysis", {})
        for k in ca:
            r["code_analysis_keys"][k] += 1
        pat = g.get("pattern", {})
        if pat.get("category"):
            r["pattern_categories"].add(pat["category"])
        sit = g.get("situation", {})
        if sit.get("topics"):
            for t in sit["topics"]:
                if isinstance(t, str):
                    r["situation_topics"].add(t)
                elif isinstance(t, dict):
                    r["situation_topics"].add(t.get("name", str(t)))

    # Convert sets to lists for JSON
    results["leetcode"]["pattern_categories"] = list(results["leetcode"]["pattern_categories"])
    results["leetcode"]["situation_topics"] = list(results["leetcode"]["situation_topics"])
    results["cre"]["pattern_categories"] = list(results["cre"]["pattern_categories"])
    results["cre"]["situation_topics"] = list(results["cre"]["situation_topics"])
    results["leetcode"]["bridges_fields_populated"] = dict(results["leetcode"]["bridges_fields_populated"])
    results["leetcode"]["situation_tags_used"] = dict(results["leetcode"]["situation_tags_used"])
    results["cre"]["code_analysis_keys"] = dict(results["cre"]["code_analysis_keys"])

    return results


def generate_gap_matrix():
    """Generate the complete gap matrix: target concept → current state → data source → transformation."""
    matrix = []

    for field_key, info in CURRENT_TO_TARGET.items():
        row = {
            "target_concept": info["target_node"],
            "target_edge": info["target_edge"],
            "current_field": field_key,
            "current_quality": info["quality"],
            "gap_description": info["gap"],
            "enrichment_sources": [],
            "transformation_required": "",
            "priority": "",
        }

        # Determine which data sources can fill this gap
        if "solution_code" in field_key:
            row["enrichment_sources"] = ["leetcode_csvs", "stackexchange_jsonl"]
            row["transformation_required"] = "Extract code from CSVs → match by problem_id → merge into guideline JSON. Also extract accepted answers' code from SE JSONL → NLP-tag by pattern → insert as reference implementations."
            row["priority"] = "CRITICAL"

        elif "bridges" in field_key:
            row["enrichment_sources"] = ["cre_enrichment", "cre_guidelines", "leetcode_guidelines"]
            row["transformation_required"] = "Build bridge index: match guideline pattern_id → CRE repo pattern_id → code_chunks collection. Cross-reference textbook_chapters by pattern domain. Generate pattern_links graph."
            row["priority"] = "CRITICAL"

        elif "anti_pattern" in field_key or "failure_mode" in field_key:
            row["enrichment_sources"] = ["stackexchange_jsonl", "leetcode_guidelines"]
            row["transformation_required"] = "Mine SE answers/comments for 'don't do X', 'common mistake', 'gotcha' patterns. Extract from solution_description sections that mention pitfalls. NLP classification into failure mode taxonomy."
            row["priority"] = "HIGH"

        elif "validation_check" in field_key:
            row["enrichment_sources"] = ["leetcode_guidelines", "stackexchange_jsonl"]
            row["transformation_required"] = "Extract edge case tests from solution_description. Mine SE for test cases in answers. Generate property-based test templates from situation.tags conditions."
            row["priority"] = "HIGH"

        elif "tradeoff" in field_key or "disqualifier" in field_key:
            row["enrichment_sources"] = ["leetcode_guidelines", "stackexchange_jsonl", "pattern_profiles"]
            row["transformation_required"] = "Mine solution_description's multiple approaches for tradeoff language ('however', 'but', 'while X is faster, Y uses less memory'). Extract disqualifier conditions from pattern profiles' when_to_use fields. Build comparison matrix."
            row["priority"] = "HIGH"

        elif "scenario_family" in field_key or "problem_shape" in field_key:
            row["enrichment_sources"] = ["pattern_index", "leetcode_guidelines", "cre_enrichment"]
            row["transformation_required"] = "Build hierarchical taxonomy from _pattern_index.json + situation.tags + situation.topics. Cluster similar_questions into scenario families. Generate problem_shape signature from tags (sorted + array + merge = IntervalMerging)."
            row["priority"] = "HIGH"

        elif "decision_graph" in field_key:
            row["enrichment_sources"] = ["ALL"]
            row["transformation_required"] = "Synthesize from ALL enriched fields: constraints → disqualifiers → approaches → tradeoffs → failure modes → checks. This is the meta-structure built AFTER all other fields are enriched."
            row["priority"] = "MEDIUM"

        elif "preferred_approach" in field_key:
            row["enrichment_sources"] = ["leetcode_guidelines", "pattern_profiles"]
            row["transformation_required"] = "Reformat existing guideline + reasoning into structured 'preferred approach' with explicit decision logic. When multiple solutions exist in solution_description, rank them with when-each-wins criteria."
            row["priority"] = "MEDIUM"

        elif "constraint" in field_key:
            row["enrichment_sources"] = ["leetcode_guidelines", "stackexchange_jsonl", "pattern_profiles"]
            row["transformation_required"] = "Replace generic constraint strings with structured predicates: {type: 'input_size', predicate: 'n ≤ 10^5', source: 'problem_description'}. Extract from solution_description complexity sections."
            row["priority"] = "HIGH"

        elif "complexity" in field_key:
            row["enrichment_sources"] = ["leetcode_csvs", "leetcode_guidelines"]
            row["transformation_required"] = "Already has time/space — needs augmentation with input-size constraints, monotonicity requirements, data structure preconditions."
            row["priority"] = "MEDIUM"

        elif "alternative" in field_key:
            row["enrichment_sources"] = ["leetcode_guidelines", "stackexchange_jsonl"]
            row["transformation_required"] = "Expand alternatives from 1-2 to N approaches. For each: add disqualifier conditions, performance characteristics under different constraints, when-this-wins criteria."
            row["priority"] = "MEDIUM"

        elif "code_analysis" in field_key:
            row["enrichment_sources"] = ["leetcode_csvs"]
            row["transformation_required"] = "Actually run AST-based code analysis on backfilled solution_code. Detect data structures, loop patterns, recursion, pointer techniques, sorting. This is Phase 2 (depends on Phase 1 code backfill)."
            row["priority"] = "HIGH"

        elif "hint" in field_key:
            row["enrichment_sources"] = ["leetcode_csvs", "stackexchange_jsonl"]
            row["transformation_required"] = "Backfill hints from CSVs. Mine SE answers for progressive revelation patterns (hint 1 → hint 2 → solution)."
            row["priority"] = "MEDIUM"

        elif "similar_question" in field_key:
            row["enrichment_sources"] = ["leetcode_csvs", "stackexchange_jsonl"]
            row["transformation_required"] = "Enhance similar_questions with SE cross-references. Link to scenario_family once taxonomy is built."
            row["priority"] = "LOW"

        else:
            row["enrichment_sources"] = ["leetcode_guidelines"]
            row["transformation_required"] = "Current content is adequate. Monitor."
            row["priority"] = "LOW"

        matrix.append(row)

    return matrix


def generate_strategy_md(depth_analysis, gap_matrix):
    """Generate comprehensive ETL strategy Markdown document."""

    lines = []
    lines.append("# Guideline Graph-Model Gap Analysis & ETL Strategy")
    lines.append(f"\n**Generated:** {datetime.now().isoformat()}")
    lines.append("**Purpose:** Map current guideline data → target graph model, identify gaps, define ETL pipeline")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 1. TARGET MODEL ──
    lines.append("## 1. Target Graph Model (End State)")
    lines.append("")
    lines.append("The guideline system must function as a **guidance graph / decision graph** with these node types:")
    lines.append("")
    lines.append("| Node | Description |")
    lines.append("|------|-------------|")
    for node, desc in GRAPH_MODEL["nodes"].items():
        lines.append(f"| `{node}` | {desc} |")
    lines.append("")
    lines.append("And these edges:")
    lines.append("")
    lines.append("| Edge | From → To | Meaning |")
    lines.append("|------|-----------|---------|")
    for edge, (src, tgt, meaning) in GRAPH_MODEL["edges"].items():
        lines.append(f"| `{edge}` | {src} → {tgt} | {meaning} |")
    lines.append("")
    lines.append("**Guideline-level concepts:** scenario_families, problem_shapes, constraints, preferred_approaches, disqualifiers, tradeoffs, anti_patterns, validation_checks")
    lines.append("")

    # ── 2. CURRENT STATE ──
    lines.append("## 2. Current State: Schema v2.0 Content Depth Analysis")
    lines.append("")

    leet = depth_analysis["leetcode"]
    cre = depth_analysis["cre"]

    lines.append(f"### 2.1 LeetCode Guidelines (sampled {leet['files']} of 3,982)")
    lines.append("")
    lines.append("| Metric | Value | Assessment |")
    lines.append("|--------|-------|------------|")
    lines.append(f"| Has solution_code | {leet['has_solution_code']}/{leet['files']} ({round(leet['has_solution_code']/leet['files']*100)}%) | 🔴 Critical gap |")
    lines.append(f"| Has solution_description | {leet['has_solution_description']}/{leet['files']} ({round(leet['has_solution_description']/leet['files']*100)}%) | 🟢 Rich multi-approach content |")
    lines.append(f"| Constraints non-empty | {leet['constraints_non_empty']}/{leet['files']} | 🟡 Generic platitudes |")
    lines.append(f"| Avg constraints per guideline | {leet.get('constraints_avg', 0)} | 🔴 Too few, not structured |")
    lines.append(f"| Alternatives non-empty | {leet['alternatives_non_empty']}/{leet['files']} | 🟡 Limited |")
    lines.append(f"| Avg alternatives per guideline | {leet.get('alternatives_avg', 0)} | 🔴 Too few |")
    lines.append(f"| Bridges populated (any field) | {sum(leet['bridges_fields_populated'].values())} total | 🔴 Almost ALL empty |")
    lines.append(f"| code_analysis detected (any flag) | {leet['code_analysis_detected']}/{leet['files']} | 🔴 Detector never ran |")
    lines.append(f"| Has hints | {leet['hints_available']}/{leet['files']} | 🟡 Partial |")
    lines.append(f"| Avg similar_questions | {leet.get('similar_questions_avg_val', 0)} | 🟢 Adequate |")
    lines.append("")

    lines.append(f"### 2.2 CRE Guidelines (sampled {cre['files']} of 1,474)")
    lines.append("")
    lines.append("| Metric | Value | Assessment |")
    lines.append("|--------|-------|------------|")
    lines.append(f"| Has solution_code | {cre['has_solution_code']}/{cre['files']} | 🔴 ZERO |")
    lines.append(f"| Constraints non-empty | {cre['constraints_non_empty']}/{cre['files']} | 🔴 Empty |")
    lines.append(f"| Alternatives non-empty | {cre['alternatives_non_empty']}/{cre['files']} | 🔴 Empty |")
    lines.append(f"| Bridges non-empty | {cre['bridges_non_empty']}/{cre['files']} | 🔴 Empty |")
    lines.append(f"| Guideline quality | Generic template | 🔴 Not useful as-is |")
    lines.append("")

    # ── 3. GAP MATRIX ──
    lines.append("## 3. Complete Gap Matrix: Current → Target")
    lines.append("")

    priorities = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
    for row in gap_matrix:
        priorities[row["priority"]].append(row)

    for priority_level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if not priorities[priority_level]:
            continue
        emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}[priority_level]
        lines.append(f"### {emoji} {priority_level} Priority Gaps")
        lines.append("")
        lines.append("| Target Concept | Current Field | Quality | Gap | Sources |")
        lines.append("|---------------|---------------|---------|-----|---------|")
        for row in priorities[priority_level]:
            sources = ", ".join(row["enrichment_sources"])
            gap_short = row["gap_description"][:120] + "..." if len(row["gap_description"]) > 120 else row["gap_description"]
            lines.append(f"| `{row['target_concept']}` via `{row['target_edge']}` | `{row['current_field']}` | {row['current_quality']} | {gap_short} | {sources} |")
        lines.append("")

    # ── 4. TRANSFORMATION PIPELINE ──
    lines.append("## 4. ETL Transformation Pipeline")
    lines.append("")
    lines.append("### Phase Dependency Graph")
    lines.append("```")
    lines.append("Phase 1: Backfill solution_code (CRITICAL)")
    lines.append("  ├── From: 8 LeetCode CSVs → 3,982 guideline JSONs")
    lines.append("  └── Unblocks: Phase 2, Phase 3, Phase 4")
    lines.append("")
    lines.append("Phase 2: Run code_analysis (HIGH)")
    lines.append("  ├── From: solution_code (backfilled in Phase 1)")
    lines.append("  └── Produces: detection flags, algorithm identification")
    lines.append("")
    lines.append("Phase 3: Build bridges (HIGH)")
    lines.append("  ├── From: CRE enrichment + guideline pattern matching")
    lines.append("  └── Produces: code_repos, code_chunks, textbook_chapters")
    lines.append("")
    lines.append("Phase 4: Structured constraints (HIGH)")
    lines.append("  ├── From: solution_description + CSVs + SE answers")
    lines.append("  └── Produces: structured constraints, disqualifiers")
    lines.append("")
    lines.append("Phase 5: Tradeoffs & anti-patterns (HIGH)")
    lines.append("  ├── From: solution_description multi-approach sections + SE answers")
    lines.append("  └── Produces: tradeoff tables, failure_modes, anti_patterns")
    lines.append("")
    lines.append("Phase 6: Scenario families (HIGH)")
    lines.append("  ├── From: pattern_index + situation.tags + similar_questions")
    lines.append("  └── Produces: hierarchical scenario taxonomy, problem_shapes")
    lines.append("")
    lines.append("Phase 7: Validation checks (MEDIUM)")
    lines.append("  ├── From: solution_description edge cases + SE test cases")
    lines.append("  └── Produces: validation_checks, property-based test templates")
    lines.append("")
    lines.append("Phase 8: Decision graph (MEDIUM)")
    lines.append("  ├── From: ALL enriched fields")
    lines.append("  └── Produces: constraint→approach→tradeoff→check decision paths")
    lines.append("```")
    lines.append("")

    # ── 5. SCHEMA EXTENSIONS ──
    lines.append("## 5. Required Schema Extensions (v2.0 → v3.0)")
    lines.append("")
    lines.append("The current v2.0 schema is a **flat attribute bag** — it lists what a guideline IS but not what it CONNECTS TO. To support the graph model, these extensions are needed:")
    lines.append("")
    lines.append("### 5.1 New Top-Level Fields (v3.0)")
    lines.append("")
    lines.append("```json")
    lines.append("{")
    lines.append('  "scenario_family": {')
    lines.append('    "family_id": "interval_manipulation",')
    lines.append('    "family_name": "Interval Manipulation",')
    lines.append('    "subfamily": "merging",')
    lines.append('    "problem_shape": {')
    lines.append('      "input_shape": "sorted_intervals[][]",')
    lines.append('      "output_shape": "merged_intervals[][]",')
    lines.append('      "transformation": "merge"')
    lines.append("    }")
    lines.append("  },")
    lines.append('  "preferred_approach": {')
    lines.append('    "primary": "linear_scan",')
    lines.append('    "fallback": "binary_search",')
    lines.append('    "decision_rules": [')
    lines.append('      {"condition": "n < 10^4", "approach": "linear_scan"},')
    lines.append('      {"condition": "n >= 10^4", "approach": "binary_search"}')
    lines.append("    ]")
    lines.append("  },")
    lines.append('  "disqualifiers": [')
    lines.append('    {"condition": "input not sorted by start", "disqualifies": "two_pointer"},')
    lines.append('    {"condition": "intervals overlap in input", "disqualifies": "binary_search"}')
    lines.append("  ],")
    lines.append('  "tradeoffs": [')
    lines.append('    {"approaches": ["linear_scan", "binary_search"],')
    lines.append('     "dimension": "time_complexity",')
    lines.append('     "comparison": "Both O(n). Binary search find-insert position is O(log n) but merge still O(n)."}')
    lines.append("  ],")
    lines.append('  "anti_patterns": [')
    lines.append('    {"name": "modifying_input_array",')
    lines.append('     "description": "Mutating the input intervals array in-place causes..."},')
    lines.append('    {"name": "off_by_one_interval_boundary",')
    lines.append('     "description": "Using < instead of <= when comparing interval boundaries..."} ')
    lines.append("  ],")
    lines.append('  "failure_modes": [')
    lines.append('    {"mode": "empty_input", "behavior": "Returns []", "edge_case": true},')
    lines.append('    {"mode": "newInterval_consumes_all", "behavior": "Returns single merged interval"}')
    lines.append("  ],")
    lines.append('  "validation_checks": [')
    lines.append('    {"check": "output_sorted", "predicate": "all(result[i][0] <= result[i+1][0])"},')
    lines.append('    {"check": "no_overlap", "predicate": "all(result[i][1] < result[i+1][0])"}')
    lines.append("  ]")
    lines.append("}")
    lines.append("```")
    lines.append("")

    # ── 6. DATA MINING STRATEGY PER SOURCE ──
    lines.append("## 6. Data Mining Strategy Per Source")
    lines.append("")

    for src_name, src_info in DATA_SOURCES.items():
        lines.append(f"### 6.{list(DATA_SOURCES.keys()).index(src_name)+1} {src_name}")
        lines.append(f"- **Path:** `{src_info['path']}`")
        lines.append(f"- **Records:** {src_info['count']}")
        lines.append("")
        lines.append("**Strengths:**")
        for s in src_info["strengths"]:
            lines.append(f"- ✅ {s}")
        lines.append("")
        lines.append("**Weaknesses:**")
        for w in src_info["weaknesses"]:
            lines.append(f"- ❌ {w}")
        lines.append("")

    # ── 7. IMPLEMENTATION PLAN ──
    lines.append("## 7. Implementation Plan (Ordered by Dependency)")
    lines.append("")
    lines.append("| Phase | Script | Input | Output | Unblocks |")
    lines.append("|-------|--------|-------|--------|----------|")
    lines.append("| 1 | `merge_csv_solution_codes.py` | 8 LeetCode CSVs + 3,982 JSONs | Enriched JSONs with solution_code | Phase 2, 4, 5 |")
    lines.append("| 2 | `run_code_analysis.py` | Enriched JSONs (from Phase 1) | code_analysis flags populated | Phase 6, 7 |")
    lines.append("| 3 | `build_bridge_index.py` | CRE enrichment + guidelines + Qdrant | bridges populated | Phase 8 |")
    lines.append("| 4 | `structure_constraints.py` | solution_description + CSVs | Structured constraints + disqualifiers | Phase 8 |")
    lines.append("| 5 | `mine_tradeoffs.py` | solution_description + SE JSONL | tradeoffs + anti_patterns + failure_modes | Phase 8 |")
    lines.append("| 6 | `build_scenario_taxonomy.py` | pattern_index + situation.tags + similar_questions | scenario_family + problem_shape | Phase 8 |")
    lines.append("| 7 | `generate_validation_checks.py` | solution_description + SE answers | validation_checks | None |")
    lines.append("| 8 | `synthesize_decision_graph.py` | All enriched fields | decision_graph edge lists | None |")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 72)
    print("  GUIDELINE GRAPH-MODEL GAP ANALYZER")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 72)

    # ── Step 1: Analyze actual content depth ───────────────────────────
    print("\n📊 Analyzing guideline content depth (sampling)...")
    depth = analyze_guideline_depth(sample_size=50)

    print(f"  LeetCode: {depth['leetcode']['files']} sampled")
    print(f"    solution_code: {depth['leetcode']['has_solution_code']}/{depth['leetcode']['files']}")
    print(f"    constraints non-empty: {depth['leetcode']['constraints_non_empty']}/{depth['leetcode']['files']}")
    print(f"    bridges populated: {sum(depth['leetcode']['bridges_fields_populated'].values())} total")
    print(f"    code_analysis detected: {depth['leetcode']['code_analysis_detected']}/{depth['leetcode']['files']}")
    print(f"  CRE: {depth['cre']['files']} sampled")
    print(f"    solution_code: {depth['cre']['has_solution_code']}/{depth['cre']['files']}")
    print(f"    constraints non-empty: {depth['cre']['constraints_non_empty']}/{depth['cre']['files']}")

    # ── Step 2: Generate gap matrix ────────────────────────────────────
    print("\n📋 Generating gap matrix...")
    gap_matrix = generate_gap_matrix()

    priority_counts = Counter(row["priority"] for row in gap_matrix)
    for p in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        print(f"  {p}: {priority_counts.get(p, 0)} gaps")

    # ── Step 3: Generate strategy document ─────────────────────────────
    strategy_md = generate_strategy_md(depth, gap_matrix)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write gap matrix JSON
    with open(os.path.join(OUTPUT_DIR, "graph_model_gap_analysis.json"), "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "target_graph_model": GRAPH_MODEL,
            "current_to_target_mapping": {k: {
                "target_node": v["target_node"],
                "target_edge": v["target_edge"],
                "quality": v["quality"],
                "gap": v["gap"],
            } for k, v in CURRENT_TO_TARGET.items()},
            "gap_matrix": gap_matrix,
            "depth_analysis": {k: {
                kk: vv for kk, vv in v.items()
                if not isinstance(vv, (set, Counter))
            } for k, v in depth.items()},
            "priority_summary": dict(priority_counts),
        }, f, indent=2, default=str)

    # Write strategy MD
    with open(os.path.join(OUTPUT_DIR, "graph_model_etl_strategy.md"), "w") as f:
        f.write(strategy_md)

    print(f"\n  ✅ Wrote: {os.path.join(OUTPUT_DIR, 'graph_model_gap_analysis.json')}")
    print(f"  ✅ Wrote: {os.path.join(OUTPUT_DIR, 'graph_model_etl_strategy.md')}")
    print(f"\n{'=' * 72}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
