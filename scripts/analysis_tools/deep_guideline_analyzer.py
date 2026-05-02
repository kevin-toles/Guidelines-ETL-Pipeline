#!/usr/bin/env python3
"""
Deep Guideline Analyzer — v1.0
Inventory & completeness analysis of ALL guideline JSON files across all sources.

Covers:
  1. LeetCode guidelines (46 pattern subdirs, ~3,647 files)
  2. CRE repo guidelines (cre-repos/, ~1,474 files)
  3. Codewars guidelines (patterns/codewars/, ~122 files)
  4. Textbook guidelines (patterns/textbooks/, ~214 files)
  5. Pattern profiles (pattern_profiles/, ~10 files)

Outputs:
  - field_schema_map.json       — every unique field across all sources, with type frequencies
  - source_field_matrix.json    — which fields exist in which source
  - completeness_by_source.json — % populated per field per source
  - outliers_report.json        — files missing critical fields or with anomalous data
  - summary_report.md           — human-readable summary

Usage:
  python3 deep_guideline_analyzer.py [--guidelines-dir PATH] [--output-dir PATH]
"""

import json, os, sys, re, gzip
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
GUIDELINES_DIR = os.environ.get(
    "GUIDELINES_DIR",
    "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines/guidelines"
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines/scripts/analysis_tools/output"
)

# Sources to classify
LEETCODE_PATTERNS = {
    "array", "backtracking", "bfs", "binary-search", "bit-manipulation",
    "brainteaser", "combinatorics", "concurrency", "counting", "database",
    "design-pattern", "dfs", "divide-and-conquer", "dynamic-programming",
    "enumeration", "fenwick-tree", "graph-algorithms", "greedy",
    "hash-based-lookup", "heap", "interactive", "line-sweep", "linked-list",
    "machine-learning", "mathematical-reasoning", "matrix-grid", "microservices",
    "number-theory", "ordered-set", "prefix-sum", "queue", "recursion",
    "rolling-hash", "segment-tree", "shell", "simulation", "sliding-window",
    "sorting", "stack", "string-processing", "tree-traversal", "trie",
    "two-pointers", "uncategorized", "union-find", "api-design"
}


def classify_source(dirname):
    """Classify which data source a guideline subdirectory belongs to."""
    if dirname == "cre-repos":
        return "cre-repos"
    elif dirname == "pattern_profiles":
        return "pattern-profiles"
    elif dirname == "patterns":
        return "patterns-mixed"  # contains codewars + textbooks subdirs
    elif dirname.lower() in LEETCODE_PATTERNS:
        return "leetcode"
    else:
        return "unknown"


def classify_patterns_subsource(filepath):
    """Within patterns/, classify whether codewars or textbooks."""
    if "/patterns/codewars/" in filepath:
        return "codewars"
    elif "/patterns/textbooks/" in filepath:
        return "textbooks"
    return "patterns-other"


def deep_inspect_value(val, depth=0, max_depth=4):
    """Inspect a value deeply to determine its type and shape."""
    if depth > max_depth:
        return {"type": "truncated", "approx": str(val)[:100]}

    if val is None:
        return {"type": "null"}
    elif isinstance(val, bool):
        return {"type": "bool"}
    elif isinstance(val, int):
        return {"type": "int", "sample": val if abs(val) < 1e9 else f"{val:.2e}"}
    elif isinstance(val, float):
        return {"type": "float", "sample": round(val, 4)}
    elif isinstance(val, str):
        length = len(val)
        return {"type": "string", "length": length, "sample": val[:120]}
    elif isinstance(val, list):
        item_types = Counter()
        for item in val[:50]:
            insp = deep_inspect_value(item, depth + 1, max_depth)
            item_types[insp.get("type", "?")] += 1
        return {
            "type": "list",
            "length": len(val),
            "item_types": dict(item_types.most_common(5)),
            "sample_item": deep_inspect_value(val[0], depth + 1, max_depth) if val else None
        }
    elif isinstance(val, dict):
        subkeys = list(val.keys())[:30]
        key_types = {}
        for k in subkeys:
            key_types[k] = deep_inspect_value(val[k], depth + 1, max_depth)
        return {
            "type": "dict",
            "key_count": len(val),
            "keys_sample": subkeys[:20],
            "key_types": key_types
        }
    else:
        return {"type": type(val).__name__}


def analyze_structure(val, prefix=""):
    """Flatten a nested JSON structure into field paths with types."""
    fields = {}
    if val is None:
        return {prefix or "$": {"type": "null"}}
    if isinstance(val, dict):
        for k, v in val.items():
            field_path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                fields.update(analyze_structure(v, field_path))
            elif isinstance(v, list):
                fields[field_path] = {
                    "type": "list",
                    "length": len(v),
                    "item_shape": analyze_structure(v[0], f"{field_path}[]") if v else {"type": "empty_list"}
                }
            else:
                fields[field_path] = {"type": type(v).__name__, "sample": repr(v)[:100]}
    elif isinstance(val, list):
        fields[prefix or "$"] = {"type": "list", "length": len(val)}
    else:
        fields[prefix or "$"] = {"type": type(val).__name__}
    return fields


def compute_completeness(fields, file_data):
    """For a set of known fields, compute which are populated."""
    completeness = {}
    for field_path in sorted(fields.keys()):
        parts = field_path.split(".")
        val = file_data
        try:
            for p in parts:
                if "[]" in p:
                    p_clean = p.replace("[]", "")
                    val = val.get(p_clean, None)
                    break  # skip array item inspection
                elif isinstance(val, dict):
                    val = val.get(p, None)
                else:
                    val = None
                    break
            completeness[field_path] = val is not None and val != "" and val != []
        except (KeyError, TypeError, AttributeError):
            completeness[field_path] = False
    return completeness


def main():
    print("=" * 72)
    print("  DEEP GUIDELINE ANALYZER")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Source: {GUIDELINES_DIR}")
    print("=" * 72)

    # ── Phase 1: Walk all files ─────────────────────────────────────────────
    all_files = []
    file_count_by_source = Counter()
    file_count_by_subsource = Counter()

    for root, dirs, files in os.walk(GUIDELINES_DIR):
        rel = os.path.relpath(root, GUIDELINES_DIR)
        source = classify_source(rel.split("/")[0] if "/" in rel else rel)
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            subsource = classify_patterns_subsource(fpath) if source == "patterns-mixed" else source
            all_files.append((fpath, source, subsource))
            file_count_by_source[source] += 1
            file_count_by_subsource[subsource] += 1

    print(f"\n📁 Found {len(all_files)} total JSON files:")
    for src, count in file_count_by_source.most_common():
        print(f"   {src}: {count}")
    print(f"\n   Sub-source breakdown:")
    for src, count in file_count_by_subsource.most_common():
        print(f"   {src}: {count}")

    # ── Phase 2: Extract all unique fields per source ────────────────────────
    schema_by_source = defaultdict(lambda: defaultdict(Counter))
    all_global_fields = set()
    completeness_data = defaultdict(list)
    file_samples = defaultdict(list)
    outlier_files = []

    # Define expected critical fields for LeetCode v2.0 schema
    CRITICAL_FIELDS = [
        "guideline_id", "title", "title_slug", "guideline", "situation",
        "acceptance_rate", "difficulty", "pattern", "topics", "link",
        "reasoning", "complexity", "solution_code"
    ]

    print(f"\n🔍 Phase 2: Deep field extraction...")
    batch_size = 500
    for batch_start in range(0, len(all_files), batch_size):
        batch = all_files[batch_start:batch_start + batch_size]
        for fpath, source, subsource in batch:
            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                outlier_files.append({
                    "file": fpath, "error": f"JSON parse error: {e}",
                    "source": source, "subsource": subsource
                })
                continue

            # Flatten structure
            flat = analyze_structure(data)

            # Register field types per source
            for field_path, info in flat.items():
                ftype = info.get("type", "?")
                schema_by_source[source][field_path][ftype] += 1
                # Also register with subsource granularity
                schema_by_source[subsource][field_path][ftype] += 1
                all_global_fields.add(field_path)

            # Compute completeness of critical fields (dict only)
            if isinstance(data, dict):
                crit_complete = {}
                for cf in CRITICAL_FIELDS:
                    crit_complete[cf] = cf in data and data.get(cf) not in (None, "", [])
                completeness_data[subsource].append(crit_complete)

                # Flag outliers
                if crit_complete.get("guideline_id") is False:
                    outlier_files.append({
                        "file": fpath, "issue": "missing guideline_id",
                        "source": source, "subsource": subsource
                    })
                if crit_complete.get("guideline") is False and crit_complete.get("solution_code") is False:
                    outlier_files.append({
                        "file": fpath, "issue": "missing both guideline and solution_code",
                        "source": source, "subsource": subsource
                    })
            elif isinstance(data, list):
                outlier_files.append({
                    "file": fpath, "issue": "top-level list (not dict)",
                    "source": source, "subsource": subsource
                })

            # Save sample files per source (first 5)
            if len(file_samples[subsource]) < 5:
                if isinstance(data, dict):
                    file_samples[subsource].append({
                        "file": fpath,
                        "guideline_id": data.get("guideline_id", data.get("id", "?")),
                        "title": (data.get("title", "?") or "")[:80],
                        "top_level_keys": list(data.keys())[:20],
                        "has_solution_code": bool(data.get("solution_code")),
                        "has_hints": bool(data.get("hints")),
                        "has_reasoning": bool(data.get("reasoning")),
                        "has_bridges": bool(data.get("bridges")),
                        "has_code_analysis": bool(data.get("code_analysis")),
                    })
                elif isinstance(data, list):
                    file_samples[subsource].append({
                        "file": fpath,
                        "guideline_id": f"list[{len(data)}]",
                        "title": str(data[0])[:80] if data else "empty",
                        "top_level_keys": ["<LIST_TOP_LEVEL>"],
                        "has_solution_code": False,
                        "has_hints": False,
                        "has_reasoning": False,
                        "has_bridges": False,
                        "has_code_analysis": False,
                    })

        if (batch_start + batch_size) % 2000 == 0 or batch_start == 0:
            print(f"   Processed {min(batch_start + batch_size, len(all_files))}/{len(all_files)} files...")

    # ── Phase 3: Compute completeness stats ──────────────────────────────────
    print(f"\n📊 Phase 3: Computing completeness statistics...")
    completeness_stats = {}
    for subsource, records in completeness_data.items():
        if not records:
            continue
        n = len(records)
        stats = {}
        for field in CRITICAL_FIELDS:
            populated = sum(1 for r in records if r.get(field, False))
            stats[field] = {
                "populated": populated,
                "total": n,
                "pct": round(populated / n * 100, 1) if n > 0 else 0
            }
        completeness_stats[subsource] = {
            "total_files": n,
            "field_completeness": stats
        }

    # ── Phase 4: Field type consolidation ────────────────────────────────────
    print(f"\n🔬 Phase 4: Building global field schema...")
    # Build a clean global schema showing dominant type per field per source
    field_schema_map = {}
    for source, fields in sorted(schema_by_source.items()):
        for field_path, type_counts in sorted(fields.items()):
            if field_path not in field_schema_map:
                field_schema_map[field_path] = {}
            dominant_type = type_counts.most_common(1)[0][0]
            total = sum(type_counts.values())
            field_schema_map[field_path][source] = {
                "dominant_type": dominant_type,
                "occurrence_count": total,
                "type_distribution": dict(type_counts.most_common())
            }

    # ── Phase 5: Cross-source field matrix ────────────────────────────────────
    print(f"\n🗺️  Phase 5: Building cross-source field matrix...")
    all_sources = sorted(set(file_count_by_subsource.keys()))
    field_matrix = {}
    for field_path in sorted(all_global_fields):
        row = {}
        for src in all_sources:
            row[src] = field_path in schema_by_source.get(src, {})
        row["present_in"] = sum(1 for v in row.values() if v)
        row["sources"] = [s for s, v in row.items() if v]
        field_matrix[field_path] = row

    # ── Phase 6: Generate identity keys for entity resolution ────────────────
    print(f"\n🔑 Phase 6: Identifying candidate identity keys...")
    identity_keys = []
    for field_path in sorted(all_global_fields):
        sources_with_it = field_matrix[field_path]["sources"]
        if field_matrix[field_path]["present_in"] >= 2:
            identity_keys.append({
                "field": field_path,
                "shared_by": len(sources_with_it),
                "sources": sources_with_it
            })

    # Sort by most shared
    identity_keys.sort(key=lambda x: -x["shared_by"])

    # ── Write outputs ────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    outputs = {
        "field_schema_map.json": field_schema_map,
        "source_field_matrix.json": {
            "all_sources": all_sources,
            "total_unique_fields": len(all_global_fields),
            "field_matrix": field_matrix,
            "identity_keys": identity_keys
        },
        "completeness_by_source.json": completeness_stats,
        "outliers_report.json": {
            "total_outliers": len(outlier_files),
            "outliers": outlier_files[:200],  # cap for readability
            "outlier_summary": Counter(o["issue"] for o in outlier_files)
        },
        "sample_files.json": {k: v for k, v in file_samples.items()}
    }

    for fname, data in outputs.items():
        outpath = os.path.join(OUTPUT_DIR, fname)
        with open(outpath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"   ✅ Wrote: {outpath}")

    # ── Generate summary report ──────────────────────────────────────────────
    print(f"\n📝 Generating summary report...")
    report_lines = []
    report_lines.append("# Deep Guideline Analysis — Summary Report")
    report_lines.append(f"\n**Generated:** {datetime.now().isoformat()}")
    report_lines.append(f"**Source:** `{GUIDELINES_DIR}`")
    report_lines.append(f"**Total files analyzed:** {len(all_files)}")
    report_lines.append("")

    report_lines.append("## 1. File Distribution by Source")
    report_lines.append("| Source | Count |")
    report_lines.append("|--------|-------|")
    for src, count in file_count_by_subsource.most_common():
        report_lines.append(f"| {src} | {count} |")
    report_lines.append("")

    report_lines.append("## 2. Global Field Inventory")
    report_lines.append(f"**Total unique field paths discovered:** {len(all_global_fields)}")
    report_lines.append("")
    report_lines.append("### Top-Level Fields (no nesting)")
    top_level = sorted([f for f in all_global_fields if "." not in f and "[]" not in f])
    for f in top_level:
        sources = [s for s in all_sources if f in schema_by_source.get(s, {})]
        report_lines.append(f"- `{f}` — present in: {', '.join(sources[:4])}{'...' if len(sources) > 4 else ''}")
    report_lines.append("")

    report_lines.append("## 3. Completeness Matrix (% populated)")
    report_lines.append("| Field | leetcode | cre-repos | codewars | textbooks | pattern-profiles |")
    report_lines.append("|-------|----------|-----------|----------|-----------|-----------------|")
    for field in CRITICAL_FIELDS:
        row = f"| `{field}` |"
        for src in ["leetcode", "cre-repos", "codewars", "textbooks", "pattern-profiles"]:
            stats = completeness_stats.get(src, {}).get("field_completeness", {}).get(field, {})
            pct = stats.get("pct", "—")
            row += f" {pct}% |"
        report_lines.append(row)
    report_lines.append("")

    report_lines.append("## 4. Cross-Source Identity Keys (shared by ≥2 sources)")
    report_lines.append("| Field | # Sources | Sources |")
    report_lines.append("|-------|-----------|---------|")
    for ik in identity_keys[:30]:
        report_lines.append(f"| `{ik['field']}` | {ik['shared_by']} | {', '.join(ik['sources'][:3])}{'...' if len(ik['sources']) > 3 else ''} |")
    report_lines.append("")

    report_lines.append("## 5. Data Quality Issues")
    report_lines.append(f"**Total files with issues:** {len(outlier_files)}")
    oc = Counter(o["issue"] for o in outlier_files)
    for issue, count in oc.most_common():
        report_lines.append(f"- {issue}: **{count}** files")
    report_lines.append("")

    report_lines.append("## 6. Enrichment Potential — Key Gaps")
    for src in ["cre-repos", "codewars", "textbooks"]:
        stats = completeness_stats.get(src, {}).get("field_completeness", {})
        if stats:
            gaps = [(f, s["pct"]) for f, s in stats.items() if s["pct"] < 30]
            if gaps:
                report_lines.append(f"### {src} (low completeness fields)")
                for f, pct in sorted(gaps, key=lambda x: x[1]):
                    report_lines.append(f"- `{f}`: **{pct}%** populated")
                report_lines.append("")

    report_lines.append("## 7. Next Steps")
    report_lines.append("1. Review `field_schema_map.json` for all discovered fields and their types")
    report_lines.append("2. Review `source_field_matrix.json` for cross-source field overlap")
    report_lines.append("3. Use `identity_keys` to plan entity resolution between LeetCode ↔ CRE ↔ Codewars")
    report_lines.append("4. Focus enrichment on fields with <30% completeness in target sources")
    report_lines.append("5. Run `analyze_leetcode_csvs.py` for CSV-to-JSON field mapping")
    report_lines.append("6. Run `analyze_stackexchange_jsonl.py` for StackExchange Q&A enrichment mapping")

    outpath = os.path.join(OUTPUT_DIR, "summary_report.md")
    with open(outpath, "w") as f:
        f.write("\n".join(report_lines))
    print(f"   ✅ Wrote: {outpath}")

    print(f"\n{'=' * 72}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files: {len(all_files)} analyzed")
    print(f"  Unique fields: {len(all_global_fields)}")
    print(f"  Identity keys: {len(identity_keys)} shared across sources")
    print(f"  Outliers: {len(outlier_files)}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
