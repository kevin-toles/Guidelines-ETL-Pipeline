#!/usr/bin/env python3
"""
StackExchange JSONL Deep Analyzer — v1.0
Deep analysis of StackExchange Q&A JSONL data across all tiers (primary, hacks, supplemental).

Analyzes:
  1. Record structure and fields per file
  2. Topic/Site distribution
  3. Quality signals (score, view_count, answer_count)
  4. Enrichment potential: which guideline fields can be enriched
  5. Tag-to-pattern mapping (StackExchange tags → LeetCode patterns)
  6. Content quality assessment (code blocks, structured answers)

Outputs:
  - stackexchange_inventory.json      — per-file field inventory and stats
  - stackexchange_quality.json        — quality score distributions
  - stackexchange_tag_analysis.json   — tag frequency and pattern mapping
  - enrichment_mapping.json           — StackExchange fields → guideline fields
  - sample_records.json               — diverse samples for manual review

Usage:
  python3 analyze_stackexchange_jsonl.py [--data-dir PATH] [--output-dir PATH]
"""

import json, os, sys, re, gzip
from collections import defaultdict, Counter
from datetime import datetime

DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/Volumes/USB321FD/Guidelines ETL Data/ai-platform-output"
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines/scripts/analysis_tools/output"
)

# Known tiers
TIERS = ["primary", "hacks", "supplemental"]

# Guideline fields that SE data could potentially enrich
ENRICHABLE_FIELDS = {
    "hints": "StackExchange answers often contain step-by-step hints",
    "solution_code": "Code blocks in answers",
    "alternatives": "Multiple answer approaches → alternative solutions",
    "constraints": "Questions often mention constraints",
    "similar_questions": "Related/linked questions",
    "situation.summary": "Question body contains problem description",
    "reasoning": "Answer explanations provide reasoning",
    "complexity.time": "Answers often discuss time complexity",
    "complexity.space": "Answers often discuss space complexity",
    "guideline": "Accepted answer can serve as guideline text",
}

# Tag-to-pattern mapping hints
TAG_PATTERN_HINTS = {
    "array": "array",
    "hash-table": "hash-based-lookup", "hash-map": "hash-based-lookup",
    "string": "string-processing", "strings": "string-processing",
    "dynamic-programming": "dynamic-programming", "dp": "dynamic-programming",
    "depth-first-search": "dfs", "dfs": "dfs",
    "breadth-first-search": "bfs", "bfs": "bfs",
    "binary-search": "binary-search",
    "two-pointers": "two-pointers",
    "sliding-window": "sliding-window",
    "stack": "stack", "monotonic-stack": "stack",
    "queue": "queue",
    "heap": "heap", "priority-queue": "heap",
    "greedy": "greedy",
    "sorting": "sorting",
    "backtracking": "backtracking",
    "tree": "tree-traversal", "binary-tree": "tree-traversal",
    "graph": "graph-algorithms",
    "linked-list": "linked-list",
    "bit-manipulation": "bit-manipulation",
    "math": "mathematical-reasoning",
    "recursion": "recursion",
    "union-find": "union-find", "disjoint-set": "union-find",
    "trie": "trie", "prefix-tree": "trie",
    "segment-tree": "segment-tree",
    "matrix": "matrix-grid",
    "prefix-sum": "prefix-sum",
    "simulation": "simulation",
    "design": "design-pattern", "ood": "design-pattern",
    "database": "database", "sql": "database",
    "concurrency": "concurrency", "multithreading": "concurrency",
    "divide-and-conquer": "divide-and-conquer",
    "combinatorics": "combinatorics",
    "number-theory": "number-theory",
    "ordered-set": "ordered-set",
    "fenwick-tree": "fenwick-tree", "binary-indexed-tree": "fenwick-tree",
    "rolling-hash": "rolling-hash",
    "interactive": "interactive",
    "counting": "counting",
    "enumeration": "enumeration",
    "line-sweep": "line-sweep",
    "api": "api-design", "api-design": "api-design",
    "microservice": "microservices", "microservices": "microservices",
    "machine-learning": "machine-learning", "ml": "machine-learning",
}


def extract_code_blocks(text):
    """Extract code blocks from text body."""
    if not text:
        return []
    # Match ```code``` blocks
    blocks = re.findall(r'```(?:\w*\n)?(.*?)```', text, re.DOTALL)
    # Also match indented code blocks (4 spaces)
    return [b.strip() for b in blocks if len(b.strip()) > 20]


def analyze_quality(record):
    """Assess quality of a StackExchange record."""
    score = record.get("score", 0)
    view_count = record.get("view_count", 0)
    answer_count = record.get("answer_count", 0)
    has_accepted = bool(record.get("accepted_answer_id"))
    body = record.get("body", "")
    tags = record.get("tags", [])

    code_blocks = extract_code_blocks(body)
    body_len = len(body) if body else 0

    return {
        "score": score,
        "view_count": view_count,
        "answer_count": answer_count,
        "has_accepted_answer": has_accepted,
        "code_blocks_count": len(code_blocks),
        "body_length": body_len,
        "tag_count": len(tags) if isinstance(tags, list) else 0,
        "quality_tier": (
            "gold" if score >= 10 and len(code_blocks) >= 1 and has_accepted
            else "silver" if score >= 5 and body_len > 500
            else "bronze" if score >= 0
            else "low"
        ),
    }


def infer_patterns_from_tags(tags):
    """Map StackExchange tags to guideline patterns."""
    if not tags or not isinstance(tags, list):
        return []
    patterns = []
    for tag in tags:
        tag_lower = tag.lower() if isinstance(tag, str) else ""
        if tag_lower in TAG_PATTERN_HINTS:
            patterns.append(TAG_PATTERN_HINTS[tag_lower])
    return list(set(patterns))


def analyze_jsonl_file(filepath, tier):
    """Analyze a single JSONL file."""
    result = {
        "file": os.path.basename(filepath),
        "tier": tier,
        "path": filepath,
        "exists": os.path.exists(filepath),
        "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2) if os.path.exists(filepath) else 0,
    }

    if not result["exists"]:
        result["error"] = "FILE NOT FOUND"
        return result

    # Sample up to 5000 records
    records = []
    fields_seen = set()
    scores = []
    view_counts = []
    tags_counter = Counter()
    quality_tiers = Counter()
    total_code_blocks = 0
    pattern_hits = Counter()

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if len(records) >= 5000:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    records.append(rec)
                    fields_seen.update(rec.keys())

                    qa = analyze_quality(rec)
                    scores.append(qa["score"])
                    view_counts.append(qa["view_count"])
                    quality_tiers[qa["quality_tier"]] += 1
                    total_code_blocks += qa["code_blocks_count"]

                    tags = rec.get("tags", [])
                    if isinstance(tags, list):
                        for t in tags:
                            if isinstance(t, str):
                                tags_counter[t.lower()] += 1

                    patterns = infer_patterns_from_tags(tags)
                    for p in patterns:
                        pattern_hits[p] += 1

                except json.JSONDecodeError:
                    pass
    except Exception as e:
        result["error"] = str(e)
        return result

    result["record_count"] = len(records)
    result["fields"] = sorted(fields_seen)
    result["field_count"] = len(fields_seen)

    # Score statistics
    if scores:
        scores_sorted = sorted(scores)
        result["score_stats"] = {
            "min": min(scores),
            "max": max(scores),
            "median": scores_sorted[len(scores_sorted) // 2],
            "avg": round(sum(scores) / len(scores), 1),
            "p90": scores_sorted[int(len(scores_sorted) * 0.9)],
            "p95": scores_sorted[int(len(scores_sorted) * 0.95)],
            "high_quality_pct": round(sum(1 for s in scores if s >= 10) / len(scores) * 100, 1),
        }

    result["quality_distribution"] = dict(quality_tiers)
    result["total_code_blocks"] = total_code_blocks
    result["top_tags"] = tags_counter.most_common(50)
    result["top_pattern_hits"] = pattern_hits.most_common(30)

    # Enrichment potential assessment
    enrichable = {}
    for field, desc in ENRICHABLE_FIELDS.items():
        # Check if related fields exist in the SE data
        related_se_fields = []
        for se_field in fields_seen:
            se_lower = se_field.lower()
            field_key = field.split(".")[-1].lower()
            if field_key in se_lower or se_lower in field_key:
                related_se_fields.append(se_field)
        if related_se_fields or field == "solution_code" and total_code_blocks > 0:
            enrichable[field] = {
                "description": desc,
                "related_se_fields": related_se_fields,
                "confidence": (
                    "high" if len(related_se_fields) >= 1 and quality_tiers.get("gold", 0) > 10
                    else "medium" if related_se_fields
                    else "low"
                ),
            }
    result["enrichment_potential"] = enrichable

    # Sample diverse records
    samples = {"gold": None, "silver": None, "bronze": None, "code_heavy": None}
    for rec in records:
        qa = analyze_quality(rec)
        tier = qa["quality_tier"]
        if tier in samples and samples[tier] is None:
            samples[tier] = {
                "title": rec.get("title", "")[:100],
                "score": rec.get("score"),
                "tags": rec.get("tags", [])[:5],
                "body_preview": (rec.get("body", "") or "")[:300],
                "code_blocks": len(extract_code_blocks(rec.get("body", ""))),
            }
        if qa["code_blocks_count"] >= 3 and samples["code_heavy"] is None:
            samples["code_heavy"] = {
                "title": rec.get("title", "")[:100],
                "score": rec.get("score"),
                "code_blocks": qa["code_blocks_count"],
                "body_preview": (rec.get("body", "") or "")[:200],
            }
        if all(v is not None for v in samples.values()):
            break

    result["samples"] = {k: v for k, v in samples.items() if v is not None}

    return result


def main():
    print("=" * 72)
    print("  STACKEXCHANGE JSONL DEEP ANALYZER")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Source: {DATA_DIR}")
    print("=" * 72)

    if not os.path.exists(DATA_DIR):
        print(f"\n  ❌ Data directory not found: {DATA_DIR}")
        print(f"  Checking alternative locations...")
        alt = "/Volumes/USB321FD"
        if os.path.exists(alt):
            subdirs = [d for d in os.listdir(alt) if os.path.isdir(os.path.join(alt, d))]
            print(f"  Available: {subdirs}")
        return

    # ── Phase 1: Find all JSONL files ──────────────────────────────────────
    all_files = []
    for tier in TIERS:
        tier_dir = os.path.join(DATA_DIR, tier)
        if os.path.exists(tier_dir):
            for fname in sorted(os.listdir(tier_dir)):
                if fname.endswith(".jsonl"):
                    all_files.append((os.path.join(tier_dir, fname), tier))

    print(f"\n📁 Found {len(all_files)} JSONL files across {len(set(t for _, t in all_files))} tiers")

    # ── Phase 2: Sample key files (not all 600 — too slow) ─────────────────
    # Strategy: analyze 2-3 files per tier to get representative sample
    results = {}
    tier_sample_count = {"primary": 3, "hacks": 2, "supplemental": 2}

    for tier in TIERS:
        tier_files = [(p, t) for p, t in all_files if t == tier]
        sample_count = min(tier_sample_count.get(tier, 2), len(tier_files))
        step = max(1, len(tier_files) // sample_count)
        sampled = tier_files[::step][:sample_count]

        for filepath, t in sampled:
            result = analyze_jsonl_file(filepath, t)
            results[os.path.basename(filepath)] = result
            err = result.get("error", "")
            if err:
                print(f"  ❌ [{tier}] {os.path.basename(filepath)}: ERROR - {err}")
            else:
                print(f"  ✅ [{tier}] {os.path.basename(filepath)}: "
                      f"{result.get('record_count', 0)} recs, "
                      f"{result.get('field_count', 0)} fields, "
                      f"score avg={result.get('score_stats', {}).get('avg', '?')}, "
                      f"gold={result.get('quality_distribution', {}).get('gold', 0)}")

    # ── Phase 3: Aggregate findings ────────────────────────────────────────
    all_fields = set()
    all_tags = Counter()
    all_pattern_hits = Counter()
    total_records = 0
    total_gold = 0
    total_silver = 0
    total_code_blocks = 0
    aggregate_enrichment = defaultdict(lambda: {"confidence_votes": Counter(), "tiers": set()})

    for r in results.values():
        if r.get("error"):
            continue
        all_fields.update(r.get("fields", []))
        total_records += r.get("record_count", 0)
        total_code_blocks += r.get("total_code_blocks", 0)
        qd = r.get("quality_distribution", {})
        total_gold += qd.get("gold", 0)
        total_silver += qd.get("silver", 0)

        for tag, count in r.get("top_tags", []):
            all_tags[tag] += count
        for pat, count in r.get("top_pattern_hits", []):
            all_pattern_hits[pat] += count

        for field, info in r.get("enrichment_potential", {}).items():
            aggregate_enrichment[field]["confidence_votes"][info["confidence"]] += 1
            aggregate_enrichment[field]["tiers"].add(r.get("tier", "?"))

    # ── Phase 4: Infer SE record structure ──────────────────────────────────
    print(f"\n📊 AGGREGATE FINDINGS:")
    print(f"  Total records sampled: {total_records:,}")
    print(f"  Unique fields in SE data: {len(all_fields)}")
    print(f"  Total code blocks found: {total_code_blocks:,}")
    print(f"  Gold quality: {total_gold} ({round(total_gold/total_records*100,1)}%)")
    print(f"  Silver quality: {total_silver} ({round(total_silver/total_records*100,1)}%)")
    print(f"  Unique tags: {len(all_tags)}")
    print(f"  Tag→Pattern mappings: {len(all_pattern_hits)} patterns covered")

    print(f"\n📋 SE Record Fields: {sorted(all_fields)}")

    print(f"\n🔗 Enrichment potential:")
    for field, info in sorted(aggregate_enrichment.items()):
        votes = info["confidence_votes"]
        best = votes.most_common(1)[0][0] if votes else "unknown"
        print(f"  {field}: {best} confidence (tiers: {sorted(info['tiers'])})")

    # ── Phase 5: Write outputs ──────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    outputs = {
        "stackexchange_inventory.json": {
            "source_dir": DATA_DIR,
            "total_files_found": len(all_files),
            "files_analyzed": len(results),
            "total_records_sampled": total_records,
            "unique_fields": sorted(all_fields),
            "total_code_blocks": total_code_blocks,
            "quality_distribution": {
                "gold": total_gold,
                "gold_pct": round(total_gold / total_records * 100, 1) if total_records > 0 else 0,
                "silver": total_silver,
                "silver_pct": round(total_silver / total_records * 100, 1) if total_records > 0 else 0,
            },
            "file_details": results,
        },
        "stackexchange_tag_analysis.json": {
            "total_unique_tags": len(all_tags),
            "top_tags": all_tags.most_common(100),
            "total_pattern_mappings": len(all_pattern_hits),
            "tag_to_pattern_mapping": dict(all_pattern_hits.most_common(50)),
            "mapping_coverage": {
                "patterns_covered": len(all_pattern_hits),
                "total_known_patterns": len(set(TAG_PATTERN_HINTS.values())),
            },
        },
        "enrichment_mapping.json": {
            "enrichable_fields": [
                {
                    "guideline_field": field,
                    "confidence": info["confidence_votes"].most_common(1)[0][0] if info["confidence_votes"] else "unknown",
                    "se_field_count": len(info["tiers"]),
                    "tiers_available": sorted(info["tiers"]),
                }
                for field, info in aggregate_enrichment.items()
            ],
        },
    }

    for fname, data in outputs.items():
        outpath = os.path.join(OUTPUT_DIR, fname)
        with open(outpath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  ✅ Wrote: {outpath}")

    print(f"\n{'=' * 72}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
