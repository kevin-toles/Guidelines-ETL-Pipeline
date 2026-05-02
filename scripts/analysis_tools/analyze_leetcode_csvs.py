#!/usr/bin/env python3
"""
LeetCode CSV Deep Analyzer — v1.0
Deep analysis of ALL LeetCode CSV files, mapping their fields to the v2.0 guideline schema.

Covers:
  1. Leetcode.csv — main problem dataset
  2. Easy/Medium/Hard CSVs — difficulty-segmented
  3. leetcode_questions.csv — question bank
  4. leetcode_dataset - lc.csv — Kaggle-style dataset
  5. leetcode-problem-set/data.csv — problem set data
  6. template_guidelines.csv — template for guideline generation

Outputs:
  - csv_inventory.json          — detailed field inventory per CSV
  - csv_to_schema_mapping.json  — how CSV fields map to guideline schema fields
  - missing_fields_report.json  — what's in the schema but NOT in any CSV
  - enrichment_potential.json   — what CSVs can enrich which guideline fields

Usage:
  python3 analyze_leetcode_csvs.py [--scripts-dir PATH] [--output-dir PATH]
"""

import csv, json, os, sys
from collections import defaultdict, Counter
from datetime import datetime

SCRIPTS_DIR = os.environ.get(
    "SCRIPTS_DIR",
    "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines/scripts"
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines/scripts/analysis_tools/output"
)

# Known CSV files and their expected locations
CSV_FILES = {
    "Leetcode.csv": "Main LeetCode problem dataset (source for most guideline generation)",
    "leetcode_problems.csv": "LeetCode problems data",
    "leetcode_problem.csv": "LeetCode problem data (singular)",
    "leetcode_questions.csv": "LeetCode questions bank",
    "leetcode_dataset - lc.csv": "Kaggle-style LeetCode dataset",
    "Easy_leetcode_problem (1).csv": "Easy difficulty problems",
    "Medium_leetcode_problem (1).csv": "Medium difficulty problems",
    "Hard_leetcode_problem (1).csv": "Hard difficulty problems",
    "template_guidelines.csv": "Template for guideline generation",
    "leetcode-problem-set/data.csv": "Problem set data (subdirectory)",
}

# Guideline schema fields (v2.0) — from deep analysis
GUIDELINE_V2_FIELDS = {
    "guideline_id": "Unique identifier (e.g., guideline_02404)",
    "schema_version": "Schema version string (2.0)",
    "source_problem_id": "Numeric LeetCode problem ID",
    "source_dataset": "Source dataset name (e.g., Leetcode.csv)",
    "title": "Problem title",
    "title_slug": "URL-friendly title slug",
    "link": "LeetCode problem URL",
    "situation.summary": "Situation summary text",
    "situation.difficulty": "Easy/Medium/Hard",
    "situation.topics": "Array of topic strings",
    "situation.tags.*": "Boolean flags for problem characteristics",
    "guideline": "The guideline text (approach description)",
    "reasoning": "Reasoning behind the guideline",
    "complexity.time": "Time complexity (e.g., O(n))",
    "complexity.space": "Space complexity (e.g., O(n))",
    "complexity.code_hints": "Code hints array",
    "pattern.pattern_id": "Pattern identifier slug",
    "pattern.pattern_name": "Human-readable pattern name",
    "pattern.category": "Pattern category",
    "code_analysis.*": "Code detection flags (uses_dict, uses_set, etc.)",
    "constraints": "Array of constraints",
    "alternatives": "Array of alternative approaches",
    "has_solution_code": "Boolean: has solution code",
    "has_hints": "Boolean: has hints",
    "hints": "Array of hint strings",
    "description": "Full HTML problem description from LeetCode",
    "acceptance_rate": "Acceptance rate percentage",
    "likes": "Number of likes",
    "dislikes": "Number of dislikes",
    "stats": "Submission statistics dict",
    "similar_questions": "Array of similar question references",
    "bridges.*": "Cross-reference bridges (code_repos, code_chunks, etc.)",
    "metadata.*": "Metadata (premium, category, stats)",
    "solution_code": "Actual solution code (missing in most files!)",
}

# Known CSV → JSON field mappings (built from extract_guidelines.py analysis)
FIELD_SYNONYMS = {
    "id": "source_problem_id",
    "problem_id": "source_problem_id",
    "question_id": "source_problem_id",
    "frontend_question_id": "source_problem_id",
    "name": "title",
    "question_title": "title",
    "problem_name": "title",
    "title_slug": "title_slug",
    "url": "link",
    "leetcode_url": "link",
    "problem_url": "link",
    "difficulty": "situation.difficulty",
    "difficulty_level": "situation.difficulty",
    "topic_tags": "situation.topics",
    "tags": "situation.topics",
    "related_topics": "situation.topics",
    "solution": "solution_code",
    "solution_code": "solution_code",
    "code_solution": "solution_code",
    "answer": "solution_code",
    "hints": "hints",
    "hint": "hints",
    "acceptance_rate": "acceptance_rate",
    "acceptance": "acceptance_rate",
    "ac_rate": "acceptance_rate",
    "likes": "likes",
    "dislikes": "dislikes",
    "description": "description",
    "content": "description",
    "problem_description": "description",
    "question": "description",
    "is_premium": "metadata.premium",
    "premium": "metadata.premium",
    "paid_only": "metadata.premium",
    "frequency": "metadata.frequency",
    "category": "metadata.category",
    "similar_questions": "similar_questions",
    "related_questions": "similar_questions",
    "companies": "metadata.companies",
    "company_tags": "metadata.companies",
    "total_accepted": "stats.totalAcceptedRaw",
    "total_submission": "stats.totalSubmissionRaw",
    "total_submissions": "stats.totalSubmissionRaw",
}


def analyze_csv(filepath, fname):
    """Deep analysis of a single CSV file."""
    result = {
        "file": fname,
        "path": filepath,
        "exists": os.path.exists(filepath),
        "size_bytes": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
        "description": CSV_FILES.get(fname, "Unknown"),
    }
    if not result["exists"]:
        result["error"] = "FILE NOT FOUND"
        return result

    # Read first 500 rows for analysis
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows = []
            for i, row in enumerate(reader):
                if i >= 500:
                    break
                rows.append(row)
    except Exception as e:
        result["error"] = str(e)
        result["headers"] = []
        return result

    result["headers"] = headers
    result["row_count_sample"] = len(rows)

    # Analyze each column
    column_analysis = {}
    for col in headers:
        vals = [r.get(col, "") for r in rows]
        non_empty = [v for v in vals if v and v.strip()]
        col_info = {
            "non_empty_count": len(non_empty),
            "empty_count": len(vals) - len(non_empty),
            "fill_rate": round(len(non_empty) / len(vals) * 100, 1) if vals else 0,
        }

        # Determine data type
        if non_empty:
            sample = non_empty[0]
            # Check types
            if sample.replace(".", "").replace("-", "").isdigit():
                col_info["inferred_type"] = "numeric"
                nums = []
                for v in non_empty[:100]:
                    try:
                        nums.append(float(v))
                    except ValueError:
                        pass
                if nums:
                    col_info["min"] = min(nums)
                    col_info["max"] = max(nums)
                    col_info["avg"] = round(sum(nums) / len(nums), 2)
            elif sample.startswith("http"):
                col_info["inferred_type"] = "url"
            elif sample.startswith("[") or sample.startswith("{"):
                col_info["inferred_type"] = "structured_string"
            elif len(sample) > 200:
                col_info["inferred_type"] = "long_text"
                col_info["avg_length"] = round(sum(len(v) for v in non_empty[:50]) / min(50, len(non_empty)), 0)
            else:
                col_info["inferred_type"] = "string"

        # Unique values for categorical columns
        unique = set(v.strip() for v in non_empty[:100] if v.strip())
        col_info["unique_sample_count"] = min(len(unique), 100)
        if len(unique) <= 30:
            col_info["unique_values"] = sorted(unique)

        # Map to guideline schema
        col_lower = col.lower().strip()
        mapped = FIELD_SYNONYMS.get(col_lower) or FIELD_SYNONYMS.get(col)
        if not mapped:
            # Try fuzzy match
            for csv_field, json_field in FIELD_SYNONYMS.items():
                if csv_field in col_lower or col_lower in csv_field:
                    mapped = json_field
                    break
        col_info["schema_mapping"] = mapped or "UNMAPPED"

        column_analysis[col] = col_info

    result["column_analysis"] = column_analysis

    # Cross-reference: which guideline fields are covered by this CSV
    covered_fields = set()
    for col, info in column_analysis.items():
        if info["schema_mapping"] and info["schema_mapping"] != "UNMAPPED":
            covered_fields.add(info["schema_mapping"])
    result["covered_schema_fields"] = sorted(covered_fields)

    # Cross-reference: which are missing
    all_schema_fields = set(k for k in GUIDELINE_V2_FIELDS.keys() if "*" not in k)
    result["missing_schema_fields"] = sorted(all_schema_fields - covered_fields)

    return result


def main():
    print("=" * 72)
    print("  LEETCODE CSV DEEP ANALYZER")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 72)

    results = {}
    all_covered = defaultdict(set)
    all_missing = set(GUIDELINE_V2_FIELDS.keys())

    for fname, desc in CSV_FILES.items():
        # Find the actual file
        found = None
        for search_dir in [SCRIPTS_DIR, os.path.join(SCRIPTS_DIR, "leetcode-problem-set")]:
            candidate = os.path.join(search_dir, fname)
            if os.path.exists(candidate):
                found = candidate
                break

        if not found:
            # Search recursively
            for root, dirs, files in os.walk(SCRIPTS_DIR):
                if fname in files:
                    found = os.path.join(root, fname)
                    break

        if not found:
            results[fname] = {"file": fname, "error": "NOT FOUND ANYWHERE"}
            print(f"  ❌ {fname}: NOT FOUND")
            continue

        result = analyze_csv(found, fname)
        results[fname] = result

        print(f"  {'✅' if not result.get('error') else '❌'} {fname}: "
              f"{len(result.get('headers', []))} cols, "
              f"{result.get('row_count_sample', 0)} rows sampled, "
              f"{len(result.get('covered_schema_fields', []))} mapped fields")

        for f in result.get("covered_schema_fields", []):
            all_covered[f].add(fname)

    # ── Build comprehensive mapping matrix ──────────────────────────────────
    print(f"\n🗺️  Building CSV-to-Schema mapping matrix...")
    schema_to_csvs = {}
    for field, description in GUIDELINE_V2_FIELDS.items():
        if "*" in field:
            continue
        csvs = sorted(all_covered.get(field, set()))
        schema_to_csvs[field] = {
            "description": description,
            "available_in_csvs": csvs,
            "csv_count": len(csvs),
        }

    # Determine enrichment priority:
    # Fields with 0 CSVs = cannot be enriched from CSV
    # Fields with 1+ CSVs = enrichment possible
    priority_0 = [f for f, v in schema_to_csvs.items() if v["csv_count"] == 0]
    priority_1 = [f for f, v in schema_to_csvs.items() if v["csv_count"] >= 1]

    # ── Generate findings ───────────────────────────────────────────────────
    print(f"\n📊 KEY FINDINGS:")
    print(f"  Fields available in CSVs: {len(priority_1)}/{len(GUIDELINE_V2_FIELDS)}")
    print(f"  Fields NOT in any CSV:    {len(priority_0)}")
    print(f"\n  Fields NOT in any CSV (require other enrichment sources):")
    for f in sorted(priority_0):
        print(f"    - {f}")

    print(f"\n  Fields available in CSVs:")
    for f in sorted(priority_1, key=lambda x: -schema_to_csvs[x]["csv_count"]):
        csvs = schema_to_csvs[f]["csv_count"]
        print(f"    - {f} ({csvs} CSV{'s' if csvs > 1 else ''})")

    # ── Write outputs ───────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prune large unique_values from column analysis
    for r in results.values():
        if "column_analysis" in r:
            for col, info in r["column_analysis"].items():
                if "unique_values" in info and len(info.get("unique_values", [])) > 50:
                    info["unique_values"] = info["unique_values"][:50] + ["...truncated"]

    outputs = {
        "csv_inventory.json": results,
        "csv_to_schema_mapping.json": {
            "total_guideline_fields": len(GUIDELINE_V2_FIELDS),
            "fields_in_csvs": len(priority_1),
            "fields_not_in_csvs": len(priority_0),
            "fields_not_in_csvs_list": sorted(priority_0),
            "mapping": schema_to_csvs,
        },
    }

    for fname, data in outputs.items():
        outpath = os.path.join(OUTPUT_DIR, fname)
        with open(outpath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  ✅ Wrote: {outpath}")

    print(f"\n{'=' * 72}")
    print(f"  ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
