#!/usr/bin/env python3
"""
LeetCode Dataset Extractor — Phase 1: Raw Extraction
=====================================================
Extracts each LeetCode dataset into per-problem JSONs organized by source.
Preserves ALL original fields without transformation, enrichment, or merging.

Output structure:
    /Users/kevintoles/POC/textbooks/Books/LeetCode JSON/
        Leetcode.csv/
            problem_1.json
            problem_2.json
            ...
        leetcode_problems.csv/
            problem_1.json
            problem_2.json
            ...
        leetcode_questions.csv/
            problem_1.json
            problem_2.json
            ...
        leetcode-problem-set/
            problem_1.json
            problem_2.json
            ...

Usage:
    python3 extract_leetcode_json.py
"""

import csv
import json
import os
import re
import sys

# ── Configuration ──────────────────────────────────────────────────
BASE_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode Datasets"
OUTPUT_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON"

# Source definitions: (name, filename, id_field, csv_kwargs)
# Each source becomes a subfolder under OUTPUT_DIR.
SOURCES = [
    {
        "name": "Leetcode.csv",
        "file": "Leetcode.csv",
        "id_field": "ID",
    },
    {
        "name": "leetcode_problems.csv",
        "file": "leetcode_problems.csv",
        "id_field": "frontendQuestionId",
    },
    {
        "name": "leetcode_questions.csv",
        "file": "leetcode_questions.csv",
        "id_field": "Question ID",
    },
    {
        "name": "leetcode-problem-set",
        "file": "leetcode-problem-set/data.csv",
        "id_field": "frontendQuestionId",
    },
]


def strip_html(text: str) -> str:
    """Strip HTML tags and decode common HTML entities."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_field(value: str) -> str:
    """Clean a field value by stripping whitespace."""
    if value is None:
        return ""
    return value.strip()


def try_json_parse(value: str):
    """Try to parse a value as JSON (for fields like 'topics', 'similar_questions')."""
    if not value or value == "[]":
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def extract_source(source: dict) -> dict:
    """
    Extract all problems from a single source CSV into per-problem JSONs.
    Returns stats dict with counts.
    """
    name = source["name"]
    filepath = os.path.join(BASE_DIR, source["file"])
    id_field = source["id_field"]
    subdir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(subdir, exist_ok=True)

    stats = {"total_rows": 0, "extracted": 0, "skipped_no_id": 0, "skipped_dup": 0}

    seen_ids = set()

    with open(filepath, "r", encoding="utf-8") as f:
        # Read first 10KB to sniff dialect (handle weird quoting)
        sample = f.read(10240)
        f.seek(0)

        # Detect delimiter
        dialect = csv.Sniffer().sniff(sample)
        reader = csv.DictReader(f, delimiter=dialect.delimiter)

        for row in reader:
            stats["total_rows"] += 1

            # Clean all fields
            cleaned = {k.strip(): clean_field(v) for k, v in row.items()}

            # Extract ID
            raw_id = cleaned.get(id_field, "")
            if not raw_id:
                stats["skipped_no_id"] += 1
                continue

            try:
                problem_id = int(raw_id)
            except ValueError:
                stats["skipped_no_id"] += 1
                continue

            if problem_id in seen_ids:
                stats["skipped_dup"] += 1
                continue
            seen_ids.add(problem_id)

            # Parse list-like fields if they look like JSON arrays
            for list_field in ["topics", "similar_questions", "topicTags",
                               "Topics", "Similar Questions", "Similar Questions Text",
                               "related_topics", "similar_questions", "companies"]:
                if list_field in cleaned:
                    cleaned[list_field] = try_json_parse(cleaned[list_field])

            # Build output: include source provenance
            output = {
                "_source": name,
                "_extracted_at": "2026-04-28",
                "_problem_id": problem_id,
                "data": cleaned,
            }

            outfile = os.path.join(subdir, f"problem_{problem_id}.json")
            with open(outfile, "w", encoding="utf-8") as outf:
                json.dump(output, outf, indent=2, ensure_ascii=False)

            stats["extracted"] += 1

    return stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("LeetCode Dataset Extractor — Phase 1: Raw Extraction")
    print("=" * 60)

    grand_total = {"extracted": 0, "skipped": 0}

    for source in SOURCES:
        print(f"\n── {source['name']} ──")
        stats = extract_source(source)
        print(f"  Total rows:     {stats['total_rows']}")
        print(f"  Extracted:      {stats['extracted']}")
        print(f"  Skipped (no ID): {stats['skipped_no_id']}")
        print(f"  Skipped (dup):   {stats['skipped_dup']}")
        grand_total["extracted"] += stats["extracted"]
        grand_total["skipped"] += stats["skipped_no_id"] + stats["skipped_dup"]

    print(f"\n{'=' * 60}")
    print(f"Total extracted: {grand_total['extracted']}")
    print(f"Total skipped:   {grand_total['skipped']}")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")

    # List output structure
    print("\nOutput structure:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        level = root.replace(OUTPUT_DIR, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/ ({len(files)} files)")


if __name__ == "__main__":
    main()
