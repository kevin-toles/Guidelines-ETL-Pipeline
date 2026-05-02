#!/usr/bin/env python3
"""
Pattern Groups Generator
=========================
Reads all guideline_*.json files from a directory and produces
_pattern_groups.json — guidelines grouped by pattern_id.

Output:
    { "patternd_id": [guideline_json, ...], ... }

Usage:
    python3 generate_pattern_groups.py \\
        --input /path/to/guidelines/ \\
        --output /path/to/guidelines/_pattern_groups.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any


def main():
    parser = argparse.ArgumentParser(
        description="Generate _pattern_groups.json from guideline files"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Directory containing guideline_*.json files",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for _pattern_groups.json",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        sys.exit(1)

    print(f"Scanning {args.input} for guideline files...")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    skipped = 0

    for fname in sorted(os.listdir(args.input)):
        if not fname.startswith("guideline_") or not fname.endswith(".json"):
            continue
        if fname.startswith("_"):  # skip _pattern_index.json, _neetcode_taxonomy.json
            continue

        fpath = os.path.join(args.input, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                guideline = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Skipping {fname}: {e}")
            skipped += 1
            continue

        pattern_id = guideline.get("pattern", {}).get("pattern_id", "uncategorized")
        groups[pattern_id].append(guideline)
        count += 1

    # Convert defaultdict to regular dict for JSON serialization
    result: dict[str, Any] = {
        "_meta": {
            "total_guidelines": count,
            "total_patterns": len(groups),
            "skipped_files": skipped,
            "source_directory": os.path.abspath(args.input),
        },
        "groups": dict(sorted(groups.items())),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    file_size_kb = os.path.getsize(args.output) / 1024

    print(f"  Guidelines processed:  {count}")
    print(f"  Skipped (errors):      {skipped}")
    print(f"  Patterns found:       {len(groups)}")
    print(f"  Output size:          {file_size_kb:.1f} KB")
    print(f"  Written to:           {args.output}")

    # Top patterns
    pattern_counts = [
        (pid, len(guidelines)) for pid, guidelines in groups.items()
    ]
    pattern_counts.sort(key=lambda x: -x[1])
    print(f"\n  Top 10 patterns:")
    for pid, cnt in pattern_counts[:10]:
        print(f"    {pid:30s}: {cnt:5d}")


if __name__ == "__main__":
    main()
