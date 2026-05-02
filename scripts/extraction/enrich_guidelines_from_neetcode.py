#!/usr/bin/env python3
"""
Phase 2c: Enrich Guidelines with NeetCode Curation Data
========================================================
Deterministic enrichment — no LLM involvement.

Adds to each existing guideline:
  - neetcode_pattern: NeetCode roadmap category (e.g., "Arrays & Hashing")
  - neetcode_difficulty: NeetCode-assigned difficulty
  - in_neetcode_150: bool — whether problem appears in NeetCode 150
  - in_blind_75: bool — whether problem appears in Blind 75
  - neetcode_video: YouTube video ID for NeetCode solution walkthrough
  - Optionally: replace solution_code with NeetCode's curated implementation

Matches by problem ID extracted from the NeetCode 'code' field prefix.
"""

import json
import os
import re
import glob
import sys

# Paths
NEETCODE_JSON = "/tmp/neetcode-check/.problemSiteData.json"
NEETCODE_PYTHON_DIR = "/tmp/neetcode-check/python/"
NEETCODE_JAVA_DIR = "/tmp/neetcode-check/java/"
NEETCODE_CPP_DIR = "/tmp/neetcode-check/cpp/"
GUIDELINES_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines/"
OUTPUT_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines/"

# Config: whether to replace solution code with NeetCode's curated version
REPLACE_SOLUTION_CODE = True  # Set to False to keep HF solution code, only add metadata

# Load NeetCode site data
print("Loading NeetCode site data...")
with open(NEETCODE_JSON) as f:
    site_data = json.load(f)
print(f"  {len(site_data)} curated problems loaded")

# Index by problem ID (extracted from 'code' field)
neetcode_by_id = {}
for d in site_data:
    code_prefix = d.get("code", "")
    m = re.match(r"(\d+)", str(code_prefix))
    if m:
        pid = str(int(m.group(1)))
        neetcode_by_id[pid] = d

# Build file lookup for each language
def build_file_lookup(directory, extension):
    """Build a dict mapping problem ID prefix -> filename."""
    lookup = {}
    if not os.path.isdir(directory):
        return lookup
    for fname in os.listdir(directory):
        m = re.match(r"(\d+)", fname)
        if m:
            pid = str(int(m.group(1)))
            lookup[pid] = os.path.join(directory, fname)
    return lookup

python_files = build_file_lookup(NEETCODE_PYTHON_DIR, ".py")
java_files = build_file_lookup(NEETCODE_JAVA_DIR, ".java")
cpp_files = build_file_lookup(NEETCODE_CPP_DIR, ".cpp")

print(f"  Python files indexed: {len(python_files)}")
print(f"  Java files indexed:   {len(java_files)}")
print(f"  C++ files indexed:    {len(cpp_files)}")

# Scan all guideline files
guideline_files = sorted(glob.glob(os.path.join(GUIDELINES_DIR, "guideline_*.json")))
print(f"  {len(guideline_files)} guideline files found")

stats = {
    "matched": 0,
    "enriched_metadata": 0,
    "enriched_code_python": 0,
    "enriched_code_java": 0,
    "enriched_code_cpp": 0,
    "skipped_no_match": 0,
    "errors": 0,
}

for fpath in guideline_files:
    try:
        with open(fpath) as f:
            guideline = json.load(f)

        sid = str(guideline["source_problem_id"])
        if sid not in neetcode_by_id:
            stats["skipped_no_match"] += 1
            continue

        nc_entry = neetcode_by_id[sid]
        stats["matched"] += 1
        changed = False

        # 1. NeetCode metadata
        nc_pattern = nc_entry.get("pattern", "")
        if nc_pattern:
            guideline["neetcode_pattern"] = nc_pattern
            changed = True

        nc_difficulty = nc_entry.get("difficulty", "")
        if nc_difficulty:
            guideline["neetcode_difficulty"] = nc_difficulty
            changed = True

        guideline["in_neetcode_150"] = nc_entry.get("neetcode150", False)
        guideline["in_blind_75"] = nc_entry.get("blind75", False)
        changed = True

        nc_video = nc_entry.get("video", "")
        if nc_video:
            guideline["neetcode_video"] = nc_video
            guideline["neetcode_video_url"] = f"https://www.youtube.com/watch?v={nc_video}"
            changed = True

        stats["enriched_metadata"] += 1

        # 2. NeetCode solution code (curated, higher quality)
        if REPLACE_SOLUTION_CODE:
            new_code = {}

            if sid in python_files:
                with open(python_files[sid]) as pf:
                    new_code["python"] = pf.read()
                stats["enriched_code_python"] += 1

            if sid in java_files:
                with open(java_files[sid]) as jf:
                    new_code["java"] = jf.read()
                stats["enriched_code_java"] += 1

            if sid in cpp_files:
                with open(cpp_files[sid]) as cf:
                    new_code["cpp"] = cf.read()
                stats["enriched_code_cpp"] += 1

            if new_code:
                guideline["solution_code"] = new_code
                guideline["has_solution_code"] = True
                guideline["solution_code_source"] = "NeetCode (curated)"
                changed = True

        # Write back if changed
        if changed:
            with open(fpath, "w") as f:
                json.dump(guideline, f, indent=2)

    except Exception as e:
        print(f"  ERROR processing {os.path.basename(fpath)}: {e}")
        stats["errors"] += 1

# Build pattern summary for output
pattern_counts = {}
for d in site_data:
    pat = d.get("pattern", "Unknown")
    pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

# Write NeetCode pattern taxonomy file
neetcode_taxonomy = {
    "source": "NeetCode Roadmap",
    "source_url": "https://neetcode.io/roadmap",
    "total_problems": len(site_data),
    "patterns": []
}
for pat, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
    neetcode_taxonomy["patterns"].append({
        "pattern_name": pat,
        "problem_count": count,
        "neetcode_150_count": sum(1 for d in site_data if d.get("pattern") == pat and d.get("neetcode150")),
        "blind_75_count": sum(1 for d in site_data if d.get("pattern") == pat and d.get("blind75")),
    })

taxonomy_path = os.path.join(GUIDELINES_DIR, "_neetcode_taxonomy.json")
with open(taxonomy_path, "w") as f:
    json.dump(neetcode_taxonomy, f, indent=2)

# Summary
print()
print("=" * 60)
print("NeetCode Enrichment Summary")
print("=" * 60)
print(f"  Guidelines matched by ID:       {stats['matched']}")
print(f"  Enriched with metadata:         {stats['enriched_metadata']}")
print(f"  Enriched with Python code:      {stats['enriched_code_python']}")
print(f"  Enriched with Java code:        {stats['enriched_code_java']}")
print(f"  Enriched with C++ code:         {stats['enriched_code_cpp']}")
print(f"  Skipped (no NeetCode match):    {stats['skipped_no_match']}")
print(f"  Errors:                         {stats['errors']}")
print()
print(f"  NeetCode taxonomy written to: {taxonomy_path}")
print(f"  Pattern categories: {len(pattern_counts)}")
print()
print("Done. Guidelines enriched with NeetCode curation data.")
