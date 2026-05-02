#!/usr/bin/env python3
"""
Phase 3a: Enrich Existing LeetCode Guidelines with HF Dataset Content
======================================================================
Deterministic enrichment — no LLM involvement.

Adds to each existing guideline:
  - solution_code_python, solution_code_java, solution_code_cpp (actual code)
  - solution_description (detailed solution walkthrough text)
  - hints (array of hint strings from LeetCode)
  - description (full problem description in HTML)

Matches by source_problem_id (LeetCode frontendQuestionId).
"""

import json
import os
import glob
import sys

# Paths
HF_JSON = "/Users/kevintoles/POC/textbooks/Books/LeetCode Datasets/leetcode_problems.json"
GUIDELINES_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines/"

# Load HF dataset
print("Loading HF dataset...")
with open(HF_JSON) as f:
    hf_data = json.load(f)

# Index by frontendQuestionId
hf_by_id = {}
for d in hf_data:
    hf_by_id[d["frontendQuestionId"]] = d

print(f"  {len(hf_data)} problems loaded from HF dataset")

# Scan all guideline files
guideline_files = sorted(glob.glob(os.path.join(GUIDELINES_DIR, "guideline_*.json")))

stats = {
    "matched": 0,
    "enriched_code": 0,
    "enriched_description": 0,
    "enriched_hints": 0,
    "enriched_problem_desc": 0,
    "skipped_no_match": 0,
    "errors": 0,
}

print(f"  {len(guideline_files)} guideline files found")

for fpath in guideline_files:
    try:
        with open(fpath) as f:
            guideline = json.load(f)

        sid = str(guideline["source_problem_id"])
        if sid not in hf_by_id:
            stats["skipped_no_match"] += 1
            continue

        hf_entry = hf_by_id[sid]
        stats["matched"] += 1
        changed = False

        # 1. Solution code (Python, Java, C++)
        code_py = hf_entry.get("solution_code_python") or ""
        code_java = hf_entry.get("solution_code_java") or ""
        code_cpp = hf_entry.get("solution_code_cpp") or ""

        if code_py or code_java or code_cpp:
            guideline["solution_code"] = {
                "python": code_py,
                "java": code_java,
                "cpp": code_cpp,
            }
            guideline["has_solution_code"] = True
            stats["enriched_code"] += 1
            changed = True

        # 2. Solution description (detailed walkthrough)
        solution_desc = hf_entry.get("solution")
        if solution_desc:
            guideline["solution_description"] = solution_desc
            stats["enriched_description"] += 1
            changed = True

        # 3. Hints
        hints = hf_entry.get("hints")
        if hints and len(hints) > 0:
            guideline["hints"] = hints
            guideline["has_hints"] = True
            stats["enriched_hints"] += 1
            changed = True

        # 4. Full problem description (HTML)
        desc = hf_entry.get("description")
        if desc:
            guideline["description"] = desc
            stats["enriched_problem_desc"] += 1
            changed = True

        # 5. Acceptance rate
        ac_rate = hf_entry.get("acceptance_rate")
        if ac_rate is not None:
            guideline["acceptance_rate"] = ac_rate
            changed = True

        # 6. Stats (total submissions, accepted)
        stats_field = hf_entry.get("stats")
        if stats_field:
            guideline["stats"] = stats_field
            changed = True

        # 7. Likes/dislikes
        likes = hf_entry.get("likes")
        dislikes = hf_entry.get("dislikes")
        if likes is not None:
            guideline["likes"] = likes
            guideline["dislikes"] = dislikes
            changed = True

        # Write back if changed
        if changed:
            with open(fpath, "w") as f:
                json.dump(guideline, f, indent=2)

    except Exception as e:
        print(f"  ERROR processing {os.path.basename(fpath)}: {e}")
        stats["errors"] += 1

# Summary
print()
print("=" * 60)
print("Enrichment Summary")
print("=" * 60)
print(f"  Guidelines matched by ID:     {stats['matched']}")
print(f"  Enriched with solution code:  {stats['enriched_code']}")
print(f"  Enriched with description:    {stats['enriched_description']}")
print(f"  Enriched with hints:          {stats['enriched_hints']}")
print(f"  Enriched with problem desc:   {stats['enriched_problem_desc']}")
print(f"  Skipped (no match):           {stats['skipped_no_match']}")
print(f"  Errors:                       {stats['errors']}")
print()
print("Done. All existing guidelines enriched with HF dataset content.")
