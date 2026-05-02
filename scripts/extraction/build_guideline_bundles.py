#!/usr/bin/env python3
"""
Phase 2: Guideline Bundle Builder
==================================
Groups LeetCode problems by algorithmic pattern and produces "pattern bundles"
— one JSON per pattern — containing representative problems, solution code,
hints, and descriptions. These bundles are the input for guideline generation.

Output:
    /Users/kevintoles/POC/textbooks/Books/LeetCode JSON/pattern-bundles/
        hash-based-lookup.json
        two-pointers.json
        sliding-window.json
        ... (one per pattern)

Usage:
    python3 build_guideline_bundles.py
"""

import json
import os
import re
import html

# ── Configuration ──────────────────────────────────────────────────
LC_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/Leetcode.csv"
PROB_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/leetcode_problems.csv"
QUEST_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/leetcode_questions.csv"
BUNDLE_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/pattern-bundles"
OUTPUT_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines"

os.makedirs(BUNDLE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Topic → Pattern mapping ──────────────────────────────────────
TOPIC_TO_PATTERN = {
    "Hash Table": ("hash-based-lookup", "Hash-based Lookup"),
    "Prefix Sum": ("prefix-sum", "Prefix Sum"),
    "Counting": ("counting", "Counting"),
    "Two Pointers": ("two-pointers", "Two Pointers"),
    "Sliding Window": ("sliding-window", "Sliding Window"),
    "Binary Search": ("binary-search", "Binary Search"),
    "Depth-First Search": ("dfs", "DFS (Depth-First Search)"),
    "Breadth-First Search": ("bfs", "BFS (Breadth-First Search)"),
    "Backtracking": ("backtracking", "Backtracking"),
    "Divide and Conquer": ("divide-and-conquer", "Divide and Conquer"),
    "Dynamic Programming": ("dynamic-programming", "Dynamic Programming"),
    "Bitmask": ("bitmask-dp", "Bitmask DP"),
    "Sorting": ("sorting", "Sorting"),
    "Ordered Set": ("ordered-set", "Ordered Set"),
    "Heap (Priority Queue)": ("heap-priority-queue", "Heap / Priority Queue"),
    "Stack": ("stack", "Stack"),
    "Monotonic Stack": ("monotonic-stack", "Monotonic Stack"),
    "Queue": ("queue", "Queue"),
    "Tree": ("tree-traversal", "Tree Traversal"),
    "Binary Tree": ("binary-tree", "Binary Tree"),
    "Graph": ("graph-algorithms", "Graph Algorithms"),
    "Union Find": ("union-find-dsu", "Union-Find / DSU"),
    "Trie": ("trie-prefix-tree", "Trie / Prefix Tree"),
    "Topological Sort": ("topological-sort", "Topological Sort"),
    "Linked List": ("linked-list", "Linked List"),
    "Bit Manipulation": ("bit-manipulation", "Bit Manipulation"),
    "Math": ("mathematical-reasoning", "Mathematical Reasoning"),
    "Number Theory": ("number-theory", "Number Theory"),
    "Combinatorics": ("combinatorics", "Combinatorics"),
    "Geometry": ("geometry", "Geometry"),
    "String": ("string-processing", "String Processing"),
    "Matrix": ("matrix-grid", "Matrix / Grid"),
    "Simulation": ("simulation", "Simulation"),
    "Design": ("design-pattern", "Design Pattern"),
}


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_problems() -> tuple:
    """Load all problems from Leetcode.csv and supplement from other sources."""
    problems = {}
    for fname in os.listdir(LC_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(LC_DIR, fname)) as f:
            d = json.load(f)
        p = d["data"]
        pid = int(p["ID"])
        problems[pid] = {
            "id": pid,
            "title": p.get("Title", ""),
            "difficulty": p.get("Difficulty", ""),
            "topics": p.get("Topics", ""),
            "category": p.get("Category", ""),
            "acceptance_rate": p.get("Acceptance Rate (%)", ""),
            "likes": p.get("Likes", "0"),
            "dislikes": p.get("Dislikes", "0"),
            "premium": p.get("Premium Only", "False"),
            "similar_questions": p.get("Similar Questions", []),
            "link": p.get("Link", ""),
        }

    # Supplement with solution code
    for fname in os.listdir(PROB_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(PROB_DIR, fname)) as f:
            d = json.load(f)
        pid = d["_problem_id"]
        if pid not in problems:
            continue
        data = d["data"]
        desc_raw = data.get("description", "")
        problems[pid].update({
            "description_html": desc_raw,
            "description": strip_html(desc_raw),
            "solution_code_python": data.get("solution_code_python", ""),
            "solution_code_java": data.get("solution_code_java", ""),
            "solution_code_cpp": data.get("solution_code_cpp", ""),
            "solution_text": data.get("solution", ""),
            "hints_json": data.get("hints", "[]"),
            "stats": data.get("stats", ""),
            "title_slug": data.get("titleSlug", ""),
        })

    # Supplement with hints from leetcode_questions
    for fname in os.listdir(QUEST_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(QUEST_DIR, fname)) as f:
            d = json.load(f)
        pid = d["_problem_id"]
        if pid not in problems:
            continue
        data = d["data"]
        hint = data.get("Hints", "")
        if hint and not problems[pid].get("hints_json"):
            problems[pid]["hints_extra"] = hint
        if not problems[pid].get("description"):
            problems[pid]["description"] = data.get("Question Text", "")

    return problems


def get_topic_list(topics_field) -> list:
    """Normalize topics field to a list of strings."""
    if isinstance(topics_field, list):
        result = []
        for t in topics_field:
            if isinstance(t, dict):
                result.append(t.get("title", str(t)))
            else:
                result.append(str(t).strip())
        return result
    elif isinstance(topics_field, str):
        return [t.strip() for t in topics_field.split(",") if t.strip()]
    return []


def assign_patterns(topic_list: list) -> list:
    """Map topic tags to canonical pattern IDs."""
    patterns = []
    for t in topic_list:
        if t in TOPIC_TO_PATTERN:
            pid, pname = TOPIC_TO_PATTERN[t]
            patterns.append({"pattern_id": pid, "pattern_name": pname})
    # Also infer patterns from similar questions or context
    if not patterns and topic_list:
        slug = topic_list[0].lower().replace(" ", "-")
        patterns.append({"pattern_id": slug, "pattern_name": topic_list[0]})
    if not patterns:
        patterns.append({"pattern_id": "uncategorized", "pattern_name": "Uncategorized"})
    return patterns


def select_representatives(probs: list, max_per_difficulty: int = 3) -> list:
    """Select representative problems for a pattern — prioritize ones with solution code."""
    by_diff = {"Easy": [], "Medium": [], "Hard": []}
    for p in probs:
        d = p.get("difficulty", "Medium")
        if d in by_diff:
            by_diff[d].append(p)

    selected = []
    for diff in ["Easy", "Medium", "Hard"]:
        pool = by_diff.get(diff, [])
        # Sort: those with solution code first, then by likes
        pool.sort(key=lambda p: (
            -(1 if p.get("solution_code_python") else 0),
            -int(p.get("likes", 0))
        ))
        selected.extend(pool[:max_per_difficulty])

    return selected


def main():
    print("=" * 60)
    print("Phase 2: Guideline Bundle Builder")
    print("=" * 60)

    problems = load_problems()
    print(f"\nLoaded {len(problems)} problems")

    # Group by pattern
    pattern_bundles = {}
    pattern_names = {}

    for pid, p in problems.items():
        topic_list = get_topic_list(p.get("topics", []))
        patterns = assign_patterns(topic_list)

        for pat in patterns:
            pid_key = pat["pattern_id"]
            if pid_key not in pattern_bundles:
                pattern_bundles[pid_key] = {
                    "pattern_id": pid_key,
                    "pattern_name": pat["pattern_name"],
                    "problems": [],
                    "derived_sources": set(),
                }
                pattern_names[pid_key] = pat["pattern_name"]
            pattern_bundles[pid_key]["problems"].append(p)
            pattern_bundles[pid_key]["derived_sources"].add(p.get("title_slug", "") or p.get("link", ""))

    # Deduplicate derived_sources
    for pid_key in pattern_bundles:
        pattern_bundles[pid_key]["derived_sources"] = list(
            pattern_bundles[pid_key]["derived_sources"]
        )

    print(f"Identified {len(pattern_bundles)} patterns\n")

    # Build and write pattern bundles
    for pid_key, bundle in sorted(pattern_bundles.items()):
        problems_list = bundle["problems"]
        representatives = select_representatives(problems_list, max_per_difficulty=2)

        pattern_bundle = {
            "_schema_version": "1.0",
            "bundle_type": "pattern_bundle",
            "pattern_id": pid_key,
            "pattern_name": bundle["pattern_name"],
            "total_problems": len(problems_list),
            "difficulty_counts": {
                "Easy": sum(1 for p in problems_list if p["difficulty"] == "Easy"),
                "Medium": sum(1 for p in problems_list if p["difficulty"] == "Medium"),
                "Hard": sum(1 for p in problems_list if p["difficulty"] == "Hard"),
            },
            "has_solution_count": sum(1 for p in problems_list if p.get("solution_code_python")),
            "representative_problems": representatives,
            "all_problem_ids": sorted([p["id"] for p in problems_list]),
            "derived_sources": bundle["derived_sources"],
        }

        outpath = os.path.join(BUNDLE_DIR, f"{pid_key}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(pattern_bundle, f, indent=2, ensure_ascii=False)

        py_count = sum(1 for r in representatives if r.get("solution_code_python"))
        dc = pattern_bundle["difficulty_counts"]
        print(f"  {pid_key:30s} {bundle['pattern_name']:25s} "
              f"({dc['Easy']:>3d}E/{dc['Medium']:>3d}M/{dc['Hard']:>3d}H) "
              f"reps:{len(representatives)} py:{py_count}")

    print(f"\nWrote {len(pattern_bundles)} pattern bundles to {BUNDLE_DIR}/")

    # Also write a summary index
    index = []
    for pid_key in sorted(pattern_bundles.keys()):
        b = pattern_bundles[pid_key]
        index.append({
            "pattern_id": pid_key,
            "pattern_name": b["pattern_name"],
            "total_problems": len(b["problems"]),
            "difficulty_counts": {
                "Easy": sum(1 for p in b["problems"] if p["difficulty"] == "Easy"),
                "Medium": sum(1 for p in b["problems"] if p["difficulty"] == "Medium"),
                "Hard": sum(1 for p in b["problems"] if p["difficulty"] == "Hard"),
            },
            "has_solution_count": sum(1 for p in b["problems"] if p.get("solution_code_python")),
        })

    with open(os.path.join(BUNDLE_DIR, "_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"Wrote index to {BUNDLE_DIR}/_index.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
