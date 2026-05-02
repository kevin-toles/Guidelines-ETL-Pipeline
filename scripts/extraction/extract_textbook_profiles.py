#!/usr/bin/env python3
"""
Phase 2e: Textbook Pattern Profile Extractor
==============================================
Extracts pattern profiles from textbook PDF JSONs (DS&A, ML, Microservices, API Design).

Each textbook chapter is mapped to a pattern profile matching the Grokking profile format.
Non-algorithmic textbooks (ML, Microservices, API Design) are extracted as domain-level
profiles that can be linked to guidelines via enrichment.

Output:
    /Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines/patterns/textbooks/
        *_profile_NNNNN.json

Usage:
    python3 extract_textbook_profiles.py
"""

import json
import os
import re
from typing import Any

# ── Configuration ──────────────────────────────────────────────────
JSON_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode Datasets/Additional Texts etc./json_output"
OUTPUT_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines/patterns/textbooks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Pattern mapping from DS&A chapter keywords ────────────────────
DSA_TOPIC_MAP: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"backtrack|recurs", re.I),                   "backtracking",           "Backtracking"),
    (re.compile(r"linked.list|node|pointer", re.I),           "linked-list",            "Linked List"),
    (re.compile(r"stack|push|pop|lifo", re.I),                "stack",                  "Stack"),
    (re.compile(r"queue|deque|fifo|priority", re.I),          "queue",                  "Queue"),
    (re.compile(r"tree|bst|binary.*tree|traversal", re.I),    "tree-traversal",         "Tree Traversal"),
    (re.compile(r"heap|priority.*queue", re.I),               "heap",                   "Heap / Priority Queue"),
    (re.compile(r"disjoint|union.find|dsu|set.*adt", re.I),  "union-find",             "Union-Find / DSU"),
    (re.compile(r"graph|bfs|dfs|shortest|topological", re.I), "graph-algorithms",       "Graph Algorithms"),
    (re.compile(r"selection.*algorithm|median|quickselect", re.I), "quickselect",        "Quickselect"),
    (re.compile(r"hash.*table|hash.*map|hash.*set|hashing|collision", re.I), "hash-based-lookup", "Hash-based Lookup"),
    (re.compile(r"string|pattern.*match|kmp|trie|suffix", re.I), "string-processing",   "String Processing"),
    (re.compile(r"trie|prefix.*tree", re.I),                  "trie",                   "Trie / Prefix Tree"),
    (re.compile(r"greedy", re.I),                             "greedy",                 "Greedy"),
    (re.compile(r"dynamic.*program|dp|memoiz", re.I),         "dynamic-programming",    "Dynamic Programming"),
    (re.compile(r"sort|order|merge|quick.*sort", re.I),       "sorting",                "Sorting"),
    (re.compile(r"binary.*search|bisect", re.I),              "binary-search",          "Binary Search"),
    (re.compile(r"bit|manipul|XOR|shift", re.I),              "bit-manipulation",       "Bit Manipulation"),
    (re.compile(r"divide.*conquer", re.I),                    "divide-and-conquer",     "Divide and Conquer"),
    (re.compile(r"sliding.*window", re.I),                    "sliding-window",         "Sliding Window"),
    (re.compile(r"two.*pointer", re.I),                       "two-pointers",           "Two Pointers"),
    (re.compile(r"math|geometric|trig", re.I),                "mathematical-reasoning", "Mathematical Reasoning"),
]


def extract_chapter_title(text: str, chapter_number: int) -> str:
    """Extract a clean chapter title from OCR-garbled text."""
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip()) > 5]
    if not lines:
        return f"Chapter {chapter_number}"

    # Skip copyright, publication info, author lines
    skip_patterns = [
        r"careermonk|publications|all rights reserved|copyright|data structure|algorithmic thinking",
        r"by\s+[A-Z][a-z]+\s+[A-Z][a-z]+",
        r"^\d+\s*$",
    ]
    title = lines[0]
    for i, line in enumerate(lines):
        if i < len(lines) - 1 and any(kw in line.lower() for kw in
            ["backtrack", "linked list", "stack", "queue", "tree", "heap",
             "graph", "hashing", "string", "greedy", "dynamic",
             "sorting", "search", "bit", "divide", "sliding", "pointer"]):
            title = line
            break

    # Clean OCR artifacts
    title = title.replace('|', '').replace('~', '').replace('{', '').replace('}', '')
    title = re.sub(r'[^a-zA-Z0-9\s/\-]', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title if title else f"Chapter {chapter_number}"


def detect_dsa_pattern(title: str, content: str) -> tuple[str, str]:
    """Detect which algorithmic pattern a DS&A chapter covers."""
    combined = f"{title} {content[:1000]}"
    for pattern, pid, pname in DSA_TOPIC_MAP:
        if pattern.search(combined):
            return pid, pname
    return "miscellaneous", "Miscellaneous"


def extract_dsa_profiles(d: dict) -> list[dict]:
    """Extract pattern profiles from DS&A textbook chapters."""
    profiles = []
    chapters = d.get("chapters", [])
    metadata = d.get("metadata", {})

    book_name = metadata.get("title", "Data Structure and Algorithmic Thinking with Python")
    source_url = metadata.get("source", metadata.get("url", ""))

    for i, ch in enumerate(chapters):
        content = ch.get("content", "")
        if not content:
            continue

        title = extract_chapter_title(content, ch["number"])
        pid, pname = detect_dsa_pattern(title, content)

        # Extract key concepts from first part of chapter
        intro_section = content[:2000]
        # Find key implementation hints
        impl_lines = []
        for line in content.split('\n'):
            line_s = line.strip()
            if any(kw in line_s.lower() for kw in
                   ["algorithm", "complexity", "o(", "time", "space",
                    "implementation", "pseudocode", "procedure"]):
                if len(line_s) > 20:
                    impl_lines.append(line_s)

        implementation = "\n".join(impl_lines[:10]) if impl_lines else intro_section[:500]

        profile = {
            "profile_id": f"dsa_{pid}_{i+1:02d}",
            "source": "Data Structure and Algorithmic Thinking with Python",
            "source_url": source_url,
            "pattern_id": pid,
            "pattern_name": pname,
            "chapter_number": ch["number"],
            "chapter_title": title,
            "when_to_use": intro_section[:500],
            "implementation": implementation[:1000],
            "complexity": {"time": "See chapter content", "space": "See chapter content"},
            "variants": [],
            "related_problems": [],
            "raw_introduction": content[:3000],
            "has_content": True,
            "page_count": ch.get("page_count", 0),
        }
        profiles.append(profile)

    return profiles


def extract_general_textbook_profiles(d: dict, book_id: str, book_title: str) -> list[dict]:
    """Extract profiles from non-algorithmic textbooks at chapter level."""
    profiles = []
    chapters = d.get("chapters", [])
    metadata = d.get("metadata", {})

    source_url = metadata.get("source", metadata.get("url", ""))

    for i, ch in enumerate(chapters):
        content = ch.get("content", "")
        if not content:
            continue

        title = extract_chapter_title(content, ch["number"])
        # Use first 500 chars as intro
        intro = content[:500]

        profile = {
            "profile_id": f"{book_id}_ch{i+1:02d}",
            "source": book_title,
            "source_url": source_url,
            "pattern_id": book_id,
            "pattern_name": title,
            "chapter_number": ch["number"],
            "chapter_title": title,
            "when_to_use": intro,
            "implementation": "",
            "complexity": {"time": "N/A", "space": "N/A"},
            "variants": [],
            "related_problems": [],
            "raw_introduction": content[:2000],
            "has_content": True,
            "page_count": ch.get("page_count", 0),
        }
        profiles.append(profile)

    return profiles


def build_profile_index(profiles: list[dict]) -> dict:
    """Build an index of all extracted profiles by pattern_id."""
    index: dict[str, list[dict]] = {}
    for p in profiles:
        pid = p["pattern_id"]
        if pid not in index:
            index[pid] = []
        index[pid].append({
            "profile_id": p["profile_id"],
            "source": p["source"],
            "chapter_title": p.get("chapter_title", ""),
            "chapter_number": p.get("chapter_number", 0),
            "page_count": p.get("page_count", 0),
        })
    return index


def main():
    books = [
        ("432327595-Data-Structure-and-Algorithmic-Thinking-with-Python-Data-Structure-and-Algorithmic-Puzzles-PDFDrive-com-pdf.json",
         "dsa", "Data Structure and Algorithmic Thinking with Python",
         extract_dsa_profiles),
        ("419796115-Full-Course-of-Machine-Learning.json",
         "ml", "Full Course of Machine Learning",
         lambda d: extract_general_textbook_profiles(d, "machine-learning", "Full Course of Machine Learning")),
        ("544166051-Strategic-Monoliths-and-Microservices-Driving-Innovation-Using-Purposeful-Architecture-18-11-2021.json",
         "microservices", "Strategic Monoliths and Microservices",
         lambda d: extract_general_textbook_profiles(d, "microservices", "Strategic Monoliths and Microservices")),
        ("713495692-AW-patterns-for-API-design.json",
         "api-design", "Patterns for API Design",
         lambda d: extract_general_textbook_profiles(d, "api-design", "Patterns for API Design")),
    ]

    total_profiles = 0
    all_profiles = []

    for filename, book_id, book_title, extractor in books:
        path = os.path.join(JSON_DIR, filename)
        if not os.path.exists(path):
            print(f"SKIP {filename} — not found")
            continue

        with open(path) as f:
            data = json.load(f)

        profiles = extractor(data)
        print(f"  {book_title}: {len(profiles)} profiles")

        for i, p in enumerate(profiles):
            out_path = os.path.join(OUTPUT_DIR, f"{book_id}_profile_{i+1:05d}.json")
            with open(out_path, "w") as f:
                json.dump(p, f, indent=2)
            total_profiles += 1
        all_profiles.extend(profiles)

    # Write profile index
    index = build_profile_index(all_profiles)
    index_path = os.path.join(OUTPUT_DIR, "_profile_index.json")
    with open(index_path, "w") as f:
        json.dump({
            "total_profiles": total_profiles,
            "sources": [b[2] for b in books],
            "by_pattern_id": {k: v for k, v in sorted(index.items())},
        }, f, indent=2)
    print(f"\nTotal profiles written: {total_profiles}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
