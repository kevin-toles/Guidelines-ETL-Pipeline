#!/usr/bin/env python3
"""
Phase 3b: Extract Grokking Coding Interview Patterns as KB-ready Guidelines
=============================================================================
Deterministic extraction — no LLM involvement.

Reads the dipjul/Grokking-the-Coding-Interview-Patterns repo markdown files and
extracts pattern introductions as standalone guideline/profile documents.

Output: pattern_profiles/ directory with one JSON per coding pattern.
Each profile contains:
  - pattern_id, pattern_name
  - when_to_use (situations/scenarios where this pattern applies)
  - implementation (how to implement the pattern)
  - complexity (time/space)
  - variants (different implementations: one-pointer-at-each-end, different-paces, etc.)
  - related_problems (list of linked LeetCode problems with difficulty)
  - common_mistakes / pitfalls (extracted from the text)
  - raw_content (the full original markdown)
"""

import json
import os
import re
import glob

# Paths
GROKKING_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode Datasets/Grokking-the-Coding-Interview-Patterns-for-Coding-Questions"
OUTPUT_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines/pattern_profiles"

# Pattern directory mapping
PATTERN_DIRS = {
    "1.-pattern-sliding-window": "sliding-window",
    "2.-pattern-two-pointers": "two-pointers",
    "9.-pattern-two-heaps": "two-heaps",
    "11.-pattern-modified-binary-search": "modified-binary-search",
    "13.-pattern-top-k-elements": "top-k-elements",
    "16.-pattern-topological-sort-graph": "topological-sort",
    "untitled": "tree-bfs",  # The "untitled" dir is actually Tree BFS
    "binary-search": "binary-search-pattern",
}

# Map directory to readable pattern name
PATTERN_NAMES = {
    "1.-pattern-sliding-window": "Sliding Window",
    "2.-pattern-two-pointers": "Two Pointers",
    "9.-pattern-two-heaps": "Two Heaps",
    "11.-pattern-modified-binary-search": "Modified Binary Search",
    "13.-pattern-top-k-elements": "Top K Elements",
    "16.-pattern-topological-sort-graph": "Topological Sort (Graph)",
    "untitled": "Tree Breadth First Search",
    "binary-search": "Binary Search (in-depth)",
}

# Additional content directories
ADDITIONAL_DIRS = {
    "revision": "coding-patterns-cheatsheet",
    "test-your-knowledge": "practice-problems",
}


def extract_leetcode_links(markdown_text):
    """Extract LeetCode problem links from markdown text."""
    links = []
    # Pattern: [text](https://leetcode.com/problems/...)
    leet_pattern = re.compile(r'\[([^\]]+)\]\(https?://leetcode\.com/problems/([^/\)]+)/?[^\)]*\)')
    for match in leet_pattern.finditer(markdown_text):
        links.append({
            "title": match.group(1),
            "slug": match.group(2),
            "url": f"https://leetcode.com/problems/{match.group(2)}/"
        })
    return links


def extract_content_sections(markdown_text):
    """Extract key sections from pattern introduction markdown."""
    sections = {}

    # Try to find "when to use", "implementation", "approach" sections by heading
    current_heading = "overview"
    current_content = []

    for line in markdown_text.split("\n"):
        heading_match = re.match(r'^#{2,4}\s+(.+)$', line)
        if heading_match:
            if current_content:
                sections[current_heading] = "\n".join(current_content).strip()
            current_heading = heading_match.group(1).strip().lower()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections[current_heading] = "\n".join(current_content).strip()

    return sections


def extract_complexity(markdown_text):
    """Extract time/space complexity mentions."""
    time_match = re.search(r'[Tt]ime\s+complexity[^.]*O\(([^)]+)\)', markdown_text)
    space_match = re.search(r'[Ss]pace\s+complexity[^.]*O\(([^)]+)\)', markdown_text)

    return {
        "time": f"O({time_match.group(1)})" if time_match else None,
        "space": f"O({space_match.group(1)})" if space_match else None,
    }


def is_substantive(content):
    """Check if markdown file has substantive content (>100 chars)."""
    text = content.strip()
    # Remove markdown formatting for length check
    plain = re.sub(r'[#*_`\[\]()>|]', '', text)
    return len(plain) > 100


def extract_problem_difficulty(filename):
    """Extract difficulty from filename (easy/medium/hard)."""
    m = re.search(r'\((easy|medium|hard)\)', filename.lower())
    return m.group(1) if m else "unknown"


def build_pattern_profile(dir_name, pattern_id, pattern_name):
    """Build a complete pattern profile from the Grokking markdown files."""
    dir_path = os.path.join(GROKKING_DIR, dir_name)
    if not os.path.isdir(dir_path):
        return None

    profile = {
        "profile_id": f"grokking_{pattern_id}",
        "source": "Grokking the Coding Interview",
        "source_url": f"https://github.com/dipjul/Grokking-the-Coding-Interview-Patterns-for-Coding-Questions/tree/master/{dir_name}",
        "pattern_id": pattern_id,
        "pattern_name": pattern_name,
        "when_to_use": "",
        "implementation": "",
        "complexity": {},
        "variants": [],
        "related_problems": [],
        "raw_introduction": "",
        "has_content": False,
    }

    # Collect all markdown files with content
    md_files = sorted(glob.glob(os.path.join(dir_path, "*.md")))

    # Find the introduction file first
    intro_content = ""
    for fpath in md_files:
        fname = os.path.basename(fpath)
        if "intro" in fname.lower() or "readme" in fname.lower() or "introduction" in fname.lower():
            with open(fpath) as f:
                content = f.read()
            if is_substantive(content):
                intro_content = content
                profile["raw_introduction"] = content
                profile["has_content"] = True

                # Extract sections
                sections = extract_content_sections(content)
                profile["when_to_use"] = sections.get("overview", "")
                profile["implementation"] = sections.get("implementation", "")

                # Extract complexity
                profile["complexity"] = extract_complexity(content)

                # Extract leetcode links from intro
                profile["related_problems"].extend(extract_leetcode_links(content))
                break

    # Collect problem-specific files
    for fpath in md_files:
        fname = os.path.basename(fpath)
        # Skip intro/readme files
        if "intro" in fname.lower() or "readme" in fname.lower() or "introduction" in fname.lower():
            continue

        with open(fpath) as f:
            content = f.read()

        if not is_substantive(content):
            continue

        # Extract LeetCode links from this problem file
        leet_links = extract_leetcode_links(content)
        difficulty = extract_problem_difficulty(fname)
        for link in leet_links:
            link["difficulty"] = difficulty
            link["file"] = fname

        profile["related_problems"].extend(leet_links)

    # Deduplicate related problems
    seen_slugs = set()
    unique_problems = []
    for p in profile["related_problems"]:
        if p["slug"] not in seen_slugs:
            seen_slugs.add(p["slug"])
            unique_problems.append(p)
    profile["related_problems"] = unique_problems

    return profile


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    profiles = []

    # Process each pattern directory
    for dir_name, pattern_id in PATTERN_DIRS.items():
        pattern_name = PATTERN_NAMES.get(dir_name, pattern_id.replace("-", " ").title())
        profile = build_pattern_profile(dir_name, pattern_id, pattern_name)
        if profile:
            profiles.append(profile)
            fpath = os.path.join(OUTPUT_DIR, f"{pattern_id}.json")
            with open(fpath, "w") as f:
                json.dump(profile, f, indent=2)
            status = "HAS CONTENT" if profile["has_content"] else "STUB (no content)"
            print(f"  {pattern_id:30s} → {len(profile['related_problems']):3d} problems  [{status}]")

    # Also process revision/cheatsheet
    revision_dir = os.path.join(GROKKING_DIR, "revision")
    if os.path.isdir(revision_dir):
        for fpath in sorted(glob.glob(os.path.join(revision_dir, "*.md"))):
            with open(fpath) as f:
                content = f.read()
            if is_substantive(content):
                fname = os.path.basename(fpath).replace(".md", "")
                profile = {
                    "profile_id": f"grokking_revision_{fname}",
                    "source": "Grokking the Coding Interview - Revision",
                    "pattern_id": "revision-cheatsheet",
                    "pattern_name": "Coding Patterns Revision Cheatsheet",
                    "when_to_use": content[:500],
                    "implementation": "",
                    "complexity": {},
                    "variants": [],
                    "related_problems": extract_leetcode_links(content),
                    "raw_introduction": content,
                    "has_content": True,
                }
                fout = os.path.join(OUTPUT_DIR, f"revision_cheatsheet.json")
                with open(fout, "w") as f:
                    json.dump(profile, f, indent=2)
                print(f"  revision_cheatsheet               → CREATED")

    print()
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Total profiles: {len(profiles)}")
    print()

    # Also output a master index
    index = {}
    for p in profiles:
        index[p["profile_id"]] = {
            "pattern_id": p["pattern_id"],
            "pattern_name": p["pattern_name"],
            "has_content": p["has_content"],
            "problem_count": len(p["related_problems"]),
        }
    index_path = os.path.join(OUTPUT_DIR, "_profile_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Index: {index_path}")


if __name__ == "__main__":
    main()
