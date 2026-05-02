#!/usr/bin/env python3
"""
Sync extracted guidelines and profiles into the coding-guidelines collection.
Organizes by pattern_id directory, updates _pattern_index.json,
and merges into existing pattern directories.

Usage:
    python3 sync_to_collection.py
"""

import json
import os
import shutil
import glob
from collections import Counter, defaultdict

# ── Configuration ──────────────────────────────────────────────────
CODING_GUIDELINES_SRC = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines"
COLLECTION_DIR = "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines"

NEW_SOURCES = {
    "codewars": {
        "src": os.path.join(CODING_GUIDELINES_SRC, "patterns", "codewars"),
        "dst": os.path.join(COLLECTION_DIR, "guidelines"),
        "glob": "*.json",
        "filter": lambda f: not f.startswith("_"),
        "tag": "codewars",
    },
    "textbooks": {
        "src": os.path.join(CODING_GUIDELINES_SRC, "patterns", "textbooks"),
        "dst": os.path.join(COLLECTION_DIR, "guidelines"),
        "glob": "*.json",
        "filter": lambda f: not f.startswith("_"),
        "tag": "textbook",
    },
}


def get_pattern_id(item: dict) -> str:
    """Extract pattern_id from a guideline or profile JSON."""
    if "pattern" in item and isinstance(item["pattern"], dict):
        return item["pattern"].get("pattern_id", "uncategorized")
    if "pattern_id" in item:
        return item.get("pattern_id", "uncategorized")
    if "guideline_id" in item:
        return "uncategorized"
    return "uncategorized"


def sync_files() -> dict[str, int]:
    """Sync all new guideline/profile files into pattern-organized directories."""
    stats = {"codewars": 0, "textbooks": 0}

    for source_name, config in NEW_SOURCES.items():
        src_dir = config["src"]
        dst_base = config["dst"]
        glob_pattern = config["glob"]
        src_files = glob.glob(os.path.join(src_dir, glob_pattern))

        for src_path in src_files:
            basename = os.path.basename(src_path)
            if not config["filter"](basename):
                continue

            with open(src_path) as f:
                item = json.load(f)

            pattern_id = get_pattern_id(item)
            dst_dir = os.path.join(dst_base, pattern_id)
            os.makedirs(dst_dir, exist_ok=True)

            # Determine destination filename
            if "guideline_id" in item:
                gid = item["guideline_id"]
                dst_name = f"{gid}.json"
            else:
                dst_name = basename

            dst_path = os.path.join(dst_dir, dst_name)

            # Avoid collisions with existing files
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(dst_name)
                counter = 1
                while os.path.exists(os.path.join(dst_dir, f"{base}_{counter}{ext}")):
                    counter += 1
                dst_path = os.path.join(dst_dir, f"{base}_{counter}{ext}")

            shutil.copy2(src_path, dst_path)
            stats[source_name] += 1

    return stats


def update_pattern_index():
    """Regenerate _pattern_index.json from all guidelines in the collection."""
    guidelines_dir = os.path.join(COLLECTION_DIR, "guidelines")
    index_path = os.path.join(COLLECTION_DIR, "metadata", "_pattern_index.json")

    # Read existing index
    existing_index = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            existing_index = {e["pattern_id"]: e for e in json.load(f)}

    # Scan all guideline files and count by pattern
    pattern_counts: dict[str, dict[str, Any]] = {}
    pattern_files: dict[str, list[str]] = defaultdict(list)

    for pattern_dir in sorted(os.listdir(guidelines_dir)):
        pdir_path = os.path.join(guidelines_dir, pattern_dir)
        if not os.path.isdir(pdir_path):
            continue

        files = [f for f in os.listdir(pdir_path) if f.endswith(".json")]
        if not files:
            continue

        # Read first file to get pattern info
        first_file = os.path.join(pdir_path, files[0])
        try:
            with open(first_file) as f:
                item = json.load(f)

            if "pattern" in item and isinstance(item["pattern"], dict):
                pid = item["pattern"].get("pattern_id", pattern_dir)
                pname = item["pattern"].get("pattern_name", pattern_dir.replace("-", " ").title())
                category = item["pattern"].get("category", "general")
            elif "pattern_name" in item:
                pid = item.get("pattern_id", pattern_dir)
                pname = item.get("pattern_name", pattern_dir.replace("-", " ").title())
                category = "general"
            else:
                pid = pattern_dir
                pname = pattern_dir.replace("-", " ").title()
                category = "general"

            pattern_counts[pid] = {
                "pattern_id": pid,
                "pattern_name": pname,
                "count": len(files),
                "category": category,
            }
        except (json.JSONDecodeError, KeyError):
            pattern_counts[pattern_dir] = {
                "pattern_id": pattern_dir,
                "pattern_name": pattern_dir.replace("-", " ").title(),
                "count": len(files),
                "category": "general",
            }

    # Merge with existing — keep existing names for existing patterns
    for pid, entry in existing_index.items():
        if pid in pattern_counts:
            pattern_counts[pid]["pattern_name"] = entry.get("pattern_name", pattern_counts[pid]["pattern_name"])
            pattern_counts[pid]["category"] = entry.get("category", pattern_counts[pid]["category"])
        else:
            pattern_counts[pid] = entry  # preserve orphaned entries

    # Write updated index
    sorted_index = sorted(pattern_counts.values(), key=lambda x: -x["count"])
    with open(index_path, "w") as f:
        json.dump(sorted_index, f, indent=2)

    print(f"Updated _pattern_index.json: {len(sorted_index)} patterns, {sum(e['count'] for e in sorted_index)} total files")
    return sorted_index


def main():
    print("=== Syncing to coding-guidelines collection ===\n")

    # Step 1: Sync files
    print("Copying files...")
    stats = sync_files()
    for source, count in stats.items():
        print(f"  {source}: {count} files synced")

    # Step 2: Copy metadata files from source
    print("\nUpdating metadata...")
    src_meta_dir = os.path.join(CODING_GUIDELINES_SRC, "metadata")
    dst_meta_dir = os.path.join(COLLECTION_DIR, "metadata")

    # Copy _neetcode_taxonomy.json if it exists in source and not in dest
    for meta_file in ["_neetcode_taxonomy.json", "_pattern_groups.json"]:
        src_meta = os.path.join(src_meta_dir, meta_file)
        dst_meta = os.path.join(dst_meta_dir, meta_file)
        if os.path.exists(src_meta) and not os.path.exists(dst_meta):
            shutil.copy2(src_meta, dst_meta)
            print(f"  Copied {meta_file} to collection metadata")

    # Sync pattern_profiles directory
    src_profiles = os.path.join(src_meta_dir, "pattern_profiles")
    dst_profiles = os.path.join(dst_meta_dir, "pattern_profiles")
    os.makedirs(dst_profiles, exist_ok=True)
    if os.path.exists(src_profiles):
        for f in os.listdir(src_profiles):
            if f.endswith(".json"):
                src_f = os.path.join(src_profiles, f)
                dst_f = os.path.join(dst_profiles, f)
                if not os.path.exists(dst_f):
                    shutil.copy2(src_f, dst_f)

    # Also copy new textbook profiles into pattern_profiles
    textbook_profiles_src = os.path.join(CODING_GUIDELINES_SRC, "patterns", "textbooks")
    if os.path.exists(textbook_profiles_src):
        for f in os.listdir(textbook_profiles_src):
            if f.endswith(".json") and not f.startswith("_"):
                shutil.copy2(os.path.join(textbook_profiles_src, f),
                             os.path.join(dst_profiles, f))
        # Copy profile index
        src_idx = os.path.join(textbook_profiles_src, "_profile_index.json")
        if os.path.exists(src_idx):
            shutil.copy2(src_idx, os.path.join(dst_profiles, "_textbook_profile_index.json"))
        print(f"  Synced textbook profiles to pattern_profiles/")

    # Step 3: Regenerate _pattern_index.json
    print("\nRegenerating pattern index...")
    update_pattern_index()

    # Step 4: Show summary
    print("\n=== Sync Complete ===")
    for pattern_dir in sorted(os.listdir(os.path.join(COLLECTION_DIR, "guidelines"))):
        pdir = os.path.join(COLLECTION_DIR, "guidelines", pattern_dir)
        if os.path.isdir(pdir):
            count = len([f for f in os.listdir(pdir) if f.endswith(".json")])
            print(f"  {pattern_dir:30s}: {count} files")

    total = sum(
        len([f for f in os.listdir(os.path.join(COLLECTION_DIR, "guidelines", d))
             if f.endswith(".json")])
        for d in os.listdir(os.path.join(COLLECTION_DIR, "guidelines"))
        if os.path.isdir(os.path.join(COLLECTION_DIR, "guidelines", d))
    )
    print(f"\n  Total: {total} files in {len(os.listdir(os.path.join(COLLECTION_DIR, 'guidelines')))} patterns")


if __name__ == "__main__":
    from typing import Any
    main()
