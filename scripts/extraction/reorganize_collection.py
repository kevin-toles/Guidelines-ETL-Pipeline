#!/usr/bin/env python3
"""
Phase 4: Reorganize Coding Guidelines Collection
=================================================
Distributes CRE-repo and textbook content into properly organized
pattern-based directories. No LLM involved — pure file operations.

Steps:
  1. Reorganize metadata/pattern_profiles/ by source
  2. Map real CRE pattern matches to existing or new directories
  3. Move fallback (templated) entries to references/
  4. Remove flat cre-repos/ directory
  5. Regenerate all metadata indexes

Usage:
    python3 reorganize_collection.py
"""

import json
import os
import shutil

# ── Configuration ──────────────────────────────────────────────────
COLLECTION = "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines"
GUIDELINES_DIR = os.path.join(COLLECTION, "guidelines")
METADATA_DIR = os.path.join(COLLECTION, "metadata")
PROFILES_DIR = os.path.join(METADATA_DIR, "pattern_profiles")
REFERENCES_DIR = os.path.join(COLLECTION, "references")

# Textbook profiles: directory under pattern_profiles/
TEXTBOOK_PROFILE_DIRS = {
    "dsa_profile": "textbooks/dsa",
    "ml_profile": "textbooks/machine-learning",
    "machine-learning_profile": "textbooks/machine-learning",
    "microservices_profile": "textbooks/microservices",
    "api-design_profile": "textbooks/api-design",
    "codewars_profile": "textbooks/codewars",
}


def step1_reorganize_pattern_profiles():
    """Organize pattern_profiles/ by source in subdirectories."""
    print("\n=== Step 1: Reorganize pattern_profiles/ ===")

    # Read existing profiles
    profile_files = [f for f in os.listdir(PROFILES_DIR)
                     if f.endswith(".json") and not f.startswith("_")]

    # Create source directories
    for prefix, subdir in TEXTBOOK_PROFILE_DIRS.items():
        target = os.path.join(PROFILES_DIR, subdir)
        os.makedirs(target, exist_ok=True)

    grokking_dir = os.path.join(PROFILES_DIR, "grokking")
    os.makedirs(grokking_dir, exist_ok=True)

    moved = 0
    for f in profile_files:
        src = os.path.join(PROFILES_DIR, f)

        # Determine target based on filename prefix
        matched = False
        for prefix, subdir in TEXTBOOK_PROFILE_DIRS.items():
            if f.startswith(prefix):
                dst = os.path.join(PROFILES_DIR, subdir, f)
                shutil.move(src, dst)
                moved += 1
                matched = True
                break

        if not matched and "_profile_" in f:
            # Likely a Grokking profile (no prefix match)
            dst = os.path.join(grokking_dir, f)
            if not os.path.exists(os.path.join(PROFILES_DIR, "grokking", f)):
                shutil.move(src, dst)
                moved += 1

    print(f"  Moved {moved} profile files into source subdirectories")
    return moved


def step2_relocate_cre_to_references():
    """
    Relocate ALL CRE-repo entries to references/repo-metadata/.

    Audit found that 100% of the 1,060+ entries contain templated filler
    ('When working with X in Y, consider the following approach...') with
    zero actual guidance content. These serve as repo metadata references,
    not as actionable coding guidelines. Move them to references/ where
    they won't pollute the pattern-based guideline directories.
    """
    print("\n=== Step 2: Relocate CRE entries to references/ ===")

    cre_dir = os.path.join(GUIDELINES_DIR, "cre-repos")
    if not os.path.exists(cre_dir):
        print("  cre-repos/ not found, skipping")
        return 0

    ref_dir = os.path.join(REFERENCES_DIR, "repo-metadata")
    os.makedirs(ref_dir, exist_ok=True)

    all_files = sorted(os.listdir(cre_dir))
    moved = 0

    for f in all_files:
        if not f.endswith(".json"):
            continue
        if f.startswith("_"):
            # Move index/metadata files too
            pass
        src = os.path.join(cre_dir, f)
        dst = os.path.join(ref_dir, f)

        # Avoid collisions
        if os.path.exists(dst):
            base, ext = os.path.splitext(f)
            counter = 1
            while os.path.exists(os.path.join(ref_dir, f"{base}_{counter}{ext}")):
                counter += 1
            dst = os.path.join(ref_dir, f"{base}_{counter}{ext}")

        shutil.move(src, dst)
        moved += 1

    # Remove empty cre-repos directory
    remaining = os.listdir(cre_dir)
    if not remaining:
        os.rmdir(cre_dir)
        print(f"  Removed empty cre-repos/ directory")

    print(f"  Moved {moved} files to {ref_dir}")
    return moved


def step3_regenerate_indexes():
    """Regenerate metadata indexes."""
    print("\n=== Step 3: Regenerate metadata indexes ===")

    # Build pattern index
    patterns = []
    for d in sorted(os.listdir(GUIDELINES_DIR)):
        pdir = os.path.join(GUIDELINES_DIR, d)
        if not os.path.isdir(pdir) or d.startswith(".") or d.startswith("_"):
            continue
        files = [f for f in os.listdir(pdir)
                 if f.endswith(".json") and not f.startswith("_")]
        if files:
            patterns.append({
                "pattern_id": d,
                "count": len(files),
                "files": files[:3],
            })

    index_path = os.path.join(METADATA_DIR, "_pattern_index.json")
    with open(index_path, "w") as f:
        json.dump(patterns, f, indent=2)

    total = sum(p["count"] for p in patterns)
    print(f"  Updated _pattern_index.json: {len(patterns)} patterns, {total} total files")

    # Also update _pattern_groups.json meta
    groups_path = os.path.join(METADATA_DIR, "_pattern_groups.json")
    if os.path.exists(groups_path):
        with open(groups_path) as f:
            groups_data = json.load(f)
        groups_data["_meta"]["total_guidelines"] = total
        groups_data["_meta"]["total_patterns"] = len(patterns)
        with open(groups_path, "w") as f:
            json.dump(groups_data, f, indent=2)
        print(f"  Updated _pattern_groups.json metadata")

    return total, len(patterns)


def main():
    print("=" * 60)
    print("  GUIDELINES COLLECTION REORGANIZATION")
    print("=" * 60)

    profile_count = step1_reorganize_pattern_profiles()
    cre_count = step2_relocate_cre_to_references()
    total, pattern_count = step3_regenerate_indexes()

    print("\n" + "=" * 60)
    print("  REORGANIZATION COMPLETE")
    print("=" * 60)
    print(f"  Pattern profiles reorganized:  {profile_count}")
    print(f"  CRE entries → references:      {cre_count}")
    print(f"  Total patterns:                {pattern_count}")
    print(f"  Total guideline files:         {total}")
    print(f"\n  Guideline directories:")
    for d in sorted(os.listdir(GUIDELINES_DIR)):
        pdir = os.path.join(GUIDELINES_DIR, d)
        if os.path.isdir(pdir) and not d.startswith(".") and not d.startswith("_"):
            count = len([f for f in os.listdir(pdir) if f.endswith(".json") and not f.startswith("_")])
            if count > 0:
                print(f"    {d:35s}: {count} files")
    print(f"\n  References:")
    ref_path = os.path.join(REFERENCES_DIR, "repo-metadata")
    if os.path.exists(ref_path):
        ref_count = len([f for f in os.listdir(ref_path) if f.endswith(".json") and not f.startswith("_")])
        print(f"    repo-metadata: {ref_count} files")


if __name__ == "__main__":
    main()
