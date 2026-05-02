#!/usr/bin/env python3
"""
CRE Enrichment Depth Analyzer — v1.0
Deep analysis of CRE enrichment data to assess content quality and identify gaps.

Analyzes:
  1. Enriched metadata files (500 files, 98 domains)
  2. Content depth: chapter counts, keyword counts, concept coverage
  3. Stub detection: files with auto-generated placeholder content
  4. Domain coverage vs repo_registry.json
  5. Bridge potential: which enriched data can fill guideline gaps

Outputs:
  - cre_enrichment_inventory.json    — per-file enrichment stats
  - cre_stub_report.json             — stub/placeholder detection
  - cre_domain_coverage.json         — domain-by-domain quality metrics
  - cre_bridge_potential.json        — which enrichment data maps to guideline fields

Usage:
  python3 analyze_cre_enrichment.py [--enriched-dir PATH] [--output-dir PATH]
"""

import json, os, sys, re
from collections import defaultdict, Counter
from datetime import datetime

ENRICHED_DIR = os.environ.get(
    "ENRICHED_DIR",
    "/Users/kevintoles/POC/ai-platform-data/repos/enriched"
)
REGISTRY_PATH = os.environ.get(
    "REGISTRY_PATH",
    "/Users/kevintoles/POC/ai-platform-data/repos/repo_registry.json"
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines/scripts/analysis_tools/output"
)

# Stub markers
STUB_MARKERS = [
    "Auto-generated stub from repo_registry.json",
    "Auto-generated stub",
    "placeholder",
    "TODO: enrich",
    "No enrichment data available",
    "Enrichment pending",
]


def detect_stub(chapter):
    """Detect if a chapter or content is a stub/placeholder."""
    content = chapter.get("content", "")
    title = chapter.get("title", "")
    summary = chapter.get("summary", "")

    if not content or len(content.strip()) < 50:
        return True, "empty_or_very_short"

    for marker in STUB_MARKERS:
        if marker.lower() in content.lower():
            return True, f"stub_marker: {marker[:80]}"

    if not title or title.strip() == "":
        return True, "no_title"

    return False, "ok"


def analyze_enriched_file(filepath, domain):
    """Deep analysis of a single enriched JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
        return {"file": os.path.basename(filepath), "domain": domain, "error": str(e)}

    result = {
        "file": os.path.basename(filepath),
        "domain": domain,
        "size_bytes": os.path.getsize(filepath),
        "size_kb": round(os.path.getsize(filepath) / 1024, 1),
    }

    # Top-level keys
    result["top_keys"] = list(data.keys())

    # Chapter analysis
    chapters = data.get("chapters", [])
    result["chapter_count"] = len(chapters)

    if chapters:
        stub_count = 0
        total_content_len = 0
        total_keywords = 0
        total_concepts = 0
        chapter_stubs = []

        for i, ch in enumerate(chapters):
            is_stub, reason = detect_stub(ch)
            if is_stub:
                stub_count += 1
                chapter_stubs.append({"index": i, "title": ch.get("title", "?")[:80], "reason": reason})

            total_content_len += len(ch.get("content", "") or "")
            total_keywords += len(ch.get("keywords", []) or [])
            total_concepts += len(ch.get("concepts", []) or [])

        result["stub_chapters"] = stub_count
        result["stub_pct"] = round(stub_count / len(chapters) * 100, 1) if chapters else 0
        result["avg_content_len"] = round(total_content_len / len(chapters), 0)
        result["avg_keywords"] = round(total_keywords / len(chapters), 1)
        result["avg_concepts"] = round(total_concepts / len(chapters), 1)
        result["chapter_stubs_sample"] = chapter_stubs[:5]

        # Quality score
        quality = 0
        if len(chapters) >= 3:
            quality += 1
        if result["avg_content_len"] > 200:
            quality += 1
        if result["avg_content_len"] > 1000:
            quality += 2
        if result["stub_pct"] < 20:
            quality += 1
        if result["stub_pct"] == 0:
            quality += 2
        if result["avg_keywords"] > 5:
            quality += 1
        if result["avg_concepts"] > 3:
            quality += 1

        result["quality_score"] = min(quality, 10)
        result["quality_tier"] = (
            "rich" if quality >= 7
            else "adequate" if quality >= 4
            else "thin" if quality >= 2
            else "stub"
        )
    else:
        result["stub_chapters"] = 0
        result["stub_pct"] = 0
        result["avg_content_len"] = 0
        result["avg_keywords"] = 0
        result["avg_concepts"] = 0
        result["quality_score"] = 0
        result["quality_tier"] = "empty"

    # Metadata analysis
    metadata = data.get("metadata", {})
    result["has_metadata"] = bool(metadata)
    result["meta_keys"] = list(metadata.keys())[:10] if metadata else []

    # Keywords / concepts at file level
    result["file_keywords"] = len(data.get("keywords", []) or [])
    result["file_concepts"] = len(data.get("concepts", []) or [])

    # Bridge potential: which fields map to guideline schema
    bridge_fields = {}
    if chapters:
        bridge_fields["situation.summary"] = "chapter content provides problem context"
        bridge_fields["guideline"] = "chapter content can be distilled into guidelines"
        bridge_fields["reasoning"] = "chapter content provides reasoning depth"
        bridge_fields["pattern.pattern_name"] = "domain+chapter_title indicate patterns"
    if result["avg_content_len"] > 500:
        bridge_fields["hints"] = "rich content can yield step-by-step hints"
        bridge_fields["constraints"] = "content discusses constraints and design considerations"

    result["bridge_potential"] = bridge_fields

    return result


def main():
    print("=" * 72)
    print("  CRE ENRICHMENT DEPTH ANALYZER")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Source: {ENRICHED_DIR}")
    print("=" * 72)

    if not os.path.exists(ENRICHED_DIR):
        print(f"  ❌ Enriched directory not found: {ENRICHED_DIR}")
        return

    # ── Phase 1: Walk all enriched files ────────────────────────────────────
    results = []
    domain_stats = defaultdict(lambda: {
        "file_count": 0,
        "total_chapters": 0,
        "stub_chapters": 0,
        "total_content_len": 0,
        "files": [],
        "quality_tiers": Counter(),
    })

    for root, dirs, files in os.walk(ENRICHED_DIR):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(root, ENRICHED_DIR)
            domain = rel.replace("/", "_") if rel != "." else "root"

            result = analyze_enriched_file(fpath, domain)
            results.append(result)

            # Aggregate domain stats
            ds = domain_stats[domain]
            ds["file_count"] += 1
            ds["total_chapters"] += result.get("chapter_count", 0)
            ds["stub_chapters"] += result.get("stub_chapters", 0)
            ds["total_content_len"] += result.get("avg_content_len", 0) * result.get("chapter_count", 0)
            ds["quality_tiers"][result.get("quality_tier", "unknown")] += 1
            ds["files"].append({
                "file": result["file"],
                "quality_tier": result.get("quality_tier"),
                "chapters": result.get("chapter_count", 0),
                "size_kb": result.get("size_kb", 0),
            })

    print(f"\n📁 Found {len(results)} enriched files across {len(domain_stats)} domains")

    # ── Phase 2: Quality distribution ────────────────────────────────────────
    quality_tiers = Counter(r.get("quality_tier", "unknown") for r in results)
    print(f"\n📊 Quality Tier Distribution:")
    for tier in ["rich", "adequate", "thin", "stub", "empty"]:
        count = quality_tiers.get(tier, 0)
        pct = round(count / len(results) * 100, 1) if results else 0
        print(f"  {tier}: {count} ({pct}%)")

    # ── Phase 3: Top and bottom files ────────────────────────────────────────
    sorted_by_quality = sorted(results, key=lambda r: r.get("quality_score", 0), reverse=True)
    print(f"\n🏆 Top 10 richest files:")
    for r in sorted_by_quality[:10]:
        print(f"  [{r['domain']}] {r['file']}: "
              f"score={r.get('quality_score')}, "
              f"chapters={r.get('chapter_count')}, "
              f"avg_content={r.get('avg_content_len', 0):.0f} chars, "
              f"stubs={r.get('stub_pct', 0)}%")

    print(f"\n💀 Bottom 10 (most stubby):")
    for r in sorted_by_quality[-10:]:
        print(f"  [{r['domain']}] {r['file']}: "
              f"score={r.get('quality_score')}, "
              f"chapters={r.get('chapter_count')}, "
              f"avg_content={r.get('avg_content_len', 0):.0f} chars, "
              f"size={r.get('size_kb', 0)} KB")

    # ── Phase 4: Bridge potential summary ────────────────────────────────────
    bridge_summary = Counter()
    for r in results:
        for field in r.get("bridge_potential", {}):
            bridge_summary[field] += 1

    print(f"\n🔗 Bridge Potential (files that can enrich each guideline field):")
    for field, count in bridge_summary.most_common():
        pct = round(count / len(results) * 100, 1) if results else 0
        print(f"  {field}: {count} files ({pct}%)")

    # ── Write outputs ────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    outputs = {
        "cre_enrichment_inventory.json": {
            "source_dir": ENRICHED_DIR,
            "total_files": len(results),
            "total_domains": len(domain_stats),
            "quality_distribution": dict(quality_tiers),
            "top_rich": [{"domain": r["domain"], "file": r["file"], "score": r.get("quality_score")}
                         for r in sorted_by_quality[:20]],
            "bridge_potential": dict(bridge_summary.most_common()),
            "file_details": results[:100],  # cap for readability
        },
        "cre_stub_report.json": {
            "total_files": len(results),
            "stub_files": [r for r in results if r.get("quality_tier") in ("stub", "empty")][:50],
            "stub_count_by_domain": {
                d: {
                    "total_files": s["file_count"],
                    "stub_pct": round(s["quality_tiers"].get("stub", 0) / s["file_count"] * 100, 1) if s["file_count"] else 0,
                }
                for d, s in domain_stats.items()
            },
        },
        "cre_domain_coverage.json": {
            "domains": {
                d: {
                    "file_count": s["file_count"],
                    "total_chapters": s["total_chapters"],
                    "stub_chapters": s["stub_chapters"],
                    "stub_chapter_pct": round(s["stub_chapters"] / s["total_chapters"] * 100, 1) if s["total_chapters"] > 0 else 0,
                    "avg_content_by_chapter": round(s["total_content_len"] / s["total_chapters"], 0) if s["total_chapters"] > 0 else 0,
                    "quality_tiers": dict(s["quality_tiers"]),
                    "top_files": sorted(s["files"], key=lambda x: x.get("size_kb", 0), reverse=True)[:5],
                }
                for d, s in sorted(domain_stats.items())
            },
        },
    }

    for fname, data in outputs.items():
        outpath = os.path.join(OUTPUT_DIR, fname)
        with open(outpath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"  ✅ Wrote: {outpath}")

    print(f"\n{'=' * 72}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
