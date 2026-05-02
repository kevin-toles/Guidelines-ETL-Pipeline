#!/usr/bin/env python3
"""
Deep Coverage Analyzer: CRE Enriched Metadata → Guideline Potential
====================================================================
Analyzes ALL enriched repo metadata files across 98 domains to map:
  - Which repos have real content vs stubs
  - Which chapters have code examples, patterns, architecture decisions
  - What topics/keywords/concepts are available per domain
  - Cross-domain relationship potential (same patterns across repos)
  - Mapping to guideline v2.0 fields (situation, pattern, code_analysis, reasoning)

Output: comprehensive JSON report + actionable summary
"""

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

ENRICHED_DIR = "/Users/kevintoles/POC/ai-platform-data/repos/enriched"
CRE_GUIDELINES_DIR = "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines/guidelines/cre-repos"
REPO_REGISTRY = "/Users/kevintoles/POC/ai-platform-data/repos/repo_registry.json"
OUTPUT_DIR = "/Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines/analysis"

STUB_MARKER = "Auto-generated stub from repo_registry.json"
MIN_CONTENT_LEN = 100  # chars — below this = stub


def load_registry():
    with open(REPO_REGISTRY) as f:
        reg = json.load(f)
    repo_to_domain = {}
    domain_to_repos = defaultdict(list)
    for domain in reg.get("domains", []):
        domain_id = domain["id"]
        for repo in domain.get("repos", []):
            repo_id = repo["id"]
            repo_to_domain[repo_id] = domain_id
            domain_to_repos[domain_id].append(repo_id)
    return repo_to_domain, domain_to_repos


def analyze_enriched_files():
    """Full deep analysis of all enriched repo metadata files."""
    stats = {
        "total_domains": 0,
        "total_files": 0,
        "total_chapters": 0,
        "chapters_with_content": 0,
        "chapters_stub_only": 0,
        "total_content_chars": 0,
        "domains": {},
        "files": [],
        "all_keywords": Counter(),
        "all_concepts": Counter(),
        "code_indicators_found": 0,
        "pattern_indicators_found": 0,
    }

    for domain_name in sorted(os.listdir(ENRICHED_DIR)):
        domain_dir = os.path.join(ENRICHED_DIR, domain_name)
        if not os.path.isdir(domain_dir):
            continue

        domain_stats = {
            "total_files": 0,
            "total_chapters": 0,
            "chapters_with_content": 0,
            "chapters_stub_only": 0,
            "total_content_chars": 0,
            "keywords": Counter(),
            "concepts": Counter(),
            "repos": [],
        }

        for fname in sorted(os.listdir(domain_dir)):
            if not fname.endswith("_enriched.json"):
                continue
            fpath = os.path.join(domain_dir, fname)
            repo_id = fname.replace("_enriched.json", "")

            try:
                with open(fpath) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"  ⚠️ Parse error: {domain_name}/{fname}: {e}")
                continue

            book = data.get("book", {})
            chapters = data.get("chapters", [])
            provenance = data.get("enrichment_provenance", {})

            repo_stats = {
                "repo_id": repo_id,
                "title": book.get("title", repo_id),
                "chapter_count": len(chapters),
                "chapters_with_content": 0,
                "chapters_stub": 0,
                "total_content_chars": 0,
                "keywords": Counter(),
                "concepts": Counter(),
                "has_code_examples": False,
                "has_pattern_mentions": False,
                "bloom_level": book.get("taxonomy", {}).get("bloom_level", ""),
                "chapter_details": [],
            }

            for ch in chapters:
                content = ch.get("content", "")
                summary = ch.get("summary", "")
                ch_keywords = ch.get("keywords", [])
                ch_concepts = ch.get("concepts", [])

                is_stub = (STUB_MARKER in summary or STUB_MARKER in content or
                          (len(content) < MIN_CONTENT_LEN and len(summary) < MIN_CONTENT_LEN))

                ch_detail = {
                    "title": ch.get("title", ""),
                    "content_len": len(content),
                    "summary_len": len(summary),
                    "is_stub": is_stub,
                    "keyword_count": len(ch_keywords),
                    "concept_count": len(ch_concepts),
                }

                repo_stats["chapter_details"].append(ch_detail)

                if not is_stub:
                    repo_stats["chapters_with_content"] += 1
                    repo_stats["total_content_chars"] += len(content)
                    domain_stats["chapters_with_content"] += 1
                    domain_stats["total_content_chars"] += len(content)
                    stats["chapters_with_content"] += 1
                    stats["total_content_chars"] += len(content)

                    # Check for code/pattern indicators
                    code_indicators = [
                        "```", "def ", "function ", "class ", "fn ",
                        "import ", "package ", "module ", "fn main",
                        "void ", "int ", "String ", "let ", "const ",
                        "var ", "use ", "impl ", "struct ", "enum ",
                    ]
                    if any(ind in content for ind in code_indicators):
                        repo_stats["has_code_examples"] = True
                        stats["code_indicators_found"] += 1

                    pattern_indicators = [
                        "pattern", "anti-pattern", "design pattern",
                        "architecture", "best practice", "principle",
                        "convention", "idiom", "rule", "standard",
                    ]
                    if any(ind in content.lower() for ind in pattern_indicators):
                        repo_stats["has_pattern_mentions"] = True
                        stats["pattern_indicators_found"] += 1
                else:
                    repo_stats["chapters_stub"] += 1
                    domain_stats["chapters_stub_only"] += 1
                    stats["chapters_stub_only"] += 1

                # Aggregate
                for kw in ch_keywords:
                    repo_stats["keywords"][kw] += 1
                    domain_stats["keywords"][kw] += 1
                    stats["all_keywords"][kw] += 1
                for c in ch_concepts:
                    repo_stats["concepts"][c] += 1
                    domain_stats["concepts"][c] += 1
                    stats["all_concepts"][c] += 1

            domain_stats["repos"].append(repo_stats)
            domain_stats["total_files"] += 1
            domain_stats["total_chapters"] += len(chapters)
            stats["total_files"] += 1
            stats["total_chapters"] += len(chapters)

        stats["domains"][domain_name] = domain_stats
        stats["total_domains"] += 1

    return stats


def analyze_cre_guidelines():
    """Analyze existing CRE guidelines for gap analysis."""
    if not os.path.exists(CRE_GUIDELINES_DIR):
        return {"error": "CRE guidelines dir not found"}

    gstats = {
        "total_guidelines": 0,
        "repos_represented": Counter(),
        "domains_represented": Counter(),
        "avg_content_richness": 0,
        "with_code_analysis": 0,
        "with_code_blocks": 0,
        "with_empty_code_analysis": 0,
        "with_empty_metadata": 0,
    }

    idx_path = os.path.join(CRE_GUIDELINES_DIR, "_cre_guideline_index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            gstats["index"] = json.load(f)

    for fname in os.listdir(CRE_GUIDELINES_DIR):
        if not fname.endswith(".json") or fname.startswith("_"):
            continue
        fpath = os.path.join(CRE_GUIDELINES_DIR, fname)
        try:
            with open(fpath) as f:
                g = json.load(f)
        except:
            continue

        gstats["total_guidelines"] += 1

        repo = g.get("pattern", {}).get("pattern_id", "unknown")
        gstats["repos_represented"][repo] += 1

        ca = g.get("code_analysis", {})
        if ca:
            has_data = any([
                ca.get("primary_languages"),
                ca.get("relevant_code_patterns"),
                ca.get("key_modules"),
            ])
            if has_data:
                gstats["with_code_analysis"] += 1
            else:
                gstats["with_empty_code_analysis"] += 1

        md = g.get("metadata", {})
        if md:
            has_meta = any([
                md.get("top_concepts"),
                md.get("top_keywords"),
                md.get("bloom_level"),
            ])
            if not has_meta:
                gstats["with_empty_metadata"] += 1

    return gstats


def compute_domain_scores(stats):
    """Compute per-domain quality scores for guideline potential."""
    scores = []
    for domain_name, ds in stats["domains"].items():
        if ds["total_chapters"] == 0:
            continue
        content_ratio = ds["chapters_with_content"] / ds["total_chapters"]
        avg_content = ds["total_content_chars"] / max(ds["chapters_with_content"], 1)
        unique_keywords = len(ds["keywords"])
        unique_concepts = len(ds["concepts"])

        # Repos with code examples
        code_repos = sum(1 for r in ds["repos"] if r["has_code_examples"])
        pattern_repos = sum(1 for r in ds["repos"] if r["has_pattern_mentions"])

        scores.append({
            "domain": domain_name,
            "repos": ds["total_files"],
            "chapters": ds["total_chapters"],
            "content_ratio": round(content_ratio, 3),
            "avg_content_chars": round(avg_content, 0),
            "unique_keywords": unique_keywords,
            "unique_concepts": unique_concepts,
            "code_repos": code_repos,
            "pattern_repos": pattern_repos,
            "composite_score": round(
                content_ratio * 0.3 +
                min(avg_content / 1000, 1) * 0.2 +
                min(unique_concepts / 50, 1) * 0.2 +
                (code_repos / max(ds["total_files"], 1)) * 0.15 +
                (pattern_repos / max(ds["total_files"], 1)) * 0.15, 3
            ),
        })

    scores.sort(key=lambda x: x["composite_score"], reverse=True)
    return scores


def analyze_repo_enrichment_gaps(stats, repo_to_domain):
    """Identify repos in registry that have NO enriched data or only stubs."""
    enriched_ids = set()
    for ds in stats["domains"].values():
        for r in ds["repos"]:
            enriched_ids.add(r["repo_id"])

    registry_ids = set(repo_to_domain.keys())
    missing = registry_ids - enriched_ids
    stub_only = []
    for ds in stats["domains"].values():
        for r in ds["repos"]:
            if r["chapters_with_content"] == 0:
                stub_only.append(r["repo_id"])

    return {
        "total_in_registry": len(registry_ids),
        "total_enriched": len(enriched_ids),
        "missing_from_enrichment": sorted(missing),
        "missing_count": len(missing),
        "stub_only": sorted(stub_only),
        "stub_only_count": len(stub_only),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("CRE ENRICHMENT COVERAGE ANALYZER")
    print("=" * 70)

    repo_to_domain, domain_to_repos = load_registry()
    print(f"\nRegistry: {len(repo_to_domain)} repos across {len(domain_to_repos)} domains")

    # ── Phase 1: Analyze enriched metadata ──
    print("\n📊 Analyzing enriched repo metadata...")
    stats = analyze_enriched_files()

    print(f"   Total domains: {stats['total_domains']}")
    print(f"   Total files: {stats['total_files']}")
    print(f"   Total chapters: {stats['total_chapters']}")
    print(f"   Chapters with content: {stats['chapters_with_content']} ({100*stats['chapters_with_content']/max(stats['total_chapters'],1):.1f}%)")
    print(f"   Chapters stub-only: {stats['chapters_stub_only']} ({100*stats['chapters_stub_only']/max(stats['total_chapters'],1):.1f}%)")
    print(f"   Total content characters: {stats['total_content_chars']:,}")
    print(f"   Files with code indicators: {stats['code_indicators_found']}")
    print(f"   Files with pattern mentions: {stats['pattern_indicators_found']}")
    print(f"   Unique keywords (all): {len(stats['all_keywords']):,}")
    print(f"   Unique concepts (all): {len(stats['all_concepts']):,}")

    # ── Phase 2: Domain scores ──
    print("\n📊 Computing domain quality scores...")
    scores = compute_domain_scores(stats)

    print(f"\n{'─' * 60}")
    print(f"{'Domain':<30} {'Repos':>5} {'Content%':>8} {'AvgLen':>7} {'KW':>5} {'Con':>5} {'Code':>5} {'Pat':>5} {'Score':>7}")
    print(f"{'─' * 60}")
    for s in scores[:30]:
        print(f"{s['domain']:<30} {s['repos']:>5} {s['content_ratio']:>7.0%} {s['avg_content_chars']:>7.0f} "
              f"{s['unique_keywords']:>5} {s['unique_concepts']:>5} {s['code_repos']:>5} {s['pattern_repos']:>5} {s['composite_score']:>7.3f}")

    # ── Phase 3: Repo enrichment gaps ──
    print("\n📊 Analyzing enrichment gaps...")
    gaps = analyze_repo_enrichment_gaps(stats, repo_to_domain)
    print(f"   Total in registry: {gaps['total_in_registry']}")
    print(f"   Total enriched: {gaps['total_enriched']}")
    print(f"   Missing from enrichment: {gaps['missing_count']}")
    print(f"   Stub-only repos: {gaps['stub_only_count']}")
    if gaps["missing_from_enrichment"]:
        print(f"   Missing repos: {', '.join(gaps['missing_from_enrichment'][:20])}...")

    # ── Phase 4: CRE guideline analysis ──
    print("\n📊 Analyzing existing CRE guidelines...")
    gstats = analyze_cre_guidelines()
    print(f"   Total CRE guidelines: {gstats['total_guidelines']}")
    print(f"   With populated code_analysis: {gstats['with_code_analysis']}")
    print(f"   With empty code_analysis: {gstats['with_empty_code_analysis']}")
    print(f"   With empty metadata: {gstats['with_empty_metadata']}")

    # ── Top keywords across all domains ──
    print(f"\n📊 Top 30 keywords across ALL enriched content:")
    for kw, count in stats["all_keywords"].most_common(30):
        print(f"   {kw:30} {count:>6}")

    print(f"\n📊 Top 30 concepts across ALL enriched content:")
    for c, count in stats["all_concepts"].most_common(30):
        print(f"   {c:30} {count:>6}")

    # ── Write full report ──
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_domains": stats["total_domains"],
            "total_files": stats["total_files"],
            "total_chapters": stats["total_chapters"],
            "content_ratio": round(stats["chapters_with_content"] / max(stats["total_chapters"], 1), 3),
            "unique_keywords": len(stats["all_keywords"]),
            "unique_concepts": len(stats["all_concepts"]),
        },
        "domain_scores": scores,
        "enrichment_gaps": gaps,
        "cre_guidelines": gstats,
        "top_keywords": stats["all_keywords"].most_common(100),
        "top_concepts": stats["all_concepts"].most_common(100),
        # Per-domain top keywords
        "per_domain_top_keywords": {
            d: ds["keywords"].most_common(20)
            for d, ds in stats["domains"].items()
        },
    }

    report_path = os.path.join(OUTPUT_DIR, "cre_coverage_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=list)
    print(f"\n✅ Full report written to: {report_path}")

    summary_path = os.path.join(OUTPUT_DIR, "cre_coverage_summary.md")
    with open(summary_path, "w") as f:
        f.write(f"# CRE Enrichment Coverage Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Overall Stats\n\n")
        f.write(f"- **{stats['total_files']}** enriched files across **{stats['total_domains']}** domains\n")
        f.write(f"- **{stats['total_chapters']}** total chapters\n")
        f.write(f"- **{stats['chapters_with_content']}** ({100*stats['chapters_with_content']/max(stats['total_chapters'],1):.1f}%) chapters have real content\n")
        f.write(f"- **{stats['chapters_stub_only']}** ({100*stats['chapters_stub_only']/max(stats['total_chapters'],1):.1f}%) are stubs\n")
        f.write(f"- **{stats['code_indicators_found']}** repos contain code examples\n")
        f.write(f"- **{stats['pattern_indicators_found']}** repos mention design patterns\n")
        f.write(f"- **{len(stats['all_keywords'])}** unique keywords, **{len(stats['all_concepts'])}** unique concepts\n\n")

        f.write(f"## Top 20 Domains by Guideline Potential Score\n\n")
        f.write(f"| Domain | Repos | Content% | Avg Len | KWs | Concepts | Code | Pattern | Score |\n")
        f.write(f"|--------|-------|----------|---------|-----|----------|------|---------|-------|\n")
        for s in scores[:20]:
            f.write(f"| {s['domain']} | {s['repos']} | {s['content_ratio']:.0%} | {s['avg_content_chars']:.0f} | {s['unique_keywords']} | {s['unique_concepts']} | {s['code_repos']} | {s['pattern_repos']} | {s['composite_score']:.3f} |\n")

        f.write(f"\n## Enrichment Gaps\n\n")
        f.write(f"- **{gaps['missing_count']}** repos in registry have NO enriched data\n")
        f.write(f"- **{gaps['stub_only_count']}** repos are stub-only (no real content)\n")
        if gaps["missing_from_enrichment"]:
            f.write(f"- Missing: {', '.join(gaps['missing_from_enrichment'][:30])}\n")

    print(f"✅ Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
