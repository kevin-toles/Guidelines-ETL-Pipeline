#!/usr/bin/env python3
"""
Phase 3: CRE Repo Guideline Extractor
=======================================
Extracts guideline-like entries from enriched CRE repo metadata.

For each enriched repo metadata JSON, reads the chapters/content and transforms
into pattern profiles that can be used as enrichment context.

Output:
    /Users/kevintoles/POC/ai-platform-data/collections/coding-guidelines/guidelines/cre-repos/
        {repo_id}_profile_NNNNN.json

Usage:
    python3 extract_cre_guidelines.py
"""

import json
import os
import re
from typing import Any

# ── Configuration ──────────────────────────────────────────────────
ENRICHED_DIRS = [
    "/Users/kevintoles/POC/ai-platform-data/repos/enriched",
]
OUTPUT_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines/patterns/cre-repos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Domain-to-pattern mapping ─────────────────────────────────────
DOMAIN_PATTERN_MAP: dict[str, list[tuple[str, str]]] = {
    "ml-frameworks": [
        ("deep-learning", "Deep Learning"),
        ("dynamic-computational-graph", "Dynamic Computational Graph"),
        ("autograd", "Automatic Differentiation"),
        ("module-api", "Module API Design"),
    ],
    "ml-inference": [
        ("model-serving", "Model Serving"),
        ("batch-inference", "Batch Inference"),
        ("quantization", "Quantization"),
        ("gpu-optimization", "GPU Optimization"),
    ],
    "ml-rag-agents": [
        ("retrieval-augmented-generation", "Retrieval Augmented Generation"),
        ("agent-orchestration", "Agent Orchestration"),
        ("tool-use", "Tool Use Pattern"),
        ("memory-management", "Memory Management"),
    ],
    "ml-agents": [
        ("multi-agent-systems", "Multi-Agent Systems"),
        ("agent-communication", "Agent Communication"),
        ("autonomous-agents", "Autonomous Agents"),
    ],
    "ml-fine-tuning": [
        ("fine-tuning", "Fine Tuning"),
        ("peft", "PEFT / Parameter Efficient Fine Tuning"),
        ("transfer-learning", "Transfer Learning"),
    ],
    "ml-similarity-search": [
        ("similarity-search", "Similarity Search"),
        ("vector-database", "Vector Database"),
        ("approximate-nearest-neighbor", "Approximate Nearest Neighbor"),
    ],
    "gpu-computing": [
        ("cuda-programming", "CUDA Programming"),
        ("gpu-kernel", "GPU Kernel Design"),
        ("memory-optimization", "Memory Optimization"),
    ],
    "computer-vision": [
        ("image-processing", "Image Processing"),
        ("feature-extraction", "Feature Extraction"),
        ("object-detection", "Object Detection"),
    ],
}


def find_enriched_files() -> list[tuple[str, str, str]]:
    """Find all enriched JSON files across domains."""
    files = []
    for base_dir in ENRICHED_DIRS:
        if not os.path.exists(base_dir):
            continue
        for domain in os.listdir(base_dir):
            domain_dir = os.path.join(base_dir, domain)
            if not os.path.isdir(domain_dir):
                continue
            for f in os.listdir(domain_dir):
                if f.endswith("_enriched.json"):
                    files.append((os.path.join(domain_dir, f), domain, f.replace("_enriched.json", "")))
    return sorted(files)


def extract_content_patterns(content: str) -> list[tuple[str, str]]:
    """Extract pattern mentions from unstructured content text."""
    patterns = []
    # Look for "Patterns:" line
    m = re.search(r'Patterns:\s*(.+?)(?:\n|$)', content)
    if m:
        for p in m.group(1).split(","):
            p = p.strip()
            if p:
                patterns.append((p, p.replace("-", " ").title()))
    return patterns


def extract_guideline_from_repo(data: dict, domain: str, repo_id: str) -> list[dict]:
    """Extract guideline entries from an enriched repo metadata JSON."""
    guidelines = []
    book = data.get("book", {})
    chapters = data.get("chapters", [])
    provenance = data.get("enrichment_provenance", {})

    title = book.get("title", repo_id)
    taxonomy = book.get("taxonomy", {})
    concepts = taxonomy.get("top_concepts_from_taxonomy", [])
    keywords = taxonomy.get("top_keywords_from_taxonomy", [])

    # Get domain patterns
    domain_patterns = DOMAIN_PATTERN_MAP.get(domain, [])

    for i, ch in enumerate(chapters):
        summary = ch.get("summary", "")
        ch_content = ch.get("content", "")
        ch_keywords = ch.get("keywords", [])
        ch_concepts = ch.get("concepts", [])

        # Combine content for analysis
        combined = f"{summary}\n{ch_content}"

        # Extract patterns mentioned in content
        content_patterns = extract_content_patterns(combined)

        # Use domain patterns as fallback
        all_patterns = content_patterns or domain_patterns

        if not all_patterns:
            all_patterns = [(repo_id, title)]

        for j, (pid, pname) in enumerate(all_patterns):
            # Build a guideline from the enriched metadata
            guideline = {
                "guideline_id": f"cre_{repo_id}_{i+1:03d}_{j+1:03d}",
                "schema_version": "2.0",
                "source_problem_id": f"{repo_id}/ch{i+1}",
                "source_dataset": "cre_repos",
                "title": f"{pname} - {title}",
                "title_slug": f"{pid}-{repo_id}",
                "link": provenance.get("system_prompt", ""),
                "situation": {
                    "summary": summary or combined[:300],
                    "tags": {
                        "has_sorted_input": False,
                        "has_unsorted_input": False,
                        "has_duplicates": False,
                        "has_contiguous": False,
                        "has_unique_elements": False,
                        "is_search_problem": False,
                        "is_optimization_problem": False,
                        "requires_path": False,
                        "requires_ordering": False,
                    },
                    "difficulty": "N/A",
                    "topics": concepts[:5] or ch_concepts[:5],
                },
                "guideline": f"When working with {pname} in {title}, consider the following approach based on the repository's design patterns and conventions.",
                "reasoning": f"This pattern is evidenced in the {title} repository ({repo_id}), which implements {pname} as a core architectural pattern. The domain ({domain}) provides context for when this pattern is applicable.",
                "complexity": {
                    "time": "N/A (architectural pattern)",
                    "space": "N/A (architectural pattern)",
                },
                "pattern": {
                    "pattern_id": pid,
                    "pattern_name": pname,
                    "category": domain.replace("-", " ").title(),
                },
                "code_analysis": {
                    "primary_languages": [],
                    "relevant_code_patterns": [],
                    "key_modules": [],
                },
                "constraints": [],
                "alternatives": [],
                "has_solution_code": True,
                "has_hints": False,
                "similar_questions": [],
                "acceptance_rate": None,
                "likes": None,
                "bridges": [],
                "metadata": {
                    "source": "cre_repos",
                    "repo_id": repo_id,
                    "domain": domain,
                    "chapter_title": ch.get("title", ""),
                    "top_concepts": concepts[:10],
                    "top_keywords": [k.get("term", k) if isinstance(k, dict) else k for k in ch_keywords[:10]],
                    "bloom_level": taxonomy.get("bloom_level", ""),
                },
            }
            guidelines.append(guideline)

    return guidelines


def build_index(guidelines: list[dict]) -> dict:
    """Build an index of extracted repo guidelines."""
    by_repo: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    by_pattern: dict[str, int] = {}

    for g in guidelines:
        repo = g.get("metadata", {}).get("repo_id", "unknown")
        domain = g.get("metadata", {}).get("domain", "unknown")
        pid = g.get("pattern", {}).get("pattern_id", "unknown")

        by_repo[repo] = by_repo.get(repo, 0) + 1
        by_domain[domain] = by_domain.get(domain, 0) + 1
        by_pattern[pid] = by_pattern.get(pid, 0) + 1

    return {
        "total": len(guidelines),
        "by_repo": dict(sorted(by_repo.items(), key=lambda x: -x[1])),
        "by_domain": dict(sorted(by_domain.items(), key=lambda x: -x[1])),
        "by_pattern": dict(sorted(by_pattern.items(), key=lambda x: -x[1])),
        "source": "cre_repos",
    }


def main():
    enriched_files = find_enriched_files()
    print(f"Found {len(enriched_files)} enriched repo files\n")

    all_guidelines = []

    for filepath, domain, repo_id in enriched_files:
        with open(filepath) as f:
            data = json.load(f)

        guidelines = extract_guideline_from_repo(data, domain, repo_id)
        print(f"  {repo_id:30s} ({domain:20s}): {len(guidelines)} guidelines")

        for i, g in enumerate(guidelines):
            out_path = os.path.join(OUTPUT_DIR, f"{repo_id}_guideline_{i+1:05d}.json")
            with open(out_path, "w") as f:
                json.dump(g, f, indent=2)

        all_guidelines.extend(guidelines)

    # Write index
    index = build_index(all_guidelines)
    index_path = os.path.join(OUTPUT_DIR, "_cre_guideline_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nTotal: {len(all_guidelines)} guidelines from {len(enriched_files)} repos")
    print(f"Index: {index_path}")

    # Summary
    print("\nBy domain:")
    for domain, count in sorted(index["by_domain"].items(), key=lambda x: -x[1]):
        print(f"  {domain:25s}: {count}")
    print("\nBy pattern (top 20):")
    for pat, count in sorted(index["by_pattern"].items(), key=lambda x: -x[1])[:20]:
        print(f"  {pat:35s}: {count}")


if __name__ == "__main__":
    main()
