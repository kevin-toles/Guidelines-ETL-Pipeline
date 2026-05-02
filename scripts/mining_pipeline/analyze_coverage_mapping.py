#!/usr/bin/env python3
"""
Analysis Step B2: Coverage Mapping
====================================
Maps discovered SE topics/clusters to the existing guideline taxonomy.
Identifies which guidelines map to which SE clusters (coverage) and
produces a coverage heatmap for gap analysis.

This is DATA ANALYSIS — deterministic mapping, not ML.

Input:  B1_cluster_interpretation.json, A3_topic_model.json, A4_association_rules.json
        + guideline taxonomy from collections/coding-guidelines/
Output: B2_coverage_map.json, B2_coverage_gaps.json
"""

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("analyze_coverage_mapping")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "mining"
GUIDELINE_DIR = Path(__file__).resolve().parent.parent.parent  # collections/coding-guidelines/


def load_cluster_interpretations(output_dir: Path) -> dict:
    cp = output_dir / "B1" / "cluster_interpretation.json"
    if not cp.exists():
        log.warning("B1_cluster_interpretation.json not found")
        return {"kmeans_clusters": [], "lda_topics": []}
    with open(cp, "r") as f:
        return json.load(f)


def load_topic_model(output_dir: Path) -> dict:
    tp = output_dir / "A3" / "topic_model.json"
    if not tp.exists():
        return {}
    with open(tp, "r") as f:
        return json.load(f)


def load_association_rules(output_dir: Path) -> dict:
    ap = output_dir / "A4" / "association_rules.json"
    if not ap.exists():
        return {"rules": []}
    with open(ap, "r") as f:
        return json.load(f)


def load_guideline_taxonomy() -> Dict[str, List[dict]]:
    """
    Load existing guideline documents and extract their taxonomy entries.
    Returns {category_name: [{"id": ..., "title": ..., "keywords": [...]}]}
    """
    taxonomy = defaultdict(list)
    docs_dir = GUIDELINE_DIR / "docs"
    if not docs_dir.exists():
        log.warning("Guideline docs directory not found: %s", docs_dir)
        return dict(taxonomy)

    for md_file in sorted(docs_dir.glob("*.md")):
        if md_file.name.startswith("."):
            continue
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            # Extract sections/headings as guideline entries
            headings = re.findall(r"^##\s+(.+)$", content, re.MULTILINE)
            # Extract keywords from tags or code spans
            keywords = set(re.findall(r"`([a-zA-Z][a-zA-Z0-9_-]{2,})`", content))
            # Simple categorization from filename
            category = md_file.stem.replace("_", " ").title()

            for heading in headings:
                taxonomy[category].append(
                    {
                        "id": f"{md_file.stem}#{heading.lower().replace(' ', '-')}",
                        "title": heading.strip(),
                        "source_file": md_file.name,
                        "category": category,
                        "keywords": list(keywords)[:50],
                    }
                )
        except Exception as e:
            log.warning("Error reading %s: %s", md_file.name, e)

    log.info("Loaded guideline taxonomy: %d categories, %d entries",
             len(taxonomy), sum(len(v) for v in taxonomy.values()))
    return dict(taxonomy)


def map_clusters_to_guidelines(
    clusters: List[dict], taxonomy: Dict[str, List[dict]]
) -> List[dict]:
    """
    For each cluster, find the best-matching guideline(s) using:
    1. Tag overlap between cluster top tags and guideline keywords
    2. String similarity between cluster label and guideline titles
    """
    mappings = []

    for cluster in clusters:
        cluster_tags = {t["tag"].lower() for t in cluster.get("top_tags", [])}
        cluster_label = cluster.get("suggested_label", "").lower()

        best_matches = []
        for category, guidelines in taxonomy.items():
            for gl in guidelines:
                gl_keywords = {kw.lower() for kw in gl.get("keywords", [])}
                gl_title = gl["title"].lower()

                # Tag overlap score
                tag_overlap = len(cluster_tags & gl_keywords)
                tag_overlap_ratio = tag_overlap / max(len(cluster_tags), 1)

                # Title similarity (simple word overlap)
                label_words = set(cluster_label.split())
                title_words = set(gl_title.split())
                title_overlap = len(label_words & title_words)
                title_overlap_ratio = title_overlap / max(len(title_words), 1)

                # Combined score
                combined = 0.6 * tag_overlap_ratio + 0.4 * title_overlap_ratio

                if combined > 0.05:  # Minimum threshold
                    best_matches.append(
                        {
                            "guideline_id": gl["id"],
                            "guideline_title": gl["title"],
                            "category": category,
                            "tag_overlap": tag_overlap,
                            "tag_overlap_ratio": round(tag_overlap_ratio, 3),
                            "title_overlap_ratio": round(title_overlap_ratio, 3),
                            "combined_score": round(combined, 3),
                            "matching_tags": list(cluster_tags & gl_keywords),
                            "matching_title_words": list(label_words & title_words),
                        }
                    )

        best_matches.sort(key=lambda x: x["combined_score"], reverse=True)

        mappings.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "algorithm": cluster.get("algorithm"),
                "cluster_label": cluster.get("suggested_label"),
                "size": cluster.get("size"),
                "top_tags": cluster.get("top_tags", [])[:5],
                "best_guideline_match": best_matches[0] if best_matches else None,
                "all_matches": best_matches[:5],
                "has_coverage": len(best_matches) > 0,
                "coverage_strength": (
                    "strong" if best_matches and best_matches[0]["combined_score"] > 0.3
                    else "weak" if best_matches
                    else "none"
                ),
            }
        )

    return mappings


def identify_gaps(mappings: List[dict]) -> dict:
    """Identify clusters with no guideline coverage."""
    no_coverage = [m for m in mappings if not m["has_coverage"]]
    weak_coverage = [m for m in mappings if m.get("coverage_strength") == "weak"]
    strong_coverage = [m for m in mappings if m.get("coverage_strength") == "strong"]

    # Sort gaps by cluster size (larger uncovered clusters = bigger gaps)
    no_coverage.sort(key=lambda x: x["size"], reverse=True)
    weak_coverage.sort(key=lambda x: x["size"], reverse=True)

    return {
        "n_clusters_analyzed": len(mappings),
        "n_covered_strong": len(strong_coverage),
        "n_covered_weak": len(weak_coverage),
        "n_uncovered": len(no_coverage),
        "coverage_rate": round(len(strong_coverage) / max(len(mappings), 1), 3),
        "uncovered_clusters": [
            {
                "cluster_label": c["cluster_label"],
                "size": c["size"],
                "top_tags": c["top_tags"],
                "gap_priority": "high" if c["size"] > 500 else "medium" if c["size"] > 100 else "low",
            }
            for c in no_coverage
        ],
        "weakly_covered_clusters": [
            {
                "cluster_label": c["cluster_label"],
                "size": c["size"],
                "best_match": c["best_guideline_match"],
                "gap_priority": "medium" if c["size"] > 500 else "low",
            }
            for c in weak_coverage
        ],
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR

    log.info("=" * 60)
    log.info("Analysis Step B2: Coverage Mapping")
    log.info("Reads intermediates from: %s", output_dir)
    log.info("Output: %s", output_dir)
    log.info("=" * 60)

    # 1. Load
    interpretations = load_cluster_interpretations(output_dir)
    topic_model = load_topic_model(output_dir)
    assoc_rules = load_association_rules(output_dir)
    taxonomy = load_guideline_taxonomy()

    # 2. Map K-Means clusters
    km_clusters = interpretations.get("kmeans_clusters", [])
    km_mappings = map_clusters_to_guidelines(km_clusters, taxonomy)

    # 3. Map DBSCAN clusters
    db_clusters = interpretations.get("dbscan_clusters", [])
    db_mappings = map_clusters_to_guidelines(db_clusters, taxonomy)

    # 4. Map LDA topics
    lda_topics = interpretations.get("lda_topics", [])
    # Adapt LDA topics to cluster format
    lda_as_clusters = []
    for t in lda_topics:
        lda_as_clusters.append(
            {
                "cluster_id": t.get("topic_id"),
                "algorithm": "lda",
                "suggested_label": t.get("suggested_label", ""),
                "size": 0,
                "top_tags": [{"tag": term, "count": 0} for term, _ in t.get("top_terms", [])[:5]],
            }
        )
    lda_mappings = map_clusters_to_guidelines(lda_as_clusters, taxonomy)

    # 5. Gap analysis
    gaps = identify_gaps(km_mappings)

    # 6. Save
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage = {
        "metadata": {
            "n_guideline_categories": len(taxonomy),
            "n_guideline_entries": sum(len(v) for v in taxonomy.values()),
            "n_se_clusters": len(km_mappings),
            "n_se_topics": len(lda_mappings),
        },
        "kmeans_coverage": km_mappings,
        "dbscan_coverage": db_mappings,
        "lda_coverage": lda_mappings,
        "gap_analysis": gaps,
        "guideline_taxonomy_summary": {
            category: [g["title"] for g in guidelines]
            for category, guidelines in taxonomy.items()
        },
    }

    with open(output_dir / "B2" / "coverage_map.json", "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    log.info("  ✓ B2_coverage_map.json")

    # Separate gaps file for quick reference
    with open(output_dir / "B2" / "coverage_gaps.json", "w", encoding="utf-8") as f:
        json.dump(gaps, f, indent=2, ensure_ascii=False)
    log.info("  ✓ B2_coverage_gaps.json")

    log.info(
        "Coverage: %d strong, %d weak, %d uncovered clusters",
        gaps["n_covered_strong"],
        gaps["n_covered_weak"],
        gaps["n_uncovered"],
    )

    log.info("B2 coverage mapping complete.")


if __name__ == "__main__":
    main()
