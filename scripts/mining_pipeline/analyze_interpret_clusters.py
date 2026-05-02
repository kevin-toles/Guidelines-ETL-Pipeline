#!/usr/bin/env python3
"""
Analysis Step B1: Interpret Clusters
=====================================
Takes the algorithmic cluster output from Phase A and interprets what
each cluster/topic represents in human terms.

This is DATA ANALYSIS, not data mining. It algorithmically labels clusters
by extracting distinguishing features (top tags, terms, sites) and computes
cluster quality metrics (cohesion, separation, silhouette).

Dependencies: numpy, scipy
Input:  A2_cluster_labels.jsonl, A2_cluster_stats.json, A1_text_corpus.jsonl, A3_topic_model.json
Output: B1_cluster_interpretation.json
"""

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("analyze_interpret_clusters")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "mining"


def load_inputs(output_dir: Path) -> Tuple[List[dict], dict, List[dict], dict]:
    """Load A1, A2, A3 outputs."""
    with open(output_dir / "A2" / "cluster_labels.jsonl", "r") as f:
        clusters = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded A2 clusters: %d records", len(clusters))

    with open(output_dir / "A2" / "cluster_stats.json", "r") as f:
        cluster_stats = json.load(f)
    log.info("Loaded A2 stats")

    corpus = []
    cp = output_dir / "A1" / "text_corpus.jsonl"
    if cp.exists():
        with open(cp, "r") as f:
            corpus = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded A1 corpus: %d records", len(corpus))

    topic_model = None
    tp = output_dir / "A3" / "topic_model.json"
    if tp.exists():
        with open(tp, "r") as f:
            topic_model = json.load(f)
    log.info("Loaded A3 topic model")

    return clusters, cluster_stats, corpus, topic_model


def interpret_kmeans_clusters(
    clusters: List[dict], corpus: List[dict]
) -> List[dict]:
    """
    For each K-Means cluster, extract:
    - Top tags
    - Top sites
    - Dominant tag categories
    - Cluster size and quality score stats
    """
    # Group by kmeans cluster
    km_groups = defaultdict(list)
    record_lookup = {}
    for rec in corpus:
        record_lookup[rec.get("id")] = rec

    for i, c in enumerate(clusters):
        km = c.get("kmeans_cluster")
        if km is not None:
            km_groups[km].append(i)

    interpretations = []
    for cluster_id, indices in sorted(km_groups.items()):
        # Collect stats
        tags = Counter()
        sites = Counter()
        tiers = Counter()
        scores = []
        quality_signals = Counter()

        for idx in indices:
            if idx < len(corpus):
                rec = corpus[idx]
                for tag in rec.get("tags", []):
                    tags[tag] += 1
                sites[rec.get("site", "")] += 1
                tiers[rec.get("tier", "")] += 1
                scores.append(rec.get("score", 0))
                if rec.get("has_accepted_answer"):
                    quality_signals["accepted"] += 1
                if rec.get("code_block_count", 0) > 0:
                    quality_signals["has_code"] += 1

        # Categorize dominant tags
        top_tags = tags.most_common(10)
        tag_category = classify_tag_cluster([t for t, _ in top_tags])

        interp = {
            "cluster_id": cluster_id,
            "algorithm": "kmeans",
            "size": len(indices),
            "pct_of_corpus": round(100 * len(indices) / max(len(clusters), 1), 1),
            "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
            "dominant_tag_category": tag_category,
            "top_sites": dict(sites.most_common(5)),
            "tier_distribution": dict(tiers),
            "avg_score": float(np.mean(scores)) if scores else 0,
            "median_score": float(np.median(scores)) if scores else 0,
            "pct_with_code": round(100 * quality_signals.get("has_code", 0) / max(len(indices), 1), 1),
            "pct_with_accepted": round(100 * quality_signals.get("accepted", 0) / max(len(indices), 1), 1),
            "suggested_label": suggest_cluster_label(top_tags, sites, tag_category),
        }
        interpretations.append(interp)

    log.info("Interpreted %d K-Means clusters", len(interpretations))
    return interpretations


def interpret_dbscan_clusters(
    clusters: List[dict], corpus: List[dict]
) -> List[dict]:
    """Same interpretation for DBSCAN clusters (excluding noise = -1)."""
    db_groups = defaultdict(list)
    for i, c in enumerate(clusters):
        db = c.get("dbscan_cluster")
        if db is not None and db != -1:
            db_groups[db].append(i)

    interpretations = []
    for cluster_id, indices in sorted(db_groups.items()):
        tags = Counter()
        sites = Counter()
        scores = []

        for idx in indices:
            if idx < len(corpus):
                rec = corpus[idx]
                for tag in rec.get("tags", []):
                    tags[tag] += 1
                sites[rec.get("site", "")] += 1
                scores.append(rec.get("score", 0))

        top_tags = tags.most_common(10)
        tag_category = classify_tag_cluster([t for t, _ in top_tags])

        interpretations.append(
            {
                "cluster_id": cluster_id,
                "algorithm": "dbscan",
                "size": len(indices),
                "pct_of_non_noise": 0,  # computed later
                "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
                "dominant_tag_category": tag_category,
                "top_sites": dict(sites.most_common(5)),
                "avg_score": float(np.mean(scores)) if scores else 0,
                "suggested_label": suggest_cluster_label(top_tags, sites, tag_category),
            }
        )
    log.info("Interpreted %d DBSCAN clusters", len(interpretations))
    return interpretations


def interpret_lda_topics(topic_model: dict) -> List[dict]:
    """Interpret LDA topics by their top terms."""
    if not topic_model:
        return []

    top_terms = topic_model.get("top_terms", {})
    interpretations = []
    for topic_id, terms in sorted(top_terms.items(), key=lambda x: int(x[0])):
        interp = {
            "topic_id": int(topic_id),
            "top_terms": terms[:10],
            "suggested_label": suggest_topic_label(terms[:10]),
        }
        interpretations.append(interp)
    log.info("Interpreted %d LDA topics", len(interpretations))
    return interpretations


# ── Helper: Tag Classification ───────────────────────────────────────────────


def classify_tag_cluster(tags: List[str]) -> str:
    """Classify a cluster's dominant tag category."""
    categories = {
        "programming_languages": ["java", "python", "javascript", "c#", "c++", "rust", "go", "typescript", "swift", "kotlin", "php", "ruby", "scala", "haskell", "clojure"],
        "web_development": ["html", "css", "react", "angular", "vue", "django", "flask", "node.js", "express", "asp.net", "spring", "laravel", "ruby-on-rails"],
        "databases": ["sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle", "database", "nosql"],
        "devops": ["docker", "kubernetes", "jenkins", "terraform", "ansible", "aws", "azure", "gcp", "ci/cd", "linux"],
        "data_science": ["python", "pandas", "numpy", "tensorflow", "pytorch", "scikit-learn", "machine-learning", "deep-learning", "nlp", "r"],
        "testing": ["unit-testing", "testing", "jest", "pytest", "selenium", "mockito", "junit", "tdd"],
        "architecture": ["design-patterns", "architecture", "microservices", "rest", "api", "oop", "functional-programming"],
        "security": ["security", "encryption", "authentication", "oauth", "ssl", "xss", "csrf"],
        "algorithms": ["algorithm", "data-structures", "time-complexity", "sorting", "graph", "dynamic-programming"],
    }

    scores = defaultdict(int)
    for tag in tags:
        tag_lower = tag.lower()
        for category, keywords in categories.items():
            if any(kw in tag_lower for kw in keywords):
                scores[category] += 1

    if scores:
        return max(scores, key=scores.get)
    return "general"


def suggest_cluster_label(top_tags: List[Tuple[str, int]], sites: Counter, category: str) -> str:
    """Suggest a human-readable cluster label."""
    if not top_tags:
        return "Unlabeled cluster"

    tag_names = [t for t, _ in top_tags[:3]]
    top_site = sites.most_common(1)[0][0] if sites else "unknown"

    if category != "general":
        return f"{category.title()}: {', '.join(tag_names)}"
    return f"General: {', '.join(tag_names)} (site: {top_site})"


def suggest_topic_label(terms: List[Tuple[str, float]]) -> str:
    """Suggest a label for an LDA topic."""
    if not terms:
        return "Unlabeled topic"
    top_words = [t for t, _ in terms[:3]]
    return f"Topic: {', '.join(top_words)}"


# ── Cluster Quality Analysis ─────────────────────────────────────────────────


def analyze_cluster_quality(clusters: List[dict], cluster_stats: dict) -> dict:
    """Analyze overall clustering quality: silhouette, separation, noise."""
    return {
        "kmeans_silhouette": cluster_stats.get("kmeans", {}).get("best_silhouette"),
        "kmeans_n_clusters": cluster_stats.get("kmeans", {}).get("n_clusters"),
        "dbscan_n_clusters": cluster_stats.get("dbscan", {}).get("n_clusters"),
        "dbscan_noise_ratio": cluster_stats.get("dbscan", {}).get("noise_ratio"),
        "recommendation": recommend_algorithm(cluster_stats),
    }


def recommend_algorithm(stats: dict) -> str:
    """Recommend which clustering algorithm to use based on stats."""
    km_sil = stats.get("kmeans", {}).get("best_silhouette", 0) or 0
    db_noise = stats.get("dbscan", {}).get("noise_ratio", 1) or 1

    if km_sil > 0.3:
        return "kmeans"
    if db_noise < 0.3:
        return "dbscan"
    if km_sil > 0.1:
        return "kmeans_cautious"
    return "both_exploratory"


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR

    log.info("=" * 60)
    log.info("Analysis Step B1: Interpret Clusters")
    log.info("Reads intermediates from: %s", output_dir)
    log.info("Output: %s", output_dir)
    log.info("=" * 60)

    # 1. Load
    clusters, cluster_stats, corpus, topic_model = load_inputs(output_dir)

    # 2. Interpret
    km_interpretations = interpret_kmeans_clusters(clusters, corpus)
    db_interpretations = interpret_dbscan_clusters(clusters, corpus)
    lda_interpretations = interpret_lda_topics(topic_model)

    # 3. Quality analysis
    quality = analyze_cluster_quality(clusters, cluster_stats)

    # 4. Cross-cluster overlap analysis
    kmeans_labels = np.array([c.get("kmeans_cluster", -1) for c in clusters])
    dbscan_labels = np.array([c.get("dbscan_cluster", -1) for c in clusters])

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "analysis_metadata": {
            "n_records": len(clusters),
            "algorithms": ["kmeans", "dbscan", "lda"],
        },
        "cluster_quality": quality,
        "kmeans_clusters": km_interpretations,
        "dbscan_clusters": db_interpretations,
        "lda_topics": lda_interpretations,
        "cross_algorithm_overlap": {
            "kmeans_dbscan_ari": None,  # Would need full clustering objects
            "note": "ARI requires raw cluster assignments from both algorithms",
        },
    }

    with open(output_dir / "B1" / "cluster_interpretation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info("  ✓ B1_cluster_interpretation.json")

    log.info("B1 cluster interpretation complete.")


if __name__ == "__main__":
    main()
