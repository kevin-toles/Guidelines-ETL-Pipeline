#!/usr/bin/env python3
"""
Analysis Step B3: Enrichment Scoring
======================================
Scores how well each SE cluster/topic can enrich existing guidelines:
  - Content richness: does the cluster contain code, accepted answers, high signal?
  - Coverage gap: is this material not already in guidelines?
  - Actionability: can it be transformed into a concrete guideline, scenario, or check?

Produces ranked enrichment recommendations.

Input:  B2_coverage_map.json, A7_quality_scores.jsonl, A6_code_stats.json
Output: B3_enrichment_scores.json
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("analyze_enrichment_scoring")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "mining"


def load_inputs(output_dir: Path) -> tuple:
    """Load B2 coverage, A7 quality, A6 code stats."""
    with open(output_dir / "B2" / "coverage_map.json", "r") as f:
        coverage = json.load(f)

    quality_scores = {}
    qp = output_dir / "A7" / "quality_scores.jsonl"
    if qp.exists():
        with open(qp, "r") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    quality_scores[rec.get("id")] = rec

    code_stats = {}
    cp = output_dir / "A6" / "code_stats.json"
    if cp.exists():
        with open(cp, "r") as f:
            code_stats = json.load(f)

    return coverage, quality_scores, code_stats


# ── Scoring Functions ────────────────────────────────────────────────────────


def score_content_richness(cluster: dict, quality_scores: dict) -> dict:
    """
    Content richness = how valuable is the SE content in this cluster?
    Components: avg quality score, code presence, accepted answer rate.
    """
    # Cluster-level estimates (quality scores indexed by question ID, not cluster)
    # We use cluster-level proxies from B1 interpretation
    pct_code = cluster.get("pct_with_code", 0)
    pct_accepted = cluster.get("pct_with_accepted", 0)
    avg_score_norm = min(cluster.get("avg_score", 0) / 100.0, 1.0)  # Normalize

    richness = 0.35 * avg_score_norm + 0.25 * (pct_code / 100) + 0.25 * (pct_accepted / 100) + 0.15 * min(cluster.get("size", 0) / 10000, 1.0)

    return {
        "score": round(richness, 3),
        "breakdown": {
            "avg_score_contribution": round(0.35 * avg_score_norm, 3),
            "code_presence_contribution": round(0.25 * pct_code / 100, 3),
            "accepted_answer_contribution": round(0.25 * pct_accepted / 100, 3),
            "volume_contribution": round(0.15 * min(cluster.get("size", 0) / 10000, 1.0), 3),
        },
        "tier": (
            "excellent" if richness > 0.7
            else "good" if richness > 0.5
            else "moderate" if richness > 0.3
            else "low"
        ),
    }


def score_coverage_gap(cluster: dict, coverage_map: dict) -> dict:
    """
    Coverage gap = how much is this cluster NOT covered by existing guidelines?
    High gap = uncovered cluster → high enrichment value.
    """
    coverage_strength = coverage_map.get("coverage_strength", "none")

    if coverage_strength == "none":
        gap_score = 1.0
    elif coverage_strength == "weak":
        gap_score = 0.7
    elif coverage_strength == "strong":
        # Check if it's a very strong match (high combined score)
        best_match = coverage_map.get("best_guideline_match", {})
        combined = best_match.get("combined_score", 0) if best_match else 0
        if combined > 0.7:
            gap_score = 0.1  # Already well covered
        elif combined > 0.4:
            gap_score = 0.3
        else:
            gap_score = 0.5
    else:
        gap_score = 0.5

    return {
        "score": round(gap_score, 3),
        "coverage_strength": coverage_strength,
        "best_guideline_match": coverage_map.get("best_guideline_match"),
        "tier": (
            "critical_gap" if gap_score > 0.8
            else "significant_gap" if gap_score > 0.5
            else "partial_coverage" if gap_score > 0.2
            else "well_covered"
        ),
    }


def score_actionability(cluster: dict, code_stats: dict) -> dict:
    """
    Actionability = can this content be transformed into a concrete artifact?
    High actionability: has code examples, clear patterns, high signal.
    """
    pct_code = cluster.get("pct_with_code", 0)

    # Code presence → can create check/example
    code_actionability = min(pct_code / 100, 1.0)

    # Size → enough material for a guideline
    size_actionability = min(cluster.get("size", 0) / 1000, 1.0)

    # Tag specificity → clear domain
    top_tags = cluster.get("top_tags", [])
    n_specific_tags = sum(
        1
        for t in top_tags
        if t.get("tag", "") not in ["general", "code", "programming", "help", "question"]
    )
    specificity = min(n_specific_tags / 5, 1.0)

    actionability = 0.4 * code_actionability + 0.3 * size_actionability + 0.3 * specificity

    return {
        "score": round(actionability, 3),
        "breakdown": {
            "code_actionability": round(0.4 * code_actionability, 3),
            "volume_actionability": round(0.3 * size_actionability, 3),
            "specificity_actionability": round(0.3 * specificity, 3),
        },
        "suggested_artifact_type": suggest_artifact_type(cluster, pct_code, size_actionability),
        "tier": (
            "highly_actionable" if actionability > 0.7
            else "actionable" if actionability > 0.4
            else "low_actionability"
        ),
    }


def suggest_artifact_type(cluster: dict, pct_code: float, size: float) -> str:
    """Suggest what type of guideline artifact this cluster could produce."""
    tag_category = cluster.get("dominant_tag_category", "general")
    size_val = cluster.get("size", 0)

    if pct_code > 50:
        if tag_category in ("testing", "security"):
            return "Check (validatable rule with code examples)"
        return "Scenario (pattern with code illustrations)"

    if size_val > 1000:
        return "Guideline (principle with multiple scenarios)"

    if tag_category in ("algorithms", "data_science"):
        return "Constraint (formulaic rule)"

    return "Concept (definition and boundary)"


def compute_composite_enrichment(
    richness: dict, gap: dict, actionability: dict
) -> dict:
    """
    Composite enrichment score: weighted combination.
    Weights favor gap (0.5) because uncovered material is most valuable for enrichment.
    """
    raw = 0.5 * gap["score"] + 0.25 * richness["score"] + 0.25 * actionability["score"]

    return {
        "score": round(raw, 3),
        "breakdown": {
            "gap_weight": 0.5,
            "richness_weight": 0.25,
            "actionability_weight": 0.25,
        },
        "priority": (
            "P0_critical" if raw > 0.7
            else "P1_high" if raw > 0.5
            else "P2_medium" if raw > 0.3
            else "P3_low"
        ),
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR

    log.info("=" * 60)
    log.info("Analysis Step B3: Enrichment Scoring")
    log.info("Reads intermediates from: %s", output_dir)
    log.info("Output: %s", output_dir)
    log.info("=" * 60)

    # 1. Load
    coverage, quality_scores, code_stats = load_inputs(output_dir)

    # 2. Score each K-Means cluster
    km_coverage = coverage.get("kmeans_coverage", [])
    enrichment_results = []

    for cluster_map in km_coverage:
        # Find matching cluster interpretation from B1
        # (cluster_map has the coverage info; we also need the original cluster stats)
        richness = score_content_richness(cluster_map, quality_scores)
        gap = score_coverage_gap(cluster_map, cluster_map)
        actionability = score_actionability(cluster_map, code_stats)
        composite = compute_composite_enrichment(richness, gap, actionability)

        enrichment_results.append(
            {
                "cluster_id": cluster_map.get("cluster_id"),
                "cluster_label": cluster_map.get("cluster_label"),
                "size": cluster_map.get("size"),
                "top_tags": cluster_map.get("top_tags", [])[:5],
                "coverage_strength": cluster_map.get("coverage_strength"),
                "content_richness": richness,
                "coverage_gap": gap,
                "actionability": actionability,
                "composite_enrichment": composite,
            }
        )

    # Sort by composite score descending
    enrichment_results.sort(key=lambda x: x["composite_enrichment"]["score"], reverse=True)

    # Priority counts
    priority_counts = defaultdict(int)
    for r in enrichment_results:
        priority_counts[r["composite_enrichment"]["priority"]] += 1

    # 3. Save
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "metadata": {
            "n_clusters_scored": len(enrichment_results),
            "scoring_weights": {
                "richness": 0.25,
                "gap": 0.5,
                "actionability": 0.25,
            },
        },
        "priority_summary": dict(priority_counts),
        "enrichment_recommendations": enrichment_results,
    }

    with open(output_dir / "B3" / "enrichment_scores.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info("  ✓ B3_enrichment_scores.json (%d clusters)", len(enrichment_results))

    # Log top recommendations
    log.info("Top enrichment opportunities:")
    for r in enrichment_results[:10]:
        log.info(
            "  [%s] %s (size=%d) — %.3f",
            r["composite_enrichment"]["priority"],
            r["cluster_label"],
            r["size"],
            r["composite_enrichment"]["score"],
        )

    log.info("B3 enrichment scoring complete.")


if __name__ == "__main__":
    main()
