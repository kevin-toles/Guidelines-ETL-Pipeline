#!/usr/bin/env python3
"""
Analysis Step B5: Viability Assessment
=======================================
Final viability analysis that synthesizes ALL prior mining and analysis
outputs into a single go/no-go recommendation for each proposed guideline
enrichment.

Produces:
  - Per-cluster viability assessment (VIABLE / CONDITIONAL / NONVIABLE)
  - Overall pipeline viability score
  - Risk register
  - Resource estimates for Phase C (Transformation)

Input:  ALL A1-A7 and B1-B4 outputs
Output: B5_viability_report.json
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("analyze_viability")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "mining"


def load_all_inputs(output_dir: Path) -> dict:
    """Load all available outputs from A and B steps."""
    inputs = {}

    # Each entry: (phase_dir, filename, key)
    phases = [
        ("A1", "stats.json", "A1_stats"),
        ("A2", "cluster_stats.json", "A2_cluster_stats"),
        ("A3", "A3_stats.json", "A3_stats"),
        ("A4", "association_rules.json", "A4_association_rules"),
        ("A5", "anomaly_stats.json", "A5_anomaly_stats"),
        ("A6", "code_stats.json", "A6_code_stats"),
        ("A7", "quality_model.json", "A7_quality_model"),
        ("B1", "cluster_interpretation.json", "B1_cluster_interpretation"),
        ("B2", "coverage_map.json", "B2_coverage_map"),
        ("B3", "enrichment_scores.json", "B3_enrichment_scores"),
        ("B4", "gap_quantification.json", "B4_gap_quantification"),
    ]

    for phase_dir, filename, key in phases:
        fp = output_dir / phase_dir / filename
        if fp.exists():
            with open(fp, "r") as f:
                inputs[key] = json.load(f)
            log.info("Loaded %s/%s", phase_dir, filename)
        else:
            log.debug("Not found: %s/%s", phase_dir, filename)

    return inputs


# ── Viability Criteria ──────────────────────────────────────────────────────


def assess_cluster_viability(enrichment_rec: dict, gap_data: dict) -> dict:
    """Assess whether enriching guidelines from this cluster is viable."""
    composite = enrichment_rec.get("composite_enrichment", {}).get("score", 0)
    priority = enrichment_rec.get("composite_enrichment", {}).get("priority", "P3_low")
    size = enrichment_rec.get("size", 0)
    richness = enrichment_rec.get("content_richness", {}).get("score", 0)
    actionability = enrichment_rec.get("actionability", {}).get("score", 0)
    cluster_label = enrichment_rec.get("cluster_label", "")

    # Viability criteria
    reasons = []

    if composite >= 0.5:
        viability = "VIABLE"
        reasons.append("High enrichment score (>=0.5)")
    elif composite >= 0.3:
        viability = "CONDITIONAL"
        reasons.append("Medium enrichment score (0.3-0.5)")
    else:
        viability = "NONVIABLE"
        reasons.append("Low enrichment score (<0.3)")

    # Size check
    if size < 50:
        viability = min_viability(viability, "CONDITIONAL")
        reasons.append("Small cluster size (<50) — may not have enough material")

    if size > 500:
        reasons.append("Large cluster — sufficient material for guideline")

    # Richness check
    if richness < 0.2:
        viability = min_viability(viability, "NONVIABLE")
        reasons.append("Low content richness — poor quality SE content")

    # Actionability check
    if actionability < 0.2:
        viability = min_viability(viability, "CONDITIONAL")
        reasons.append("Low actionability — hard to transform into guideline artifact")

    # Artifact type
    artifact = enrichment_rec.get("actionability", {}).get("suggested_artifact_type", "")

    return {
        "cluster_id": enrichment_rec.get("cluster_id"),
        "cluster_label": cluster_label,
        "size": size,
        "composite_score": composite,
        "viability": viability,
        "reasons": reasons,
        "suggested_artifact": artifact,
        "estimated_effort_hours": estimate_cluster_effort(viability, size, artifact),
    }


def min_viability(a: str, b: str) -> str:
    """Return the less favorable viability."""
    order = {"VIABLE": 3, "CONDITIONAL": 2, "NONVIABLE": 1}
    return a if order.get(a, 0) < order.get(b, 0) else b


def estimate_cluster_effort(viability: str, size: int, artifact: str) -> float:
    """Estimate effort hours per cluster for Phase C transformation."""
    base = {"VIABLE": 3.0, "CONDITIONAL": 5.0, "NONVIABLE": 0.0}[viability]

    if "Check" in artifact:
        base *= 0.7  # Checks are simpler
    elif "Scenario" in artifact:
        base *= 1.2  # Scenarios need examples
    elif "Guideline" in artifact:
        base *= 1.5  # Full guidelines need more

    # Scale with size
    size_factor = 0.5 + 0.5 * min(size / 1000, 1.0)
    return round(base * size_factor, 1)


def compute_overall_viability(per_cluster: List[dict]) -> dict:
    """Compute overall pipeline viability."""
    n_total = len(per_cluster)
    n_viable = sum(1 for c in per_cluster if c["viability"] == "VIABLE")
    n_conditional = sum(1 for c in per_cluster if c["viability"] == "CONDITIONAL")
    n_nonviable = sum(1 for c in per_cluster if c["viability"] == "NONVIABLE")

    viable_ratio = n_viable / max(n_total, 1)
    conditional_ratio = n_conditional / max(n_total, 1)

    # Weighted viability score
    viable_score = viable_ratio * 1.0 + conditional_ratio * 0.5

    # Total estimated effort
    total_hours = sum(c["estimated_effort_hours"] for c in per_cluster)

    if viable_ratio >= 0.3:
        overall = "GO"
        recommendation = "Sufficient viable clusters — proceed with Phase C transformation."
    elif viable_ratio >= 0.1:
        overall = "CONDITIONAL_GO"
        recommendation = "Some viable clusters but limited. Consider broadening SE corpus or lowering thresholds."
    else:
        overall = "NO_GO"
        recommendation = "Insufficient viable clusters. Revisit data sources or mining parameters."

    return {
        "overall_verdict": overall,
        "recommendation": recommendation,
        "viable_ratio": round(viable_ratio, 3),
        "conditional_ratio": round(conditional_ratio, 3),
        "n_viable": n_viable,
        "n_conditional": n_conditional,
        "n_nonviable": n_nonviable,
        "n_total": n_total,
        "weighted_score": round(viable_score, 3),
        "total_estimated_hours": total_hours,
        "estimated_weeks": round(total_hours / 40, 1),
    }


def build_risk_register(inputs: dict, overall: dict) -> List[dict]:
    """Build risk register for the transformation phase."""
    risks = []

    # Risk: Low quality SE content
    quality = inputs.get("A7_quality_model", {})
    quality_stats = quality.get("quality_stats", {})
    avg_quality = quality_stats.get("score_mean", 0.5)
    if avg_quality < 0.4:
        risks.append(
            {
                "id": "RISK-001",
                "severity": "HIGH",
                "category": "Data Quality",
                "description": f"Low average SE quality score ({avg_quality:.2f})",
                "mitigation": "Apply stricter quality filtering before Phase C",
            }
        )

    # Risk: High anomaly ratio
    anomaly = inputs.get("A5_anomaly_stats", {})
    consensus = anomaly.get("consensus", {})
    anomaly_ratio = consensus.get("anomaly_ratio", 0)
    if anomaly_ratio > 0.05:
        risks.append(
            {
                "id": "RISK-002",
                "severity": "MEDIUM",
                "category": "Data Integrity",
                "description": f"High anomaly ratio ({anomaly_ratio:.2%})",
                "mitigation": "Investigate anomalies manually before transformation",
            }
        )

    # Risk: Low clustering quality
    cluster = inputs.get("A2_cluster_stats", {})
    km_sil = cluster.get("kmeans", {}).get("best_silhouette", 0)
    if km_sil and km_sil < 0.1:
        risks.append(
            {
                "id": "RISK-003",
                "severity": "MEDIUM",
                "category": "Methodology",
                "description": f"Low K-Means silhouette ({km_sil:.3f})",
                "mitigation": "Consider alternative clustering parameters or algorithms",
            }
        )

    # Risk: Insufficient code examples
    code = inputs.get("A6_code_stats", {})
    code_blocks = code.get("language_distribution", {}).get("total_blocks", 0)
    if code_blocks < 1000:
        risks.append(
            {
                "id": "RISK-004",
                "severity": "LOW",
                "category": "Content",
                "description": f"Low code block count ({code_blocks})",
                "mitigation": "Supplement with CRE code corpus",
            }
        )

    # Risk: Too many nonviable clusters
    if overall["viable_ratio"] < 0.15:
        risks.append(
            {
                "id": "RISK-005",
                "severity": "HIGH",
                "category": "Feasibility",
                "description": f"Only {overall['n_viable']}/{overall['n_total']} clusters viable",
                "mitigation": "Lower enrichment thresholds or expand SE data sources",
            }
        )

    # Risk: High effort estimate
    if overall["total_estimated_hours"] > 200:
        risks.append(
            {
                "id": "RISK-006",
                "severity": "MEDIUM",
                "category": "Resourcing",
                "description": f"High estimated effort ({overall['total_estimated_hours']:.0f} hours)",
                "mitigation": "Phase transformation: start with top-10 clusters only",
            }
        )

    return risks


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR

    log.info("=" * 60)
    log.info("Analysis Step B5: Viability Assessment")
    log.info("Reads intermediates from: %s", output_dir)
    log.info("Output: %s", output_dir)
    log.info("=" * 60)

    # 1. Load all inputs
    inputs = load_all_inputs(output_dir)

    # 2. Per-cluster viability
    enrichment = inputs.get("B3_enrichment_scores", {})
    enrichment_recs = enrichment.get("enrichment_recommendations", [])
    gap_data = inputs.get("B2_coverage_map", {}).get("gap_analysis", {})

    per_cluster = [assess_cluster_viability(rec, gap_data) for rec in enrichment_recs]

    # 3. Overall viability
    overall = compute_overall_viability(per_cluster)

    # 4. Risk register
    risks = build_risk_register(inputs, overall)

    # 5. Save
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "metadata": {
            "assessment_date": None,  # Set at runtime
            "input_sources": list(inputs.keys()),
            "n_clusters_assessed": len(per_cluster),
        },
        "overall_viability": overall,
        "per_cluster_viability": per_cluster,
        "risk_register": risks,
        "recommendations": {
            "go": overall["overall_verdict"].startswith("GO"),
            "top_priority_clusters": [
                {
                    "cluster_label": c["cluster_label"],
                    "viability": c["viability"],
                    "suggested_artifact": c["suggested_artifact"],
                    "estimated_hours": c["estimated_effort_hours"],
                }
                for c in per_cluster
                if c["viability"] == "VIABLE"
            ][:10],
            "next_steps": build_next_steps(overall, risks),
        },
    }

    with open(output_dir / "B5" / "viability_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info("  ✓ B5_viability_report.json")

    # Summary
    log.info("=" * 60)
    log.info("VIABILITY VERDICT: %s", overall["overall_verdict"])
    log.info(
        "  Viable: %d | Conditional: %d | Nonviable: %d",
        overall["n_viable"],
        overall["n_conditional"],
        overall["n_nonviable"],
    )
    log.info("  Estimated effort: %.0f hours (%.1f weeks)", overall["total_estimated_hours"], overall["estimated_weeks"])
    log.info("  Risks: %d identified", len(risks))
    log.info("=" * 60)

    log.info("B5 viability assessment complete.")


def build_next_steps(overall: dict, risks: List[dict]) -> List[str]:
    """Build actionable next steps."""
    steps = []

    if overall["overall_verdict"] == "GO":
        steps.append("Proceed to Phase C (Transformation) with VIABLE clusters")
        steps.append("Implement Phase C scripts: transform_guidelines.py enrichment path")
        steps.append("Start with top-5 viable clusters as pilot")
    elif overall["overall_verdict"] == "CONDITIONAL_GO":
        steps.append("Review CONDITIONAL clusters for manual curation")
        steps.append("Consider lowering quality thresholds")
        steps.append("Expand SE corpus with additional sites or tiers")
    else:
        steps.append("Re-evaluate data source quality")
        steps.append("Consider alternative mining parameters")
        steps.append("Explore complementary data sources (GitHub, blogs, documentation)")

    if any(r["severity"] == "HIGH" for r in risks):
        steps.append("Mitigate HIGH-severity risks before Phase C")

    return steps


if __name__ == "__main__":
    main()
