#!/usr/bin/env python3
"""
Analysis Step B4: Gap Quantification
=====================================
Quantifies specific gaps in the guideline taxonomy by comparing:
  - SE topic coverage (what SE covers) vs guideline coverage (what guidelines cover)
  - Identifies underserved domains, common failure patterns, and missing constraint types
  - Produces quantified gap items with severity and estimated effort to close

Input:  B2_coverage_map.json, B2_coverage_gaps.json, B3_enrichment_scores.json, A5_anomaly_stats.json
Output: B4_gap_quantification.json
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("analyze_gap_quantification")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "mining"

# ── Target Graph Model Node Types ───────────────────────────────────────────

NODE_TYPES = ["Guideline", "Scenario", "Constraint", "Concept", "FailureMode", "Check"]

# ── Gap Categories ──────────────────────────────────────────────────────────

GAP_CATEGORIES = {
    "missing_guideline": "No guideline exists for this domain",
    "incomplete_guideline": "Existing guideline lacks scenarios or checks",
    "missing_failure_mode": "Common failure patterns not documented",
    "missing_check": "No validatable checks or constraints exist",
    "underrepresented_domain": "Domain has some coverage but is underserved",
    "code_language_gap": "Missing language-specific guidance for popular language",
}


def load_inputs(output_dir: Path) -> tuple:
    with open(output_dir / "B2" / "coverage_map.json", "r") as f:
        coverage = json.load(f)
    with open(output_dir / "B2" / "coverage_gaps.json", "r") as f:
        gaps = json.load(f)
    with open(output_dir / "B3" / "enrichment_scores.json", "r") as f:
        enrichment = json.load(f)

    anomaly_stats = {}
    ap = output_dir / "A5" / "anomaly_stats.json"
    if ap.exists():
        with open(ap, "r") as f:
            anomaly_stats = json.load(f)

    return coverage, gaps, enrichment, anomaly_stats


def quantify_domain_gaps(gaps_data: dict) -> List[dict]:
    """Quantify gaps per domain category."""
    domain_gaps = []
    uncovered = gaps_data.get("uncovered_clusters", [])
    weakly_covered = gaps_data.get("weakly_covered_clusters", [])

    # Group uncovered by tag category
    by_category = defaultdict(lambda: {"count": 0, "total_size": 0, "clusters": []})

    for cluster in uncovered:
        # Infer category from tags
        tags = [t.get("tag", "") for t in cluster.get("top_tags", [])]
        category = classify_domain(tags)
        by_category[category]["count"] += 1
        by_category[category]["total_size"] += cluster.get("size", 0)
        by_category[category]["clusters"].append(
            {
                "label": cluster.get("cluster_label"),
                "size": cluster.get("size"),
                "priority": cluster.get("gap_priority", "medium"),
            }
        )

    for category, data in sorted(by_category.items()):
        domain_gaps.append(
            {
                "domain": category,
                "n_uncovered_clusters": data["count"],
                "total_uncovered_size": data["total_size"],
                "severity": (
                    "critical" if data["total_size"] > 5000
                    else "high" if data["total_size"] > 1000
                    else "medium" if data["total_size"] > 200
                    else "low"
                ),
                "top_uncovered_clusters": sorted(
                    data["clusters"], key=lambda x: x["size"], reverse=True
                )[:5],
                "gap_type": "missing_guideline",
                "estimated_effort": estimate_effort(data["count"], data["total_size"]),
            }
        )

    return domain_gaps


def quantify_node_type_gaps(coverage_data: dict) -> List[dict]:
    """Quantify which node types are missing from guidelines."""
    km_coverage = coverage_data.get("kmeans_coverage", [])
    taxonomy_summary = coverage_data.get("guideline_taxonomy_summary", {})

    # Count existing guideline entries by node type (heuristic from titles)
    existing_nodes = defaultdict(int)
    for category, titles in taxonomy_summary.items():
        for title in titles:
            node_type = infer_node_type(title)
            existing_nodes[node_type] += 1

    # What SE clusters need but don't exist
    needed_nodes = defaultdict(int)
    for cluster in km_coverage:
        if cluster.get("coverage_strength") in ("none", "weak"):
            suggested = suggest_needed_node_type(cluster)
            needed_nodes[suggested] += 1

    node_gaps = []
    for node_type in NODE_TYPES:
        existing = existing_nodes.get(node_type, 0)
        needed = needed_nodes.get(node_type, 0)
        gap = max(0, needed - existing)
        node_gaps.append(
            {
                "node_type": node_type,
                "existing_count": existing,
                "needed_estimate": needed,
                "gap": gap,
                "severity": (
                    "high" if gap > 10 else "medium" if gap > 3 else "low" if gap > 0 else "none"
                ),
            }
        )

    return node_gaps


def quantify_language_gaps(coverage_data: dict, code_stats: dict) -> List[dict]:
    """Identify languages with code but no guidelines."""
    lang_dist = code_stats.get("language_distribution", {}).get("language_distribution", {})
    if not lang_dist:
        return []

    taxonomy_summary = coverage_data.get("guideline_taxonomy_summary", {})

    # Languages present in SE code but missing from guidelines
    language_gaps = []
    for lang, info in lang_dist.items():
        if lang == "unknown":
            continue
        count = info.get("count", 0)
        # Check if language is covered in guidelines
        covered = any(
            lang.lower() in title.lower()
            for titles in taxonomy_summary.values()
            for title in titles
        )

        if not covered and count > 100:
            language_gaps.append(
                {
                    "language": lang,
                    "code_blocks": count,
                    "severity": "high" if count > 1000 else "medium",
                    "gap_type": "code_language_gap",
                }
            )

    language_gaps.sort(key=lambda x: x["code_blocks"], reverse=True)
    return language_gaps


def quantify_anomaly_gaps(anomaly_stats: dict) -> dict:
    """Quantify how many anomalies suggest guideline gaps."""
    if not anomaly_stats:
        return {}

    consensus = anomaly_stats.get("consensus", {})
    return {
        "n_consensus_anomalies": consensus.get("n_anomalies", 0),
        "anomaly_ratio": consensus.get("anomaly_ratio", 0),
        "gap_relevance": (
            "high" if consensus.get("anomaly_ratio", 0) > 0.05
            else "medium" if consensus.get("anomaly_ratio", 0) > 0.02
            else "low"
        ),
        "interpretation": (
            "Many anomalous records suggest undiscovered topic areas — likely guideline gaps."
            if consensus.get("anomaly_ratio", 0) > 0.05
            else "Moderate anomalies may indicate some gaps or simply noise."
        ),
    }


# ── Helpers ─────────────────────────────────────────────────────────────────


def classify_domain(tags: List[str]) -> str:
    """Classify tags into a domain category."""
    domains = {
        "Web Development": ["html", "css", "javascript", "react", "angular", "vue", "node", "django", "flask", "php"],
        "Backend & Systems": ["java", "spring", "c#", ".net", "go", "rust", "api", "microservices"],
        "Data & ML": ["python", "pandas", "numpy", "tensorflow", "pytorch", "machine-learning", "sql"],
        "DevOps & Cloud": ["docker", "kubernetes", "aws", "azure", "terraform", "linux", "ci/cd"],
        "Mobile": ["android", "ios", "swift", "kotlin", "react-native", "flutter"],
        "Testing & QA": ["testing", "unit-test", "selenium", "junit", "pytest", "jest", "tdd"],
        "Security": ["security", "encryption", "authentication", "oauth", "ssl"],
        "Architecture & Design": ["design-patterns", "architecture", "oop", "solid", "patterns"],
        "Algorithms & CS": ["algorithm", "data-structures", "complexity", "sorting"],
    }

    tags_lower = [t.lower() for t in tags]
    scores = defaultdict(int)
    for domain, keywords in domains.items():
        for kw in keywords:
            if any(kw in t for t in tags_lower):
                scores[domain] += 1

    return max(scores, key=scores.get) if scores else "General"


def infer_node_type(title: str) -> str:
    """Infer node type from guideline title."""
    title_lower = title.lower()
    if any(w in title_lower for w in ["pattern", "example", "scenario", "how-to", "recipe"]):
        return "Scenario"
    if any(w in title_lower for w in ["check", "test", "verify", "validate", "audit"]):
        return "Check"
    if any(w in title_lower for w in ["constraint", "rule", "must", "shall", "ensure"]):
        return "Constraint"
    if any(w in title_lower for w in ["concept", "definition", "overview", "introduction"]):
        return "Concept"
    if any(w in title_lower for w in ["failure", "mistake", "pitfall", "anti-pattern", "bug"]):
        return "FailureMode"
    return "Guideline"


def suggest_needed_node_type(cluster: dict) -> str:
    """Suggest what node type a cluster needs."""
    tags = [t.get("tag", "").lower() for t in cluster.get("top_tags", [])]
    label = cluster.get("cluster_label", "").lower()

    if any(w in label for w in ["fail", "error", "bug", "mistake", "pitfall"]):
        return "FailureMode"
    if any(w in tags for w in ["testing", "unit-test", "validation", "lint"]):
        return "Check"
    if any(w in label for w in ["pattern", "example", "how"]):
        return "Scenario"
    if any(w in label for w in ["rule", "constraint", "must"]):
        return "Constraint"

    return "Guideline"


def estimate_effort(n_clusters: int, total_size: int) -> dict:
    """Estimate effort to close the gap."""
    # Rough heuristic: 1-2 hours per cluster for research + drafting
    research_hours = n_clusters * 1.5
    drafting_hours = total_size * 0.01  # 1 hour per 100 questions of material
    review_hours = n_clusters * 0.5

    total = research_hours + drafting_hours + review_hours

    return {
        "research_hours": round(research_hours, 1),
        "drafting_hours": round(drafting_hours, 1),
        "review_hours": round(review_hours, 1),
        "total_hours_estimate": round(total, 1),
        "confidence": "low" if n_clusters > 20 else "medium",
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR

    log.info("=" * 60)
    log.info("Analysis Step B4: Gap Quantification")
    log.info("Reads intermediates from: %s", output_dir)
    log.info("Output: %s", output_dir)
    log.info("=" * 60)

    # 1. Load
    coverage, gaps_data, enrichment, anomaly_stats = load_inputs(output_dir)

    # 2. Quantify gaps across dimensions
    domain_gaps = quantify_domain_gaps(gaps_data)
    node_gaps = quantify_node_type_gaps(coverage)
    language_gaps = quantify_language_gaps(coverage, {})  # Will load code_stats if available
    anomaly_gap = quantify_anomaly_gaps(anomaly_stats)

    # 3. Composite gap severity
    total_uncovered = sum(d["total_uncovered_size"] for d in domain_gaps)
    n_critical = sum(1 for d in domain_gaps if d["severity"] == "critical")
    n_high_node = sum(1 for n in node_gaps if n["severity"] in ("high", "medium"))

    overall_severity = (
        "critical" if n_critical >= 3 or total_uncovered > 10000
        else "high" if n_critical >= 1 or total_uncovered > 5000
        else "medium" if total_uncovered > 1000
        else "low"
    )

    # 4. Save
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "overall_severity": overall_severity,
        "summary": {
            "total_uncovered_size": total_uncovered,
            "n_critical_domain_gaps": n_critical,
            "n_domains_with_gaps": len(domain_gaps),
            "n_node_type_gaps": n_high_node,
            "n_language_gaps": len(language_gaps),
            "anomaly_gap_interpretation": anomaly_gap,
        },
        "domain_gaps": domain_gaps,
        "node_type_gaps": node_gaps,
        "language_gaps": language_gaps,
        "anomaly_gap": anomaly_gap,
    }

    with open(output_dir / "B4" / "gap_quantification.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info("  ✓ B4_gap_quantification.json")

    log.info(
        "Gap severity: %s | %d uncovered clusters across %d domains",
        overall_severity.upper(),
        gaps_data.get("n_uncovered", 0),
        len(domain_gaps),
    )

    log.info("B4 gap quantification complete.")


if __name__ == "__main__":
    main()
