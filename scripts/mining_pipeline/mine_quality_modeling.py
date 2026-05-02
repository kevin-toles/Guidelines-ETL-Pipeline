#!/usr/bin/env python3
"""
Mining Step A7: Quality Signal Modeling
========================================
Builds a composite quality model for each SE record by combining:
  - SE-internal signals (score, view_count, answer_count, accepted answer)
  - Structural signals (code block count, answer ratio, text length)
  - External signals (cross-site citations, tag authority)

No LLMs. Pure statistical modeling with configurable weights.

Dependencies: numpy, scipy
Input:  A1_text_corpus.jsonl
Output: A7_quality_scores.jsonl, A7_quality_model.json
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import rankdata, zscore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("mine_quality_modeling")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "mining"

# ── Feature Extractors ──────────────────────────────────────────────────────


def extract_features(corpus: List[dict]) -> Tuple[np.ndarray, List[str], List[dict]]:
    """
    Extract raw feature vectors from corpus.
    Returns (feature_matrix, feature_names, metadata).
    """
    feature_names = [
        "score",
        "view_count",
        "answer_count",
        "has_accepted_answer",
        "code_block_count",
        "title_length",
        "body_length",
        "tag_count",
        "tier_primary",  # Binary: is primary tier?
        "tier_hacks",
        "tier_supplemental",
    ]

    X = np.zeros((len(corpus), len(feature_names)))
    metadata = []

    for i, rec in enumerate(corpus):
        score = float(rec.get("score", 0) or 0)
        view_count = float(rec.get("view_count", 0) or 0)
        answer_count = float(rec.get("answer_count", 0) or 0)
        has_accepted = 1.0 if rec.get("has_accepted_answer") else 0.0
        code_count = float(rec.get("code_block_count", 0))
        title_len = float(len(rec.get("title", "")))
        body_len = float(len(rec.get("accepted_answer_text", "")))
        tag_count = float(len(rec.get("tags", [])))

        tier = rec.get("tier", "")
        tier_primary = 1.0 if tier == "primary" else 0.0
        tier_hacks = 1.0 if tier == "hacks" else 0.0
        tier_supplemental = 1.0 if tier == "supplemental" else 0.0

        X[i] = [
            score,
            view_count,
            answer_count,
            has_accepted,
            code_count,
            title_len,
            body_len,
            tag_count,
            tier_primary,
            tier_hacks,
            tier_supplemental,
        ]

        metadata.append(
            {
                "id": rec.get("id"),
                "title": rec.get("title", "")[:150],
                "site": rec.get("site", ""),
                "tags": rec.get("tags", []),
                "tier": tier,
            }
        )

    return X, feature_names, metadata


# ── Quality Scoring ──────────────────────────────────────────────────────────


def compute_composite_quality(
    X: np.ndarray, feature_names: List[str], weights: Dict[str, float] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute composite quality score:
    1. Z-score normalize each feature
    2. Apply weights
    3. Sum weighted z-scores
    4. Normalize to [0, 1] via sigmoid
    """
    if weights is None:
        # Default weights — higher = more important for quality
        weights = {
            "score": 0.25,
            "view_count": 0.05,
            "answer_count": 0.10,
            "has_accepted_answer": 0.20,
            "code_block_count": 0.10,
            "title_length": 0.02,
            "body_length": 0.05,
            "tag_count": 0.03,
            "tier_primary": 0.15,
            "tier_hacks": 0.05,
            "tier_supplemental": -0.00,  # Not negative, just zero contribution
        }

    # Z-score each feature (robust: replace NaN/Inf with 0)
    X_z = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        if col.std() > 0:
            X_z[:, j] = (col - col.mean()) / col.std()
        # else: column is constant, leave as 0

    X_z = np.nan_to_num(X_z, nan=0.0, posinf=0.0, neginf=0.0)

    # Weighted sum
    w = np.array([weights.get(fn, 0.0) for fn in feature_names])
    raw_scores = X_z @ w

    # Sigmoid normalize to [0, 1]
    quality_scores = 1.0 / (1.0 + np.exp(-raw_scores))

    # Feature contribution breakdown
    feature_contributions = X_z * w  # Per-feature contribution

    # Quality tier assignment
    tiers = np.where(quality_scores >= 0.8, "excellent",
                     np.where(quality_scores >= 0.6, "good",
                              np.where(quality_scores >= 0.4, "average",
                                       np.where(quality_scores >= 0.2, "below_average", "poor"))))

    # Stats
    stats = {
        "weights": weights,
        "score_mean": float(quality_scores.mean()),
        "score_std": float(quality_scores.std()),
        "score_min": float(quality_scores.min()),
        "score_max": float(quality_scores.max()),
        "score_p25": float(np.percentile(quality_scores, 25)),
        "score_p50": float(np.percentile(quality_scores, 50)),
        "score_p75": float(np.percentile(quality_scores, 75)),
        "score_p90": float(np.percentile(quality_scores, 90)),
        "tier_distribution": {
            t: int(np.sum(tiers_arr == t))
            for t in ["excellent", "good", "average", "below_average", "poor"]
        },
    }

    return quality_scores, feature_contributions, stats


def compute_signal_correlations(X: np.ndarray, quality: np.ndarray, feature_names: List[str]) -> dict:
    """Compute correlation of each input feature with the composite quality score."""
    correlations = {}
    for j, name in enumerate(feature_names):
        if X[:, j].std() > 0:
            corr = float(np.corrcoef(X[:, j], quality)[0, 1])
        else:
            corr = 0.0
        correlations[name] = corr
    return correlations


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR

    log.info("=" * 60)
    log.info("Mining Step A7: Quality Signal Modeling")
    log.info("Reads from: %s", output_dir)
    log.info("Output: %s", output_dir)
    log.info("=" * 60)

    # 1. Load corpus
    corpus_path = output_dir / "A1" / "text_corpus.jsonl"
    if not corpus_path.exists():
        log.error("A1_text_corpus.jsonl not found. Run A1 first.")
        sys.exit(1)

    corpus = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    log.info("Loaded %d records", len(corpus))

    # 2. Extract features
    X, feature_names, metadata = extract_features(corpus)
    log.info("Extracted %d features", len(feature_names))

    # 3. Compute composite quality
    quality_scores, contributions, quality_stats = compute_composite_quality(X, feature_names)

    # 4. Signal correlations
    correlations = compute_signal_correlations(X, quality_scores, feature_names)

    # 5. Save
    output_dir.mkdir(parents=True, exist_ok=True)

    # A7_quality_scores.jsonl
    with open(output_dir / "A7" / "quality_scores.jsonl", "w", encoding="utf-8") as f:
        for i, meta in enumerate(metadata):
            entry = {
                **meta,
                "quality_score": float(quality_scores[i]),
                "quality_tier": str(
                    "excellent" if quality_scores[i] >= 0.8
                    else "good" if quality_scores[i] >= 0.6
                    else "average" if quality_scores[i] >= 0.4
                    else "below_average" if quality_scores[i] >= 0.2
                    else "poor"
                ),
                "feature_contributions": {
                    fn: float(contributions[i, j])
                    for j, fn in enumerate(feature_names)
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("  ✓ A7_quality_scores.jsonl (%d records)", len(corpus))

    # A7_quality_model.json
    model = {
        "n_records": len(corpus),
        "feature_names": feature_names,
        "quality_stats": quality_stats,
        "feature_correlations_with_quality": correlations,
    }
    with open(output_dir / "A7" / "quality_model.json", "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
    log.info("  ✓ A7_quality_model.json")

    log.info("A7 quality signal modeling complete.")


if __name__ == "__main__":
    main()
