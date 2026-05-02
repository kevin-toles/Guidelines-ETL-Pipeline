#!/usr/bin/env python3
"""
Mining Step A5: Anomaly Detection
==================================
Detects anomalous SE records using:
  - Isolation Forest (unsupervised outlier detection)
  - One-Class SVM (novelty detection)
  - Statistical baselines: Z-score on feature vectors

Anomalies = records that differ significantly from the corpus norm, indicating:
  - Undiscovered topic areas (high-value outliers)
  - Noise/spam (low-value outliers)
  - Signal gaps where guidelines may be missing

Dependencies: scikit-learn, numpy
Input:  A1_embeddings.npy, A1_text_corpus.jsonl, A2_cluster_stats.json
Output: A5_anomaly_scores.jsonl, A5_anomaly_stats.json
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("mine_anomaly_detection")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "mining"


def load_inputs(output_dir: Path) -> Tuple[Optional[np.ndarray], List[dict]]:
    """Load embeddings and corpus."""
    emb_path = output_dir / "A1" / "embeddings.npy"
    embeddings = None
    if emb_path.exists():
        embeddings = np.load(emb_path)
        log.info("Loaded embeddings: %s", embeddings.shape)

    corpus = []
    cp = output_dir / "A1" / "text_corpus.jsonl"
    if cp.exists():
        with open(cp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))
    log.info("Loaded corpus: %d records", len(corpus))
    return embeddings, corpus


# ── Isolation Forest ─────────────────────────────────────────────────────────


def isolation_forest_detect(X: np.ndarray, contamination: float = 0.05, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Run Isolation Forest.
    Returns (labels: 1=inlier, -1=outlier, anomaly_scores, stats).
    """
    log.info("Running Isolation Forest (contamination=%.3f)...", contamination)
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    labels = iso.fit_predict(X)
    scores = iso.score_samples(X)
    n_anomalies = int(np.sum(labels == -1))
    log.info("  Anomalies: %d / %d (%.2f%%)", n_anomalies, len(labels), 100 * n_anomalies / len(labels))

    stats = {
        "algorithm": "IsolationForest",
        "contamination": contamination,
        "n_estimators": 200,
        "n_anomalies": n_anomalies,
        "anomaly_ratio": float(n_anomalies / len(labels)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
    }
    return labels, scores, stats


# ── One-Class SVM ────────────────────────────────────────────────────────────


def one_class_svm_detect(X: np.ndarray, nu: float = 0.05, subsample: int = 20000) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Run One-Class SVM (subsampled for scalability).
    Returns (labels, decision_scores, stats).
    """
    log.info("Running One-Class SVM (nu=%.3f, subsample=%d)...", nu, subsample)

    # Subsample for training if too large
    if X.shape[0] > subsample:
        indices = np.random.choice(X.shape[0], subsample, replace=False)
        X_train = X[indices]
    else:
        X_train = X

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    svm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale", max_iter=5000)
    svm.fit(X_scaled)

    # Predict on ALL data (batch to manage memory)
    X_all = scaler.transform(X)
    labels = svm.predict(X_all)
    scores = svm.decision_function(X_all)

    n_anomalies = int(np.sum(labels == -1))
    log.info("  Anomalies: %d / %d (%.2f%%)", n_anomalies, len(labels), 100 * n_anomalies / len(labels))

    stats = {
        "algorithm": "OneClassSVM",
        "nu": nu,
        "n_anomalies": n_anomalies,
        "anomaly_ratio": float(n_anomalies / len(labels)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
    }
    return labels, scores, stats


# ── Z-Score Baseline ─────────────────────────────────────────────────────────


def zscore_detect(X: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Simple Z-score anomaly detection baseline.
    Per-sample: ||z-normalized vector|| > threshold.
    """
    log.info("Running Z-score baseline (threshold=%.1f)...", threshold)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-10] = 1.0  # Avoid div by zero

    z_scores = np.abs((X - mean) / std)
    anomaly_scores = z_scores.max(axis=1)  # Max z-score per sample
    labels = np.where(anomaly_scores > threshold, -1, 1)

    n_anomalies = int(np.sum(labels == -1))
    log.info("  Anomalies: %d / %d (%.2f%%)", n_anomalies, len(labels), 100 * n_anomalies / len(labels))

    stats = {
        "algorithm": "ZScore",
        "threshold": threshold,
        "n_anomalies": n_anomalies,
        "anomaly_ratio": float(n_anomalies / len(labels)),
        "score_min": float(anomaly_scores.min()),
        "score_max": float(anomaly_scores.max()),
        "score_mean": float(anomaly_scores.mean()),
        "score_std": float(anomaly_scores.std()),
    }
    return labels, anomaly_scores, stats


# ── Consensus ────────────────────────────────────────────────────────────────


def compute_consensus(labels_list: List[np.ndarray]) -> np.ndarray:
    """
    Consensus: record is anomalous if 2+ of the 3 algorithms flag it.
    """
    stacked = np.stack(labels_list, axis=0)  # (3, n_samples)
    n_negative = np.sum(stacked == -1, axis=0)  # Per-sample count
    consensus = np.where(n_negative >= 2, -1, 1)
    return consensus, n_negative


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR

    log.info("=" * 60)
    log.info("Mining Step A5: Anomaly Detection")
    log.info("Reads intermediates from: %s", output_dir)
    log.info("Output: %s", output_dir)
    log.info("=" * 60)

    # 1. Load
    embeddings, corpus = load_inputs(output_dir)
    if embeddings is None or embeddings.size == 0:
        log.error("No embeddings found. Run A1 first.")
        sys.exit(1)

    X = embeddings

    # 2. Run all three detectors
    iso_labels, iso_scores, iso_stats = isolation_forest_detect(X)
    svm_labels, svm_scores, svm_stats = one_class_svm_detect(X)
    z_labels, z_scores, z_stats = zscore_detect(X)

    # 3. Consensus
    consensus_labels, n_negative = compute_consensus([iso_labels, svm_labels, z_labels])
    n_consensus_anomalies = int(np.sum(consensus_labels == -1))
    log.info(
        "Consensus anomalies (2+ algorithms): %d / %d (%.2f%%)",
        n_consensus_anomalies, len(consensus_labels),
        100 * n_consensus_anomalies / len(consensus_labels),
    )

    # 4. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # A5_anomaly_scores.jsonl
    with open(output_dir / "A5" / "anomaly_scores.jsonl", "w", encoding="utf-8") as f:
        for i, rec in enumerate(corpus):
            entry = {
                "id": rec.get("id"),
                "title": rec.get("title", "")[:200],  # Truncate for file size
                "site": rec.get("site", ""),
                "tags": rec.get("tags", []),
                "tier": rec.get("tier"),
                "signal_score": rec.get("signal_score", 0),
                "score": rec.get("score", 0),
                "isolation_forest": {
                    "label": int(iso_labels[i]),
                    "score": float(iso_scores[i]),
                },
                "one_class_svm": {
                    "label": int(svm_labels[i]),
                    "score": float(svm_scores[i]),
                },
                "zscore": {
                    "label": int(z_labels[i]),
                    "score": float(z_scores[i]),
                },
                "consensus": {
                    "label": int(consensus_labels[i]),
                    "n_algorithms_flagging": int(n_negative[i]),
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("  ✓ A5_anomaly_scores.jsonl (%d records)", len(corpus))

    # A5_anomaly_stats.json
    stats = {
        "n_records": len(corpus),
        "isolation_forest": iso_stats,
        "one_class_svm": svm_stats,
        "zscore": z_stats,
        "consensus": {
            "n_anomalies": n_consensus_anomalies,
            "anomaly_ratio": float(n_consensus_anomalies / len(consensus_labels)),
        },
    }
    with open(output_dir / "A5" / "anomaly_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info("  ✓ A5_anomaly_stats.json")

    log.info("A5 anomaly detection complete.")


if __name__ == "__main__":
    main()
