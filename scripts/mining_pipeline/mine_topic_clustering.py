#!/usr/bin/env python3
"""
Mining Step A2: Topic Clustering
=================================
Performs K-Means and DBSCAN clustering on TF-IDF and/or SBERT embeddings
computed by A1 to discover latent topic clusters in the SE corpus.

Algorithms:
  - K-Means (k=20..200, silhouette scoring for best k)
  - DBSCAN (eps grid search, min_samples grid)
  - PCA for dimensionality reduction and 2D visualization

Dependencies: scikit-learn, numpy, scipy
Input:  A1_tfidf_matrix.npz, A1_embeddings.npy, A1_text_corpus.jsonl
Output: A2_cluster_labels.jsonl, A2_cluster_stats.json
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import load_npz
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("mine_topic_clustering")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "mining"


def load_inputs(output_dir: Path) -> Tuple[np.ndarray, Optional[np.ndarray], List[dict]]:
    """Load A1 outputs."""
    log.info("Loading A1 outputs from %s", output_dir)

    # TF-IDF matrix
    tfidf = load_npz(output_dir / "A1" / "tfidf_matrix.npz")
    log.info("  TF-IDF: %s", tfidf.shape)

    # Embeddings (optional)
    embeddings_path = output_dir / "A1" / "embeddings.npy"
    embeddings = None
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        log.info("  Embeddings: %s", embeddings.shape)

    # Text corpus metadata
    corpus = []
    corpus_path = output_dir / "A1" / "text_corpus.jsonl"
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
    log.info("  Corpus: %d records", len(corpus))

    return tfidf, embeddings, corpus


# ── K-Means Clustering ───────────────────────────────────────────────────────


def kmeans_cluster(
    X: np.ndarray, k_values: List[int] = None, random_state: int = 42
) -> Tuple[np.ndarray, int, float, dict]:
    """
    Run K-Means for multiple k values and select best by silhouette score.
    Returns (best_labels, best_k, best_silhouette, all_results).
    """
    if k_values is None:
        k_values = [10, 20, 30, 40, 50, 75, 100, 150, 200]

    log.info("Running K-Means for k in %s...", k_values)
    results = {}
    best_k = None
    best_score = -1.0
    best_labels = None

    for k in k_values:
        if k >= X.shape[0]:
            log.warning("  k=%d >= n_samples=%d, skipping", k, X.shape[0])
            continue

        km = KMeans(n_clusters=k, random_state=random_state, n_init=5, max_iter=300)
        labels = km.fit_predict(X)
        inertia = km.inertia_

        # Silhouette on a sample if too large
        if X.shape[0] > 50000:
            indices = np.random.choice(X.shape[0], 50000, replace=False)
            score = silhouette_score(X[indices], labels[indices], sample_size=50000)
        else:
            score = silhouette_score(X, labels)

        results[k] = {"silhouette": float(score), "inertia": float(inertia), "n_clusters": k}
        log.info("  k=%d  silhouette=%.4f  inertia=%.0f", k, score, inertia)

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    log.info("Best K-Means: k=%d, silhouette=%.4f", best_k, best_score)
    return best_labels, best_k, best_score, results


# ── DBSCAN Clustering ────────────────────────────────────────────────────────


def dbscan_cluster(
    X: np.ndarray,
    eps_values: List[float] = None,
    min_samples_values: List[int] = None,
) -> Tuple[np.ndarray, float, float, dict]:
    """
    Grid search over DBSCAN parameters.
    Returns (best_labels, best_eps, best_min_samples, all_results).
    """
    if eps_values is None:
        eps_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    if min_samples_values is None:
        min_samples_values = [3, 5, 10, 20, 50]

    log.info("Running DBSCAN grid search...")
    results = {}
    best_score = -1.0
    best_params = None
    best_labels = None

    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
            labels = db.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))

            # Skip degenerate solutions
            if n_clusters < 2 or n_clusters > 200:
                results.setdefault(str(eps), {})[str(ms)] = {
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "noise_ratio": float(n_noise / len(labels)),
                    "silhouette": None,
                    "status": "degenerate" if n_clusters < 2 else "too_many",
                }
                continue

            # Compute silhouette on non-noise points
            mask = labels != -1
            if mask.sum() > 1:
                # Sample if too large
                X_masked = X[mask]
                if X_masked.shape[0] > 50000:
                    indices = np.random.choice(X_masked.shape[0], 50000, replace=False)
                    score = silhouette_score(X_masked[indices], labels[mask][indices], sample_size=50000)
                else:
                    score = silhouette_score(X_masked, labels[mask])
            else:
                score = -1.0

            results.setdefault(str(eps), {})[str(ms)] = {
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": float(n_noise / len(labels)),
                "silhouette": float(score),
                "status": "ok",
            }
            log.info(
                "  eps=%.1f  ms=%d  clusters=%d  noise=%d (%.1f%%)  silhouette=%.4f",
                eps, ms, n_clusters, n_noise, 100 * n_noise / len(labels), score,
            )

            if score > best_score:
                best_score = score
                best_params = (eps, ms)
                best_labels = labels

    log.info(
        "Best DBSCAN: eps=%.1f, min_samples=%d, silhouette=%.4f",
        best_params[0], best_params[1], best_score,
    )
    return best_labels, best_params[0], best_params[1], results


# ── PCA Reduction ────────────────────────────────────────────────────────────


def pca_reduce(X: np.ndarray, n_components: int = 50) -> Tuple[np.ndarray, PCA, np.ndarray]:
    """Reduce dimensionality with PCA."""
    log.info("Running PCA to %d components...", n_components)
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.cumsum()
    log.info("  PCA %d components explains %.2f%% variance", n_components, explained[-1] * 100)
    return X_pca, pca, explained


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR

    log.info("=" * 60)
    log.info("Mining Step A2: Topic Clustering")
    log.info("Reads intermediates from: %s", output_dir)
    log.info("Output: %s", output_dir)
    log.info("=" * 60)

    # 1. Load inputs
    tfidf, embeddings, corpus = load_inputs(output_dir)

    # 2. Choose feature matrix: prefer embeddings if available, else TF-IDF
    if embeddings is not None and embeddings.size > 0:
        log.info("Using SBERT embeddings for clustering")
        X_raw = embeddings
    else:
        log.info("Using TF-IDF matrix for clustering")
        # Convert sparse to dense safely
        if tfidf.shape[0] > 50000:
            log.info("  Large matrix — using PCA-reduced TF-IDF")
            X_raw = pca_reduce(tfidf, n_components=100)[0]
        else:
            X_raw = tfidf.toarray()

    # 3. PCA reduction for clustering efficiency
    X, pca_model, explained_var = pca_reduce(X_raw, n_components=min(50, X_raw.shape[1], X_raw.shape[0] - 1))

    # 4. K-Means clustering
    kmeans_labels, best_k, kmeans_silhouette, kmeans_results = kmeans_cluster(X)

    # 5. DBSCAN clustering (on PCA-reduced)
    log.info("\n")
    dbscan_labels, best_eps, best_ms, dbscan_results = dbscan_cluster(X)

    # 6. PCA 2D for visualization
    log.info("Computing 2D PCA for visualization...")
    pca_2d = PCA(n_components=2, random_state=42)
    coords_2d = pca_2d.fit_transform(X)

    # 7. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # A2_cluster_labels.jsonl
    with open(output_dir / "A2" / "cluster_labels.jsonl", "w", encoding="utf-8") as f:
        for i, rec in enumerate(corpus):
            entry = {
                "id": rec.get("id"),
                "title": rec.get("title", ""),
                "tags": rec.get("tags", []),
                "site": rec.get("site", ""),
                "tier": rec.get("tier"),
                "kmeans_cluster": int(kmeans_labels[i]),
                "dbscan_cluster": int(dbscan_labels[i]),
                "pca_x": float(coords_2d[i, 0]),
                "pca_y": float(coords_2d[i, 1]),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("  ✓ A2_cluster_labels.jsonl (%d records)", len(corpus))

    # A2_cluster_stats.json
    n_kmeans = len(set(kmeans_labels))
    n_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_dbscan_noise = int(np.sum(dbscan_labels == -1))

    stats = {
        "n_records": len(corpus),
        "kmeans": {
            "best_k": best_k,
            "best_silhouette": float(kmeans_silhouette),
            "n_clusters": n_kmeans,
            "all_results": kmeans_results,
        },
        "dbscan": {
            "best_eps": best_eps,
            "best_min_samples": best_ms,
            "n_clusters": n_dbscan,
            "n_noise": n_dbscan_noise,
            "noise_ratio": float(n_dbscan_noise / len(corpus)),
            "all_results": dbscan_results,
        },
        "pca": {
            "n_components": X.shape[1],
            "explained_variance_ratio": explained_var.tolist(),
        },
    }
    with open(output_dir / "A2" / "cluster_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info("  ✓ A2_cluster_stats.json")

    log.info("A2 clustering complete.")


if __name__ == "__main__":
    main()
