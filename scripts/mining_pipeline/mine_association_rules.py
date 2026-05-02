#!/usr/bin/env python3
"""
Mining Step A4: Association Rule Mining
========================================
Discovers co-occurrence patterns between SE tags, sites, and LDA topics
using the Apriori algorithm from mlxtend.

Mining target: "When tags [A,B] appear together, topic X is likely" or
              "site X questions with tag Y frequently have accepted Z code patterns"

Dependencies: mlxtend, numpy
Input:  A1_text_corpus.jsonl, A2_cluster_labels.jsonl, A3_doc_topics.jsonl
Output: A4_association_rules.json
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("mine_association_rules")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "mining"


def load_all(output_dir: Path) -> Tuple[List[dict], Optional[List[dict]], Optional[List[dict]]]:
    """Load A1, A2, A3 outputs."""
    # A1 corpus with metadata
    corpus = []
    cp = output_dir / "A1" / "text_corpus.jsonl"
    if cp.exists():
        with open(cp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    corpus.append(json.loads(line))
    log.info("Loaded A1 corpus: %d records", len(corpus))

    # A2 cluster labels
    clusters = []
    cp2 = output_dir / "A2" / "cluster_labels.jsonl"
    if cp2.exists():
        with open(cp2, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    clusters.append(json.loads(line))
    log.info("Loaded A2 clusters: %d records", len(clusters))

    # A3 document topics
    doc_topics = []
    cp3 = output_dir / "A3" / "doc_topics.jsonl"
    if cp3.exists():
        with open(cp3, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc_topics.append(json.loads(line))
    log.info("Loaded A3 topics: %d records", len(doc_topics))

    return corpus, clusters if clusters else None, doc_topics if doc_topics else None


def build_transaction_sets(
    corpus: List[dict],
    clusters: Optional[List[dict]],
    doc_topics: Optional[List[dict]],
    min_tag_freq: int = 50,
) -> List[List[str]]:
    """
    Build transaction sets for Apriori.
    Each transaction = itemset of (tag_, site_, cluster_, topic_, quality_) features.
    """
    # Count tag frequencies first
    tag_counts = defaultdict(int)
    for rec in corpus:
        for tag in rec.get("tags", []):
            tag_counts[tag] += 1

    # Keep only frequent tags
    frequent_tags = {t for t, c in tag_counts.items() if c >= min_tag_freq}
    log.info("Frequent tags (>=%d occurrences): %d / %d", min_tag_freq, len(frequent_tags), len(tag_counts))

    transactions = []
    for i, rec in enumerate(corpus):
        items = []

        # Tags (prefixed)
        for tag in rec.get("tags", []):
            if tag in frequent_tags:
                items.append(f"tag:{tag}")

        # Site
        site = rec.get("site", "")
        if site:
            items.append(f"site:{site}")

        # Tier
        tier = rec.get("tier", "")
        if tier:
            items.append(f"tier:{tier}")

        # Score bucket
        score = rec.get("score", 0)
        if score >= 100:
            items.append("score:high")
        elif score >= 10:
            items.append("score:medium")
        elif score > 0:
            items.append("score:low")
        else:
            items.append("score:zero")

        # Has accepted answer
        if rec.get("has_accepted_answer"):
            items.append("accepted_answer")

        # Has code
        code_count = rec.get("code_block_count", 0)
        if code_count > 0:
            items.append("has_code")
        if code_count >= 3:
            items.append("code_rich")

        # K-Means cluster (from A2)
        if clusters and i < len(clusters):
            kc = clusters[i].get("kmeans_cluster")
            if kc is not None:
                items.append(f"km_cluster:{kc}")

        # Dominant LDA topic (from A3)
        if doc_topics and i < len(doc_topics):
            dt = doc_topics[i].get("dominant_topic")
            if dt is not None and dt >= 0:
                items.append(f"lda_topic:{dt}")

        # Signal score bucket
        signal = rec.get("signal_score", 0)
        if signal > 0:
            if signal >= 0.8:
                items.append("signal:high")
            elif signal >= 0.5:
                items.append("signal:medium")
            else:
                items.append("signal:low")

        if len(items) >= 2:
            transactions.append(items)

    log.info("Built %d transactions (avg items: %.1f)", len(transactions),
             sum(len(t) for t in transactions) / max(len(transactions), 1))
    return transactions


def mine_rules(
    transactions: List[List[str]], min_support: float = 0.01, min_confidence: float = 0.5
) -> List[dict]:
    """Run Apriori and extract association rules."""
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    log.info("Running Apriori: min_support=%.3f, min_confidence=%.3f", min_support, min_confidence)

    # Encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    import pandas as pd

    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Apriori
    log.info("  Computing frequent itemsets...")
    frequent = apriori(df, min_support=min_support, use_colnames=True, max_len=4)
    log.info("  Found %d frequent itemsets", len(frequent))

    if len(frequent) < 2:
        log.warning("  Too few frequent itemsets — lowering min_support")
        return []

    # Association rules
    log.info("  Computing association rules...")
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence, num_itemsets=len(frequent))

    # Filter and sort by lift
    rules = rules.sort_values("lift", ascending=False)

    # Convert to serializable dicts
    result = []
    for _, row in rules.head(500).iterrows():
        result.append(
            {
                "antecedents": list(row["antecedents"]),
                "consequents": list(row["consequents"]),
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
                "leverage": float(row.get("leverage", 0)),
                "conviction": float(row.get("conviction", 0)),
            }
        )

    log.info("  Extracted %d rules (top 500 kept)", len(rules))
    return result


def analyze_rules(rules: List[dict]) -> dict:
    """Analyze rule patterns."""
    if not rules:
        return {"total_rules": 0}

    # Count rules by type
    tag_to_topic = 0
    tag_to_tag = 0
    tag_to_cluster = 0
    site_to_tag = 0

    for r in rules:
        ant = set(r["antecedents"])
        con = set(r["consequents"])

        has_tag_ant = any(a.startswith("tag:") for a in ant)
        has_topic_con = any(c.startswith("lda_topic:") for c in con)
        has_tag_con = any(c.startswith("tag:") for c in con)
        has_cluster_con = any(c.startswith("km_cluster:") for c in con)
        has_site_ant = any(a.startswith("site:") for a in ant)

        if has_tag_ant and has_topic_con:
            tag_to_topic += 1
        if has_tag_ant and has_tag_con:
            tag_to_tag += 1
        if has_tag_ant and has_cluster_con:
            tag_to_cluster += 1
        if has_site_ant and has_tag_con:
            site_to_tag += 1

    return {
        "total_rules": len(rules),
        "tag_to_topic_rules": tag_to_topic,
        "tag_to_tag_rules": tag_to_tag,
        "tag_to_cluster_rules": tag_to_cluster,
        "site_to_tag_rules": site_to_tag,
        "top_10_lift": [
            {
                "antecedents": r["antecedents"],
                "consequents": r["consequents"],
                "lift": r["lift"],
                "confidence": r["confidence"],
            }
            for r in rules[:10]
        ],
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR
    min_support = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
    min_confidence = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    log.info("=" * 60)
    log.info("Mining Step A4: Association Rule Mining")
    log.info("Reads intermediates from: %s", output_dir)
    log.info("Output: %s", output_dir)
    log.info("=" * 60)

    # 1. Load all inputs
    corpus, clusters, doc_topics = load_all(output_dir)
    if not corpus:
        log.error("No corpus. Run A1 first.")
        sys.exit(1)

    # 2. Build transaction sets
    transactions = build_transaction_sets(corpus, clusters, doc_topics)
    if not transactions:
        log.error("No transactions built.")
        sys.exit(1)

    # 3. Mine rules
    rules = mine_rules(transactions, min_support=min_support, min_confidence=min_confidence)

    # 4. Analyze
    analysis = analyze_rules(rules)

    # 5. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = {"rules": rules, "analysis": analysis, "parameters": {"min_support": min_support, "min_confidence": min_confidence, "n_transactions": len(transactions)}}

    with open(output_dir / "A4" / "association_rules.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    log.info("  ✓ A4_association_rules.json (%d rules)", len(rules))

    log.info("A4 association rule mining complete.")


if __name__ == "__main__":
    main()
