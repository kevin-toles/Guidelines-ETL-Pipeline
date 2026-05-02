#!/usr/bin/env python3
"""
Mining Step A3: Topic Modeling (LDA)
=====================================
Performs Latent Dirichlet Allocation on the tokenized SE corpus to discover
latent topic distributions per document and top terms per topic.

Uses gensim's LdaMulticore for scalability on large corpora.

Dependencies: gensim, numpy
Input:  A1_text_corpus.jsonl (tokenized texts), A1_tfidf_matrix.npz
Output: A3_topic_model.json, A3_doc_topics.jsonl
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("mine_topic_modeling")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "mining"


def load_corpus(output_dir: Path) -> List[List[str]]:
    """Load tokenized texts from A1 output."""
    corpus_path = output_dir / "A1" / "text_corpus.jsonl"
    log.info("Loading tokenized corpus from %s", corpus_path)
    documents = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tokens = rec.get("tokens", [])
            if tokens:
                documents.append(tokens)
    log.info("  Loaded %d documents with tokens", len(documents))
    return documents


def build_gensim_corpus(documents: List[List[str]], no_below: int = 10, no_above: float = 0.5):
    """Build gensim dictionary and bag-of-words corpus."""
    import gensim.corpora as corpora
    from gensim.models import LdaMulticore

    log.info("Building gensim dictionary (no_below=%d, no_above=%.2f)...", no_below, no_above)
    dictionary = corpora.Dictionary(documents)
    # Filter extremes
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    dictionary.compactify()

    log.info("  Dictionary size: %d", len(dictionary))
    corpus_bow = [dictionary.doc2bow(doc) for doc in documents]
    log.info("  Corpus BOW: %d documents", len(corpus_bow))
    return dictionary, corpus_bow


def run_lda(
    corpus_bow, dictionary, num_topics: int = 50, passes: int = 10, alpha: str = "auto"
) -> Tuple[object, Dict]:
    """Run LDA and collect results."""
    from gensim.models import LdaMulticore

    log.info("Running LDA: num_topics=%d, passes=%d, alpha=%s...", num_topics, passes, alpha)

    model = LdaMulticore(
        corpus=corpus_bow,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        alpha=alpha,
        eta="auto",
        random_state=42,
        workers=4,
        chunksize=2000,
    )

    # Extract top terms per topic
    top_terms = {}
    for topic_id in range(num_topics):
        terms = model.get_topic_terms(topic_id, topn=20)
        top_terms[topic_id] = [(dictionary[term_id], float(weight)) for term_id, weight in terms]

    # Extract per-document topic distributions
    doc_topics = []
    for i, bow in enumerate(corpus_bow):
        topics = model.get_document_topics(bow, minimum_probability=0.01)
        doc_topics.append(topics)

    # Coherence scores (if available)
    coherence_cv = None
    coherence_umass = None
    try:
        from gensim.models.coherencemodel import CoherenceModel

        coherence_cv_model = CoherenceModel(
            model=model, corpus=corpus_bow, dictionary=dictionary, coherence="c_v"
        )
        coherence_cv = float(coherence_cv_model.get_coherence())

        coherence_umass_model = CoherenceModel(
            model=model, corpus=corpus_bow, dictionary=dictionary, coherence="u_mass"
        )
        coherence_umass = float(coherence_umass_model.get_coherence())
        log.info("  Coherence: c_v=%.4f, u_mass=%.4f", coherence_cv, coherence_umass)
    except Exception as e:
        log.warning("  Coherence computation failed: %s", e)

    results = {
        "num_topics": num_topics,
        "passes": passes,
        "alpha": alpha,
        "dictionary_size": len(dictionary),
        "coherence_cv": coherence_cv,
        "coherence_umass": coherence_umass,
        "top_terms": {str(k): v for k, v in top_terms.items()},
    }

    log.info("LDA complete: %d topics", num_topics)
    return model, results, doc_topics


def save_outputs(results: Dict, doc_topics: List, output_dir: Path, corpus_size: int):
    """Save topic model and per-document topic distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # A3_topic_model.json
    with open(output_dir / "A3" / "topic_model.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("  ✓ A3_topic_model.json")

    # A3_doc_topics.jsonl
    with open(output_dir / "A3" / "doc_topics.jsonl", "w", encoding="utf-8") as f:
        for i, topics in enumerate(doc_topics):
            entry = {
                "doc_index": i,
                "topics": [{"topic_id": int(t[0]), "probability": float(t[1])} for t in topics],
                "dominant_topic": int(max(topics, key=lambda x: x[1])[0]) if topics else -1,
                "topic_entropy": float(
                    -sum(p * np.log(max(p, 1e-10)) for _, p in topics)
                ) if topics else 0.0,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("  ✓ A3_doc_topics.jsonl (%d records)", len(doc_topics))

    # A3_stats.json
    topic_counts = defaultdict(int)
    for entry in [
        {
            "dominant_topic": int(max(topics, key=lambda x: x[1])[0]) if topics else -1,
        }
        for topics in doc_topics
    ]:
        topic_counts[entry["dominant_topic"]] += 1

    stats = {"n_documents": corpus_size, "topic_document_counts": dict(topic_counts)}
    with open(output_dir / "A3" / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info("  ✓ A3_stats.json")


from collections import defaultdict


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR
    num_topics = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    log.info("=" * 60)
    log.info("Mining Step A3: Topic Modeling (LDA)")
    log.info("Reads intermediates from: %s", output_dir)
    log.info("Output: %s", output_dir)
    log.info("Topics: %d", num_topics)
    log.info("=" * 60)

    # 1. Load tokenized corpus
    documents = load_corpus(output_dir)
    if not documents:
        log.error("No valid documents found. Run A1 first.")
        sys.exit(1)

    # 2. Build gensim dictionary and BOW corpus
    dictionary, corpus_bow = build_gensim_corpus(documents)

    # 3. Run LDA
    model, results, doc_topics = run_lda(corpus_bow, dictionary, num_topics=num_topics)

    # 4. Save outputs
    save_outputs(results, doc_topics, output_dir, len(documents))

    log.info("A3 topic modeling complete.")


if __name__ == "__main__":
    main()
