#!/usr/bin/env python3
"""
Convert Textbook JSONs → A1 Corpus Format
===========================================
Reads the 8 textbook-json files (PDF-extracted {metadata, pages, chapters} format)
from the textbook-json/ directory and converts them to the standard A1 corpus
format — same output schema as convert_collections_to_corpus.py:

  <output_dir>/A1/text_corpus.jsonl     — cleaned text + metadata
  <output_dir>/A1/code_corpus.jsonl     — extracted code blocks
  <output_dir>/A1/tfidf_matrix.npz      — TF-IDF sparse matrix
  <output_dir>/A1/feature_names.npy     — TF-IDF vocabulary
  <output_dir>/A1/embeddings.npy        — SBERT embeddings
  <output_dir>/A1/stats.json            — processing statistics

Usage:
  python3 convert_textbook_json.py <input_dir> <output_dir>

  <input_dir>  : path to textbook-json/ directory
  <output_dir> : where to write A1/ output files
                 (e.g. /Volumes/.../mining_output/textbook_mining/)

Dependencies: scikit-learn, sentence-transformers, numpy, scipy
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("convert_textbook_json")

# ── Constants ────────────────────────────────────────────────────────────────

MIN_TEXT_LENGTH = 50  # Minimum chars to include a document

# ── Code Detection Heuristics ────────────────────────────────────────────────

CODE_PATTERNS = re.compile(
    r"(def |class |function |import |from |return |if __name__|"
    r"```|{CODE|#include|int main|void |public class|"
    r"private |protected |const |let |var |function\s+\w+\s*\()",
    re.IGNORECASE,
)

LANG_PATTERNS = {
    "python": re.compile(
        r"(def |import |from |class |print\(|\.py\b|lambda |yield |async def)", re.IGNORECASE
    ),
    "javascript": re.compile(
        r"(function |const |let |var |=>|console\.|document\.|require\()", re.IGNORECASE
    ),
    "java": re.compile(
        r"(public class|private |protected |void main|@Override|import java\.)", re.IGNORECASE
    ),
    "cpp": re.compile(
        r"(#include|int main|std::|cout|cin|template|class\s+\w+\s*\{)", re.IGNORECASE
    ),
    "bash": re.compile(
        r"(#!/bin/bash|#!/usr/bin|export |source |apt-get|brew |pip install)", re.IGNORECASE
    ),
}


def detect_language(text: str) -> str:
    """Simple language detection for code blocks."""
    for lang, pattern in LANG_PATTERNS.items():
        if pattern.search(text):
            return lang
    matches = CODE_PATTERNS.findall(text)
    if matches:
        return "unknown"
    return ""


def extract_code_blocks_from_text(text: str) -> List[Dict[str, str]]:
    """Extract code-like blocks from plain text using heuristic patterns."""
    blocks = []
    lines = text.split("\n")
    current_block = []
    in_code = False

    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                code_text = "\n".join(current_block)
                if len(code_text) >= 20:
                    lang = detect_language(code_text)
                    blocks.append({"code": code_text, "language": lang})
                current_block = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            current_block.append(line)
            continue

        if line.startswith("    ") and len(line.strip()) > 10:
            current_block.append(line)
        else:
            if current_block:
                code_text = "\n".join(current_block)
                if len(code_text) >= 20 and CODE_PATTERNS.search(code_text):
                    lang = detect_language(code_text)
                    blocks.append({"code": code_text, "language": lang})
                current_block = []

    if current_block:
        code_text = "\n".join(current_block)
        if len(code_text) >= 20 and CODE_PATTERNS.search(code_text):
            lang = detect_language(code_text)
            blocks.append({"code": code_text, "language": lang})

    return blocks


# ── Document Processing ──────────────────────────────────────────────────────


def process_document(filepath: Path) -> Optional[Dict]:
    """
    Read a single textbook JSON file and extract text + metadata.

    Returns a dict with fields matching A1_text_corpus.jsonl schema:
      id, title, tags[], site, tier, source_file,
      text (merged page content), code_blocks[]
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log.warning("  ⚠ Skipping %s: %s", filepath.name, e)
        return None

    meta = data.get("metadata", {})
    chapters = data.get("chapters", [])
    pages = data.get("pages", [])

    title = meta.get("title", filepath.stem)
    source_file = meta.get("source_file", filepath.name)

    # Merge all chapter content (preferred) or fall back to page text
    if chapters:
        full_text = "\n\n".join(
            ch.get("content", "") for ch in chapters if ch.get("content")
        )
    else:
        full_text = "\n\n".join(
            pg.get("text", "") for pg in pages if pg.get("text")
        )

    if len(full_text) < MIN_TEXT_LENGTH:
        log.debug("  ⚠ Skipping %s: text too short (%d chars)", filepath.name, len(full_text))
        return None

    # Extract code blocks
    code_blocks = extract_code_blocks_from_text(full_text)

    # Derive tier/tags from filename heuristics
    stem = filepath.stem.lower()
    if "codewars" in stem or "code-wars" in stem or "code_wars" in stem:
        tier = "codewars"
    elif "microservice" in stem or "monolith" in stem:
        tier = "architecture"
    elif stem.startswith("leon-solved"):
        tier = "codewars"
    elif stem.isdigit():
        tier = "codewars"
    else:
        tier = "textbook"

    tags = ["textbook-json", tier] if tier != "codewars" else ["textbook-json", "codewars"]

    record = {
        "id": f"textbook:{tier}:{filepath.stem}",
        "title": title,
        "tags": tags,
        "site": "pdf-extraction",
        "tier": tier,
        "source_file": source_file,
        "total_pages": meta.get("total_pages", len(pages)),
        "chapter_count": len(chapters),
        "score": 0,
        "view_count": 0,
        "answer_count": 0,
        "signal_score": 0,
        "has_accepted_answer": False,
        "code_block_count": len(code_blocks),
        "accepted_answer_text": "",
        "text": full_text,
        "_code_blocks": code_blocks,
    }

    return record


# ── Corpus Assembly ──────────────────────────────────────────────────────────


def build_corpus(input_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Walk textbook-json dir and build:
      - text_records: list of document records
      - code_entries: list of {doc_id, code, language}
    """
    text_records = []
    code_entries = []

    # Find all JSON files, skip ._* Apple Double files
    json_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix == ".json" and not f.name.startswith("._")
    )

    log.info("  Found %d JSON files in %s", len(json_files), input_dir)

    for fpath in json_files:
        record = process_document(fpath)
        if record is None:
            continue

        doc_id = record["id"]
        text_records.append(record)

        for cb in record.pop("_code_blocks", []):
            code_entries.append(
                {
                    "doc_id": doc_id,
                    "title": record["title"],
                    "code": cb["code"],
                    "language": cb["language"] or "unknown",
                    "source_file": record["source_file"],
                    "tier": record["tier"],
                    "tags": record["tags"],
                }
            )

    log.info(
        "Corpus built: %d documents, %d code blocks",
        len(text_records),
        len(code_entries),
    )
    return text_records, code_entries


# ── TF-IDF ───────────────────────────────────────────────────────────────────


def compute_tfidf(texts: List[str], output_dir: Path):
    """Compute TF-IDF matrix and save."""
    log.info("Computing TF-IDF matrix (%d documents)...", len(texts))
    t0 = time.time()

    if len(texts) < 2:
        log.warning("  ⚠ Too few documents for meaningful TF-IDF — saving empty matrix")
        # Create an empty matrix to avoid breaking downstream scripts
        matrix = csr_matrix((len(texts), 0))
        feature_names = np.array([])
    else:
        vectorizer = TfidfVectorizer(
            max_df=0.85,
            min_df=1,
            max_features=50000,
            stop_words="english",
            sublinear_tf=True,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

    save_npz(output_dir / "A1" / "tfidf_matrix.npz", matrix)
    np.save(output_dir / "A1" / "feature_names.npy", feature_names)

    elapsed = time.time() - t0
    log.info("  ✓ TF-IDF matrix: %s × %s (%.1f sec)", matrix.shape[0], matrix.shape[1], elapsed)


# ── SBERT Embeddings ─────────────────────────────────────────────────────────


def compute_embeddings(texts: List[str], output_dir: Path) -> np.ndarray:
    """Compute SBERT embeddings and save."""
    log.info("Computing SBERT embeddings (%d documents)...", len(texts))
    t0 = time.time()

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(emb)

        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    except ImportError:
        log.warning("  ⚠ sentence-transformers not available — using TF-IDF SVD fallback")
        from scipy.sparse import load_npz
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        tfidf_path = output_dir / "A1" / "tfidf_matrix.npz"
        if tfidf_path.exists():
            matrix = load_npz(tfidf_path)
            svd = TruncatedSVD(n_components=min(384, matrix.shape[1]), random_state=42)
            embeddings = normalize(svd.fit_transform(matrix), norm="l2")
        else:
            rng = np.random.RandomState(42)
            embeddings = rng.randn(len(texts), 384).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    np.save(output_dir / "A1" / "embeddings.npy", embeddings)
    elapsed = time.time() - t0
    log.info("  ✓ SBERT embeddings: %s (%.1f sec)", str(embeddings.shape), elapsed)
    return embeddings


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 convert_textbook_json.py <input_dir> <output_dir>")
        print()
        print("  <input_dir>  : path to textbook-json/ directory")
        print("  <output_dir> : where to write A1/ output files")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) >= 3 else input_dir.parent / "mining_output" / "textbook_mining"

    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Textbook JSON → A1 Corpus Converter")
    log.info("Input:  %s", input_dir.resolve())
    log.info("Output: %s", output_dir.resolve())
    log.info("=" * 60)

    # ── Phase 1: Walk & Extract ──
    t_start = time.time()
    text_records, code_entries = build_corpus(input_dir)
    if not text_records:
        log.error("No documents extracted — aborting.")
        sys.exit(1)

    # ── Phase 2: Write Text Corpus ──
    (output_dir / "A1").mkdir(parents=True, exist_ok=True)
    log.info("Writing text corpus...")
    text_path = output_dir / "A1" / "text_corpus.jsonl"
    with open(text_path, "w", encoding="utf-8") as f:
        for rec in text_records:
            entry = {k: v for k, v in rec.items() if not k.startswith("_")}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("  ✓ text_corpus.jsonl (%d records)", len(text_records))

    # ── Phase 3: Write Code Corpus ──
    if code_entries:
        log.info("Writing code corpus...")
        code_path = output_dir / "A1" / "code_corpus.jsonl"
        with open(code_path, "w", encoding="utf-8") as f:
            for entry in code_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info("  ✓ code_corpus.jsonl (%d code blocks)", len(code_entries))
    else:
        log.info("  (no code blocks extracted, skipping code corpus)")

    # ── Phase 4: TF-IDF ──
    texts = [rec["text"] for rec in text_records]
    compute_tfidf(texts, output_dir)

    # ── Phase 5: SBERT Embeddings ──
    compute_embeddings(texts, output_dir)

    # ── Phase 6: Stats ──
    elapsed = time.time() - t_start
    stats = {
        "script": "convert_textbook_json",
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "documents_processed": len(text_records),
        "code_blocks_extracted": len(code_entries),
        "total_text_chars": sum(len(rec["text"]) for rec in text_records),
        "elapsed_seconds": round(elapsed, 1),
        "tiers": {},
    }
    for rec in text_records:
        tier = rec["tier"]
        if tier not in stats["tiers"]:
            stats["tiers"][tier] = {"documents": 0, "code_blocks": 0}
        stats["tiers"][tier]["documents"] += 1
        stats["tiers"][tier]["code_blocks"] += rec.get("code_block_count", 0)

    stats_path = output_dir / "A1" / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info("  ✓ stats.json")
    log.info("")
    log.info("=" * 60)
    log.info("DONE — %d docs, %d code blocks (%.1f sec)", len(text_records), len(code_entries), elapsed)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
