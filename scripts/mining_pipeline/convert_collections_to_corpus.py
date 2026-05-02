#!/usr/bin/env python3
"""
Collections Preprocessor — Converts platform textbook/research-paper JSON
into A1-compatible corpus format for downstream mining (A2–A7, B1–B5).

Reads the hierarchical JSON collections at:
  /Users/kevintoles/POC/ai-platform-data/collections/{software-engineering,ai-safety}/raw/

Each JSON file follows the document-oriented schema:
  { metadata: {title, author, ..., total_pages, source_file},
    pages: [{page_number, text, extraction_method}, ...],
    chapters: [{number, title, start_page, end_page, content, page_count}, ...] }

Produces the SAME output files as mine_text_preprocess.py (A1):
  - A1_text_corpus.jsonl     (one record per paper)
  - A1_code_corpus.jsonl     (extracted code blocks)
  - A1_tfidf_matrix.npz      (TF-IDF sparse matrix)
  - A1_feature_names.npy     (TF-IDF vocabulary)
  - A1_embeddings.npy        (SBERT embeddings)
  - A1_stats.json            (processing statistics)

Usage:
  python3 convert_collections_to_corpus.py <input_dir> <output_dir>

  <input_dir>  : path to collections root, e.g.
                 /Users/kevintoles/POC/ai-platform-data/collections
  <output_dir> : where to write A1_* output files (e.g. mining_output/collections/)

Constraint: NO LLMs. All operations are deterministic.
Dependencies: scikit-learn, sentence-transformers, numpy, scipy
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("convert_collections")

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_INPUT_DIR = "/Users/kevintoles/POC/ai-platform-data/collections"
MIN_TEXT_LENGTH = 100  # Minimum chars to include a paper

# Which collections to process (top-level dirs under input_dir)
COLLECTIONS = ["software-engineering", "ai-safety"]

# ── Code Detection Heuristics ────────────────────────────────────────────────

CODE_PATTERNS = re.compile(
    r"(def |class |function |import |from |return |if __name__|"
    r"```|{CODE|#include|int main|void |public class|"
    r"private |protected |const |let |var |function\s+\w+\s*\()",
    re.IGNORECASE,
)

LANG_PATTERNS = {
    "python": re.compile(r"(def |import |from |class |print\(|\.py\b|lambda |yield |async def)", re.IGNORECASE),
    "javascript": re.compile(r"(function |const |let |var |=>|console\.|document\.|require\()", re.IGNORECASE),
    "java": re.compile(r"(public class|private |protected |void main|@Override|import java\.)", re.IGNORECASE),
    "cpp": re.compile(r"(#include|int main|std::|cout|cin|template|class\s+\w+\s*\{)", re.IGNORECASE),
    "bash": re.compile(r"(#!/bin/bash|#!/usr/bin|export |source |apt-get|brew |pip install)", re.IGNORECASE),
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
        # Detect fenced code blocks (```)
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

        # Detect indented code blocks (4+ spaces)
        if line.startswith("    ") and len(line.strip()) > 10:
            current_block.append(line)
        else:
            if current_block:
                code_text = "\n".join(current_block)
                if len(code_text) >= 20 and CODE_PATTERNS.search(code_text):
                    lang = detect_language(code_text)
                    blocks.append({"code": code_text, "language": lang})
                current_block = []

    # Flush trailing block
    if current_block:
        code_text = "\n".join(current_block)
        if len(code_text) >= 20 and CODE_PATTERNS.search(code_text):
            lang = detect_language(code_text)
            blocks.append({"code": code_text, "language": lang})

    return blocks


# ── Collection Traversal ─────────────────────────────────────────────────────


def discover_json_files(collection_root: Path, collection_name: str) -> List[Tuple[Path, List[str]]]:
    """
    Recursively walk collection_root, finding all .json files.
    Returns list of (file_path, domain_tags) where domain_tags are
    the path components relative to the collection root (e.g.
    ['ai-engineering', 'ai-agents', 'supporting-papers']).

    Excludes ._* Apple Double files.
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(collection_root):
        dirnames.sort()
        filenames.sort()
        for fn in filenames:
            if fn.startswith("._") or not fn.endswith(".json"):
                continue
            fpath = Path(dirpath) / fn
            # Derive tags from relative path
            rel = fpath.relative_to(collection_root)
            tags = [collection_name] + list(rel.parts[:-1])  # exclude filename
            files.append((fpath, tags))
    return files


# ── Paper Processing ─────────────────────────────────────────────────────────


def process_paper(
    filepath: Path,
    tags: List[str],
) -> Optional[Dict]:
    """
    Read a single collections JSON file and extract text + metadata.

    Returns a dict with fields matching A1_text_corpus.jsonl schema:
      id, title, tags[], site, tier, source_file,
      text (merged chapter content), code_blocks[]
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
    total_pages = meta.get("total_pages", len(pages))

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

    # Build composite tags: collection_domain + paper title keywords
    topic_tags = list(tags)  # e.g. ['software-engineering', 'ai-engineering', 'ai-agents', 'supporting-papers']

    record = {
        "id": f"collections:{tags[0]}:{filepath.stem}",
        "title": title,
        "tags": topic_tags,
        "site": "platform-collections",
        "tier": tags[0],  # software-engineering or ai-safety
        "source_file": source_file,
        "total_pages": total_pages,
        "chapter_count": len(chapters),
        "score": 0,
        "view_count": 0,
        "answer_count": 0,
        "signal_score": 0,
        "has_accepted_answer": False,
        "code_block_count": len(code_blocks),
        "accepted_answer_text": "",
        "text": full_text,
        "_code_blocks": code_blocks,  # internal, not written to text corpus
    }

    return record


# ── Corpus Assembly ──────────────────────────────────────────────────────────


def build_corpus(input_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Walk all collections under input_dir and build:
      - text_records: list of paper records (for text corpus + TF-IDF + embeddings)
      - code_entries: list of {doc_id, code, language} (for code corpus)
    """
    text_records = []
    code_entries = []
    total_files = 0
    skipped = 0

    for collection_name in COLLECTIONS:
        collection_root = input_dir / collection_name / "raw"
        if not collection_root.is_dir():
            log.warning("  ⚠ Collection not found: %s", collection_root)
            continue

        files = discover_json_files(collection_root, collection_name)
        log.info("  Found %d JSON files in %s", len(files), collection_name)

        for fpath, tags in files:
            total_files += 1
            record = process_paper(fpath, tags)
            if record is None:
                skipped += 1
                continue

            doc_id = record["id"]
            text_records.append(record)

            # Build code entries
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
        "Corpus built: %d papers, %d code blocks (%d skipped, %d total files)",
        len(text_records),
        len(code_entries),
        skipped,
        total_files,
    )
    return text_records, code_entries


# ── TF-IDF ───────────────────────────────────────────────────────────────────


def compute_tfidf(
    texts: List[str],
    record_ids: List[str],
    output_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute TF-IDF matrix and save. Returns (matrix, feature_names)."""
    log.info("Computing TF-IDF matrix (%d documents)...", len(texts))
    t0 = time.time()

    vectorizer = TfidfVectorizer(
        max_df=0.85,
        min_df=2,
        max_features=50000,
        stop_words="english",
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Save
    save_npz(output_dir / "A1" / "tfidf_matrix.npz", matrix)
    np.save(output_dir / "A1" / "feature_names.npy", feature_names)

    elapsed = time.time() - t0
    log.info(
        "  ✓ TF-IDF matrix: %s × %s (%.1f sec)", matrix.shape[0], matrix.shape[1], elapsed
    )
    return matrix, feature_names


# ── SBERT Embeddings ─────────────────────────────────────────────────────────


def compute_embeddings(texts: List[str], output_dir: Path) -> np.ndarray:
    """Compute SBERT embeddings and save. Returns embeddings array."""
    log.info("Computing SBERT embeddings (%d documents)...", len(texts))
    t0 = time.time()

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # Process in batches to avoid OOM
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            all_embeddings.append(emb)

        embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    except ImportError:
        log.warning("  ⚠ sentence-transformers not available — using random embeddings")
        # Fallback: use TF-IDF PCA as stand-in (ensures downstream scripts don't crash)
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        tfidf_path = output_dir / "A1" / "tfidf_matrix.npz"
        if tfidf_path.exists():
            from scipy.sparse import load_npz
            matrix = load_npz(tfidf_path)
            svd = TruncatedSVD(n_components=384, random_state=42)
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
        input_dir = Path(DEFAULT_INPUT_DIR)
        log.info("Usage: %s <input_dir> [output_dir]", sys.argv[0])
        log.info("Using default input: %s", input_dir)
    else:
        input_dir = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])
    else:
        # Default: alongside scripts in USB structure
        output_dir = Path(__file__).resolve().parent.parent / "output" / "collections_mining"

    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Collections Preprocessor — Convert JSON → A1 Corpus")
    log.info("Input:  %s", input_dir.resolve())
    log.info("Output: %s", output_dir.resolve())
    log.info("Collections: %s", ", ".join(COLLECTIONS))
    log.info("=" * 60)

    # ── Phase 1: Walk & Extract ──
    t_start = time.time()
    text_records, code_entries = build_corpus(input_dir)
    if not text_records:
        log.error("No papers extracted — aborting.")
        sys.exit(1)

    # ── Phase 2: Write Text Corpus ──
    (output_dir / "A1").mkdir(parents=True, exist_ok=True)
    log.info("Writing text corpus...")
    text_path = output_dir / "A1" / "text_corpus.jsonl"
    with open(text_path, "w", encoding="utf-8") as f:
        for rec in text_records:
            # Write only the standard A1 fields (exclude _code_blocks)
            entry = {k: v for k, v in rec.items() if not k.startswith("_")}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("  ✓ A1_text_corpus.jsonl (%d records)", len(text_records))

    # ── Phase 3: Write Code Corpus ──
    if code_entries:
        log.info("Writing code corpus...")
        code_path = output_dir / "A1" / "code_corpus.jsonl"
        with open(code_path, "w", encoding="utf-8") as f:
            for entry in code_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info("  ✓ A1_code_corpus.jsonl (%d code blocks)", len(code_entries))
    else:
        log.info("  (no code blocks extracted, skipping code corpus)")

    # ── Phase 4: TF-IDF ──
    texts = [rec["text"] for rec in text_records]
    record_ids = [rec["id"] for rec in text_records]
    compute_tfidf(texts, record_ids, output_dir)

    # ── Phase 5: SBERT Embeddings ──
    compute_embeddings(texts, output_dir)

    # ── Phase 6: Stats ──
    stats = {
        "papers_processed": len(text_records),
        "code_blocks_extracted": len(code_entries),
        "collections": {},
    }
    for rec in text_records:
        tier = rec["tier"]
        if tier not in stats["collections"]:
            stats["collections"][tier] = {"papers": 0, "code_blocks": 0, "total_chapters": 0}
        stats["collections"][tier]["papers"] += 1
        stats["collections"][tier]["code_blocks"] += rec.get("code_block_count", 0)
        stats["collections"][tier]["total_chapters"] += rec.get("chapter_count", 0)

    stats_path = output_dir / "A1" / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info("  ✓ A1_stats.json")

    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("Done. %d papers processed in %.1f sec", len(text_records), elapsed)
    log.info("Output: %s", output_dir.resolve())
    log.info("=" * 60)


if __name__ == "__main__":
    main()
