#!/usr/bin/env python3
"""
Mining Step A1: Text Preprocessing Pipeline (Streaming + Checkpoint)
====================================================================
Reads 315 StackExchange JSONL files (primary, hacks, supplemental tiers),
performs deterministic text preprocessing with streaming file-level
processing and per-file checkpointing.

If interrupted (OOM, crash), re-run with --resume to skip completed files
and pick up where you left off.

Output structure (under <output_base>/A1/):

  text_corpus.jsonl     — cleaned text + tokens + metadata (one JSON line per record)
  code_corpus.jsonl     — extracted code blocks (one JSON line per block)
  tfidf_matrix.npz      — TF-IDF document-term sparse matrix
  feature_names.npy     — TF-IDF vocabulary array
  embeddings.npy        — SBERT embeddings (skipped if sentence-transformers absent)
  stats.json            — pipeline statistics
  _batches/             — per-file checkpoint markers (name = <tier>.<filename>.done)

Constraint: NO LLMs. All operations are deterministic.

Dependencies: scikit-learn, sentence-transformers, beautifulsoup4, nltk, pygments
"""

import gc
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("mine_text_preprocess")

# ── Constants ────────────────────────────────────────────────────────────────

SE_INPUT_DIR = "/Volumes/USB321FD/Guidelines ETL Data/ai-platform-output"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "mining"

MIN_TEXT_LENGTH = 50   # Drop records shorter than this after cleaning
SBERT_BATCH_SIZE = 256

# ── HTML Cleaning ────────────────────────────────────────────────────────────


def clean_html(html_text: str) -> str:
    """Strip HTML tags, extract text content only. Preserves code blocks."""
    if not html_text:
        return ""
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup.find_all(["code", "pre"]):
            tag.decompose()
        return soup.get_text(separator=" ").strip()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html_text)
        return re.sub(r"\s+", " ", text).strip()


def extract_code_blocks(html_text: str) -> List[Dict[str, str]]:
    """Extract code blocks from HTML body with language detection."""
    if not html_text:
        return []
    try:
        soup = BeautifulSoup(html_text, "html.parser")
        blocks = []
        for tag in soup.find_all(["code", "pre"]):
            code_text = tag.get_text().strip()
            if len(code_text) >= 10:
                lang = ""
                classes = tag.get("class", [])
                if classes:
                    for cls in classes:
                        if cls.startswith(("lang-", "language-")):
                            lang = cls.split("-", 1)[1]
                            break
                blocks.append({"code": code_text, "language": lang})
        return blocks
    except Exception:
        return []


# ── Tokenization ─────────────────────────────────────────────────────────────


def _ensure_nltk_data():
    """Download NLTK data if missing (idempotent)."""
    import nltk
    for resource, kind in [("punkt", "tokenizers"), ("punkt_tab", "tokenizers"),
                           ("stopwords", "corpora")]:
        try:
            nltk.data.find(f"{kind}/{resource}")
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


def build_stop_words() -> set:
    """Build stop words set including SE-specific noise words."""
    _ensure_nltk_data()
    sw = set(stopwords.words("english"))
    sw.update([
        "question", "answer", "thanks", "help", "please",
        "anyone", "somebody", "problem", "issue", "trying",
        "using", "want", "need", "know", "like", "would",
        "get", "one", "also", "still",
    ])
    return sw


def tokenize_text(text: str, stemmer: PorterStemmer, stop_words: set) -> List[str]:
    """Tokenize, clean, stem, remove stop words."""
    if not text:
        return []
    try:
        tokens = word_tokenize(text.lower())
    except Exception:
        tokens = text.lower().split()
    cleaned = []
    for t in tokens:
        t = t.strip()
        if len(t) < 2 or not t.isalpha() or t in stop_words:
            continue
        cleaned.append(stemmer.stem(t))
    return cleaned


# ── Per-Record Processing ────────────────────────────────────────────────────


def process_record(rec: dict, tier: str, source_file: str,
                   stemmer: PorterStemmer, stop_words: set) -> Optional[dict]:
    """Clean, tokenize, and extract code from one SE record.

    Returns a dict (or None if the record is too short).

    Fields stored in text_corpus.jsonl:
      id, title, tags, site, tier, source_file, score, view_count,
      answer_count, signal_score, has_accepted_answer, code_block_count,
      accepted_answer_text, tokens, _cleaned_text
    """
    combined_html = ""
    code_blocks = []

    title = rec.get("title", "")
    if title:
        combined_html += title + " "

    body_html = rec.get("body_html", rec.get("body", ""))
    if body_html:
        combined_html += body_html + " "
        code_blocks.extend(extract_code_blocks(body_html))

    accepted_text = ""
    for ans in rec.get("answers", []):
        ans_html = ans.get("body_html", ans.get("body", ""))
        if ans_html:
            combined_html += ans_html + " "
            code_blocks.extend(extract_code_blocks(ans_html))
            if ans.get("is_accepted"):
                accepted_text += clean_html(ans_html) + " "

    cleaned_text = clean_html(combined_html)
    if len(cleaned_text) < MIN_TEXT_LENGTH:
        return None

    tokens = tokenize_text(cleaned_text, stemmer, stop_words)

    return {
        "id": rec.get("id"),
        "title": title,
        "tags": rec.get("tags", []),
        "site": rec.get("site", ""),
        "tier": tier,
        "source_file": source_file,
        "score": rec.get("score", 0),
        "view_count": rec.get("view_count", 0),
        "answer_count": rec.get("answer_count", 0),
        "signal_score": rec.get("signal_score", 0),
        "has_accepted_answer": any(a.get("is_accepted") for a in rec.get("answers", [])),
        "code_block_count": len(code_blocks),
        "accepted_answer_text": accepted_text,
        "tokens": tokens,
        "_cleaned_text": cleaned_text,
    }


def extract_code_from_record(rec: dict, tier: str, source_file: str) -> List[dict]:
    """Extract code blocks from one SE record for the code corpus."""
    entries = []
    body_html = rec.get("body_html", rec.get("body", ""))
    for cb in extract_code_blocks(body_html):
        entries.append({
            "question_id": rec.get("id"), "source": "question",
            "language": cb["language"], "code": cb["code"],
            "site": rec.get("site", ""), "tier": tier,
            "question_score": rec.get("score", 0),
        })
    for ans in rec.get("answers", []):
        ans_html = ans.get("body_html", ans.get("body", ""))
        for cb in extract_code_blocks(ans_html):
            entries.append({
                "question_id": rec.get("id"), "answer_id": ans.get("id"),
                "source": "answer", "is_accepted": ans.get("is_accepted", False),
                "language": cb["language"], "code": cb["code"],
                "site": rec.get("site", ""), "tier": tier,
                "answer_score": ans.get("score", 0),
            })
    return entries


# ── File Loading ─────────────────────────────────────────────────────────────


def discover_files(input_dir: str) -> List[Tuple[str, str, Path]]:
    """Discover all JSONL files across tiers.

    Returns sorted list of (tier, source_name, Path) tuples.
    """
    files = []
    for tier in ["primary", "hacks", "supplemental"]:
        tier_dir = os.path.join(input_dir, tier)
        if not os.path.isdir(tier_dir):
            log.warning("Tier dir not found: %s", tier_dir)
            continue
        for jf in sorted(Path(tier_dir).glob("*.jsonl")):
            if jf.name.startswith("._"):
                continue
            files.append((tier, jf.stem, jf))
    return files


def load_jsonl_file(filepath: Path, tier: str) -> List[dict]:
    """Load all records from a single JSONL file."""
    records = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    rec["_tier"] = tier
                    rec["_source_file"] = filepath.name
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log.error("Error reading %s: %s", filepath.name, e)
    return records


# ── TF-IDF Matrix ────────────────────────────────────────────────────────────


def build_tfidf_matrix(texts: List[str]) -> Tuple[csr_matrix, np.ndarray]:
    """Build TF-IDF document-term sparse matrix."""
    vectorizer = TfidfVectorizer(
        max_features=10000, min_df=5, max_df=0.7,
        ngram_range=(1, 2), sublinear_tf=True,
        strip_accents="unicode", stop_words="english",
    )
    tfidf = vectorizer.fit_transform(texts)
    names = vectorizer.get_feature_names_out()
    log.info("TF-IDF: %d × %d, %.2f%% non-zero",
             tfidf.shape[0], tfidf.shape[1],
             tfidf.nnz / (tfidf.shape[0] * tfidf.shape[1]) * 100)
    return tfidf, names


# ── SBERT Embeddings ─────────────────────────────────────────────────────────


def build_embeddings(texts: List[str]) -> np.ndarray:
    """Generate SBERT embeddings. Returns empty array if not available."""
    try:
        from sentence_transformers import SentenceTransformer
        log.info("Loading SBERT model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Embedding %d texts (batch_size=%d)...", len(texts), SBERT_BATCH_SIZE)
        emb = model.encode(texts, batch_size=SBERT_BATCH_SIZE,
                           show_progress_bar=True, normalize_embeddings=True)
        log.info("Embeddings shape: %s", emb.shape)
        return emb
    except ImportError:
        log.warning("sentence-transformers not installed — skipping embeddings")
        return np.array([])


# ── Phase 2: Build final artifacts ────────────────────────────────────────────


def build_final_artifacts(a1_dir: Path):
    """Build TF-IDF, embeddings, and stats after streaming phase.

    Re-reads the text corpus from disk to keep peak memory low.
    Cleans up any partial outputs from a prior Phase-2 OOM before starting.
    """
    log.info("=" * 60)
    log.info("Phase 2: Building corpus-level artifacts")
    log.info("=" * 60)

    text_corpus_path = a1_dir / "text_corpus.jsonl"

    # ── Re-read texts from disk ──────────────────────────────────────────
    texts: List[str] = []
    token_counts: List[int] = []
    code_block_count = 0
    with open(text_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            ct = entry.get("_cleaned_text", "")
            if ct:
                texts.append(ct)
            token_counts.append(len(entry.get("tokens", [])))
            code_block_count += entry.get("code_block_count", 0)
    log.info("  Re-read %d texts from %s", len(texts), text_corpus_path.name)

    if not texts:
        log.error("No texts found in corpus — aborting Phase 2.")
        return

    # ── Clean stale phase-2 outputs (previous OOM may have left partials) ──
    for stale in ["tfidf_matrix.npz", "feature_names.npy",
                   "embeddings.npy", "stats.json"]:
        p = a1_dir / stale
        if p.exists():
            p.unlink()
            log.info("  🧹 Removed stale Phase-2 output: %s", stale)

    tfidf, names = build_tfidf_matrix(texts)
    tfidf_shape = list(tfidf.shape)
    tfidf_nnz = int(tfidf.nnz)

    save_npz(a1_dir / "tfidf_matrix.npz", tfidf)
    np.save(a1_dir / "feature_names.npy", names)
    vocab_size = len(names)

    # Free TF-IDF before loading SBERT model (reduces peak memory)
    tfidf = None
    gc.collect()

    embeddings = build_embeddings(texts)

    # Free texts after embeddings (no further use)
    del texts
    gc.collect()

    if embeddings.size > 0:
        np.save(a1_dir / "embeddings.npy", embeddings)

    stats = {
        "total_records": len(token_counts),
        "total_code_blocks": code_block_count,
        "tfidf_shape": tfidf_shape,
        "tfidf_nnz": tfidf_nnz,
        "embeddings_shape": list(embeddings.shape) if embeddings.size > 0 else [],
        "vocabulary_size": vocab_size,
        "avg_tokens_per_doc": sum(token_counts) / max(len(token_counts), 1),
    }
    with open(a1_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    log.info("  ✓ stats.json — %d records, %d code blocks", len(token_counts), code_block_count)
    log.info("A1 pipeline complete — all outputs in %s", a1_dir)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else SE_INPUT_DIR
    output_base = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR
    a1_dir = output_base / "A1"
    batches_dir = a1_dir / "_batches"
    text_corpus_path = a1_dir / "text_corpus.jsonl"
    code_corpus_path = a1_dir / "code_corpus.jsonl"

    resume = "--resume" in sys.argv or os.environ.get("RESUME") == "1"

    log.info("=" * 60)
    log.info("Mining Step A1: Text Preprocessing (Streaming + Checkpoint)")
    log.info("Input SE data: %s", input_dir)
    log.info("Output:       %s", a1_dir)
    log.info("Resume mode:  %s", resume)
    log.info("=" * 60)

    # ── Discover files ───────────────────────────────────────────────────
    all_files = discover_files(input_dir)
    if not all_files:
        log.error("No JSONL files found in %s/{primary,hacks,supplemental}", input_dir)
        sys.exit(1)
    log.info("Discovered %d JSONL files across 3 tiers", len(all_files))

    # ── Ensure output dirs ───────────────────────────────────────────────
    a1_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume: read checkpoints + rebuild texts list ──────────────────
    stemmer = PorterStemmer()
    stop_words = build_stop_words()

    # Checkpoint format:
    #   .done        — fully processed file; content: {"records_written": N, "total": M}
    #   .processing  — partially processed file (crash/OOM mid-file);
    #                  content: {"records_written": N, "total": M}
    completed: set = set()         # batch_id -> fully done
    partial: dict = {}             # batch_id -> records_written so far

    if resume and batches_dir.exists():
        for bf in batches_dir.glob("*.done"):
            batch_id = bf.stem.rsplit(".", 1)[0]
            completed.add(batch_id)
        for bf in batches_dir.glob("*.processing"):
            batch_id = bf.stem.rsplit(".", 1)[0]
            try:
                with open(bf) as _f:
                    pc = json.load(_f)
                partial[batch_id] = pc.get("records_written", 0)
            except (json.JSONDecodeError, OSError):
                partial[batch_id] = 0

    # ── OOM detection ──────────────────────────────────────────────────
    orphaned_processing = [bf.name for bf in batches_dir.glob("*.processing")]
    if orphaned_processing:
        log.warning("⚠  Detected %d orphaned .processing files — "
                     "previous run likely OOM/crashed mid-file.",
                     len(orphaned_processing))
        for pname in orphaned_processing:
            batch_id = pname.rsplit(".", 1)[0]
            rec_count = partial.get(batch_id, "?")
            log.warning("    ↳ %s  (%d records committed, will resume from there)",
                        batch_id, rec_count)

    texts: List[str] = []
    token_counts: List[int] = []

    if resume and text_corpus_path.exists():
        log.info("Resume: rebuilding texts list from existing corpus...")
        with open(text_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                ct = entry.get("_cleaned_text", "")
                if ct:
                    texts.append(ct)
                token_counts.append(len(entry.get("tokens", [])))
        log.info("  Rebuilt: %d texts from existing corpus", len(texts))
        log.info("  Checkpointed: %d done + %d partial / %d total files",
                 len(completed), len(partial), len(all_files))

    # ── Phase 1: Stream through files ────────────────────────────────────
    mode = "a" if resume else "w"
    text_fh = open(text_corpus_path, mode, encoding="utf-8")
    code_fh = open(code_corpus_path, mode, encoding="utf-8")

    total_code = 0
    skipped = 0

    for tier, src_name, fp in all_files:
        batch_id = f"{tier}.{src_name}"
        ckpt_done = batches_dir / f"{batch_id}.done"

        if batch_id in completed:
            skipped += 1
            if skipped <= 3 or skipped % 50 == 0:
                log.info("  [%s] ✓ checkpoint found — skipped", batch_id)
            continue

        records = load_jsonl_file(fp, tier)
        total_in_file = len(records)
        if total_in_file == 0:
            ckpt_done.write_text(json.dumps({"records_written": 0, "total": 0}))
            continue

        # ── Partial resume: skip records already committed to corpus ──
        records_written_before = partial.get(batch_id, 0)
        if records_written_before >= total_in_file:
            # Fully processed but .done marker was lost (rare edge case)
            completed.add(batch_id)
            ckpt_done.write_text(json.dumps({"records_written": total_in_file,
                                              "total": total_in_file}))
            skipped += 1
            continue

        # ── Write .processing marker (enables OOM recovery mid-file) ───
        ckpt_processing = batches_dir / f"{batch_id}.processing"
        with open(ckpt_processing, "w") as _pf:
            json.dump({"records_written": records_written_before, "total": total_in_file}, _pf)

        file_rec = 0
        file_code = 0

        for idx, rec in enumerate(records):
            if idx < records_written_before:
                # Already committed to corpus in previous interrupted run
                # (texts list already rebuilt from corpus, so skip here)
                continue

            result = process_record(rec, tier, fp.name, stemmer, stop_words)
            if result is None:
                continue

            text_fh.write(json.dumps(result, ensure_ascii=False) + "\n")

            for ce in extract_code_from_record(rec, tier, fp.name):
                code_fh.write(json.dumps(ce, ensure_ascii=False) + "\n")

            texts.append(result["_cleaned_text"])
            token_counts.append(len(result["tokens"]))
            file_rec += 1
            file_code += len(extract_code_from_record(rec, tier, fp.name))

            # ── Live-update .processing checkpoint every 500 recs ──────
            if file_rec % 500 == 0:
                with open(ckpt_processing, "w") as _pf:
                    json.dump({"records_written": records_written_before + file_rec,
                               "total": total_in_file}, _pf)

        # ── Finalize: .done replaces .processing ────────────────────────
        ckpt_done.write_text(json.dumps({"records_written": records_written_before + file_rec,
                                         "total": total_in_file}))
        if ckpt_processing.exists():
            ckpt_processing.unlink()
        total_code += file_code

        log.info("  [%s] %4d rec, %3d code blocks  (→ %d total)",
                 batch_id, file_rec, file_code, len(texts))

    text_fh.close()
    code_fh.close()

    log.info("Phase 1 done: %d files (%d resumed + %d new), "
             "%d records, %d code blocks",
             len(all_files), len(completed),
             len(all_files) - len(completed),
             len(texts), total_code)

    if not texts:
        log.error("No records processed. Aborting.")
        sys.exit(1)

    # ── Free Phase 1 memory before Phase 2 ───────────────────────────────
    # texts and token_counts are held in memory during streaming but also
    # persisted to text_corpus.jsonl.  Release them now so that Phase 2
    # (TF-IDF + SBERT) can re-read from disk with much lower peak memory.
    del texts
    del token_counts
    gc.collect()
    log.info("Phase-1 text list freed — re-reading corpus for Phase 2")

    # ── Phase 2: Build TF-IDF + embeddings + stats ───────────────────────
    build_final_artifacts(a1_dir)


if __name__ == "__main__":
    main()
