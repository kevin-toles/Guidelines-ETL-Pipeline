#!/usr/bin/env python3
"""
Mining Step A6: Code Pattern Extraction
========================================
Mines the code corpus (A1_code_corpus.jsonl) for:
  - Language distribution statistics
  - Code block frequency by tag and topic
  - Language-tag association patterns
  - Code complexity signals (line count, nesting indicators)
  - Cross-site language usage patterns

This step is deterministic — no LLMs. Uses regex heuristics for
language detection and code structure analysis.

Dependencies: pygments (for language detection), numpy
Input:  A1_code_corpus.jsonl, A3_doc_topics.jsonl, A2_cluster_labels.jsonl
Output: A6_code_stats.json, A6_code_patterns.json
"""

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("mine_code_patterns")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "mining"

# ── Code Analysis ────────────────────────────────────────────────────────────


# Recognized language patterns (tag -> language name)
LANGUAGE_PATTERNS = {
    "python": ["python", "py", "python3"],
    "javascript": ["javascript", "js", "node", "nodejs", "react", "angular", "vue", "jquery"],
    "typescript": ["typescript", "ts"],
    "java": ["java", "spring", "android"],
    "csharp": ["csharp", "c#", "c-sharp", "dotnet", ".net"],
    "cpp": ["cpp", "c++", "cplusplus", "c++11", "c++14", "c++17", "stl"],
    "c": ["c", "ansi-c"],
    "rust": ["rust", "rust-lang"],
    "go": ["go", "golang"],
    "ruby": ["ruby", "rails"],
    "php": ["php", "laravel", "symfony"],
    "sql": ["sql", "mysql", "postgresql", "sqlite", "tsql"],
    "bash": ["bash", "shell", "sh", "zsh", "terminal"],
    "html": ["html", "html5", "xhtml"],
    "css": ["css", "css3", "scss", "sass", "less"],
    "r": ["r", "rlang"],
    "swift": ["swift", "ios"],
    "kotlin": ["kotlin"],
}

LANG_ALIAS_MAP = {}
for canonical, aliases in LANGUAGE_PATTERNS.items():
    for alias in aliases:
        LANG_ALIAS_MAP[alias] = canonical


def detect_language(code_text: str, declared_lang: str = "") -> str:
    """Detect programming language via pygments (primary) or regex fallback."""
    # 1. Use declared language if present and recognized
    if declared_lang:
        lang_lower = declared_lang.lower()
        if lang_lower in LANG_ALIAS_MAP:
            return LANG_ALIAS_MAP[lang_lower]

    # 2. Try pygments
    try:
        from pygments.lexers import guess_lexer

        lexer = guess_lexer(code_text[:2000])  # Guess from first 2K chars
        name = lexer.name.lower()
        for canonical, aliases in LANGUAGE_PATTERNS.items():
            if canonical in name or any(a in name for a in aliases):
                return canonical
    except Exception:
        pass

    # 3. Regex heuristics as fallback
    code_lower = code_text.lower()

    # Python patterns
    if re.search(r"\bdef\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import", code_text):
        return "python"
    # Java patterns
    if re.search(r"public\s+(static\s+)?(void|class|int|String)\s", code_text):
        return "java"
    # JavaScript patterns
    if re.search(r"\b(const|let|var)\s+\w+\s*=|function\s+\w+\s*\(|=>\s*\{", code_text):
        return "javascript"
    # SQL patterns
    if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE TABLE)\b", code_text, re.IGNORECASE):
        return "sql"
    # Bash
    if re.search(r"^#!(/bin/|/usr/bin/)", code_text) or re.search(r"\becho\s+[\"']", code_text):
        return "bash"

    return "unknown"


def compute_code_metrics(code_text: str) -> dict:
    """Compute basic code complexity metrics (heuristic, no AST)."""
    lines = code_text.split("\n")
    n_lines = len(lines)
    n_nonblank = sum(1 for l in lines if l.strip())
    n_blank = n_lines - n_nonblank

    # Indent-based nesting estimate
    indent_levels = []
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            indent_levels.append(indent // 2)  # Assume 2-space indents

    max_nesting = max(indent_levels) if indent_levels else 0
    avg_indent = np.mean(indent_levels) if indent_levels else 0.0

    # Comment ratio (estimate via // # -- ''' patterns)
    comment_lines = sum(
        1
        for l in lines
        if l.strip().startswith(("#", "//", "/*", "*", "'''", '"""', "--", "REM "))
    )
    comment_ratio = comment_lines / max(n_lines, 1)

    return {
        "n_lines": n_lines,
        "n_nonblank": n_nonblank,
        "n_blank": n_blank,
        "max_nesting": max_nesting,
        "avg_indent": round(float(avg_indent), 1),
        "comment_ratio": round(float(comment_ratio), 2),
    }


# ── Main Pipeline ────────────────────────────────────────────────────────────


def load_code_corpus(output_dir: Path) -> List[dict]:
    cp = output_dir / "A1" / "code_corpus.jsonl"
    if not cp.exists():
        log.error("A1_code_corpus.jsonl not found. Run A1 first.")
        return []
    entries = []
    with open(cp, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    log.info("Loaded %d code blocks", len(entries))
    return entries


def analyze_language_distribution(entries: List[dict]) -> dict:
    """Language distribution with cross-tabulations."""
    lang_counts = Counter()
    site_lang = defaultdict(Counter)
    tier_lang = defaultdict(Counter)
    tag_lang = defaultdict(Counter)

    for e in entries:
        lang = detect_language(e.get("code", ""), e.get("language", ""))
        lang_counts[lang] += 1
        site_lang[e.get("site", "unknown")][lang] += 1
        tier_lang[e.get("tier", "unknown")][lang] += 1

    total = len(entries)
    lang_dist = {
        lang: {"count": count, "pct": round(100 * count / total, 1)}
        for lang, count in lang_counts.most_common(30)
    }

    return {
        "total_blocks": total,
        "n_distinct_languages": len(lang_counts),
        "language_distribution": lang_dist,
        "top_5": lang_counts.most_common(5),
        "site_language_matrix": {
            site: dict(counter.most_common(10)) for site, counter in site_lang.items()
        },
        "tier_language_matrix": {
            tier: dict(counter.most_common(10)) for tier, counter in tier_lang.items()
        },
    }


def analyze_code_metrics(entries: List[dict]) -> dict:
    """Aggregate code complexity metrics."""
    all_metrics = []
    for e in entries:
        metrics = compute_code_metrics(e.get("code", ""))
        metrics["language"] = e.get("language", "")
        all_metrics.append(metrics)

    n_lines = [m["n_lines"] for m in all_metrics]
    max_nestings = [m["max_nesting"] for m in all_metrics]
    comment_ratios = [m["comment_ratio"] for m in all_metrics]

    # By language
    lang_metrics = defaultdict(list)
    for m in all_metrics:
        lang = detect_language("", m.get("language", ""))
        lang_metrics[lang].append(m["n_lines"])

    return {
        "n_blocks": len(all_metrics),
        "avg_lines": float(np.mean(n_lines)),
        "median_lines": float(np.median(n_lines)),
        "p90_lines": float(np.percentile(n_lines, 90)),
        "p99_lines": float(np.percentile(n_lines, 99)),
        "max_lines": int(max(n_lines)),
        "avg_nesting": float(np.mean(max_nestings)),
        "p90_nesting": float(np.percentile(max_nestings, 90)),
        "avg_comment_ratio": float(np.mean(comment_ratios)),
        "lang_avg_lines": {lang: float(np.mean(counts)) for lang, counts in lang_metrics.items()},
    }


def extract_code_patterns(entries: List[dict]) -> List[dict]:
    """Extract notable code patterns (framework usage, common idioms)."""
    patterns = []

    # Framework detection regex patterns
    framework_patterns = [
        ("pandas", re.compile(r"\bimport\s+pandas\b|\bfrom\s+pandas\b")),
        ("numpy", re.compile(r"\bimport\s+numpy\b|\bfrom\s+numpy\b")),
        ("scikit-learn", re.compile(r"\bimport\s+sklearn\b|\bfrom\s+sklearn\b")),
        ("tensorflow", re.compile(r"\bimport\s+tensorflow\b|\bfrom\s+tensorflow\b")),
        ("pytorch", re.compile(r"\bimport\s+torch\b|\bfrom\s+torch\b")),
        ("react", re.compile(r"\bimport\s+React\b|\bfrom\s+['\"]react['\"]")),
        ("express", re.compile(r"\brequire\s*\(\s*['\"]express['\"]")),
        ("django", re.compile(r"\bfrom\s+django\b|\bimport\s+django\b")),
        ("flask", re.compile(r"\bfrom\s+flask\b|\bimport\s+flask\b")),
        ("spring", re.compile(r"@SpringBootApplication|import\s+org\.springframework")),
        ("docker", re.compile(r"\bFROM\s+\w+|docker\s+(build|run|compose)")),
        ("kubernetes", re.compile(r"\bapiVersion:|kind:\s+(Pod|Deployment|Service)\b")),
        ("terraform", re.compile(r'\bresource\s+"\w+"\s+"\w+"\b')),
        ("ansible", re.compile(r"\bhosts:\s*\n\s*tasks:")),
    ]

    framework_counts = defaultdict(int)
    for e in entries:
        code = e.get("code", "")
        for fw_name, pattern in framework_patterns:
            if pattern.search(code):
                framework_counts[fw_name] += 1

    # Top frameworks
    for fw, count in framework_counts.most_common(20):
        patterns.append(
            {
                "type": "framework_usage",
                "framework": fw,
                "occurrences": count,
                "pct_of_code_blocks": round(100 * count / max(len(entries), 1), 1),
            }
        )

    return patterns


def save_outputs(
    lang_stats: dict, metrics: dict, patterns: List[dict], output_dir: Path
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # A6_code_stats.json
    stats = {"language_distribution": lang_stats, "complexity_metrics": metrics}
    with open(output_dir / "A6" / "code_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info("  ✓ A6_code_stats.json")

    # A6_code_patterns.json
    with open(output_dir / "A6" / "code_patterns.json", "w", encoding="utf-8") as f:
        json.dump({"patterns": patterns, "n_code_blocks": lang_stats["total_blocks"]}, f, indent=2, ensure_ascii=False)
    log.info("  ✓ A6_code_patterns.json (%d patterns)", len(patterns))


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR

    log.info("=" * 60)
    log.info("Mining Step A6: Code Pattern Extraction")
    log.info("Reads intermediates from: %s", output_dir)
    log.info("Output: %s", output_dir)
    log.info("=" * 60)

    # 1. Load code corpus
    entries = load_code_corpus(output_dir)
    if not entries:
        sys.exit(1)

    # 2. Language distribution
    lang_stats = analyze_language_distribution(entries)
    log.info("Language distribution: %d distinct languages", lang_stats["n_distinct_languages"])

    # 3. Code complexity metrics
    metrics = analyze_code_metrics(entries)
    log.info("Code metrics: avg %.1f lines, p90=%.0f lines", metrics["avg_lines"], metrics["p90_lines"])

    # 4. Code patterns
    patterns = extract_code_patterns(entries)
    log.info("Extracted %d code patterns", len(patterns))

    # 5. Save
    save_outputs(lang_stats, metrics, patterns, output_dir)

    log.info("A6 code pattern extraction complete.")


if __name__ == "__main__":
    main()
