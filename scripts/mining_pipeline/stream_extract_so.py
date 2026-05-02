#!/usr/bin/env python3
"""
Streaming Stack Overflow Relevance Extractor
=============================================
Streams the 21GB stackoverflow.com-Posts.7z via 7z pipe, uses SAX XML parsing
to score each post for coding-guideline relevance WITHOUT decompressing to disk.
Extracts only high-value Q&A pairs into stackoverflow.jsonl.

Usage:
  7z x -so /path/to/stackoverflow.com-Posts.7z | python3 stream_extract_so.py

Output:
  - stackoverflow.jsonl   → One JSON record per question, with answers[] embedded
  - stackoverflow_stats.json → Extraction statistics

Constraints: NO LLMs. Tags + score + code-block heuristics only. Deterministic.
"""

import html
import json
import logging
import re
import sys
import time
import xml.sax
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("stream_extract_so")

# ── Output path ──────────────────────────────────────────────────────────────
# Write alongside the other SE output
OUTPUT_DIR = Path("/Volumes/USB321FD/Guidelines ETL Data/data/stackexchange")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "stackoverflow.jsonl"
STATS_FILE = OUTPUT_DIR / "stackoverflow_stats.json"

# ── Tag Classification ───────────────────────────────────────────────────────

HIGH_VALUE_TAGS: Set[str] = {
    # Architecture & Design
    "architecture", "software-architecture", "system-design", "design-patterns",
    "microservices", "domain-driven-design", "clean-architecture",
    "hexagonal-architecture", "onion-architecture", "layered-architecture",
    "solid-principles", "single-responsibility", "open-closed",
    "liskov-substitution", "interface-segregation", "dependency-inversion",
    "dependency-injection", "inversion-of-control", "separation-of-concerns",
    "loose-coupling", "high-cohesion", "modularity", "modular-design",
    "event-driven-architecture", "cqrs", "event-sourcing", "soa",
    "service-oriented", "rest", "restful", "graphql", "api-design", "api",
    "web-api", "api-gateway",
    # Testing
    "testing", "unit-testing", "unit-test", "integration-testing",
    "integration-test", "functional-testing", "e2e", "end-to-end",
    "test-driven-development", "tdd", "bdd", "behavior-driven-development",
    "mocking", "mock", "stub", "fake", "test-doubles", "test-coverage",
    "code-coverage", "testability", "regression-testing", "acceptance-testing",
    "pytest", "junit", "nunit", "xunit", "jest", "mocha", "chai",
    # Code Quality
    "code-quality", "code-review", "static-analysis", "static-code-analysis",
    "linting", "linter", "refactoring", "technical-debt", "code-smell",
    "code-smells", "anti-patterns", "anti-pattern", "code-organization",
    "code-structure", "naming-conventions", "naming", "code-style",
    "code-standards", "best-practices",
    # Security
    "security", "authentication", "authorization", "oauth", "jwt",
    "encryption", "cryptography", "secure-coding", "owasp", "xss",
    "sql-injection", "csrf", "cors", "https", "ssl", "tls",
    "vulnerability", "penetration-testing", "penetration-test",
    "access-control", "rbac", "abac", "identity", "saml", "openid",
    # Performance
    "performance", "optimization", "caching", "cache", "scalability",
    "concurrency", "parallelism", "async", "asynchronous", "non-blocking",
    "latency", "throughput", "profiling", "benchmarking",
    # DevOps / CI-CD
    "devops", "ci-cd", "continuous-integration", "continuous-delivery",
    "continuous-deployment", "deployment", "infrastructure-as-code",
    "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins",
    "github-actions", "gitlab-ci", "monitoring", "observability",
    "logging", "telemetry", "prometheus", "grafana", "opentelemetry",
    # Databases & Data
    "database-design", "database-schema", "data-modeling", "normalization",
    "sql", "nosql", "orm", "object-relational-mapping", "jpa", "hibernate",
    "entity-framework", "sqlalchemy", "mongodb", "postgresql", "mysql",
    "redis", "elasticsearch", "indexing", "query-optimization",
    "data-integrity", "transactions", "acid", "cap-theorem",
    # Error Handling
    "error-handling", "exception-handling", "exception", "error-handling",
    "logging", "structured-logging", "fault-tolerance", "resilience",
    "resiliency", "circuit-breaker", "retry", "timeout", "dead-letter",
    "idempotency", "idempotent", "retry-logic",
    # Documentation
    "documentation", "api-documentation", "swagger", "openapi",
    "code-documentation", "docstring", "javadoc", "xml-doc",
    "readme", "wiki",
    # Version Control
    "version-control", "git", "github", "gitlab", "merge-strategy",
    "branching-strategy", "git-flow", "trunk-based-development",
    "code-review", "pull-request",
}

BOOLEAN_FILTER_TAGS: Set[str] = {
    # Broad engineering categories — post must have at least one to pass,
    # unless it's extremely highly scored
    "python", "javascript", "typescript", "java", "c#", "c++", "rust", "go",
    "ruby", "swift", "kotlin", "scala", "php", "perl", "haskell", "clojure",
    "elixir", "erlang", "dart", "lua", "r", "matlab",
    "reactjs", "react-native", "angular", "vue.js", "svelte", "next.js",
    "nuxt.js", "django", "flask", "fastapi", "spring", "spring-boot",
    "asp.net", "asp.net-core", "rails", "ruby-on-rails", "express",
    "node.js", "deno", "bun",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "aws", "azure", "gcp", "google-cloud", "cloud-computing",
    "functional-programming", "object-oriented", "oop",
    "algorithm", "data-structures", "data-structure",
    "regex", "regular-expression",
    "multithreading", "multiprocessing", "thread-safety",
    "memory-management", "garbage-collection", "resource-management",
}

# Tags that indicate OFF-TOPIC for coding guidelines
OFF_TOPIC_TAGS: Set[str] = {
    "gaming", "games", "cooking", "travel", "photography", "music",
    "video-production", "audio", "homework", "opinion-based",
}

# ── Scoring Weights ──────────────────────────────────────────────────────────

SCORE_WEIGHTS = {
    "tag_match": 40,       # Max: all tags in high-value set
    "score_tier": 25,      # Based on log(score) bucket
    "has_code": 15,        # Body contains <code> blocks
    "has_accepted": 10,    # Has an accepted answer
    "view_tier": 10,       # Based on log(view_count) bucket
    "answer_tier": 5,      # Multiple answers → more discussion
    "title_quality": 5,    # Title contains technical keywords
}

MIN_PASS_SCORE = 40  # Total score needed to extract


# ── SAX Content Handler ──────────────────────────────────────────────────────


class PostsHandler(xml.sax.ContentHandler):
    """Streaming SAX parser for Stack Exchange Posts.xml."""

    def __init__(self):
        self.questions: List[Dict] = []
        self._questions_by_id: Dict[str, Dict] = {}  # O(1) lookup for answer attachment
        self.extracted_qids: Set[str] = set()  # Question IDs extracted so far
        # No pending_answers needed — questions precede answers in dump order (by PostId)
        # Answers to non-extracted questions are dropped.
        self.current_row = ""
        self.total_questions = 0
        self.total_answers = 0
        self.extracted_count = 0
        self.start_time = time.time()

    def startElement(self, name: str, attrs):
        if name != "row":
            return

        # Extract all attributes
        row = dict(attrs)
        post_type = row.get("PostTypeId", "")

        if post_type == "1":  # Question
            self.total_questions += 1
            self._process_question(row)

        elif post_type == "2":  # Answer
            self.total_answers += 1
            self._process_answer(row)

        # Periodic progress
        total = self.total_questions + self.total_answers
        if total % 500_000 == 0:
            elapsed = time.time() - self.start_time
            rate = total / elapsed if elapsed > 0 else 0
            log.info(
                f"Processed {total:,} posts ({self.total_questions:,}Q / "
                f"{self.total_answers:,}A) | Extracted: {self.extracted_count:,} "
                f"| {rate:,.0f} posts/sec"
            )

    def _process_question(self, row: Dict):
        """Score and potentially extract a question."""
        score = self._score_question(row)
        tags_raw = row.get("Tags", "")
        tags = [t.strip() for t in tags_raw.strip("|").split("|") if t.strip()]
        has_off_topic = any(t.lower() in OFF_TOPIC_TAGS for t in tags)

        if score >= MIN_PASS_SCORE and not has_off_topic:
            qid = row["Id"]
            body_html = row.get("Body", "")
            question = {
                "id": qid,
                "title": html.unescape(row.get("Title", "")),
                "body": self._strip_html(body_html),
                "body_html": body_html,
                "score": int(row.get("Score", 0)),
                "view_count": int(row.get("ViewCount", 0)),
                "answer_count": int(row.get("AnswerCount", 0)),
                "accepted_answer_id": row.get("AcceptedAnswerId", None),
                "tags": tags,
                "creation_date": row.get("CreationDate", ""),
                "closed_date": row.get("ClosedDate", None),
                "signal_score": score,
                "route": "primary",
                "route_metadata": {"route_reason": "so_relevance_filter"},
                "answers": [],
                "site": "stackoverflow.com",
            }

            # Record this question as extracted
            self.extracted_qids.add(qid)
            self._questions_by_id[qid] = question

            self.questions.append(question)
            self.extracted_count += 1

    def _process_answer(self, row: Dict):
        """Extract answer if its parent question was extracted."""
        parent_id = row.get("ParentId", "")
        if not parent_id:
            return

        body_html = row.get("Body", "")
        answer = {
            "id": row["Id"],
            "score": int(row.get("Score", 0)),
            "is_accepted": False,  # Will be marked after merging
            "body": self._strip_html(body_html),
            "body_html": body_html,
            "creation_date": row.get("CreationDate", ""),
        }

        # Questions always precede their answers in the SE XML dump (ordered by PostId).
        # If parent wasn't extracted, it was below threshold — drop the answer.
        if parent_id in self.extracted_qids:
            q = self._questions_by_id.get(parent_id)
            if q is not None:
                q["answers"].append(answer)
                q["answer_count"] = len(q["answers"])

    def _score_question(self, row: Dict) -> int:
        """Calculate a relevance score [0-100] for a question."""
        tags_raw = row.get("Tags", "")
        tags = [t.strip().lower() for t in tags_raw.strip("|").split("|") if t.strip()]
        body_html = row.get("Body", "")
        title = row.get("Title", "")

        score = 0

        # 1. Tag match (0-40)
        high_value_matches = sum(1 for t in tags if t in HIGH_VALUE_TAGS)
        boolean_matches = sum(1 for t in tags if t in BOOLEAN_FILTER_TAGS)
        tag_score = min(40, (high_value_matches * 10) + (boolean_matches * 5))
        score += tag_score

        # 2. Score tier (0-25)
        raw_score = int(row.get("Score", 0))
        if raw_score >= 100:
            score += 25
        elif raw_score >= 50:
            score += 20
        elif raw_score >= 20:
            score += 15
        elif raw_score >= 10:
            score += 10
        elif raw_score >= 5:
            score += 5

        # 3. Has code blocks (0-15)
        code_count = len(re.findall(r"<code>", body_html, re.IGNORECASE))
        if code_count >= 3:
            score += 15
        elif code_count >= 1:
            score += 10

        # 4. Has accepted answer (0-10)
        if row.get("AcceptedAnswerId"):
            score += 10

        # 5. View count tier (0-10)
        views = int(row.get("ViewCount", 0))
        if views >= 50_000:
            score += 10
        elif views >= 10_000:
            score += 7
        elif views >= 1_000:
            score += 4

        # 6. Answer count (0-5)
        answers = int(row.get("AnswerCount", 0))
        if answers >= 5:
            score += 5
        elif answers >= 2:
            score += 3

        # 7. Title quality (0-5)
        title_lower = title.lower()
        technical_patterns = [
            r"\bhow\b", r"\bwhy\b", r"\bbest\b", r"\bdifference\b",
            r"\bvs\b", r"\bdesign\b", r"\bpattern\b", r"\barchitect",
            r"\btest", r"\bsecurity\b", r"\bperform",
            r"\bimplement", r"\boptimiz",
            r"\berror\b", r"\bexception\b",
        ]
        quality_hits = sum(1 for p in technical_patterns if re.search(p, title_lower))
        score += min(5, quality_hits * 2)

        return score

    @staticmethod
    def _strip_html(html_text: str) -> str:
        """Remove HTML tags, decode entities, collapse whitespace."""
        text = re.sub(r"<[^>]+>", " ", html_text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def endDocument(self):
        """Log final summary."""
        elapsed = time.time() - self.start_time
        total = self.total_questions + self.total_answers
        rate = total / elapsed if elapsed > 0 else 0
        log.info("=" * 60)
        log.info(f"Stream complete in {elapsed:.1f}s ({rate:,.0f} posts/sec)")
        log.info(f"Total posts: {total:,} ({self.total_questions:,}Q / {self.total_answers:,}A)")
        log.info(f"Extracted: {self.extracted_count:,} questions")
        # All answers matching extracted questions have been attached


# ── Main ─────────────────────────────────────────────────────────────────────


def mark_accepted_answers(questions: List[Dict]):
    """Mark which answers are accepted."""
    for q in questions:
        accepted_id = q.get("accepted_answer_id")
        if accepted_id:
            for a in q["answers"]:
                if a["id"] == accepted_id:
                    a["is_accepted"] = True


def main():
    log.info("Starting Stack Overflow streaming extractor...")
    log.info(f"Reading from stdin (pipe from 7z x -so)")
    log.info(f"Output: {OUTPUT_FILE}")

    handler = PostsHandler()
    xml.sax.parse(sys.stdin, handler)

    questions = handler.questions
    mark_accepted_answers(questions)

    # Sort questions by score descending for pipeline priority
    questions.sort(key=lambda q: q["score"], reverse=True)

    log.info(f"Writing {len(questions):,} questions to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Stats
    total_answers = sum(len(q["answers"]) for q in questions)
    stats = {
        "total_posts_processed": handler.total_questions + handler.total_answers,
        "total_questions": handler.total_questions,
        "total_answers": handler.total_answers,
        "extracted_questions": len(questions),
        "extracted_answers": total_answers,
        "extraction_rate_pct": round(
            len(questions) / max(handler.total_questions, 1) * 100, 2
        ),
        "elapsed_seconds": round(time.time() - handler.start_time, 1),
        "output_file": str(OUTPUT_FILE),
    }

    log.info(f"Stats: {json.dumps(stats, indent=2)}")
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("Done.")


if __name__ == "__main__":
    main()
