#!/usr/bin/env python3
"""
StackExchange 5-Stage ETL Pipeline — streaming extraction with quality gating.

Stages:
  1. Extract   — 7z streaming → iterparse → Post objects
  2. Quality Gate — Answer-first filtering (Score≥1 or accepted)
  3. Suitability Router — Opinion/version/debug detection → 3 collections
  4. Signal Score — 100-point composite ranking (per-site normalized)
  5. Index — Write JSONL output + tag graph CSV for Neo4j import

Usage:
  # Process all sites in archive directory (largest first)
  python process_stackexchange.py \\
    --archive-dir /Volumes/USB321FD/Guidelines\\ ETL\\ Data/stackexchange \\
    --output-dir ./output

  # Process specific sites only
  python process_stackexchange.py \\
    --archive-dir /Volumes/USB321FD/Guidelines\\ ETL\\ Data/stackexchange \\
    --output-dir ./output \\
    --sites security,datascience,ai

  # Run only stages 1-3 (skip signal scoring)
  python process_stackexchange.py \\
    --archive-dir /Volumes/USB321FD/Guidelines\\ ETL\\ Data/stackexchange \\
    --output-dir ./output \\
    --stages 1-3

  # Dry run — print stats without writing output
  python process_stackexchange.py \\
    --archive-dir /Volumes/USB321FD/Guidelines\\ ETL\\ Data/stackexchange \\
    --output-dir ./output \\
    --dry-run
"""

import os
import sys
import re
import json
import math
import csv
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# PATTERNS (synced with stackexchange_assessment_v3.py)
# ═══════════════════════════════════════════════════════════════════

OPINION_PATTERNS = [
    'best language', 'which language', 'what language should',
    'better than', 'vs ', 'versus', 'which is better',
    'which framework', 'what framework', 'which tool',
    'what tool should', 'which ide', 'which editor',
    'most popular', 'recommend me', 'suggest me a',
]

VERSION_SPECIFIC_PATTERNS = [
    r'python\s+\d+\.\d+', r'django\s+\d+\.\d+',
    r'rails\s+\d+\.\d+', r'angular\s+\d+',
    r'react\s+\d+', r'vue\s+\d+\.\d+',
    r'jquery', r'internet\s+explorer|ie\s+\d+',
    r'ios\s+\d+', r'android\s+\d+',
    r'swift\s+\d+', r'xcode\s+\d+',
    r'node\s+\d+', r'\.net\s+\d+\.\d+',
    r'ruby\s+\d+\.\d+', r'php\s+\d+\.\d+',
    r'gcc\s+\d+', r'java\s+\d+',
    r'postgres\s+\d+', r'mysql\s+\d+',
    r'ubuntu\s+\d+\.\d+', r'debian\s+\d+',
    r'flutter\s+\d+', r'kotlin\s+\d+',
]

DEBUG_PATTERNS = [
    'help me debug', 'why does this crash', 'segfault',
    'segmentation fault', 'null pointer', 'nullpointer',
    'index out of bounds', 'array out of bounds',
    'why is this not working', 'why does this not work',
    'please help me fix', 'stuck on', 'cannot figure out',
]

HACK_PATTERNS = [
    'workaround', 'hack', 'hacky', 'might work',
    'not sure but', 'just guessing', 'untested',
    'should work', 'probably works', 'probably fine',
    "probably won't", 'probably not',
]

# ── Constants ──
SCORE_OVERRIDE = 5  # Questions with Score>=5 bypass opinion/version/debug rejection
SIGNAL_THRESHOLDS = {
    "primary": 40,
    "hacks": 30,
    "supplemental": 25,
}
TEMPORAL_DECAY_LAMBDA = 0.3  # e^(-λ × age_years) — see §4.6.1


def parse_tags(tags_str):
    """Parse tags from a StackExchange Posts.xml Tags attribute.
    
    Handles both the current pipe-delimited format and the legacy
    angle-bracket format, so the pipeline works with dumps from
    any era.
    
    Current format: |tag1|tag2|tag3|
    Legacy format:  <tag1><tag2><tag3>
    
    Returns a list of tag names (empty list if tags_str is empty).
    """
    if not tags_str:
        return []
    # Pipe-delimited (current StackExchange dump format)
    if '|' in tags_str:
        return [t for t in tags_str.split('|') if t]
    # Angle-bracket (legacy format)
    return re.findall(r'<([^>]+)>', tags_str)


# ═══════════════════════════════════════════════════════════════════
# STAGE 1: EXTRACT — Stream 7z XML into Post objects
# ═══════════════════════════════════════════════════════════════════

def xml_iterparse_stream(xml_path):
    """Yield (event, element) from an XML file with element clearing.
    
    Keeps memory bounded by calling elem.clear() after each row element
    is yielded. Handles Posts.xml files up to 100+ GB uncompressed.
    """
    for event, elem in ET.iterparse(xml_path, events=('end',)):
        if elem.tag == 'row':
            yield event, elem
        elem.clear()


def extract_posts(archive_path, temp_dir, progress_interval=500000):
    """Stream 7z x -so → temp file on external drive → iterparse.
    
    Decompresses directly to a temporary file on the provided temp_dir
    (typically the output directory on an external drive). This avoids:
    - Loading the entire decompressed XML into memory (5.8+ GB for math)
    - Writing large temporary files to /tmp on the internal SSD
    
    Args:
        archive_path: Path to the .7z archive
        temp_dir: Directory for temporary decompressed XML file
        progress_interval: How often to print progress (in rows)
    
    Yields:
        (post_type_id, attrs_dict) for each row in Posts.xml
    """
    archive_name = Path(archive_path).name
    print(f"  [Stage 1] Extracting {archive_name}...")
    
    # Stream 7z output directly to a temp file on the external drive
    # (never loads the full XML into memory)
    with tempfile.NamedTemporaryFile(
        dir=temp_dir, suffix='.xml', delete=True
    ) as tmp:
        # Stream decompression: 7z stdout → temp file (no in-memory buffer)
        with subprocess.Popen(
            ['7z', 'x', '-so', str(archive_path), 'Posts.xml'],
            stdout=tmp, stderr=subprocess.PIPE
        ) as proc:
            _, stderr = proc.communicate(timeout=7200)  # 2 hours for large archives
            if proc.returncode != 0:
                raise RuntimeError(
                    f"7z failed on {archive_path}: {stderr.decode()}"
                )
        
        tmp.flush()
        tmp.seek(0)
        xml_size = tmp.tell()
        xml_mb = xml_size / 1e6
        print(f"  [Stage 1] Decompressed Posts.xml: {xml_mb:.0f} MB")
        
        row_count = 0
        for event, elem in xml_iterparse_stream(tmp):
            attrs = dict(elem.attrib)
            pt = attrs.get('PostTypeId', '')
            row_count += 1
            if row_count % progress_interval == 0:
                print(f"    ... {row_count:,} rows extracted")
            yield pt, attrs
        
        print(f"  [Stage 1] Done: {row_count:,} rows extracted")


def load_posts_from_archive(archive_path, temp_dir):
    """Load all posts from an archive, separating questions and answers.
    
    Args:
        archive_path: Path to the .7z archive
        temp_dir: Directory for temporary decompressed XML file
    
    Returns:
        (questions: dict[id→attrs], answers: dict[id→attrs], stats: dict)
    """
    questions = {}
    answers = {}
    other = 0
    
    for post_type, attrs in extract_posts(archive_path, temp_dir):
        pid = attrs.get('Id')
        if post_type == '1':
            questions[pid] = attrs
        elif post_type == '2':
            answers[pid] = attrs
        else:
            other += 1
    
    return questions, answers, {
        'total': len(questions) + len(answers) + other,
        'questions': len(questions),
        'answers': len(answers),
        'other': other,
    }


# ═══════════════════════════════════════════════════════════════════
# STAGE 2: QUALITY GATE — Answer-first filtering
# ═══════════════════════════════════════════════════════════════════

def quality_gate(questions, answers):
    """Apply answer-first quality gate.
    
    Step 2a: Filter answers by Score ≥ 1 or accepted status.
    Step 2b: Filter questions by having retained answers AND (Score ≥ 3 or accepted answer).
    
    Returns:
        (open_pass: list, closed_pass: list, stats: dict)
        Each pass item is (qid, q_attrs, retained_answers, is_closed)
    """
    # Step 2a: Answer filter
    retained_answers = {}  # qid → [answer_attrs]
    answers_retained = 0
    answers_score_reject = 0
    answers_unlinked = 0
    answers_deleted = 0
    
    for aid, a in answers.items():
        parent_qid = a.get('ParentId')
        if not parent_qid or parent_qid not in questions:
            answers_unlinked += 1
            continue
        if a.get('DeletionDate'):
            answers_deleted += 1
            continue
        score = int(a.get('Score', 0))
        q = questions.get(parent_qid, {})
        is_accepted = aid == q.get('AcceptedAnswerId')
        if score >= 1 or is_accepted:
            answers_retained += 1
            retained_answers.setdefault(parent_qid, []).append(a)
        else:
            answers_score_reject += 1
    
    # Sort retained answers by Score desc, accepted first
    for qid in retained_answers:
        accepted_id = questions[qid].get('AcceptedAnswerId')
        retained_answers[qid].sort(
            key=lambda a: (
                0 if a.get('Id') == accepted_id else 1,
                -int(a.get('Score', 0))
            )
        )
    
    # Step 2b: Question filter
    open_pass = []
    closed_pass = []
    reject_no_answers = 0
    reject_empty = 0
    reject_deleted = 0
    reject_no_signal = 0
    
    for qid, q in questions.items():
        if q.get('DeletionDate'):
            reject_deleted += 1
            continue
        body = q.get('Body', '') or ''
        if not body or len(body.strip()) < 10:
            reject_empty += 1
            continue
        
        has_retained = qid in retained_answers and len(retained_answers[qid]) > 0
        if not has_retained:
            reject_no_answers += 1
            continue
        
        score = int(q.get('Score', 0))
        accepted_id = q.get('AcceptedAnswerId')
        is_closed = bool(q.get('ClosedDate'))
        
        has_quality = score >= 3 or (
            accepted_id and any(
                a.get('Id') == accepted_id
                for a in retained_answers.get(qid, [])
            )
        )
        
        if not has_quality:
            reject_no_signal += 1
            continue
        
        item = (qid, q, retained_answers.get(qid, []), is_closed)
        if is_closed:
            closed_pass.append(item)
        else:
            open_pass.append(item)
    
    stats = {
        'answers_retained': answers_retained,
        'answers_score_reject': answers_score_reject,
        'answers_unlinked': answers_unlinked,
        'answers_deleted': answers_deleted,
        'open_pass': len(open_pass),
        'closed_pass': len(closed_pass),
        'reject_no_answers': reject_no_answers,
        'reject_empty': reject_empty,
        'reject_deleted': reject_deleted,
        'reject_no_signal': reject_no_signal,
    }
    
    return open_pass, closed_pass, stats


# ═══════════════════════════════════════════════════════════════════
# STAGE 3: SUITABILITY ROUTER — Classify into 3 collections
# ═══════════════════════════════════════════════════════════════════

def compute_tag_counts(open_pass):
    """Count tag frequency across all questions for popular-tag bootstrap."""
    tag_count = defaultdict(int)
    for _, q, _, _ in open_pass:
        tags_str = q.get('Tags', '') or ''
        if tags_str:
            tags = parse_tags(tags_str)
            for tag in tags:
                tag_count[tag] += 1
    return tag_count


def get_popular_tags(tag_counts, top_n=200):
    """Get the top N most frequent tags."""
    sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])
    return set(tag for tag, _ in sorted_tags[:top_n])


def route_questions(open_pass, closed_pass, popular_tags=None):
    """Route questions through Stage 3 suitability classifier.
    
    Returns:
        {collection_name: [(qid, q_attrs, retained_answers, metadata)], ...}
        where collection_name ∈ {primary, hacks, supplemental, rejected}
    """
    if popular_tags is None:
        popular_tags = set()
    
    primary = []
    hacks = []
    supplemental = []
    rejected = defaultdict(list)  # reason → items
    
    # ── Process open questions ──
    for qid, q, retained_as, _ in open_pass:
        title = (q.get('Title', '') or '').lower()
        body = (q.get('Body', '') or '').lower()
        score = int(q.get('Score', 0))
        
        # Basic checks
        if not q.get('Title', '').strip():
            rejected['no_title'].append((qid, q, retained_as))
            continue
        if len(body) < 100:
            rejected['body_too_short'].append((qid, q, retained_as))
            continue
        
        # Opinion check
        is_opinion = any(p in title or p in body for p in OPINION_PATTERNS)
        if is_opinion and score < SCORE_OVERRIDE:
            rejected['opinion'].append((qid, q, retained_as))
            continue
        
        # Version-specific check (regex)
        is_version = any(re.search(p, title) or re.search(p, body)
                        for p in VERSION_SPECIFIC_PATTERNS)
        if is_version and score < SCORE_OVERRIDE:
            rejected['version_specific'].append((qid, q, retained_as))
            continue
        
        # Debug-only check
        is_debug = any(p in title or p in body for p in DEBUG_PATTERNS)
        if is_debug and score < SCORE_OVERRIDE:
            rejected['debug_only'].append((qid, q, retained_as))
            continue
        
        # Disputed accepted answer
        accepted_id = q.get('AcceptedAnswerId')
        has_dispute = False
        if accepted_id and len(retained_as) >= 2:
            accepted_score = 0
            max_other = 0
            for a in retained_as:
                sv = int(a.get('Score', 0))
                if a.get('Id') == accepted_id:
                    accepted_score = sv
                else:
                    max_other = max(max_other, sv)
            if max_other > accepted_score + 5:
                has_dispute = True
        if has_dispute:
            rejected['disputed_answers'].append((qid, q, retained_as))
            continue
        
        # Classify answers: hack vs clean
        hack_as, clean_as = [], []
        for a in retained_as:
            a_body = (a.get('Body', '') or '').lower()
            if any(p in a_body for p in HACK_PATTERNS):
                hack_as.append(a)
            else:
                clean_as.append(a)
        
        # Route based on hack ratio
        if len(retained_as) > 0 and len(hack_as) / len(retained_as) >= 0.5:
            # Majority hack answers → hacks collection
            meta = {'route_reason': 'hack_dominant', 'hack_ratio': len(hack_as) / len(retained_as)}
            hacks.append((qid, q, retained_as, meta))
        else:
            # Clean answers → primary
            meta = {'route_reason': 'clean'}
            primary.append((qid, q, clean_as, meta))
            # Also route hack answers separately if any
            if hack_as:
                hack_meta = {'route_reason': 'hack_minority', 'hack_ratio': len(hack_as) / len(retained_as)}
                hacks.append((qid, q, hack_as, hack_meta))
    
    # ── Process closed questions → supplemental ──
    for qid, q, retained_as, _ in closed_pass:
        title = (q.get('Title', '') or '').lower()
        body = (q.get('Body', '') or '').lower()
        
        if not q.get('Title', '').strip():
            continue
        if len(body) < 50:  # More lenient for closed
            continue
        
        # Separate hack answers
        hack_as, clean_as = [], []
        for a in retained_as:
            a_body = (a.get('Body', '') or '').lower()
            if any(p in a_body for p in HACK_PATTERNS):
                hack_as.append(a)
            else:
                clean_as.append(a)
        
        if clean_as:
            meta = {'route_reason': 'closed_rescue', 'closed_date': q.get('ClosedDate', '')}
            supplemental.append((qid, q, clean_as, meta))
        if hack_as:
            hack_meta = {'route_reason': 'closed_hack'}
            hacks.append((qid, q, hack_as, hack_meta))
    
    return {
        'primary': primary,
        'hacks': hacks,
        'supplemental': supplemental,
        'rejected': {reason: items for reason, items in rejected.items()},
    }


# ═══════════════════════════════════════════════════════════════════
# STAGE 4: SIGNAL SCORE — 100-point composite ranking
# ═══════════════════════════════════════════════════════════════════

# Global containers for per-site percentile normalization
# Filled by compute_per_site_percentiles() before scoring
# Structure: {site_name: {metric: {value: percentile_rank}}}
_SITE_PERCENTILES = {}


def compute_per_site_percentiles(all_scored_questions, site_name):
    """Pre-compute percentile rank lookup tables for a site's questions.
    
    Builds dict-of-dicts: {site_name: {metric: {value: percentile}}} so any
    question (primary, hacks, supplemental) can look up its percentile rank
    against the full site distribution.
    
    Args:
        all_scored_questions: List of (qid, q_attrs, retained_as, meta) tuples
                              from ALL collections (primary + hacks + supplemental)
        site_name: Site identifier
    """
    if not all_scored_questions:
        _SITE_PERCENTILES[site_name] = {}
        return
    
    qscores = []
    view_counts = []
    ans_scores = []
    
    for _, q, retained_as, _ in all_scored_questions:
        qscores.append(int(q.get('Score', 0)))
        view_counts.append(int(q.get('ViewCount', 0)))
        if retained_as:
            ans_scores.append(
                sum(int(a.get('Score', 0)) for a in retained_as) / len(retained_as)
            )
        else:
            ans_scores.append(0)
    
    def build_percentile_map(values):
        """Build {value: percentile_rank} dict for O(1) lookup."""
        if not values:
            return {}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        pct_map = {}
        for v in set(values):
            less_than = sum(1 for sv in sorted_vals if sv < v)
            pct_map[v] = (less_than / n) * 100
        return pct_map
    
    _SITE_PERCENTILES[site_name] = {
        'qscore': build_percentile_map(qscores),
        'view': build_percentile_map(view_counts),
        'ans_score': build_percentile_map(ans_scores),
    }


def compute_signal_score(q, retained_answers, site_name, popular_tags=None, now=None):
    """Compute 100-point signal score for a question.
    
    Dimensions:
      A. Community Validation (max 30) — log₂(score + 1) × 6
      B. Answer Quality (max 35) — accepted + avg score + ideal count
      C. Discovery Value (max 15) — log₂(view_count + 1) × temporal_decay × 0.75
      D. Content Depth (max 15) — body length + code blocks + formatting
      E. Tag Relevance (max 5) — ideal count + popular tags
      F. Penalties (subtract) — short title, wall-of-code, link-only
    
    Args:
        q: Question attrs dict
        retained_answers: List of answer attrs (already filtered)
        site_name: Site identifier for percentile lookup
        popular_tags: Set of top-200 tag names (optional)
        now: Current datetime for age calculation (default: now)
    
    Returns:
        Integer score 0-100
    """
    if now is None:
        now = datetime.now(timezone.utc)
    
    score = int(q.get('Score', 0))
    view_count = int(q.get('ViewCount', 0))
    accepted_id = q.get('AcceptedAnswerId')
    body = q.get('Body', '') or ''
    title = q.get('Title', '') or ''
    tags_str = q.get('Tags', '') or ''
    
    # ── Get percentile ranks (dict-based lookup, O(1)) ──
    pct_maps = _SITE_PERCENTILES.get(site_name, {})
    qscore_pct = pct_maps.get('qscore', {}).get(score, 50)
    view_pct = pct_maps.get('view', {}).get(view_count, 50)
    
    avg_ans_score = 0
    if retained_answers:
        avg_ans_score = sum(int(a.get('Score', 0)) for a in retained_answers) / len(retained_answers)
    ans_score_pct = pct_maps.get('ans_score', {}).get(avg_ans_score, 50)
    
    # ── A. Community Validation (max 30) ──
    community = math.log2(score + 1) * 6
    
    # ── B. Answer Quality (max 35) ──
    has_accepted = 1 if (accepted_id and any(
        a.get('Id') == accepted_id for a in retained_answers
    )) else 0
    accepted_bonus = has_accepted * 10
    
    avg_ans_bonus = min(avg_ans_score, 20) * 0.75
    
    num_answers = len(retained_answers)
    if 2 <= num_answers <= 3:
        count_bonus = 10
    elif 4 <= num_answers <= 5:
        count_bonus = 8
    elif num_answers == 1:
        count_bonus = 5
    elif 6 <= num_answers <= 8:
        count_bonus = 3
    else:
        count_bonus = 0
    
    answer_quality = accepted_bonus + avg_ans_bonus + count_bonus
    
    # ── C. Discovery Value (max 15) ──
    # Age calculation
    creation_date = q.get('CreationDate', '')
    age_years = 5  # Default: assume 5 years if can't parse
    if creation_date:
        try:
            # ISO format: 2020-01-15T12:00:00.000
            created = datetime.fromisoformat(creation_date.replace('Z', '+00:00'))
            age_years = (now - created).total_seconds() / (365.25 * 24 * 3600)
        except (ValueError, TypeError):
            pass
    
    temporal_decay = math.exp(-TEMPORAL_DECAY_LAMBDA * age_years)
    discovery = math.log2(view_count + 1) * temporal_decay * 0.75
    discovery = min(discovery, 15)
    
    # ── D. Content Depth (max 15) ──
    body_len = len(body)
    if 200 <= body_len <= 3000:
        body_bonus = 5
    elif 100 <= body_len < 200 or 3000 < body_len <= 10000:
        body_bonus = 3
    else:
        body_bonus = 1
    
    code_block_bonus = 5 if '<code>' in body else 0
    has_structure = any(marker in body for marker in ['<h1>', '<h2>', '<h3>', '<ul>', '<ol>', '<strong>', '<em>'])
    structure_bonus = 5 if has_structure else 2
    
    content = body_bonus + code_block_bonus + structure_bonus
    
    # ── E. Tag Relevance (max 5) ──
    tags = parse_tags(tags_str)
    tag_count = len(tags)
    if 1 <= tag_count <= 3:
        tag_count_bonus = 2.5
    elif 4 <= tag_count <= 5:
        tag_count_bonus = 1.5
    else:
        tag_count_bonus = 0
    
    popular_tag_bonus = 0
    if popular_tags:
        popular_tag_bonus = min(sum(0.5 for t in tags if t in popular_tags), 2.5)
    
    tags_score = tag_count_bonus + popular_tag_bonus
    
    # ── F. Penalties ──
    penalties = 0
    if len(title) < 20:
        penalties += 5
    
    # Wall of code: >80% of body is code blocks
    code_chars = sum(len(m) for m in re.findall(r'<code>(.*?)</code>', body))
    if body_len > 0 and code_chars / body_len > 0.8:
        penalties += 5
    
    # Link-only: >60% of body is external links
    link_count = len(re.findall(r'<a\s+href=', body))
    if body_len > 0 and link_count > 10 and body_len < 500:
        penalties += 10
    
    # No accepted answer on high-acceptance-rate sites
    # (Applied globally for simplicity; could be per-site)
    if not has_accepted:
        penalties += 5
    
    # ── Composite ──
    total = community + answer_quality + discovery + content + tags_score - penalties
    return max(0, min(100, round(total)))


# ═══════════════════════════════════════════════════════════════════
# STAGE 5: INDEX — Write JSONL output
# ═══════════════════════════════════════════════════════════════════

def build_output_doc(qid, q_attrs, retained_answers, signal_score, route, metadata):
    """Build a JSON-serializable output document for a question."""
    tags_str = q_attrs.get('Tags', '') or ''
    tags = parse_tags(tags_str)
    
    # Extract text from HTML body (simple tag stripping)
    body_html = q_attrs.get('Body', '') or ''
    body_text = re.sub(r'<[^>]+>', ' ', body_html)
    body_text = re.sub(r'\s+', ' ', body_text).strip()
    
    # Build answer list with text extraction
    answers_out = []
    for a in retained_answers:
        a_body_html = a.get('Body', '') or ''
        a_body_text = re.sub(r'<[^>]+>', ' ', a_body_html)
        a_body_text = re.sub(r'\s+', ' ', a_body_text).strip()
        answers_out.append({
            'id': a.get('Id'),
            'score': int(a.get('Score', 0)),
            'is_accepted': a.get('Id') == q_attrs.get('AcceptedAnswerId'),
            'body': a_body_text[:2000],  # Truncate for index
            'body_html': a_body_html[:5000],
            'creation_date': a.get('CreationDate', ''),
        })
    
    return {
        'id': qid,
        'title': q_attrs.get('Title', ''),
        'body': body_text[:3000],
        'body_html': body_html[:10000],
        'score': int(q_attrs.get('Score', 0)),
        'view_count': int(q_attrs.get('ViewCount', 0)),
        'answer_count': int(q_attrs.get('AnswerCount', 0)),
        'accepted_answer_id': q_attrs.get('AcceptedAnswerId'),
        'tags': tags,
        'creation_date': q_attrs.get('CreationDate', ''),
        'closed_date': q_attrs.get('ClosedDate'),
        'signal_score': signal_score,
        'route': route,
        'route_metadata': metadata,
        'answers': answers_out,
    }


def write_collection_jsonl(docs, output_path):
    """Write collection documents to a JSONL file (one JSON per line)."""
    with open(output_path, 'w') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')


def write_tag_graph_csv(all_collections, output_dir):
    """Write tag→question relationships for Neo4j import.
    
    Format: tag,question_id,site_name,tag_count_on_question
    """
    tag_path = output_dir / 'tag_graph.csv'
    with open(tag_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tag', 'question_id', 'site', 'tag_count'])
        for collection_name, docs in all_collections.items():
            for doc in docs:
                site = doc.get('site', '')
                qid = doc['id']
                tags = doc.get('tags', [])
                for tag in tags:
                    writer.writerow([tag, qid, site, len(tags)])
    return tag_path


# ═══════════════════════════════════════════════════════════════════
# PER-SITE PROCESSING
# ═══════════════════════════════════════════════════════════════════

def process_site(archive_path, output_dir, stages=frozenset({1,2,3,4,5}),
                 popular_tags=None, dry_run=False):
    """Run the full 5-stage pipeline on a single site archive.
    
    Args:
        archive_path: Path to .7z archive
        output_dir: Directory for JSONL output
        stages: Set of stages to run {1,2,3,4,5}
        popular_tags: Set of top-200 popular tags (for Stage 4)
        dry_run: If True, skip writing output files
    
    Returns:
        stats dict with per-stage metrics
    """
    site_name = Path(archive_path).stem.replace('.stackexchange.com', '').replace('.stackexchange', '')
    sz_mb = os.path.getsize(archive_path) / 1e6
    print(f"\n{'='*70}")
    print(f"SITE: {site_name} ({sz_mb:.0f} MB)")
    print(f"{'='*70}")
    
    # ═══ Stage 1: Extract ═══
    # Use the archive's directory as temp storage (on external drive, not /tmp)
    temp_dir = Path(archive_path).parent
    questions, answers, extract_stats = load_posts_from_archive(archive_path, temp_dir)
    print(f"  [Stage 1] {extract_stats['total']:,} posts ({extract_stats['questions']:,} Q, {extract_stats['answers']:,} A)")
    
    if 2 not in stages:
        return {'site': site_name, 'stage1': extract_stats}
    
    # ═══ Stage 2: Quality Gate ═══
    print(f"  [Stage 2] Applying answer-first quality gate...")
    open_pass, closed_pass, qg_stats = quality_gate(questions, answers)
    print(f"  [Stage 2] Open: {qg_stats['open_pass']:,}  Closed (suppl): {qg_stats['closed_pass']:,}  "
          f"Rejected: no-answers={qg_stats['reject_no_answers']:,} no-signal={qg_stats['reject_no_signal']:,}")
    
    if 3 not in stages:
        return {'site': site_name, 'stage1': extract_stats, 'stage2': qg_stats}
    
    # ═══ Stage 3: Suitability Router ═══
    print(f"  [Stage 3] Routing questions to collections...")
    collections = route_questions(open_pass, closed_pass, popular_tags)
    
    stage3_stats = {}
    for col_name, items in collections.items():
        if col_name == 'rejected':
            total_rej = sum(len(v) for v in items.values())
            stage3_stats['rejected'] = total_rej
            for reason, reason_items in sorted(items.items(), key=lambda x: -len(x[1])):
                if reason_items:
                    stage3_stats[f'reject_{reason}'] = len(reason_items)
        else:
            stage3_stats[col_name] = len(items)
    
    print(f"  [Stage 3] Primary: {stage3_stats.get('primary', 0):,}  "
          f"Hacks: {stage3_stats.get('hacks', 0):,}  "
          f"Supplemental: {stage3_stats.get('supplemental', 0):,}  "
          f"Rejected: {stage3_stats.get('rejected', 0):,}")
    
    if 4 not in stages:
        return {'site': site_name, 'stage1': extract_stats, 'stage2': qg_stats, 'stage3': stage3_stats}
    
    # ═══ Stage 4: Signal Score ═══
    print(f"  [Stage 4] Computing signal scores (per-site normalized)...")
    
    # Pre-compute percentiles from ALL scored collections (primary + hacks + supplemental)
    all_scored = (collections.get('primary', []) +
                  collections.get('hacks', []) +
                  collections.get('supplemental', []))
    if all_scored:
        compute_per_site_percentiles(all_scored, site_name)
    
    # Score and filter each collection
    output_docs = {}
    total_scored = 0
    total_indexed = 0
    
    for col_name in ['primary', 'hacks', 'supplemental']:
        items = collections.get(col_name, [])
        threshold = SIGNAL_THRESHOLDS.get(col_name, 30)
        scored = []
        
        for qid, q, retained_as, meta in items:
            try:
                sig = compute_signal_score(q, retained_as, site_name, popular_tags)
            except Exception:
                sig = 30  # Fallback for malformed data
            
            doc = build_output_doc(qid, q, retained_as, sig, col_name, meta)
            doc['site'] = site_name
            scored.append((sig, doc))
            total_scored += 1
        
        # Sort by score desc, take top by threshold
        scored.sort(key=lambda x: -x[0])
        passing = [(sig, doc) for sig, doc in scored if sig >= threshold]
        total_indexed += len(passing)
        
        if not dry_run:
            output_docs[col_name] = [doc for _, doc in passing]
        
        above_threshold = len(passing)
        print(f"    {col_name}: {above_threshold:,}/{len(scored):,} above threshold ({threshold})")
    
    # ═══ Stage 5: Write Output ═══
    if not dry_run and 5 in stages:
        print(f"  [Stage 5] Writing output files...")
        for col_name, docs in output_docs.items():
            col_dir = output_dir / col_name
            col_dir.mkdir(parents=True, exist_ok=True)
            output_path = col_dir / f"{site_name}.jsonl"
            write_collection_jsonl(docs, output_path)
            print(f"    {col_name}/{site_name}.jsonl: {len(docs):,} documents")
    
    stats = {
        'site': site_name,
        'size_mb': sz_mb,
        'stage1': extract_stats,
        'stage2': qg_stats,
        'stage3': stage3_stats,
        'stage4': {
            'scored': total_scored,
            'indexed': total_indexed,
            'by_collection': {
                col: len(output_docs.get(col, []))
                for col in ['primary', 'hacks', 'supplemental']
            } if not dry_run else {}
        },
    }
    
    return stats


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def get_archive_list(archive_dir, sites_filter=None):
    """Get sorted list of .7z archives, smallest first. Optionally filter by name."""
    archives = sorted(
        Path(archive_dir).glob('*.7z'),
        key=lambda p: os.path.getsize(p)
    )
    if sites_filter:
        site_set = set(sites_filter)
        archives = [a for a in archives
                    if any(s in a.name for s in site_set)]
    return archives


def main():
    parser = argparse.ArgumentParser(
        description='StackExchange 5-Stage ETL Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sites, largest first
  %(prog)s -a /path/to/archives -o ./output

  # Process 3 specific sites, stages 1-3 only (dry run)
  %(prog)s -a /path/to/archives -o ./output --sites security,datascience,ai --stages 1-3 --dry-run

  # Resume from stage 3 (skip extraction)
  %(prog)s -a /path/to/archives -o ./output --stages 3-5
        """
    )
    parser.add_argument('-a', '--archive-dir', required=True,
                        help='Directory containing .7z StackExchange archives')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Directory for JSONL output (creates primary/hacks/supplemental subdirs)')
    parser.add_argument('--sites', default=None,
                        help='Comma-separated site names to process (default: all .7z in archive-dir)')
    parser.add_argument('--stages', default='1-5',
                        help='Stage range to run, e.g. "1-5" (default), "1-3", "4-5"')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run pipeline without writing output files')
    parser.add_argument('--popular-tags-file', default=None,
                        help='Pre-computed JSON file with {"tag": count} for popular tags')
    parser.add_argument('--skip-larger-than', type=float, default=None,
                        help='Skip archives larger than N GB (useful for skipping stackoverflow.com)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip sites whose JSONL output files already exist')
    args = parser.parse_args()
    
    archive_dir = Path(args.archive_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse stage range
    stage_range = args.stages.split('-')
    stage_start = int(stage_range[0])
    stage_end = int(stage_range[1]) if len(stage_range) > 1 else stage_start
    stages = frozenset(range(stage_start, stage_end + 1))
    
    # Parse site filter
    sites_filter = None
    if args.sites:
        sites_filter = [s.strip() for s in args.sites.split(',')]
    
    # Get archive list (largest first)
    archives = get_archive_list(archive_dir, sites_filter)
    
    # Filter by size
    if args.skip_larger_than:
        max_bytes = args.skip_larger_than * 1e9
        skipped = [a for a in archives if os.path.getsize(a) > max_bytes]
        archives = [a for a in archives if os.path.getsize(a) <= max_bytes]
        for s in skipped:
            print(f"SKIP ({os.path.getsize(s)/1e9:.1f} GB > {args.skip_larger_than} GB): {s.name}")
    
    print(f"Processing {len(archives)} site(s)")
    print(f"Stages: {min(stages)}-{max(stages)}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Load popular tags
    popular_tags = set()
    if args.popular_tags_file:
        with open(args.popular_tags_file) as f:
            tag_counts = json.load(f)
        sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])
        popular_tags = set(tag for tag, _ in sorted_tags[:200])
        print(f"Loaded {len(popular_tags)} popular tags from {args.popular_tags_file}")
    
    # Resume: skip sites whose output already exists
    if args.resume and not args.dry_run:
        resumed_archives = []
        for a in archives:
            site_name = a.stem.replace('.stackexchange.com', '').replace('.stackexchange', '')
            missing = []
            for col in ['primary', 'hacks', 'supplemental']:
                out_file = output_dir / col / f"{site_name}.jsonl"
                if not out_file.exists():
                    missing.append(col)
            if missing:
                resumed_archives.append(a)
            else:
                print(f"SKIP (already complete): {site_name}")
        skipped_count = len(archives) - len(resumed_archives)
        if skipped_count:
            print(f"Resuming — {skipped_count} site(s) already complete, {len(resumed_archives)} remaining\n")
        archives = resumed_archives
    
    # Process each site
    all_stats = []
    for i, archive_path in enumerate(archives, 1):
        try:
            stats = process_site(
                archive_path, output_dir, stages=stages,
                popular_tags=popular_tags, dry_run=args.dry_run
            )
            all_stats.append(stats)
            print(f"  ✓ {stats['site']} complete ({i}/{len(archives)})")
        except Exception as e:
            print(f"  ✗ ERROR processing {archive_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Site':<35} {'Q':>10} {'A':>10} {'Primary':>10} {'Hacks':>10} {'Suppl':>10} {'Indexed':>10}")
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    total_q = 0
    total_a = 0
    total_primary = 0
    total_hacks = 0
    total_suppl = 0
    total_indexed = 0
    
    for s in all_stats:
        q = s.get('stage1', {}).get('questions', 0)
        a = s.get('stage1', {}).get('answers', 0)
        prim = s.get('stage3', {}).get('primary', 0)
        hack = s.get('stage3', {}).get('hacks', 0)
        suppl = s.get('stage3', {}).get('supplemental', 0)
        indexed = s.get('stage4', {}).get('indexed', 0)
        
        total_q += q
        total_a += a
        total_primary += prim
        total_hacks += hack
        total_suppl += suppl
        total_indexed += indexed
        
        name = s.get('site', '')[:34]
        print(f"{name:<35} {q:>10,} {a:>10,} {prim:>10,} {hack:>10,} {suppl:>10,} {indexed:>10,}")
    
    print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"{'TOTAL':<35} {total_q:>10,} {total_a:>10,} {total_primary:>10,} {total_hacks:>10,} {total_suppl:>10,} {total_indexed:>10,}")
    
    # Write tag graph across all collections
    if not args.dry_run and 5 in stages:
        # Collect all output docs for tag graph
        all_docs = {}
        for col_name in ['primary', 'hacks', 'supplemental']:
            col_dir = output_dir / col_name
            if col_dir.exists():
                for jsonl_file in col_dir.glob('*.jsonl'):
                    if jsonl_file.name.startswith('._'):
                        continue
                    site = jsonl_file.stem
                    with open(jsonl_file) as f:
                        for line in f:
                            doc = json.loads(line.strip())
                            doc['site'] = site
                            all_docs.setdefault(col_name, []).append(doc)
        
        if all_docs:
            tag_path = write_tag_graph_csv(all_docs, output_dir)
            print(f"\nTag graph written to {tag_path}")
    
    # Write summary JSON
    summary_path = output_dir / 'pipeline_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'run_time': datetime.now(timezone.utc).isoformat(),
            'total_sites_processed': len(all_stats),
            'dry_run': args.dry_run,
            'stages': f'{min(stages)}-{max(stages)}',
            'totals': {
                'questions': total_q,
                'answers': total_a,
                'primary': total_primary,
                'hacks': total_hacks,
                'supplemental': total_suppl,
                'indexed': total_indexed,
            },
            'per_site': all_stats,
        }, f, indent=2, default=str)
    print(f"Summary written to {summary_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
