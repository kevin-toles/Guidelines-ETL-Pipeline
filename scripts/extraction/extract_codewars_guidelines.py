#!/usr/bin/env python3
"""
Phase 2d: CodeWars Guideline Extractor
========================================
Extracts problem→pattern→guideline entries from CodeWars contest PDF JSONs
(2018 Barcelona, 2021 Spain Virtual, 2025 León).

Works without topic tags — uses keyword/import analysis on problem statements
and solution code to detect algorithmic patterns.

Output:
    /Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines/patterns/codewars/
        cw2018_guideline_NNNNN.json
        cw2021_guideline_NNNNN.json
        cw2025_guideline_NNNNN.json

Usage:
    python3 extract_codewars_guidelines.py
"""

import json
import os
import re
from typing import Any, Optional

# ── Configuration ──────────────────────────────────────────────────
JSON_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode Datasets/Additional Texts etc./json_output"
OUTPUT_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines/patterns/codewars"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Pattern Mapping (subset of TOPIC_PATTERN_MAP from extract_guidelines.py) ──
PATTERN_MAP = {
    "hash-based-lookup":     ("hash-based-lookup",     "Hash-based Lookup",     "lookup"),
    "two-pointers":          ("two-pointers",          "Two Pointers",          "scanning"),
    "sliding-window":        ("sliding-window",        "Sliding Window",        "window"),
    "binary-search":         ("binary-search",         "Binary Search",         "divide"),
    "sorting":               ("sorting",               "Sorting",               "ordering"),
    "dfs":                   ("dfs",                   "DFS",                   "traversal"),
    "bfs":                   ("bfs",                   "BFS",                   "traversal"),
    "dynamic-programming":   ("dynamic-programming",   "Dynamic Programming",   "optimization"),
    "greedy":                ("greedy",                "Greedy",                "optimization"),
    "heap":                  ("heap",                  "Heap / Priority Queue", "ordering"),
    "stack":                 ("stack",                 "Stack",                 "lifo"),
    "queue":                 ("queue",                 "Queue",                 "fifo"),
    "linked-list":           ("linked-list",           "Linked List",           "linear"),
    "string-processing":     ("string-processing",     "String Processing",     "transformation"),
    "mathematical-reasoning":("mathematical-reasoning","Mathematical Reasoning","mathematical"),
    "simulation":            ("simulation",            "Simulation",            "simulation"),
    "number-theory":         ("number-theory",         "Number Theory",         "mathematical"),
    "combinatorics":         ("combinatorics",         "Combinatorics",         "mathematical"),
}

# ── Keyword → pattern detection for problem statements ────────────
PROBLEM_KEYWORDS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"sort|order|arrang", re.I),              "sorting",          "Sorting"),
    (re.compile(r"hash|map|dictionary|frequency|counter", re.I), "hash-based-lookup", "Hash-based Lookup"),
    (re.compile(r"two pointer|pair|complement", re.I),     "two-pointers",     "Two Pointers"),
    (re.compile(r"window|subarray|substring.*consecutive", re.I), "sliding-window", "Sliding Window"),
    (re.compile(r"binary search|bisect|find.*log n", re.I),"binary-search",    "Binary Search"),
    (re.compile(r"depth.first|dfs|backtrack|permutation|combination", re.I), "dfs", "DFS / Backtracking"),
    (re.compile(r"breadth.first|bfs|shortest path|level", re.I), "bfs", "BFS"),
    (re.compile(r"dynamic|dp|memoization|optimal substruct|overlapping", re.I), "dynamic-programming", "Dynamic Programming"),
    (re.compile(r"greedy|locally optimal|exchange|priority", re.I), "greedy", "Greedy"),
    (re.compile(r"heap|priority|kth largest|k smallest", re.I), "heap", "Heap / Priority Queue"),
    (re.compile(r"stack|parenthes|bracket.*match|monotonic", re.I), "stack", "Stack"),
    (re.compile(r"queue|bfs|deque|fifo", re.I),            "queue",            "Queue"),
    (re.compile(r"linked.list|null|next.*pointer", re.I),  "linked-list",      "Linked List"),
    (re.compile(r"string|palindrome|anagram|subsequence|split", re.I), "string-processing", "String Processing"),
    (re.compile(r"prime|gcd|lcm|modulo|divisib|factor", re.I), "number-theory", "Number Theory"),
    (re.compile(r"permutation|combination|binomial|choose|count.*ways", re.I), "combinatorics", "Combinatorics"),
    (re.compile(r"simulat|process.*step|iterate.*rule", re.I), "simulation",    "Simulation"),
    (re.compile(r"math|equation|formula|pattern.*number|sequence", re.I), "mathematical-reasoning", "Mathematical Reasoning"),
]

# ── Code import → pattern detection for solution code ─────────────
CODE_IMPORT_PATTERNS: list[tuple[str, str, str]] = [
    ("dict",           "hash-based-lookup",     "Hash-based Lookup"),
    ("defaultdict",     "hash-based-lookup",     "Hash-based Lookup"),
    ("Counter",         "hash-based-lookup",     "Hash-based Lookup"),
    ("heapq",           "heap",                  "Heap / Priority Queue"),
    ("bisect",          "binary-search",         "Binary Search"),
    ("deque",           "queue",                 "Queue"),
    ("collections",     "hash-based-lookup",     "Hash-based Lookup"),
    ("itertools",       "backtracking",          "DFS / Backtracking"),
    ("functools.lru_cache", "dynamic-programming", "Dynamic Programming"),
    ("math.gcd",        "number-theory",         "Number Theory"),
    ("math.comb",       "combinatorics",         "Combinatorics"),
    ("LinkedList",      "linked-list",           "Linked List"),
    ("ListNode",        "linked-list",           "Linked List"),
    ("TreeNode",        "dfs",                   "DFS / Tree"),
    ("PriorityQueue",   "heap",                  "Heap / Priority Queue"),
    ("ArrayList",       "hash-based-lookup",     "Hash-based Lookup"),
    ("HashMap",         "hash-based-lookup",     "Hash-based Lookup"),
    ("HashSet",         "hash-based-lookup",     "Hash-based Lookup"),
    ("Arrays.sort",     "sorting",               "Sorting"),
    ("Collections.sort","sorting",               "Sorting"),
]

# ── GUIDELINE TEMPLATES (abbreviated) ─────────────────────────────
GUIDELINE_TEMPLATES = {
    "hash-based-lookup": {
        "situation": "Looking up, counting, or detecting duplicates in a collection where O(1) access is needed.",
        "guideline": "Use a hash map for key→value lookups or a hash set for membership checks. Trade O(n) space for O(1) average-time operations.",
        "reasoning": "Hash structures provide amortized O(1) insertion and lookup, reducing O(n²) nested iteration to O(n) single-pass.",
    },
    "two-pointers": {
        "situation": "Searching for pairs or processing a sorted (or conditionally sorted) sequence in a single pass.",
        "guideline": "Position two pointers at start/end or different speeds to converge on the solution in O(n) time with O(1) space.",
        "reasoning": "Two pointers eliminate the need for nested loops by exploiting ordering or monotonic properties.",
    },
    "sliding-window": {
        "situation": "Finding optimal subarrays, substrings, or contiguous subsequences satisfying a constraint.",
        "guideline": "Expand and contract a window over the sequence, maintaining state to track constraint satisfaction.",
        "reasoning": "Each element is added once and removed once, achieving O(n) time instead of O(n²) from enumerating all windows.",
    },
    "binary-search": {
        "situation": "Searching in a sorted range or finding a boundary where a monotonic condition flips.",
        "guideline": "Maintain a search interval [lo, hi], halving it each iteration to achieve O(log n) convergence.",
        "reasoning": "Each comparison eliminates half the search space, reducing linear O(n) to logarithmic O(log n) time.",
    },
    "sorting": {
        "situation": "Ordering elements to enable efficient subsequent processing or satisfy output requirements.",
        "guideline": "Use built-in sort (Timsort in Python, Dual-Pivot QuickSort in Java) for O(n log n) stable sorting.",
        "reasoning": "Sorting is often a preprocessing step that enables O(n) or O(log n) algorithms on the sorted result.",
    },
    "dfs": {
        "situation": "Exploring all paths, permutations, or combinations in a state space; tree/graph traversal.",
        "guideline": "Use recursion or an explicit stack to traverse deeply; backtrack when exploration of a branch is complete.",
        "reasoning": "DFS explores each path fully, using the call stack to automatically backtrack. Space O(h) for recursion depth.",
    },
    "bfs": {
        "situation": "Finding shortest path (in unweighted graphs), level-order traversal, or multi-source propagation.",
        "guideline": "Use a queue to process nodes level by level; mark visited when enqueued to prevent cycles.",
        "reasoning": "BFS guarantees shortest path in unweighted graphs. Processing by layers ensures first discovery = shortest.",
    },
    "dynamic-programming": {
        "situation": "Optimization problems with overlapping subproblems and optimal substructure.",
        "guideline": "Define a state DP[i] representing the solution up to step i. Build from base cases upward, reusing previously computed values.",
        "reasoning": "DP avoids recomputation by solving each subproblem once and storing results, often reducing exponential to polynomial time.",
    },
    "greedy": {
        "situation": "Making the locally optimal choice leads to globally optimal solution; matroid-like properties.",
        "guideline": "Sort or process elements by a key that guarantees the greedy choice property. Prove correctness via exchange argument.",
        "reasoning": "Greedy works when local optimal decisions never need revisiting. O(n log n) for sorting + O(n) for selection.",
    },
    "heap": {
        "situation": "Repeatedly accessing or removing the smallest/largest element; kth order statistics.",
        "guideline": "Maintain a heap to efficiently get min/max. Use min-heap for smallest elements, max-heap for largest.",
        "reasoning": "Heap operations are O(log n) for push/pop. Extracting k elements is O(n + k log n), optimal for streaming. kth largest: maintain a min-heap of size k.",
    },
    "stack": {
        "situation": "Processing nested structures, matching brackets, evaluating expressions, or monotonic sequences.",
        "guideline": "Use a stack for LIFO processing. For monotonic stack, maintain increasing/decreasing order to find next greater/smaller element.",
        "reasoning": "The stack's LIFO property naturally handles nesting. Monotonic stacks solve range-based problems in O(n) time.",
    },
    "queue": {
        "situation": "Processing elements in FIFO order, buffering, BFS, or sliding windows.",
        "guideline": "Use a queue (or deque for double-ended operations) to maintain processing order. Deque supports O(1) push/pop at both ends.",
        "reasoning": "FIFO ordering ensures fair processing. Deque enables advanced sliding window algorithms with O(1) pop from either end.",
    },
    "linked-list": {
        "situation": "Sequential data with frequent insertions/deletions at arbitrary positions. When array size is unknown.",
        "guideline": "Use sentinel/dummy nodes to simplify edge cases. For two-pointer, use fast/slow to detect cycles or find middle.",
        "reasoning": "Linked lists offer O(1) insertion/deletion given a pointer, at the cost of O(n) random access. Two-pointer tricks avoid length precomputation.",
    },
    "string-processing": {
        "situation": "Text manipulation: pattern matching, palindrome checking, anagram detection, or formatting.",
        "guideline": "Prefer character arrays for in-place mutation. Use hash maps for frequency counting of characters.",
        "reasoning": "String immutability in many languages means concatenation creates new objects. Array-based approaches allocate once.",
    },
    "mathematical-reasoning": {
        "situation": "Problems solvable via formula, number theory, or direct computation rather than algorithmic iteration.",
        "guideline": "Identify the mathematical operation (sum formula, modular arithmetic, parity, etc.) before implementing loops.",
        "reasoning": "Closed-form mathematical solutions are O(1) versus O(n) or worse for iterative approaches.",
    },
    "simulation": {
        "situation": "Problems describing a step-by-step process to be faithfully implemented.",
        "guideline": "Implement the process as described, using appropriate data structures. Optimize only if simulation boundaries are large.",
        "reasoning": "Simulation problems test faithful implementation. Premature optimization risks deviating from the spec.",
    },
    "number-theory": {
        "situation": "Problems involving primes, GCD/LCM, modular arithmetic, or divisibility properties.",
        "guideline": "Use Sieve of Eratosthenes for multiple primality tests. Use Euclidean algorithm for GCD. Precompute factorials and inverses modulo M.",
        "reasoning": "Sieve is O(n log log n) for all primes up to n. Euclidean GCD is O(log min(a,b)). Precomputation amortizes cost across queries.",
    },
    "combinatorics": {
        "situation": "Counting arrangements, selections, or configurations under constraints.",
        "guideline": "Use factorial, permutation, and combination formulas. For DP counting, define state as number of ways to reach subproblem.",
        "reasoning": "Combinatorial counts grow exponentially. Using formulas directly avoids enumerating all possibilities.",
    },
}


# ═══════════════════════════════════════════════════════════════════
#  PARSERS
# ═══════════════════════════════════════════════════════════════════

def parse_cw2021(d: dict) -> list[dict]:
    """Parse CodeWars 2021 Spain Virtual — alternating statement/solution chapters."""
    problems = []
    chapters = d.get("chapters", [])
    pages = d.get("pages", [])
    all_text = " ".join(p.get("text", "") for p in pages)

    # Pattern: "N \n Title" followed by a points line
    prob_pattern = re.compile(r"(?:^|\n)\s*(\d+)\s*\n\s*([A-Z][A-Za-z\s\-/]+?)\s*\n\s*(\d+)\s*points?", re.M)

    # Extract all solution code blocks
    code_blocks = extract_code_blocks(all_text)

    seen_titles = set()
    for match in prob_pattern.finditer(all_text):
        num, title, points = match.group(1), match.group(2).strip(), match.group(3)
        if title in seen_titles:
            continue
        seen_titles.add(title)

        # Find surrounding context (problem statement between this match and next)
        start = match.end()
        next_match = prob_pattern.search(all_text, start)
        end = next_match.start() if next_match else len(all_text)
        problem_text = all_text[start:end]

        # Extract code specific to this problem via proximity
        problem_end_pos = end
        context = all_text[match.start():problem_end_pos]

        problem_obj = {
            "id": f"cw2021_{num}",
            "number": int(num),
            "title": title,
            "points": int(points),
            "statement": clean_text(problem_text[:2000]),
            "context": clean_text(context[:3000]),
            "source": "codewars_2021",
            "difficulty": "Easy" if int(points) <= 2 else ("Medium" if int(points) <= 4 else "Hard"),
        }

        # Extract solutions from code blocks within context
        solutions = extract_nearby_solutions(code_blocks, match.start(), end)
        problem_obj["solutions"] = solutions

        # Detect pattern
        pid, pname = detect_pattern(problem_obj)
        problem_obj["pattern_id"] = pid
        problem_obj["pattern_name"] = pname
        problem_obj["category"] = PATTERN_MAP.get(pid, (pid, pname, "general"))[2]

        problems.append(problem_obj)

    return problems


def parse_cw2018(d: dict) -> list[dict]:
    """Parse CodeWars 2018 Barcelona — batched chapters (statements, hints, solutions)."""
    problems = []
    pages = d.get("pages", [])
    all_text = "\n".join(p.get("text", "") for p in pages)

    # Pattern: number + title + points
    prob_pattern = re.compile(r"(?:^|\n)\s*(\d+)\s+([A-Z][A-Za-z\s\-/]+?)\s+(\d+)\s*$", re.M)

    code_blocks = extract_code_blocks(all_text)
    seen_titles = set()

    for match in prob_pattern.finditer(all_text):
        num, title, points = match.group(1), match.group(2).strip(), match.group(3)
        if title in seen_titles:
            continue
        seen_titles.add(title)

        start = match.end()
        next_match = prob_pattern.search(all_text, start)
        end = next_match.start() if next_match else len(all_text)
        problem_text = all_text[start:end]

        problem_obj = {
            "id": f"cw2018_{num}",
            "number": int(num),
            "title": title,
            "points": int(points),
            "statement": clean_text(problem_text[:2000]),
            "context": clean_text(all_text[match.start():end][:3000]),
            "source": "codewars_2018",
            "difficulty": "Easy" if int(points) <= 2 else ("Medium" if int(points) <= 4 else "Hard"),
        }

        solutions = extract_nearby_solutions(code_blocks, match.start(), end)
        problem_obj["solutions"] = solutions

        pid, pname = detect_pattern(problem_obj)
        problem_obj["pattern_id"] = pid
        problem_obj["pattern_name"] = pname
        problem_obj["category"] = PATTERN_MAP.get(pid, (pid, pname, "general"))[2]

        problems.append(problem_obj)

    return problems


def parse_cw2025(d: dict) -> list[dict]:
    """Parse CodeWars León 2025 — each chapter = one problem."""
    problems = []
    chapters = d.get("chapters", [])

    next_id = 1
    for ch in chapters:
        content = ch.get("content", "")
        if not content:
            continue

        # Extract title from section info + first line after heading
        title_match = re.search(r"(?:Section\s+\d+|^\s*)([A-Z][A-Za-z\s\-/&]+?)(?:\s*\n)", content, re.M)
        title = title_match.group(1).strip() if title_match else f"Problem {next_id}"

        points_match = re.search(r"(\d+)\s*points?", content, re.I)
        points = int(points_match.group(1)) if points_match else 1

        # Introduction section
        intro_match = re.search(r"Introduction\s*\n(.*?)(?:\n\s*\n|\n\s*(?:Example|Input|Output|Solutions))", content, re.DOTALL)
        statement = intro_match.group(1).strip()[:2000] if intro_match else content[:2000]

        # Solutions section
        solutions = {}
        sol_section = re.search(r"Solutions\s*\n(.*)", content, re.DOTALL)
        if sol_section:
            sol_text = sol_section.group(1)
            # Extract Python
            py_match = re.search(r"Python\s*\n(.*?)(?:\n\s*(?:C\+\+|Java|$))", sol_text, re.DOTALL)
            if py_match:
                solutions["python"] = py_match.group(1).strip()
            cpp_match = re.search(r"C\+\+\s*\n(.*?)(?:\n\s*(?:Java|$))", sol_text, re.DOTALL)
            if cpp_match:
                solutions["cpp"] = cpp_match.group(1).strip()
            java_match = re.search(r"Java\s*\n(.*)", sol_text, re.DOTALL)
            if java_match:
                solutions["java"] = java_match.group(1).strip()

        problem_obj = {
            "id": f"cw2025_{next_id:03d}",
            "number": next_id,
            "title": title,
            "points": points,
            "statement": clean_text(statement),
            "context": clean_text(content[:3000]),
            "source": "codewars_2025",
            "difficulty": "Easy" if points <= 2 else ("Medium" if points <= 4 else "Hard"),
            "solutions": solutions,
        }

        pid, pname = detect_pattern(problem_obj)
        problem_obj["pattern_id"] = pid
        problem_obj["pattern_name"] = pname
        problem_obj["category"] = PATTERN_MAP.get(pid, (pid, pname, "general"))[2]

        problems.append(problem_obj)
        next_id += 1

    return problems


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Remove excessive whitespace and garbage characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)  # Keep printable + newline
    return text.strip()


def extract_code_blocks(text: str) -> list[dict]:
    """Find probable code blocks in text."""
    blocks = []
    # Match lines that look like code (indented, with keywords like import, def, class, int main, etc.)
    code_pattern = re.compile(r"(?:^|\n)((?:import |def |class |int main|public |private |#include|using |var |let |const |function )[^\n]+(?:\n(?:  .*|    .*|\{.*|return .*))*)", re.M)
    for match in code_pattern.finditer(text):
        blocks.append({
            "code": match.group(1).strip(),
            "start": match.start(),
            "end": match.end(),
        })
    return blocks


def extract_nearby_solutions(code_blocks: list[dict], prob_start: int, prob_end: int) -> dict:
    """Find solution code blocks near a problem's text range."""
    solutions = {}
    # Expand range to capture code within reasonable proximity
    window_start = max(0, prob_start - 500)
    window_end = prob_end + 2000

    py_snippets = []
    cpp_snippets = []
    java_snippets = []

    for block in code_blocks:
        if window_start <= block["start"] <= window_end:
            code = block["code"]
            if "def " in code or "import " in code:
                py_snippets.append(code)
            elif "#include" in code or "int main" in code:
                cpp_snippets.append(code)
            elif "class " in code or "public static" in code:
                java_snippets.append(code)

    if py_snippets:
        solutions["python"] = "\n".join(py_snippets)
    if cpp_snippets:
        solutions["cpp"] = "\n".join(cpp_snippets)
    if java_snippets:
        solutions["java"] = "\n".join(java_snippets)

    return solutions


def detect_pattern(problem: dict) -> tuple[str, str]:
    """
    Detect the algorithmic pattern for a problem via:
    1. Keyword matching on the problem statement
    2. Import/data-structure analysis on solution code
    3. Fallback to 'simulation' or 'mathematical-reasoning'
    """
    statement = problem.get("statement", "")
    solutions = problem.get("solutions", {})
    combined_code = " ".join(solutions.values())

    score: dict[str, int] = {}

    # 1. Keyword matching on statement
    for pattern, pid, pname in PROBLEM_KEYWORDS:
        if pattern.search(statement):
            score[pid] = score.get(pid, 0) + 2

    # 2. Code import analysis
    all_code = combined_code.lower()
    for keyword, pid, pname in CODE_IMPORT_PATTERNS:
        if keyword.lower() in all_code:
            score[pid] = score.get(pid, 0) + 2

    # 3. Bonus for specific code structure hints
    if "for" in combined_code and "range" in combined_code:
        pass  # generic iteration — weak signal
    if "while" in combined_code:
        pass  # also weak

    if score:
        best = max(score, key=score.get)
        best_score = score[best]
        info = PATTERN_MAP.get(best, (best, best.replace("-", " ").title(), "general"))
        return info[0], info[1]

    # Fallback
    if any(kw in statement.lower() for kw in ["compute", "calculate", "formula", "sum", "product", "average"]):
        return "mathematical-reasoning", "Mathematical Reasoning"
    return "simulation", "Simulation"


def build_guideline(problem: dict) -> dict:
    """Build a guideline JSON object from a parsed problem."""
    pid = problem["pattern_id"]
    template = GUIDELINE_TEMPLATES.get(pid, {
        "situation": "A computational problem requiring analysis and implementation.",
        "guideline": "Analyze the problem, identify the core algorithmic pattern, and implement a solution matching the required complexity.",
        "reasoning": "Breaking the problem into known patterns enables leveraging established, efficient solutions."
    })

    topics = []
    if problem.get("solutions", {}).get("python"):
        topics.append("Python")
    if problem.get("solutions", {}).get("cpp"):
        topics.append("C++")
    if problem.get("solutions", {}).get("java"):
        topics.append("Java")

    # Determine time/space from difficulty and pattern
    diff = problem.get("difficulty", "Medium")
    if diff == "Easy":
        time_complexity, space_complexity = "O(n)", "O(1)"
    elif diff == "Medium":
        time_complexity, space_complexity = "O(n log n)", "O(n)"
    else:
        time_complexity, space_complexity = "O(n²)", "O(n)"

    return {
        "guideline_id": problem["id"],
        "schema_version": "2.0",
        "source_problem_id": problem["number"],
        "source_dataset": problem["source"],
        "title": problem["title"],
        "title_slug": problem["title"].lower().replace(" ", "-").replace("/", "-"),
        "situation": {
            "summary": template["situation"],
            "tags": {"has_sorted_input": False},  # minimal tags for CodeWars
            "difficulty": diff,
            "topics": topics,
        },
        "guideline": template["guideline"],
        "reasoning": template["reasoning"],
        "complexity": {
            "time": time_complexity,
            "space": space_complexity,
        },
        "pattern": {
            "pattern_id": pid,
            "pattern_name": problem["pattern_name"],
            "category": problem["category"],
        },
        "solutions": problem.get("solutions", {}),
        "statement": problem.get("statement", ""),
        "constraints": [
            f"Solution available in {', '.join(topics) if topics else 'multiple languages'}",
            "Pattern detected via keyword analysis of problem statement and solution code"
        ],
        "alternatives": [],
    }


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    files = [
        ("416871339-Code-Wars.json", parse_cw2018, "cw2018"),
        ("536143839-StatementsAndSolutionsCodeWarsBarcelona2021-v1-4.json", parse_cw2021, "cw2021"),
        ("leon-solved_2025.json", parse_cw2025, "cw2025"),
    ]

    total = 0
    for filename, parser, prefix in files:
        path = os.path.join(JSON_DIR, filename)
        if not os.path.exists(path):
            print(f"SKIP {filename} — not found")
            continue

        with open(path) as f:
            data = json.load(f)

        problems = parser(data)
        print(f"Parsed {len(problems)} problems from {filename}")

        for i, prob in enumerate(problems):
            g = build_guideline(prob)
            out_path = os.path.join(OUTPUT_DIR, f"{prefix}_guideline_{i+1:05d}.json")
            with open(out_path, "w") as f:
                json.dump(g, f, indent=2)
            total += 1

    print(f"\nTotal guidelines written: {total}")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
