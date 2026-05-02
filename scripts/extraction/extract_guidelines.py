#!/usr/bin/env python3
"""
Phase 2: Bottom-Up Guideline Extractor
========================================
Transforms each LeetCode problem into a granular, situation-specific guideline.
NO LLM involvement — 100% deterministic code analysis + template filling.

Each guideline answers: "In this specific situation, what is the best practice?"

Output:
    /Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines/
        guideline_00001.json
        guideline_00002.json
        ...

Each guideline JSON:
    {
        "guideline_id": "guideline_00001",
        "source_problem_id": 1,
        "title": "Two Sum",
        "situation": { ... },       # structured metadata about when this applies
        "guideline": "...",          # human-readable best practice (templated)
        "reasoning": "...",          # why this works (templated)
        "constraints": [ ... ],      # when this guideline holds
        "alternatives": [ ... ],     # other approaches with trade-offs
        "complexity": { ... },       # time/space complexity
        "pattern": { ... },          # the algorithmic pattern it encodes
        "difficulty": "Easy",
        "topics": ["Array", "Hash Table"],
        "bridges": { ... }           # (filled later by Phase 3)
    }

Usage:
    python3 extract_guidelines.py
"""

import ast
import json
import os
import re
import html
from typing import Any, Optional

# ── Configuration ──────────────────────────────────────────────────
LC_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/Leetcode.csv"
PROB_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/leetcode_problems.csv"
QUEST_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/leetcode_questions.csv"
OUTPUT_DIR = "/Users/kevintoles/POC/textbooks/Books/LeetCode JSON/guidelines"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────
# 1. PATTERN DETECTION — Code Analysis
# ────────────────────────────────────────────────────────────────────

# Mapping from topic tag → (pattern_id, pattern_name, category)
TOPIC_PATTERN_MAP = {
    "Hash Table":          ("hash-based-lookup",     "Hash-based Lookup",     "lookup"),
    "Prefix Sum":          ("prefix-sum",            "Prefix Sum",            "accumulation"),
    "Counting":            ("counting",              "Counting",               "frequency"),
    "Two Pointers":        ("two-pointers",          "Two Pointers",           "scanning"),
    "Sliding Window":      ("sliding-window",        "Sliding Window",         "window"),
    "Binary Search":       ("binary-search",         "Binary Search",          "divide"),
    "Depth-First Search":  ("dfs",                   "DFS",                    "traversal"),
    "Breadth-First Search":("bfs",                   "BFS",                    "traversal"),
    "Backtracking":        ("backtracking",          "Backtracking",           "exploration"),
    "Divide and Conquer":  ("divide-and-conquer",    "Divide and Conquer",     "divide"),
    "Dynamic Programming": ("dynamic-programming",   "Dynamic Programming",    "optimization"),
    "Sorting":             ("sorting",               "Sorting",                "ordering"),
    "Heap (Priority Queue)": ("heap",                "Heap / Priority Queue",  "ordering"),
    "Stack":               ("stack",                 "Stack",                  "lifo"),
    "Monotonic Stack":     ("monotonic-stack",       "Monotonic Stack",        "lifo"),
    "Queue":               ("queue",                 "Queue",                  "fifo"),
    "Tree":                ("tree-traversal",        "Tree Traversal",         "traversal"),
    "Binary Tree":         ("binary-tree",           "Binary Tree",            "traversal"),
    "Graph":               ("graph-algorithms",      "Graph Algorithms",       "traversal"),
    "Union Find":          ("union-find",            "Union-Find / DSU",       "connectivity"),
    "Trie":                ("trie",                  "Trie / Prefix Tree",     "retrieval"),
    "Topological Sort":    ("topological-sort",      "Topological Sort",       "ordering"),
    "Linked List":         ("linked-list",           "Linked List",            "linear"),
    "Bit Manipulation":    ("bit-manipulation",      "Bit Manipulation",       "bitwise"),
    "String":              ("string-processing",     "String Processing",      "transformation"),
    "Matrix":              ("matrix-grid",           "Matrix / Grid",          "grid"),
    "Simulation":          ("simulation",            "Simulation",             "simulation"),
    "Design":              ("design-pattern",        "Design Pattern",         "design"),
    "Math":                ("mathematical-reasoning","Mathematical Reasoning", "mathematical"),
    "Greedy":              ("greedy",                "Greedy",                 "optimization"),
    "Memoization":         ("memoization",           "Memoization",            "optimization"),
    "Recursion":           ("recursion",             "Recursion",              "divide"),
    "Ordered Set":         ("ordered-set",           "Ordered Set",            "ordering"),
    "Bucket Sort":         ("bucket-sort",           "Bucket Sort",            "ordering"),
    "Radix Sort":          ("radix-sort",            "Radix Sort",             "ordering"),
    "Counting Sort":       ("counting-sort",         "Counting Sort",          "ordering"),
    "Merge Sort":          ("merge-sort",            "Merge Sort",             "divide"),
    "Quickselect":         ("quickselect",           "Quickselect",            "selection"),
    "Reservoir Sampling":  ("reservoir-sampling",    "Reservoir Sampling",     "sampling"),
    "Rejection Sampling":  ("rejection-sampling",    "Rejection Sampling",     "sampling"),
    "Geometry":            ("geometry",              "Geometry",               "mathematical"),
    "Number Theory":       ("number-theory",         "Number Theory",          "mathematical"),
    "Combinatorics":       ("combinatorics",         "Combinatorics",          "mathematical"),
    "Game Theory":         ("game-theory",           "Game Theory",            "mathematical"),
    "Probability and Statistics": ("probability",    "Probability / Statistics","mathematical"),
    "Randomized":          ("randomized",            "Randomized",             "randomized"),
    "Rolling Hash":        ("rolling-hash",          "Rolling Hash",           "hashing"),
    "Hash Function":       ("hash-function",         "Hash Function",          "hashing"),
    "Database":            ("database",              "Database",               "data-management"),
    "Shell":               ("shell",                 "Shell",                  "automation"),
    "Concurrency":         ("concurrency",           "Concurrency",            "concurrency"),
    "Iterator":            ("iterator",              "Iterator",               "iteration"),
    "Brainteaser":         ("brainteaser",           "Brainteaser",            "puzzle"),
    "Data Stream":         ("data-stream",           "Data Stream",            "streaming"),
    "Interactive":         ("interactive",           "Interactive",            "interactive"),
    "Shortest Path":       ("shortest-path",         "Shortest Path",          "pathfinding"),
    "Minimum Spanning Tree": ("mst",                 "Minimum Spanning Tree",  "connectivity"),
    "Strongly Connected Component": ("scc",          "SCC",                    "connectivity"),
    "Eulerian Circuit":    ("eulerian-circuit",      "Eulerian Circuit",       "traversal"),
    "Line Sweep":          ("line-sweep",            "Line Sweep",             "sweeping"),
    "Doubly-Linked List":  ("doubly-linked-list",    "Doubly-Linked List",     "linear"),
    "Binary Indexed Tree": ("fenwick-tree",          "Fenwick Tree / BIT",     "accumulation"),
    "Segment Tree":        ("segment-tree",          "Segment Tree",           "range-query"),
    "Skip List":           ("skip-list",             "Skip List",              "ordering"),
    "Biconnected Component": ("biconnected",         "Biconnected Component",  "connectivity"),
    "Inclusion-Exclusion": ("inclusion-exclusion",   "Inclusion-Exclusion",    "counting"),
    "Topological Sort":    ("topological-sort",      "Topological Sort",       "ordering"),
    "String Matching":     ("string-matching",       "String Matching",        "pattern-matching"),
    "Suffix Array":        ("suffix-array",          "Suffix Array",           "string-index"),
    "Trie":                ("trie",                  "Trie",                   "retrieval"),
    "Meet in the Middle":  ("meet-in-middle",        "Meet in the Middle",     "divide"),
}

INFERRED_PATTERNS = {
    "array": "array-processing",
    "hash map": "hash-based-lookup",
    "hash set": "hash-based-lookup",
    "counter": "counting",
    "defaultdict": "hash-based-lookup",
    "deque": "queue",
    "heapq": "heap",
    "bisect": "binary-search",
    "bisect_left": "binary-search",
    "bisect_right": "binary-search",
    "sort": "sorting",
    "sorted": "sorting",
    "lru_cache": "memoization",
    "cache": "memoization",
    "accumulate": "prefix-sum",
    "product": "prefix-sum",
    "permutations": "backtracking",
    "combinations": "backtracking",
    "combinations_with_replacement": "backtracking",
    "re": "string-matching",
}

# ── GUIDELINE TEMPLATES ───────────────────────────────────────────
# Keyed by pattern_id. Each template has:
#   situation_template: description of the situation context
#   guideline_template: the best practice recommendation
#   reasoning_template: why this approach works
#   constraint_template: when this guideline is valid

GUIDELINE_TEMPLATES = {
    "hash-based-lookup": {
        "category": "lookup",
        "situation": (
            "Searching for a complementary element or detecting duplicates "
            "in an unsorted or mutable collection where direct indexing is not available."
        ),
        "guideline": (
            "Use a hash-based structure (hash map or hash set) to store previously seen elements "
            "for O(1) lookups. As you iterate over the collection, check if the required complement "
            "or condition-satisfying element already exists in the hash structure before inserting "
            "the current element. This defers the matching cost to a single pass."
        ),
        "reasoning": (
            "Hash structures provide amortized O(1) insertion and lookup. By trading O(n) space, "
            "the time complexity reduces from O(n²) (nested iteration) to O(n). The hash map variant "
            "preserves index association, while the hash set variant is sufficient when only presence "
            "matters."
        ),
        "constraints": [
            "Collection elements must be hashable (immutable in Python).",
            "Hash collisions can degrade performance to O(n) worst-case per operation.",
            "Does not preserve insertion order in older implementations.",
            "Space overhead scales with the number of unique elements stored."
        ],
        "alternatives": [
            {
                "approach": "Brute force nested iteration",
                "complexity": {"time": "O(n²)", "space": "O(1)"},
                "when": "Collection is very small (n < 50) or memory-constrained."
            },
            {
                "approach": "Sort + binary search / two pointers",
                "complexity": {"time": "O(n log n)", "space": "O(1) or O(n)"},
                "when": "Collection is sortable and multiple queries are needed against the same data."
            }
        ]
    },
    "two-pointers": {
        "category": "scanning",
        "situation": (
            "Searching for pairs or partitions in a sorted or sequentially traversable collection, "
            "especially when the search condition is monotonic (moving one pointer changes the result in a predictable direction)."
        ),
        "guideline": (
            "Position two pointers at strategic starting locations (both ends, same start with different speeds, "
            "or adjacent positions) and advance them based on the comparison result. Move the left pointer forward "
            "when the current sum/condition is insufficient, and the right pointer backward when it exceeds the target. "
            "The pointers converge toward each other, visiting each element at most once."
        ),
        "reasoning": (
            "Two-pointer scanning exploits the monotonic property of sorted data: if the sum of left and right "
            "elements is too small, advancing the left pointer increases the sum; if too large, retreating the "
            "right pointer decreases it. This eliminates the need for nested iteration, reducing O(n²) to O(n)."
        ),
        "constraints": [
            "Input must be sorted (or sortable) for the classic two-pointer technique.",
            "Works for exactly one pair — finding all pairs requires additional handling.",
            "Assumes the operation (sum, comparison) is monotonic with respect to pointer movement."
        ],
        "alternatives": [
            {
                "approach": "Hash-based lookup",
                "complexity": {"time": "O(n)", "space": "O(n)"},
                "when": "Collection is unsorted and cannot be sorted in-place, or when O(n) space is acceptable."
            },
            {
                "approach": "Binary search per element",
                "complexity": {"time": "O(n log n)", "space": "O(1)"},
                "when": "Only one pass with binary search lookups is sufficient."
            }
        ]
    },
    "sliding-window": {
        "category": "window",
        "situation": (
            "Processing contiguous subarrays or substrings where the optimal window size is unknown, "
            "or the window must satisfy a constraint (maximum sum, minimum length, distinct elements, etc.)."
        ),
        "guideline": (
            "Expand the right boundary of a window incrementally while tracking window state. "
            "When the window becomes invalid or optimal, shrink the left boundary until validity is restored. "
            "Record the optimal window state at each validity checkpoint. The window slides forward monotonically, "
            "never backtracking."
        ),
        "reasoning": (
            "A sliding window maintains a running aggregate (sum, count, frequency map) that updates in O(1) "
            "per element added or removed. Each element enters the window once and leaves once, yielding O(n) "
            "total time. This avoids recomputing from scratch for each subarray candidate."
        ),
        "constraints": [
            "The condition must be monotonic: expanding the window always makes it 'more' of something.",
            "Requires the ability to add/remove elements from the window boundary in O(1).",
            "Fixed-size windows are simpler than variable-size windows (no shrink logic needed)."
        ],
        "alternatives": [
            {
                "approach": "Prefix sum array",
                "complexity": {"time": "O(n²)", "space": "O(n)"},
                "when": "Window size is fixed and only sum queries are needed."
            },
            {
                "approach": "Dynamic programming for optimal substructure",
                "complexity": {"time": "O(n)", "space": "O(n)"},
                "when": "The window constraint involves complex state beyond simple aggregation."
            }
        ]
    },
    "binary-search": {
        "category": "divide",
        "situation": (
            "Searching for a value or boundary in a sorted collection, or finding a threshold where "
            "a monotonic predicate transitions from false to true."
        ),
        "guideline": (
            "Maintain a search interval [low, high] that contains the target. Examine the midpoint and "
            "compare it to the target: if the target is greater, discard the left half; if smaller, discard "
            "the right half. Halve the interval each iteration. When searching for a boundary, check the "
            "predicate at the midpoint to decide which half to keep."
        ),
        "reasoning": (
            "Binary search halves the search space each iteration, achieving O(log n) time. The key invariant "
            "is that the search interval always contains the target (or the boundary predicate transition point). "
            "The loop terminates when the interval is empty or the target is found."
        ),
        "constraints": [
            "Input must be sorted (or have a monotonic predicate).",
            "Random access to elements by index is required (array, not linked list).",
            "The predicate must be monotonic: once it becomes true, it stays true (or vice versa)."
        ],
        "alternatives": [
            {
                "approach": "Linear scan",
                "complexity": {"time": "O(n)", "space": "O(1)"},
                "when": "Collection is small, unsorted, or sequential-access only."
            },
            {
                "approach": "Interpolation search",
                "complexity": {"time": "O(log log n) average", "space": "O(1)"},
                "when": "Values are uniformly distributed and cost of probe is cheap."
            }
        ]
    },
    "dfs": {
        "category": "traversal",
        "situation": (
            "Exploring all paths, checking connectivity, or enumerating all states in a tree or graph "
            "where depth is bounded and the search space must be fully explored."
        ),
        "guideline": (
            "Use recursion or an explicit stack to traverse as deep as possible along each branch before "
            "backtracking. Mark visited nodes to prevent cycles. Process the current node (pre-order), "
            "children (in-order for binary trees), or after children (post-order) depending on the goal."
        ),
        "reasoning": (
            "DFS uses the call stack or an explicit stack to remember the traversal path. It naturally "
            "fits recursive structures and can detect cycles, compute connected components, and perform "
            "topological ordering with O(V + E) time and O(h) space where h is the maximum depth."
        ),
        "constraints": [
            "Deep recursion can overflow the call stack (mitigate with iterative stack or sys.setrecursionlimit).",
            "DFS does not guarantee shortest path (use BFS for unweighted shortest path).",
            "Requires visited tracking for graphs with cycles."
        ],
        "alternatives": [
            {
                "approach": "BFS",
                "complexity": {"time": "O(V+E)", "space": "O(V)"},
                "when": "Shortest path in unweighted graph, or when the goal is expected to be shallow."
            },
            {
                "approach": "Iterative deepening DFS",
                "complexity": {"time": "O(V+E)", "space": "O(d)"},
                "when": "Combining DFS memory efficiency with BFS completeness."
            }
        ]
    },
    "bfs": {
        "category": "traversal",
        "situation": (
            "Finding the shortest path in an unweighted graph, level-order traversal, or exploring "
            "all nodes at the current depth before moving deeper."
        ),
        "guideline": (
            "Use a queue to process nodes level by level. Enqueue the starting node, then repeatedly "
            "dequeue a node, process it, and enqueue its unvisited neighbors. Track visited nodes and "
            "the distance/level of each node from the start."
        ),
        "reasoning": (
            "BFS processes nodes in order of their distance from the start. The queue ensures FIFO order, "
            "so the first time a target is reached, it is via the shortest path (in edge count). "
            "Time is O(V + E), space is O(V) for the queue and visited set."
        ),
        "constraints": [
            "Only guarantees shortest path for unweighted graphs.",
            "Space can be O(V) for wide graphs (BFS queue).",
            "All edges must have equal weight for shortest-path guarantee."
        ],
        "alternatives": [
            {
                "approach": "DFS",
                "complexity": {"time": "O(V+E)", "space": "O(h)"},
                "when": "Memory-constrained or when any path is acceptable."
            },
            {
                "approach": "Dijkstra's algorithm",
                "complexity": {"time": "O((V+E) log V)", "space": "O(V)"},
                "when": "Graph has weighted edges and shortest weighted path is needed."
            }
        ]
    },
    "dynamic-programming": {
        "category": "optimization",
        "situation": (
            "Optimizing a decision process where overlapping subproblems exist — the optimal solution "
            "to the whole problem can be constructed from optimal solutions to subproblems."
        ),
        "guideline": (
            "Define a state representation that captures the essential information needed to make a decision. "
            "Establish a recurrence relation that expresses the optimal value at state i in terms of previous states. "
            "Solve bottom-up (tabulation) by iterating over states in dependency order, or top-down (memoization) "
            "by caching results of recursive calls."
        ),
        "reasoning": (
            "DP trades exponential time (O(2^n) for naive recursion) for polynomial time by solving each "
            "subproblem once and reusing the result. The key insight is the optimal substructure property: "
            "the optimal solution contains optimal solutions to subproblems."
        ),
        "constraints": [
            "Problem must have optimal substructure and overlapping subproblems.",
            "State space must be bounded (state explosion is the main failure mode).",
            "Bottom-up requires understanding dependency order; top-down avoids it but has recursion overhead."
        ],
        "alternatives": [
            {
                "approach": "Greedy algorithm",
                "complexity": {"time": "O(n log n)", "space": "O(1)"},
                "when": "Local optimum = global optimum (greedy choice property holds)."
            },
            {
                "approach": "Divide and conquer",
                "complexity": "varies",
                "when": "Subproblems are non-overlapping; no reuse of results."
            }
        ]
    },
    "heap": {
        "category": "ordering",
        "situation": (
            "Repeatedly accessing the smallest or largest element in a dynamic collection, "
            "or merging sorted streams where the ordering must be maintained incrementally."
        ),
        "guideline": (
            "Use a heap (priority queue) to maintain a partial ordering with O(log n) insertion and "
            "O(log n) extraction of the extreme element. For k-th smallest/largest problems, maintain "
            "a min-heap or max-heap of size k, discarding elements that fall outside the desired range."
        ),
        "reasoning": (
            "A heap is a complete binary tree stored in an array. The heap property ensures the root "
            "is always the minimum (min-heap) or maximum (max-heap). Each operation restructures the "
            "tree in O(log n). This beats the O(n) scan needed to find the extreme in an unsorted collection."
        ),
        "constraints": [
            "Heap provides fast extreme access, not fast search for arbitrary elements.",
            "Cannot find arbitrary elements without O(n) scan (unless combined with a hash map).",
            "Python's heapq is min-heap only; max-heap requires negating values."
        ],
        "alternatives": [
            {
                "approach": "Sort once, access repeatedly",
                "complexity": {"time": "O(n log n) sort + O(1) access", "space": "O(n)"},
                "when": "Collection is static and all accesses are known in advance."
            },
            {
                "approach": "Ordered set (balanced BST)",
                "complexity": {"time": "O(log n) per operation", "space": "O(n)"},
                "when": "Need both extreme access AND arbitrary search/update."
            }
        ]
    },
    "stack": {
        "category": "lifo",
        "situation": (
            "Processing nested or hierarchical structures where the most recently encountered element "
            "determines the current operation — parsing, matching, or evaluating expressions."
        ),
        "guideline": (
            "Push elements onto the stack as they are encountered. When a closing or resolving condition "
            "is met, pop from the stack and process the pair. The stack naturally tracks the nesting "
            "depth and provides access to the most recent context in O(1)."
        ),
        "reasoning": (
            "The LIFO property of a stack exactly mirrors the last-in-first-out nature of nested constructs "
            "(parentheses, HTML tags, function calls). Each push records context, each pop resolves it. "
            "All operations are O(1)."
        ),
        "constraints": [
            "Only the topmost element is accessible — no random access.",
            "Stack overflow with unbounded nesting (mitigate with explicit stack in iterative approach).",
            "Not suitable for queue-like processing (FIFO) — use deque for that."
        ],
        "alternatives": [
            {
                "approach": "Recursion (implicit call stack)",
                "complexity": {"time": "O(n)", "space": "O(depth)"},
                "when": "Language supports tail recursion optimization or depth is bounded."
            },
            {
                "approach": "Two-pointer scan",
                "complexity": {"time": "O(n)", "space": "O(1)"},
                "when": "Structure is simple and doesn't require tracking arbitrary nesting."
            }
        ]
    },
    "monotonic-stack": {
        "category": "lifo",
        "situation": (
            "Finding the next greater/smaller element, or determining the nearest element that satisfies "
            "an inequality, across an array where each element's result depends on future elements."
        ),
        "guideline": (
            "Maintain a stack of elements with a monotonically increasing (or decreasing) property. "
            "When a new element violates the monotonic property, pop elements from the stack and record "
            "the current element as their result. The popped elements have found their 'next greater/smaller'."
        ),
        "reasoning": (
            "Monotonic stack processes each element once, pushing and popping at most once, yielding O(n) time. "
            "The invariant (monotonic order) guarantees that when a violating element arrives, all elements "
            "in the stack that it 'resolves' are contiguous at the top of the stack."
        ),
        "constraints": [
            "Requires element comparison to be well-defined (total order).",
            "Only resolves the 'next' relation — not arbitrary distance queries.",
            "Decreasing vs increasing monotonic property depends on the problem (next greater vs next smaller)."
        ],
        "alternatives": [
            {
                "approach": "Brute force with nested loops",
                "complexity": {"time": "O(n²)", "space": "O(1)"},
                "when": "Array is very small or one-time computation."
            },
            {
                "approach": "Segment tree / Fenwick tree",
                "complexity": {"time": "O(n log n)", "space": "O(n)"},
                "when": "Need range queries beyond 'next element'."
            }
        ]
    },
    "backtracking": {
        "category": "exploration",
        "situation": (
            "Generating all permutations, combinations, or subsets where the search space is finite "
            "but too large for brute force enumeration without pruning."
        ),
        "guideline": (
            "Build a candidate solution incrementally. At each step, extend the candidate with an available option. "
            "Before recursing, check if the candidate can lead to a valid solution (pruning). If not, skip (prune). "
            "After recursion, remove the last choice (backtrack) to try the next option. Continue until all options "
            "are explored or the target is found."
        ),
        "reasoning": (
            "Backtracking explores a decision tree where each branch is a choice. Pruning rejects branches "
            "that cannot yield a valid solution, reducing the effective search space. Without pruning, time is "
            "O(k^n) where k is branching factor; with pruning, it can be much smaller."
        ),
        "constraints": [
            "Search space can still be exponential even with pruning.",
            "Requires the ability to express constraints as an incremental validity check.",
            "Best with problems where early pruning is possible (N-Queens, Sudoku)."
        ],
        "alternatives": [
            {
                "approach": "Dynamic programming",
                "complexity": {"time": "polynomial", "space": "polynomial"},
                "when": "Overlapping subproblems exist; DP avoids re-exploration."
            },
            {
                "approach": "Iterative generation (next permutation)",
                "complexity": {"time": "O(n!) total", "space": "O(1)"},
                "when": "Only sequential iteration over all states is needed."
            }
        ]
    },
    "prefix-sum": {
        "category": "accumulation",
        "situation": (
            "Answering multiple range sum queries on a static array, or detecting subarrays where the "
            "sum satisfies a condition without recomputing from scratch."
        ),
        "guideline": (
            "Compute a prefix sum array where prefix[i] = sum of elements [0..i). Then any subarray "
            "sum from i to j can be computed in O(1) as prefix[j] - prefix[i]. For non-sum operations, "
            "extend to prefix counts (frequency) or prefix product when the operation is invertible."
        ),
        "reasoning": (
            "Prefix sums precompute O(n) values that each subsequent query uses in O(1). The difference "
            "property works because addition is invertible (subtraction). This reduces per-query time from "
            "O(n) to O(1), at the cost of O(n) preprocessing."
        ),
        "constraints": [
            "Array data must be static (no updates) or require a Fenwick tree for incremental updates.",
            "The operation must be invertible (sum, xor, count work; min, max do not).",
            "Off-by-one errors are common — be precise about inclusive/exclusive bounds."
        ],
        "alternatives": [
            {
                "approach": "Sliding window",
                "complexity": {"time": "O(n)", "space": "O(1)"},
                "when": "Only one specific subarray condition is needed, not arbitrary queries."
            },
            {
                "approach": "Fenwick tree (Binary Indexed Tree)",
                "complexity": {"time": "O(log n) per query/update", "space": "O(n)"},
                "when": "Array data changes between queries."
            }
        ]
    },
    "union-find": {
        "category": "connectivity",
        "situation": (
            "Tracking connected components in a dynamically growing graph, detecting cycles, "
            "or determining if two elements belong to the same group."
        ),
        "guideline": (
            "Initialize each element as its own parent. To union two elements, find their roots and make one "
            "root the parent of the other. Apply path compression (make each traversed node point to the root) "
            "and union by rank (attach smaller tree under larger tree) to achieve near-constant amortized time."
        ),
        "reasoning": (
            "Union-Find with path compression and union by rank achieves amortized O(α(n)) per operation, "
            "where α(n) is the inverse Ackermann function — effectively constant. This makes it the fastest "
            "possible algorithm for dynamic connectivity."
        ),
        "constraints": [
            "Elements must be discrete and identifiable (integer IDs or hashable objects).",
            "Cannot easily split components once merged (union only, no disunion).",
            "Path compression assumes no structural queries between union operations."
        ],
        "alternatives": [
            {
                "approach": "DFS/BFS for connectivity check",
                "complexity": {"time": "O(V+E) per query", "space": "O(V)"},
                "when": "Graph is static or only a few connectivity queries are needed."
            },
            {
                "approach": "Graph adjacency list + Tarjan's algorithm",
                "complexity": {"time": "O(V+E)", "space": "O(V)"},
                "when": "Need biconnected components or articulation points."
            }
        ]
    },
    "tree-traversal": {
        "category": "traversal",
        "situation": (
            "Processing all nodes in a tree structure where the order of visiting children relative to "
            "the parent determines the traversal semantics (pre-order for copying, in-order for sorted, "
            "post-order for deletion)."
        ),
        "guideline": (
            "Choose the traversal order based on the operation: pre-order (parent before children) for "
            "copying or serialization; in-order (left-parent-right) for sorted output in BSTs; post-order "
            "(children before parent) for deletion or computing subtree properties. Use recursion for clarity "
            "or an explicit stack for deeper trees."
        ),
        "reasoning": (
            "Tree traversal visits each node exactly once, yielding O(n) time. The order determines what "
            "state is available when processing a node: pre-order has parent state, in-order has left-subtree "
            "state, post-order has both children's state."
        ),
        "constraints": [
            "Recursive traversal may overflow the call stack for deep trees.",
            "Morris traversal achieves O(1) space but modifies the tree structure temporarily.",
            "Level-order requires a queue (BFS), not a stack."
        ],
        "alternatives": [
            {
                "approach": "Morris traversal",
                "complexity": {"time": "O(n)", "space": "O(1)"},
                "when": "Memory-constrained; modifies tree temporarily."
            },
            {
                "approach": "BFS / Level-order traversal",
                "complexity": {"time": "O(n)", "space": "O(w)"},
                "when": "Need node-by-level processing (tree width w)."
            }
        ]
    },
    "graph-algorithms": {
        "category": "traversal",
        "situation": (
            "Analyzing relationships and paths in a graph structure where nodes are connected by edges, "
            "and the goal involves connectivity, shortest paths, or cycle detection."
        ),
        "guideline": (
            "Choose the traversal strategy based on the goal: DFS for connectivity and cycle detection, "
            "BFS for unweighted shortest paths, Dijkstra for weighted shortest paths, Floyd-Warshall for "
            "all-pairs shortest paths, and Floyd's cycle detection for linked lists. Always track visited "
            "nodes to prevent infinite loops."
        ),
        "reasoning": (
            "Graph algorithms trade time complexity for the information they provide. BFS/DFS are O(V+E), "
            "Dijkstra is O((V+E) log V), Floyd-Warshall is O(V³). The right choice depends on what information "
            "is needed (single-source vs all-pairs, weighted vs unweighted)."
        ),
        "constraints": [
            "Graph representation matters: adjacency list is O(V+E) space; adjacency matrix is O(V²) space.",
            "Negative weights require Bellman-Ford (O(VE)) instead of Dijkstra.",
            "Dense graphs may benefit from Floyd-Warshall (V³) over repeated Dijkstra (V * (V+E) log V)."
        ],
        "alternatives": [
            {
                "approach": "A* search",
                "complexity": {"time": "O(E)", "space": "O(V)"},
                "when": "A heuristic is available to guide the search toward the goal."
            },
            {
                "approach": "Bellman-Ford",
                "complexity": {"time": "O(VE)", "space": "O(V)"},
                "when": "Graph has negative weight edges (detects negative cycles)."
            }
        ]
    },
    "string-processing": {
        "category": "transformation",
        "situation": (
            "Manipulating text data where character-level operations, substring matching, "
            "or pattern-based transformation is required."
        ),
        "guideline": (
            "Use character arrays when in-place mutation is needed (strings are immutable in most languages). "
            "For substring search, prefer built-in find/index methods over manual iteration. For pattern matching, "
            "use regex for complex patterns or specialized algorithms (KMP, Boyer-Moore, Rabin-Karp) for "
            "high-performance substring search."
        ),
        "reasoning": (
            "String immutability guarantees safety but means every modification creates a new copy. "
            "Building strings incrementally via concatenation is O(n²) — use a list of characters and join, "
            "or StringBuilder/StringBuilder-equivalent, for O(n) construction."
        ),
        "constraints": [
            "Strings are immutable in Python, Java, and most modern languages.",
            "Repeated concatenation creates O(n²) time — collect parts and join.",
            "Character encoding (ASCII vs Unicode) affects memory and comparison behavior."
        ],
        "alternatives": [
            {
                "approach": "Regular expressions",
                "complexity": {"time": "O(n) average, O(2^n) worst with backtracking", "space": "O(n)"},
                "when": "Complex pattern matching with groups, alternatives, and quantifiers."
            },
            {
                "approach": "Two-pointer scanning",
                "complexity": {"time": "O(n)", "space": "O(1)"},
                "when": "In-place string reversal, palindrome checking, or character comparison."
            }
        ]
    },
    "bit-manipulation": {
        "category": "bitwise",
        "situation": (
            "Operating at the bit level for space-efficient flags, fast arithmetic, or "
            "problems involving power-of-two properties, XOR properties, or bit masking."
        ),
        "guideline": (
            "Use XOR (^) to find unique elements (x^x = 0, x^0 = x), AND (&) to check parity or "
            "clear bits, OR (|) to set bits, and shifts (<<, >>) for fast multiplication/division by powers of two. "
            "For subsets, iterate from 0 to (1<<n)-1 to generate all 2^n bitmasks."
        ),
        "reasoning": (
            "Bit operations are hardware-level instructions — O(1) per operation. Using bitsets reduces "
            "space by a factor of the word size (64× on modern CPUs) and enables vectorized operations "
            "via SIMD."
        ),
        "constraints": [
            "Limited to integer types — not applicable to floating-point bit patterns.",
            "Signed integer right shift behavior varies by language (arithmetic vs logical).",
            "Python integers are arbitrarily large — shifts don't overflow but can create large memory."
        ],
        "alternatives": [
            {
                "approach": "Boolean array / hash set",
                "complexity": {"time": "O(n)", "space": "O(n)"},
                "when": "More than 64 flags or readability is more important than performance."
            },
            {
                "approach": "Mathematical approach (division/modulo)",
                "complexity": {"time": "O(log n)", "space": "O(1)"},
                "when": "Operation is not easily expressed as bit manipulation."
            }
        ]
    },
    "greedy": {
        "category": "optimization",
        "situation": (
            "Making a sequence of choices where the locally optimal choice at each step leads to "
            "the globally optimal solution."
        ),
        "guideline": (
            "Sort the input by a criterion that ensures the greedy choice is safe, then iterate, "
            "making the best immediate decision at each step. Verify the greedy choice property: the "
            "global optimum can be reached by a series of local optima."
        ),
        "reasoning": (
            "Greedy algorithms are efficient (O(n log n) typical for sorting + O(n) for the pass) "
            "because they avoid exploring alternatives. The correctness depends on the problem having "
            "the greedy choice property and optimal substructure."
        ),
        "constraints": [
            "Must prove greedy choice property holds — local optimum = global optimum.",
            "Sorting changes the original data order (or requires tuple of (key, value)).",
            "Not all optimization problems can be solved greedily — verify optimal substructure."
        ],
        "alternatives": [
            {
                "approach": "Dynamic programming",
                "complexity": {"time": "O(n²) typical", "space": "O(n)"},
                "when": "Greedy choice property does not hold; need to explore trade-offs."
            },
            {
                "approach": "Divide and conquer",
                "complexity": "varies",
                "when": "Problem can be split into independent subproblems."
            }
        ]
    },
    "trie": {
        "category": "retrieval",
        "situation": (
            "Storing and querying a dynamic set of strings where prefix matching, autocomplete, "
            "or spell-checking is required."
        ),
        "guideline": (
            "Build a tree where each node represents a character and paths from root to leaf represent "
            "stored strings. Mark nodes as terminal when they represent the end of a stored word. "
            "Traverse character by character to check existence, prefix matches, or collect all words "
            "with a given prefix."
        ),
        "reasoning": (
            "Trie operations take O(k) time where k is the key length, independent of the number of "
            "stored strings. Hash tables are also O(k) but cannot do prefix matching. Tries use O(total "
            "characters) space, which can be larger than a hash table due to node overhead."
        ),
        "constraints": [
            "Space overhead: each character requires a node with children references.",
            "Alphabet size affects node size: 26 for lowercase letters, 256 for extended ASCII.",
            "Compressed (radix tree / Patricia trie) reduces space at the cost of insertion complexity."
        ],
        "alternatives": [
            {
                "approach": "Hash set / hash map",
                "complexity": {"time": "O(k)", "space": "O(n)"},
                "when": "Prefix matching is not needed; only exact string lookup."
            },
            {
                "approach": "Binary search on sorted array of strings",
                "complexity": {"time": "O(k log n)", "space": "O(n)"},
                "when": "Static set of strings; prefix range queries via bisect."
            }
        ]
    },
    "linked-list": {
        "category": "linear",
        "situation": (
            "Maintaining an ordered sequence where insertions and deletions occur at arbitrary positions, "
            "or when memory locality is not a primary concern."
        ),
        "guideline": (
            "Use a sentinel (dummy head node) to unify edge case handling for empty lists. For doubly-linked "
            "lists, maintain prev and next pointers. For singly-linked lists, track the previous node during "
            "traversal to enable deletion. Use the fast-slow pointer technique (tortoise and hare) for "
            "cycle detection and finding the middle element."
        ),
        "reasoning": (
            "Linked lists provide O(1) insertion/deletion at known positions (head or given node reference). "
            "Random access is O(n) because each element must be traversed. Fast-slow pointer solves cycle "
            "detection and middle-element finding in O(n) time and O(1) space."
        ),
        "constraints": [
            "No O(1) random access — cannot binary search.",
            "Extra memory per element for pointer(s): 8 bytes per pointer in 64-bit systems.",
            "Poor cache locality — elements may be scattered in memory."
        ],
        "alternatives": [
            {
                "approach": "Dynamic array (list/ArrayList/vector)",
                "complexity": {"time": "O(1) access, O(n) insert in middle", "space": "O(n)"},
                "when": "Random access is needed or insertions are at the end."
            },
            {
                "approach": "Skip list",
                "complexity": {"time": "O(log n) average search", "space": "O(n log n)"},
                "when": "Need ordered structure with both fast access and fast insertion."
            }
        ]
    },
    "queue": {
        "category": "fifo",
        "situation": (
            "Processing elements in the order they arrive (FIFO), such as task scheduling, "
            "buffering, or breadth-first traversal."
        ),
        "guideline": (
            "Use collections.deque for O(1) appends and pops from both ends. For thread-safe operations, "
            "use queue.Queue. Enqueue elements at one end and dequeue from the other. The element that "
            "has been in the queue longest is always processed first."
        ),
        "reasoning": (
            "FIFO ordering ensures fairness — each element is processed in arrival order. Deque provides "
            "O(1) operations at both ends by using a doubly-linked block structure internally. This is "
            "essential for BFS where node discovery order determines processing."
        ),
        "constraints": [
            "No random access — only the front element is visible.",
            "Python's deque is thread-safe for appends/pops at each end but not atomic across ends.",
            "PriorityQueue changes ordering from FIFO to priority-based."
        ],
        "alternatives": [
            {
                "approach": "List with index pointer (circular buffer)",
                "complexity": {"time": "O(1) amortized", "space": "O(capacity)"},
                "when": "Maximum queue size is known in advance and fixed capacity is acceptable."
            },
            {
                "approach": "Stack (LIFO)",
                "complexity": {"time": "O(1)", "space": "O(n)"},
                "when": "Processing order should be reversed (last-in-first-out)."
            }
        ]
    },
    "counting": {
        "category": "frequency",
        "situation": (
            "Determining the frequency of elements, finding the most/least common element, "
            "or detecting elements that appear a specific number of times."
        ),
        "guideline": (
            "Use a hash map (Counter in Python) to count element frequencies in O(n). For bounded integer "
            "ranges, use an array indexed by the element value for O(1) per increment. For majority element "
            "detection (> n/2), Boyer-Moore voting achieves O(n) time and O(1) space."
        ),
        "reasoning": (
            "Frequency counting is the foundation for many patterns. A hash map gives O(n) time and O(k) space "
            "where k is distinct elements. Boyer-Moore voting eliminates the space cost entirely when only "
            "the majority element is needed — it cancels pairs of different elements, leaving the majority."
        ),
        "constraints": [
            "Hash map approach requires hashable elements.",
            "Bounded integer range counting requires knowing the max value in advance.",
            "Boyer-Moore only finds elements with frequency > n/2 (or > n/3 with two candidates)."
        ],
        "alternatives": [
            {
                "approach": "Sort and count runs",
                "complexity": {"time": "O(n log n)", "space": "O(1)"},
                "when": "Sorting is acceptable and memory is constrained."
            },
            {
                "approach": "Boyer-Moore majority vote",
                "complexity": {"time": "O(n)", "space": "O(1)"},
                "when": "Only need majority element, not full frequency distribution."
            }
        ]
    },
    "sorting": {
        "category": "ordering",
        "situation": (
            "Ordering elements by a key to enable efficient search, identify duplicates, "
            "or satisfy a constraint that requires ordered input."
        ),
        "guideline": (
            "Use built-in sort (Timsort, O(n log n)) for general-purpose sorting. For stable sorting, "
            "Timsort is stable by default. For custom order, provide a key function rather than cmp "
            "(key is O(n) evaluated once per element vs O(n log n) for cmp). Use partial sorting "
            "(nlargest/nsmallest) when only the top k are needed."
        ),
        "reasoning": (
            "Timsort is a hybrid stable sort that exploits existing order in data, achieving O(n) on "
            "nearly-sorted input and O(n log n) worst case. Using key=custom_key instead of cmp=cmp_func "
            "computes keys once per element (O(n)) rather than on every comparison (O(n log n))."
        ),
        "constraints": [
            "Sorting an array of length n is at least O(n log n) for comparison-based sorts.",
            "Non-comparison sorts (counting, radix, bucket) can achieve O(n) but have constraints on data."
        ],
        "alternatives": [
            {
                "approach": "Counting sort",
                "complexity": {"time": "O(n + k)", "space": "O(k)"},
                "when": "Integer data with a known, small range k."
            },
            {
                "approach": "Quickselect (partial sort)",
                "complexity": {"time": "O(n) average, O(n²) worst", "space": "O(1)"},
                "when": "Need only the k-th smallest element, not the full ordering."
            }
        ]
    },
    "matrix-grid": {
        "category": "grid",
        "situation": (
            "Processing a 2D grid where elements have spatial relationships (up, down, left, right) "
            "and operations involve adjacency, path-finding, or region detection."
        ),
        "guideline": (
            "Define direction arrays (dr = [-1,1,0,0], dc = [0,0,-1,1]) to iterate neighbors cleanly. "
            "Use DFS for exploring connected regions, BFS for shortest paths, and in-place marking for "
            "visited cells. Check bounds before accessing grid[r][c] to avoid index errors."
        ),
        "reasoning": (
            "Direction arrays reduce 4 or 8 directional checks to a loop. DFS/BFS on a grid is O(mn) "
            "for an m×n grid because each cell is visited once. In-place marking avoids O(mn) visited "
            "array overhead when the grid values can be modified."
        ),
        "constraints": [
            "Grid traversal is O(mn) regardless of which search algorithm is used.",
            "In-place marking destroys original data — copy the grid if original values are needed.",
            "Recursive DFS can overflow the stack on large grids (1000×1000+)."
        ],
        "alternatives": [
            {
                "approach": "Disjoint Set Union (Union-Find)",
                "complexity": {"time": "O(mn α(mn))", "space": "O(mn)"},
                "when": "Dynamic connectivity across multiple union operations."
            },
            {
                "approach": "Dynamic programming on grid",
                "complexity": {"time": "O(mn)", "space": "O(mn)"},
                "when": "Optimal path with cumulative costs (DP for path sums, etc.)."
            }
        ]
    },
    "mathematical-reasoning": {
        "category": "mathematical",
        "situation": (
            "Solving problems where the core logic relies on mathematical properties, number theory, "
            "or geometric relationships rather than data structure manipulation."
        ),
        "guideline": (
            "Identify the underlying mathematical operation: Euclidean algorithm for GCD, "
            "modular arithmetic for cyclic behavior, combinatorics for counting arrangements, "
            "Pythagorean theorem for distances. Simplify the problem to its mathematical essence "
            "before implementing — the math often reduces the problem to a formula."
        ),
        "reasoning": (
            "Mathematical approaches often reduce multi-step logic to a closed-form expression. "
            "For example, the sum 1+2+...+n = n(n+1)/2 avoids iteration. This can reduce time from "
            "O(n) to O(1) when a formula exists."
        ),
        "constraints": [
            "Integer overflow for large computations (Python handles big ints, C++/Java need 64-bit or BigInteger).",
            "Floating-point precision errors for division or square roots — prefer integer arithmetic.",
            "Modulo operations for large results (10^9+7 is common)."
        ],
        "alternatives": [
            {
                "approach": "Simulation / iteration",
                "complexity": {"time": "O(n) or O(result)", "space": "O(1)"},
                "when": "No closed-form formula exists or n is small."
            }
        ]
    },
    "simulation": {
        "category": "simulation",
        "situation": (
            "Modeling a real-world process step by step, where the state transitions are explicitly "
            "defined and must be followed precisely."
        ),
        "guideline": (
            "Define the state explicitly, then iterate through each step/rule, updating the state "
            "according to the problem's rules. Use a loop with a clear termination condition. "
            "Break complex rules into helper functions for readability."
        ),
        "reasoning": (
            "Simulation is straightforward but may be slow if the number of steps is large. "
            "The key is to identify the invariant that determines termination, and to handle edge "
            "cases (boundary conditions, cycles) before they cause infinite loops."
        ),
        "constraints": [
            "Simulation time grows with the number of steps — potentially unbounded.",
            "Must detect cycles to avoid infinite loops."
        ],
        "alternatives": [
            {
                "approach": "Mathematical formula",
                "complexity": {"time": "O(1)", "space": "O(1)"},
                "when": "The process has a closed-form solution (e.g., arithmetic series)."
            },
            {
                "approach": "State compression / pattern detection",
                "complexity": {"time": "varies", "space": "O(1)"},
                "when": "The simulation has repeating states that can be detected and skipped."
            }
        ]
    },
    "database": {
        "category": "data-management",
        "situation": (
            "Querying, transforming, or analyzing structured data using set-based operations "
            "rather than procedural iteration."
        ),
        "guideline": (
            "Use set-based operations (JOIN, GROUP BY, window functions) rather than row-by-row processing. "
            "Index columns used in WHERE, JOIN, and ORDER BY clauses. Use CTEs for readability when a query "
            "has multiple steps. Prefer EXISTS over IN for subqueries when checking for existence."
        ),
        "reasoning": (
            "SQL is declarative — the database optimizer chooses the execution plan. Set-based operations "
            "are orders of magnitude faster than procedural (cursor/loop) equivalents. Window functions "
            "avoid self-joins for rank, running totals, and lag/lead operations."
        ),
        "constraints": [
            "Indexing strategy depends on query patterns, not just schema.",
            "CTEs can materialize differently across databases (PostgreSQL materializes, SQL Server may not).",
            "N+1 query problem: avoid querying in loops — use JOIN or IN with subquery."
        ],
        "alternatives": [
            {
                "approach": "In-memory processing (pandas, polars)",
                "complexity": "varies with data size",
                "when": "Data fits in memory and requires complex transformations beyond SQL expressiveness."
            }
        ]
    },
    "design-pattern": {
        "category": "design",
        "situation": (
            "Implementing an object-oriented design with specific behavioral constraints, "
            "such as caching, iteration, or state management."
        ),
        "guideline": (
            "Identify the core abstraction and code to the interface, not the implementation. "
            "Use composition over inheritance — delegate behavior to helper objects. "
            "Consider the Singleton, Factory, Observer, or Strategy patterns if they map cleanly "
            "to the problem's requirements."
        ),
        "reasoning": (
            "Design patterns provide tested, reusable solutions to common design problems. "
            "Using a known pattern makes the code readable to other developers who recognize it. "
            "However, patterns should solve a genuine problem — don't force patterns where a simple "
            "function suffices."
        ),
        "constraints": [
            "Over-engineering: don't apply a pattern when a simple solution works.",
            "Patterns add abstraction layers — each layer adds complexity and indirection.",
            "In dynamic languages (Python), some patterns are simpler or unnecessary (e.g., Strategy = higher-order function)."
        ],
        "alternatives": [
            {
                "approach": "Higher-order functions / lambdas",
                "complexity": {"time": "O(n)", "space": "O(1)"},
                "when": "Language supports first-class functions; simplifies Strategy, Command patterns."
            },
            {
                "approach": "Simple procedural code",
                "complexity": "varies",
                "when": "The design is simple enough that patterns add unnecessary overhead."
            }
        ]
    },
}


# ── DEFAULT TEMPLATE for uncategorized patterns ───────────────────
DEFAULT_TEMPLATE = {
    "category": "general",
    "situation": (
        "Applying a specific technique to transform input data into a desired output "
        "where the constraints and operations are explicitly defined."
    ),
    "guideline": (
        "Analyze the problem to identify the core operation that transforms inputs to outputs. "
        "Implement that operation directly, handling edge cases (empty input, boundary values, "
        "invalid states) before the main logic. Test with the provided examples first."
    ),
    "reasoning": (
        "Direct implementation of the defined transformation is the baseline approach. "
        "Once working, optimize by identifying bottlenecks (repeated computation, unnecessary allocations)."
    ),
    "constraints": [
        "Handle edge cases before implementing the main logic.",
        "Prefer simplicity — the simplest correct solution is usually best."
    ],
    "alternatives": [
        {
            "approach": "Brute force with optimization",
            "complexity": {"time": "varies", "space": "varies"},
            "when": "Constraints are small enough that optimization is unnecessary."
        }
    ]
}


# ────────────────────────────────────────────────────────────────────
# 2. CODE ANALYSIS — Parse solution code for pattern detection
# ────────────────────────────────────────────────────────────────────

def detect_code_patterns(source_code: str) -> dict:
    """Analyze Python solution code to detect data structures, patterns, and operations.
    Returns structured metadata about the code's approach."""
    
    analysis = {
        "data_structures": [],
        "algorithms": [],
        "imports": [],
        "loop_types": [],
        "complexity_hints": [],
        "uses_dict": False,
        "uses_set": False,
        "uses_list": False,
        "uses_deque": False,
        "uses_heap": False,
        "uses_sorting": False,
        "uses_recursion": False,
        "has_nested_loops": False,
        "has_while_loop": False,
        "has_for_loop": False,
        "pointer_technique": False,
        "detected_pattern_id": None,
    }
    
    if not source_code:
        return analysis
    
    # Check imports
    import_patterns = {
        "collections.deque": "deque",
        "collections.defaultdict": "defaultdict", 
        "collections.Counter": "counter",
        "collections.OrderedDict": "ordered-dict",
        "heapq": "heapq",
        "bisect": "bisect",
        "itertools": "itertools",
        "functools.lru_cache": "lru_cache",
        "functools.cache": "cache",
    }
    for imp, name in import_patterns.items():
        if imp in source_code:
            analysis["imports"].append(name)
    
    # Detect data structures
    if "dict(" in source_code or "{}" in source_code or "defaultdict" in source_code or "Counter" in source_code:
        analysis["data_structures"].append("hash_map")
        analysis["uses_dict"] = True
    if "set(" in source_code or "set()" in source_code:
        analysis["data_structures"].append("hash_set")
        analysis["uses_set"] = True
    if "deque" in source_code:
        analysis["data_structures"].append("deque")
        analysis["uses_deque"] = True
    if "heapq" in source_code:
        analysis["data_structures"].append("heap")
        analysis["uses_heap"] = True
    if "[" in source_code and "List[" in source_code:
        analysis["data_structures"].append("list")
        analysis["uses_list"] = True
    
    # Detect sorting
    if ".sort()" in source_code or "sorted(" in source_code:
        analysis["algorithms"].append("sorting")
        analysis["uses_sorting"] = True
    
    # Detect recursion
    if "def " in source_code:
        # Check if a function calls itself
        func_matches = re.findall(r'def (\w+)\(', source_code)
        for fname in func_matches:
            if fname != "Solution" and fname != "__init__":
                call_pattern = rf'(?<!def )\b{re.escape(fname)}\('
                if re.search(call_pattern, source_code):
                    analysis["uses_recursion"] = True
                    analysis["algorithms"].append("recursion")
                    break
    
    # Detect loops
    if "for " in source_code:
        analysis["has_for_loop"] = True
        analysis["loop_types"].append("for")
    if "while " in source_code:
        analysis["has_while_loop"] = True
        analysis["loop_types"].append("while")
    
    # Detect nested loops (heuristic: two for/while keywords)
    loop_count = len(re.findall(r'\b(for|while)\s+', source_code))
    if loop_count >= 2:
        analysis["has_nested_loops"] = True
    
    # Detect two-pointer technique
    if ("left" in source_code and "right" in source_code) or \
       ("l " in source_code and "r " in source_code) or \
       ("slow" in source_code and "fast" in source_code):
        analysis["pointer_technique"] = True
        analysis["algorithms"].append("two_pointers")
    
    # Detect specific patterns
    if analysis["uses_dict"] and not analysis["uses_sorting"]:
        analysis["detected_pattern_id"] = "hash-based-lookup"
    elif analysis["pointer_technique"]:
        analysis["detected_pattern_id"] = "two-pointers"
    elif "heapq" in source_code:
        analysis["detected_pattern_id"] = "heap"
    elif analysis["uses_deque"] and "queue" in source_code.lower():
        analysis["detected_pattern_id"] = "queue"
    elif "bisect" in source_code:
        analysis["detected_pattern_id"] = "binary-search"
    
    # Complexity hints
    if analysis["uses_dict"] or analysis["uses_set"]:
        analysis["complexity_hints"].append("O(n)_space_for_hash")
    if analysis["has_nested_loops"] and not (analysis["uses_dict"] or analysis["uses_set"]):
        analysis["complexity_hints"].append("O(n²)_brute_force")
    if analysis["uses_sorting"]:
        analysis["complexity_hints"].append("O(n_log_n)_sort")
    if not analysis["has_nested_loops"] and (analysis["has_for_loop"] or analysis["has_while_loop"]):
        analysis["complexity_hints"].append("O(n)_single_pass")
    
    return analysis


def timeframe_from_difficulty(difficulty: str) -> dict:
    """Provide typical complexity ranges for each difficulty level."""
    ranges = {
        "Easy": {"typical_time": ["O(n)", "O(n log n)", "O(n²)"],
                  "typical_space": ["O(1)", "O(n)"]},
        "Medium": {"typical_time": ["O(n)", "O(n log n)", "O(n²)"],
                    "typical_space": ["O(1)", "O(n)", "O(n²)"]},
        "Hard": {"typical_time": ["O(n²)", "O(2^n)", "O(n!)", "O(n³)"],
                  "typical_space": ["O(n)", "O(n²)", "O(2^n)"]},
    }
    return ranges.get(difficulty, ranges["Easy"])



def format_alternatives_text(alternatives: list, preferred_time: str, preferred_space: str) -> str:
    """Format alternatives as bullet points for the guideline text."""
    parts = []
    for alt in alternatives:
        parts.append(
            f"  {alt['approach']} ({alt['complexity']['time']} time, "
            f"{alt['complexity']['space']} space): {alt['when']}"
        )
    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────
# 3. DATA LOADING
# ────────────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_topic_list(topics_field) -> list:
    """Normalize topics field to a list of strings."""
    if isinstance(topics_field, list):
        result = []
        for t in topics_field:
            if isinstance(t, dict):
                result.append(t.get("title", str(t)))
            else:
                result.append(str(t).strip())
        return result
    elif isinstance(topics_field, str):
        return [t.strip() for t in topics_field.split(",") if t.strip()]
    return []


def load_all_problems() -> list:
    """Load all problems from all 4 sources, merging where possible."""
    
    # Load master list from Leetcode.csv first
    problems = {}
    for fname in os.listdir(LC_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(LC_DIR, fname)) as f:
            d = json.load(f)
        p = d["data"]
        pid = int(p["ID"])
        problems[pid] = {
            "id": pid,
            "title": p.get("Title", ""),
            "difficulty": p.get("Difficulty", ""),
            "topics_raw": p.get("Topics", []),
            "category": p.get("Category", ""),
            "acceptance_rate": p.get("Acceptance Rate (%)", ""),
            "likes": p.get("Likes", "0"),
            "dislikes": p.get("Dislikes", "0"),
            "premium": p.get("Premium Only", "False"),
            "similar_questions_raw": p.get("Similar Questions", "[]"),
            "link": p.get("Link", ""),
        }
    
    # Supplement with descriptions, solution code from leetcode_problems.csv
    for fname in os.listdir(PROB_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(PROB_DIR, fname)) as f:
            d = json.load(f)
        pid = d["_problem_id"]
        if pid not in problems:
            continue
        data = d["data"]
        problems[pid].update({
            "description_html": data.get("description", ""),
            "description": strip_html(data.get("description", "")),
            "solution_code_python": data.get("solution_code_python", ""),
            "solution_code_java": data.get("solution_code_java", ""),
            "solution_code_cpp": data.get("solution_code_cpp", ""),
            "solution_text": data.get("solution", "")[:500] if data.get("solution") else "",
            "hints_json": data.get("hints", "[]"),
            "stats": data.get("stats", "{}"),
            "title_slug": data.get("titleSlug", ""),
            "paid_only": data.get("paidOnly", "False"),
            "topics_list": data.get("topics", []),
        })
    
    # Supplement with hints from leetcode_questions.csv
    for fname in os.listdir(QUEST_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(QUEST_DIR, fname)) as f:
            d = json.load(f)
        pid = d["_problem_id"]
        if pid not in problems:
            continue
        data = d["data"]
        hint = data.get("Hints", "")
        if hint:
            problems[pid]["hints_text"] = hint
        if not problems[pid].get("description"):
            problems[pid]["description"] = data.get("Question Text", "")
    
    return list(problems.values())


# ────────────────────────────────────────────────────────────────────
# 4. GUIDELINE GENERATION
# ────────────────────────────────────────────────────────────────────

def build_situation_text(problem: dict, pattern_info: dict, template: dict) -> str:
    """Build the situation description from problem context + template."""
    topics = get_topic_list(problem.get("topics_raw", []))
    topics_str = ", ".join(topics[:3]) if topics else pattern_info.get("pattern_name", "this pattern")
    difficulty = problem.get("difficulty", "Medium")
    
    # Start with the template's situation
    situation = template["situation"]
    
    # Add specific context from the problem
    desc = problem.get("description", "")
    # Extract key constraint from description
    constraints_found = []
    if "sorted" in desc.lower():
        constraints_found.append("sorted input")
    if "unsorted" in desc.lower() or "not sorted" in desc.lower():
        constraints_found.append("unsorted input")
    if "duplicate" in desc.lower():
        constraints_found.append("possible duplicates")
    if "contiguous" in desc.lower() or "subarray" in desc.lower() or "substring" in desc.lower():
        constraints_found.append("contiguous sub-structure")
    if "shortest" in desc.lower():
        constraints_found.append("shortest path or minimum length")
    if "unique" in desc.lower():
        constraints_found.append("uniqueness requirement")
    
    constraint_note = ""
    if constraints_found:
        constraint_note = f" The specific context involves {', '.join(constraints_found[:3])}."
    
    return f"{situation.strip()}{constraint_note}"


def build_guideline_text(problem: dict, pattern_info: dict, template: dict) -> str:
    """Build the specific guideline from template + problem context."""
    title = problem.get("title", "")
    difficulty = problem.get("difficulty", "Easy")
    topics = get_topic_list(problem.get("topics_raw", []))
    topics_str = ", ".join(topics[:3]) if topics else ""
    
    guideline = template["guideline"]
    
    # For problems with solution code, add specificity from the code analysis
    code = problem.get("solution_code_python", "")
    code_analysis = detect_code_patterns(code)
    
    # If we detected specific patterns from code, add concrete detail
    specifics = []
    if "hash_map" in code_analysis["data_structures"]:
        specifics.append("Uses a hash map to track seen elements with O(1) access.")
    if code_analysis["pointer_technique"]:
        specifics.append("Employs pointer-based traversal to scan the data in a single pass.")
    if code_analysis["uses_sorting"]:
        specifics.append("Sorts the input as a preprocessing step to enable ordered traversal.")
    if code_analysis["uses_recursion"]:
        specifics.append("Recursively decomposes the problem into smaller subproblems.")
    
    if specifics:
        guideline += "\n\nIn the specific case of \"" + title + "\": "
        guideline += " ".join(specifics)
    
    return guideline


def build_reasoning_text(problem: dict, template: dict, time_hint: str, space_hint: str) -> str:
    """Build the reasoning/why section."""
    reasoning = template["reasoning"]
    
    # Append trade-off specific to this problem
    # Try to extract acceptance rate as a signal
    acc_rate = problem.get("acceptance_rate", "")
    likes = problem.get("likes", "0")
    try:
        likes_int = int(likes)
    except (ValueError, TypeError):
        likes_int = 0
    
    if likes_int > 10000:
        reasoning += f" This problem ({likes_int}+ likes) is widely recognized as a canonical example of this pattern in practice."
    
    return reasoning


def build_constraints(problem: dict, template: dict) -> list:
    """Build the list of constraints for when this guideline applies."""
    constraints = list(template["constraints"])  # copy
    
    # Add problem-specific constraints
    desc = problem.get("description", "").lower()
    premium = problem.get("premium", "False")
    
    if premium == "True":
        constraints.append("This is a premium problem — the guideline is inferred from similar non-premium problems.")
    
    if "follow-up" in desc or "follow up" in desc:
        constraints.append("A follow-up optimization is expected — the shown solution may not be the final optimal form.")
    
    return constraints


def extract_situation_tags(problem: dict) -> dict:
    """Extract structured situation metadata for bridging later."""
    desc = problem.get("description", "").lower()
    
    return {
        "has_sorted_input": "sorted" in desc,
        "has_unsorted_input": "unsorted" in desc or "not sorted" in desc,
        "has_duplicates": "duplicate" in desc,
        "has_contiguous": "contiguous" in desc or "subarray" in desc or "substring" in desc,
        "has_unique_elements": "unique" in desc,
        "is_search_problem": "search" in desc or "find" in desc[:200],
        "is_optimization_problem": "minimum" in desc or "maximum" in desc or "shortest" in desc or "longest" in desc,
        "requires_path": "path" in desc or "route" in desc,
        "requires_ordering": "order" in desc or "sort" in desc,
    }


def generate_guideline(problem: dict, idx: int) -> Optional[dict]:
    """Generate a single guideline JSON from a problem."""
    
    # Skip problems with no real data
    title = problem.get("title", "")
    if not title:
        return None
    
    pid = problem["id"]
    difficulty = problem.get("difficulty", "Easy")
    topics = get_topic_list(problem.get("topics_raw", []))
    
    # Determine pattern from topics
    pattern_id = None
    pattern_name = None
    pattern_category = None
    
    for topic in topics:
        if topic in TOPIC_PATTERN_MAP:
            pid_key, pname, pcat = TOPIC_PATTERN_MAP[topic]
            # Prefer more specific patterns over general ones
            if pattern_id is None or pid_key in ("hash-based-lookup", "two-pointers", "sliding-window"):
                pattern_id = pid_key
                pattern_name = pname
                pattern_category = pcat
    
    # Fallback: try code analysis
    if pattern_id is None:
        code = problem.get("solution_code_python", "")
        code_analysis = detect_code_patterns(code)
        inferred = code_analysis.get("detected_pattern_id")
        if inferred:
            pattern_id = inferred
            pattern_name = inferred.replace("-", " ").title()
            pattern_category = "inferred"
    
    # Last resort: derive from first topic
    if pattern_id is None and topics:
        first_topic = topics[0].lower().replace(" ", "-")
        pattern_id = first_topic
        pattern_name = topics[0]
        pattern_category = "topic-derived"
    
    # Final fallback: uncategorized
    if pattern_id is None:
        pattern_id = "uncategorized"
        pattern_name = "Uncategorized"
        pattern_category = "general"
    
    # Get template
    template = GUIDELINE_TEMPLATES.get(pattern_id, DEFAULT_TEMPLATE)
    
    pattern_info = {
        "pattern_id": pattern_id,
        "pattern_name": pattern_name,
        "category": pattern_category,
    }
    
    # Build complexity from code analysis
    code = problem.get("solution_code_python", "")
    code_analysis = detect_code_patterns(code)
    
    # Build situation
    situation_text = build_situation_text(problem, pattern_info, template)
    situation_tags = extract_situation_tags(problem)
    
    # Build guideline
    guideline_text = build_guideline_text(problem, pattern_info, template)
    
    # Build reasoning
    time_hint = ", ".join(code_analysis.get("complexity_hints", []))
    space_hint = ""
    reasoning_text = build_reasoning_text(problem, template, time_hint, space_hint)
    
    # Build complexity estimate
    if code_analysis["has_nested_loops"] and not (code_analysis["uses_dict"] or code_analysis["uses_set"]):
        time_complexity = "O(n²)"
        space_complexity = "O(1)"
    elif template == DEFAULT_TEMPLATE:
        time_complexity = "varies"
        space_complexity = "varies"
    else:
        # Use pattern-appropriate defaults
        pattern_complexities = {
            "hash-based-lookup": ("O(n)", "O(n)"),
            "two-pointers": ("O(n)", "O(1)"),
            "sliding-window": ("O(n)", "O(1) or O(k)"),
            "binary-search": ("O(log n)", "O(1)"),
            "dfs": ("O(V+E)", "O(h)"),
            "bfs": ("O(V+E)", "O(V)"),
            "dynamic-programming": ("O(n * state)", "O(n) or O(1)"),
            "heap": ("O(n log k)", "O(k)"),
            "stack": ("O(n)", "O(n)"),
            "monotonic-stack": ("O(n)", "O(n)"),
            "backtracking": ("O(k^n) worst", "O(n)"),
            "prefix-sum": ("O(n)", "O(n)"),
            "union-find": ("O(α(n)) amortized", "O(n)"),
            "tree-traversal": ("O(n)", "O(h)"),
            "graph-algorithms": ("O(V+E)", "O(V)"),
            "string-processing": ("O(n)", "O(n)"),
            "bit-manipulation": ("O(n)", "O(1)"),
            "greedy": ("O(n log n)", "O(1)"),
            "trie": ("O(k)", "O(total_chars)"),
            "linked-list": ("O(n)", "O(1)"),
            "queue": ("O(n)", "O(n)"),
            "counting": ("O(n)", "O(k)"),
            "sorting": ("O(n log n)", "O(n)"),
            "matrix-grid": ("O(mn)", "O(mn)"),
            "mathematical-reasoning": ("O(1) to O(n)", "O(1)"),
            "simulation": ("O(steps)", "O(state)"),
            "database": ("O(n log n)", "O(n)"),
            "design-pattern": ("O(n)", "O(n)"),
        }
        tc, sc = pattern_complexities.get(pattern_id, ("varies", "varies"))
        time_complexity = tc
        space_complexity = sc
    
    # Build alternatives from template
    alternatives = template["alternatives"]
    
    # Build constraints
    constraints = build_constraints(problem, template)
    
    # Parse similar questions
    similar_raw = problem.get("similar_questions_raw", [])
    similar_questions = []
    if isinstance(similar_raw, list):
        for sq in similar_raw:
            if isinstance(sq, dict):
                similar_questions.append({
                    "title": sq.get("title", ""),
                    "id": sq.get("titleSlug", ""),
                })
            elif isinstance(sq, str):
                similar_questions.append({"title": sq, "id": sq.lower().replace(" ", "-")})
    
    # Detect if hints exist
    hints = problem.get("hints_json", "") or problem.get("hints_text", "")
    has_hints = bool(hints and len(hints) > 10)
    
    # Build the guideline object
    guideline = {
        "guideline_id": f"guideline_{pid:05d}",
        "schema_version": "2.0",
        "source_problem_id": pid,
        "source_dataset": "Leetcode.csv",
        "title": title,
        "title_slug": problem.get("title_slug", ""),
        "link": problem.get("link", ""),
        
        "situation": {
            "summary": situation_text,
            "tags": situation_tags,
            "difficulty": difficulty,
            "topics": topics,
        },
        
        "guideline": guideline_text,
        "reasoning": reasoning_text,
        
        "complexity": {
            "time": time_complexity,
            "space": space_complexity,
            "code_hints": code_analysis["complexity_hints"],
        },
        
        "pattern": pattern_info,
        "code_analysis": code_analysis,
        
        "constraints": constraints,
        "alternatives": alternatives,
        
        "has_solution_code": bool(code),
        "has_hints": has_hints,
        
        "similar_questions": similar_questions[:10],  # limit to top 10
        
        "acceptance_rate": problem.get("acceptance_rate", ""),
        "likes": problem.get("likes", "0"),
        
        # Bridges — to be filled in Phase 3
        "bridges": {
            "code_repos": [],
            "code_chunks": [],
            "textbook_chapters": [],
            "diagrams": [],
            "pattern_links": [],
        },
        
        "metadata": {
            "premium": problem.get("premium", "False"),
            "category": problem.get("category", ""),
            "stats": problem.get("stats", "{}"),
        },
    }
    
    return guideline


# ────────────────────────────────────────────────────────────────────
# 5. MAIN
# ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 2: Bottom-Up Guideline Extractor")
    print("=" * 60)
    print("100% deterministic — no LLM involvement")
    print("=" * 60)
    
    # Load all problems
    problems = load_all_problems()
    print(f"\nLoaded {len(problems)} problems from all sources")
    
    # Track stats
    stats = {
        "total": 0,
        "extracted": 0,
        "skipped_no_title": 0,
        "patterns_found": {},
    }
    
    for idx, problem in enumerate(problems):
        stats["total"] += 1
        guideline = generate_guideline(problem, idx)
        
        if guideline is None:
            stats["skipped_no_title"] += 1
            continue
        
        # Track pattern
        p_id = guideline["pattern"]["pattern_id"]
        if p_id not in stats["patterns_found"]:
            stats["patterns_found"][p_id] = 0
        stats["patterns_found"][p_id] += 1
        
        # Write to file
        outpath = os.path.join(OUTPUT_DIR, f"{guideline['guideline_id']}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(guideline, f, indent=2, ensure_ascii=False)
        
        stats["extracted"] += 1
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Guideline Extraction Summary:")
    print(f"  Total problems loaded: {stats['total']}")
    print(f"  Guidelines extracted:  {stats['extracted']}")
    print(f"  Skipped (empty title): {stats['skipped_no_title']}")
    print(f"\n  Patterns found: {len(stats['patterns_found'])}")
    for p_id, count in sorted(stats['patterns_found'].items(), key=lambda x: -x[1]):
        print(f"    {p_id:25s}: {count:5d}")
    
    print(f"\n  Output: {OUTPUT_DIR}/")
    
    # Write pattern index
    index_path = os.path.join(OUTPUT_DIR, "_pattern_index.json")
    pattern_index = [
        {"pattern_id": p_id, "pattern_name": GUIDELINE_TEMPLATES.get(p_id, {}).get("guideline", "")[:50],
         "count": count, "category": GUIDELINE_TEMPLATES.get(p_id, {}).get("category", "general")}
        for p_id, count in sorted(stats['patterns_found'].items(), key=lambda x: -x[1])
    ]
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(pattern_index, f, indent=2, ensure_ascii=False)
    print(f"  Pattern index: {index_path}")
    
    # Write difficulty index
    diff_counts = {}
    for p in problems:
        d = p.get("difficulty", "Unknown")
        diff_counts[d] = diff_counts.get(d, 0) + 1
    print(f"\n  Difficulty distribution: {diff_counts}")
    
    print(f"{'='} * 60")


if __name__ == "__main__":
    main()
