#!/usr/bin/env python3
"""
Generic Guideline Generator — CSV/Excel → Guideline JSONs
==========================================================
Converts any CSV or Excel file with the requisite columns into individual
guideline_NNNNN.json files compatible with the platform's guideline data model.

Features:
  - Handles .csv, .xlsx, and .xls inputs (requires pandas for Excel)
  - Column mapping via --column-map JSON (adapts to any CSV schema)
  - Reuses the 56-pattern TOPIC_PATTERN_MAP and GUIDELINE_TEMPLATES engine
  - 100% deterministic — no LLM involvement
  - Idempotent — re-running produces identical output

Usage:
    python3 generate_guidelines_from_csv.py \\
        --input my_problems.csv \\
        --output /path/to/guidelines/ \\
        --dataset-name "MyDataset"

    python3 generate_guidelines_from_csv.py \\
        --input my_problems.xlsx \\
        --output /path/to/guidelines/ \\
        --column-map my_column_mapping.json

See CSV_COLUMN_REFERENCE.md for full column documentation.
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Optional

# ── Try pandas for Excel support ───────────────────────────────────
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ══════════════════════════════════════════════════════════════════════
# 1. PATTERN DETECTION — Code Analysis
# ══════════════════════════════════════════════════════════════════════

TOPIC_PATTERN_MAP: dict[str, tuple[str, str, str]] = {
    "Hash Table": ("hash-based-lookup", "Hash-based Lookup", "lookup"),
    "Prefix Sum": ("prefix-sum", "Prefix Sum", "accumulation"),
    "Counting": ("counting", "Counting", "frequency"),
    "Two Pointers": ("two-pointers", "Two Pointers", "scanning"),
    "Sliding Window": ("sliding-window", "Sliding Window", "window"),
    "Binary Search": ("binary-search", "Binary Search", "divide"),
    "Depth-First Search": ("dfs", "DFS", "traversal"),
    "Breadth-First Search": ("bfs", "BFS", "traversal"),
    "Backtracking": ("backtracking", "Backtracking", "exploration"),
    "Divide and Conquer": ("divide-and-conquer", "Divide and Conquer", "divide"),
    "Dynamic Programming": ("dynamic-programming", "Dynamic Programming", "optimization"),
    "Sorting": ("sorting", "Sorting", "ordering"),
    "Heap (Priority Queue)": ("heap", "Heap / Priority Queue", "ordering"),
    "Stack": ("stack", "Stack", "lifo"),
    "Monotonic Stack": ("monotonic-stack", "Monotonic Stack", "lifo"),
    "Queue": ("queue", "Queue", "fifo"),
    "Tree": ("tree-traversal", "Tree Traversal", "traversal"),
    "Binary Tree": ("binary-tree", "Binary Tree", "traversal"),
    "Graph": ("graph-algorithms", "Graph Algorithms", "traversal"),
    "Union Find": ("union-find", "Union-Find / DSU", "connectivity"),
    "Trie": ("trie", "Trie / Prefix Tree", "retrieval"),
    "Topological Sort": ("topological-sort", "Topological Sort", "ordering"),
    "Linked List": ("linked-list", "Linked List", "linear"),
    "Bit Manipulation": ("bit-manipulation", "Bit Manipulation", "bitwise"),
    "String": ("string-processing", "String Processing", "transformation"),
    "Matrix": ("matrix-grid", "Matrix / Grid", "grid"),
    "Simulation": ("simulation", "Simulation", "simulation"),
    "Design": ("design-pattern", "Design Pattern", "design"),
    "Math": ("mathematical-reasoning", "Mathematical Reasoning", "mathematical"),
    "Greedy": ("greedy", "Greedy", "optimization"),
    "Memoization": ("memoization", "Memoization", "optimization"),
    "Recursion": ("recursion", "Recursion", "divide"),
    "Ordered Set": ("ordered-set", "Ordered Set", "ordering"),
    "Bucket Sort": ("bucket-sort", "Bucket Sort", "ordering"),
    "Radix Sort": ("radix-sort", "Radix Sort", "ordering"),
    "Counting Sort": ("counting-sort", "Counting Sort", "ordering"),
    "Merge Sort": ("merge-sort", "Merge Sort", "divide"),
    "Quickselect": ("quickselect", "Quickselect", "selection"),
    "Reservoir Sampling": ("reservoir-sampling", "Reservoir Sampling", "sampling"),
    "Rejection Sampling": ("rejection-sampling", "Rejection Sampling", "sampling"),
    "Geometry": ("geometry", "Geometry", "mathematical"),
    "Number Theory": ("number-theory", "Number Theory", "mathematical"),
    "Combinatorics": ("combinatorics", "Combinatorics", "mathematical"),
    "Game Theory": ("game-theory", "Game Theory", "mathematical"),
    "Probability and Statistics": ("probability", "Probability / Statistics", "mathematical"),
    "Randomized": ("randomized", "Randomized", "randomized"),
    "Rolling Hash": ("rolling-hash", "Rolling Hash", "hashing"),
    "Hash Function": ("hash-function", "Hash Function", "hashing"),
    "Database": ("database", "Database", "data-management"),
    "Shell": ("shell", "Shell", "automation"),
    "Concurrency": ("concurrency", "Concurrency", "concurrency"),
    "Iterator": ("iterator", "Iterator", "iteration"),
    "Brainteaser": ("brainteaser", "Brainteaser", "puzzle"),
    "Data Stream": ("data-stream", "Data Stream", "streaming"),
    "Interactive": ("interactive", "Interactive", "interactive"),
    "Shortest Path": ("shortest-path", "Shortest Path", "pathfinding"),
    "Minimum Spanning Tree": ("mst", "Minimum Spanning Tree", "connectivity"),
    "Strongly Connected Component": ("scc", "SCC", "connectivity"),
    "Eulerian Circuit": ("eulerian-circuit", "Eulerian Circuit", "traversal"),
    "Line Sweep": ("line-sweep", "Line Sweep", "sweeping"),
    "Doubly-Linked List": ("doubly-linked-list", "Doubly-Linked List", "linear"),
    "Binary Indexed Tree": ("fenwick-tree", "Fenwick Tree / BIT", "accumulation"),
    "Segment Tree": ("segment-tree", "Segment Tree", "range-query"),
    "Skip List": ("skip-list", "Skip List", "ordering"),
    "Biconnected Component": ("biconnected", "Biconnected Component", "connectivity"),
    "String Matching": ("string-matching", "String Matching", "pattern-matching"),
    "Suffix Array": ("suffix-array", "Suffix Array", "string-index"),
    "Meet in the Middle": ("meet-in-middle", "Meet in the Middle", "divide"),
}

# Code pattern inference from structural signatures in solution code
INFERRED_PATTERNS: dict[str, str] = {
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

# ══════════════════════════════════════════════════════════════════════
# 2. DEFAULT COLUMN MAPPING
# ══════════════════════════════════════════════════════════════════════

# Maps standardized field names → CSV column name.
# Override with --column-map JSON file.
DEFAULT_COLUMN_MAP: dict[str, str] = {
    "id": "id",
    "title": "title",
    "title_slug": "title_slug",
    "link": "link",
    "difficulty": "difficulty",
    "topics_raw": "topics_raw",
    "description": "description",
    "solution_code_python": "solution_code_python",
    "acceptance_rate": "acceptance_rate",
    "likes": "likes",
    "premium": "premium",
    "category": "category",
    "hints_text": "hints_text",
    "similar_questions_raw": "similar_questions_raw",
    "tags_structured": "tags_structured",
    "pattern_override": "pattern_override",
}

# ══════════════════════════════════════════════════════════════════════
# 3. GUIDELINE TEMPLATES (abridged — same as extract_guidelines.py)
# ══════════════════════════════════════════════════════════════════════

# Default template used when no pattern-specific template exists
DEFAULT_TEMPLATE: dict[str, Any] = {
    "category": "general",
    "situation": "A general algorithmic problem.",
    "guideline": "Analyze the problem constraints, data structures, and input characteristics to select the most appropriate algorithmic approach.",
    "reasoning": "Different problem characteristics favor different algorithmic paradigms. Matching the approach to the problem structure yields the best complexity trade-offs.",
    "constraints": [
        "Solution depends on specific problem constraints.",
        "May require combining multiple patterns for optimal solution.",
    ],
    "alternatives": [],
}

# Full template library — pattern_id → template dict
GUIDELINE_TEMPLATES: dict[str, dict[str, Any]] = {
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
            "Space overhead scales with the number of unique elements stored.",
        ],
        "alternatives": [
            {
                "approach": "Brute force nested iteration",
                "complexity": {"time": "O(n²)", "space": "O(1)"},
                "when": "Collection is very small (n < 50) or memory-constrained.",
            },
            {
                "approach": "Sort + binary search / two pointers",
                "complexity": {"time": "O(n log n)", "space": "O(1) or O(n)"},
                "when": "Collection is sortable and multiple queries are needed against the same data.",
            },
        ],
    },
    "two-pointers": {
        "category": "scanning",
        "situation": (
            "Processing a sorted array or linked list where you need to find pairs, "
            "triplets, or subarrays that satisfy a condition, and the collection can "
            "be traversed from both ends."
        ),
        "guideline": (
            "Initialize two pointers at opposite ends (or both at start with different speeds). "
            "Move pointers inward based on comparison with target: if sum < target, move left "
            "pointer right to increase sum; if sum > target, move right pointer left to decrease. "
            "For in-place array modification (e.g., removing duplicates), use a slow pointer to "
            "track the write position and a fast pointer to scan."
        ),
        "reasoning": (
            "Two pointers eliminate the need for nested iteration by exploiting sorted order or "
            "spatial relationships. Each pointer moves at most n steps, yielding O(n) time with "
            "O(1) extra space."
        ),
        "constraints": [
            "Array must be sorted (or sortable) for opposite-end two-pointer technique.",
            "In-place modification requires the collection to be mutable.",
        ],
        "alternatives": [
            {
                "approach": "Hash map for complement lookup",
                "complexity": {"time": "O(n)", "space": "O(n)"},
                "when": "Array is unsorted and cannot be sorted (e.g., original indices needed).",
            },
        ],
    },
    "sliding-window": {
        "category": "window",
        "situation": (
            "Processing contiguous subarrays or substrings where you need to find optimal "
            "windows satisfying constraints (max/min sum, distinct elements, longest substring "
            "without repeating characters, etc.)."
        ),
        "guideline": (
            "Maintain a window [left, right] that expands by moving right forward. When the window "
            "violates constraints, shrink by moving left forward until constraints are satisfied again. "
            "Track the optimal window state (size, sum, etc.) at each valid configuration."
        ),
        "reasoning": (
            "The sliding window converts what would be O(n²) nested iteration into O(n) by ensuring "
            "each element is added and removed at most once. The window expands/shrinks monotonically."
        ),
        "constraints": [
            "Problem must involve contiguous subarrays/substrings.",
            "Window constraint must be monotonic (if violated at [L,R], also violated at [L,R+1]).",
        ],
        "alternatives": [
            {
                "approach": "Prefix sum with binary search",
                "complexity": {"time": "O(n log n)", "space": "O(n)"},
                "when": "Window size is fixed, or non-contiguous combinations are needed.",
            },
        ],
    },
    "binary-search": {
        "category": "divide",
        "situation": (
            "Searching in a sorted collection, or searching over a monotonic answer space "
            "(e.g., minimum possible value, maximum feasible capacity)."
        ),
        "guideline": (
            "Define a predicate function P(x) → bool that is monotonic (false for x < threshold, "
            "true for x ≥ threshold). Binary search over the answer space: if P(mid) is true, "
            "search left (smaller values); if false, search right. Return the split point."
        ),
        "reasoning": (
            "Binary search reduces O(n) linear scan to O(log n). The key insight is recognizing "
            "monotonicity in the search space — even when the collection itself isn't sorted."
        ),
        "constraints": [
            "The search space must be monotonic with respect to the predicate.",
            "Array variant requires sorted input (or cost of pre-sorting).",
        ],
        "alternatives": [],
    },
    "dynamic-programming": {
        "category": "optimization",
        "situation": (
            "Optimization problems with overlapping subproblems and optimal substructure — "
            "the solution can be built from solutions to smaller instances of the same problem."
        ),
        "guideline": (
            "Define a state representation dp[...] that captures the minimal information needed "
            "to make optimal future decisions. Derive a recurrence relation: dp[i] = f(dp[i-1], "
            "dp[i-2], ...). Choose between top-down (memoization) for sparse state spaces or "
            "bottom-up (tabulation) for dense, well-defined iteration order."
        ),
        "reasoning": (
            "DP avoids recomputing subproblems by storing their results. Optimal substructure "
            "guarantees that locally optimal decisions compose to a globally optimal solution."
        ),
        "constraints": [
            "Problem must exhibit optimal substructure.",
            "State space must be manageable (polynomial in input size).",
        ],
        "alternatives": [
            {
                "approach": "Greedy algorithm",
                "complexity": {"time": "O(n log n)", "space": "O(1)"},
                "when": "Problem has greedy-choice property (locally optimal = globally optimal).",
            },
        ],
    },
    "dfs": {
        "category": "traversal",
        "situation": (
            "Exploring all paths in a tree or graph, or when you need to exhaustively search "
            "a state space (e.g., backtracking, topological ordering, cycle detection)."
        ),
        "guideline": (
            "Use recursion or an explicit stack to traverse depth-first. For trees: process current "
            "node, then recursively process children. For graphs: maintain a visited set to avoid "
            "cycles. Choose pre-order (process before children), in-order (between children, "
            "binary trees only), or post-order (after children) based on the problem."
        ),
        "reasoning": (
            "DFS naturally mirrors divide-and-conquer and is memory-efficient on deep narrow graphs "
            "(O(h) stack depth vs BFS's O(w) queue). It's preferred when the solution requires "
            "exploring full paths before backtracking."
        ),
        "constraints": [
            "Recursive DFS risks stack overflow on deep trees (>1000 levels in Python).",
            "Must track visited nodes to avoid infinite loops in cyclic graphs.",
        ],
        "alternatives": [
            {
                "approach": "BFS",
                "complexity": {"time": "O(V+E)", "space": "O(V)"},
                "when": "Shortest path in unweighted graph, or level-order processing needed.",
            },
        ],
    },
    "bfs": {
        "category": "traversal",
        "situation": (
            "Finding the shortest path in an unweighted graph, level-order tree traversal, "
            "or exploring nearest neighbors first."
        ),
        "guideline": (
            "Use a queue. Enqueue the starting node(s). While the queue is not empty, dequeue "
            "a node, process it, and enqueue all unvisited neighbors. Track distance/depth from "
            "start for shortest-path applications."
        ),
        "reasoning": (
            "BFS explores nodes in order of increasing distance from the start, guaranteeing "
            "the first time a node is reached is via the shortest path (in unweighted graphs)."
        ),
        "constraints": [
            "BFS uses O(V) memory for the queue — problematic for very wide graphs.",
            "Only guarantees shortest path in unweighted graphs.",
        ],
        "alternatives": [
            {
                "approach": "DFS with iterative deepening",
                "complexity": {"time": "O(b^d)", "space": "O(d)"},
                "when": "Graph is very wide and depth is bounded.",
            },
        ],
    },
    "backtracking": {
        "category": "exploration",
        "situation": (
            "Generating all valid combinations, permutations, or subsets that satisfy constraints. "
            "The search space is a tree of decisions where some branches can be pruned early."
        ),
        "guideline": (
            "Build the solution incrementally. At each step: (1) make a choice, (2) recurse to "
            "explore that branch, (3) undo the choice (backtrack). Prune branches that cannot "
            "lead to valid solutions based on current partial state."
        ),
        "reasoning": (
            "Backtracking systematically explores the decision tree while pruning invalid branches "
            "early, avoiding full enumeration. The undo step ensures each branch starts from a clean state."
        ),
        "constraints": [
            "Worst-case exponential time — only practical for small input sizes (n ≤ 20).",
            "Pruning heuristics dramatically affect performance.",
        ],
        "alternatives": [],
    },
    "heap": {
        "category": "ordering",
        "situation": (
            "Finding the k smallest/largest elements, merging sorted streams, or maintaining "
            "a dynamic set where you always need the minimum/maximum element."
        ),
        "guideline": (
            "Use a min-heap (Python: heapq) for k-largest or a max-heap (negate values) for "
            "k-smallest. Push elements and pop when heap size exceeds k. The heap root always "
            "holds the k-th largest/smallest element."
        ),
        "reasoning": (
            "Heaps provide O(log k) insertion and O(1) access to the extremum. Processing n elements "
            "with a size-k heap yields O(n log k) total — better than O(n log n) full sort when k ≪ n."
        ),
        "constraints": [
            "Only the root element is guaranteed to be the extremum — not suitable for full ordering.",
        ],
        "alternatives": [
            {
                "approach": "Quickselect",
                "complexity": {"time": "O(n) average", "space": "O(1)"},
                "when": "Only the k-th element is needed (not the full top-k list) and array is mutable.",
            },
        ],
    },
    "stack": {
        "category": "lifo",
        "situation": (
            "Processing nested structures (parentheses, HTML tags), evaluating expressions "
            "(postfix/prefix notation), or when the most recently seen element needs to be "
            "compared against the current element."
        ),
        "guideline": (
            "Push elements onto a stack as you encounter them. When the current element relates "
            "to the top-of-stack (e.g., closing bracket matching opening bracket, or monotonic "
            "property violation), pop and process the relationship."
        ),
        "reasoning": (
            "Stacks provide LIFO access — the most recently seen element is always available. "
            "This naturally models nested/balanced structures and last-seen relationships."
        ),
        "constraints": [
            "Only the top element is accessible — random access is not supported.",
        ],
        "alternatives": [],
    },
    "prefix-sum": {
        "category": "accumulation",
        "situation": (
            "Computing range sums or range operations repeatedly on a static array. "
            "Multiple queries need to be answered efficiently after preprocessing."
        ),
        "guideline": (
            "Precompute a prefix sum array prefix[i] = sum(nums[0..i-1]). Then any range sum "
            "nums[L..R] = prefix[R+1] - prefix[L]. For 2D grids, precompute a 2D prefix matrix: "
            "prefix[i][j] = sum of submatrix (0,0) to (i-1,j-1)."
        ),
        "reasoning": (
            "Prefix sums trade O(n) preprocessing space for O(1) per-query time. Each query avoids "
            "re-scanning the entire range."
        ),
        "constraints": [
            "Array must be static (or prefix array must be rebuilt on mutation).",
            "Only supports associative and invertible operations (sum, XOR; not max/min).",
        ],
        "alternatives": [
            {
                "approach": "Segment Tree / Fenwick Tree",
                "complexity": {"time": "O(log n) per query/update", "space": "O(n)"},
                "when": "Array is updated dynamically between queries.",
            },
        ],
    },
    "union-find": {
        "category": "connectivity",
        "situation": (
            "Determining connected components in a dynamic graph, detecting cycles, or grouping "
            "elements into disjoint sets that can be merged over time."
        ),
        "guideline": (
            "Implement with parent array + rank/size optimization. find(x) locates the set "
            "representative with path compression (recursively reparent nodes to root). "
            "union(x,y) merges two sets by attaching the smaller root to the larger root. "
            "After processing all unions, components are identified by unique find(x) values."
        ),
        "reasoning": (
            "Union-Find achieves near-constant amortized time per operation via two optimizations: "
            "path compression flattens the tree during finds, and union-by-rank keeps trees balanced."
        ),
        "constraints": [
            "Elements must be mappable to integer indices (0 to n-1).",
            "Does not support splitting sets — only merging.",
        ],
        "alternatives": [],
    },
    "greedy": {
        "category": "optimization",
        "situation": (
            "Optimization problems where a locally optimal choice at each step leads to a globally "
            "optimal solution (greedy-choice property). Common in scheduling, interval selection, "
            "and Huffman coding."
        ),
        "guideline": (
            "Sort items by a key metric (e.g., end time for interval scheduling, value/weight ratio "
            "for fractional knapsack). Iterate and select the best available item that satisfies "
            "remaining constraints. Validate that the greedy-choice property holds for the problem."
        ),
        "reasoning": (
            "Greedy algorithms are simple and fast (O(n log n) dominated by sorting) but only work "
            "when optimal substructure guarantees that local optimality implies global optimality."
        ),
        "constraints": [
            "Greedy-choice property must be proven — greedy is incorrect for many problems "
            "(e.g., 0/1 knapsack, longest path).",
        ],
        "alternatives": [
            {
                "approach": "Dynamic Programming",
                "complexity": {"time": "O(n * state)", "space": "O(state)"},
                "when": "Greedy-choice property does not hold (e.g., 0/1 knapsack).",
            },
        ],
    },
    "linked-list": {
        "category": "linear",
        "situation": (
            "Manipulating linked lists: reversing, merging, detecting cycles, finding middle, "
            "or performing operations where O(1) insertion/deletion at known positions matters."
        ),
        "guideline": (
            "Use a dummy head node to simplify edge cases (empty list, operations at head). "
            "For cycle detection, use Floyd's tortoise-and-hare algorithm (slow + fast pointer). "
            "For reversal, maintain prev/curr/next pointers and reverse links in-place."
        ),
        "reasoning": (
            "Linked lists support O(1) insertion/deletion at known positions (vs O(n) for arrays). "
            "Two-pointer techniques on linked lists detect cycles and find middle in O(n) with O(1) space."
        ),
        "constraints": [
            "Random access is O(n) — not suitable for index-based access patterns.",
            "Extra memory overhead per node for the next pointer.",
        ],
        "alternatives": [
            {
                "approach": "Array / ArrayList",
                "complexity": {"time": "O(1) access, O(n) insert/delete", "space": "O(n)"},
                "when": "Random access by index is needed frequently.",
            },
        ],
    },
    "trie": {
        "category": "retrieval",
        "situation": (
            "Storing and searching strings by prefix — autocomplete, spell checking, "
            "dictionary lookups, or querying all keys with a common prefix."
        ),
        "guideline": (
            "Implement a TrieNode with children dict/map and an is_end flag. Insert: traverse/copy "
            "characters, creating missing nodes, mark last node as end. Search: traverse characters; "
            "return is_end flag for exact match, or collect all descendants for prefix query."
        ),
        "reasoning": (
            "Tries provide O(k) lookup/insert per string (where k = string length), independent of "
            "the total number of stored strings. This beats hash tables for prefix-based operations."
        ),
        "constraints": [
            "Memory overhead is high — each node stores a child pointer per possible character.",
            "Not suitable for very large alphabets (e.g., full Unicode).",
        ],
        "alternatives": [
            {
                "approach": "Hash set + prefix iteration",
                "complexity": {"time": "O(n * k)", "space": "O(n)"},
                "when": "Only exact-match lookups are needed, not prefix queries.",
            },
        ],
    },
    "bit-manipulation": {
        "category": "bitwise",
        "situation": (
            "Operations on binary representations: finding single-occurrence elements (XOR), "
            "subset enumeration (bitmask DP), or low-level optimization where bit operations "
            "replace arithmetic."
        ),
        "guideline": (
            "Key techniques: (1) n & (n-1) clears the lowest set bit, (2) n & -n isolates the "
            "lowest set bit, (3) XOR of all elements cancels duplicates leaving the singleton, "
            "(4) bitmask enumeration: for mask in range(1<<n) generates all subsets."
        ),
        "reasoning": (
            "Bit operations execute in O(1) CPU cycles. Bitmask DP compresses state representation "
            "from n! to 2^n, making many combinatorial problems tractable for n ≤ 20."
        ),
        "constraints": [
            "Solution code is often less readable than equivalent array-based approaches.",
            "Practical for n ≤ 20 in DP contexts (2^20 ≈ 1M states).",
        ],
        "alternatives": [],
    },
    "string-processing": {
        "category": "transformation",
        "situation": (
            "Manipulating strings: pattern matching, substring searches, palindrome detection, "
            "anagram grouping, or string transformations."
        ),
        "guideline": (
            "For palindrome checks: expand around center in O(n²) or use Manacher's for O(n). "
            "For substring search: use KMP or rolling hash (Rabin-Karp) for O(n+m). "
            "For anagram grouping: sort each string as key, or use character frequency tuple. "
            "For common prefix queries: use a Trie (see trie template)."
        ),
        "reasoning": (
            "String algorithms exploit character-level properties and the limited alphabet size. "
            "The right approach depends on whether you need single query vs repeated queries, "
            "and whether preprocessing is amortized."
        ),
        "constraints": [
            "Character set matters — Unicode increases alphabet size significantly.",
            "Some algorithms (KMP) have high constant factors for short strings.",
        ],
        "alternatives": [],
    },
}

# ══════════════════════════════════════════════════════════════════════
# 4. PATTERN COMPLEXITY DEFAULTS
# ══════════════════════════════════════════════════════════════════════

PATTERN_COMPLEXITIES: dict[str, tuple[str, str]] = {
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


# ══════════════════════════════════════════════════════════════════════
# 5. HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════


def load_column_map(column_map_path: Optional[str] = None) -> dict[str, str]:
    """Load column mapping from JSON, falling back to DEFAULT_COLUMN_MAP."""
    if column_map_path:
        with open(column_map_path, "r", encoding="utf-8") as f:
            custom = json.load(f)
        # Merge: custom overrides default for matching keys
        merged = dict(DEFAULT_COLUMN_MAP)
        merged.update(custom)
        return merged
    return dict(DEFAULT_COLUMN_MAP)


def read_input(input_path: str, column_map: dict[str, str]) -> list[dict[str, Any]]:
    """Read CSV or Excel file, remap columns, return list of row dicts."""
    ext = os.path.splitext(input_path)[1].lower()

    if ext in (".xlsx", ".xls"):
        if not HAS_PANDAS:
            print(
                "ERROR: pandas is required for Excel files. Install with: pip install pandas openpyxl"
            )
            sys.exit(1)
        df = pd.read_excel(input_path)
        # Convert DataFrame to list of dicts
        rows = df.to_dict(orient="records")
        # Convert numpy types to native Python types
        import numpy as np

        clean_rows: list[dict[str, Any]] = []
        for row in rows:
            clean_row: dict[str, Any] = {}
            for k, v in row.items():
                if isinstance(v, (np.integer,)):
                    clean_row[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean_row[k] = float(v)
                elif isinstance(v, (np.bool_,)):
                    clean_row[k] = bool(v)
                elif isinstance(v, float) and np.isnan(v):
                    clean_row[k] = ""
                else:
                    clean_row[k] = v
            clean_rows.append(clean_row)
        rows = clean_rows
    elif ext == ".csv":
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    else:
        print(f"ERROR: Unsupported file format '{ext}'. Use .csv, .xlsx, or .xls.")
        sys.exit(1)

    # Remap columns using column_map
    # column_map[standard_name] = csv_column_name
    reverse_map = {v: k for k, v in column_map.items()}
    remapped: list[dict[str, Any]] = []
    for row in rows:
        new_row: dict[str, Any] = {}
        for csv_col, value in row.items():
            std_name = reverse_map.get(csv_col, csv_col)
            new_row[std_name] = value
        remapped.append(new_row)

    return remapped


def get_topic_list(topics_raw: Any) -> list[str]:
    """Parse topics from raw input (comma-separated string, list, or JSON array)."""
    if isinstance(topics_raw, list):
        return [str(t).strip() for t in topics_raw if t]
    if isinstance(topics_raw, str):
        s = topics_raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return [t.strip().strip("\"'") for t in json.loads(s)]
            except json.JSONDecodeError:
                pass
        return [t.strip() for t in s.split(",") if t.strip()]
    return []


def detect_code_patterns(code: str) -> dict[str, Any]:
    """Analyze solution code for structural signatures and infer pattern."""
    if not code:
        return {
            "detected_pattern_id": None,
            "complexity_hints": [],
            "has_nested_loops": False,
            "uses_dict": False,
            "uses_set": False,
            "uses_heap": False,
            "uses_deque": False,
            "uses_recursion": False,
            "uses_sort": False,
            "pointer_technique": False,
        }

    code_lower = code.lower()
    result: dict[str, Any] = {
        "detected_pattern_id": None,
        "complexity_hints": [],
        "has_nested_loops": False,
        "uses_dict": "dict(" in code_lower or "defaultdict" in code_lower or "{}" in code_lower or "hashmap" in code_lower,
        "uses_set": "set(" in code_lower or "set()" in code_lower,
        "uses_heap": "heap" in code_lower,
        "uses_deque": "deque" in code_lower,
        "uses_recursion": "def " in code and code.count("def ") >= 2
        or "self." in code_lower and "return self." in code_lower,
        "uses_sort": ".sort(" in code_lower or "sorted(" in code_lower,
        "pointer_technique": "left" in code_lower and "right" in code_lower,
    }

    # Detect nested loops
    lines = code.split("\n")
    indent_level = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("for ") or stripped.startswith("while "):
            if indent_level > 0:
                result["has_nested_loops"] = True
            indent_level += 1
        elif stripped and not stripped.startswith("#"):
            pass  # keep indent_level

    # Complexity hints
    if result["has_nested_loops"] and not (result["uses_dict"] or result["uses_set"]):
        result["complexity_hints"].append("Nested loops detected — likely O(n²)")
    if result["uses_dict"] or result["uses_set"]:
        result["complexity_hints"].append("Hash structure — amortized O(1) lookups")
    if result["uses_heap"]:
        result["complexity_hints"].append("Heap operations — O(log k) per element")
    if result["pointer_technique"]:
        result["complexity_hints"].append("Two-pointer pattern — O(n) linear scan")
    if result["uses_recursion"]:
        result["complexity_hints"].append("Recursive calls — consider stack depth")

    # Infer pattern from code
    for keyword, pattern_id in INFERRED_PATTERNS.items():
        if keyword in code_lower:
            result["detected_pattern_id"] = pattern_id
            break

    return result


def extract_situation_tags(problem: dict[str, Any]) -> dict[str, bool]:
    """Extract structured situation metadata from description."""
    desc = str(problem.get("description", "")).lower()
    return {
        "has_sorted_input": "sorted" in desc,
        "has_unsorted_input": "unsorted" in desc or "not sorted" in desc,
        "has_duplicates": "duplicate" in desc,
        "has_contiguous": "contiguous" in desc
        or "subarray" in desc
        or "substring" in desc,
        "has_unique_elements": "unique" in desc,
        "is_search_problem": "search" in desc or "find" in desc[:200],
        "is_optimization_problem": "minimum" in desc
        or "maximum" in desc
        or "shortest" in desc
        or "longest" in desc,
        "requires_path": "path" in desc or "route" in desc,
        "requires_ordering": "order" in desc or "sort" in desc,
    }


def build_situation_text(
    problem: dict[str, Any], pattern_info: dict[str, str], template: dict[str, Any]
) -> str:
    """Build a human-readable situation summary."""
    title = problem.get("title", "Unknown Problem")
    difficulty = problem.get("difficulty", "Medium")
    pattern_name = pattern_info.get("pattern_name", "general approach")
    template_situation = template.get("situation", "")

    # Truncate description to first sentence for summary
    desc = str(problem.get("description", ""))
    first_sentence = desc.split(".")[0] if desc else ""
    if len(first_sentence) > 200:
        first_sentence = first_sentence[:197] + "..."

    return (
        f"Problem '{title}' ({difficulty}) requires {pattern_name}. "
        f"{first_sentence}. "
        f"{template_situation}"
    )


def build_guideline_text(
    problem: dict[str, Any], pattern_info: dict[str, str], template: dict[str, Any]
) -> str:
    """Build the best-practice recommendation text."""
    pattern_name = pattern_info.get("pattern_name", "the appropriate approach")
    template_guideline = template.get("guideline", "")

    return f"For this {pattern_name} problem: {template_guideline}"


def build_reasoning_text(
    problem: dict[str, Any],
    template: dict[str, Any],
    time_hint: str,
    space_hint: str,
) -> str:
    """Build the reasoning/justification text."""
    template_reasoning = template.get("reasoning", "")
    parts = [template_reasoning]
    if time_hint:
        parts.append(f"Time analysis: {time_hint}.")
    if space_hint:
        parts.append(f"Space analysis: {space_hint}.")
    return " ".join(parts)


def build_constraints(problem: dict[str, Any], template: dict[str, Any]) -> list[str]:
    """Build constraint list from template."""
    return list(template.get("constraints", []))


def parse_similar_questions(raw: Any) -> list[dict[str, str]]:
    """Parse similar_questions from raw string or JSON."""
    if isinstance(raw, list):
        result: list[dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict):
                result.append(
                    {
                        "title": str(item.get("title", "")),
                        "id": str(item.get("titleSlug", item.get("id", ""))),
                    }
                )
            elif isinstance(item, str):
                result.append({"title": item, "id": item.lower().replace(" ", "-")})
        return result[:10]
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return parse_similar_questions(json.loads(s))
            except json.JSONDecodeError:
                pass
    return []


def parse_tags_structured(raw: Any) -> dict[str, bool]:
    """Parse pre-computed situation tags from JSON string or dict."""
    if isinstance(raw, dict):
        return {k: bool(v) for k, v in raw.items()}
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return {k: bool(v) for k, v in parsed.items()}
            except json.JSONDecodeError:
                pass
    return {}


def generate_guideline(
    problem: dict[str, Any], idx: int, dataset_name: str
) -> Optional[dict[str, Any]]:
    """Generate a single guideline JSON from a problem row dict."""
    title = str(problem.get("title", "")).strip()
    if not title:
        return None

    # Resolve ID
    raw_id = problem.get("id", idx + 1)
    try:
        pid = int(raw_id)
    except (ValueError, TypeError):
        pid = idx + 1

    difficulty = str(problem.get("difficulty", "Medium")).strip()
    topics = get_topic_list(problem.get("topics_raw", ""))

    # ── Pattern detection ──────────────────────────────────────────
    pattern_override = str(problem.get("pattern_override", "")).strip()
    if pattern_override and pattern_override in TOPIC_PATTERN_MAP:
        pid_key, pname, pcat = TOPIC_PATTERN_MAP[pattern_override]
    elif pattern_override and pattern_override in GUIDELINE_TEMPLATES:
        pid_key = pattern_override
        pname = pattern_override.replace("-", " ").title()
        pcat = GUIDELINE_TEMPLATES.get(pattern_override, {}).get("category", "general")
    else:
        # Topic-based pattern detection
        pid_key = None
        pname = None
        pcat = None
        for topic in topics:
            if topic in TOPIC_PATTERN_MAP:
                candidate_id, candidate_name, candidate_cat = TOPIC_PATTERN_MAP[topic]
                # Prefer more specific patterns
                if pid_key is None or candidate_id in (
                    "hash-based-lookup",
                    "two-pointers",
                    "sliding-window",
                ):
                    pid_key = candidate_id
                    pname = candidate_name
                    pcat = candidate_cat

        # Code analysis fallback
        if pid_key is None:
            code = str(problem.get("solution_code_python", ""))
            code_analysis = detect_code_patterns(code)
            inferred = code_analysis.get("detected_pattern_id")
            if inferred:
                pid_key = inferred
                pname = inferred.replace("-", " ").title()
                pcat = "inferred"

        # Topic derivation fallback
        if pid_key is None and topics:
            first_topic = topics[0].lower().replace(" ", "-")
            pid_key = first_topic
            pname = topics[0]
            pcat = "topic-derived"

        # Uncategorized fallback
        if pid_key is None:
            pid_key = "uncategorized"
            pname = "Uncategorized"
            pcat = "general"

    pattern_info = {
        "pattern_id": pid_key,
        "pattern_name": pname,
        "category": pcat,
    }

    # ── Template ───────────────────────────────────────────────────
    template = GUIDELINE_TEMPLATES.get(pid_key, DEFAULT_TEMPLATE)

    # ── Code analysis ──────────────────────────────────────────────
    code = str(problem.get("solution_code_python", "")).strip()
    code_analysis = detect_code_patterns(code)

    # ── Situation ──────────────────────────────────────────────────
    situation_text = build_situation_text(problem, pattern_info, template)
    tags_structured = parse_tags_structured(problem.get("tags_structured", ""))
    if not tags_structured:
        tags_structured = extract_situation_tags(problem)

    # ── Guideline text ─────────────────────────────────────────────
    guideline_text = build_guideline_text(problem, pattern_info, template)

    # ── Reasoning ──────────────────────────────────────────────────
    time_hint = ", ".join(code_analysis.get("complexity_hints", []))
    reasoning_text = build_reasoning_text(problem, template, time_hint, "")

    # ── Complexity ─────────────────────────────────────────────────
    if code_analysis["has_nested_loops"] and not (
        code_analysis["uses_dict"] or code_analysis["uses_set"]
    ):
        time_complexity = "O(n²)"
        space_complexity = "O(1)"
    elif template == DEFAULT_TEMPLATE:
        time_complexity = "varies"
        space_complexity = "varies"
    else:
        tc, sc = PATTERN_COMPLEXITIES.get(pid_key, ("varies", "varies"))
        time_complexity = tc
        space_complexity = sc

    # ── Build output ───────────────────────────────────────────────
    hints = str(problem.get("hints_text", "")).strip()
    has_hints = len(hints) > 10

    guideline: dict[str, Any] = {
        "guideline_id": f"guideline_{pid:05d}",
        "schema_version": "2.0",
        "source_problem_id": pid,
        "source_dataset": dataset_name,
        "title": title,
        "title_slug": str(problem.get("title_slug", "")).strip(),
        "link": str(problem.get("link", "")).strip(),
        "situation": {
            "summary": situation_text,
            "tags": tags_structured,
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
        "constraints": build_constraints(problem, template),
        "alternatives": list(template.get("alternatives", [])),
        "has_solution_code": bool(code),
        "has_hints": has_hints,
        "similar_questions": parse_similar_questions(
            problem.get("similar_questions_raw", "")
        ),
        "acceptance_rate": str(problem.get("acceptance_rate", "")).strip(),
        "likes": str(problem.get("likes", "0")).strip(),
        "bridges": {
            "code_repos": [],
            "code_chunks": [],
            "textbook_chapters": [],
            "diagrams": [],
            "pattern_links": [],
        },
        "metadata": {
            "premium": str(problem.get("premium", "False")).strip(),
            "category": str(problem.get("category", "")).strip(),
            "stats": "{}",
        },
    }

    return guideline


# ══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Generic Guideline Generator — CSV/Excel → Guideline JSONs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_guidelines_from_csv.py --input problems.csv --output ./guidelines/
  python3 generate_guidelines_from_csv.py --input problems.xlsx --output ./guidelines/ --column-map mapping.json
  python3 generate_guidelines_from_csv.py --input problems.csv --output ./out/ --dataset-name "HackerRank"
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input CSV or Excel file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Directory for output guideline JSON files"
    )
    parser.add_argument(
        "--dataset-name",
        "-d",
        default=None,
        help="Dataset name for source_dataset field (default: derived from filename)",
    )
    parser.add_argument(
        "--column-map",
        "-c",
        default=None,
        help="Path to JSON file mapping standard field names to CSV column names",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing files",
    )
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    dataset_name = args.dataset_name or os.path.splitext(os.path.basename(args.input))[
        0
    ]

    print("=" * 60)
    print("Generic Guideline Generator")
    print("=" * 60)
    print(f"  Input:       {args.input}")
    print(f"  Output:      {args.output}")
    print(f"  Dataset:     {dataset_name}")
    print(f"  Column map:  {args.column_map or 'default'}")
    print("=" * 60)

    # Load column mapping
    column_map = load_column_map(args.column_map)

    # Read input
    print("\nReading input...")
    rows = read_input(args.input, column_map)
    print(f"  Loaded {len(rows)} rows")

    if not rows:
        print("ERROR: No data rows found")
        sys.exit(1)

    # Run extraction
    if not args.dry_run:
        os.makedirs(args.output, exist_ok=True)

    stats: dict[str, Any] = {
        "total": 0,
        "extracted": 0,
        "skipped_no_title": 0,
        "patterns_found": {},
        "difficulty_counts": {},
    }

    for idx, row in enumerate(rows):
        stats["total"] += 1
        guideline = generate_guideline(row, idx, dataset_name)

        if guideline is None:
            stats["skipped_no_title"] += 1
            continue

        # Track pattern
        p_id = guideline["pattern"]["pattern_id"]
        stats["patterns_found"][p_id] = stats["patterns_found"].get(p_id, 0) + 1

        # Track difficulty
        diff = guideline["situation"]["difficulty"]
        stats["difficulty_counts"][diff] = stats["difficulty_counts"].get(diff, 0) + 1

        if not args.dry_run:
            outpath = os.path.join(
                args.output, f"{guideline['guideline_id']}.json"
            )
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump(guideline, f, indent=2, ensure_ascii=False)

        stats["extracted"] += 1

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Extraction Summary:")
    print(f"  Total rows:             {stats['total']}")
    print(f"  Guidelines extracted:   {stats['extracted']}")
    print(f"  Skipped (no title):     {stats['skipped_no_title']}")
    print(f"\n  Patterns found:         {len(stats['patterns_found'])}")
    for p_id, count in sorted(
        stats["patterns_found"].items(), key=lambda x: -x[1]
    )[:15]:
        print(f"    {p_id:30s}: {count:5d}")
    if len(stats["patterns_found"]) > 15:
        print(f"    ... and {len(stats['patterns_found']) - 15} more")

    print(f"\n  Difficulty distribution:")
    for diff, count in sorted(stats["difficulty_counts"].items()):
        print(f"    {diff:10s}: {count:5d}")

    if not args.dry_run:
        print(f"\n  Output directory: {args.output}/")
        print(f"  Files written:    {stats['extracted']}")

    # Write pattern index
    if not args.dry_run and stats["patterns_found"]:
        index_path = os.path.join(args.output, "_pattern_index.json")
        pattern_index = [
            {
                "pattern_id": p_id,
                "count": count,
                "category": GUIDELINE_TEMPLATES.get(p_id, {}).get(
                    "category", "general"
                ),
            }
            for p_id, count in sorted(
                stats["patterns_found"].items(), key=lambda x: -x[1]
            )
        ]
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(pattern_index, f, indent=2, ensure_ascii=False)
        print(f"  Pattern index:    {index_path}")

    print(f"\n{'=' * 60}")
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
