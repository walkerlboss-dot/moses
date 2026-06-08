"""
Safe Code Mutator for Moses v4.0

Parses Python source into an AST and applies a whitelist of safe mutations.
Never touches imports, class definitions, or safety-critical regions.
Generates unified diffs for human review.
"""

from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import inspect
import random
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------

_PROTECTED_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.ClassDef,
)

_SAFETY_CRITICAL_PATTERNS = [
    r"def\s+__init__\b",
    r"def\s+__enter__\b",
    r"def\s+__exit__\b",
    r"def\s+rollback\b",
    r"def\s+restore\b",
    r"class\s+Safety",
    r"class\s+Guard",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mutation:
    """A single mutation operation."""
    name: str
    node_type: str
    line_start: int
    line_end: int
    description: str


@dataclass
class Mutant:
    """A mutated version of a source file."""
    source: str
    original_hash: str
    mutations: List[Mutation] = field(default_factory=list)
    fitness: Optional[float] = None

    def diff(self, original_source: str) -> str:
        """Unified diff against original source."""
        original_lines = original_source.splitlines(keepends=True)
        mutant_lines = self.source.splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                original_lines,
                mutant_lines,
                fromfile="original.py",
                tofile="mutant.py",
                lineterm="",
            )
        )


# ---------------------------------------------------------------------------
# Core mutator
# ---------------------------------------------------------------------------

class CodeMutator:
    """
    Safely mutate Python source code using AST transformations.

    Usage:
        mutator = CodeMutator()
        mutant = mutator.mutate_file("path/to/file.py", mutation_count=3)
        print(mutant.diff(original_source))
    """

    def __init__(
        self,
        protected_patterns: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        self.protected_patterns = protected_patterns or _SAFETY_CRITICAL_PATTERNS
        self._rng = random.Random(seed)
        self._mutation_registry: List[Callable[[ast.AST], Optional[ast.AST]]] = [
            self._mutate_numeric_constant,
            self._mutate_string_constant,
            self._swap_adjacent_functions,
            self._add_early_return,
            self._negate_boolean,
            self._swap_binop_operands,
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mutate_file(
        self,
        path: Union[str, Path],
        mutation_count: int = 1,
    ) -> Mutant:
        """Read *path*, apply *mutation_count* mutations, return Mutant."""
        path = Path(path)
        original_source = path.read_text(encoding="utf-8")
        return self.mutate_source(original_source, mutation_count)

    def mutate_source(
        self,
        source: str,
        mutation_count: int = 1,
    ) -> Mutant:
        """Apply *mutation_count* mutations to *source* and return Mutant."""
        original_hash = hashlib.sha256(source.encode()).hexdigest()[:16]
        tree = ast.parse(source)

        if not self._is_safe_to_mutate(tree, source):
            raise UnsafeMutationError("Source contains protected constructs that block mutation.")

        mutations_log: List[Mutation] = []
        current_source = source

        for _ in range(mutation_count):
            tree = ast.parse(current_source)
            mutator = self._rng.choice(self._mutation_registry)
            new_tree, mutation = mutator(tree)
            if mutation is not None:
                try:
                    current_source = ast.unparse(new_tree)
                    mutations_log.append(mutation)
                except Exception as exc:
                    # Unparse failed; keep previous source
                    pass

        return Mutant(
            source=current_source,
            original_hash=original_hash,
            mutations=mutations_log,
        )

    def list_possible_mutations(self, source: str) -> List[Mutation]:
        """Return a list of mutations that *could* be applied (dry-run)."""
        tree = ast.parse(source)
        possible: List[Mutation] = []
        for mutator in self._mutation_registry:
            _, mutation = mutator(tree)
            if mutation is not None:
                possible.append(mutation)
        return possible

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def _is_safe_to_mutate(self, tree: ast.AST, source: str) -> bool:
        """Return False if source matches safety-critical patterns."""
        for node in ast.walk(tree):
            if isinstance(node, _PROTECTED_NODE_TYPES):
                return False
        for pattern in self.protected_patterns:
            if re.search(pattern, source):
                return False
        return True

    def _is_protected_function(self, node: ast.FunctionDef) -> bool:
        """Check if a function is safety-critical by name."""
        for pattern in self.protected_patterns:
            if re.search(pattern, f"def {node.name}"):
                return True
        return False

    # ------------------------------------------------------------------
    # Mutation operators
    # ------------------------------------------------------------------

    def _mutate_numeric_constant(self, tree: ast.AST) -> Tuple[ast.AST, Optional[Mutation]]:
        """Change a numeric constant by +/- 1 or scale by 2."""
        candidates = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ]
        if not candidates:
            return tree, None

        target = self._rng.choice(candidates)
        original_value = target.value
        operation = self._rng.choice(["inc", "dec", "double", "half"])

        if operation == "inc":
            new_value = original_value + 1
        elif operation == "dec":
            new_value = original_value - 1
        elif operation == "double":
            new_value = original_value * 2
        else:  # half
            new_value = original_value / 2 if original_value != 0 else 0

        target.value = type(original_value)(new_value) if isinstance(new_value, float) and isinstance(original_value, int) else new_value

        mutation = Mutation(
            name="numeric_constant",
            node_type="Constant",
            line_start=getattr(target, "lineno", 0),
            line_end=getattr(target, "end_lineno", 0),
            description=f"Changed numeric constant {original_value} -> {new_value}",
        )
        return tree, mutation

    def _mutate_string_constant(self, tree: ast.AST) -> Tuple[ast.AST, Optional[Mutation]]:
        """Append or prepend a random character to a string constant."""
        candidates = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        if not candidates:
            return tree, None

        target = self._rng.choice(candidates)
        original = target.value
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        op = self._rng.choice(["append", "prepend"])
        ch = self._rng.choice(chars)
        new_value = original + ch if op == "append" else ch + original
        target.value = new_value

        mutation = Mutation(
            name="string_constant",
            node_type="Constant",
            line_start=getattr(target, "lineno", 0),
            line_end=getattr(target, "end_lineno", 0),
            description=f"Modified string constant (len {len(original)} -> {len(new_value)})",
        )
        return tree, mutation

    def _swap_adjacent_functions(self, tree: ast.AST) -> Tuple[ast.AST, Optional[Mutation]]:
        """Swap order of two adjacent top-level function definitions."""
        funcs = [
            (idx, node) for idx, node in enumerate(tree.body)
            if isinstance(node, ast.FunctionDef) and not self._is_protected_function(node)
        ]
        if len(funcs) < 2:
            return tree, None

        idx1, idx2 = self._rng.sample(range(len(funcs)), 2)
        pos1, _ = funcs[idx1]
        pos2, _ = funcs[idx2]
        tree.body[pos1], tree.body[pos2] = tree.body[pos2], tree.body[pos1]

        mutation = Mutation(
            name="swap_functions",
            node_type="FunctionDef",
            line_start=min(pos1, pos2),
            line_end=max(pos1, pos2),
            description=f"Swapped functions at positions {pos1} and {pos2}",
        )
        return tree, mutation

    def _add_early_return(self, tree: ast.AST) -> Tuple[ast.AST, Optional[Mutation]]:
        """Add an early return in a function if a simple condition is met."""
        funcs = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and not self._is_protected_function(node)
            and len(node.body) > 1
        ]
        if not funcs:
            return tree, None

        target = self._rng.choice(funcs)
        # Insert 'if not True: return None' at start -- semantically safe no-op
        guard = ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=ast.Constant(value=True)),
            body=[ast.Return(value=ast.Constant(value=None))],
            orelse=[],
        )
        ast.copy_location(guard, target.body[0])
        target.body.insert(0, guard)

        mutation = Mutation(
            name="early_return",
            node_type="FunctionDef",
            line_start=target.lineno,
            line_end=target.end_lineno or target.lineno,
            description=f"Added no-op early return to function '{target.name}'",
        )
        return tree, mutation

    def _negate_boolean(self, tree: ast.AST) -> Tuple[ast.AST, Optional[Mutation]]:
        """Negate a boolean constant."""
        candidates = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, bool)
        ]
        if not candidates:
            return tree, None

        target = self._rng.choice(candidates)
        target.value = not target.value

        mutation = Mutation(
            name="negate_boolean",
            node_type="Constant",
            line_start=getattr(target, "lineno", 0),
            line_end=getattr(target, "end_lineno", 0),
            description=f"Negated boolean constant to {target.value}",
        )
        return tree, mutation

    def _swap_binop_operands(self, tree: ast.AST) -> Tuple[ast.AST, Optional[Mutation]]:
        """Swap left and right operands of a commutative binary operation."""
        commutative_ops = (ast.Add, ast.Mult, ast.BitOr, ast.BitXor, ast.And, ast.Or)
        candidates = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.BinOp) and isinstance(node.op, commutative_ops)
        ]
        if not candidates:
            return tree, None

        target = self._rng.choice(candidates)
        target.left, target.right = target.right, target.left

        mutation = Mutation(
            name="swap_binop",
            node_type="BinOp",
            line_start=getattr(target, "lineno", 0),
            line_end=getattr(target, "end_lineno", 0),
            description="Swapped commutative binary operands",
        )
        return tree, mutation


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class UnsafeMutationError(Exception):
    """Raised when a mutation would violate safety constraints."""
    pass
