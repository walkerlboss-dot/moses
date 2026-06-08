"""
Moses Self-Modification System v4.0

A safe, reversible, evolutionary code improvement framework.

Modules:
    code_mutator  -- AST-based safe code mutations
    ab_tester     -- Side-by-side A/B testing with statistical validation
    rollback      -- Git-based versioning and automatic recovery
    evolution     -- Genetic algorithm for generational code improvement

Safety Guarantees:
    - Imports and class definitions are never modified
    - All mutations generate human-readable diffs
    - Every change is versioned and can be rolled back
    - A/B tests validate improvements before adoption

Example:
    from moses.self_modify import EvolutionEngine
    engine = EvolutionEngine(target_file="moses/brain.py")
    best_variant = engine.evolve(generations=10, population_size=8)
"""

__version__ = "4.0.0"
__all__ = [
    "CodeMutator",
    "ABTester",
    "RollbackManager",
    "EvolutionEngine",
]

from .code_mutator import CodeMutator
from .ab_tester import ABTester
from .rollback import RollbackManager
from .evolution import EvolutionEngine
