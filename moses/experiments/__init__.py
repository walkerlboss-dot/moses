"""
Moses v5.0 Experiments Package

Automated experimentation framework for continuous training.
Provides experiment orchestration, search spaces, budget management,
and statistical comparison for PPO, SAC, GR00T algorithms.
"""

from .runner import ExperimentRunner, ExperimentConfig, ExperimentResult
from .search_space import (
    SearchSpace,
    PPOSearchSpace,
    SACSearchSpace,
    GR00TSearchSpace,
    ArchitectureSearchSpace,
    EnvironmentSearchSpace,
    ComposableSearchSpace,
)
from .budget import BudgetManager, ComputeBudget, ExperimentBudget

__all__ = [
    # Runner
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentResult",
    # Search Spaces
    "SearchSpace",
    "PPOSearchSpace",
    "SACSearchSpace",
    "GR00TSearchSpace",
    "ArchitectureSearchSpace",
    "EnvironmentSearchSpace",
    "ComposableSearchSpace",
    # Budget
    "BudgetManager",
    "ComputeBudget",
    "ExperimentBudget",
]

__version__ = "5.0.0"
