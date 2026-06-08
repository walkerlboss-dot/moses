"""
Moses Meta-Learning Engine v4.0

Provides automatic discovery of training configurations, neural architectures,
curricula, and reward functions for the Moses robot locomotion framework.

Modules
-------
hyperparameter_search : Optuna-based HPO with early pruning and resumption.
neural_architecture_search : Efficient NAS with weight sharing.
curriculum_learning : Adaptive difficulty scheduling with regression.
reward_shaping : Evolutionary auto-reward design with hacking detection.

Example
-------
>>> from moses.meta_learning import HyperparameterSearch, CurriculumScheduler
>>> hpo = HyperparameterSearch(study_name="moses_ppo_v1")
>>> best = hpo.run(n_trials=100)
"""

from __future__ import annotations

__version__ = "4.0.0"

from .hyperparameter_search import HyperparameterSearch, SearchSpace
from .neural_architecture_search import NeuralArchitectureSearch, ArchitectureConfig
from .curriculum_learning import CurriculumScheduler, DifficultyConfig
from .reward_shaping import RewardShaper, RewardComponent

__all__ = [
    "HyperparameterSearch",
    "SearchSpace",
    "NeuralArchitectureSearch",
    "ArchitectureConfig",
    "CurriculumScheduler",
    "DifficultyConfig",
    "RewardShaper",
    "RewardComponent",
]
