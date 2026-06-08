"""
Moses v4.0 Deep Recursion Package
==================================

The deepest recursion layers of the Moses architecture:

    meta_meta_learning  →  Learns how to design meta-learning strategies
    self_healing        →  Monitors health, detects anomalies, auto-repairs
    world_model         →  Predictive environment dynamics for planning
    predictor           →  Surrogate model for training outcome prediction

These modules form a self-improving stack:

    ┌─────────────────────────────────────────────┐
    │  predictor          →  Guides experiments   │
    │  world_model        →  Simulates outcomes   │
    │  self_healing       →  Maintains stability  │
    │  meta_meta_learning →  Discovers algorithms │
    └─────────────────────────────────────────────┘

Each layer can operate independently or compose into the full recursion stack.

Example:
    from moses.recursion import MetaMetaTrainer, SelfHealingSystem
    from moses.recursion import WorldModelTrainer, TrainingOutcomePredictor

    # Meta-meta learning discovers an optimizer
    meta_trainer = MetaMetaTrainer(config)
    results = meta_trainer.train(task_distribution)

    # Self-healing monitors training
    healer = SelfHealingSystem()
    model, optimizer, health = healer.step(step, loss, model, optimizer)

    # World model enables imagination-based planning
    world_model = EnsembleTransitionModel(config)
    planner = MPCPlanner(world_model, config)
    action = planner.plan(state)

    # Predictor saves compute
    predictor = TrainingOutcomePredictor()
    predictor.add_experiment(record)
    predictor.fit()
    should_run, info = predictor.should_run(arch_config, hparams)
"""

from __future__ import annotations

# Meta-meta-learning
from .meta_meta_learning import (
    AlgorithmDiscoveryCell,
    LSTMOptimizer,
    MetaMetaConfig,
    MetaMetaLearner,
    MetaMetaTrainer,
    create_sine_task_distribution,
)

# Self-healing
from .self_healing import (
    AnomalyType,
    Diagnosis,
    FixEngine,
    FixResult,
    FixType,
    HealthMonitor,
    HealthSnapshot,
    HealthStatus,
    HealingConfig,
    RootCauseAnalyzer,
    SelfHealingSystem,
)

# World model
from .world_model import (
    EnsembleTransitionModel,
    LatentWorldModel,
    MPCPlanner,
    StateActionEncoder,
    TransitionModel,
    WorldModelConfig,
    WorldModelTrainer,
)

# Predictor
from .predictor import (
    AcquisitionFunction,
    EnsemblePredictor,
    ExperimentRecord,
    FeatureEncoder,
    GaussianProcessSurrogate,
    NeuralSurrogate,
    PredictorConfig,
    TrainingOutcomePredictor,
)

__version__ = "4.0.0"
__all__ = [
    # Meta-meta-learning
    "MetaMetaConfig",
    "MetaMetaLearner",
    "MetaMetaTrainer",
    "LSTMOptimizer",
    "AlgorithmDiscoveryCell",
    "create_sine_task_distribution",
    # Self-healing
    "HealthStatus",
    "AnomalyType",
    "FixType",
    "HealthSnapshot",
    "Diagnosis",
    "FixResult",
    "HealingConfig",
    "HealthMonitor",
    "RootCauseAnalyzer",
    "FixEngine",
    "SelfHealingSystem",
    # World model
    "WorldModelConfig",
    "StateActionEncoder",
    "TransitionModel",
    "EnsembleTransitionModel",
    "LatentWorldModel",
    "MPCPlanner",
    "WorldModelTrainer",
    # Predictor
    "PredictorConfig",
    "ExperimentRecord",
    "FeatureEncoder",
    "NeuralSurrogate",
    "GaussianProcessSurrogate",
    "EnsemblePredictor",
    "AcquisitionFunction",
    "TrainingOutcomePredictor",
]