"""
Moses Emergence Engine v4.0

Emergent behavior systems for the Moses robot locomotion framework:
swarm intelligence, behavior arbitration, curiosity-driven exploration,
and lightweight consciousness simulation.

These modules enable collective learning, adaptive controller selection,
intrinsic motivation, and structured meta-cognition. They are grounded
in established robotics and AI research — not magical properties.

Modules
-------
swarm_intelligence : Population-based multi-agent collective learning with
    experience sharing, voting, and emergent strategy discovery.
behavior_arbitration : Confidence-weighted arbitration among competing
    controllers (GR00T, PPO, MPC, heuristic) with online learning.
curiosity : Intrinsic motivation via prediction error (ICM), random network
    distillation (RND), episodic memory, and count-based exploration.
consciousness_sim : Lightweight simulation of self-modeling, introspection,
    goal hierarchies, and attention for improved decision-making.

References
----------
- Pathak et al. (2017) — Curiosity-driven Exploration by Self-supervised
  Prediction (ICM, ICML)
- Burda et al. (2018) — Large-Scale Study of Curiosity-Driven Learning
  (RND, arXiv)
- Jaderberg et al. (2019) — Human-level performance in 3D multiplayer games
  with population-based RL (Nature)
- Doya (2002) — Metalearning and neuromodulation (Neural Networks)
- Graziano (2013) — Consciousness and the Social Brain (attention schema)
- Savinov et al. (2018) — Episodic Curiosity through Reachability

Example
-------
>>> from moses.emergence import SwarmIntelligence, BehaviorArbitrator
>>> from moses.emergence import CuriosityEngine, ConsciousnessSimulation
>>>
>>> # Swarm of 16 agents
>>> swarm = SwarmIntelligence(population_size=16)
>>>
>>> # Arbitration among controllers
>>> controllers = [GR00TController(), PPOController(), MPCController()]
>>> arb = BehaviorArbitrator(controllers=controllers, mode="soft_blend")
>>>
>>> # Curiosity-driven exploration
>>> curiosity = CuriosityEngine(state_dim=48, action_dim=12, method="icm")
>>>
>>> # Consciousness simulation for meta-cognition
>>> cog = ConsciousnessSimulation(modalities=["proprio", "imu", "depth"])
"""

from __future__ import annotations

__version__ = "4.0.0"

from .swarm_intelligence import (
    SwarmIntelligence,
    AgentConfig,
    ExperiencePacket,
    Vote,
    EmergentStrategy,
    compress_trajectory,
)

from .behavior_arbitration import (
    BehaviorArbitrator,
    Controller,
    ControllerContext,
    ControllerProposal,
    ArbitrationDecision,
    GR00TController,
    PPOController,
    MPCController,
    HeuristicController,
)

from .curiosity import (
    CuriosityEngine,
    CuriosityConfig,
    ForwardDynamicsModel,
    RandomNetworkDistillation,
    EpisodicCuriosity,
    CountBasedExploration,
    adaptive_curiosity_schedule,
)

from .consciousness_sim import (
    ConsciousnessSimulation,
    SelfModelManager,
    IntrospectionMonitor,
    GoalHierarchyManager,
    AttentionMechanism,
    Goal,
    GoalStatus,
    SelfModel,
    IntrospectionSnapshot,
    AttentionFocus,
)

__all__ = [
    # Swarm Intelligence
    "SwarmIntelligence",
    "AgentConfig",
    "ExperiencePacket",
    "Vote",
    "EmergentStrategy",
    "compress_trajectory",
    # Behavior Arbitration
    "BehaviorArbitrator",
    "Controller",
    "ControllerContext",
    "ControllerProposal",
    "ArbitrationDecision",
    "GR00TController",
    "PPOController",
    "MPCController",
    "HeuristicController",
    # Curiosity
    "CuriosityEngine",
    "CuriosityConfig",
    "ForwardDynamicsModel",
    "RandomNetworkDistillation",
    "EpisodicCuriosity",
    "CountBasedExploration",
    "adaptive_curiosity_schedule",
    # Consciousness Simulation
    "ConsciousnessSimulation",
    "SelfModelManager",
    "IntrospectionMonitor",
    "GoalHierarchyManager",
    "AttentionMechanism",
    "Goal",
    "GoalStatus",
    "SelfModel",
    "IntrospectionSnapshot",
    "AttentionFocus",
]
