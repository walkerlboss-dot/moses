"""
moses/emergence/swarm_intelligence.py
Multi-Agent Collective Learning for Moses v4.0

A population of Moses instances with diverse configurations collaborates via
a shared knowledge graph. Agents vote on hyperparameters, share experiences,
and discover emergent strategies no single agent finds alone.

Inspired by:
- Jaderberg et al. (2019) — "Human-level performance in 3D multiplayer games
  with population-based reinforcement learning" (Nature)
- Vinyals et al. (2019) — "Grandmaster level in StarCraft II using multi-agent
  reinforcement learning" (Nature)
- Colas et al. (2020) — "How many random seeds?" + population-based training

This is NOT magical swarm consciousness. It is explicit experience sharing
and voting among independent learners.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Configuration fingerprint for one Moses agent in the swarm."""
    agent_id: str
    policy_arch: str = "mlp"          # mlp, lstm, transformer, gr00t
    observation_keys: Tuple[str, ...] = ("proprio", "imu")
    action_space: str = "joint_pos"   # joint_pos, joint_vel, ee_pose
    lr: float = 3e-4
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    batch_size: int = 4096
    env_seed: int = 0
    # Unique fingerprint for deduplication
    fingerprint: str = ""

    def __post_init__(self):
        if not self.fingerprint:
            payload = json.dumps(asdict(self), sort_keys=True)
            self.fingerprint = hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class ExperiencePacket:
    """Compressed experience shared between swarm agents."""
    agent_id: str
    episode_id: str
    timestamp: str
    # State summary (not full trajectories — too large)
    state_summary: Dict[str, float]   # mean/std of obs dimensions
    action_summary: Dict[str, float]
    reward_breakdown: Dict[str, float]  # per-component rewards
    total_reward: float
    episode_length: int
    success: bool
    # What the agent learned
    value_estimate: float             # V(s_0) at episode start
    policy_entropy: float
    # Context for matching
    task_id: str
    difficulty: float
    # Hyperparams used
    hyperparams: Dict[str, Any]


@dataclass
class Vote:
    """One agent's vote on a hyperparameter or strategy."""
    agent_id: str
    target: str                       # e.g., "lr_for_humanoid_walk"
    proposal: Any
    confidence: float                 # 0..1 based on local evidence
    evidence_reward: float            # mean reward that supports this vote
    timestamp: str


@dataclass
class EmergentStrategy:
    """A strategy discovered by the swarm, not by any single agent."""
    strategy_id: str
    description: str
    # How to recognize when this strategy applies
    context_signature: Dict[str, Tuple[float, float]]  # feature -> (mean, std)
    # What to do
    recommended_hyperparams: Dict[str, Any]
    recommended_controller: str
    # Evidence
    discovered_by: List[str]          # agent IDs that contributed
    episodes_validated: int
    mean_reward: float
    std_reward: float
    success_rate: float
    created_at: str


# ─── Swarm Intelligence Core ─────────────────────────────────────────────────

class SwarmIntelligence:
    """
    Population-based collective learning for Moses.

    Each agent trains independently but shares compressed experiences
    and votes on what works. The swarm tracks emergent strategies —
    combinations of hyperparameters and controllers that outperform
    anything any single agent found.

    Parameters
    ----------
    population_size : int
        Number of agents in the swarm (typically 8-32).
    knowledge_graph : KnowledgeGraph or None
        Shared graph for persistent cross-agent memory.
    experience_buffer_size : int
        How many recent packets to keep per agent.
    vote_threshold : float
        Minimum confidence for a vote to count.
    """

    def __init__(
        self,
        population_size: int = 16,
        knowledge_graph=None,
        experience_buffer_size: int = 1000,
        vote_threshold: float = 0.6,
        strategy_discovery_threshold: float = 0.15,  # 15% above baseline
        state_dir: Optional[Path] = None,
    ):
        self.population_size = population_size
        self.kg = knowledge_graph
        self.experience_buffer_size = experience_buffer_size
        self.vote_threshold = vote_threshold
        self.strategy_discovery_threshold = strategy_discovery_threshold
        self.state_dir = state_dir or Path(".moses_swarm")
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Population: agent_id -> AgentConfig
        self.population: Dict[str, AgentConfig] = {}
        # Experiences: agent_id -> deque-like ring buffer
        self.experiences: Dict[str, List[ExperiencePacket]] = defaultdict(list)
        # Votes: target -> list of Vote
        self.votes: Dict[str, List[Vote]] = defaultdict(list)
        # Discovered strategies
        self.strategies: Dict[str, EmergentStrategy] = {}
        # Performance baselines per task
        self.baselines: Dict[str, float] = {}
        # Thread safety
        self._lock = threading.RLock()

        self._init_population()

    def _init_population(self):
        """Create diverse initial population."""
        arch_choices = ["mlp", "lstm", "transformer"]
        lr_choices = [1e-4, 3e-4, 1e-3]
        entropy_choices = [0.0, 0.01, 0.02]
        obs_choices = [
            ("proprio",),
            ("proprio", "imu"),
            ("proprio", "imu", "depth"),
            ("proprio", "imu", "depth", "lidar"),
        ]

        for i in range(self.population_size):
            cfg = AgentConfig(
                agent_id=f"swarm_agent_{i:03d}",
                policy_arch=random.choice(arch_choices),
                observation_keys=random.choice(obs_choices),
                lr=random.choice(lr_choices),
                entropy_coef=random.choice(entropy_choices),
                env_seed=i,
            )
            self.population[cfg.agent_id] = cfg
            logger.info(f"Swarm init: {cfg.agent_id} arch={cfg.policy_arch} "
                        f"lr={cfg.lr} obs={cfg.observation_keys}")

    # ── Experience Sharing ──────────────────────────────────────────────────

    def submit_experience(self, packet: ExperiencePacket) -> None:
        """An agent shares a compressed experience packet."""
        with self._lock:
            buf = self.experiences[packet.agent_id]
            buf.append(packet)
            if len(buf) > self.experience_buffer_size:
                buf.pop(0)

            # Update knowledge graph if available
            if self.kg is not None:
                self._update_kg_from_packet(packet)

            # Check for emergent strategy
            self._check_strategy_discovery(packet)

    def _update_kg_from_packet(self, packet: ExperiencePacket) -> None:
        """Write experience into the knowledge graph."""
        try:
            self.kg.add_experiment_result(
                experiment_id=packet.episode_id,
                hyperparams=packet.hyperparams,
                architecture={"policy": packet.agent_id},  # simplified
                env_config={"task": packet.task_id, "difficulty": packet.difficulty},
                metrics={
                    "reward_mean": packet.total_reward,
                    "success_rate": 1.0 if packet.success else 0.0,
                    "episode_length": packet.episode_length,
                },
            )
        except Exception as e:
            logger.warning(f"KG update failed: {e}")

    def get_relevant_experiences(
        self,
        agent_id: str,
        task_id: str,
        top_k: int = 10,
    ) -> List[ExperiencePacket]:
        """
        Retrieve experiences from OTHER agents on similar tasks.
        Uses simple feature matching; could use embedding similarity.
        """
        with self._lock:
            candidates = []
            for aid, buf in self.experiences.items():
                if aid == agent_id:
                    continue
                for pkt in buf:
                    if pkt.task_id == task_id:
                        candidates.append(pkt)

            # Sort by total_reward descending
            candidates.sort(key=lambda p: p.total_reward, reverse=True)
            return candidates[:top_k]

    # ── Collective Voting ───────────────────────────────────────────────────

    def cast_vote(self, vote: Vote) -> None:
        """Submit a vote from an agent."""
        if vote.confidence < self.vote_threshold:
            logger.debug(f"Vote from {vote.agent_id} below threshold, ignored.")
            return
        with self._lock:
            self.votes[vote.target].append(vote)

    def get_consensus(
        self,
        target: str,
        min_votes: int = 3,
    ) -> Optional[Tuple[Any, float]]:
        """
        Aggregate votes via weighted averaging (for continuous params)
        or plurality (for discrete choices).

        Returns (winning_proposal, consensus_confidence) or None.
        """
        with self._lock:
            votes = self.votes.get(target, [])
            if len(votes) < min_votes:
                return None

            # Separate by proposal type
            proposals = [v.proposal for v in votes]
            if all(isinstance(p, (int, float)) for p in proposals):
                # Weighted average
                weights = [v.confidence for v in votes]
                vals = [v.proposal for v in votes]
                consensus = np.average(vals, weights=weights)
                conf = np.mean(weights)
                return consensus, conf
            else:
                # Plurality voting
                counts = defaultdict(list)
                for v in votes:
                    counts[str(v.proposal)].append(v.confidence)
                best = max(counts.items(), key=lambda kv: (len(kv[1]), np.mean(kv[1])))
                # Parse back to original type if possible
                try:
                    proposal = json.loads(best[0])
                except Exception:
                    proposal = best[0]
                conf = np.mean(best[1])
                return proposal, conf

    def vote_on_hyperparams(
        self,
        agent_id: str,
        task_id: str,
        hyperparam_name: str,
        local_best_value: Any,
        local_best_reward: float,
        local_episodes: int,
    ) -> None:
        """
        An agent votes for its best hyperparameter based on local evidence.
        Confidence scales with episode count and reward margin over baseline.
        """
        baseline = self.baselines.get(task_id, 0.0)
        reward_margin = max(0.0, local_best_reward - baseline)
        # Confidence: more episodes + higher margin = more confident
        conf = min(1.0, (local_episodes / 100) * (1 + reward_margin / max(baseline, 1.0)))

        target = f"{hyperparam_name}_for_{task_id}"
        vote = Vote(
            agent_id=agent_id,
            target=target,
            proposal=local_best_value,
            confidence=conf,
            evidence_reward=local_best_reward,
            timestamp=datetime.utcnow().isoformat(),
        )
        self.cast_vote(vote)

    # ── Emergent Strategy Discovery ─────────────────────────────────────────

    def _check_strategy_discovery(self, packet: ExperiencePacket) -> None:
        """
        Detect when an experience suggests a novel, high-performing approach.
        A strategy emerges when multiple agents independently achieve high
        reward with similar hyperparameter/controller combinations.
        """
        baseline = self.baselines.get(packet.task_id, 0.0)
        if baseline == 0:
            self.baselines[packet.task_id] = packet.total_reward
            return

        improvement = (packet.total_reward - baseline) / max(baseline, 1.0)
        if improvement < self.strategy_discovery_threshold:
            return

        # Look for similar high-performing experiences
        similar = self._find_similar_experiences(packet, threshold=0.1)
        if len(similar) >= 2:  # At least 2 other agents
            strategy = self._synthesize_strategy(packet, similar)
            if strategy.strategy_id not in self.strategies:
                self.strategies[strategy.strategy_id] = strategy
                logger.info(
                    f"EMERGENT STRATEGY discovered: {strategy.strategy_id}\n"
                    f"  Context: {strategy.context_signature}\n"
                    f"  Recommended: {strategy.recommended_hyperparams}\n"
                    f"  Mean reward: {strategy.mean_reward:.1f} "
                    f"(+{100*(strategy.mean_reward/baseline-1):.1f}%)\n"
                    f"  Discovered by: {strategy.discovered_by}"
                )

    def _find_similar_experiences(
        self,
        packet: ExperiencePacket,
        threshold: float = 0.1,
    ) -> List[ExperiencePacket]:
        """Find experiences with similar hyperparams and high reward."""
        similar = []
        for aid, buf in self.experiences.items():
            if aid == packet.agent_id:
                continue
            for other in buf:
                if other.task_id != packet.task_id:
                    continue
                if self._hyperparam_distance(packet.hyperparams, other.hyperparams) < threshold:
                    similar.append(other)
        return similar

    @staticmethod
    def _hyperparam_distance(a: Dict, b: Dict) -> float:
        """Normalized distance between hyperparameter dicts."""
        keys = set(a.keys()) | set(b.keys())
        if not keys:
            return 0.0
        diffs = []
        for k in keys:
            av = a.get(k, 0)
            bv = b.get(k, 0)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                diffs.append(abs(av - bv) / max(abs(av), abs(bv), 1e-8))
            elif av != bv:
                diffs.append(1.0)
        return np.mean(diffs) if diffs else 0.0

    def _synthesize_strategy(
        self,
        packet: ExperiencePacket,
        similar: List[ExperiencePacket],
    ) -> EmergentStrategy:
        """Create an EmergentStrategy from a cluster of experiences."""
        all_packets = [packet] + similar
        rewards = [p.total_reward for p in all_packets]
        agent_ids = list({p.agent_id for p in all_packets})

        # Aggregate hyperparams by median for continuous, mode for discrete
        hp_keys = set()
        for p in all_packets:
            hp_keys.update(p.hyperparams.keys())

        rec_hp = {}
        for k in hp_keys:
            vals = [p.hyperparams[k] for p in all_packets if k in p.hyperparams]
            if all(isinstance(v, (int, float)) for v in vals):
                rec_hp[k] = float(np.median(vals))
            else:
                # Plurality
                rec_hp[k] = max(set(vals), key=vals.count)

        # Context signature from state summaries
        features = defaultdict(list)
        for p in all_packets:
            for fk, fv in p.state_summary.items():
                features[fk].append(fv)

        sig = {}
        for fk, vals in features.items():
            sig[fk] = (float(np.mean(vals)), float(np.std(vals)))

        strategy_id = hashlib.sha256(
            json.dumps({"task": packet.task_id, "hp": rec_hp}, sort_keys=True).encode()
        ).hexdigest()[:16]

        return EmergentStrategy(
            strategy_id=strategy_id,
            description=f"Auto-discovered strategy for {packet.task_id}",
            context_signature=sig,
            recommended_hyperparams=rec_hp,
            recommended_controller="auto",  # inferred from packet
            discovered_by=agent_ids,
            episodes_validated=len(all_packets),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            success_rate=sum(1 for p in all_packets if p.success) / len(all_packets),
            created_at=datetime.utcnow().isoformat(),
        )

    def get_best_strategy(self, task_id: str) -> Optional[EmergentStrategy]:
        """Return the highest-performing emergent strategy for a task."""
        candidates = [
            s for s in self.strategies.values()
            if task_id in s.description
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.mean_reward)

    # ── Population Evolution ────────────────────────────────────────────────

    def evolve_population(self, top_k_ratio: float = 0.25) -> List[AgentConfig]:
        """
        Replace bottom-performing agents with mutated versions of top performers.
        This is Population-Based Training (PBT) style evolution.

        Returns the new population configs.
        """
        with self._lock:
            # Score each agent by mean reward of last N experiences
            scores = {}
            for aid, buf in self.experiences.items():
                if not buf:
                    scores[aid] = -float("inf")
                    continue
                recent = buf[-50:]
                scores[aid] = np.mean([p.total_reward for p in recent])

            sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            n_top = max(1, int(self.population_size * top_k_ratio))
            top_ids = [aid for aid, _ in sorted_agents[:n_top]]
            bottom_ids = [aid for aid, _ in sorted_agents[n_top:]]

            new_population = {}
            for aid in top_ids:
                new_population[aid] = self.population[aid]

            for i, aid in enumerate(bottom_ids):
                parent = self.population[random.choice(top_ids)]
                child = self._mutate_config(parent, new_id=aid)
                new_population[aid] = child
                logger.info(f"Evolved {aid} from {parent.agent_id}: "
                            f"lr {parent.lr} -> {child.lr}, "
                            f"arch {parent.policy_arch} -> {child.policy_arch}")

            self.population = new_population
            return list(new_population.values())

    def _mutate_config(self, parent: AgentConfig, new_id: str) -> AgentConfig:
        """Create a mutated copy of a parent config."""
        mutations = {
            "lr": lambda x: x * random.choice([0.5, 1.0, 2.0]),
            "entropy_coef": lambda x: max(0.0, x + random.choice([-0.005, 0, 0.005])),
            "gamma": lambda x: min(0.999, max(0.9, x + random.choice([-0.01, 0, 0.01]))),
            "policy_arch": lambda x: random.choice(["mlp", "lstm", "transformer"]),
        }

        child_dict = asdict(parent)
        child_dict["agent_id"] = new_id
        child_dict["fingerprint"] = ""  # will regenerate

        # Apply 1-2 mutations
        for key in random.sample(list(mutations.keys()), k=random.randint(1, 2)):
            child_dict[key] = mutations[key](child_dict[key])

        return AgentConfig(**child_dict)

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self) -> None:
        """Serialize swarm state to disk."""
        state = {
            "population": {aid: asdict(cfg) for aid, cfg in self.population.items()},
            "strategies": {sid: asdict(s) for sid, s in self.strategies.items()},
            "baselines": self.baselines,
            "timestamp": datetime.utcnow().isoformat(),
        }
        path = self.state_dir / "swarm_state.json"
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"Swarm state saved to {path}")

    def load(self) -> None:
        """Load swarm state from disk."""
        path = self.state_dir / "swarm_state.json"
        if not path.exists():
            return
        with open(path) as f:
            state = json.load(f)

        self.population = {
            aid: AgentConfig(**d) for aid, d in state.get("population", {}).items()
        }
        self.strategies = {
            sid: EmergentStrategy(**d) for sid, d in state.get("strategies", {}).items()
        }
        self.baselines = state.get("baselines", {})
        logger.info(f"Swarm state loaded from {path}")


# ─── Utility: Experience Compressor ──────────────────────────────────────────

def compress_trajectory(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    agent_id: str,
    task_id: str,
    hyperparams: Dict[str, Any],
    success: bool,
) -> ExperiencePacket:
    """
    Compress a full trajectory into an ExperiencePacket for swarm sharing.

    We do NOT share full trajectories (privacy + bandwidth). Instead we share
    statistical summaries that are sufficient for hyperparameter voting and
    strategy discovery.
    """
    return ExperiencePacket(
        agent_id=agent_id,
        episode_id=hashlib.sha256(
            f"{agent_id}_{time.time()}".encode()
        ).hexdigest()[:16],
        timestamp=datetime.utcnow().isoformat(),
        state_summary={
            "obs_mean": float(np.mean(observations)),
            "obs_std": float(np.std(observations)),
            "obs_max": float(np.max(np.abs(observations))),
        },
        action_summary={
            "act_mean": float(np.mean(actions)),
            "act_std": float(np.std(actions)),
            "act_max": float(np.max(np.abs(actions))),
        },
        reward_breakdown={"total": float(np.sum(rewards))},
        total_reward=float(np.sum(rewards)),
        episode_length=len(rewards),
        success=success,
        value_estimate=0.0,  # populated by caller if available
        policy_entropy=0.0,  # populated by caller if available
        task_id=task_id,
        difficulty=0.5,  # populated by curriculum if available
        hyperparams=hyperparams,
    )
