"""
moses/emergence/behavior_arbitration.py
Emergent Behavior Selection for Moses v4.0

Multiple competing controllers (GR00T, PPO, MPC, heuristic) submit actions
for every timestep. An arbitrator selects or blends actions based on
controller confidence and current context. Over time, the system learns
which controller to trust in which situations.

Inspired by:
- Doya (2002) — "Metalearning and neuromodulation" (Neural Networks)
- Frank et al. (2004) — "By carrot or by stick" on arbitration between
  model-based and model-free control (Science)
- Heess et al. (2017) — "Emergence of locomotion behaviours in rich environments"
  (DeepMind, mixture-of-experts style)
- Kumar et al. (2019) — "Conservative Q-Learning" (confidence estimation)

This is NOT a magical "best of all worlds." It is explicit confidence-weighted
blending with online learning of arbitration rules.
"""

from __future__ import annotations

import copy
import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ControllerContext:
    """Context features used by the arbitrator to decide."""
    task_phase: str = "unknown"       # startup, steady, transition, recovery
    terrain_type: str = "flat"        # flat, rough, slope, stairs
    robot_stability: float = 1.0      # 0..1 from IMU
    contact_quality: float = 1.0      # 0..1 foot contact consistency
    recent_fall_count: int = 0
    time_since_reset: float = 0.0     # seconds
    commanded_velocity: float = 0.0


@dataclass
class ControllerProposal:
    """One controller's proposed action + metadata."""
    controller_name: str
    action: np.ndarray
    confidence: float                 # 0..1, controller's self-assessment
    value_estimate: float             # expected return if this action is taken
    computation_time_ms: float
    # Optional: uncertainty estimate
    action_std: Optional[np.ndarray] = None


@dataclass
class ArbitrationDecision:
    """Output of the arbitrator."""
    selected_controller: str
    blended_action: np.ndarray
    blend_weights: Dict[str, float]
    arbitration_mode: str             # "winner_takes_all", "soft_blend", "safety_override"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ─── Controller Interface ────────────────────────────────────────────────────

class Controller(ABC):
    """
    Abstract base for all controllers that can participate in arbitration.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def compute_action(
        self,
        observation: np.ndarray,
        context: ControllerContext,
    ) -> ControllerProposal:
        """
        Compute action and self-assessed confidence.
        Confidence should reflect the controller's estimate of how likely
        its action is to succeed in the current context.
        """
        ...

    @abstractmethod
    def report_outcome(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Feedback to update the controller's internal model (if any)."""
        ...


# ─── Concrete Controller Stubs ───────────────────────────────────────────────

class GR00TController(Controller):
    """
    NVIDIA GR00T-style foundation model controller.
    High capacity, generalizes across tasks, but slower and may hallucinate.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._name = "gr00t"
        self._recent_rewards = deque(maxlen=100)

    @property
    def name(self) -> str:
        return self._name

    def compute_action(
        self,
        observation: np.ndarray,
        context: ControllerContext,
    ) -> ControllerProposal:
        # Placeholder: in production, this calls the GR00T model
        action = np.zeros(12)  # 12-DOF placeholder
        # Confidence drops when terrain is rough or robot is unstable
        base_conf = 0.85
        if context.terrain_type in ("rough", "stairs"):
            base_conf -= 0.15
        if context.robot_stability < 0.7:
            base_conf -= 0.20
        conf = max(0.1, base_conf)
        return ControllerProposal(
            controller_name=self.name,
            action=action,
            confidence=conf,
            value_estimate=0.0,  # from critic if available
            computation_time_ms=50.0,
        )

    def report_outcome(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self._recent_rewards.append(reward)


class PPOController(Controller):
    """
    Proximal Policy Optimization controller.
    Fast, reliable for trained tasks, but brittle outside training distribution.
    """

    def __init__(self, policy_path: Optional[str] = None):
        self.policy_path = policy_path
        self._name = "ppo"
        self._recent_rewards = deque(maxlen=100)
        self._training_task = "default"

    @property
    def name(self) -> str:
        return self._name

    def compute_action(
        self,
        observation: np.ndarray,
        context: ControllerContext,
    ) -> ControllerProposal:
        action = np.zeros(12)
        # PPO is confident on flat ground, less so on stairs
        base_conf = 0.80
        if context.terrain_type == "stairs":
            base_conf -= 0.30
        if context.recent_fall_count > 2:
            base_conf -= 0.25
        conf = max(0.1, base_conf)
        return ControllerProposal(
            controller_name=self.name,
            action=action,
            confidence=conf,
            value_estimate=0.0,
            computation_time_ms=5.0,
        )

    def report_outcome(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self._recent_rewards.append(reward)


class MPCController(Controller):
    """
    Model Predictive Control controller.
    Excellent for precise maneuvers, but computationally expensive and
    requires accurate dynamics model.
    """

    def __init__(self, horizon: int = 16):
        self.horizon = horizon
        self._name = "mpc"
        self._solve_failures = 0

    @property
    def name(self) -> str:
        return self._name

    def compute_action(
        self,
        observation: np.ndarray,
        context: ControllerContext,
    ) -> ControllerProposal:
        action = np.zeros(12)
        # MPC is confident for precise tasks, but slow
        base_conf = 0.75
        if context.time_since_reset < 1.0:
            base_conf += 0.10  # Good at startup stabilization
        if context.computation_time_ms > 20:  # If we're running slow
            base_conf -= 0.20
        conf = max(0.1, base_conf)
        return ControllerProposal(
            controller_name=self.name,
            action=action,
            confidence=conf,
            value_estimate=0.0,
            computation_time_ms=30.0,
        )

    def report_outcome(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        pass


class HeuristicController(Controller):
    """
    Hand-crafted heuristic controller (e.g., standing policy, fall recovery).
    Low capability but maximum reliability in known situations.
    """

    def __init__(self):
        self._name = "heuristic"

    @property
    def name(self) -> str:
        return self._name

    def compute_action(
        self,
        observation: np.ndarray,
        context: ControllerContext,
    ) -> ControllerProposal:
        action = np.zeros(12)
        # Heuristic is always moderately confident — it's simple and predictable
        base_conf = 0.60
        if context.robot_stability < 0.5 or context.recent_fall_count > 0:
            base_conf += 0.25  # Best for recovery
        conf = min(0.95, base_conf)
        return ControllerProposal(
            controller_name=self.name,
            action=action,
            confidence=conf,
            value_estimate=0.0,
            computation_time_ms=0.1,
        )

    def report_outcome(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        pass


# ─── Arbitration Engine ──────────────────────────────────────────────────────

class BehaviorArbitrator:
    """
    Selects or blends actions from multiple controllers.

    Two modes:
    1. Winner-takes-all: pick the controller with highest adjusted confidence.
    2. Soft blend: weighted average of actions by confidence.

    Adjusted confidence = raw_confidence * learned_context_multiplier.
    The context multiplier is learned online from outcome feedback.

    Parameters
    ----------
    controllers : List[Controller]
        All available controllers.
    mode : str
        "winner_takes_all" or "soft_blend".
    blend_temperature : float
        Temperature for soft blending (lower = more winner-takes-all).
    safety_controller : str
        Controller to fall back to when stability is critical.
    """

    def __init__(
        self,
        controllers: List[Controller],
        mode: str = "soft_blend",
        blend_temperature: float = 0.5,
        safety_controller: str = "heuristic",
        context_feature_dim: int = 6,
        learning_rate: float = 0.01,
        state_path: Optional[Path] = None,
    ):
        self.controllers = {c.name: c for c in controllers}
        self.controller_names = list(self.controllers.keys())
        self.mode = mode
        self.blend_temperature = blend_temperature
        self.safety_controller = safety_controller
        self.learning_rate = learning_rate
        self.state_path = state_path or Path(".moses_arbitrator.json")

        # Learned context multipliers: controller -> feature weights
        # Simple linear model: adjusted_conf = raw_conf * sigmoid(w · context)
        self.context_weights: Dict[str, np.ndarray] = {
            name: np.zeros(context_feature_dim)
            for name in self.controller_names
        }
        self.context_bias: Dict[str, float] = {
            name: 0.0 for name in self.controller_names
        }

        # Running statistics for context normalization
        self.context_mean = np.zeros(context_feature_dim)
        self.context_var = np.ones(context_feature_dim)
        self.context_count = 0

        # Performance history: controller -> deque of (context, reward)
        self.performance_history: Dict[str, deque] = {
            name: deque(maxlen=1000) for name in self.controller_names
        }

        self._load()

    def _context_to_vector(self, ctx: ControllerContext) -> np.ndarray:
        """Convert ControllerContext to normalized feature vector."""
        terrain_map = {"flat": 0.0, "rough": 0.5, "slope": 0.3, "stairs": 1.0}
        vec = np.array([
            1.0 if ctx.task_phase == "recovery" else 0.0,
            terrain_map.get(ctx.terrain_type, 0.0),
            ctx.robot_stability,
            ctx.contact_quality,
            min(ctx.recent_fall_count / 5.0, 1.0),
            min(ctx.time_since_reset / 10.0, 1.0),
        ], dtype=np.float32)

        # Online normalization
        self.context_count += 1
        delta = vec - self.context_mean
        self.context_mean += delta / self.context_count
        delta2 = vec - self.context_mean
        self.context_var = ((self.context_count - 1) * self.context_var + delta * delta2) / self.context_count

        std = np.sqrt(self.context_var + 1e-8)
        return (vec - self.context_mean) / std

    def _adjusted_confidence(
        self,
        controller_name: str,
        raw_confidence: float,
        context_vec: np.ndarray,
    ) -> float:
        """Apply learned context multiplier to raw confidence."""
        w = self.context_weights[controller_name]
        b = self.context_bias[controller_name]
        logit = np.dot(w, context_vec) + b
        multiplier = 1.0 / (1.0 + np.exp(-logit))  # sigmoid, 0..1
        # Map to 0.5..1.5 range so it can boost or reduce
        multiplier = 0.5 + multiplier
        return np.clip(raw_confidence * multiplier, 0.0, 1.0)

    def arbitrate(
        self,
        observation: np.ndarray,
        context: ControllerContext,
    ) -> ArbitrationDecision:
        """
        Main arbitration loop. Collect proposals, compute adjusted confidences,
        select or blend.
        """
        # Collect proposals
        proposals: Dict[str, ControllerProposal] = {}
        for name, ctrl in self.controllers.items():
            try:
                prop = ctrl.compute_action(observation, context)
                proposals[name] = prop
            except Exception as e:
                logger.warning(f"Controller {name} failed: {e}")
                continue

        if not proposals:
            raise RuntimeError("All controllers failed")

        # Compute adjusted confidences
        context_vec = self._context_to_vector(context)
        adjusted_confs = {}
        for name, prop in proposals.items():
            adj = self._adjusted_confidence(name, prop.confidence, context_vec)
            adjusted_confs[name] = adj

        # Safety override: if stability is critical, force safety controller
        if context.robot_stability < 0.3 or context.recent_fall_count > 3:
            if self.safety_controller in proposals:
                logger.info("SAFETY OVERRIDE: forcing heuristic controller")
                return ArbitrationDecision(
                    selected_controller=self.safety_controller,
                    blended_action=proposals[self.safety_controller].action,
                    blend_weights={self.safety_controller: 1.0},
                    arbitration_mode="safety_override",
                )

        # Winner-takes-all or soft blend
        if self.mode == "winner_takes_all":
            winner = max(adjusted_confs.items(), key=lambda x: x[1])[0]
            weights = {name: 0.0 for name in proposals}
            weights[winner] = 1.0
            action = proposals[winner].action
            mode_str = "winner_takes_all"
        else:
            # Soft blend with temperature
            temps = np.array(list(adjusted_confs.values())) / self.blend_temperature
            exp_scores = np.exp(temps - np.max(temps))  # numerical stability
            weights_arr = exp_scores / exp_scores.sum()
            weights = {
                name: float(w) for name, w in zip(adjusted_confs.keys(), weights_arr)
            }
            action = sum(
                weights[name] * proposals[name].action for name in proposals
            )
            winner = max(weights.items(), key=lambda x: x[1])[0]
            mode_str = "soft_blend"

        return ArbitrationDecision(
            selected_controller=winner,
            blended_action=action,
            blend_weights=weights,
            arbitration_mode=mode_str,
        )

    def report_outcome(
        self,
        decision: ArbitrationDecision,
        context: ControllerContext,
        reward: float,
    ) -> None:
        """
        Update arbitration model based on what happened.
        The key insight: we attribute reward to each controller proportional
        to its blend weight, then update its context model.
        """
        context_vec = self._context_to_vector(context)

        for name, weight in decision.blend_weights.items():
            if weight < 0.01:
                continue

            # Store for batch updates
            self.performance_history[name].append((context_vec.copy(), reward, weight))

            # Online gradient update: we want high weight controllers that
            # got high reward to have their context multiplier increased.
            w = self.context_weights[name]
            b = self.context_bias[name]
            logit = np.dot(w, context_vec) + b
            sig = 1.0 / (1.0 + np.exp(-logit))

            # Gradient of (reward * sig) w.r.t. weights
            # We want to increase sig when reward is high, decrease when low
            grad = sig * (1 - sig) * reward * weight
            self.context_weights[name] += self.learning_rate * grad * context_vec
            self.context_bias[name] += self.learning_rate * grad

        logger.debug(
            f"Arbitration outcome: reward={reward:.3f}, "
            f"weights={decision.blend_weights}"
        )

    def get_controller_ranking(
        self,
        context: ControllerContext,
    ) -> List[Tuple[str, float]]:
        """Return controllers ranked by adjusted confidence for a context."""
        context_vec = self._context_to_vector(context)
        rankings = []
        for name in self.controller_names:
            # Use a nominal raw confidence of 0.8 for ranking
            adj = self._adjusted_confidence(name, 0.8, context_vec)
            rankings.append((name, adj))
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def discover_rules(self) -> List[Dict[str, Any]]:
        """
        Extract human-readable rules about when to use which controller.
        Returns a list of discovered arbitration rules.
        """
        rules = []
        feature_names = [
            "recovery_phase", "terrain_difficulty", "robot_stability",
            "contact_quality", "recent_falls", "time_since_reset",
        ]

        for name in self.controller_names:
            w = self.context_weights[name]
            b = self.context_bias[name]
            # Find features with large positive/negative weights
            top_positive = []
            top_negative = []
            for i, fn in enumerate(feature_names):
                if w[i] > 0.5:
                    top_positive.append(fn)
                elif w[i] < -0.5:
                    top_negative.append(fn)

            if top_positive or top_negative:
                rules.append({
                    "controller": name,
                    "boosted_when": top_positive,
                    "suppressed_when": top_negative,
                    "bias": round(b, 3),
                })

        return rules

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self) -> None:
        state = {
            "context_weights": {k: v.tolist() for k, v in self.context_weights.items()},
            "context_bias": self.context_bias,
            "context_mean": self.context_mean.tolist(),
            "context_var": self.context_var.tolist(),
            "context_count": self.context_count,
            "mode": self.mode,
            "blend_temperature": self.blend_temperature,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        with open(self.state_path) as f:
            state = json.load(f)
        self.context_weights = {
            k: np.array(v) for k, v in state.get("context_weights", {}).items()
        }
        self.context_bias = state.get("context_bias", self.context_bias)
        self.context_mean = np.array(state.get("context_mean", self.context_mean.tolist()))
        self.context_var = np.array(state.get("context_var", self.context_var.tolist()))
        self.context_count = state.get("context_count", 0)
        self.mode = state.get("mode", self.mode)
        self.blend_temperature = state.get("blend_temperature", self.blend_temperature)
