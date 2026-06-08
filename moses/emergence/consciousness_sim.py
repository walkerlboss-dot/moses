"""
moses/emergence/consciousness_sim.py
Lightweight Consciousness Simulation for Moses v4.0

This module simulates aspects of consciousness — NOT claiming actual
consciousness for the robot, but using computational analogues of
self-modeling, introspection, goal hierarchies, and attention to
improve decision-making.

Inspired by:
- Graziano (2013) — "Consciousness and the Social Brain". Attention schema
  theory: consciousness as a model of attention.
- Friston (2010) — "The free-energy principle: a unified brain theory?"
  (Nature Neuroscience). Self-modeling as predictive coding.
- Ha & Schmidhuber (2018) — "World Models" (NeurIPS). Agent learns model
  of itself and environment for planning.
- Butlin et al. (2023) — "Consciousness in Artificial Agents" (arXiv).
  Framework for assessing consciousness-like properties in AI systems.

What this IS:
- A self-model that tracks the agent's own state and capabilities
- Introspection that monitors internal confidence, resource usage, and goals
- Goal hierarchy from mission down to motor commands
- Attention mechanism that selects relevant observations

What this is NOT:
- Phenomenal consciousness (qualia, subjective experience)
- A claim that the robot "feels" anything
- Magical emergence of true understanding
"""

from __future__ import annotations

import copy
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────

class GoalStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Goal:
    """A goal at any level of the hierarchy."""
    goal_id: str
    description: str
    level: int                          # 0=mission, 1=objective, 2=action
    parent_id: Optional[str] = None
    status: GoalStatus = GoalStatus.PENDING
    priority: float = 1.0               # 0..1
    progress: float = 0.0               # 0..1 completion
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    subgoals: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    deadline: Optional[str] = None


@dataclass
class SelfModel:
    """The agent's model of itself — capabilities, state, and history."""
    # Physical capabilities
    dof_count: int = 12
    max_joint_torque: float = 50.0      # Nm
    max_linear_velocity: float = 2.0    # m/s
    # Current state estimates
    estimated_battery: float = 1.0      # 0..1
    estimated_temperature: float = 40.0  # Celsius
    estimated_stability: float = 1.0    # 0..1 from IMU
    # Operational status
    controllers_available: List[str] = field(default_factory=list)
    sensors_operational: List[str] = field(default_factory=list)
    current_controller: str = "none"
    # Performance history
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_falls: int = 0
    total_episodes: int = 0
    total_steps: int = 0
    # Calibration
    last_calibration: Optional[str] = None


@dataclass
class IntrospectionSnapshot:
    """A momentary snapshot of internal state for monitoring."""
    timestamp: str
    # Confidence estimates
    policy_confidence: float            # How sure the policy is
    value_estimate: float               # Expected return
    model_prediction_error: float       # Forward model surprise
    # Resource monitoring
    compute_time_ms: float
    memory_usage_mb: float
    # Emotional analogues (operational states, not feelings)
    stress_level: float                 # High when prediction errors spike
    boredom_level: float                # High when reward variance is low
    curiosity_level: float              # Current intrinsic motivation
    # Goal status
    active_goal_id: Optional[str]
    goal_stack_depth: int


@dataclass
class AttentionFocus:
    """What the agent is currently attending to."""
    attended_modalities: List[str]      # e.g., ["proprio", "depth"]
    attention_weights: Dict[str, float]  # modality -> weight
    spatial_focus: Optional[Tuple[float, float, float]] = None  # xyz
    temporal_focus: str = "present"     # present, recent_past, predicted_future
    confidence: float = 1.0


# ─── Self-Model Manager ──────────────────────────────────────────────────────

class SelfModelManager:
    """
    Maintains and updates the agent's self-model.

    The self-model is a simplified representation of the agent's own
    body and capabilities. It is used for:
    - Predicting what actions are physically possible
    - Detecting anomalies ("my leg shouldn't bend that way")
    - Selecting appropriate controllers based on capability
    """

    def __init__(self, initial_model: Optional[SelfModel] = None):
        self.model = initial_model or SelfModel()
        self._history: deque = deque(maxlen=1000)

    def update_from_observation(self, observation: Dict[str, np.ndarray]) -> None:
        """
        Update self-model based on current sensor readings.
        Observations should include keys like 'proprio', 'imu', 'battery'.
        """
        if "battery" in observation:
            self.model.estimated_battery = float(np.mean(observation["battery"]))
        if "imu" in observation:
            imu = observation["imu"]
            # Simple stability estimate from accelerometer variance
            if len(imu) >= 3:
                acc_var = np.var(imu[:3])
                self.model.estimated_stability = float(np.exp(-acc_var * 10))
        if "temperature" in observation:
            self.model.estimated_temperature = float(np.mean(observation["temperature"]))

        self.model.total_steps += 1
        self._history.append(copy.deepcopy(self.model))

    def update_from_outcome(
        self,
        reward: float,
        fell: bool,
        controller_used: str,
    ) -> None:
        """Update self-model based on what just happened."""
        self.model.recent_rewards.append(reward)
        self.model.current_controller = controller_used
        if fell:
            self.model.recent_falls += 1
        self.model.total_episodes += 1

    def is_capable_of(self, action_description: str) -> Tuple[bool, float]:
        """
        Check if the self-model believes an action is feasible.
        Returns (feasible, confidence).
        """
        # Simple rule-based capability checking
        if "sprint" in action_description and self.model.estimated_battery < 0.2:
            return False, 0.9
        if "stairs" in action_description and "depth" not in self.model.sensors_operational:
            return False, 0.7
        if self.model.estimated_stability < 0.3:
            return False, 0.95
        return True, 0.8

    def detect_anomaly(self, observation: Dict[str, np.ndarray]) -> Optional[str]:
        """
        Detect when observations contradict the self-model.
        E.g., joint position outside expected range.
        """
        if "proprio" in observation:
            proprio = observation["proprio"]
            # Check for impossible joint angles (simplified)
            if np.any(np.abs(proprio) > np.pi * 2):
                return "joint_angle_anomaly"
        if "imu" in observation:
            imu = observation["imu"]
            # Free-fall detection
            acc_norm = np.linalg.norm(imu[:3])
            if acc_norm < 1.0 and self.model.estimated_stability > 0.8:
                return "unexpected_freefall"
        return None

    def get_model(self) -> SelfModel:
        return self.model


# ─── Introspection Monitor ───────────────────────────────────────────────────

class IntrospectionMonitor:
    """
    Monitors internal states and provides meta-cognitive information.

    This is analogous to introspection in humans: awareness of one's own
    thought processes. Here it means the agent tracks:
    - How confident its policy is
    - Whether its world model is making good predictions
    - Whether it is stuck, bored, or overloaded
    """

    def __init__(
        self,
        history_size: int = 1000,
        stress_threshold: float = 0.7,
        boredom_threshold: float = 0.8,
    ):
        self.history_size = history_size
        self.stress_threshold = stress_threshold
        self.boredom_threshold = boredom_threshold
        self.snapshots: deque = deque(maxlen=history_size)
        self.prediction_errors: deque = deque(maxlen=100)
        self.reward_history: deque = deque(maxlen=100)

    def record(
        self,
        policy_confidence: float,
        value_estimate: float,
        model_prediction_error: float,
        compute_time_ms: float,
        memory_usage_mb: float,
        active_goal_id: Optional[str],
        goal_stack_depth: int,
    ) -> IntrospectionSnapshot:
        """Record a new introspection snapshot."""
        self.prediction_errors.append(model_prediction_error)
        self.reward_history.append(value_estimate)

        # Compute operational-state analogues
        recent_errors = list(self.prediction_errors)
        stress = min(1.0, np.mean(recent_errors[-10:]) * 5) if len(recent_errors) >= 10 else 0.0

        recent_rewards = list(self.reward_history)
        boredom = 0.0
        if len(recent_rewards) >= 20:
            reward_var = np.var(recent_rewards[-20:])
            boredom = 1.0 - min(1.0, reward_var * 10)

        # Curiosity = inverse of prediction accuracy
        curiosity = min(1.0, np.mean(recent_errors[-10:]) * 3) if len(recent_errors) >= 10 else 0.5

        snap = IntrospectionSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            policy_confidence=policy_confidence,
            value_estimate=value_estimate,
            model_prediction_error=model_prediction_error,
            compute_time_ms=compute_time_ms,
            memory_usage_mb=memory_usage_mb,
            stress_level=stress,
            boredom_level=boredom,
            curiosity_level=curiosity,
            active_goal_id=active_goal_id,
            goal_stack_depth=goal_stack_depth,
        )
        self.snapshots.append(snap)
        return snap

    def get_current_state(self) -> Optional[IntrospectionSnapshot]:
        """Return the most recent introspection snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    def should_switch_strategy(self) -> Tuple[bool, str]:
        """
        Based on introspection, recommend whether to change behavior.
        Returns (should_switch, reason).
        """
        if len(self.snapshots) < 10:
            return False, "insufficient_history"

        recent = list(self.snapshots)[-10:]
        avg_stress = np.mean([s.stress_level for s in recent])
        avg_boredom = np.mean([s.boredom_level for s in recent])
        avg_confidence = np.mean([s.policy_confidence for s in recent])

        if avg_stress > self.stress_threshold:
            return True, f"high_stress ({avg_stress:.2f})"
        if avg_boredom > self.boredom_threshold:
            return True, f"high_boredom ({avg_boredom:.2f})"
        if avg_confidence < 0.3:
            return True, f"low_confidence ({avg_confidence:.2f})"

        return False, "stable"

    def is_overloaded(self) -> bool:
        """Check if compute or memory usage is too high."""
        if not self.snapshots:
            return False
        recent = list(self.snapshots)[-5:]
        avg_compute = np.mean([s.compute_time_ms for s in recent])
        avg_memory = np.mean([s.memory_usage_mb for s in recent])
        return avg_compute > 100 or avg_memory > 8000  # >100ms or >8GB


# ─── Goal Hierarchy Manager ──────────────────────────────────────────────────

class GoalHierarchyManager:
    """
    Manages hierarchical goals: mission -> objectives -> actions.

    The agent always has one active mission. Missions decompose into
    objectives, which decompose into low-level actions. The manager
    tracks progress and handles failure by replanning.

    Inspired by hierarchical reinforcement learning (Kulkarni et al. 2016,
    "Hierarchical Deep Reinforcement Learning").
    """

    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.active_mission: Optional[str] = None
        self.goal_stack: List[str] = []  # Current path from mission to leaf

    def set_mission(self, description: str, mission_id: str = "mission_0") -> Goal:
        """Set the top-level mission."""
        mission = Goal(
            goal_id=mission_id,
            description=description,
            level=0,
            status=GoalStatus.ACTIVE,
        )
        self.goals[mission_id] = mission
        self.active_mission = mission_id
        self.goal_stack = [mission_id]
        logger.info(f"Mission set: {description}")
        return mission

    def add_objective(
        self,
        description: str,
        parent_id: Optional[str] = None,
        priority: float = 1.0,
    ) -> Goal:
        """Add an objective under a mission or another objective."""
        parent_id = parent_id or self.active_mission
        if parent_id not in self.goals:
            raise ValueError(f"Parent goal {parent_id} not found")

        obj_id = f"obj_{len(self.goals)}"
        obj = Goal(
            goal_id=obj_id,
            description=description,
            level=1,
            parent_id=parent_id,
            priority=priority,
        )
        self.goals[obj_id] = obj
        self.goals[parent_id].subgoals.append(obj_id)
        return obj

    def add_action(
        self,
        description: str,
        parent_id: str,
        success_criteria: Dict[str, Any],
    ) -> Goal:
        """Add a low-level action under an objective."""
        if parent_id not in self.goals:
            raise ValueError(f"Parent goal {parent_id} not found")

        act_id = f"act_{len(self.goals)}"
        action = Goal(
            goal_id=act_id,
            description=description,
            level=2,
            parent_id=parent_id,
            success_criteria=success_criteria,
        )
        self.goals[act_id] = action
        self.goals[parent_id].subgoals.append(act_id)
        return action

    def get_current_action(self) -> Optional[Goal]:
        """Return the current leaf action to execute."""
        if not self.goal_stack:
            return None
        # Find deepest active goal
        for gid in reversed(self.goal_stack):
            g = self.goals.get(gid)
            if g and g.status == GoalStatus.ACTIVE:
                # If it has subgoals, find first pending
                if g.subgoals:
                    for sg_id in g.subgoals:
                        sg = self.goals.get(sg_id)
                        if sg and sg.status == GoalStatus.PENDING:
                            sg.status = GoalStatus.ACTIVE
                            self.goal_stack.append(sg_id)
                            return self.get_current_action()
                return g
        return None

    def report_progress(self, goal_id: str, progress: float) -> None:
        """Update progress on a goal (0..1)."""
        if goal_id not in self.goals:
            return
        self.goals[goal_id].progress = np.clip(progress, 0.0, 1.0)
        if progress >= 1.0:
            self.goals[goal_id].status = GoalStatus.COMPLETED
            self._propagate_completion(goal_id)

    def report_failure(self, goal_id: str, reason: str) -> None:
        """Mark a goal as failed and trigger replanning."""
        if goal_id not in self.goals:
            return
        self.goals[goal_id].status = GoalStatus.FAILED
        logger.warning(f"Goal {goal_id} failed: {reason}")
        self._handle_failure(goal_id)

    def _propagate_completion(self, goal_id: str) -> None:
        """Check if parent goals are now complete."""
        goal = self.goals[goal_id]
        if goal.parent_id and goal.parent_id in self.goals:
            parent = self.goals[goal.parent_id]
            if all(self.goals[sg].status == GoalStatus.COMPLETED for sg in parent.subgoals):
                parent.status = GoalStatus.COMPLETED
                parent.progress = 1.0
                self._propagate_completion(parent.goal_id)

    def _handle_failure(self, goal_id: str) -> None:
        """Replan when a goal fails."""
        goal = self.goals[goal_id]
        # Simple strategy: if action fails, try alternative
        if goal.level == 2 and goal.parent_id:
            parent = self.goals[goal.parent_id]
            # Find next pending sibling
            for sg_id in parent.subgoals:
                sg = self.goals.get(sg_id)
                if sg and sg.status == GoalStatus.PENDING:
                    sg.status = GoalStatus.ACTIVE
                    self.goal_stack = [g for g in self.goal_stack if g != goal_id]
                    self.goal_stack.append(sg_id)
                    logger.info(f"Replanning: switched to {sg_id}")
                    return
        # If no alternative, mark parent as blocked
        if goal.parent_id:
            self.goals[goal.parent_id].status = GoalStatus.BLOCKED

    def get_goal_tree(self) -> Dict[str, Any]:
        """Return the full goal hierarchy as a nested dict."""
        def build_tree(gid: str) -> Dict[str, Any]:
            g = self.goals[gid]
            return {
                "id": gid,
                "description": g.description,
                "status": g.status.value,
                "progress": g.progress,
                "subgoals": [build_tree(sg) for sg in g.subgoals],
            }

        if not self.active_mission:
            return {}
        return build_tree(self.active_mission)


# ─── Attention Mechanism ─────────────────────────────────────────────────────

class AttentionMechanism:
    """
    Selects which observations to focus on.

    Not a transformer attention — a decision-theoretic attention that
    weights sensor modalities based on current goals and context.

    Inspired by attention schema theory (Graziano 2013) and active
    perception in robotics (Bajcsy et al. 2018).
    """

    def __init__(
        self,
        modalities: List[str],
        default_weights: Optional[Dict[str, float]] = None,
    ):
        self.modalities = modalities
        self.weights = default_weights or {m: 1.0 / len(modalities) for m in modalities}
        self._history: deque = deque(maxlen=100)

    def compute_attention(
        self,
        observations: Dict[str, np.ndarray],
        goal: Optional[Goal],
        context: Dict[str, Any],
    ) -> AttentionFocus:
        """
        Compute attention weights for each modality.
        Weights depend on:
        - Current goal (navigating -> attend to lidar/depth)
        - Recent prediction errors (high error -> attend more)
        - Modality reliability
        """
        weights = dict(self.weights)

        # Goal-driven attention modulation
        if goal:
            desc = goal.description.lower()
            if any(w in desc for w in ("walk", "run", "navigate")):
                weights["lidar"] = weights.get("lidar", 0) + 0.3
                weights["depth"] = weights.get("depth", 0) + 0.2
            if any(w in desc for w in ("manipulate", "grasp", "pick")):
                weights["rgb"] = weights.get("rgb", 0) + 0.3
                weights["proprio"] = weights.get("proprio", 0) + 0.2
            if "balance" in desc:
                weights["imu"] = weights.get("imu", 0) + 0.4

        # Prediction error-driven attention
        if "prediction_errors" in context:
            errors = context["prediction_errors"]
            for mod, err in errors.items():
                if mod in weights:
                    weights[mod] += err * 0.5  # Higher error -> more attention

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Determine attended modalities (above threshold)
        threshold = 1.0 / len(self.modalities) * 0.5
        attended = [m for m, w in weights.items() if w > threshold]

        focus = AttentionFocus(
            attended_modalities=attended,
            attention_weights=weights,
            confidence=min(1.0, max(weights.values()) * 2),
        )
        self._history.append(focus)
        return focus

    def get_attended_observation(
        self,
        observations: Dict[str, np.ndarray],
        focus: AttentionFocus,
    ) -> Dict[str, np.ndarray]:
        """
        Return observation dict with unattended modalities optionally
        downweighted or masked.
        """
        result = {}
        for mod, obs in observations.items():
            w = focus.attention_weights.get(mod, 0.0)
            if w > 0.05:
                # Scale observation by attention weight
                result[mod] = obs * w
            else:
                # Zero out unattended modalities
                result[mod] = np.zeros_like(obs)
        return result


# ─── Consciousness Simulation Orchestrator ───────────────────────────────────

class ConsciousnessSimulation:
    """
    Orchestrates self-model, introspection, goal hierarchy, and attention.

    This is the top-level interface. It does NOT make the robot conscious.
    It provides structured meta-cognitive information that other modules
    (arbitration, curiosity, swarm) can use for better decisions.
    """

    def __init__(
        self,
        modalities: List[str] = ("proprio", "imu", "rgb", "depth", "lidar"),
        state_path: Optional[Path] = None,
    ):
        self.self_model = SelfModelManager()
        self.introspection = IntrospectionMonitor()
        self.goals = GoalHierarchyManager()
        self.attention = AttentionMechanism(list(modalities))
        self.state_path = state_path or Path(".moses_consciousness.json")

        # Integration state
        self.current_focus: Optional[AttentionFocus] = None
        self.current_introspection: Optional[IntrospectionSnapshot] = None

    def perceive(
        self,
        observations: Dict[str, np.ndarray],
        policy_confidence: float,
        value_estimate: float,
        model_error: float,
        compute_time_ms: float,
        memory_usage_mb: float,
    ) -> Dict[str, Any]:
        """
        Main perception loop. Call this every timestep.
        Updates self-model, introspection, and attention.
        Returns a dict with current cognitive state.
        """
        # Update self-model
        self.self_model.update_from_observation(observations)
        anomaly = self.self_model.detect_anomaly(observations)
        if anomaly:
            logger.warning(f"Self-model anomaly detected: {anomaly}")

        # Update introspection
        current_goal = self.goals.get_current_action()
        self.current_introspection = self.introspection.record(
            policy_confidence=policy_confidence,
            value_estimate=value_estimate,
            model_prediction_error=model_error,
            compute_time_ms=compute_time_ms,
            memory_usage_mb=memory_usage_mb,
            active_goal_id=current_goal.goal_id if current_goal else None,
            goal_stack_depth=len(self.goals.goal_stack),
        )

        # Compute attention
        context = {"prediction_errors": {"proprio": model_error}}
        self.current_focus = self.attention.compute_attention(
            observations=observations,
            goal=current_goal,
            context=context,
        )
        attended_obs = self.attention.get_attended_observation(
            observations, self.current_focus
        )

        # Check if strategy switch is recommended
        should_switch, reason = self.introspection.should_switch_strategy()

        return {
            "attended_observation": attended_obs,
            "attention_weights": self.current_focus.attention_weights,
            "self_model": self.self_model.get_model(),
            "introspection": self.current_introspection,
            "anomaly": anomaly,
            "should_switch_strategy": should_switch,
            "switch_reason": reason,
            "current_goal": current_goal,
            "is_overloaded": self.introspection.is_overloaded(),
        }

    def act(
        self,
        action: np.ndarray,
        reward: float,
        fell: bool,
        controller_used: str,
    ) -> None:
        """Report action outcome for self-model update."""
        self.self_model.update_from_outcome(reward, fell, controller_used)

    def set_mission(self, description: str) -> None:
        """Set the top-level mission."""
        self.goals.set_mission(description)

    def add_objective(self, description: str, priority: float = 1.0) -> Goal:
        """Add an objective to the current mission."""
        return self.goals.add_objective(description, priority=priority)

    def report_goal_progress(self, goal_id: str, progress: float) -> None:
        """Report progress on a goal."""
        self.goals.report_progress(goal_id, progress)

    def report_goal_failure(self, goal_id: str, reason: str) -> None:
        """Report goal failure for replanning."""
        self.goals.report_failure(goal_id, reason)

    def get_cognitive_state(self) -> Dict[str, Any]:
        """Return complete cognitive state for logging/debugging."""
        return {
            "self_model": {
                "battery": self.self_model.model.estimated_battery,
                "stability": self.self_model.model.estimated_stability,
                "total_steps": self.self_model.model.total_steps,
                "recent_falls": self.self_model.model.recent_falls,
            },
            "introspection": self.current_introspection,
            "attention": {
                "modalities": self.current_focus.attended_modalities if self.current_focus else [],
                "weights": self.current_focus.attention_weights if self.current_focus else {},
            },
            "goals": self.goals.get_goal_tree(),
        }

    def save(self) -> None:
        """Save cognitive state to disk."""
        state = self.get_cognitive_state()
        state["saved_at"] = datetime.utcnow().isoformat()
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load(self) -> None:
        """Load cognitive state from disk."""
        if not self.state_path.exists():
            return
        with open(self.state_path) as f:
            state = json.load(f)
        # Note: full restoration would require reconstructing all objects
        logger.info(f"Consciousness state loaded from {self.state_path}")
