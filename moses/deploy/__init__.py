"""
Moses v5.0 — Safe Deployment System for Physical Humanoid Robots
================================================================
Provides staged rollouts, A/B testing, shadow mode, pre-deployment validation,
and automatic rollback for policy deployment to physical robots.

Safety invariants:
  1. Never deploy an untested policy.
  2. Never jump to 100% rollout without passing lower stages.
  3. Rollback must be achievable with a single command.
  4. Shadow mode must never execute new-policy actions on hardware.
"""

from __future__ import annotations

__version__ = "5.0.0"
__all__ = [
    "Policy",
    "RobotInterface",
    "MetricsClient",
    "DeploymentConfig",
    "RolloutStage",
    "ValidationResult",
    "DeploymentError",
    "RollbackError",
    "ValidationError",
    "ShadowMismatchError",
]

import abc
import dataclasses
import enum
import numpy as np
from typing import Any, Dict, List, Optional, Protocol, Tuple


# ---------------------------------------------------------------------------
# Core protocols
# ---------------------------------------------------------------------------

class Policy(Protocol):
    """Generic policy interface."""

    def reset(self) -> None:
        ...

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return action given observation."""
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def version(self) -> str:
        ...


class RobotInterface(Protocol):
    """Abstraction over physical robot hardware."""

    def is_estopped(self) -> bool:
        ...

    def emergency_stop(self, reason: str) -> None:
        ...

    def get_joint_positions(self) -> np.ndarray:
        ...

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) joint limits."""
        ...

    def get_sensor_calibration_status(self) -> Dict[str, bool]:
        ...

    def execute_action(self, action: np.ndarray) -> None:
        ...

    def get_telemetry(self) -> Dict[str, float]:
        """Return latest telemetry snapshot."""
        ...


class MetricsClient(Protocol):
    """Backend for pushing / querying deployment metrics."""

    def push(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        ...

    def query(
        self,
        metric_name: str,
        tags: Optional[Dict[str, str]] = None,
        window_seconds: int = 60,
    ) -> List[float]:
        ...


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class RolloutStage(enum.Enum):
    SHADOW = "shadow"
    PCT_1 = "1%"
    PCT_10 = "10%"
    PCT_50 = "50%"
    PCT_100 = "100%"
    ROLLED_BACK = "rolled_back"


@dataclasses.dataclass(frozen=True)
class DeploymentConfig:
    """Immutable configuration for a deployment."""

    new_policy: Policy
    old_policy: Policy
    robot: RobotInterface
    metrics: MetricsClient
    # Rollout thresholds
    stage_durations_minutes: Dict[RolloutStage, int] = dataclasses.field(
        default_factory=lambda: {
            RolloutStage.PCT_1: 10,
            RolloutStage.PCT_10: 30,
            RolloutStage.PCT_50: 60,
            RolloutStage.PCT_100: 0,
        }
    )
    # Metric degradation thresholds (fraction of old policy baseline)
    min_success_rate_ratio: float = 0.95
    max_energy_ratio: float = 1.20
    min_stability_ratio: float = 0.90
    # Shadow mode
    shadow_max_mismatch_fraction: float = 0.05
    shadow_min_samples: int = 1000
    # Validation
    sim_min_episodes: int = 100
    sim_min_success_rate: float = 0.98
    hitl_max_torque_fraction: float = 0.5


@dataclasses.dataclass(frozen=True)
class ValidationResult:
    passed: bool
    stage: str
    details: Dict[str, Any]
    timestamp: str


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DeploymentError(Exception):
    """Base deployment exception."""
    pass


class RollbackError(DeploymentError):
    """Raised when rollback itself fails."""
    pass


class ValidationError(DeploymentError):
    """Raised when pre-deployment validation fails."""
    pass


class ShadowMismatchError(DeploymentError):
    """Raised when shadow-mode divergence exceeds threshold."""
    pass
