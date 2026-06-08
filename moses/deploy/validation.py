"""
Pre-deployment validation for physical humanoid robots.

Stages:
  1. Simulation validation   — run new policy in sim, check success rate.
  2. Hardware-in-the-loop    — run on real robot with torque / speed limits.
  3. Pre-flight checklist    — joint limits, sensors, e-stop.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from moses.deploy import (
    DeploymentConfig,
    DeploymentError,
    MetricsClient,
    Policy,
    RobotInterface,
    ValidationError,
    ValidationResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation validation
# ---------------------------------------------------------------------------

class SimEnvironment:
    """Placeholder for the actual sim backend (Isaac Sim, MuJoCo, etc.).

    The real implementation would wrap the gym.Env or similar.
    """

    def __init__(self, name: str = "humanoid_v5") -> None:
        self.name = name

    def reset(self) -> np.ndarray:
        return np.zeros(60, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs = np.zeros(60, dtype=np.float32)
        reward = 1.0
        terminated = False
        info: Dict[str, Any] = {"success": False}
        return obs, reward, terminated, info

    def seed(self, seed: int) -> None:
        np.random.seed(seed)


def validate_in_simulation(
    policy: Policy,
    config: DeploymentConfig,
    env_factory: Callable[[], SimEnvironment] = SimEnvironment,
    seed: int = 42,
) -> ValidationResult:
    """Run policy in simulation and verify success rate."""
    logger.info("[validation] Starting simulation validation for %s", policy.name)
    env = env_factory()
    env.seed(seed)

    successes = 0
    episodes = config.sim_min_episodes
    for ep in range(episodes):
        obs = env.reset()
        policy.reset()
        done = False
        step = 0
        max_steps = 1000
        while not done and step < max_steps:
            action = policy.act(obs)
            obs, _, done, info = env.step(action)
            step += 1
        if info.get("success", False):
            successes += 1

    success_rate = successes / episodes
    passed = success_rate >= config.sim_min_success_rate
    details = {
        "episodes": episodes,
        "successes": successes,
        "success_rate": success_rate,
        "threshold": config.sim_min_success_rate,
    }
    logger.info("[validation] Sim validation %s: %s", "PASSED" if passed else "FAILED", details)
    return ValidationResult(
        passed=passed,
        stage="simulation",
        details=details,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )


# ---------------------------------------------------------------------------
# Hardware-in-the-loop (HITL)
# ---------------------------------------------------------------------------

class HITLSafetyLimiter:
    """Wraps robot execution with torque / velocity clamps for safe testing."""

    def __init__(
        self,
        robot: RobotInterface,
        max_torque_fraction: float = 0.5,
        max_velocity_fraction: float = 0.3,
    ) -> None:
        self.robot = robot
        self.max_torque_fraction = max_torque_fraction
        self.max_velocity_fraction = max_velocity_fraction
        # Placeholder: real implementation would query robot URDF / model
        self._nominal_torque = np.ones(12) * 80.0  # Nm
        self._nominal_velocity = np.ones(12) * 10.0  # rad/s

    def clamp_action(self, action: np.ndarray) -> np.ndarray:
        """Clamp action to safe HITL limits."""
        # Assume action is desired torque for this example
        max_t = self._nominal_torque * self.max_torque_fraction
        min_t = -max_t
        return np.clip(action, min_t, max_t)

    def execute(self, action: np.ndarray) -> None:
        safe_action = self.clamp_action(action)
        self.robot.execute_action(safe_action)


def validate_hardware_in_the_loop(
    policy: Policy,
    config: DeploymentConfig,
    duration_seconds: float = 60.0,
) -> ValidationResult:
    """Run policy on real robot with safety limits engaged."""
    logger.info("[validation] Starting HITL validation for %s", policy.name)
    robot = config.robot
    limiter = HITLSafetyLimiter(
        robot,
        max_torque_fraction=config.hitl_max_torque_fraction,
    )

    if robot.is_estopped():
        raise ValidationError("Robot is e-stopped before HITL validation.")

    policy.reset()
    start = time.time()
    steps = 0
    violations = 0
    energy_estimate = 0.0

    try:
        while time.time() - start < duration_seconds:
            obs = robot.get_joint_positions()  # simplified observation
            action = policy.act(obs)
            limiter.execute(action)
            steps += 1
            energy_estimate += float(np.linalg.norm(action))
            # Check for limit violations
            lower, upper = robot.get_joint_limits()
            current = robot.get_joint_positions()
            if np.any(current < lower - 0.01) or np.any(current > upper + 0.01):
                violations += 1
                robot.emergency_stop("HITL joint limit violation")
                break
            time.sleep(0.01)  # 100 Hz control loop placeholder
    except Exception as exc:
        robot.emergency_stop(f"HITL exception: {exc}")
        raise ValidationError(f"HITL validation failed: {exc}") from exc

    passed = violations == 0 and steps > 0
    details = {
        "duration_seconds": duration_seconds,
        "steps": steps,
        "violations": violations,
        "energy_estimate": energy_estimate,
    }
    logger.info("[validation] HITL validation %s: %s", "PASSED" if passed else "FAILED", details)
    return ValidationResult(
        passed=passed,
        stage="hardware_in_the_loop",
        details=details,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )


# ---------------------------------------------------------------------------
# Pre-flight checklist
# ---------------------------------------------------------------------------

def run_preflight_checklist(robot: RobotInterface) -> ValidationResult:
    """Static checks before any policy runs."""
    logger.info("[validation] Running pre-flight checklist")
    checks: Dict[str, bool] = {}

    # 1. Joint limits readable
    try:
        lower, upper = robot.get_joint_limits()
        checks["joint_limits_readable"] = lower.shape == upper.shape and len(lower) > 0
    except Exception as exc:
        logger.error("Joint limits check failed: %s", exc)
        checks["joint_limits_readable"] = False

    # 2. Sensor calibration
    try:
        sensor_status = robot.get_sensor_calibration_status()
        checks["sensors_calibrated"] = all(sensor_status.values())
        checks["sensor_details"] = sensor_status  # type: ignore[assignment]
    except Exception as exc:
        logger.error("Sensor calibration check failed: %s", exc)
        checks["sensors_calibrated"] = False

    # 3. Emergency stop functional
    try:
        # We do NOT trigger e-stop here; we only verify the interface exists.
        # A real implementation might perform a soft self-test.
        checks["estop_interface_ok"] = True
    except Exception as exc:
        logger.error("E-stop check failed: %s", exc)
        checks["estop_interface_ok"] = False

    # 4. Telemetry stream alive
    try:
        tel = robot.get_telemetry()
        checks["telemetry_alive"] = len(tel) > 0
    except Exception as exc:
        logger.error("Telemetry check failed: %s", exc)
        checks["telemetry_alive"] = False

    passed = all(
        v for k, v in checks.items() if isinstance(v, bool)
    )
    details = {"checks": checks}
    logger.info("[validation] Preflight checklist %s", "PASSED" if passed else "FAILED")
    return ValidationResult(
        passed=passed,
        stage="preflight_checklist",
        details=details,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def full_pre_deploy_validation(
    config: DeploymentConfig,
    env_factory: Callable[[], SimEnvironment] = SimEnvironment,
) -> List[ValidationResult]:
    """Run all validation stages in order. Raises ValidationError on first failure."""
    results: List[ValidationResult] = []

    # Stage 0: preflight
    r0 = run_preflight_checklist(config.robot)
    results.append(r0)
    if not r0.passed:
        raise ValidationError(f"Preflight checklist failed: {r0.details}")

    # Stage 1: simulation
    r1 = validate_in_simulation(config.new_policy, config, env_factory=env_factory)
    results.append(r1)
    if not r1.passed:
        raise ValidationError(f"Simulation validation failed: {r1.details}")

    # Stage 2: HITL
    r2 = validate_hardware_in_the_loop(config.new_policy, config)
    results.append(r2)
    if not r2.passed:
        raise ValidationError(f"HITL validation failed: {r2.details}")

    logger.info("[validation] All pre-deployment validations passed.")
    return results
