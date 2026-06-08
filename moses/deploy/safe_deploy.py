"""
Safe staged rollout for physical humanoid robots.

Stages:
  SHADOW  →  1%  →  10%  →  50%  →  100%

At each stage we A/B test old vs new policy on a traffic fraction.
Metrics (success rate, energy, stability) are monitored continuously.
Auto-rollback triggers if new policy degrades relative to baseline.

Rollback is one command: `deployment.rollback()`
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from moses.deploy import (
    DeploymentConfig,
    DeploymentError,
    MetricsClient,
    Policy,
    RobotInterface,
    RollbackError,
    RolloutStage,
)
from moses.deploy.shadow import ShadowRunner
from moses.deploy.validation import full_pre_deploy_validation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics snapshot
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MetricsSnapshot:
    success_rate: float
    energy_per_step: float
    stability_score: float
    sample_count: int
    timestamp: float

    @staticmethod
    def from_queries(metrics: MetricsClient, policy_version: str, window_seconds: int = 60) -> MetricsSnapshot:
        success = metrics.query("policy.success", tags={"version": policy_version}, window_seconds=window_seconds)
        energy = metrics.query("policy.energy", tags={"version": policy_version}, window_seconds=window_seconds)
        stability = metrics.query("policy.stability", tags={"version": policy_version}, window_seconds=window_seconds)

        def _mean(vals: List[float]) -> float:
            return float(np.mean(vals)) if vals else 0.0

        return MetricsSnapshot(
            success_rate=_mean(success),
            energy_per_step=_mean(energy),
            stability_score=_mean(stability),
            sample_count=len(success),
            timestamp=time.time(),
        )


# ---------------------------------------------------------------------------
# Deployment engine
# ---------------------------------------------------------------------------

class SafeDeployment:
    """Orchestrates staged rollout with A/B testing and auto-rollback."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.new_policy = config.new_policy
        self.old_policy = config.old_policy
        self.robot = config.robot
        self.metrics = config.metrics

        self._stage = RolloutStage.SHADOW
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Baseline metrics from old policy (populated at stage entry)
        self._baseline: Optional[MetricsSnapshot] = None
        # Current active policy for execution
        self._active_policy: Policy = self.old_policy

        # History for audit
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deploy(self) -> None:
        """Full deployment pipeline: validate → shadow → staged rollout."""
        logger.info("[deploy] Starting safe deployment of %s", self.new_policy.name)

        # 1. Pre-deployment validation
        logger.info("[deploy] Running pre-deployment validation...")
        full_pre_deploy_validation(self.config)
        self._log_event("validation_passed")

        # 2. Shadow mode
        self._enter_stage(RolloutStage.SHADOW)
        shadow_runner = ShadowRunner(self.config)
        shadow_runner.start()
        shadow_duration = 300.0  # 5 minutes default shadow
        logger.info("[deploy] Shadow mode running for %.0f seconds...", shadow_duration)
        time.sleep(shadow_duration)
        shadow_summary = shadow_runner.stop()
        self._log_event("shadow_complete", details=shadow_summary)
        logger.info("[deploy] Shadow complete: %s", shadow_summary)

        # 3. Staged rollout
        stages = [RolloutStage.PCT_1, RolloutStage.PCT_10, RolloutStage.PCT_50, RolloutStage.PCT_100]
        for stage in stages:
            self._enter_stage(stage)
            if not self._run_stage(stage):
                logger.error("[deploy] Stage %s failed metrics; initiating rollback.", stage.value)
                self.rollback()
                return

        logger.info("[deploy] Deployment complete at 100%%.")
        self._log_event("deploy_complete")

    def rollback(self) -> None:
        """One-command rollback to old policy."""
        logger.warning("[deploy] ROLLBACK initiated!")
        with self._lock:
            self._active_policy = self.old_policy
            self._stage = RolloutStage.ROLLED_BACK
            self._running = False

        # Ensure robot is safe
        try:
            self.robot.emergency_stop("Rollback triggered")
        except Exception as exc:
            logger.error("[deploy] E-stop during rollback failed: %s", exc)
            raise RollbackError(f"Rollback e-stop failed: {exc}") from exc

        # Signal metrics
        self.metrics.push("deployment.rollback", 1.0, tags={
            "new_policy": self.new_policy.version,
            "old_policy": self.old_policy.version,
        })
        self._log_event("rollback")
        logger.warning("[deploy] Rollback complete. Robot e-stopped. Old policy is active.")

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "stage": self._stage.value,
                "active_policy": self._active_policy.name,
                "active_version": self._active_policy.version,
                "running": self._running,
                "baseline": dataclasses.asdict(self._baseline) if self._baseline else None,
                "history": self._history,
            }

    # ------------------------------------------------------------------
    # Stage machinery
    # ------------------------------------------------------------------

    def _enter_stage(self, stage: RolloutStage) -> None:
        with self._lock:
            self._stage = stage
        self._baseline = MetricsSnapshot.from_queries(self.metrics, self.old_policy.version)
        self._log_event("stage_enter", {"stage": stage.value})
        logger.info("[deploy] Entered stage %s. Baseline: %s", stage.value, self._baseline)

    def _run_stage(self, stage: RolloutStage) -> bool:
        """Run a rollout stage. Return True if metrics pass, False if rollback needed."""
        traffic_fraction = self._stage_to_fraction(stage)
        duration_minutes = self.config.stage_durations_minutes.get(stage, 10)
        duration_seconds = duration_minutes * 60

        logger.info(
            "[deploy] Running stage %s at %.0f%% traffic for %d minutes",
            stage.value, traffic_fraction * 100, duration_minutes,
        )

        self._running = True
        self._thread = threading.Thread(
            target=self._control_loop,
            args=(traffic_fraction,),
            daemon=True,
        )
        self._thread.start()

        # Monitor during stage
        monitor_interval = 10.0  # seconds
        elapsed = 0.0
        while elapsed < duration_seconds and self._running:
            time.sleep(monitor_interval)
            elapsed += monitor_interval
            if not self._check_metrics():
                self._running = False
                self._thread.join(timeout=5.0)
                return False

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        # Final check
        return self._check_metrics()

    def _control_loop(self, new_policy_fraction: float) -> None:
        """Execute policies according to traffic split."""
        while self._running:
            if self.robot.is_estopped():
                logger.error("[deploy] E-stop detected; halting control loop.")
                self._running = False
                break

            try:
                obs = self.robot.get_joint_positions()
                # A/B traffic split
                if random.random() < new_policy_fraction:
                    policy = self.new_policy
                    tag = self.new_policy.version
                else:
                    policy = self.old_policy
                    tag = self.old_policy.version

                action = policy.act(obs)
                self.robot.execute_action(action)

                # Push per-step metrics (simplified)
                telemetry = self.robot.get_telemetry()
                self.metrics.push("policy.success", telemetry.get("success", 1.0), tags={"version": tag})
                self.metrics.push("policy.energy", telemetry.get("energy", 0.0), tags={"version": tag})
                self.metrics.push("policy.stability", telemetry.get("stability", 1.0), tags={"version": tag})

            except Exception as exc:
                logger.exception("[deploy] Control loop error: %s", exc)
                self.robot.emergency_stop(f"Control loop error: {exc}")
                self._running = False
                break

            time.sleep(0.01)

    # ------------------------------------------------------------------
    # Metric checks
    # ------------------------------------------------------------------

    def _check_metrics(self) -> bool:
        """Compare new policy metrics against baseline. Return False if degraded."""
        if self._baseline is None:
            return True

        new_metrics = MetricsSnapshot.from_queries(self.metrics, self.new_policy.version, window_seconds=60)
        if new_metrics.sample_count < 10:
            return True  # Not enough data yet

        old = self._baseline
        checks = {
            "success_rate_ok": (
                new_metrics.success_rate >= old.success_rate * self.config.min_success_rate_ratio
            ),
            "energy_ok": (
                new_metrics.energy_per_step <= old.energy_per_step * self.config.max_energy_ratio
                or old.energy_per_step == 0.0  # avoid div-by-zero
            ),
            "stability_ok": (
                new_metrics.stability_score >= old.stability_score * self.config.min_stability_ratio
            ),
        }

        if not all(checks.values()):
            logger.error(
                "[deploy] Metric degradation detected! new=%s old=%s checks=%s",
                new_metrics, old, checks,
            )
            self._log_event("metric_degradation", {
                "new": dataclasses.asdict(new_metrics),
                "old": dataclasses.asdict(old),
                "checks": checks,
            })
            return False
        return True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _stage_to_fraction(stage: RolloutStage) -> float:
        mapping = {
            RolloutStage.PCT_1: 0.01,
            RolloutStage.PCT_10: 0.10,
            RolloutStage.PCT_50: 0.50,
            RolloutStage.PCT_100: 1.00,
        }
        return mapping.get(stage, 0.0)

    def _log_event(self, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "event": event,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "stage": self._stage.value,
            "details": details or {},
        }
        self._history.append(entry)


# ---------------------------------------------------------------------------
# One-command helpers
# ---------------------------------------------------------------------------

def deploy_policy(config: DeploymentConfig) -> SafeDeployment:
    """High-level API: create deployment and run it."""
    deployment = SafeDeployment(config)
    # Run in background thread so caller can inspect status / trigger rollback
    thread = threading.Thread(target=deployment.deploy, daemon=True)
    thread.start()
    return deployment


def rollback_deployment(deployment: SafeDeployment) -> None:
    """One-command rollback."""
    deployment.rollback()
