"""
Shadow mode deployment for physical humanoid robots.

The new policy runs in parallel with the old (production) policy.
Observations are fed to both; only the old policy's actions are executed.
Outputs are compared to validate safety before real deployment.

SAFETY INVARIANT: new_policy.act() must NEVER result in a robot action.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from moses.deploy import (
    DeploymentConfig,
    DeploymentError,
    Policy,
    RobotInterface,
    ShadowMismatchError,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ShadowSample:
    timestamp: float
    observation: np.ndarray
    old_action: np.ndarray
    new_action: np.ndarray
    mismatch: float  # e.g., L2 norm or max component diff


class ShadowRunner:
    """Runs new policy in shadow mode alongside production policy."""

    def __init__(self, config: DeploymentConfig) -> None:
        self.config = config
        self.new_policy = config.new_policy
        self.old_policy = config.old_policy
        self.robot = config.robot
        self.metrics = config.metrics

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._samples: Deque[ShadowSample] = deque(maxlen=100_000)

        # Statistics
        self._total_samples = 0
        self._mismatch_count = 0
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin shadow mode in a background thread."""
        if self._running:
            logger.warning("[shadow] Already running.")
            return
        logger.info("[shadow] Starting shadow mode: old=%s new=%s",
                    self.old_policy.name, self.new_policy.name)
        self.new_policy.reset()
        self.old_policy.reset()
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop shadow mode and return summary statistics."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        summary = self._compute_summary()
        logger.info("[shadow] Stopped. Summary: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main shadow loop: observe, compute both actions, execute only old."""
        while self._running:
            if self.robot.is_estopped():
                logger.error("[shadow] Robot e-stopped; halting shadow mode.")
                self._running = False
                break

            try:
                obs = self._get_observation()
                old_action = self.old_policy.act(obs)
                new_action = self.new_policy.act(obs)

                # SAFETY: only old action reaches hardware
                self.robot.execute_action(old_action)

                mismatch = float(np.linalg.norm(old_action - new_action))
                sample = ShadowSample(
                    timestamp=time.time(),
                    observation=obs.copy(),
                    old_action=old_action.copy(),
                    new_action=new_action.copy(),
                    mismatch=mismatch,
                )
                with self._lock:
                    self._samples.append(sample)
                    self._total_samples += 1
                    if mismatch > self._mismatch_threshold():
                        self._mismatch_count += 1

                self._push_metrics(mismatch)
                self._maybe_check_divergence()

            except Exception as exc:
                logger.exception("[shadow] Loop error: %s", exc)
                self.robot.emergency_stop(f"Shadow mode error: {exc}")
                self._running = False
                break

            time.sleep(0.01)  # 100 Hz placeholder

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Gather observation from robot telemetry."""
        # Simplified: use joint positions as proxy observation.
        # Real system would stack IMU, proprioception, vision, etc.
        return self.robot.get_joint_positions()

    def _mismatch_threshold(self) -> float:
        """Dynamic threshold based on action scale. Placeholder."""
        return 0.5  # rad or Nm — tune per robot

    def _push_metrics(self, mismatch: float) -> None:
        self.metrics.push(
            "shadow.mismatch",
            mismatch,
            tags={
                "old_policy": self.old_policy.version,
                "new_policy": self.new_policy.version,
            },
        )

    def _maybe_check_divergence(self) -> None:
        """Raise if mismatch fraction exceeds config threshold."""
        with self._lock:
            total = self._total_samples
            bad = self._mismatch_count
        if total < self.config.shadow_min_samples:
            return
        fraction = bad / total
        if fraction > self.config.shadow_max_mismatch_fraction:
            msg = (
                f"Shadow divergence exceeded threshold: {fraction:.3%} "
                f"({bad}/{total}) > {self.config.shadow_max_mismatch_fraction:.3%}"
            )
            logger.error("[shadow] %s", msg)
            self.robot.emergency_stop(msg)
            self._running = False
            raise ShadowMismatchError(msg)

    def _compute_summary(self) -> Dict[str, Any]:
        with self._lock:
            samples = list(self._samples)
            total = self._total_samples
            bad = self._mismatch_count
        if not samples:
            return {"total_samples": 0, "duration_seconds": 0.0}

        mismatches = np.array([s.mismatch for s in samples])
        duration = samples[-1].timestamp - samples[0].timestamp if len(samples) > 1 else 0.0

        return {
            "old_policy": self.old_policy.version,
            "new_policy": self.new_policy.version,
            "total_samples": total,
            "mismatch_count": bad,
            "mismatch_fraction": bad / total if total else 0.0,
            "duration_seconds": duration,
            "mismatch_mean": float(np.mean(mismatches)),
            "mismatch_std": float(np.std(mismatches)),
            "mismatch_p99": float(np.percentile(mismatches, 99)),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }

    # ------------------------------------------------------------------
    # Public introspection
    # ------------------------------------------------------------------

    def get_recent_samples(self, n: int = 100) -> List[ShadowSample]:
        with self._lock:
            return list(self._samples)[-n:]

    def is_running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def run_shadow_validation(
    config: DeploymentConfig,
    duration_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Run shadow mode for a fixed duration or until divergence threshold hit.

    Returns summary dict. Raises ShadowMismatchError if divergence too high.
    """
    runner = ShadowRunner(config)
    runner.start()
    try:
        if duration_seconds is not None:
            time.sleep(duration_seconds)
        else:
            # Run until manually stopped or divergence triggered
            while runner.is_running():
                time.sleep(1.0)
    finally:
        summary = runner.stop()
    return summary
