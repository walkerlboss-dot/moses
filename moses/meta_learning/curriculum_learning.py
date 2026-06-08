"""
Adaptive Curriculum Learning for Moses Locomotion

Automatically schedules environment difficulty based on training success rate.
Supports progressive difficulty increases and regression to easier tasks when
performance collapses.

Example
-------
>>> from moses.meta_learning import CurriculumScheduler, DifficultyConfig
>>> sched = CurriculumScheduler(
...     difficulty=DifficultyConfig(max_slope=15.0, max_obstacle_density=0.5),
...     success_window=50,
...     promote_threshold=0.85,
...     regress_threshold=0.40,
... )
>>> for episode in range(10000):
...     diff = sched.get_difficulty()
...     # train one episode with `diff`
...     success = env.run(diff)
...     sched.update(success)
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DifficultyConfig:
    """
    Bounds for curriculum difficulty parameters.

    Attributes
    ----------
    max_slope : float
        Maximum terrain slope in degrees.
    max_obstacle_density : float
        Maximum fraction of terrain covered by obstacles (0..1).
    max_speed : float
        Maximum target forward velocity (m/s).
    max_roughness : float
        Maximum terrain roughness amplitude (m).
    max_step_height : float
        Maximum step height (m).
    """

    max_slope: float = 15.0
    max_obstacle_density: float = 0.5
    max_speed: float = 3.0
    max_roughness: float = 0.05
    max_step_height: float = 0.15

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DifficultyConfig":
        return cls(**d)


class CurriculumScheduler:
    """
    Adaptive curriculum scheduler.

    Difficulty is represented as a normalized scalar ``level`` in ``[0, 1]``.
    The scheduler translates ``level`` into concrete environment parameters.
    Success rate over a sliding window drives level adjustments.

    Parameters
    ----------
    difficulty : DifficultyConfig
        Upper bounds for each difficulty dimension.
    success_window : int
        Number of recent episodes to average for success-rate estimation.
    promote_threshold : float
        Success rate above which difficulty increases.
    regress_threshold : float
        Success rate below which difficulty decreases.
    promote_delta : float
        Amount to increase ``level`` on promotion.
    regress_delta : float
        Amount to decrease ``level`` on regression.
    min_level : float
        Floor for difficulty level.
    max_level : float
        Ceiling for difficulty level.
    cooldown : int
        Minimum episodes between difficulty changes.
    smoothing : float
        EMA decay for success rate (0 = pure window average).
    """

    def __init__(
        self,
        difficulty: Optional[DifficultyConfig] = None,
        success_window: int = 50,
        promote_threshold: float = 0.85,
        regress_threshold: float = 0.40,
        promote_delta: float = 0.05,
        regress_delta: float = 0.10,
        min_level: float = 0.0,
        max_level: float = 1.0,
        cooldown: int = 10,
        smoothing: float = 0.0,
    ) -> None:
        self.difficulty = difficulty or DifficultyConfig()
        self.success_window = max(1, success_window)
        self.promote_threshold = promote_threshold
        self.regress_threshold = regress_threshold
        self.promote_delta = promote_delta
        self.regress_delta = regress_delta
        self.min_level = min_level
        self.max_level = max_level
        self.cooldown = cooldown
        self.smoothing = np.clip(smoothing, 0.0, 1.0)

        # State
        self._level: float = 0.0
        self._history: Deque[bool] = deque(maxlen=self.success_window)
        self._ema_success: float = 0.0
        self._steps_since_change: int = 0
        self._total_episodes: int = 0
        self._promotions: int = 0
        self._regressions: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def level(self) -> float:
        """Current normalized difficulty level."""
        return self._level

    def get_difficulty(self) -> Dict[str, float]:
        """
        Translate the current level into concrete environment parameters.

        Returns
        -------
        dict
            Keys: ``slope``, ``obstacle_density``, ``target_speed``,
            ``roughness``, ``step_height``.
        """
        t = self._level
        # Non-linear scaling: early levels stay very easy
        ease = 1.0 - (1.0 - t) ** 2
        return {
            "slope": ease * self.difficulty.max_slope,
            "obstacle_density": ease * self.difficulty.max_obstacle_density,
            "target_speed": (0.5 + 0.5 * ease) * self.difficulty.max_speed,
            "roughness": ease * self.difficulty.max_roughness,
            "step_height": ease * self.difficulty.max_step_height,
        }

    def update(self, success: bool, extra_reward: Optional[float] = None) -> Optional[str]:
        """
        Record episode outcome and optionally adjust difficulty.

        Parameters
        ----------
        success : bool
            Whether the episode achieved the task objective.
        extra_reward : float, optional
            Additional scalar signal (e.g., episode return) logged but not used
            for scheduling unless ``smoothing > 0`` and no window is available.

        Returns
        -------
        str or None
            ``"promote"``, ``"regress"``, or ``None`` if no change occurred.
        """
        self._history.append(success)
        self._total_episodes += 1
        self._steps_since_change += 1

        # Update EMA
        val = 1.0 if success else 0.0
        if self._ema_success == 0.0 and len(self._history) == 1:
            self._ema_success = val
        else:
            self._ema_success = self.smoothing * self._ema_success + (1 - self.smoothing) * val

        if self._steps_since_change < self.cooldown:
            return None

        rate = self._success_rate()
        if rate >= self.promote_threshold:
            return self._promote()
        if rate <= self.regress_threshold:
            return self._regress()
        return None

    def reset(self, level: float = 0.0) -> None:
        """Reset scheduler to a given difficulty level."""
        self._level = float(np.clip(level, self.min_level, self.max_level))
        self._history.clear()
        self._ema_success = 0.0
        self._steps_since_change = 0
        self._total_episodes = 0
        self._promotions = 0
        self._regressions = 0
        logger.info("Curriculum reset to level %.3f", self._level)

    def get_stats(self) -> Dict[str, Any]:
        """Return scheduler statistics."""
        return {
            "level": self._level,
            "success_rate": self._success_rate(),
            "ema_success": self._ema_success,
            "total_episodes": self._total_episodes,
            "promotions": self._promotions,
            "regressions": self._regressions,
            "history_len": len(self._history),
        }

    def save(self, path: str) -> None:
        """Persist state to JSON."""
        state = {
            "difficulty": self.difficulty.to_dict(),
            "config": {
                "success_window": self.success_window,
                "promote_threshold": self.promote_threshold,
                "regress_threshold": self.regress_threshold,
                "promote_delta": self.promote_delta,
                "regress_delta": self.regress_delta,
                "min_level": self.min_level,
                "max_level": self.max_level,
                "cooldown": self.cooldown,
                "smoothing": self.smoothing,
            },
            "state": {
                "level": self._level,
                "history": list(self._history),
                "ema_success": self._ema_success,
                "steps_since_change": self._steps_since_change,
                "total_episodes": self._total_episodes,
                "promotions": self._promotions,
                "regressions": self._regressions,
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        logger.info("Curriculum state saved to %s", path)

    def load(self, path: str) -> None:
        """Restore state from JSON."""
        with open(path, "r", encoding="utf-8") as fh:
            payload: Dict[str, Any] = json.load(fh)
        cfg = payload["config"]
        self.success_window = cfg["success_window"]
        self.promote_threshold = cfg["promote_threshold"]
        self.regress_threshold = cfg["regress_threshold"]
        self.promote_delta = cfg["promote_delta"]
        self.regress_delta = cfg["regress_delta"]
        self.min_level = cfg["min_level"]
        self.max_level = cfg["max_level"]
        self.cooldown = cfg["cooldown"]
        self.smoothing = cfg["smoothing"]

        st = payload["state"]
        self._level = st["level"]
        self._history = deque(st["history"], maxlen=self.success_window)
        self._ema_success = st["ema_success"]
        self._steps_since_change = st["steps_since_change"]
        self._total_episodes = st["total_episodes"]
        self._promotions = st["promotions"]
        self._regressions = st["regressions"]
        logger.info("Curriculum state loaded from %s", path)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _success_rate(self) -> float:
        if not self._history:
            return 0.0
        return float(np.mean(self._history))

    def _promote(self) -> str:
        old = self._level
        self._level = min(self._level + self.promote_delta, self.max_level)
        self._steps_since_change = 0
        self._promotions += 1
        logger.info(
            "Promoted difficulty %.3f -> %.3f (success_rate=%.2f)",
            old,
            self._level,
            self._success_rate(),
        )
        return "promote"

    def _regress(self) -> str:
        old = self._level
        self._level = max(self._level - self.regress_delta, self.min_level)
        self._steps_since_change = 0
        self._regressions += 1
        logger.warning(
            "Regressed difficulty %.3f -> %.3f (success_rate=%.2f)",
            old,
            self._level,
            self._success_rate(),
        )
        return "regress"
