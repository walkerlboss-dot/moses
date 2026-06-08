"""
Automatic Reward Design for Moses Locomotion

Learns reward-function weights via evolutionary search while monitoring for
reward hacking (e.g., exploiting simulator bugs or proxy mismatches).

Example
-------
>>> from moses.meta_learning import RewardShaper, RewardComponent
>>> shaper = RewardShaper(
...     components=[
...         RewardComponent("forward_velocity", weight=1.0),
...         RewardComponent("energy_penalty", weight=-0.01),
...         RewardComponent("alive_bonus", weight=0.1),
...     ]
... )
>>> best_weights = shaper.evolve(
...     evaluate_fn=lambda w: train_and_score(w),
...     generations=20,
...     population_size=30,
... )
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardComponent:
    """
    A single term in the shaped reward function.

    Attributes
    ----------
    name : str
        Human-readable identifier.
    weight : float
        Current weight (can be negative for penalties).
    baseline : float
        Expected raw value magnitude (used for normalization).
    bounds : Tuple[float, float]
        Min / max allowed weight during search.
    normalize : bool
        Whether to divide raw value by ``baseline`` before weighting.
    """

    name: str
    weight: float = 1.0
    baseline: float = 1.0
    bounds: Tuple[float, float] = (-10.0, 10.0)
    normalize: bool = True

    def compute(self, raw_value: float) -> float:
        """Apply normalization and weighting."""
        if self.normalize and self.baseline != 0.0:
            raw_value = raw_value / self.baseline
        return self.weight * raw_value


class RewardShaper:
    """
    Evolutionary reward-weight search with anti-hacking safeguards.

    The reward function is a weighted sum of :class:`RewardComponent` terms.
    CMA-ES-style evolution strategies optimize the weight vector.  Several
    monitors detect reward hacking:

    1. **Proxy mismatch**: High reward but low true success rate.
    2. **Exploit detection**: Sudden spikes in individual component raw values.
    3. **Correlation collapse**: Reward becomes uncorrelated with episode length.

    Parameters
    ----------
    components : List[RewardComponent]
        Reward terms.
    mutation_sigma : float
        Standard deviation of Gaussian weight perturbations.
    elite_frac : float
        Fraction of population kept as elites.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        components: List[RewardComponent],
        mutation_sigma: float = 0.3,
        elite_frac: float = 0.2,
        seed: int = 42,
    ) -> None:
        if not components:
            raise ValueError("At least one reward component is required")
        self.components = components
        self.n_components = len(components)
        self.mutation_sigma = mutation_sigma
        self.elite_frac = elite_frac
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Hacking detection thresholds
        self.proxy_mismatch_threshold: float = 0.5
        self.exploit_zscore_threshold: float = 4.0
        self.correlation_min: float = 0.3

        # History for diagnostics
        self._generation_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_weights(self) -> np.ndarray:
        """Return the current weight vector."""
        return np.array([c.weight for c in self.components], dtype=np.float64)

    def set_weights(self, weights: np.ndarray) -> None:
        """Assign a new weight vector."""
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != self.n_components:
            raise ValueError("Weight vector length mismatch")
        for c, w in zip(self.components, weights):
            c.weight = float(np.clip(w, c.bounds[0], c.bounds[1]))

    def compute_reward(self, raw_values: Dict[str, float]) -> float:
        """
        Compute total shaped reward from a dictionary of raw component values.

        Parameters
        ----------
        raw_values : dict
            Mapping from component ``name`` to raw scalar.

        Returns
        -------
        float
            Total shaped reward.
        """
        total = 0.0
        for comp in self.components:
            raw = raw_values.get(comp.name, 0.0)
            total += comp.compute(raw)
        return total

    def evolve(
        self,
        evaluate_fn: Callable[[np.ndarray], Tuple[float, Dict[str, Any]]],
        generations: int = 20,
        population_size: int = 30,
        initial_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run evolutionary search over reward weights.

        Parameters
        ----------
        evaluate_fn : callable
            Receives a weight vector and must return ``(score, metadata)``.
            ``score`` is maximized.  ``metadata`` should include keys:
            ``"success_rate"``, ``"episode_return"``, ``"episode_length"``,
            and optionally ``"component_raw"`` (dict of raw values).
        generations : int
            Number of generations.
        population_size : int
            Individuals per generation.
        initial_weights : ndarray, optional
            Starting point. Defaults to current component weights.

        Returns
        -------
        ndarray
            Best weight vector found.
        """
        if initial_weights is None:
            initial_weights = self.get_weights()
        else:
            self.set_weights(initial_weights)

        pop: List[Tuple[float, np.ndarray, Dict[str, Any]]] = []
        best_score = -float("inf")
        best_weights = initial_weights.copy()

        for gen in range(generations):
            # Build population
            candidates = self._generate_population(population_size, best_weights)
            scores: List[Tuple[float, np.ndarray, Dict[str, Any]]] = []

            for w in candidates:
                score, meta = evaluate_fn(w)
                # Anti-hacking penalty
                penalty = self._compute_hacking_penalty(meta)
                adjusted = score - penalty
                scores.append((adjusted, w, meta))
                logger.debug(
                    "Gen %d raw_score=%.3f penalty=%.3f adjusted=%.3f",
                    gen,
                    score,
                    penalty,
                    adjusted,
                )

            # Sort by adjusted score
            scores.sort(key=lambda t: t[0], reverse=True)
            pop = scores

            # Update best
            if scores[0][0] > best_score:
                best_score = scores[0][0]
                best_weights = scores[0][1].copy()
                self.set_weights(best_weights)
                logger.info(
                    "Gen %d new best adjusted=%.3f raw=%.3f weights=%s",
                    gen,
                    best_score,
                    scores[0][2].get("episode_return", 0.0),
                    np.round(best_weights, 3).tolist(),
                )

            # Logging
            self._generation_history.append(
                {
                    "generation": gen,
                    "best_adjusted": scores[0][0],
                    "mean_adjusted": float(np.mean([s[0] for s in scores])),
                    "best_meta": scores[0][2],
                }
            )

        logger.info("Evolution complete. Best score=%.3f", best_score)
        return best_weights

    def diagnose(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run hacking-detection diagnostics on a single evaluation metadata dict.

        Returns
        -------
        dict
            Keys: ``proxy_mismatch``, ``exploit_detected``, ``correlation_collapse``,
            ``overall_risk`` (0..1).
        """
        proxy_mismatch = self._check_proxy_mismatch(meta)
        exploit = self._check_exploit(meta)
        corr = self._check_correlation_collapse(meta)
        risk = float(np.clip((proxy_mismatch + exploit + (1 - corr)) / 3, 0.0, 1.0))
        return {
            "proxy_mismatch": proxy_mismatch,
            "exploit_detected": exploit,
            "correlation_collapse": corr,
            "overall_risk": risk,
        }

    def save(self, path: str) -> None:
        """Serialize components and history to JSON."""
        payload = {
            "components": [asdict(c) for c in self.components],
            "config": {
                "mutation_sigma": self.mutation_sigma,
                "elite_frac": self.elite_frac,
                "seed": self.seed,
            },
            "history": self._generation_history,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("RewardShaper state saved to %s", path)

    def load(self, path: str) -> None:
        """Restore components and history from JSON."""
        with open(path, "r", encoding="utf-8") as fh:
            payload: Dict[str, Any] = json.load(fh)
        self.components = [RewardComponent(**d) for d in payload["components"]]
        self.n_components = len(self.components)
        cfg = payload["config"]
        self.mutation_sigma = cfg["mutation_sigma"]
        self.elite_frac = cfg["elite_frac"]
        self.seed = cfg["seed"]
        self._rng = np.random.default_rng(self.seed)
        self._generation_history = payload.get("history", [])
        logger.info("RewardShaper state loaded from %s", path)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _generate_population(
        self, size: int, center: np.ndarray
    ) -> List[np.ndarray]:
        """Sample a population around ``center`` respecting component bounds."""
        pop: List[np.ndarray] = []
        for _ in range(size):
            noise = self._rng.normal(0, self.mutation_sigma, size=self.n_components)
            candidate = center + noise
            # Clip to per-component bounds
            for i, comp in enumerate(self.components):
                candidate[i] = float(np.clip(candidate[i], comp.bounds[0], comp.bounds[1]))
            pop.append(candidate)
        return pop

    def _compute_hacking_penalty(self, meta: Dict[str, Any]) -> float:
        """Compute a penalty that down-weights suspected hacked rewards."""
        diag = self.diagnose(meta)
        # Quadratic penalty on overall risk
        penalty = 1000.0 * (diag["overall_risk"] ** 2)
        return penalty

    def _check_proxy_mismatch(self, meta: Dict[str, Any]) -> float:
        """Return 0..1 mismatch score (higher = more mismatch)."""
        success_rate = meta.get("success_rate", 1.0)
        episode_return = meta.get("episode_return", 0.0)
        # If return is very high but success is low, something is wrong
        if episode_return <= 0.0:
            return 0.0
        mismatch = max(0.0, 1.0 - success_rate - self.proxy_mismatch_threshold)
        # Scale by return magnitude
        mismatch *= min(1.0, episode_return / 100.0)
        return float(mismatch)

    def _check_exploit(self, meta: Dict[str, Any]) -> float:
        """Detect sudden spikes in raw component values."""
        raw = meta.get("component_raw", {})
        if not raw:
            return 0.0
        # Simple z-score using historical means if available
        scores = []
        for name, val in raw.items():
            # Find component baseline
            comp = next((c for c in self.components if c.name == name), None)
            if comp is None or comp.baseline == 0.0:
                continue
            z = abs(val - comp.baseline) / max(1e-6, abs(comp.baseline))
            scores.append(z)
        if not scores:
            return 0.0
        max_z = max(scores)
        if max_z > self.exploit_zscore_threshold:
            return float(min(1.0, (max_z - self.exploit_zscore_threshold) / self.exploit_zscore_threshold))
        return 0.0

    def _check_correlation_collapse(self, meta: Dict[str, Any]) -> float:
        """
        Return correlation-like score between reward and episode length.

        Low correlation may indicate the agent is gaming the reward without
        surviving longer.
        """
        ep_return = meta.get("episode_return", 0.0)
        ep_length = meta.get("episode_length", 0.0)
        if ep_length <= 0:
            return 0.0
        # Heuristic: return per step should be within reasonable bounds
        rps = ep_return / ep_length
        # Typical healthy range: [-1, 5] per step for locomotion
        if -1.0 <= rps <= 5.0:
            return 1.0
        # Collapse if return per step is extreme
        return float(max(0.0, 1.0 - (abs(rps) - 5.0) / 10.0))
