"""
moses/data/preprocessing.py
===========================
Data preprocessing for Moses v5.0 continuous training.

Modules:
  • Normalization   — observation scaling, action clipping
  • Augmentation    — noise injection, time warping, mirroring
  • Filtering       — remove bad episodes, outliers
  • Balancing       — ensure diverse task coverage

Design decisions (locked 2026-06-08):
  - All transforms are stateful where needed (fit on train, apply to val).
  - Streaming / chunk-based to handle TB-scale data.
  - NumPy-first; torch.Tensor conversion happens at dataloader time.
  - Augmentations are applied per-episode, not per-step, to preserve dynamics.
"""

from __future__ import annotations

import abc
import copy
import logging
import math
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from moses.data.ingestion import Episode

logger = logging.getLogger("moses.data.preprocessing")

# ---------------------------------------------------------------------------
# Optional soft-deps
# ---------------------------------------------------------------------------
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

try:
    import scipy.ndimage  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    scipy = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Base transform protocol
# ---------------------------------------------------------------------------

class Transform(abc.ABC):
    """Base class for all preprocessing transforms."""

    @abc.abstractmethod
    def __call__(self, episode: Episode) -> Episode:
        ...

    def fit(self, episodes: Iterator[Episode]) -> "Transform":
        """Stateful transforms override this. Default: no-op."""
        return self


class IdentityTransform(Transform):
    def __call__(self, episode: Episode) -> Episode:
        return episode


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ObservationNormalizer(Transform):
    """Z-score normalizer for observation features.

    Supports both vector observations and image observations (per-channel).
    """

    keys: Sequence[str] = ("proprioception", "state")
    image_keys: Sequence[str] = ("rgb", "image", "depth")
    eps: float = 1e-6
    _mean: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _std: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _img_mean: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _img_std: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def fit(self, episodes: Iterator[Episode]) -> "ObservationNormalizer":
        # Accumulate running stats for vector obs
        sums: dict[str, np.ndarray] = {}
        sumsqs: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        img_sums: dict[str, np.ndarray] = {}
        img_sumsqs: dict[str, np.ndarray] = {}
        img_counts: dict[str, int] = {}

        for ep in episodes:
            for step_obs in ep.observations:
                for key in self.keys:
                    val = step_obs.get(key)
                    if not isinstance(val, np.ndarray):
                        continue
                    if key not in sums:
                        sums[key] = np.zeros(val.shape, dtype=np.float64)
                        sumsqs[key] = np.zeros(val.shape, dtype=np.float64)
                        counts[key] = 0
                    sums[key] += val.astype(np.float64)
                    sumsqs[key] += (val.astype(np.float64) ** 2)
                    counts[key] += 1
                for key in self.image_keys:
                    val = step_obs.get(key)
                    if not isinstance(val, np.ndarray) or val.ndim not in (2, 3):
                        continue
                    # Per-channel mean/std for images
                    if val.ndim == 2:
                        val = val[..., np.newaxis]
                    if key not in img_sums:
                        img_sums[key] = np.zeros(val.shape[-1], dtype=np.float64)
                        img_sumsqs[key] = np.zeros(val.shape[-1], dtype=np.float64)
                        img_counts[key] = 0
                    img_sums[key] += val.mean(axis=(0, 1)).astype(np.float64)
                    img_sumsqs[key] += (val.astype(np.float64) ** 2).mean(axis=(0, 1))
                    img_counts[key] += val.shape[0] * val.shape[1]

        for key in sums:
            n = counts[key]
            self._mean[key] = (sums[key] / n).astype(np.float32)
            self._std[key] = np.sqrt(np.maximum(sumsqs[key] / n - self._mean[key] ** 2, 0)).astype(np.float32) + self.eps
        for key in img_sums:
            n = img_counts[key]
            self._img_mean[key] = (img_sums[key] / max(n, 1)).astype(np.float32)
            var = np.maximum(img_sumsqs[key] / max(n, 1) - self._img_mean[key] ** 2, 0)
            self._img_std[key] = (np.sqrt(var) + self.eps).astype(np.float32)
        return self

    def __call__(self, episode: Episode) -> Episode:
        ep = copy.deepcopy(episode)
        for step_obs in ep.observations:
            for key in self.keys:
                if key in self._mean and key in step_obs:
                    val = np.array(step_obs[key], dtype=np.float32)
                    step_obs[key] = (val - self._mean[key]) / self._std[key]
            for key in self.image_keys:
                if key in self._img_mean and key in step_obs:
                    val = np.array(step_obs[key], dtype=np.float32)
                    if val.ndim == 2:
                        val = val[..., np.newaxis]
                    step_obs[key] = ((val - self._img_mean[key]) / self._img_std[key]).astype(np.float32)
        return ep


@dataclass(slots=True)
class ActionNormalizer(Transform):
    """Min-max scaler for actions (or Z-score if *zscore* is True)."""

    zscore: bool = False
    eps: float = 1e-6
    clip: tuple[float, float] | None = (-5.0, 5.0)
    _min: np.ndarray | None = field(default=None, repr=False)
    _max: np.ndarray | None = field(default=None, repr=False)
    _mean: np.ndarray | None = field(default=None, repr=False)
    _std: np.ndarray | None = field(default=None, repr=False)

    def fit(self, episodes: Iterator[Episode]) -> "ActionNormalizer":
        actions_list: list[np.ndarray] = []
        for ep in episodes:
            if ep.actions.size > 0:
                actions_list.append(ep.actions)
        if not actions_list:
            logger.warning("ActionNormalizer.fit: no actions found")
            return self
        all_actions = np.concatenate(actions_list, axis=0)
        self._min = all_actions.min(axis=0).astype(np.float32)
        self._max = all_actions.max(axis=0).astype(np.float32)
        self._mean = all_actions.mean(axis=0).astype(np.float32)
        self._std = all_actions.std(axis=0).astype(np.float32) + self.eps
        return self

    def __call__(self, episode: Episode) -> Episode:
        ep = copy.deepcopy(episode)
        if ep.actions.size == 0:
            return ep
        if self.zscore and self._mean is not None and self._std is not None:
            ep.actions = (ep.actions - self._mean) / self._std
        elif self._min is not None and self._max is not None:
            rng = (self._max - self._min) + self.eps
            ep.actions = 2.0 * (ep.actions - self._min) / rng - 1.0
        if self.clip:
            ep.actions = np.clip(ep.actions, self.clip[0], self.clip[1])
        return ep


@dataclass(slots=True)
class ActionClipper(Transform):
    """Hard clip actions to a fixed range (no fitting required)."""

    low: float = -1.0
    high: float = 1.0

    def __call__(self, episode: Episode) -> Episode:
        ep = copy.deepcopy(episode)
        ep.actions = np.clip(ep.actions, self.low, self.high)
        return ep


@dataclass(slots=True)
class RewardNormalizer(Transform):
    """Z-score normalizer for rewards (useful for PPO-style training)."""

    eps: float = 1e-6
    _mean: float = 0.0
    _std: float = 1.0

    def fit(self, episodes: Iterator[Episode]) -> "RewardNormalizer":
        rewards: list[np.ndarray] = []
        for ep in episodes:
            if ep.rewards.size > 0:
                rewards.append(ep.rewards)
        if not rewards:
            return self
        all_r = np.concatenate(rewards)
        self._mean = float(all_r.mean())
        self._std = float(all_r.std()) + self.eps
        return self

    def __call__(self, episode: Episode) -> Episode:
        ep = copy.deepcopy(episode)
        if ep.rewards.size > 0:
            ep.rewards = (ep.rewards - self._mean) / self._std
        return ep


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ActionNoise(Transform):
    """Inject Gaussian noise into actions."""

    std: float = 0.01
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, episode: Episode) -> Episode:
        ep = copy.deepcopy(episode)
        if ep.actions.size > 0:
            noise = self._rng.normal(0, self.std, size=ep.actions.shape).astype(np.float32)
            ep.actions = ep.actions + noise
        return ep


@dataclass(slots=True)
class TimeWarp(Transform):
    """Apply smooth temporal warping to an episode via cubic interpolation.

    Preserves episode length but non-linearly stretches/compresses time.
    """

    sigma: float = 0.2
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, episode: Episode) -> Episode:
        if len(episode) < 4:
            return episode
        ep = copy.deepcopy(episode)
        n = len(ep)
        # Generate random warp knots
        knots = np.linspace(0, n - 1, num=max(4, n // 20))
        warp = knots + self._rng.normal(0, self.sigma * n, size=knots.shape)
        warp = np.clip(warp, 0, n - 1)
        # Interpolate to uniform grid
        new_indices = np.interp(np.arange(n), knots, warp)
        ep.actions = self._interp1d(ep.actions, new_indices)
        ep.rewards = self._interp1d(ep.rewards[:, np.newaxis], new_indices)[:, 0]
        ep.dones = ep.dones  # keep dones as-is (binary)
        ep.timestamps = np.interp(np.arange(n), knots, ep.timestamps[knots.astype(int)])
        # Observations: interpolate vector fields, keep images
        new_observations: list[dict[str, Any]] = []
        for i in range(n):
            src_idx = int(np.clip(new_indices[i], 0, n - 1))
            frame = copy.deepcopy(ep.observations[src_idx])
            for key, val in frame.items():
                if isinstance(val, np.ndarray) and val.ndim == 1 and val.dtype.kind == "f":
                    # Simple linear interp for 1D float arrays
                    lo = int(np.floor(new_indices[i]))
                    hi = int(np.ceil(new_indices[i]))
                    alpha = new_indices[i] - lo
                    lo = np.clip(lo, 0, n - 1)
                    hi = np.clip(hi, 0, n - 1)
                    frame[key] = (1 - alpha) * ep.observations[lo].get(key, val) + alpha * ep.observations[hi].get(key, val)
            new_observations.append(frame)
        ep.observations = new_observations
        return ep

    @staticmethod
    def _interp1d(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Linear interpolation of *arr* at *indices* along axis 0."""
        if arr.ndim == 1:
            return np.interp(indices, np.arange(len(arr)), arr).astype(arr.dtype)
        result = np.zeros((len(indices), arr.shape[1]), dtype=arr.dtype)
        for d in range(arr.shape[1]):
            result[:, d] = np.interp(indices, np.arange(len(arr)), arr[:, d])
        return result


@dataclass(slots=True)
class MirrorTransform(Transform):
    """Mirror an episode (flip left-right for images, negate lateral actions).

    Assumes action index 0 corresponds to lateral (x) movement.
    """

    lateral_action_idx: int = 0
    image_keys: Sequence[str] = ("rgb", "image")

    def __call__(self, episode: Episode) -> Episode:
        if cv2 is None:
            return episode
        ep = copy.deepcopy(episode)
        if ep.actions.ndim >= 2 and ep.actions.shape[1] > self.lateral_action_idx:
            ep.actions[:, self.lateral_action_idx] *= -1.0
        for step_obs in ep.observations:
            for key in self.image_keys:
                if key in step_obs and isinstance(step_obs[key], np.ndarray):
                    img = step_obs[key]
                    if img.ndim == 3:
                        step_obs[key] = cv2.flip(img, 1)
                    elif img.ndim == 2:
                        step_obs[key] = cv2.flip(img, 1)
        return ep


@dataclass(slots=True)
class ImageColorJitter(Transform):
    """Random brightness/contrast/saturation jitter for RGB images."""

    brightness: float = 0.1
    contrast: float = 0.1
    saturation: float = 0.1
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, episode: Episode) -> Episode:
        if cv2 is None:
            return episode
        ep = copy.deepcopy(episode)
        for step_obs in ep.observations:
            for key in ("rgb", "image"):
                if key not in step_obs:
                    continue
                img = step_obs[key]
                if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
                    continue
                # Convert to HSV, jitter, convert back
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 2] *= 1.0 + self._rng.uniform(-self.brightness, self.brightness)
                hsv[:, :, 1] *= 1.0 + self._rng.uniform(-self.saturation, self.saturation)
                hsv[:, :, 2] = (hsv[:, :, 2] - 128.0) * (1.0 + self._rng.uniform(-self.contrast, self.contrast)) + 128.0
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                step_obs[key] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return ep


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LengthFilter(Transform):
    """Drop episodes outside length bounds."""

    min_steps: int = 10
    max_steps: int = 10_000

    def __call__(self, episode: Episode) -> Episode | None:
        n = len(episode)
        if n < self.min_steps or n > self.max_steps:
            return None
        return episode


@dataclass(slots=True)
class OutlierFilter(Transform):
    """Drop episodes with outlier action magnitudes (robust Z-score)."""

    threshold: float = 5.0

    def __call__(self, episode: Episode) -> Episode | None:
        if episode.actions.size == 0:
            return episode
        median = np.median(episode.actions)
        mad = np.median(np.abs(episode.actions - median)) + 1e-6
        robust_z = np.abs((episode.actions - median) / (1.4826 * mad))
        if np.any(robust_z > self.threshold):
            return None
        return episode


@dataclass(slots=True)
class RewardFilter(Transform):
    """Drop episodes with suspicious reward statistics."""

    min_total_reward: float | None = None
    max_total_reward: float | None = None
    drop_all_nan: bool = True

    def __call__(self, episode: Episode) -> Episode | None:
        if self.drop_all_nan and np.isnan(episode.rewards).all():
            return None
        total = float(episode.rewards.sum())
        if self.min_total_reward is not None and total < self.min_total_reward:
            return None
        if self.max_total_reward is not None and total > self.max_total_reward:
            return None
        return episode


@dataclass(slots=True)
class SuccessRateFilter(Transform):
    """Drop episodes below a success threshold (based on final done/reward)."""

    min_final_reward: float = 0.0
    require_done: bool = True

    def __call__(self, episode: Episode) -> Episode | None:
        if len(episode) == 0:
            return None
        if self.require_done and not episode.dones[-1]:
            return None
        if episode.rewards.size > 0 and episode.rewards[-1] < self.min_final_reward:
            return None
        return episode


# ---------------------------------------------------------------------------
# Balancing
# ---------------------------------------------------------------------------

@dataclass
class TaskBalancer:
    """Re-sample episodes so that tasks are represented equally.

    Supports oversampling (duplicate minority tasks) and undersampling
    (subsample majority tasks).  Operates on an iterator, so it buffers
    in memory — use only after filtering has reduced dataset size.
    """

    strategy: str = "oversample"  # "oversample" | "undersample" | "cap"
    target_count: int | None = None  # if None, uses median task count
    max_count: int | None = None  # hard cap per task (for "cap")
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def balance(self, episodes: Iterator[Episode]) -> Iterator[Episode]:
        # Buffer all episodes (assumes filtered set fits in RAM)
        buffer = list(episodes)
        from collections import defaultdict

        by_task: dict[str, list[Episode]] = defaultdict(list)
        for ep in buffer:
            by_task[ep.task_label].append(ep)
        if not by_task:
            return iter([])

        counts = {task: len(eps) for task, eps in by_task.items()}
        if self.strategy == "oversample":
            target = self.target_count or max(counts.values())
            balanced: list[Episode] = []
            for task, eps in by_task.items():
                n = len(eps)
                if n >= target:
                    balanced.extend(self._rng.choice(eps, size=target, replace=False).tolist())
                else:
                    extra = self._rng.choice(eps, size=target - n, replace=True).tolist()
                    balanced.extend(eps + extra)
            self._rng.shuffle(balanced)
            return iter(balanced)
        elif self.strategy == "undersample":
            target = self.target_count or min(counts.values())
            balanced = []
            for task, eps in by_task.items():
                balanced.extend(self._rng.choice(eps, size=min(target, len(eps)), replace=False).tolist())
            self._rng.shuffle(balanced)
            return iter(balanced)
        elif self.strategy == "cap":
            cap = self.max_count or (self.target_count or max(counts.values()))
            balanced = []
            for task, eps in by_task.items():
                n = min(len(eps), cap)
                balanced.extend(self._rng.choice(eps, size=n, replace=False).tolist())
            self._rng.shuffle(balanced)
            return iter(balanced)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


@dataclass
class QualityWeightedSampler:
    """Sample episodes weighted by a quality score (e.g. success rate, diversity)."""

    temperature: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def sample(self, episodes: list[Episode], n_samples: int) -> list[Episode]:
        if not episodes:
            return []
        scores = np.array([float(ep.metadata.get("quality_score", 1.0)) for ep in episodes], dtype=np.float64)
        # Softmax with temperature
        exp_scores = np.exp((scores - scores.max()) / max(self.temperature, 1e-6))
        probs = exp_scores / exp_scores.sum()
        indices = self._rng.choice(len(episodes), size=n_samples, replace=True, p=probs)
        return [episodes[i] for i in indices]


# ---------------------------------------------------------------------------
# Pipeline composer
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """Composable preprocessing pipeline.

    Usage:
        pipe = PreprocessingPipeline()
        pipe.add(ObservationNormalizer().fit(train_eps))
        pipe.add(ActionNormalizer().fit(train_eps))
        pipe.add(ActionNoise(std=0.02))
        pipe.add(LengthFilter(min_steps=20))
        for ep in pipe.apply(raw_eps):
            ...
    """

    def __init__(self) -> None:
        self.transforms: list[Transform] = []
        self.filters: list[Callable[[Episode], Episode | None]] = []

    def add(self, transform: Transform | Callable[[Episode], Episode | None]) -> "PreprocessingPipeline":
        if isinstance(transform, Transform):
            self.transforms.append(transform)
        else:
            self.filters.append(transform)
        return self

    def apply(self, episodes: Iterator[Episode]) -> Iterator[Episode]:
        for ep in episodes:
            # Apply transforms
            for t in self.transforms:
                ep = t(ep)
                if ep is None:
                    break
            if ep is None:
                continue
            # Apply filters
            for f in self.filters:
                ep = f(ep)
                if ep is None:
                    break
            if ep is not None:
                yield ep

    def fit(self, episodes: Iterator[Episode]) -> "PreprocessingPipeline":
        """Fit all stateful transforms on the provided episodes."""
        # Materialise once for fitting (transforms may need multiple passes)
        buffer = list(episodes)
        for t in self.transforms:
            if hasattr(t, "fit"):
                t.fit(iter(buffer))
        return self


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def standard_pipeline(
    fit_episodes: Iterator[Episode] | None = None,
    action_noise_std: float = 0.0,
    mirror_prob: float = 0.0,
    time_warp_prob: float = 0.0,
    min_steps: int = 10,
    max_steps: int = 10_000,
    action_clip: tuple[float, float] = (-1.0, 1.0),
    seed: int | None = None,
) -> PreprocessingPipeline:
    """Build a standard preprocessing pipeline with optional augmentations."""
    pipe = PreprocessingPipeline()

    # Normalisation (stateful)
    obs_norm = ObservationNormalizer()
    act_norm = ActionNormalizer()
    rew_norm = RewardNormalizer()
    if fit_episodes is not None:
        buf = list(fit_episodes)
        obs_norm.fit(iter(buf))
        act_norm.fit(iter(buf))
        rew_norm.fit(iter(buf))
    pipe.add(obs_norm)
    pipe.add(act_norm)
    pipe.add(rew_norm)
    pipe.add(ActionClipper(low=action_clip[0], high=action_clip[1]))

    # Augmentation (probabilistic)
    if action_noise_std > 0:
        pipe.add(ActionNoise(std=action_noise_std, seed=seed))
    if mirror_prob > 0:
        # Mirror is not a standard Transform (returns Episode not Episode|None)
        # We'll wrap it as a conditional transform
        class _MaybeMirror(Transform):
            def __init__(self, p: float, s: int | None) -> None:
                self.p = p
                self.rng = np.random.default_rng(s)
            def __call__(self, ep: Episode) -> Episode:
                if self.rng.random() < self.p:
                    return MirrorTransform()(ep)
                return ep
        pipe.add(_MaybeMirror(mirror_prob, seed))
    if time_warp_prob > 0:
        class _MaybeWarp(Transform):
            def __init__(self, p: float, s: int | None) -> None:
                self.p = p
                self.rng = np.random.default_rng(s)
            def __call__(self, ep: Episode) -> Episode:
                if self.rng.random() < self.p:
                    return TimeWarp()(ep)
                return ep
        pipe.add(_MaybeWarp(time_warp_prob, seed))

    # Filtering
    pipe.add(LengthFilter(min_steps=min_steps, max_steps=max_steps))
    pipe.add(OutlierFilter(threshold=5.0))
    pipe.add(RewardFilter(drop_all_nan=True))

    return pipe
