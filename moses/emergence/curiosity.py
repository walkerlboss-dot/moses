"""
moses/emergence/curiosity.py
Intrinsic Motivation & Curiosity-Driven Exploration for Moses v4.0

The agent explores not just for external reward but because it is curious
about prediction errors — situations where the world surprises it.
This prevents local optima and discovers diverse behaviors.

Inspired by:
- Pathak et al. (2017) — "Curiosity-driven Exploration by Self-supervised
  Prediction" (ICML). Forward dynamics model + prediction error as intrinsic
  reward. The "ICM" (Intrinsic Curiosity Module).
- Burda et al. (2018) — "Large-Scale Study of Curiosity-Driven Learning"
  (arXiv). Random Network Distillation (RND) as a simpler alternative.
- Ecoffet et al. (2019) — "Go-Explore" (Nature). Remembering and returning
  to promising states.
- Pathak et al. (2019) — "Self-Supervised Exploration via Disagreement"
  (ICML). Ensemble disagreement as exploration bonus.

This module provides both ICM-style prediction-error curiosity and
RND-style novelty bonuses. The user selects which to use.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class CuriosityConfig:
    """Configuration for curiosity-driven exploration."""
    method: str = "icm"               # "icm", "rnd", "disagreement", "hybrid"
    icm_forward_loss_coef: float = 10.0
    icm_inverse_loss_coef: float = 0.8
    icm_feature_dim: int = 256
    rnd_output_dim: int = 512
    ensemble_size: int = 5            # for disagreement method
    curiosity_weight: float = 1.0     # scale of intrinsic reward
    reward_clip: float = 1.0          # clip intrinsic reward magnitude
    episodic_memory_size: int = 10000
    novelty_decay: float = 0.99       # decay for visited-state novelty
    use_episodic_curiosity: bool = True
    use_count_based: bool = False     # hash-based count bonus (Tang et al. 2017)


# ─── Forward Dynamics Model (ICM) ────────────────────────────────────────────

class ForwardDynamicsModel:
    """
    Predicts next state feature given current state feature and action.
    Prediction error = curiosity signal.

    Reference: Pathak et al. (2017) — ICM module.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int = 256,
        learning_rate: float = 1e-3,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.lr = learning_rate

        # Simple MLP for feature encoding: state -> feature
        self._init_encoder()
        # Forward model: (feature, action) -> next_feature
        self._init_forward_model()
        # Inverse model: (feature, next_feature) -> action
        self._init_inverse_model()

    def _init_encoder(self):
        """Initialize encoding network weights."""
        d = self.state_dim
        h = self.feature_dim
        self.enc_W1 = np.random.randn(d, h) * np.sqrt(2.0 / d)
        self.enc_b1 = np.zeros(h)
        self.enc_W2 = np.random.randn(h, h) * np.sqrt(2.0 / h)
        self.enc_b2 = np.zeros(h)

    def _init_forward_model(self):
        """Initialize forward dynamics weights."""
        h = self.feature_dim
        a = self.action_dim
        self.fwd_W1 = np.random.randn(h + a, h) * np.sqrt(2.0 / (h + a))
        self.fwd_b1 = np.zeros(h)
        self.fwd_W2 = np.random.randn(h, h) * np.sqrt(2.0 / h)
        self.fwd_b2 = np.zeros(h)

    def _init_inverse_model(self):
        """Initialize inverse dynamics weights."""
        h = self.feature_dim
        self.inv_W1 = np.random.randn(h * 2, h) * np.sqrt(2.0 / (h * 2))
        self.inv_b1 = np.zeros(h)
        self.inv_W2 = np.random.randn(h, self.action_dim) * np.sqrt(2.0 / h)
        self.inv_b2 = np.zeros(self.action_dim)

    def _encode(self, state: np.ndarray) -> np.ndarray:
        """Encode state to feature space."""
        h = np.maximum(0, state @ self.enc_W1 + self.enc_b1)  # ReLU
        return np.tanh(h @ self.enc_W2 + self.enc_b2)

    def _forward(self, feature: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict next feature."""
        x = np.concatenate([feature, action])
        h = np.maximum(0, x @ self.fwd_W1 + self.fwd_b1)
        return h @ self.fwd_W2 + self.fwd_b2

    def _inverse(self, feat1: np.ndarray, feat2: np.ndarray) -> np.ndarray:
        """Predict action from state transition."""
        x = np.concatenate([feat1, feat2])
        h = np.maximum(0, x @ self.inv_W1 + self.inv_b1)
        return h @ self.inv_W2 + self.inv_b2

    def compute_intrinsic_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute curiosity reward as forward prediction error.
        Also returns losses for training.
        """
        f_t = self._encode(state)
        f_t1 = self._encode(next_state)

        # Forward prediction
        f_t1_pred = self._forward(f_t, action)
        forward_error = np.mean((f_t1_pred - f_t1) ** 2)

        # Inverse prediction
        action_pred = self._inverse(f_t, f_t1)
        inverse_error = np.mean((action_pred - action) ** 2)

        # Curiosity = forward error (what we couldn't predict)
        intrinsic_reward = forward_error

        return intrinsic_reward, {
            "forward_error": forward_error,
            "inverse_error": inverse_error,
        }

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        forward_coef: float = 10.0,
        inverse_coef: float = 0.8,
    ) -> Dict[str, float]:
        """
        One gradient step on forward and inverse models.
        Uses finite differences for simplicity; in production, use PyTorch/TF.
        """
        f_t = self._encode(state)
        f_t1 = self._encode(next_state)

        # Forward model update
        f_t1_pred = self._forward(f_t, action)
        fwd_err = f_t1_pred - f_t1
        fwd_loss = 0.5 * np.mean(fwd_err ** 2)

        # Simple SGD on forward model (simplified — real impl uses autograd)
        grad_scale = self.lr * forward_coef
        self.fwd_W2 -= grad_scale * np.outer(np.maximum(0, np.concatenate([f_t, action]) @ self.fwd_W1 + self.fwd_b1), fwd_err)
        self.fwd_b2 -= grad_scale * fwd_err

        # Inverse model update
        action_pred = self._inverse(f_t, f_t1)
        inv_err = action_pred - action
        inv_loss = 0.5 * np.mean(inv_err ** 2)

        grad_scale_inv = self.lr * inverse_coef
        self.inv_W2 -= grad_scale_inv * np.outer(np.maximum(0, np.concatenate([f_t, f_t1]) @ self.inv_W1 + self.inv_b1), inv_err)
        self.inv_b2 -= grad_scale_inv * inv_err

        return {"forward_loss": fwd_loss, "inverse_loss": inv_loss}


# ─── Random Network Distillation (RND) ───────────────────────────────────────

class RandomNetworkDistillation:
    """
    RND: Fixed random network generates target features.
    Trainable network tries to match them.
    Prediction error = novelty of state.

    Reference: Burda et al. (2018) — "Large-Scale Study of Curiosity-Driven Learning"
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int = 512,
        learning_rate: float = 1e-4,
    ):
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        # Target network: fixed random initialization, never updated
        self.target_W1 = np.random.randn(state_dim, 512) * np.sqrt(2.0 / state_dim)
        self.target_b1 = np.zeros(512)
        self.target_W2 = np.random.randn(512, output_dim) * np.sqrt(2.0 / 512)
        self.target_b2 = np.zeros(output_dim)

        # Predictor network: trained to match target
        self.pred_W1 = np.random.randn(state_dim, 512) * np.sqrt(2.0 / state_dim)
        self.pred_b1 = np.zeros(512)
        self.pred_W2 = np.random.randn(512, output_dim) * np.sqrt(2.0 / 512)
        self.pred_b2 = np.zeros(output_dim)

        # Running mean/std for normalization (critical for RND stability)
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0

    def _target_features(self, state: np.ndarray) -> np.ndarray:
        h = np.maximum(0, state @ self.target_W1 + self.target_b1)
        return h @ self.target_W2 + self.target_b2

    def _predict_features(self, state: np.ndarray) -> np.ndarray:
        h = np.maximum(0, state @ self.pred_W1 + self.pred_b1)
        return h @ self.pred_W2 + self.pred_b2

    def compute_intrinsic_reward(self, state: np.ndarray) -> float:
        """RND bonus = prediction error, normalized by running statistics."""
        target = self._target_features(state)
        pred = self._predict_features(state)
        error = np.mean((pred - target) ** 2)

        # Update running statistics
        self.count += 1
        delta = error - self.running_mean
        self.running_mean += delta / self.count
        delta2 = error - self.running_mean
        self.running_var = ((self.count - 1) * self.running_var + delta * delta2) / self.count

        # Normalize
        normalized_error = error / (np.sqrt(self.running_var) + 1e-8)
        return float(normalized_error)

    def update(self, state: np.ndarray) -> float:
        """Train predictor to better match target on this state."""
        target = self._target_features(state)
        pred = self._predict_features(state)
        err = pred - target
        loss = 0.5 * np.mean(err ** 2)

        # Simplified gradient step
        grad_scale = self.lr
        h = np.maximum(0, state @ self.pred_W1 + self.pred_b1)
        self.pred_W2 -= grad_scale * np.outer(h, err)
        self.pred_b2 -= grad_scale * err

        return loss


# ─── Episodic Curiosity (ECR) ────────────────────────────────────────────────

class EpisodicCuriosity:
    """
    Episodic memory of visited states. Bonus for reaching states
    far from any in current episode's memory.

    Reference: Savinov et al. (2018) — "Episodic Curiosity through Reachability"
    """

    def __init__(
        self,
        state_dim: int,
        memory_size: int = 10000,
        similarity_threshold: float = 0.5,
        kernel: str = "euclidean",
    ):
        self.state_dim = state_dim
        self.memory_size = memory_size
        self.similarity_threshold = similarity_threshold
        self.kernel = kernel

        # Current episode memory
        self.episode_memory: List[np.ndarray] = []
        # Global memory (optional, for across-episode novelty)
        self.global_memory: deque = deque(maxlen=memory_size)

    def _distance(self, s1: np.ndarray, s2: np.ndarray) -> float:
        if self.kernel == "euclidean":
            return np.linalg.norm(s1 - s2)
        elif self.kernel == "cosine":
            return 1.0 - np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2) + 1e-8)
        else:
            return np.linalg.norm(s1 - s2)

    def compute_bonus(self, state: np.ndarray) -> float:
        """
        Compute episodic curiosity bonus.
        High if state is far from all states in episode memory.
        """
        if len(self.episode_memory) == 0:
            self.episode_memory.append(state.copy())
            return 1.0  # First state is novel

        distances = [self._distance(state, m) for m in self.episode_memory]
        min_dist = min(distances)

        # Bonus decreases as we visit similar states
        bonus = 1.0 if min_dist > self.similarity_threshold else 0.0

        # Add to memory if sufficiently novel
        if bonus > 0.5:
            self.episode_memory.append(state.copy())
            if len(self.episode_memory) > self.memory_size:
                self.episode_memory.pop(0)

        return bonus

    def reset_episode(self) -> None:
        """Call at episode end."""
        # Optionally merge into global memory
        for s in self.episode_memory:
            self.global_memory.append(s.copy())
        self.episode_memory = []


# ─── Count-Based Exploration (Hashing) ───────────────────────────────────────

class CountBasedExploration:
    """
    Simple hash-based count bonus (Tang et al. 2017).
    States hashed to discrete bins; bonus = 1/sqrt(count).
    """

    def __init__(self, hash_dim: int = 64):
        self.hash_dim = hash_dim
        self.state_counts: Dict[str, int] = {}
        # Random projection for hashing
        self.projection = np.random.randn(hash_dim)

    def _hash_state(self, state: np.ndarray) -> str:
        """LSH-style hash of state."""
        projected = (state @ self.projection[:len(state)]) > 0
        return "".join("1" if b else "0" for b in projected)

    def compute_bonus(self, state: np.ndarray) -> float:
        h = self._hash_state(state)
        count = self.state_counts.get(h, 0) + 1
        self.state_counts[h] = count
        return 1.0 / np.sqrt(count)


# ─── Curiosity Engine ────────────────────────────────────────────────────────

class CuriosityEngine:
    """
    Unified interface for intrinsic motivation in Moses.

    Combines multiple curiosity signals:
    - ICM: prediction error in learned feature space
    - RND: novelty via random network distillation
    - Episodic: reachability-based episodic memory
    - Count: hash-based count bonus

    The user configures which methods to use. Hybrid mode combines
    ICM/RND with episodic curiosity for robust exploration.

    Parameters
    ----------
    state_dim : int
        Dimensionality of observation vector.
    action_dim : int
        Dimensionality of action vector.
    config : CuriosityConfig
        Which methods to use and their hyperparameters.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[CuriosityConfig] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or CuriosityConfig()

        self.icm: Optional[ForwardDynamicsModel] = None
        self.rnd: Optional[RandomNetworkDistillation] = None
        self.episodic: Optional[EpisodicCuriosity] = None
        self.count_based: Optional[CountBasedExploration] = None

        if self.config.method in ("icm", "hybrid"):
            self.icm = ForwardDynamicsModel(
                state_dim=state_dim,
                action_dim=action_dim,
                feature_dim=self.config.icm_feature_dim,
            )

        if self.config.method in ("rnd", "hybrid"):
            self.rnd = RandomNetworkDistillation(
                state_dim=state_dim,
                output_dim=self.config.rnd_output_dim,
            )

        if self.config.use_episodic_curiosity:
            self.episodic = EpisodicCuriosity(
                state_dim=state_dim,
                memory_size=self.config.episodic_memory_size,
            )

        if self.config.use_count_based:
            self.count_based = CountBasedExploration()

        # Statistics tracking
        self.intrinsic_rewards = deque(maxlen=10000)
        self.extrinsic_rewards = deque(maxlen=10000)

    def compute_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        extrinsic_reward: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward = extrinsic + curiosity_weight * intrinsic.

        Returns (total_reward, info_dict with component breakdown).
        """
        intrinsic = 0.0
        info = {"extrinsic": extrinsic_reward}

        if self.icm is not None:
            icm_reward, icm_info = self.icm.compute_intrinsic_reward(
                state, action, next_state
            )
            intrinsic += icm_reward
            info["icm_reward"] = icm_reward
            info["icm_forward_error"] = icm_info["forward_error"]
            info["icm_inverse_error"] = icm_info["inverse_error"]

        if self.rnd is not None:
            rnd_reward = self.rnd.compute_intrinsic_reward(next_state)
            intrinsic += rnd_reward
            info["rnd_reward"] = rnd_reward

        if self.episodic is not None:
            epi_bonus = self.episodic.compute_bonus(next_state)
            intrinsic += epi_bonus
            info["episodic_bonus"] = epi_bonus

        if self.count_based is not None:
            count_bonus = self.count_based.compute_bonus(next_state)
            intrinsic += count_bonus
            info["count_bonus"] = count_bonus

        # Clip and weight
        intrinsic = np.clip(
            intrinsic * self.config.curiosity_weight,
            -self.config.reward_clip,
            self.config.reward_clip,
        )

        total_reward = extrinsic_reward + intrinsic

        self.intrinsic_rewards.append(intrinsic)
        self.extrinsic_rewards.append(extrinsic_reward)

        info["intrinsic_total"] = intrinsic
        info["total_reward"] = total_reward

        return total_reward, info

    def update_models(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> Dict[str, float]:
        """
        Update curiosity models (ICM, RND) after observing a transition.
        Call this after compute_reward in the training loop.
        """
        losses = {}

        if self.icm is not None:
            icm_losses = self.icm.update(
                state, action, next_state,
                forward_coef=self.config.icm_forward_loss_coef,
                inverse_coef=self.config.icm_inverse_loss_coef,
            )
            losses.update({f"icm_{k}": v for k, v in icm_losses.items()})

        if self.rnd is not None:
            rnd_loss = self.rnd.update(next_state)
            losses["rnd_loss"] = rnd_loss

        return losses

    def reset_episode(self) -> None:
        """Call at the end of each episode."""
        if self.episodic is not None:
            self.episodic.reset_episode()

    def get_statistics(self) -> Dict[str, float]:
        """Return running statistics of intrinsic vs extrinsic rewards."""
        if not self.intrinsic_rewards:
            return {}
        return {
            "intrinsic_mean": float(np.mean(self.intrinsic_rewards)),
            "intrinsic_std": float(np.std(self.intrinsic_rewards)),
            "extrinsic_mean": float(np.mean(self.extrinsic_rewards)),
            "intrinsic_ratio": float(np.mean(self.intrinsic_rewards)) / (
                abs(float(np.mean(self.extrinsic_rewards))) + 1e-8
            ),
        }

    def is_stuck_in_local_optimum(
        self,
        window: int = 100,
        threshold: float = 0.01,
    ) -> bool:
        """
        Detect if extrinsic reward has plateaued while intrinsic is high.
        This suggests the agent is exploring but not finding better policies.
        """
        if len(self.extrinsic_rewards) < window:
            return False
        recent_ext = list(self.extrinsic_rewards)[-window:]
        recent_int = list(self.intrinsic_rewards)[-window:]
        ext_std = np.std(recent_ext)
        int_mean = np.mean(recent_int)
        return ext_std < threshold and int_mean > threshold * 10


# ─── Utility: Adaptive Curiosity Annealing ───────────────────────────────────

def adaptive_curiosity_schedule(
    episode: int,
    total_episodes: int,
    base_weight: float = 1.0,
    min_weight: float = 0.1,
    exploration_fraction: float = 0.5,
) -> float:
    """
    Decay curiosity weight over training.
    Early: explore heavily. Late: exploit known rewards.

    Similar to epsilon-greedy decay but for intrinsic reward weight.
    """
    progress = episode / total_episodes
    if progress < exploration_fraction:
        return base_weight
    else:
        # Linear decay from base_weight to min_weight
        decay_progress = (progress - exploration_fraction) / (1 - exploration_fraction)
        return base_weight - decay_progress * (base_weight - min_weight)
