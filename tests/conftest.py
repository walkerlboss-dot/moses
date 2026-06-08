"""
conftest.py — Shared pytest fixtures for the Moses humanoid test suite.

Provides:
  - Mock Isaac Sim / Isaac Lab when not available
  - Temporary directories for checkpoints and artifacts
  - Sample configuration objects used across tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Isaac Sim / Isaac Lab availability probe
# ---------------------------------------------------------------------------

def _isaac_available() -> bool:
    """Return True if Isaac Lab can be imported."""
    try:
        import isaaclab  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures: temporary directories
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for checkpoint I/O tests."""
    d = tmp_path / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_log_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for log outputs."""
    d = tmp_path / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_design_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for design artifacts."""
    d = tmp_path / "designs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Fixtures: mock Isaac Sim environment
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_isaac_env() -> Generator[Any, None, None]:
    """
    Yield a minimal mock Isaac Sim environment.

    The mock exposes the gymnasium-like API (reset, step, close) and
    reports observation/action space shapes compatible with the humanoid.
    """

    class _MockObsSpace:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
            self.dtype = np.float32

    class _MockActSpace:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape
            self.dtype = np.float32

    class _MockEnv:
        """Minimal stand-in for an Isaac Lab ManagerBasedRLEnv."""

        def __init__(self, num_envs: int = 4) -> None:
            self.num_envs = num_envs
            self.num_obs = 69
            self.num_actions = 21
            self.observation_space = _MockObsSpace((self.num_obs,))
            self.action_space = _MockActSpace((self.num_actions,))
            self._step_count = 0
            self._max_steps = 100

        def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
            if seed is not None:
                np.random.seed(seed)
            obs = np.random.randn(self.num_envs, self.num_obs).astype(np.float32)
            info = {"episode": 0}
            self._step_count = 0
            return obs, info

        def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
            self._step_count += 1
            obs = np.random.randn(self.num_envs, self.num_obs).astype(np.float32)
            reward = np.random.randn(self.num_envs).astype(np.float32)
            terminated = np.zeros(self.num_envs, dtype=bool)
            truncated = np.zeros(self.num_envs, dtype=bool)
            # Randomly terminate ~1% of envs
            mask = np.random.rand(self.num_envs) < 0.01
            terminated |= mask
            truncated |= self._step_count >= self._max_steps
            info = {"episode": self._step_count}
            return obs, reward, terminated, truncated, info

        def close(self) -> None:
            self._step_count = 0

    yield _MockEnv()


# ---------------------------------------------------------------------------
# Fixtures: sample configs
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_train_config() -> Dict[str, Any]:
    """Return a minimal training configuration dict for smoke tests."""
    return {
        "experiment_name": "test_humanoid",
        "seed": 42,
        "device": "cpu",
        "headless": True,
        "env": {
            "name": "MosesHumanoid-v1",
            "num_envs": 4,
            "env_spacing": 4.0,
            "episode_length_s": 5.0,
        },
        "policy": {
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],
            "activation": "relu",
            "init_noise_std": 1.0,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "learning_rate": 3.0e-4,
            "gamma": 0.99,
            "lam": 0.95,
            "max_grad_norm": 1.0,
        },
        "runner": {
            "num_steps_per_env": 24,
            "max_iterations": 10,
            "save_interval": 5,
            "log_interval": 2,
        },
    }


@pytest.fixture
def sample_checkpoint_state() -> Dict[str, Any]:
    """Return a mock checkpoint state dict compatible with train_humanoid.py."""
    return {
        "iteration": 5,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "args": {
            "observation_space": 69,
            "action_space": 21,
            "hidden_dims": [64, 64],
            "activation": "relu",
        },
    }


# ---------------------------------------------------------------------------
# Fixtures: kinematics helpers (domain stubs)
# ---------------------------------------------------------------------------

@pytest.fixture
def seven_dof_arm() -> "SevenDOFArm":
    """Provide a 7-DOF serial manipulator model for kinematics tests."""
    return SevenDOFArm()


# ---------------------------------------------------------------------------
# Domain stubs used by fixtures and tests
# ---------------------------------------------------------------------------

@dataclass
class SevenDOFArm:
    """
    Minimal 7-DOF serial manipulator for unit testing kinematics.

    Uses the DH convention with arbitrary link lengths for testability.
    """

    link_lengths: List[float] = field(default_factory=lambda: [0.15, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08])
    joint_limits: List[tuple[float, float]] = field(
        default_factory=lambda: [
            (-np.pi, np.pi),
            (-np.pi / 2, np.pi / 2),
            (-np.pi, np.pi),
            (-np.pi / 2, np.pi / 2),
            (-np.pi, np.pi),
            (-np.pi / 2, np.pi / 2),
            (-np.pi, np.pi),
        ]
    )

    def fk(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for the end-effector pose.

        Returns a 4x4 homogeneous transform matrix.
        """
        assert joint_angles.shape == (7,), f"Expected 7 joint angles, got {joint_angles.shape}"
        T = np.eye(4)
        for i in range(7):
            theta = joint_angles[i]
            a = self.link_lengths[i]
            # Simplified DH: rotation about z, translation along x
            ct, st = np.cos(theta), np.sin(theta)
            Ti = np.array([
                [ct, -st, 0.0, a * ct],
                [st,  ct, 0.0, a * st],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            T = T @ Ti
        return T

    def ik(self, target_pose: np.ndarray, initial_guess: np.ndarray | None = None) -> np.ndarray:
        """
        Compute inverse kinematics via numerical optimization (Jacobian pseudo-inverse).

        Returns joint angles that place the end-effector near target_pose.
        """
        if initial_guess is None:
            initial_guess = np.zeros(7)
        q = initial_guess.copy()
        target_pos = target_pose[:3, 3]
        for _ in range(100):
            T = self.fk(q)
            error = target_pos - T[:3, 3]
            if np.linalg.norm(error) < 1e-4:
                break
            J = self.jacobian(q)
            # Use only position part of Jacobian (3x7)
            J_pos = J[:3, :]
            dq = np.linalg.pinv(J_pos) @ error
            q += dq * 0.5
            # Clamp to joint limits
            for i in range(7):
                lo, hi = self.joint_limits[i]
                q[i] = np.clip(q[i], lo, hi)
        return q

    def jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute the geometric Jacobian (6x7) at the given configuration.

        Columns are [Jv_i; Jw_i] for each joint i.
        """
        assert joint_angles.shape == (7,)
        J = np.zeros((6, 7))
        T = np.eye(4)
        # Forward pass: compute transforms
        transforms = [T.copy()]
        for i in range(7):
            theta = joint_angles[i]
            a = self.link_lengths[i]
            ct, st = np.cos(theta), np.sin(theta)
            Ti = np.array([
                [ct, -st, 0.0, a * ct],
                [st,  ct, 0.0, a * st],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            T = T @ Ti
            transforms.append(T.copy())

        # End-effector position
        p_ee = transforms[-1][:3, 3]
        z_axis = np.array([0.0, 0.0, 1.0])

        for i in range(7):
            p_i = transforms[i][:3, 3]
            z_i = transforms[i][:3, :3] @ z_axis
            J[:3, i] = np.cross(z_i, p_ee - p_i)
            J[3:, i] = z_i
        return J

    def is_singular(self, joint_angles: np.ndarray, threshold: float = 1e-3) -> bool:
        """
        Detect kinematic singularity by checking the smallest singular value
        of the position Jacobian.
        """
        J = self.jacobian(joint_angles)
        J_pos = J[:3, :]
        s = np.linalg.svd(J_pos, compute_uv=False)
        return float(s[-1]) < threshold


# ---------------------------------------------------------------------------
# Fixtures: reward helpers (domain stubs)
# ---------------------------------------------------------------------------

@pytest.fixture
def reward_functions() -> "RewardFunctions":
    """Provide a set of reward functions for unit testing."""
    return RewardFunctions()


@dataclass
class RewardFunctions:
    """Reward function suite for the humanoid locomotion task."""

    def velocity_tracking(self, desired_vel: np.ndarray, actual_vel: np.ndarray, sigma: float = 0.5) -> float:
        """
        Gaussian tracking reward for matching desired velocity commands.

        Args:
            desired_vel: [vx, vy, yaw_rate] target velocities.
            actual_vel: [vx, vy, yaw_rate] actual velocities.
            sigma: bandwidth of the Gaussian kernel.

        Returns:
            Scalar reward in [0, 1].
        """
        error = np.linalg.norm(desired_vel - actual_vel)
        return float(np.exp(-error ** 2 / (2 * sigma ** 2)))

    def energy_penalty(self, torques: np.ndarray, joint_velocities: np.ndarray) -> float:
        """
        Penalize mechanical power consumption.

        Args:
            torques: joint torques.
            joint_velocities: joint angular velocities.

        Returns:
            Negative scalar penalty (more negative for higher power).
        """
        power = np.abs(torques * joint_velocities).sum()
        return -0.01 * float(power)

    def stability_bonus(self, base_z: float, target_z: float = 0.85, tolerance: float = 0.05) -> float:
        """
        Bonus for maintaining nominal torso height.

        Args:
            base_z: current base link z-coordinate.
            target_z: desired standing height.
            tolerance: acceptable deviation.

        Returns:
            Positive scalar bonus if within tolerance, zero otherwise.
        """
        if abs(base_z - target_z) <= tolerance:
            return 1.0
        return 0.0

    def termination_conditions(
        self,
        base_z: float,
        base_orientation: np.ndarray,
        max_tilt_rad: float = np.pi / 3,
        min_height: float = 0.3,
    ) -> tuple[bool, str]:
        """
        Determine whether the episode should terminate.

        Args:
            base_z: current base height.
            base_orientation: roll/pitch/yaw or quaternion.
            max_tilt_rad: maximum allowable base tilt.
            min_height: minimum allowable base height.

        Returns:
            (terminated, reason) tuple.
        """
        if base_z < min_height:
            return True, "fallen"
        # Interpret first two elements as roll/pitch if array-like
        if base_orientation.size >= 2:
            tilt = np.linalg.norm(base_orientation[:2])
            if tilt > max_tilt_rad:
                return True, "excessive_tilt"
        return False, ""
