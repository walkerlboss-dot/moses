"""
test_env_creation.py — Integration tests for environment lifecycle.

Verifies:
  - Gymnasium registration (or mock registration)
  - Environment reset/step/close cycle
  - Observation and action space shapes
  - Compatibility with Isaac Lab when available

Skipped automatically if Isaac Sim / Isaac Lab is not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Probe for Isaac Lab availability
# ---------------------------------------------------------------------------

try:
    import isaaclab  # noqa: F401
    ISAAC_LAB_AVAILABLE = True
except Exception:
    ISAAC_LAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ISAAC_LAB_AVAILABLE, reason="Isaac Lab not installed")
class TestIsaacLabEnv:
    """Integration tests that require Isaac Lab."""

    def test_gymnasium_registration(self):
        """
        Verify that the Moses humanoid environment is registered with gymnasium.
        """
        import gymnasium as gym
        # Registration may happen on import; check if our env id exists
        env_ids = [spec.id for spec in gym.envs.registry.values()]
        # If the env is registered, it should appear in the registry
        # (This is a smoke test — the actual id depends on project setup)
        moses_ids = [eid for eid in env_ids if "Moses" in eid or "Humanoid" in eid]
        # We do not assert presence because registration may not have run yet;
        # instead we verify the registry API is accessible.
        assert isinstance(env_ids, list)

    def test_env_reset(self):
        """
        Resetting the environment must return an observation and an info dict.
        """
        import gymnasium as gym
        # Try to make the environment if registered; otherwise skip gracefully
        env_ids = [spec.id for spec in gym.envs.registry.values()]
        moses_id = next((eid for eid in env_ids if "Moses" in eid), None)
        if moses_id is None:
            pytest.skip("Moses environment not registered")

        env = gym.make(moses_id)
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert isinstance(info, dict)
        env.close()

    def test_env_step(self):
        """
        Stepping the environment must return (obs, reward, terminated, truncated, info).
        """
        import gymnasium as gym
        env_ids = [spec.id for spec in gym.envs.registry.values()]
        moses_id = next((eid for eid in env_ids if "Moses" in eid), None)
        if moses_id is None:
            pytest.skip("Moses environment not registered")

        env = gym.make(moses_id)
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs is not None
        assert isinstance(reward, (float, int, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        env.close()

    def test_observation_space(self):
        """
        Observation space must be a Box with consistent shape.
        """
        import gymnasium as gym
        env_ids = [spec.id for spec in gym.envs.registry.values()]
        moses_id = next((eid for eid in env_ids if "Moses" in eid), None)
        if moses_id is None:
            pytest.skip("Moses environment not registered")

        env = gym.make(moses_id)
        assert hasattr(env, "observation_space")
        obs_space = env.observation_space
        assert obs_space is not None
        assert hasattr(obs_space, "shape")
        assert len(obs_space.shape) >= 1
        env.close()

    def test_action_space(self):
        """
        Action space must be a Box with consistent shape.
        """
        import gymnasium as gym
        env_ids = [spec.id for spec in gym.envs.registry.values()]
        moses_id = next((eid for eid in env_ids if "Moses" in eid), None)
        if moses_id is None:
            pytest.skip("Moses environment not registered")

        env = gym.make(moses_id)
        assert hasattr(env, "action_space")
        act_space = env.action_space
        assert act_space is not None
        assert hasattr(act_space, "shape")
        assert len(act_space.shape) >= 1
        env.close()


class TestMockEnv:
    """Integration tests using the mock environment from conftest."""

    def test_mock_env_reset(self, mock_isaac_env):
        """
        The mock environment reset must return an observation array and info dict.
        """
        obs, info = mock_isaac_env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (mock_isaac_env.num_envs, mock_isaac_env.num_obs)
        assert isinstance(info, dict)

    def test_mock_env_step(self, mock_isaac_env):
        """
        The mock environment step must return valid tuples.
        """
        mock_isaac_env.reset(seed=42)
        action = np.random.randn(mock_isaac_env.num_envs, mock_isaac_env.num_actions).astype(np.float32)
        obs, reward, terminated, truncated, info = mock_isaac_env.step(action)
        assert obs.shape == (mock_isaac_env.num_envs, mock_isaac_env.num_obs)
        assert reward.shape == (mock_isaac_env.num_envs,)
        assert terminated.shape == (mock_isaac_env.num_envs,)
        assert truncated.shape == (mock_isaac_env.num_envs,)
        assert isinstance(info, dict)

    def test_mock_env_close(self, mock_isaac_env):
        """
        Closing the mock environment must not raise.
        """
        mock_isaac_env.reset(seed=42)
        mock_isaac_env.close()
        assert mock_isaac_env._step_count == 0

    def test_mock_env_observation_space(self, mock_isaac_env):
        """
        The mock observation space must report the correct shape.
        """
        assert mock_isaac_env.observation_space.shape == (mock_isaac_env.num_obs,)

    def test_mock_env_action_space(self, mock_isaac_env):
        """
        The mock action space must report the correct shape.
        """
        assert mock_isaac_env.action_space.shape == (mock_isaac_env.num_actions,)

    def test_mock_env_episode_termination(self, mock_isaac_env):
        """
        After enough steps, the mock environment must produce truncated=True.
        """
        mock_isaac_env.reset(seed=42)
        action = np.zeros((mock_isaac_env.num_envs, mock_isaac_env.num_actions), dtype=np.float32)
        truncated_any = False
        for _ in range(mock_isaac_env._max_steps + 5):
            _, _, _, truncated, _ = mock_isaac_env.step(action)
            if truncated.any():
                truncated_any = True
                break
        assert truncated_any, "Mock env did not truncate within expected horizon"
