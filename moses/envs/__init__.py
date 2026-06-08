"""
Moses Environment Registration

Registers all Moses environments with Gymnasium so they can be created
via `gymnasium.make("Moses-Humanoid-v0")`.

Author: Moses Team
Version: 3.0.0
"""

import gymnasium as gym
from gymnasium.envs.registration import register

# ------------------------------------------------------------------------------
# Register Moses Humanoid Environment
# ------------------------------------------------------------------------------
register(
    id="Moses-Humanoid-v0",
    entry_point="moses.envs.humanoid_env:MosesHumanoidEnv",
    kwargs={
        "cfg_entry_point": "moses.envs.humanoid_env:MosesHumanoidEnvCfg",
    },
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

# ------------------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------------------
def make_moses_humanoid(
    num_envs: int = 4096,
    device: str = "cuda:0",
    headless: bool = True,
    **kwargs,
) -> gym.Env:
    """
    Create a vectorized Moses Humanoid environment.

    Args:
        num_envs: Number of parallel environments.
        device: Torch device ("cuda:0" or "cpu").
        headless: Run without GUI.
        **kwargs: Additional arguments passed to the environment config.

    Returns:
        A Gymnasium-compatible vectorized environment.
    """
    env = gym.make(
        "Moses-Humanoid-v0",
        num_envs=num_envs,
        device=device,
        headless=headless,
        **kwargs,
    )
    return env


__all__ = ["make_moses_humanoid"]
