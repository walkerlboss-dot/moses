#!/usr/bin/env python3
"""
eval_policy.py

Policy evaluation script for the Moses humanoid.

Loads a trained policy checkpoint, runs evaluation episodes,
computes metrics (reward, survival time, energy efficiency),
generates trajectory plots, and exports the policy to ONNX.

Target: Isaac Lab 1.x, PyTorch 2.x, CUDA 12.x
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval_policy")


# ---------------------------------------------------------------------------
# Lazy module loader
# ---------------------------------------------------------------------------
class _LazyModules:
    """Container for lazily imported modules."""
    _loaded = False
    torch: Any = None
    nn: Any = None
    distributions: Any = None

    @classmethod
    def load(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        import torch
        import torch.nn as nn
        import torch.distributions
        cls.torch = torch
        cls.nn = nn
        cls.distributions = torch.distributions


def _build_actor_critic(num_obs: int, num_actions: int, hidden_dims: list[int], activation: str):
    """Build and return the actor-critic model class."""
    _LazyModules.load()
    torch = _LazyModules.torch
    nn = _LazyModules.nn
    act_cls = nn.ELU if activation == "elu" else nn.ReLU

    class HumanoidActorCritic(nn.Module):
        """Actor-Critic network for the humanoid policy."""

        def __init__(self) -> None:
            super().__init__()
            self.num_obs = num_obs
            self.num_actions = num_actions

            actor_layers: list[Any] = []
            prev = num_obs
            for h in hidden_dims:
                actor_layers.extend([nn.Linear(prev, h), act_cls()])
                prev = h
            actor_layers.append(nn.Linear(prev, num_actions))
            self.actor = nn.Sequential(*actor_layers)

            critic_layers: list[Any] = []
            prev = num_obs
            for h in hidden_dims:
                critic_layers.extend([nn.Linear(prev, h), act_cls()])
                prev = h
            critic_layers.append(nn.Linear(prev, 1))
            self.critic = nn.Sequential(*critic_layers)

            self.std = nn.Parameter(torch.ones(num_actions) * 0.5)

        def forward(self) -> None:
            raise NotImplementedError("Use act() and evaluate()")

        def act(self, obs: Any) -> tuple[Any, Any]:
            mean = self.actor(obs)
            std = self.std.expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob

        def evaluate(self, obs: Any, action: Any) -> tuple[Any, Any, Any]:
            mean = self.actor(obs)
            std = self.std.expand_as(mean)
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            value = self.critic(obs).squeeze(-1)
            return value, log_prob, entropy

    return HumanoidActorCritic


# ---------------------------------------------------------------------------
# Dummy env for standalone evaluation
# ---------------------------------------------------------------------------
class DummyHumanoidEnv:
    """Stand-in environment when Isaac Lab is not installed."""

    def __init__(self, num_envs: int, device: str) -> None:
        _LazyModules.load()
        self.num_envs = num_envs
        self.device = device
        self.num_obs = 69
        self.num_actions = 21
        self.episode_length_buf = _LazyModules.torch.zeros(num_envs, device=device, dtype=_LazyModules.torch.long)
        self.max_episode_length = 4000

    def reset(self) -> tuple[Any, dict]:
        obs = _LazyModules.torch.randn(self.num_envs, self.num_obs, device=self.device)
        return obs, {}

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict]:
        obs = _LazyModules.torch.randn(self.num_envs, self.num_obs, device=self.device)
        reward = _LazyModules.torch.randn(self.num_envs, device=self.device)
        terminated = _LazyModules.torch.zeros(self.num_envs, device=self.device, dtype=_LazyModules.torch.bool)
        truncated = _LazyModules.torch.zeros(self.num_envs, device=self.device, dtype=_LazyModules.torch.bool)
        self.episode_length_buf += 1
        mask = _LazyModules.torch.rand(self.num_envs, device=self.device) < 0.001
        terminated |= mask
        truncated |= self.episode_length_buf >= self.max_episode_length
        reset_mask = terminated | truncated
        self.episode_length_buf[reset_mask] = 0
        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class EpisodeMetrics:
    """Collects per-episode metrics."""

    def __init__(self) -> None:
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_energy: list[float] = []
        self.survival_times: list[float] = []

    def add(
        self,
        reward_sum: float,
        length: int,
        energy_sum: float,
        survival_time: float,
    ) -> None:
        self.episode_rewards.append(reward_sum)
        self.episode_lengths.append(length)
        self.episode_energy.append(energy_sum)
        self.survival_times.append(survival_time)

    def summary(self) -> dict[str, float]:
        if not self.episode_rewards:
            return {}
        import numpy as np
        rewards = np.array(self.episode_rewards)
        lengths = np.array(self.episode_lengths, dtype=float)
        energies = np.array(self.episode_energy)
        survivals = np.array(self.survival_times)
        efficiency = np.where(energies > 0, rewards / energies, 0.0)

        return {
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "min_reward": float(rewards.min()),
            "max_reward": float(rewards.max()),
            "mean_length": float(lengths.mean()),
            "mean_survival_time_s": float(survivals.mean()),
            "mean_energy": float(energies.mean()),
            "mean_efficiency": float(efficiency.mean()),
            "total_episodes": len(rewards),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Moses humanoid policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel envs for eval")
    parser.add_argument("--max-episode-steps", type=int, default=4000, help="Max steps per episode")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy (mean only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="Directory for outputs")
    parser.add_argument("--export-onnx", action="store_true", help="Export policy to ONNX")
    parser.add_argument("--onnx-path", type=str, default="", help="ONNX output path (default: auto)")
    parser.add_argument("--plot-trajectories", action="store_true", help="Generate trajectory plots")
    parser.add_argument("--use-wandb", action="store_true", help="Log results to W&B")
    parser.add_argument("--wandb-project", type=str, default="moses-humanoid")
    parser.add_argument("--wandb-run-name", type=str, default="")
    return parser.parse_args()


def load_policy(checkpoint_path: str, device: Any) -> tuple[Any, dict]:
    """Load policy from checkpoint."""
    _LazyModules.load()
    torch = _LazyModules.torch
    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args_dict = checkpoint.get("args", {})

    num_obs = args_dict.get("observation_space", 69)
    num_actions = args_dict.get("action_space", 21)
    hidden_dims = args_dict.get("hidden_dims", [512, 256, 128])
    activation = args_dict.get("activation", "elu")

    ActorCriticCls = _build_actor_critic(num_obs, num_actions, hidden_dims, activation)
    model = ActorCriticCls().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Policy loaded: obs=%d actions=%d", num_obs, num_actions)
    return model, args_dict


def run_evaluation(
    model: Any,
    env: Any,
    num_episodes: int,
    max_steps: int,
    deterministic: bool,
    device: Any,
) -> EpisodeMetrics:
    """Run evaluation and collect metrics."""
    _LazyModules.load()
    torch = _LazyModules.torch
    metrics = EpisodeMetrics()
    num_envs = env.num_envs

    episode_reward = torch.zeros(num_envs, device=device)
    episode_length = torch.zeros(num_envs, device=device, dtype=torch.long)
    episode_energy = torch.zeros(num_envs, device=device)
    episodes_done = 0

    obs, _ = env.reset()

    while episodes_done < num_episodes:
        with torch.no_grad():
            if deterministic:
                action = model.actor(obs)
            else:
                action, _ = model.act(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        energy = (action ** 2).sum(dim=-1)

        episode_reward += reward
        episode_length += 1
        episode_energy += energy

        done = terminated | truncated

        for i in range(num_envs):
            if done[i]:
                metrics.add(
                    reward_sum=episode_reward[i].item(),
                    length=int(episode_length[i].item()),
                    energy_sum=episode_energy[i].item(),
                    survival_time=float(episode_length[i].item()) * 0.02,
                )
                episodes_done += 1
                episode_reward[i] = 0.0
                episode_length[i] = 0
                episode_energy[i] = 0.0
                if episodes_done >= num_episodes:
                    break

        # Hard truncation
        if (episode_length >= max_steps).any():
            trunc_mask = episode_length >= max_steps
            for i in range(num_envs):
                if trunc_mask[i]:
                    metrics.add(
                        reward_sum=episode_reward[i].item(),
                        length=int(episode_length[i].item()),
                        energy_sum=episode_energy[i].item(),
                        survival_time=float(episode_length[i].item()) * 0.02,
                    )
                    episodes_done += 1
                    episode_reward[i] = 0.0
                    episode_length[i] = 0
                    episode_energy[i] = 0.0
                    if episodes_done >= num_episodes:
                        break

        obs = next_obs

    return metrics


def plot_trajectories(metrics: EpisodeMetrics, output_dir: Path) -> None:
    """Generate trajectory summary plots."""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available; skipping plots")
        return

    import numpy as np
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    rewards = np.array(metrics.episode_rewards)
    lengths = np.array(metrics.episode_lengths, dtype=float)
    energies = np.array(metrics.episode_energy)
    efficiencies = np.where(energies > 0, rewards / energies, 0.0)

    axes[0, 0].hist(rewards, bins=30, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Episode Reward Distribution")
    axes[0, 0].set_xlabel("Reward")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(lengths * 0.02, bins=30, color="forestgreen", edgecolor="black")
    axes[0, 1].set_title("Survival Time Distribution")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].hist(energies, bins=30, color="coral", edgecolor="black")
    axes[1, 0].set_title("Energy Consumption Distribution")
    axes[1, 0].set_xlabel("Energy")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(efficiencies, bins=30, color="mediumpurple", edgecolor="black")
    axes[1, 1].set_title("Energy Efficiency Distribution")
    axes[1, 1].set_xlabel("Reward / Energy")
    axes[1, 1].set_ylabel("Count")

    plt.tight_layout()
    plot_path = output_dir / "evaluation_metrics.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("Plots saved: %s", plot_path)


def export_policy_to_onnx(model: Any, num_obs: int, onnx_path: Path, device: Any) -> None:
    """Export the actor network to ONNX format."""
    _LazyModules.load()
    torch = _LazyModules.torch
    model.eval()
    dummy_input = torch.randn(1, num_obs, device=device)
    torch.onnx.export(
        model.actor,
        dummy_input,
        onnx_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=11,
    )
    logger.info("ONNX exported: %s", onnx_path)


def evaluate(args: argparse.Namespace) -> int:
    """Main evaluation entry point."""
    _LazyModules.load()
    torch = _LazyModules.torch

    logger.info("=" * 60)
    logger.info("Moses Humanoid Policy Evaluation")
    logger.info("=" * 60)

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt_args = load_policy(args.checkpoint, device)
    num_obs = ckpt_args.get("observation_space", 69)
    num_actions = ckpt_args.get("action_space", 21)

    env = DummyHumanoidEnv(num_envs=args.num_envs, device=str(device))

    logger.info("Running %d episodes across %d envs...", args.num_episodes, args.num_envs)
    start = time.time()
    metrics = run_evaluation(
        model=model,
        env=env,
        num_episodes=args.num_episodes,
        max_steps=args.max_episode_steps,
        deterministic=args.deterministic,
        device=device,
    )
    elapsed = time.time() - start

    summary = metrics.summary()
    summary["eval_time_s"] = elapsed
    summary["deterministic"] = float(args.deterministic)

    logger.info("-" * 40)
    logger.info("Evaluation Results (%d episodes):", summary.get("total_episodes", 0))
    for key, value in summary.items():
        if isinstance(value, float):
            logger.info("  %-25s : %10.4f", key, value)
        else:
            logger.info("  %-25s : %10s", key, str(value))
    logger.info("-" * 40)

    json_path = output_dir / "eval_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary JSON: %s", json_path)

    if args.plot_trajectories:
        plot_trajectories(metrics, output_dir)

    if args.export_onnx:
        onnx_path = Path(args.onnx_path) if args.onnx_path else output_dir / "policy.onnx"
        export_policy_to_onnx(model, num_obs, onnx_path, device)

    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name)
        wandb.log(summary)
        wandb.finish()

    env.close()
    logger.info("Evaluation complete.")
    return 0


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    try:
        return evaluate(args)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted")
        return 130
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
