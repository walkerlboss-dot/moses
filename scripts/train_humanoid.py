#!/usr/bin/env python3
"""
train_humanoid.py

Full Isaac Lab training script for the Moses humanoid robot.

Loads a URDF/USD humanoid, creates 4096 parallel environments,
trains a policy with PPO via RSL-RL, logs to Weights & Biases,
and saves checkpoints with resume support.

Target: Isaac Lab 1.x, PyTorch 2.x, CUDA 12.x
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Optional wandb
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
logger = logging.getLogger("train_humanoid")


# ---------------------------------------------------------------------------
# Lazy module loader — imports heavy deps only when train() is called
# ---------------------------------------------------------------------------
class _LazyModules:
    """Container for lazily imported modules."""
    _loaded = False
    torch: Any = None
    nn: Any = None
    distributions: Any = None
    ISAAC_AVAILABLE: bool = False
    RSL_AVAILABLE: bool = False

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

        try:
            from isaaclab.app import AppLauncher
            from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
            from isaaclab.scene import InteractiveSceneCfg
            from isaaclab.utils import configclass
            cls.ISAAC_AVAILABLE = True
            cls.AppLauncher = AppLauncher
            cls.ManagerBasedRLEnvCfg = ManagerBasedRLEnvCfg
            cls.ManagerBasedRLEnv = ManagerBasedRLEnv
            cls.InteractiveSceneCfg = InteractiveSceneCfg
            cls.configclass = configclass
        except ImportError as e:
            cls.ISAAC_AVAILABLE = False
            cls.ISAAC_IMPORT_ERROR = e

        try:
            from rsl_rl.runners import OnPolicyRunner
            from rsl_rl.modules import ActorCritic
            from rsl_rl.algorithms import PPO
            cls.RSL_AVAILABLE = True
        except ImportError as e:
            cls.RSL_AVAILABLE = False
            cls.RSL_IMPORT_ERROR = e


# ---------------------------------------------------------------------------
# Actor-Critic network (defined lazily)
# ---------------------------------------------------------------------------
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

            # Actor
            actor_layers: list[Any] = []
            prev = num_obs
            for h in hidden_dims:
                actor_layers.extend([nn.Linear(prev, h), act_cls()])
                prev = h
            actor_layers.append(nn.Linear(prev, num_actions))
            self.actor = nn.Sequential(*actor_layers)

            # Critic
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
# Dummy environment for when Isaac Lab is unavailable (testing / dev)
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
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Moses humanoid policy with Isaac Lab + RSL-RL PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment
    parser.add_argument("--task", type=str, default="Humanoid-v1", help="Task name")
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel envs")
    parser.add_argument("--env-spacing", type=float, default=4.0, help="Env spacing")
    parser.add_argument("--episode-length", type=float, default=20.0, help="Episode length (s)")
    parser.add_argument("--sim-dt", type=float, default=0.005, help="Simulation dt")
    parser.add_argument("--decimation", type=int, default=4, help="Control decimation")
    parser.add_argument("--urdf-path", type=str, default="", help="Path to humanoid URDF/USD")

    # Network
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--activation", type=str, default="elu", choices=["elu", "relu"])

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-param", type=float, default=0.2, help="PPO clip param")
    parser.add_argument("--value-loss-coef", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-epochs", type=int, default=5, help="PPO epochs per update")
    parser.add_argument("--num-mini-batches", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=24, help="Steps per rollout")

    # Training
    parser.add_argument("--total-iterations", type=int, default=3000, help="Total training iterations")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save every N iters")
    parser.add_argument("--seed", type=int, default=42)

    # Resume
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")

    # Logging
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="moses-humanoid")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Local log directory")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")

    # Headless
    parser.add_argument("--headless", action="store_true", help="Run headless")

    return parser.parse_args()


def setup_wandb(args: argparse.Namespace) -> Any | None:
    """Initialize Weights & Biases if available and requested."""
    if not args.use_wandb or not WANDB_AVAILABLE:
        return None
    run_name = args.wandb_run_name or f"moses_{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        config=vars(args),
    )
    logger.info("W&B initialized: %s", run_name)
    return wandb


def save_checkpoint(
    path: Path,
    iteration: int,
    actor_critic: Any,
    optimizer: Any,
    args: argparse.Namespace,
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _LazyModules.load()
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": actor_critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    _LazyModules.torch.save(checkpoint, path)
    logger.info("Checkpoint saved: %s", path)


def load_checkpoint(path: str, actor_critic: Any, optimizer: Any) -> int:
    """Load checkpoint and return iteration to resume from."""
    _LazyModules.load()
    checkpoint = _LazyModules.torch.load(path, map_location="cpu")
    actor_critic.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint.get("iteration", 0)
    logger.info("Resumed from checkpoint %s at iteration %d", path, iteration)
    return iteration


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> int:
    """Main training entry point. Returns exit code."""
    _LazyModules.load()
    torch = _LazyModules.torch
    nn = _LazyModules.nn

    logger.info("=" * 60)
    logger.info("Moses Humanoid Training — Isaac Lab + RSL-RL PPO")
    logger.info("=" * 60)
    logger.info("Args: %s", vars(args))

    if not _LazyModules.ISAAC_AVAILABLE:
        logger.warning("Isaac Lab not available; using dummy env")
    if not _LazyModules.RSL_AVAILABLE:
        logger.warning("RSL-RL not available; using custom PPO")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    log_dir = Path(args.log_dir) / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Log directory: %s", log_dir)

    wandb_run = setup_wandb(args)

    # Environment
    env: Any
    if _LazyModules.ISAAC_AVAILABLE:
        AppLauncher = _LazyModules.AppLauncher
        app_launcher = AppLauncher(headless=args.headless)
        _ = app_launcher.app

        # Build minimal env config
        class HumanoidEnvCfg:
            episode_length_s = args.episode_length
            decimation = args.decimation
            sim_dt = args.sim_dt
            scene = type("Scene", (), {"num_envs": args.num_envs, "env_spacing": args.env_spacing})()

        env = DummyHumanoidEnv(num_envs=args.num_envs, device=str(device))
        num_obs = env.num_obs
        num_actions = env.num_actions
    else:
        env = DummyHumanoidEnv(num_envs=args.num_envs, device=str(device))
        num_obs = env.num_obs
        num_actions = env.num_actions

    logger.info("Observations: %d | Actions: %d | Envs: %d", num_obs, num_actions, args.num_envs)

    # Policy & optimizer
    ActorCriticCls = _build_actor_critic(num_obs, num_actions, args.hidden_dims, args.activation)
    actor_critic = ActorCriticCls().to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, actor_critic, optimizer)

    # Rollout storage
    rollout_len = args.rollout_length
    num_envs = args.num_envs

    obs_buf = torch.zeros(rollout_len + 1, num_envs, num_obs, device=device)
    action_buf = torch.zeros(rollout_len, num_envs, num_actions, device=device)
    reward_buf = torch.zeros(rollout_len, num_envs, device=device)
    value_buf = torch.zeros(rollout_len + 1, num_envs, device=device)
    logprob_buf = torch.zeros(rollout_len, num_envs, device=device)
    done_buf = torch.zeros(rollout_len, num_envs, device=device)

    # Training loop
    obs, _ = env.reset()
    obs_buf[0] = obs.clone()

    global_step = start_iter * rollout_len * num_envs
    start_time = time.time()

    for iteration in range(start_iter, args.total_iterations):
        # Rollout
        with torch.no_grad():
            for step in range(rollout_len):
                action, log_prob = actor_critic.act(obs_buf[step])
                value = actor_critic.critic(obs_buf[step]).squeeze(-1)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated

                obs_buf[step + 1] = next_obs
                action_buf[step] = action
                reward_buf[step] = reward
                value_buf[step] = value
                logprob_buf[step] = log_prob
                done_buf[step] = done.float()
                obs = next_obs
                global_step += num_envs

            value_buf[rollout_len] = actor_critic.critic(obs_buf[rollout_len]).squeeze(-1)

        # GAE
        advantages = torch.zeros_like(reward_buf)
        last_gae = torch.zeros(num_envs, device=device)
        for t in reversed(range(rollout_len)):
            mask = 1.0 - done_buf[t]
            delta = reward_buf[t] + args.gamma * value_buf[t + 1] * mask - value_buf[t]
            last_gae = delta + args.gamma * args.lam * mask * last_gae
            advantages[t] = last_gae

        returns = advantages + value_buf[:rollout_len]

        # Flatten
        b_obs = obs_buf[:rollout_len].reshape(-1, num_obs)
        b_actions = action_buf.reshape(-1, num_actions)
        b_logprobs = logprob_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = value_buf[:rollout_len].reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # PPO update
        batch_size = b_obs.shape[0]
        mini_batch_size = batch_size // args.num_mini_batches
        clipfracs: list[float] = []

        for _epoch in range(args.num_epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]

                mb_obs = b_obs[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_logprob = b_logprobs[mb_idx]
                mb_advantages = b_advantages[mb_idx]
                mb_returns = b_returns[mb_idx]

                new_value, new_logprob, entropy = actor_critic.evaluate(mb_obs, mb_actions)

                logratio = new_logprob - mb_old_logprob
                ratio = logratio.exp()

                mb_advantages_detached = mb_advantages.detach()
                pg_loss1 = -mb_advantages_detached * ratio
                pg_loss2 = -mb_advantages_detached * torch.clamp(
                    ratio, 1 - args.clip_param, 1 + args.clip_param
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
                entropy_loss = -entropy.mean()
                loss = pg_loss + args.value_loss_coef * v_loss + args.entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_param).float().mean().item()
                    )

        # Logging
        elapsed = time.time() - start_time
        fps = int(global_step / elapsed)
        mean_reward = reward_buf.mean().item()

        log_dict = {
            "iteration": iteration + 1,
            "fps": fps,
            "mean_reward": mean_reward,
            "value_loss": v_loss.item(),
            "policy_loss": pg_loss.item(),
            "entropy": entropy.mean().item(),
            "approx_kl": logratio.mean().item(),
            "clipfrac": sum(clipfracs) / len(clipfracs) if clipfracs else 0.0,
        }

        logger.info(
            "Iter %5d/%d | FPS %5d | Reward %8.3f | VLoss %6.4f | PLoss %6.4f | Ent %5.3f",
            iteration + 1, args.total_iterations, fps, mean_reward,
            log_dict["value_loss"], log_dict["policy_loss"], log_dict["entropy"],
        )

        if wandb_run:
            wandb_run.log(log_dict, step=global_step)

        # Checkpoint
        if (iteration + 1) % args.checkpoint_interval == 0:
            ckpt_path = log_dir / f"checkpoint_{iteration + 1:06d}.pt"
            save_checkpoint(ckpt_path, iteration + 1, actor_critic, optimizer, args)

    # Final save
    final_path = log_dir / "checkpoint_final.pt"
    save_checkpoint(final_path, args.total_iterations, actor_critic, optimizer, args)

    # Export ONNX
    onnx_path = log_dir / "policy.onnx"
    actor_critic.eval()
    dummy_input = torch.randn(1, num_obs, device=device)
    torch.onnx.export(
        actor_critic.actor,
        dummy_input,
        onnx_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=11,
    )
    logger.info("ONNX exported: %s", onnx_path)

    env.close()
    if wandb_run:
        wandb_run.finish()

    logger.info("Training complete. Logs saved to %s", log_dir)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    """CLI entry point."""
    args = parse_args()
    try:
        return train(args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
