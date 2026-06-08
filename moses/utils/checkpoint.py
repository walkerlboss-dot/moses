"""
Checkpoint Manager for Moses Training

Handles saving and loading of:
  - Policy network state dict
  - Optimizer state dict
  - Environment state (rng, step counters, domain randomization params)
  - Training metadata (iteration, reward history, config hash)

Compatible with RSL-RL, SKRL, RL-Games, and custom trainers.

Author: Moses Team
Version: 3.0.0
"""

from __future__ import annotations

import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Optimizer

# Isaac Lab
from isaaclab.envs import ManagerBasedEnv, DirectRLEnv


# ------------------------------------------------------------------------------
# Protocols for policy / trainer interoperability
# ------------------------------------------------------------------------------

@runtime_checkable
class PolicyProtocol(Protocol):
    """Minimal interface for policies that can be checkpointed."""

    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


@runtime_checkable
class TrainerProtocol(Protocol):
    """Minimal interface for trainers that expose iteration count."""

    current_iteration: int


# ------------------------------------------------------------------------------
# Checkpoint dataclass
# ------------------------------------------------------------------------------

@dataclass
class CheckpointMetadata:
    """Metadata stored alongside every checkpoint."""

    iteration: int
    timestamp: str
    config_hash: str
    reward_mean: float | None = None
    reward_std: float | None = None
    episode_length_mean: float | None = None
    env_seed: int | None = None
    git_commit: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Checkpoint:
    """Full checkpoint payload."""

    policy_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None
    env_state: dict[str, Any]
    metadata: CheckpointMetadata


# ------------------------------------------------------------------------------
# Checkpoint Manager
# ------------------------------------------------------------------------------

class CheckpointManager:
    """
    Save / load training checkpoints with automatic versioning and cleanup.

    Directory structure::

        checkpoint_dir/
        ├── checkpoint_latest.pt
        ├── checkpoint_best.pt
        ├── checkpoint_iter_00001000.pt
        ├── checkpoint_iter_00002000.pt
        └── metadata.json

    Args:
        checkpoint_dir: Root directory for checkpoint files.
        max_checkpoints: Maximum number of iteration checkpoints to retain.
        keep_every_n: Retain every Nth checkpoint regardless of max_checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 10,
        keep_every_n: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.keep_every_n = keep_every_n

        self._metadata_path = self.checkpoint_dir / "metadata.json"
        self._checkpoint_history: list[Path] = []
        self._best_reward: float = -float("inf")

        self._load_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        policy: PolicyProtocol,
        optimizer: Optimizer | None,
        env: ManagerBasedEnv | DirectRLEnv | None,
        iteration: int,
        reward_mean: float | None = None,
        reward_std: float | None = None,
        episode_length_mean: float | None = None,
        config_dict: dict[str, Any] | None = None,
        notes: str = "",
        is_best: bool = False,
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            policy: The policy network (must implement state_dict).
            optimizer: The optimizer (optional).
            env: The environment (optional, for RNG state).
            iteration: Current training iteration.
            reward_mean: Mean episodic reward (for metadata).
            reward_std: Std episodic reward.
            episode_length_mean: Mean episode length.
            config_dict: Training configuration dict (hashed for provenance).
            notes: Free-form notes.
            is_best: If True, also save as checkpoint_best.pt.

        Returns:
            Path to the saved checkpoint file.
        """
        # Build checkpoint payload
        policy_state = policy.state_dict()
        optimizer_state = optimizer.state_dict() if optimizer is not None else None
        env_state = self._capture_env_state(env) if env is not None else {}

        config_hash = self._hash_config(config_dict) if config_dict else ""
        metadata = CheckpointMetadata(
            iteration=iteration,
            timestamp=datetime.utcnow().isoformat() + "Z",
            config_hash=config_hash,
            reward_mean=reward_mean,
            reward_std=reward_std,
            episode_length_mean=episode_length_mean,
            notes=notes,
        )

        checkpoint = Checkpoint(
            policy_state=policy_state,
            optimizer_state=optimizer_state,
            env_state=env_state,
            metadata=metadata,
        )

        # Determine filename
        iter_str = f"{iteration:010d}"
        filename = f"checkpoint_iter_{iter_str}.pt"
        filepath = self.checkpoint_dir / filename

        # Save
        torch.save(checkpoint, filepath)

        # Always update latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        shutil.copy2(filepath, latest_path)

        # Save best if warranted
        if is_best or (reward_mean is not None and reward_mean > self._best_reward):
            self._best_reward = reward_mean if reward_mean is not None else self._best_reward
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            shutil.copy2(filepath, best_path)

        # Update history
        self._checkpoint_history.append(filepath)
        self._write_metadata()
        self._cleanup_old_checkpoints()

        return filepath

    def load(
        self,
        policy: PolicyProtocol,
        optimizer: Optimizer | None = None,
        env: ManagerBasedEnv | DirectRLEnv | None = None,
        checkpoint_path: str | Path | None = None,
        load_optimizer: bool = True,
        load_env_state: bool = True,
        strict: bool = True,
    ) -> CheckpointMetadata:
        """
        Load a checkpoint into policy, optimizer, and environment.

        Args:
            policy: Policy to load state into.
            optimizer: Optimizer to load state into (optional).
            env: Environment to restore RNG state (optional).
            checkpoint_path: Path to checkpoint. If None, loads latest.
            load_optimizer: Whether to restore optimizer state.
            load_env_state: Whether to restore environment RNG state.
            strict: Passed to policy.load_state_dict().

        Returns:
            CheckpointMetadata from the loaded checkpoint.
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "checkpoint_latest.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint: Checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load policy
        policy.load_state_dict(checkpoint.policy_state, strict=strict)

        # Load optimizer
        if load_optimizer and optimizer is not None and checkpoint.optimizer_state is not None:
            optimizer.load_state_dict(checkpoint.optimizer_state)

        # Load env state
        if load_env_state and env is not None and checkpoint.env_state:
            self._restore_env_state(env, checkpoint.env_state)

        return checkpoint.metadata

    def load_latest(
        self,
        policy: PolicyProtocol,
        optimizer: Optimizer | None = None,
        env: ManagerBasedEnv | DirectRLEnv | None = None,
        **kwargs,
    ) -> CheckpointMetadata:
        """Convenience: load the most recent checkpoint."""
        return self.load(policy, optimizer, env, checkpoint_path=None, **kwargs)

    def load_best(
        self,
        policy: PolicyProtocol,
        optimizer: Optimizer | None = None,
        env: ManagerBasedEnv | DirectRLEnv | None = None,
        **kwargs,
    ) -> CheckpointMetadata:
        """Convenience: load the best checkpoint."""
        best_path = self.checkpoint_dir / "checkpoint_best.pt"
        return self.load(policy, optimizer, env, checkpoint_path=best_path, **kwargs)

    def resume_training(
        self,
        policy: PolicyProtocol,
        optimizer: Optimizer,
        env: ManagerBasedEnv | DirectRLEnv | None = None,
        trainer: TrainerProtocol | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> int:
        """
        Full resume: load checkpoint and return the iteration to resume from.

        Args:
            policy: Policy network.
            optimizer: Optimizer.
            env: Environment.
            trainer: Trainer object with `current_iteration` attribute.
            checkpoint_path: Specific checkpoint, or latest if None.

        Returns:
            The iteration number to resume from (metadata.iteration + 1).
        """
        metadata = self.load(
            policy=policy,
            optimizer=optimizer,
            env=env,
            checkpoint_path=checkpoint_path,
            load_optimizer=True,
            load_env_state=True,
        )

        if trainer is not None:
            trainer.current_iteration = metadata.iteration

        return metadata.iteration + 1

    def list_checkpoints(self) -> list[Path]:
        """Return sorted list of all iteration checkpoints."""
        return sorted(
            self.checkpoint_dir.glob("checkpoint_iter_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _capture_env_state(self, env: ManagerBasedEnv | DirectRLEnv) -> dict[str, Any]:
        """Serialize environment RNG and step counters."""
        state: dict[str, Any] = {}

        # Torch RNG
        state["torch_rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()

        # Numpy RNG (if accessible)
        try:
            import numpy as np
            state["numpy_rng_state"] = np.random.get_state()
        except Exception:
            pass

        # Isaac Sim RNG (if available)
        try:
            state["sim_seed"] = env.cfg.scene.seed if hasattr(env.cfg.scene, "seed") else None
        except Exception:
            pass

        # Episode counters
        if hasattr(env, "_episode_step_count"):
            state["episode_step_count"] = env._episode_step_count.cpu().tolist()

        # Command buffers
        if hasattr(env, "_commands"):
            state["commands"] = env._commands.cpu().tolist()

        return state

    def _restore_env_state(
        self,
        env: ManagerBasedEnv | DirectRLEnv,
        state: dict[str, Any],
    ) -> None:
        """Deserialize environment RNG and step counters."""
        # Torch RNG
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if "cuda_rng_state" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])

        # Numpy RNG
        if "numpy_rng_state" in state:
            import numpy as np
            np.random.set_state(state["numpy_rng_state"])

        # Episode counters
        if "episode_step_count" in state and hasattr(env, "_episode_step_count"):
            env._episode_step_count = torch.tensor(
                state["episode_step_count"],
                dtype=torch.int32,
                device=env.device,
            )

        # Command buffers
        if "commands" in state and hasattr(env, "_commands"):
            env._commands = torch.tensor(
                state["commands"],
                dtype=torch.float32,
                device=env.device,
            )

    def _hash_config(self, config_dict: dict[str, Any]) -> str:
        """Create a deterministic hash of the config dict."""
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints, keeping every Nth."""
        if len(self._checkpoint_history) <= self.max_checkpoints:
            return

        to_remove = []
        for idx, path in enumerate(self._checkpoint_history):
            # Keep every Nth
            if (idx + 1) % self.keep_every_n == 0:
                continue
            # Keep the most recent max_checkpoints
            if idx >= len(self._checkpoint_history) - self.max_checkpoints:
                continue
            to_remove.append(path)

        for path in to_remove:
            if path.exists():
                path.unlink()
            self._checkpoint_history.remove(path)

    def _write_metadata(self) -> None:
        """Persist checkpoint history to JSON."""
        data = {
            "checkpoint_history": [str(p.name) for p in self._checkpoint_history],
            "best_reward": self._best_reward,
            "last_updated": datetime.utcnow().isoformat() + "Z",
        }
        with open(self._metadata_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_history(self) -> None:
        """Restore checkpoint history from JSON if present."""
        if not self._metadata_path.exists():
            return
        with open(self._metadata_path, "r") as f:
            data = json.load(f)
        self._checkpoint_history = [
            self.checkpoint_dir / name for name in data.get("checkpoint_history", [])
        ]
        self._best_reward = data.get("best_reward", -float("inf"))


# ------------------------------------------------------------------------------
# Integration helpers for common RL frameworks
# ------------------------------------------------------------------------------

class RSLRLCheckpointAdapter:
    """
    Adapter to integrate CheckpointManager with RSL-RL runners.

    Usage in RSL-RL runner::

        ckpt_mgr = RSLRLCheckpointAdapter("./checkpoints", policy, env_cfg)
        ckpt_mgr.save(algo.actor_critic, algo.optimizer, env, it)
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        env_cfg: dict[str, Any] | None = None,
        max_checkpoints: int = 10,
    ):
        self.manager = CheckpointManager(checkpoint_dir, max_checkpoints)
        self.env_cfg = env_cfg or {}

    def save(
        self,
        actor_critic: nn.Module,
        optimizer: Optimizer,
        env: ManagerBasedEnv | DirectRLEnv,
        iteration: int,
        reward_mean: float | None = None,
    ) -> Path:
        """Save RSL-RL checkpoint."""
        return self.manager.save(
            policy=actor_critic,
            optimizer=optimizer,
            env=env,
            iteration=iteration,
            reward_mean=reward_mean,
            config_dict=self.env_cfg,
        )

    def load(
        self,
        actor_critic: nn.Module,
        optimizer: Optimizer,
        env: ManagerBasedEnv | DirectRLEnv | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> int:
        """Load and return iteration to resume from."""
        return self.manager.resume_training(
            policy=actor_critic,
            optimizer=optimizer,
            env=env,
            checkpoint_path=checkpoint_path,
        )


class SKRLCheckpointAdapter:
    """
    Adapter to integrate CheckpointManager with SKRL trainers.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        env_cfg: dict[str, Any] | None = None,
        max_checkpoints: int = 10,
    ):
        self.manager = CheckpointManager(checkpoint_dir, max_checkpoints)
        self.env_cfg = env_cfg or {}

    def save(
        self,
        agent: Any,  # skrl.agents.torch.Agent
        env: ManagerBasedEnv | DirectRLEnv,
        iteration: int,
        reward_mean: float | None = None,
    ) -> Path:
        """Save SKRL checkpoint."""
        policy = agent.policy if hasattr(agent, "policy") else agent
        optimizer = agent.optimizer if hasattr(agent, "optimizer") else None
        return self.manager.save(
            policy=policy,
            optimizer=optimizer,
            env=env,
            iteration=iteration,
            reward_mean=reward_mean,
            config_dict=self.env_cfg,
        )

    def load(
        self,
        agent: Any,
        env: ManagerBasedEnv | DirectRLEnv | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> int:
        """Load and return iteration to resume from."""
        policy = agent.policy if hasattr(agent, "policy") else agent
        optimizer = agent.optimizer if hasattr(agent, "optimizer") else None
        return self.manager.resume_training(
            policy=policy,
            optimizer=optimizer,
            env=env,
            checkpoint_path=checkpoint_path,
        )
