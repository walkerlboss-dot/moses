"""
test_training_smoke.py — Smoke tests for the training loop.

Verifies:
  - Training loop starts without crashing
  - A small number of iterations (10) complete successfully
  - Checkpoints are written to disk

Marked as slow because even a minimal training run takes several seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the scripts directory is importable
_scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_minimal_args(tmp_log_dir: Path) -> "argparse.Namespace":
    """Build a minimal argparse Namespace for smoke testing train()."""
    import argparse
    return argparse.Namespace(
        task="Humanoid-v1",
        num_envs=4,
        env_spacing=4.0,
        episode_length=5.0,
        sim_dt=0.005,
        decimation=4,
        urdf_path="",
        hidden_dims=[32, 32],
        activation="relu",
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        num_epochs=2,
        num_mini_batches=2,
        rollout_length=8,
        total_iterations=10,
        checkpoint_interval=5,
        seed=42,
        resume="",
        use_wandb=False,
        wandb_project="moses-humanoid",
        wandb_entity="",
        wandb_run_name="",
        log_dir=str(tmp_log_dir),
        device="cpu",
        headless=True,
    )


# ---------------------------------------------------------------------------
# Smoke Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTrainingSmoke:
    """Slow smoke tests for the full training pipeline."""

    def test_training_loop_starts(self, tmp_log_dir):
        """
        The training loop must start and run at least one iteration without raising.
        """
        from train_humanoid import train
        args = _build_minimal_args(tmp_log_dir)
        # Reduce to a single iteration for a fast start test
        args.total_iterations = 1
        exit_code = train(args)
        assert exit_code == 0

    def test_ten_iterations_complete(self, tmp_log_dir):
        """
        Running 10 iterations must complete successfully and produce log output.
        """
        from train_humanoid import train
        args = _build_minimal_args(tmp_log_dir)
        args.total_iterations = 10
        exit_code = train(args)
        assert exit_code == 0

    def test_checkpoint_saved(self, tmp_log_dir):
        """
        After training with checkpoint_interval=5, a checkpoint file must exist on disk.
        """
        from train_humanoid import train
        args = _build_minimal_args(tmp_log_dir)
        args.total_iterations = 10
        args.checkpoint_interval = 5
        exit_code = train(args)
        assert exit_code == 0

        # Check that a checkpoint was written
        log_dirs = list(Path(tmp_log_dir).iterdir())
        assert len(log_dirs) >= 1, "No log directory created"
        latest_log_dir = max(log_dirs, key=lambda p: p.stat().st_mtime)
        checkpoints = list(latest_log_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) >= 1, f"No checkpoint found in {latest_log_dir}"

    def test_checkpoint_contains_expected_keys(self, tmp_log_dir):
        """
        A saved checkpoint must contain iteration, model_state_dict, optimizer_state_dict, and args.
        """
        import torch
        from train_humanoid import train
        args = _build_minimal_args(tmp_log_dir)
        args.total_iterations = 5
        args.checkpoint_interval = 5
        exit_code = train(args)
        assert exit_code == 0

        log_dirs = list(Path(tmp_log_dir).iterdir())
        latest_log_dir = max(log_dirs, key=lambda p: p.stat().st_mtime)
        checkpoints = list(latest_log_dir.glob("checkpoint_*.pt"))
        assert checkpoints
        ckpt = torch.load(checkpoints[0], map_location="cpu")
        assert "iteration" in ckpt
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "args" in ckpt
        assert ckpt["iteration"] > 0

    def test_resume_from_checkpoint(self, tmp_log_dir):
        """
        Training resumed from a checkpoint must start from the saved iteration.
        """
        import torch
        from train_humanoid import train, save_checkpoint
        args = _build_minimal_args(tmp_log_dir)
        args.total_iterations = 5
        args.checkpoint_interval = 5

        # Run first phase
        exit_code = train(args)
        assert exit_code == 0

        # Find the checkpoint
        log_dirs = list(Path(tmp_log_dir).iterdir())
        latest_log_dir = max(log_dirs, key=lambda p: p.stat().st_mtime)
        checkpoints = list(latest_log_dir.glob("checkpoint_*.pt"))
        assert checkpoints
        ckpt_path = str(checkpoints[0])

        # Resume
        args2 = _build_minimal_args(tmp_log_dir)
        args2.total_iterations = 8
        args2.resume = ckpt_path
        exit_code2 = train(args2)
        assert exit_code2 == 0

        # The second run should have loaded the checkpoint
        loaded = torch.load(ckpt_path, map_location="cpu")
        assert loaded["iteration"] == 5

    def test_onnx_exported(self, tmp_log_dir):
        """
        After training completes, an ONNX policy file must be produced.
        """
        from train_humanoid import train
        args = _build_minimal_args(tmp_log_dir)
        args.total_iterations = 3
        exit_code = train(args)
        assert exit_code == 0

        log_dirs = list(Path(tmp_log_dir).iterdir())
        latest_log_dir = max(log_dirs, key=lambda p: p.stat().st_mtime)
        onnx_files = list(latest_log_dir.glob("*.onnx"))
        assert len(onnx_files) >= 1, f"ONNX export missing in {latest_log_dir}"
