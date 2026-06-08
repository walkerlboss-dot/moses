"""GR00T Fine-tuning Pipeline for Moses v4.0.

This module provides a high-level wrapper around NVIDIA Isaac GR00T N1.7's
fine-tuning infrastructure, adapted for Moses-specific tasks (walking,
manipulation, whole-body control).

It handles:

- Loading a GR00T base checkpoint.
- Preparing Moses Isaac Lab rollouts into GR00T LeRobot v2 format.
- Launching fine-tuning with Moses-optimized hyperparameters.
- Saving fine-tuned checkpoints.
- Evaluating the fine-tuned model against a baseline (base model or previous
  checkpoint) using open-loop action prediction metrics (MSE, MAE).

Dependencies
------------
- ``gr00t`` (NVIDIA Isaac-GR00T package)
- ``lerobot`` (for dataset reading/writing)
- ``torch``, ``numpy``, ``tyro``, ``wandb`` (optional)

Example
-------
>>> from moses.gr00t.finetune import Gr00TFineTuner
>>> from moses.gr00t.embodiment import MOSES_H2_SHARPA_CONFIG
>>> tuner = Gr00TFineTuner(
...     base_model="nvidia/GR00T-N1.7-3B",
...     dataset_path="./data/moses_rollouts",
...     output_dir="./checkpoints/moses_gr00t",
...     embodiment_config_path="moses/gr00t/embodiment.py",
... )
>>> tuner.finetune(max_steps=5000)
>>> metrics = tuner.evaluate(baseline_model="nvidia/GR00T-N1.7-3B")
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gr00t.experiment.launch_finetune import load_modality_config
from gr00t.eval.open_loop_eval import evaluate_single_trajectory
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.policy.gr00t_policy import Gr00tPolicy

logger = logging.getLogger(__name__)


@dataclass
class MosesFinetuneConfig:
    """Configuration for the Moses GR00T fine-tuning pipeline.

    These parameters map closely to GR00T's :class:`FinetuneConfig` but are
    curated for Moses tasks (humanoid locomotion + manipulation).
    """

    # Paths
    base_model_path: str = "nvidia/GR00T-N1.7-3B"
    """Path to the pretrained base model checkpoint or HuggingFace model ID."""

    dataset_path: str = "./data/moses_rollouts"
    """Path to the LeRobot v2 dataset generated from Moses rollouts."""

    output_dir: str = "./checkpoints/moses_gr00t"
    """Directory where fine-tuned checkpoints and logs are saved."""

    embodiment_tag: str = "NEW_EMBODIMENT"
    """Embodiment tag. Use NEW_EMBODIMENT for Moses H2+Sharpa custom config."""

    modality_config_path: str = "moses/gr00t/embodiment.py"
    """Path to the Python file that registers the Moses embodiment modality config."""

    # Model tuning flags (selective fine-tuning akin to LoRA)
    tune_llm: bool = False
    """Whether to fine-tune the LLM backbone (Cosmos-Reason2-2B). Memory-heavy."""

    tune_visual: bool = False
    """Whether to fine-tune the visual encoder."""

    tune_projector: bool = True
    """Whether to fine-tune the multimodal projector layers. Recommended."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion action decoder. Recommended."""

    state_dropout_prob: float = 0.2
    """Dropout probability on state inputs for regularization."""

    # Data augmentation
    random_rotation_angle: int | None = None
    """Max rotation angle (degrees) for image augmentation. None = disabled."""

    color_jitter_params: dict[str, float] = field(default_factory=lambda: {
        "brightness": 0.3,
        "contrast": 0.4,
        "saturation": 0.5,
        "hue": 0.08,
    })
    """Color jitter parameters for image augmentation."""

    # Training hyperparameters
    max_steps: int = 10000
    """Total training steps."""

    learning_rate: float = 1e-4
    """Initial learning rate for AdamW."""

    weight_decay: float = 1e-5
    """Weight decay (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Fraction of training steps used for LR warm-up."""

    global_batch_size: int = 32
    """Effective batch size across all GPUs and accumulation steps."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps. Increase if GPU memory is limited."""

    # Checkpointing
    save_steps: int = 1000
    """Save a checkpoint every N steps."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to retain (oldest deleted)."""

    save_only_model: bool = False
    """If True, save only model weights (smaller, but not resumable)."""

    # Hardware
    num_gpus: int = 1
    """Number of GPUs for training."""

    dataloader_num_workers: int = 4
    """Parallel data-loading workers."""

    # Logging
    use_wandb: bool = False
    """Enable Weights & Biases logging."""

    wandb_project: str = "moses-gr00t-finetune"
    """W&B project name."""

    experiment_name: str | None = None
    """Optional experiment name (defaults to output_dir basename)."""

    # Evaluation
    eval_traj_ids: list[int] = field(default_factory=lambda: [0, 1, 2])
    """Trajectory IDs to use for open-loop evaluation after training."""

    eval_steps: int = 200
    """Max steps per trajectory during evaluation."""

    eval_action_horizon: int = 16
    """Action horizon for evaluation (typically 16 for open-loop)."""

    # Dataset generation
    video_fps: int = 30
    """Frames per second for video encoding in generated datasets."""

    image_size: tuple[int, int] = (224, 224)
    """Target (H, W) for dataset video frames."""


class Gr00TFineTuner:
    """High-level fine-tuning orchestrator for GR00T within Moses.

    Parameters
    ----------
    config : MosesFinetuneConfig
        Fine-tuning configuration dataclass.
    """

    def __init__(self, config: MosesFinetuneConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve embodiment tag early to catch typos
        self.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)

        # Register modality config if provided
        if config.modality_config_path:
            mod_path = Path(config.modality_config_path)
            if not mod_path.exists():
                raise FileNotFoundError(
                    f"Modality config path does not exist: {mod_path}"
                )
            load_modality_config(str(mod_path))
            logger.info("Registered modality config from %s", mod_path)

        logger.info(
            "Gr00TFineTuner initialized: base=%s, embodiment=%s, output=%s",
            config.base_model_path,
            self.embodiment_tag.name,
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_dataset_from_rollouts(
        self,
        rollout_dir: str | Path,
        output_path: str | Path | None = None,
    ) -> Path:
        """Convert Moses Isaac Lab rollout files into GR00T LeRobot v2 format.

        This method expects *rollout_dir* to contain a set of episode files
        (format is flexible—npz, pickle, or HDF5). Each episode should provide:

        - ``images/{cam_name}``: array of shape ``(T, H, W, 3) uint8``
        - ``states/joint_pos``: array of shape ``(T, D) float32``
        - ``actions``: array of shape ``(T, D_action) float32``
        - ``task_text``: str

        Parameters
        ----------
        rollout_dir : str or Path
            Directory containing raw rollout episode files.
        output_path : str or Path, optional
            Destination for the LeRobot dataset. Defaults to
            ``self.config.dataset_path``.

        Returns
        -------
        Path
            Path to the generated LeRobot v2 dataset root.
        """
        rollout_dir = Path(rollout_dir)
        if output_path is None:
            output_path = Path(self.config.dataset_path)
        else:
            output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Preparing dataset from rollouts in %s -> %s",
            rollout_dir,
            output_path,
        )

        # TODO: Implement full LeRobot v2 dataset writer integration.
        # The writer must produce:
        #   meta/info.json
        #   meta/episodes.jsonl
        #   meta/tasks.jsonl
        #   meta/modality.json
        #   data/chunk-000/*.parquet
        #   videos/chunk-000/*.mp4
        #
        # For now, we log a clear stub message and return the path so that
        # downstream code can check existence.
        logger.warning(
            "Dataset generation is a stub. Please ensure %s contains a valid "
            "LeRobot v2 dataset before calling finetune().",
            output_path,
        )
        return output_path

    # ------------------------------------------------------------------
    # Fine-tuning
    # ------------------------------------------------------------------

    def finetune(self) -> Path:
        """Launch the GR00T fine-tuning job.

        This method invokes ``gr00t/experiment/launch_finetune.py`` via
        subprocess with the parameters from :attr:`config`. Alternatively,
        you can import and call the launcher's ``run()`` function directly
        if running inside the same Python process.

        Returns
        -------
        Path
            Path to the final checkpoint directory.

        Raises
        ------
        RuntimeError
            If the training subprocess returns a non-zero exit code.
        """
        cfg = self.config

        # Build CLI arguments for launch_finetune.py
        cmd = [
            sys.executable,
            "-m",
            "gr00t.experiment.launch_finetune",
            "--base-model-path", cfg.base_model_path,
            "--dataset-path", cfg.dataset_path,
            "--embodiment-tag", self.embodiment_tag.name,
            "--output-dir", str(self.output_dir),
            "--max-steps", str(cfg.max_steps),
            "--learning-rate", str(cfg.learning_rate),
            "--weight-decay", str(cfg.weight_decay),
            "--warmup-ratio", str(cfg.warmup_ratio),
            "--global-batch-size", str(cfg.global_batch_size),
            "--gradient-accumulation-steps", str(cfg.gradient_accumulation_steps),
            "--save-steps", str(cfg.save_steps),
            "--save-total-limit", str(cfg.save_total_limit),
            "--num-gpus", str(cfg.num_gpus),
            "--dataloader-num-workers", str(cfg.dataloader_num_workers),
            "--tune-projector", str(cfg.tune_projector),
            "--tune-diffusion-model", str(cfg.tune_diffusion_model),
            "--state-dropout-prob", str(cfg.state_dropout_prob),
        ]

        if cfg.modality_config_path:
            cmd.extend(["--modality-config-path", cfg.modality_config_path])

        if cfg.tune_llm:
            cmd.append("--tune-llm")
        if cfg.tune_visual:
            cmd.append("--tune-visual")
        if cfg.save_only_model:
            cmd.append("--save-only-model")
        if cfg.use_wandb:
            cmd.append("--use-wandb")
            cmd.extend(["--wandb-project", cfg.wandb_project])
        if cfg.experiment_name:
            cmd.extend(["--experiment-name", cfg.experiment_name])
        if cfg.random_rotation_angle is not None:
            cmd.extend(["--random-rotation-angle", str(cfg.random_rotation_angle)])
        if cfg.color_jitter_params:
            cj = cfg.color_jitter_params
            cmd.extend([
                "--color-jitter-params",
                "brightness", str(cj.get("brightness", 0.3)),
                "contrast", str(cj.get("contrast", 0.4)),
                "saturation", str(cj.get("saturation", 0.5)),
                "hue", str(cj.get("hue", 0.08)),
            ])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(cfg.num_gpus))
        env["LOGURU_LEVEL"] = env.get("LOGURU_LEVEL", "INFO")

        logger.info("Launching fine-tuning: %s", " ".join(cmd))
        result = subprocess.run(cmd, env=env, capture_output=False, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Fine-tuning subprocess failed with exit code {result.returncode}."
            )

        # Determine the latest checkpoint
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not checkpoints:
            raise RuntimeError(
                f"No checkpoints found in {self.output_dir} after training."
            )
        latest_ckpt = checkpoints[-1]
        logger.info("Fine-tuning complete. Latest checkpoint: %s", latest_ckpt)
        return latest_ckpt

    def finetune_in_process(self) -> Path:
        """Launch fine-tuning in the current Python process (no subprocess).

        This is useful when running inside a long-lived training worker or
        Jupyter notebook where you want to keep the process alive.

        Returns
        -------
        Path
            Path to the final checkpoint directory.
        """
        from gr00t.configs.base_config import get_default_config
        from gr00t.experiment.experiment import run

        cfg = self.config
        dataset_paths = [p for p in cfg.dataset_path.split(os.pathsep) if p]

        config = get_default_config().load_dict(
            {
                "data": {
                    "download_cache": False,
                    "datasets": [
                        {
                            "dataset_paths": dataset_paths,
                            "mix_ratio": 1.0,
                            "embodiment_tag": self.embodiment_tag.value,
                        }
                    ],
                }
            }
        )
        config.load_config_path = None

        # Model tuning flags
        config.model.tune_llm = cfg.tune_llm
        config.model.tune_visual = cfg.tune_visual
        config.model.tune_projector = cfg.tune_projector
        config.model.tune_diffusion_model = cfg.tune_diffusion_model
        config.model.state_dropout_prob = cfg.state_dropout_prob
        config.model.random_rotation_angle = cfg.random_rotation_angle
        config.model.color_jitter_params = cfg.color_jitter_params
        config.model.extra_augmentation_config = None

        config.model.load_bf16 = False
        config.model.reproject_vision = False
        config.model.model_name = "nvidia/Cosmos-Reason2-2B"
        config.model.backbone_trainable_params_fp32 = True
        config.model.use_relative_action = True

        # Training config
        config.training.experiment_name = cfg.experiment_name
        config.training.start_from_checkpoint = cfg.base_model_path
        config.training.optim = "adamw_torch"
        config.training.global_batch_size = cfg.global_batch_size
        config.training.dataloader_num_workers = cfg.dataloader_num_workers
        config.training.learning_rate = cfg.learning_rate
        config.training.gradient_accumulation_steps = cfg.gradient_accumulation_steps
        config.training.output_dir = cfg.output_dir
        config.training.save_steps = cfg.save_steps
        config.training.save_total_limit = cfg.save_total_limit
        config.training.num_gpus = cfg.num_gpus
        config.training.use_wandb = cfg.use_wandb
        config.training.max_steps = cfg.max_steps
        config.training.weight_decay = cfg.weight_decay
        config.training.warmup_ratio = cfg.warmup_ratio
        config.training.wandb_project = cfg.wandb_project
        config.training.save_only_model = cfg.save_only_model
        config.training.skip_weight_loading = False

        config.data.shard_size = 2**10
        config.data.episode_sampling_rate = 0.1
        config.data.num_shards_per_epoch = int(1e5)

        logger.info("Starting in-process fine-tuning …")
        run(config)

        checkpoints = sorted(
            Path(cfg.output_dir).glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        if not checkpoints:
            raise RuntimeError(f"No checkpoints found in {cfg.output_dir}.")
        return checkpoints[-1]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        checkpoint_path: str | Path | None = None,
        baseline_model: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a fine-tuned checkpoint against a baseline using open-loop metrics.

        Parameters
        ----------
        checkpoint_path : str or Path, optional
            Path to the fine-tuned checkpoint. If ``None``, the latest
            checkpoint in :attr:`config.output_dir` is used.
        baseline_model : str, optional
            Path or HuggingFace ID of the baseline model for comparison.
            If ``None``, only the fine-tuned model is evaluated.

        Returns
        -------
        dict
            Evaluation results containing:

            - ``"finetuned_mse"``: float, mean squared error of fine-tuned model
            - ``"finetuned_mae"``: float, mean absolute error of fine-tuned model
            - ``"baseline_mse"``: float | None, baseline MSE
            - ``"baseline_mae"``: float | None, baseline MAE
            - ``"improvement_mse"``: float | None, relative improvement
            - ``"improvement_mae"``: float | None, relative improvement
        """
        if checkpoint_path is None:
            checkpoints = sorted(
                self.output_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]),
            )
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints in {self.output_dir}")
            checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = Path(checkpoint_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load fine-tuned policy
        logger.info("Evaluating fine-tuned model: %s", checkpoint_path)
        ft_policy = Gr00tPolicy(
            embodiment_tag=self.embodiment_tag,
            model_path=str(checkpoint_path),
            device=device,
            strict=False,
        )

        # Load dataset
        dataset = LeRobotEpisodeLoader(
            dataset_path=self.config.dataset_path,
            modality_configs=ft_policy.get_modality_config(),
            video_backend="torchcodec",
            video_backend_kwargs=None,
        )

        ft_mse_list: list[float] = []
        ft_mae_list: list[float] = []

        for traj_id in self.config.eval_traj_ids:
            if traj_id >= len(dataset):
                logger.warning("Trajectory %d out of range (dataset len=%d). Skipping.",
                               traj_id, len(dataset))
                continue
            mse, mae = evaluate_single_trajectory(
                policy=ft_policy,
                loader=dataset,
                traj_id=traj_id,
                embodiment_tag=self.embodiment_tag,
                modality_keys=None,
                steps=self.config.eval_steps,
                action_horizon=self.config.eval_action_horizon,
                save_plot_path=str(self.output_dir / f"eval_traj_{traj_id}.png"),
            )
            ft_mse_list.append(float(mse))
            ft_mae_list.append(float(mae))

        results: dict[str, Any] = {
            "finetuned_mse": float(np.mean(ft_mse_list)) if ft_mse_list else None,
            "finetuned_mae": float(np.mean(ft_mae_list)) if ft_mae_list else None,
            "baseline_mse": None,
            "baseline_mae": None,
            "improvement_mse": None,
            "improvement_mae": None,
        }

        # Baseline evaluation
        if baseline_model is not None:
            logger.info("Evaluating baseline model: %s", baseline_model)
            try:
                base_policy = Gr00tPolicy(
                    embodiment_tag=self.embodiment_tag,
                    model_path=baseline_model,
                    device=device,
                    strict=False,
                )
                base_mse_list: list[float] = []
                base_mae_list: list[float] = []
                for traj_id in self.config.eval_traj_ids:
                    if traj_id >= len(dataset):
                        continue
                    mse, mae = evaluate_single_trajectory(
                        policy=base_policy,
                        loader=dataset,
                        traj_id=traj_id,
                        embodiment_tag=self.embodiment_tag,
                        modality_keys=None,
                        steps=self.config.eval_steps,
                        action_horizon=self.config.eval_action_horizon,
                        save_plot_path=str(self.output_dir / f"baseline_traj_{traj_id}.png"),
                    )
                    base_mse_list.append(float(mse))
                    base_mae_list.append(float(mae))

                results["baseline_mse"] = float(np.mean(base_mse_list)) if base_mse_list else None
                results["baseline_mae"] = float(np.mean(base_mae_list)) if base_mae_list else None

                if results["baseline_mse"] and results["finetuned_mse"]:
                    results["improvement_mse"] = (
                        (results["baseline_mse"] - results["finetuned_mse"])
                        / results["baseline_mse"]
                    )
                if results["baseline_mae"] and results["finetuned_mae"]:
                    results["improvement_mae"] = (
                        (results["baseline_mae"] - results["finetuned_mae"])
                        / results["baseline_mae"]
                    )
            except Exception as exc:
                logger.error("Baseline evaluation failed: %s", exc)

        # Save results
        result_path = self.output_dir / "evaluation_results.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Evaluation results saved to %s", result_path)
        return results
