"""GR00T Adapter for Moses v4.0.

This module bridges between Moses observation/action spaces and the NVIDIA Isaac GR00T
N1.7 policy API. It handles:

- Loading pre-trained or fine-tuned GR00T models via :class:`Gr00tPolicy`.
- Converting Moses observations (Isaac Lab / gymnasium format) into GR00T's expected
  multimodal dictionary format ``{"video": ..., "state": ..., "language": ...}``.
- Extracting and converting GR00T action outputs back into Moses-compatible action tensors.
- Providing a fine-tuning wrapper that prepares Moses rollout data for GR00T training.

Dependencies
------------
- ``gr00t`` (NVIDIA Isaac-GR00T package)
- ``numpy``, ``torch``, ``gymnasium`` (or ``isaaclab``)

Example
-------
>>> from moses.gr00t.adapter import Gr00TAdapter
>>> adapter = Gr00TAdapter(
...     model_path="nvidia/GR00T-N1.7-3B",
...     embodiment_tag="NEW_EMBODIMENT",
...     device="cuda:0",
... )
>>> gr00t_obs = adapter.moses_obs_to_gr00t(moses_obs, task_text="pick up the cube")
>>> action = adapter.get_action(gr00t_obs)
>>> moses_action = adapter.gr00t_action_to_moses(action)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag

logger = logging.getLogger(__name__)


class Gr00TAdapter:
    """Adapter that bridges Moses environments with NVIDIA GR00T policies.

    Parameters
    ----------
    model_path : str
        Path to a GR00T checkpoint directory or a HuggingFace model ID
        (e.g. ``"nvidia/GR00T-N1.7-3B"``).
    embodiment_tag : str or EmbodimentTag
        Embodiment tag that defines the modality configuration. Use
        ``"NEW_EMBODIMENT"`` for custom robots (requires a modality config
        registered in the current process).
    device : str or int
        Torch device string (e.g. ``"cuda:0"``, ``"cpu"``) or integer GPU index.
    strict : bool, optional
        Enable strict observation/action validation in the underlying
        :class:`Gr00tPolicy`. Recommended during development. Default is ``True``.
    camera_key_map : dict[str, str] | None, optional
        Mapping from Moses camera names to GR00T video keys. If ``None``, a
        default mapping is used (``{"rgb_front": "front", "rgb_wrist": "wrist"}``).
    state_key_map : dict[str, str] | None, optional
        Mapping from Moses state keys to GR00T state keys. If ``None``, a
        default mapping is used (``{"joint_pos": "joint", "gripper": "gripper"}``).
    language_key : str, optional
        Key used for the language instruction inside the GR00T observation dict.
        Default is ``"annotation.human.task_description"``.
    action_horizon : int, optional
        Number of future action steps the model should predict per inference call.
        The GR00T N1.7 maximum is 16. Default is 8 for deployment (re-planning
        frequency) and 16 for open-loop evaluation.

    Attributes
    ----------
    policy : Gr00tPolicy
        The underlying GR00T policy instance.
    modality_configs : dict
        Modality configuration dict retrieved from the policy processor.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str | EmbodimentTag,
        device: str | int,
        *,
        strict: bool = True,
        camera_key_map: dict[str, str] | None = None,
        state_key_map: dict[str, str] | None = None,
        language_key: str = "annotation.human.task_description",
        action_horizon: int = 8,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.strict = strict
        self.camera_key_map = camera_key_map or {
            "rgb_front": "front",
            "rgb_wrist": "wrist",
            "rgb_head": "head",
        }
        self.state_key_map = state_key_map or {
            "joint_pos": "joint",
            "joint_vel": "joint_vel",
            "gripper": "gripper",
            "base_pose": "base_pose",
        }
        self.language_key = language_key
        self.action_horizon = action_horizon

        # Validate action horizon against GR00T hard limit
        if self.action_horizon > 16:
            raise ValueError(
                f"action_horizon ({self.action_horizon}) exceeds GR00T N1.7 maximum of 16."
            )

        logger.info(
            "Loading GR00T policy from %s with embodiment_tag=%s on device=%s",
            model_path,
            embodiment_tag,
            device,
        )

        try:
            self.policy = Gr00tPolicy(
                embodiment_tag=embodiment_tag,
                model_path=model_path,
                device=device,
                strict=strict,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Gr00tPolicy from {model_path} with tag {embodiment_tag}. "
                f"Ensure the checkpoint supports this embodiment tag."
            ) from exc

        self.modality_configs = self.policy.get_modality_config()
        logger.info("GR00T policy loaded. Modality configs: %s", list(self.modality_configs.keys()))

    # ------------------------------------------------------------------
    # Observation conversion: Moses → GR00T
    # ------------------------------------------------------------------

    def moses_obs_to_gr00t(
        self,
        moses_obs: dict[str, Any],
        task_text: str,
        *,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        """Convert a Moses observation into GR00T's expected observation dict.

        The Moses observation is expected to contain:

        - One or more RGB camera images as ``np.ndarray`` of shape ``(H, W, 3)``
          or ``(T, H, W, 3)`` with dtype ``uint8``.
        - Proprioceptive state as ``np.ndarray`` of shape ``(D,)`` or ``(T, D)``
          with dtype ``float32``.
        - Optionally a ``task_text`` string (passed as a separate argument).

        Parameters
        ----------
        moses_obs : dict
            Raw observation from the Moses environment.
        task_text : str
            Natural-language task instruction (e.g. ``"pick up the red cube"``).
        batch_size : int, optional
            Batch dimension to prepend. Default is 1.

        Returns
        -------
        dict
            GR00T observation dict with keys ``"video"``, ``"state"``, ``"language"``.

        Raises
        ------
        KeyError
            If a required Moses camera or state key is missing.
        ValueError
            If image/state shapes or dtypes are invalid.
        """
        gr00t_obs: dict[str, Any] = {"video": {}, "state": {}, "language": {}}

        # ---- Video -----------------------------------------------------
        expected_video_keys = self.modality_configs["video"].modality_keys
        video_horizon = len(self.modality_configs["video"].delta_indices)

        for gr00t_key in expected_video_keys:
            # Find the Moses key that maps to this GR00T key
            moses_key = self._reverse_lookup(self.camera_key_map, gr00t_key)
            if moses_key not in moses_obs:
                raise KeyError(
                    f"Missing camera observation '{moses_key}' (maps to GR00T '{gr00t_key}'). "
                    f"Available keys: {list(moses_obs.keys())}"
                )

            img = moses_obs[moses_key]
            img = self._ensure_video_format(img, video_horizon, moses_key)

            # Prepend batch dimension if missing
            if img.ndim == 4:  # (T, H, W, C)
                img = np.expand_dims(img, axis=0)  # (1, T, H, W, C)
            if img.shape[0] != batch_size:
                img = np.repeat(img, batch_size, axis=0)

            gr00t_obs["video"][gr00t_key] = img

        # ---- State -----------------------------------------------------
        expected_state_keys = self.modality_configs["state"].modality_keys
        state_horizon = len(self.modality_configs["state"].delta_indices)

        for gr00t_key in expected_state_keys:
            moses_key = self._reverse_lookup(self.state_key_map, gr00t_key)
            if moses_key not in moses_obs:
                # Some state keys may be optional (e.g. gripper not present in all obs)
                logger.warning(
                    "Missing state observation '%s' (maps to GR00T '%s'); using zeros.",
                    moses_key,
                    gr00t_key,
                )
                # Infer dimension from modality config if possible; fallback to 1
                dim = self._infer_state_dim(gr00t_key)
                state = np.zeros((batch_size, state_horizon, dim), dtype=np.float32)
            else:
                state = moses_obs[moses_key]
                state = self._ensure_state_format(state, state_horizon, moses_key)
                if state.ndim == 2:  # (T, D)
                    state = np.expand_dims(state, axis=0)  # (1, T, D)
                if state.shape[0] != batch_size:
                    state = np.repeat(state, batch_size, axis=0)

            gr00t_obs["state"][gr00t_key] = state

        # ---- Language --------------------------------------------------
        # GR00T expects: {key: [[str]]} with shape (B, T)
        gr00t_obs["language"][self.language_key] = [
            [task_text] for _ in range(batch_size)
        ]

        return gr00t_obs

    # ------------------------------------------------------------------
    # Action conversion: GR00T → Moses
    # ------------------------------------------------------------------

    def gr00t_action_to_moses(self, gr00t_action: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convert GR00T action output into Moses-compatible action dict.

        GR00T returns actions as a dict of arrays with shape ``(B, T_action, D)``.
        This method extracts the *first* action step (index 0) from the horizon
        and optionally renames keys back to Moses conventions.

        Parameters
        ----------
        gr00t_action : dict[str, np.ndarray]
            Raw action dict from :meth:`Gr00tPolicy.get_action`.

        Returns
        -------
        dict[str, np.ndarray]
            Moses action dict where each array has shape ``(B, D)``.
        """
        moses_action: dict[str, np.ndarray] = {}
        for gr00t_key, arr in gr00t_action.items():
            if arr.ndim != 3:
                raise ValueError(
                    f"Expected GR00T action '{gr00t_key}' to have shape (B, T, D), "
                    f"got {arr.shape}."
                )
            # Extract the first action step in the chunk
            first_step = arr[:, 0, :]  # (B, D)

            # Map back to Moses key if possible
            moses_key = self.state_key_map.get(gr00t_key, gr00t_key)
            moses_action[moses_key] = first_step

        return moses_action

    # ------------------------------------------------------------------
    # Unified inference interface
    # ------------------------------------------------------------------

    def get_action(
        self,
        moses_obs: dict[str, Any],
        task_text: str,
    ) -> dict[str, np.ndarray]:
        """End-to-end action computation from a Moses observation.

        This is a convenience wrapper that converts the observation, runs the
        GR00T policy, and converts the action back to Moses format.

        Parameters
        ----------
        moses_obs : dict
            Raw Moses environment observation.
        task_text : str
            Natural-language task instruction.

        Returns
        -------
        dict[str, np.ndarray]
            Moses-compatible action dict.
        """
        gr00t_obs = self.moses_obs_to_gr00t(moses_obs, task_text)
        gr00t_action, _info = self.policy.get_action(gr00t_obs)
        return self.gr00t_action_to_moses(gr00t_action)

    def reset(self) -> dict[str, Any]:
        """Reset the policy state between episodes.

        Returns
        -------
        dict
            Reset info dict (currently empty, reserved for future use).
        """
        return self.policy.reset()

    # ------------------------------------------------------------------
    # Fine-tuning dataset preparation
    # ------------------------------------------------------------------

    def prepare_finetune_dataset(
        self,
        rollout_buffer: list[dict[str, Any]],
        output_dir: str | Path,
        *,
        video_fps: int = 30,
        image_size: tuple[int, int] = (224, 224),
    ) -> Path:
        """Convert a Moses rollout buffer into GR00T LeRobot v2 format.

        This is a *stub* that should be extended with the actual LeRobot dataset
        writer once the ``lerobot`` package is available in the Moses environment.

        Parameters
        ----------
        rollout_buffer : list[dict]
            List of timestep dicts, each containing ``"obs"``, ``"action"``,
            ``"task_text"``, and optionally ``"image"`` or camera frames.
        output_dir : str or Path
            Directory where the LeRobot-format dataset will be written.
        video_fps : int, optional
            Frames per second for encoded video files. Default is 30.
        image_size : tuple[int, int], optional
            Target (H, W) for video frames. Default is (224, 224).

        Returns
        -------
        Path
            Path to the generated dataset root.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Integrate with lerobot v2 dataset writer once available.
        # The expected structure is:
        #   output_dir/
        #   ├── meta/
        #   │   ├── info.json
        #   │   ├── episodes.jsonl
        #   │   ├── tasks.jsonl
        #   │   └── modality.json
        #   ├── data/chunk-000/
        #   └── videos/chunk-000/
        logger.info(
            "Dataset preparation stub called for %d timesteps -> %s",
            len(rollout_buffer),
            output_dir,
        )
        return output_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reverse_lookup(mapping: dict[str, str], value: str) -> str:
        """Return the first key in *mapping* whose value equals *value*.

        If no match is found, returns *value* unchanged (assume 1:1 identity).
        """
        for k, v in mapping.items():
            if v == value:
                return k
        return value

    @staticmethod
    def _ensure_video_format(
        img: Any,
        expected_horizon: int,
        key_name: str,
    ) -> np.ndarray:
        """Validate and coerce an image array into GR00T video format.

        Expected output shape: ``(T, H, W, 3)`` with dtype ``uint8``.
        """
        if not isinstance(img, np.ndarray):
            raise ValueError(
                f"Video key '{key_name}' must be a numpy array, got {type(img)}."
            )

        # Handle single frame (H, W, 3) -> (1, H, W, 3)
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)

        if img.ndim != 4:
            raise ValueError(
                f"Video key '{key_name}' must have shape (T, H, W, 3), got {img.shape}."
            )

        if img.shape[0] != expected_horizon:
            # Temporal padding / truncation
            img = Gr00TAdapter._temporal_resample(img, expected_horizon)

        if img.shape[-1] != 3:
            raise ValueError(
                f"Video key '{key_name}' must have 3 channels (RGB), got {img.shape[-1]}."
            )

        if img.dtype != np.uint8:
            if img.dtype in (np.float32, np.float64):
                # Assume [0, 1] float images
                img = (img * 255).clip(0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        return img

    @staticmethod
    def _ensure_state_format(
        state: Any,
        expected_horizon: int,
        key_name: str,
    ) -> np.ndarray:
        """Validate and coerce a state array into GR00T state format.

        Expected output shape: ``(T, D)`` with dtype ``float32``.
        """
        if not isinstance(state, np.ndarray):
            raise ValueError(
                f"State key '{key_name}' must be a numpy array, got {type(state)}."
            )

        # Handle single step (D,) -> (1, D)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)

        if state.ndim != 2:
            raise ValueError(
                f"State key '{key_name}' must have shape (T, D), got {state.shape}."
            )

        if state.shape[0] != expected_horizon:
            state = Gr00TAdapter._temporal_resample(state, expected_horizon)

        if state.dtype != np.float32:
            state = state.astype(np.float32)

        return state

    @staticmethod
    def _temporal_resample(arr: np.ndarray, target_len: int) -> np.ndarray:
        """Resample a temporal sequence to *target_len* via nearest-neighbor indexing."""
        current_len = arr.shape[0]
        if current_len == target_len:
            return arr
        indices = np.linspace(0, current_len - 1, target_len, dtype=int)
        return arr[indices]

    def _infer_state_dim(self, gr00t_key: str) -> int:
        """Attempt to infer state dimension from modality config or fallback."""
        # GR00T does not expose dims directly in ModalityConfig; we use heuristics.
        # Override this method if your embodiment config provides explicit dims.
        known_dims = {
            "joint": 20,
            "joint_vel": 20,
            "gripper": 1,
            "base_pose": 3,
            "single_arm": 6,
        }
        return known_dims.get(gr00t_key, 1)
