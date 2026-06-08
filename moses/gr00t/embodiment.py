"""Moses Embodiment Configuration for Unitree H2 Plus + Sharpa Wave.

This module defines the modality configuration for the Moses humanoid platform:
- **Body**: Unitree H2 Plus (31 DoF, ~182 cm, 360 N·m joint torque)
- **Hands**: Sharpa Wave (22 DoF per hand, Dynamic Tactile Array sensing)
- **Compute**: NVIDIA Jetson Thor (T5000 module, 2070 FP4 TFLOPS)

The configuration is registered under ``EmbodimentTag.NEW_EMBODIMENT`` so that
GR00T N1.7 can be fine-tuned on Moses-specific walking and manipulation tasks.

Usage
-----
Import this module before launching fine-tuning or inference::

    from moses.gr00t.embodiment import MOSES_H2_SHARPA_CONFIG
    # The config is auto-registered on import.

    policy = Gr00tPolicy(
        model_path="./checkpoints/moses_gr00t/checkpoint-5000",
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        device="cuda:0",
    )

References
----------
- Unitree H2 Plus: https://www.originofbots.com/robot/unitree-h2-plus-by-unitree-robotics-details-specifications-rating
- Sharpa Wave: https://humanoid.guide/product/sharpawave/
- Jetson Thor: https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/
- GR00T Embodiment Config Guide: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md
"""

from __future__ import annotations

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

# =============================================================================
# Unitree H2 Plus Specifications
# =============================================================================

UNITREE_H2_PLUS_SPECS = {
    "name": "Unitree H2 Plus",
    "height_m": 1.82,
    "weight_kg": None,  # Not publicly specified
    "degrees_of_freedom": 31,
    "joint_torque_nm": 360,
    "max_speed": None,  # Not publicly specified
    "battery_life_hours": 3,
    "compute_platform": "NVIDIA Jetson-class (up to 2070 TOPS)",
    "ota_updates": True,
    "price_usd": 29900,
    "launch_year": 2026,
}

# Joint layout (inferred from 31 DoF humanoid topology)
# These are the *action* keys that GR00T will predict.
UNITREE_H2_JOINTS = {
    # Legs (12 DoF)
    "left_hip_yaw": {"limit": (-2.5, 2.5), "actuator": "high_torque"},
    "left_hip_roll": {"limit": (-1.5, 1.5), "actuator": "high_torque"},
    "left_hip_pitch": {"limit": (-2.5, 2.5), "actuator": "high_torque"},
    "left_knee_pitch": {"limit": (-2.5, 0.5), "actuator": "high_torque"},
    "left_ankle_pitch": {"limit": (-1.0, 1.0), "actuator": "medium_torque"},
    "left_ankle_roll": {"limit": (-0.8, 0.8), "actuator": "medium_torque"},
    "right_hip_yaw": {"limit": (-2.5, 2.5), "actuator": "high_torque"},
    "right_hip_roll": {"limit": (-1.5, 1.5), "actuator": "high_torque"},
    "right_hip_pitch": {"limit": (-2.5, 2.5), "actuator": "high_torque"},
    "right_knee_pitch": {"limit": (-2.5, 0.5), "actuator": "high_torque"},
    "right_ankle_pitch": {"limit": (-1.0, 1.0), "actuator": "medium_torque"},
    "right_ankle_roll": {"limit": (-0.8, 0.8), "actuator": "medium_torque"},
    # Waist (2 DoF)
    "waist_yaw": {"limit": (-2.5, 2.5), "actuator": "high_torque"},
    "waist_pitch": {"limit": (-1.5, 1.5), "actuator": "high_torque"},
    # Arms (14 DoF, excluding hands)
    "left_shoulder_pitch": {"limit": (-2.5, 2.5), "actuator": "medium_torque"},
    "left_shoulder_roll": {"limit": (-1.5, 1.5), "actuator": "medium_torque"},
    "left_shoulder_yaw": {"limit": (-2.5, 2.5), "actuator": "medium_torque"},
    "left_elbow_pitch": {"limit": (-2.5, 0.0), "actuator": "medium_torque"},
    "left_wrist_roll": {"limit": (-1.5, 1.5), "actuator": "low_torque"},
    "left_wrist_pitch": {"limit": (-1.5, 1.5), "actuator": "low_torque"},
    "left_wrist_yaw": {"limit": (-2.5, 2.5), "actuator": "low_torque"},
    "right_shoulder_pitch": {"limit": (-2.5, 2.5), "actuator": "medium_torque"},
    "right_shoulder_roll": {"limit": (-1.5, 1.5), "actuator": "medium_torque"},
    "right_shoulder_yaw": {"limit": (-2.5, 2.5), "actuator": "medium_torque"},
    "right_elbow_pitch": {"limit": (-2.5, 0.0), "actuator": "medium_torque"},
    "right_wrist_roll": {"limit": (-1.5, 1.5), "actuator": "low_torque"},
    "right_wrist_pitch": {"limit": (-1.5, 1.5), "actuator": "low_torque"},
    "right_wrist_yaw": {"limit": (-2.5, 2.5), "actuator": "low_torque"},
    # Neck (3 DoF, optional depending on configuration)
    "neck_yaw": {"limit": (-2.5, 2.5), "actuator": "low_torque"},
    "neck_pitch": {"limit": (-1.0, 1.0), "actuator": "low_torque"},
    "neck_roll": {"limit": (-0.8, 0.8), "actuator": "low_torque"},
}

# =============================================================================
# Sharpa Wave Hand Specifications
# =============================================================================

SHARPA_WAVE_SPECS = {
    "name": "Sharpa Wave",
    "degrees_of_freedom": 22,
    "weight_kg": 0.9,
    "dimensions_cm": (50, 50, 50),  # per-hand envelope
    "fingertip_force_n": 20,
    "grip_cycles": 1_000_000,
    "gestures_per_second": 4,
    "tactile_sensing": "Dynamic Tactile Array (DTA)",
    "tactile_pixels_per_fingertip": 1000,
    "backdrivable": True,
    "price_usd": 52000,
    "manufacturer": "Sharpa Robotics (Singapore)",
    "availability": "In production",
}

# Hand joint layout (22 DoF per hand)
# We model each hand as a single action key for GR00T (latent hand pose),
# but the full breakdown is documented here for reference.
SHARPA_WAVE_JOINTS = {
    # Thumb (4 DoF)
    "thumb_cmc_yaw": {"limit": (-0.8, 0.8)},
    "thumb_cmc_pitch": {"limit": (-0.5, 1.5)},
    "thumb_mcp": {"limit": (-0.5, 1.5)},
    "thumb_ip": {"limit": (-0.5, 1.5)},
    # Index (4 DoF)
    "index_mcp_yaw": {"limit": (-0.3, 0.3)},
    "index_mcp_pitch": {"limit": (-0.5, 1.5)},
    "index_pip": {"limit": (-0.5, 1.5)},
    "index_dip": {"limit": (-0.5, 1.5)},
    # Middle (4 DoF)
    "middle_mcp_yaw": {"limit": (-0.3, 0.3)},
    "middle_mcp_pitch": {"limit": (-0.5, 1.5)},
    "middle_pip": {"limit": (-0.5, 1.5)},
    "middle_dip": {"limit": (-0.5, 1.5)},
    # Ring (4 DoF)
    "ring_mcp_yaw": {"limit": (-0.3, 0.3)},
    "ring_mcp_pitch": {"limit": (-0.5, 1.5)},
    "ring_pip": {"limit": (-0.5, 1.5)},
    "ring_dip": {"limit": (-0.5, 1.5)},
    # Pinky (4 DoF)
    "pinky_mcp_yaw": {"limit": (-0.3, 0.3)},
    "pinky_mcp_pitch": {"limit": (-0.5, 1.5)},
    "pinky_pip": {"limit": (-0.5, 1.5)},
    "pinky_dip": {"limit": (-0.5, 1.5)},
    # Wrist (2 DoF)
    "wrist_roll": {"limit": (-1.5, 1.5)},
    "wrist_pitch": {"limit": (-1.5, 1.5)},
}

# =============================================================================
# Sensor Configuration
# =============================================================================

SENSOR_CONFIG = {
    # Cameras
    "cameras": {
        "head_rgb": {"resolution": (1280, 720), "fps": 30, "fov_deg": 90},
        "chest_rgb": {"resolution": (1280, 720), "fps": 30, "fov_deg": 120},
        "left_wrist_rgb": {"resolution": (640, 480), "fps": 30, "fov_deg": 75},
        "right_wrist_rgb": {"resolution": (640, 480), "fps": 30, "fov_deg": 75},
    },
    # Proprioception
    "proprioception": {
        "joint_positions": {"dim": 31, "units": "rad", "dtype": "float32"},
        "joint_velocities": {"dim": 31, "units": "rad/s", "dtype": "float32"},
        "joint_torques": {"dim": 31, "units": "N·m", "dtype": "float32"},
        "base_linear_velocity": {"dim": 3, "units": "m/s", "dtype": "float32"},
        "base_angular_velocity": {"dim": 3, "units": "rad/s", "dtype": "float32"},
        "base_orientation": {"dim": 4, "units": "quat (xyzw)", "dtype": "float32"},
        "imu_acceleration": {"dim": 3, "units": "m/s²", "dtype": "float32"},
        "imu_gyroscope": {"dim": 3, "units": "rad/s", "dtype": "float32"},
    },
    # Tactile (Sharpa Wave DTA)
    "tactile": {
        "left_hand_dta": {"pixels": 5000, "resolution": "1000 per fingertip", "dtype": "float32"},
        "right_hand_dta": {"pixels": 5000, "resolution": "1000 per fingertip", "dtype": "float32"},
    },
}

# =============================================================================
# GR00T Modality Configuration
# =============================================================================

# We define a compact action space for GR00T fine-tuning:
# - body: 31 joint position deltas (relative to current state)
# - left_hand: 22-DoF hand pose (absolute target, low-frequency control)
# - right_hand: 22-DoF hand pose (absolute target)
#
# Video uses 4 cameras (head, chest, left wrist, right wrist).
# State uses joint positions + base velocity + IMU.
# Language uses the task description.

MOSES_H2_SHARPA_CONFIG = {
    # ------------------------------------------------------------------
    # Video: current frame only (delta_indices=[0])
    # ------------------------------------------------------------------
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "head",          # head-mounted RGB camera
            "chest",         # chest-mounted wide-angle RGB camera
            "left_wrist",    # left wrist egocentric camera
            "right_wrist",   # right wrist egocentric camera
        ],
    ),
    # ------------------------------------------------------------------
    # State: current proprioceptive reading
    # ------------------------------------------------------------------
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "body_joints",       # 31-DOF joint positions (rad)
            "base_velocity",     # 6-DOF base linear + angular velocity
            "imu",               # 6-DOF IMU acceleration + gyroscope
        ],
    ),
    # ------------------------------------------------------------------
    # Action: 16-step prediction horizon
    # ------------------------------------------------------------------
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # predict 16 future steps
        modality_keys=[
            "body_joints",       # 31-DOF joint position deltas (relative)
            "left_hand",         # 22-DOF left hand pose (absolute)
            "right_hand",        # 22-DOF right hand pose (absolute)
        ],
        action_configs=[
            # body_joints: RELATIVE delta from current state (better generalization)
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="body_joints",
            ),
            # left_hand: ABSOLUTE target pose (hand control works better absolute)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand: ABSOLUTE target pose
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # Language: task instruction
    # ------------------------------------------------------------------
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# =============================================================================
# Register the configuration
# =============================================================================

register_modality_config(
    MOSES_H2_SHARPA_CONFIG,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)

# Convenience alias for imports
__all__ = [
    "MOSES_H2_SHARPA_CONFIG",
    "UNITREE_H2_PLUS_SPECS",
    "UNITREE_H2_JOINTS",
    "SHARPA_WAVE_SPECS",
    "SHARPA_WAVE_JOINTS",
    "SENSOR_CONFIG",
]
