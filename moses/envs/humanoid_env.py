"""
Moses Humanoid Environment for Isaac Lab v1.2.0

A 28-DOF humanoid locomotion environment with velocity tracking,
energy penalty, and stability rewards. Compatible with RSL-RL,
SKRL, and RL-Games training frameworks.

Author: Moses Team
Version: 3.0.0
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Literal, Sequence

# Isaac Lab core imports
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
)
from isaaclab.sensors import (
    ContactSensorCfg,
    ImuCfg,
    RayCasterCfg,
    patterns,
)
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, NoiseModelWithAdditiveBias

# Isaac Lab actuator / robot configs
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets import RigidObject

# Isaac Sim core
import omni.isaac.core.utils.prims as prim_utils
from pxr import Usd, UsdGeom, Gf

# Gymnasium
import gymnasium as gym


@configclass
class MosesHumanoidEnvCfg(DirectRLEnvCfg):
    """Configuration for the Moses Humanoid environment."""

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
    )

    # ------------------------------------------------------------------
    # Robot asset
    # ------------------------------------------------------------------
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=None,  # loaded from USD
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
                effort_limit=300.0,
                velocity_limit=10.0,
                stiffness={".*": 80.0},
                damping={".*": 5.0},
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*", ".*_wrist_.*"],
                effort_limit=150.0,
                velocity_limit=10.0,
                stiffness={".*": 40.0},
                damping={".*": 2.5},
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["waist.*", "neck.*"],
                effort_limit=200.0,
                velocity_limit=8.0,
                stiffness={".*": 60.0},
                damping={".*": 4.0},
            ),
        },
    )

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------
    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,
        history_length=6,
        track_air_time=True,
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/torso"],
    )

    imu_cfg: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/torso",
        update_period=0.0,
        gravity_bias=True,
    )

    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/head/camera",
        update_period=0.05,
        height=240,
        width=320,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=None,
        offset=CameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------
    randomize_mass: bool = True
    mass_range: tuple[float, float] = (0.8, 1.2)

    randomize_friction: bool = True
    friction_range: tuple[float, float] = (0.5, 1.25)

    randomize_gravity: bool = True
    gravity_range: tuple[float, float] = (-10.5, -9.3)

    # ------------------------------------------------------------------
    # Command: velocity tracking (x, y, yaw)
    # ------------------------------------------------------------------
    command_x_range: tuple[float, float] = (-1.0, 1.0)
    command_y_range: tuple[float, float] = (-0.5, 0.5)
    command_yaw_range: tuple[float, float] = (-1.0, 1.0)
    command_resample_interval: int = 500  # steps

    # ------------------------------------------------------------------
    # Reward scales
    # ------------------------------------------------------------------
    rew_scale_lin_vel: float = 1.0
    rew_scale_ang_vel: float = 0.5
    rew_scale_energy: float = -0.005
    rew_scale_stability: float = 0.1
    rew_scale_alive: float = 0.05
    rew_scale_foot_clearance: float = 0.1
    rew_scale_contact: float = 0.05

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    termination_height: float = 0.5
    termination_pitch_roll: float = 1.2  # rad (~68 deg)
    max_episode_length: int = 1000

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    num_observations: int = 133  # 3 base lin + 3 base ang + 28 pos + 28 vel + 28 force + 28 prev action + 3 command + 12 contact
    num_actions: int = 28
    observation_noise: float = 0.01

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------
    decimation: int = 4
    physics_dt: float = 0.005  # 200 Hz
    render_dt: float = 0.02    # 50 Hz


class MosesHumanoidEnv(DirectRLEnv):
    """
    Direct RL environment for a 28-DOF humanoid robot.

    Observations (133-D):
        - base linear velocity (3)
        - base angular velocity (3)
        - joint positions (28)
        - joint velocities (28)
        - joint actuator forces (28)
        - previous actions (28)
        - velocity commands (3)
        - contact sensor booleans (12)

    Actions (28-D):
        - Target joint positions (position control) or
        - Joint position offsets (delta control)

    Rewards:
        - Linear velocity tracking
        - Angular velocity tracking
        - Energy penalty (torque * velocity)
        - Stability bonus (upright torso)
        - Alive bonus
        - Foot clearance
        - Contact timing

    Terminations:
        - Base height below threshold
        - Base pitch/roll exceeds threshold
        - Episode length exceeded
    """

    cfg: MosesHumanoidEnvCfg

    def __init__(self, cfg: MosesHumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Internal buffers
        self._prev_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Joint indices (populated after asset init)
        self._hip_indices: list[int] = []
        self._knee_indices: list[int] = []
        self._ankle_indices: list[int] = []
        self._shoulder_indices: list[int] = []
        self._elbow_indices: list[int] = []
        self._wrist_indices: list[int] = []
        self._waist_indices: list[int] = []
        self._neck_indices: list[int] = []

        # Contact sensor body indices
        self._contact_body_ids: list[int] = []

        # Noise model
        self._obs_noise = NoiseModelWithAdditiveBias(
            noise_cfg=NoiseModelWithAdditiveBiasCfg(
                noise_type="gaussian",
                noise_std=self.cfg.observation_noise,
            ),
            num_envs=self.num_envs,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup_scene(self) -> None:
        """Configure the interactive scene with robot, ground, and sensors."""
        self.scene = InteractiveScene(self.cfg.scene)

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # Add robot
        self._robot = self.scene.articulations["robot"]

        # Add contact sensor
        self._contact_sensor = self.scene.sensors["contact"]

        # Add IMU
        self._imu = self.scene.sensors["imu"]

        # Add camera (optional, for vision tasks)
        if "camera" in self.scene.sensors:
            self._camera: Camera = self.scene.sensors["camera"]
        else:
            self._camera = None

        # Add ground plane
        cfg_ground = RigidObjectCfg(
            prim_path="/World/ground",
            spawn=None,
        )
        self._ground = RigidObject(cfg_ground)

        # Light
        prim_utils.create_prim(
            "/World/Light",
            "DomeLight",
            attributes={"intensity": 1000.0, "color": (1.0, 1.0, 1.0)},
        )

        # Resolve joint names to indices
        self._resolve_joint_indices()

    def _resolve_joint_indices(self) -> None:
        """Map joint name patterns to tensor indices."""
        joint_names = self._robot.joint_names
        for i, name in enumerate(joint_names):
            if "hip" in name:
                self._hip_indices.append(i)
            elif "knee" in name:
                self._knee_indices.append(i)
            elif "ankle" in name:
                self._ankle_indices.append(i)
            elif "shoulder" in name:
                self._shoulder_indices.append(i)
            elif "elbow" in name:
                self._elbow_indices.append(i)
            elif "wrist" in name:
                self._wrist_indices.append(i)
            elif "waist" in name:
                self._waist_indices.append(i)
            elif "neck" in name:
                self._neck_indices.append(i)

    # ------------------------------------------------------------------
    # Pre-physics step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics simulation."""
        self._prev_actions = actions.clone()

        # Scale actions to joint position targets
        # Action space: [-1, 1] -> [lower_limit, upper_limit]
        joint_limits = self._robot.data.joint_limits
        action_scaled = actions * (joint_limits[:, :, 1] - joint_limits[:, :, 0]) / 2.0
        self._robot.set_joint_position_target(action_scaled)

    def _apply_action(self) -> None:
        """Write actions to simulation (handled by articulation)."""
        pass  # Articulation writes targets in pre_physics_step

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Compute environment observations."""
        obs = self._compute_observations()
        obs_dict = {"policy": obs}

        # Include camera if available
        if self._camera is not None:
            rgb = self._camera.data.output.get("rgb", None)
            depth = self._camera.data.output.get("distance_to_image_plane", None)
            if rgb is not None:
                obs_dict["rgb"] = rgb.clone()
            if depth is not None:
                obs_dict["depth"] = depth.clone()

        return obs_dict

    def _compute_observations(self) -> torch.Tensor:
        """Build the 133-D observation vector."""
        # Base velocities (world frame)
        base_lin_vel = self._robot.data.root_lin_vel_b  # (N, 3) body frame
        base_ang_vel = self._robot.data.root_ang_vel_b  # (N, 3) body frame

        # Joint states
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        joint_force = self._robot.data.applied_torque

        # Contact sensor
        contact_forces = self._contact_sensor.data.net_forces_w
        contact_bool = (contact_forces.norm(dim=-1) > 1.0).float()  # (N, num_bodies)
        # Flatten / select key bodies
        contact_obs = contact_bool[:, :12]  # truncate/pad to 12

        # Concatenate
        obs = torch.cat(
            [
                base_lin_vel,           # 3
                base_ang_vel,           # 3
                joint_pos,              # 28
                joint_vel,              # 28
                joint_force,            # 28
                self._prev_actions,     # 28
                self._commands,         # 3
                contact_obs,            # 12
            ],
            dim=-1,
        )

        # Additive observation noise
        obs = self._obs_noise.add_noise(obs)

        return obs

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """Compute scalar reward for each environment."""
        # --- Velocity tracking ---
        lin_vel_error = torch.sum(
            (self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]) ** 2, dim=1
        )
        yaw_rate_error = (self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]) ** 2

        rew_lin_vel = torch.exp(-lin_vel_error / 0.25) * self.cfg.rew_scale_lin_vel
        rew_ang_vel = torch.exp(-yaw_rate_error / 0.25) * self.cfg.rew_scale_ang_vel

        # --- Energy penalty ---
        energy = torch.sum(
            torch.abs(self._robot.data.applied_torque * self._robot.data.joint_vel), dim=1
        )
        rew_energy = energy * self.cfg.rew_scale_energy

        # --- Stability (upright torso) ---
        projected_gravity = self._robot.data.projected_gravity_b
        stability = 1.0 - projected_gravity[:, 2].clamp(0.0, 1.0)  # 0 = upright, 1 = fallen
        rew_stability = (1.0 - stability) * self.cfg.rew_scale_stability

        # --- Alive bonus ---
        rew_alive = torch.ones(self.num_envs, device=self.device) * self.cfg.rew_scale_alive

        # --- Foot clearance (mid-swing) ---
        foot_z = self._robot.data.body_pos_w[:, self._ankle_indices, 2]
        rew_foot_clearance = torch.mean(foot_z.clamp(min=0.0, max=0.3), dim=1) * self.cfg.rew_scale_foot_clearance

        # --- Contact timing (reward alternating contact) ---
        contact_bool = (self._contact_sensor.data.net_forces_w.norm(dim=-1) > 1.0).float()
        left_contact = contact_bool[:, self._ankle_indices[0]] if len(self._ankle_indices) > 0 else torch.zeros(self.num_envs, device=self.device)
        right_contact = contact_bool[:, self._ankle_indices[1]] if len(self._ankle_indices) > 1 else torch.zeros(self.num_envs, device=self.device)
        rew_contact = (left_contact * right_contact) * self.cfg.rew_scale_contact  # penalize double stance slightly

        # Total
        total_reward = (
            rew_lin_vel
            + rew_ang_vel
            + rew_energy
            + rew_stability
            + rew_alive
            + rew_foot_clearance
            + rew_contact
        )

        return total_reward

    # ------------------------------------------------------------------
    # Terminations / resets
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (time_out, termination) booleans."""
        time_out = self._episode_step_count >= self.cfg.max_episode_length

        # Fall detection: base height
        base_height = self._robot.data.root_pos_w[:, 2]
        fallen = base_height < self.cfg.termination_height

        # Orientation violation
        projected_gravity = self._robot.data.projected_gravity_b
        pitch_roll = torch.acos(projected_gravity[:, 2].clamp(-1.0, 1.0))
        bad_orientation = pitch_roll > self.cfg.termination_pitch_roll

        # Joint limit violation (soft check)
        joint_pos = self._robot.data.joint_pos
        joint_limits = self._robot.data.joint_limits
        limit_violation = (
            (joint_pos < joint_limits[:, :, 0] - 0.1).any(dim=1)
            | (joint_pos > joint_limits[:, :, 1] + 0.1).any(dim=1)
        )

        terminated = fallen | bad_orientation | limit_violation

        return time_out, terminated

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Reset specified environments."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.tensor(list(env_ids), dtype=torch.int32, device=self.device)

        # Reset robot state
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset joints
        default_joint_pos = self._robot.data.default_joint_pos[env_ids]
        default_joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids)

        # Reset buffers
        self._prev_actions[env_ids] = 0.0
        self._episode_step_count[env_ids] = 0

        # Resample commands
        self._resample_commands(env_ids)

        # Domain randomization
        self._apply_domain_randomization(env_ids)

        # Reset sensors
        self._contact_sensor.reset(env_ids)
        self._imu.reset(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        """Sample new velocity commands."""
        self._commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (
            self.cfg.command_x_range[1] - self.cfg.command_x_range[0]
        ) + self.cfg.command_x_range[0]
        self._commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (
            self.cfg.command_y_range[1] - self.cfg.command_y_range[0]
        ) + self.cfg.command_y_range[0]
        self._commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (
            self.cfg.command_yaw_range[1] - self.cfg.command_yaw_range[0]
        ) + self.cfg.command_yaw_range[0]

    # ------------------------------------------------------------------
    # Domain randomization
    # ------------------------------------------------------------------
    def _apply_domain_randomization(self, env_ids: torch.Tensor) -> None:
        """Apply randomization to physics parameters."""
        if self.cfg.randomize_mass:
            mass_scale = torch.rand(len(env_ids), device=self.device) * (
                self.cfg.mass_range[1] - self.cfg.mass_range[0]
            ) + self.cfg.mass_range[0]
            # Apply to robot bodies via articulation API
            for i, env_id in enumerate(env_ids):
                body_masses = self._robot.data.default_mass[env_id]
                new_masses = body_masses * mass_scale[i]
                self._robot.root_physx_view.set_masses(new_masses, env_id)

        if self.cfg.randomize_friction:
            friction_scale = torch.rand(len(env_ids), device=self.device) * (
                self.cfg.friction_range[1] - self.cfg.friction_range[0]
            ) + self.cfg.friction_range[0]
            # Modify material properties via PhysX
            for i, env_id in enumerate(env_ids):
                # This requires per-env material handling; simplified here
                pass

        if self.cfg.randomize_gravity:
            gravity_z = torch.rand(len(env_ids), device=self.device) * (
                self.cfg.gravity_range[1] - self.cfg.gravity_range[0]
            ) + self.cfg.gravity_range[0]
            for i, env_id in enumerate(env_ids):
                self.sim.set_gravity(
                    torch.tensor([0.0, 0.0, gravity_z[i]], device=self.device), env_id
                )

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def _set_seed(self, seed: int) -> None:
        """Set random seed."""
        torch.manual_seed(seed)
        np.random.seed(seed)

    def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Override step to increment episode counters."""
        obs, reward, terminated, truncated, info = super().step(action)
        self._episode_step_count += 1
        return obs, reward, terminated, truncated, info
