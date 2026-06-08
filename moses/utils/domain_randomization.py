"""
Domain Randomization for Moses Humanoid Training

Provides Isaac Lab-compatible randomization operations for physics
parameters, lighting, and material properties. Designed to be called
from within the environment's `_reset_idx()` or `_pre_physics_step()`.

Author: Moses Team
Version: 3.0.0
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Sequence, Callable
from dataclasses import dataclass

# Isaac Lab / Isaac Sim
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform


# ------------------------------------------------------------------------------
# Randomization Event Configs
# ------------------------------------------------------------------------------

@dataclass
class RandomizationEventCfg:
    """Base configuration for a randomization event."""
    func: Callable
    mode: str = "reset"  # "reset" | "step" | "interval"
    interval: int = 1
    params: dict = None


@dataclass
class RandomizeMassCfg(RandomizationEventCfg):
    """Randomize link masses."""
    asset_name: str = "robot"
    mass_range: tuple[float, float] = (0.8, 1.2)
    distribution: str = "uniform"


@dataclass
class RandomizeFrictionCfg(RandomizationEventCfg):
    """Randomize ground / body friction coefficients."""
    asset_name: str = "robot"
    static_friction_range: tuple[float, float] = (0.5, 1.25)
    dynamic_friction_range: tuple[float, float] = (0.3, 1.0)
    restitution_range: tuple[float, float] = (0.0, 0.5)


@dataclass
class RandomizeGravityCfg(RandomizationEventCfg):
    """Randomize simulation gravity vector."""
    gravity_range: tuple[float, float] = (-10.5, -9.3)
    distribution: str = "uniform"


@dataclass
class RandomizeJointPropertiesCfg(RandomizationEventCfg):
    """Randomize joint stiffness, damping, armature."""
    asset_name: str = "robot"
    stiffness_range: tuple[float, float] = (0.9, 1.1)
    damping_range: tuple[float, float] = (0.9, 1.1)
    armature_range: tuple[float, float] = (0.9, 1.1)


@dataclass
class RandomizePushCfg(RandomizationEventCfg):
    """Apply random external pushes to the base."""
    asset_name: str = "robot"
    push_velocity_range: tuple[float, float] = (-1.0, 1.0)
    push_interval: int = 15  # every N steps
    push_duration: int = 1


# ------------------------------------------------------------------------------
# Randomization Functions (Isaac Lab compatible)
# ------------------------------------------------------------------------------

def randomize_link_masses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int],
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float] = (0.8, 1.2),
    distribution: str = "uniform",
) -> None:
    """
    Randomize the mass of each rigid body in the specified asset.

    Args:
        env: The Isaac Lab environment.
        env_ids: Environment indices to randomize.
        asset_cfg: Scene entity configuration for the target asset.
        mass_range: Multiplicative range [min, max] relative to default mass.
        distribution: "uniform" or "log_uniform".
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if isinstance(env_ids, Sequence):
        env_ids = torch.tensor(list(env_ids), dtype=torch.int32, device=asset.device)

    # Get default masses (N_envs, N_bodies)
    default_masses = asset.data.default_mass[env_ids]  # (n_ids, n_bodies)

    if distribution == "uniform":
        scale = sample_uniform(
            mass_range[0], mass_range[1], default_masses.shape, device=asset.device
        )
    elif distribution == "log_uniform":
        log_min = np.log(mass_range[0])
        log_max = np.log(mass_range[1])
        log_scale = sample_uniform(log_min, log_max, default_masses.shape, device=asset.device)
        scale = torch.exp(log_scale)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    new_masses = default_masses * scale

    # Write to PhysX
    for i, env_id in enumerate(env_ids):
        asset.root_physx_view.set_masses(new_masses[i], env_id)


def randomize_friction_coefficients(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int],
    asset_cfg: SceneEntityCfg,
    static_friction_range: tuple[float, float] = (0.5, 1.25),
    dynamic_friction_range: tuple[float, float] = (0.3, 1.0),
    restitution_range: tuple[float, float] = (0.0, 0.5),
) -> None:
    """
    Randomize material friction and restitution for the asset.

    In Isaac Sim, material properties are stored per-shape. This function
    applies uniform randomization across all collision shapes.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if isinstance(env_ids, Sequence):
        env_ids = torch.tensor(list(env_ids), dtype=torch.int32, device=asset.device)

    n_ids = len(env_ids)

    # Sample random values
    static_friction = sample_uniform(
        static_friction_range[0], static_friction_range[1], (n_ids,), device=asset.device
    )
    dynamic_friction = sample_uniform(
        dynamic_friction_range[0], dynamic_friction_range[1], (n_ids,), device=asset.device
    )
    restitution = sample_uniform(
        restitution_range[0], restitution_range[1], (n_ids,), device=asset.device
    )

    # Apply via PhysX material API
    for i, env_id in enumerate(env_ids):
        # Access the articulation's physx view
        n_shapes = asset.root_physx_view.max_shapes
        for shape_idx in range(n_shapes):
            asset.root_physx_view.set_friction_coefficients(
                static_friction[i], env_id, shape_idx
            )
            # Note: dynamic friction and restitution may require
            # direct material property access depending on Isaac Sim version


def randomize_gravity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int],
    gravity_range: tuple[float, float] = (-10.5, -9.3),
    distribution: str = "uniform",
) -> None:
    """
    Randomize the simulation gravity vector (Z-axis only).

    Args:
        env: The Isaac Lab environment.
        env_ids: Environment indices.
        gravity_range: Range for gravity magnitude in m/s^2 (negative = down).
        distribution: "uniform" or "gaussian".
    """
    if isinstance(env_ids, Sequence):
        env_ids = torch.tensor(list(env_ids), dtype=torch.int32, device=env.device)

    n_ids = len(env_ids)

    if distribution == "uniform":
        gravity_z = sample_uniform(
            gravity_range[0], gravity_range[1], (n_ids,), device=env.device
        )
    elif distribution == "gaussian":
        mean = (gravity_range[0] + gravity_range[1]) / 2.0
        std = (gravity_range[1] - gravity_range[0]) / 4.0
        gravity_z = torch.randn(n_ids, device=env.device) * std + mean
        gravity_z = gravity_z.clamp(gravity_range[0], gravity_range[1])
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    for i, env_id in enumerate(env_ids):
        env.sim.set_gravity(
            torch.tensor([0.0, 0.0, gravity_z[i]], device=env.device),
            env_id,
        )


def randomize_joint_properties(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int],
    asset_cfg: SceneEntityCfg,
    stiffness_range: tuple[float, float] = (0.9, 1.1),
    damping_range: tuple[float, float] = (0.9, 1.1),
    armature_range: tuple[float, float] = (0.9, 1.1),
) -> None:
    """
    Randomize joint stiffness, damping, and armature per environment.

    This modifies the articulation's drive parameters at reset.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if isinstance(env_ids, Sequence):
        env_ids = torch.tensor(list(env_ids), dtype=torch.int32, device=asset.device)

    n_ids = len(env_ids)
    n_joints = asset.num_joints

    stiffness_scale = sample_uniform(
        stiffness_range[0], stiffness_range[1], (n_ids, n_joints), device=asset.device
    )
    damping_scale = sample_uniform(
        damping_range[0], damping_range[1], (n_ids, n_joints), device=asset.device
    )
    armature_scale = sample_uniform(
        armature_range[0], armature_range[1], (n_ids, n_joints), device=asset.device
    )

    # Get default values from articulation
    default_stiffness = asset.data.default_joint_stiffness[env_ids]  # (n_ids, n_joints)
    default_damping = asset.data.default_joint_damping[env_ids]
    default_armature = asset.data.default_joint_armature[env_ids]

    new_stiffness = default_stiffness * stiffness_scale
    new_damping = default_damping * damping_scale
    new_armature = default_armature * armature_scale

    # Write to PhysX articulation
    for i, env_id in enumerate(env_ids):
        asset.write_joint_stiffness_to_sim(new_stiffness[i], env_id)
        asset.write_joint_damping_to_sim(new_damping[i], env_id)
        asset.write_joint_armature_to_sim(new_armature[i], env_id)


def apply_random_push(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int],
    asset_cfg: SceneEntityCfg,
    push_velocity_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    """
    Apply an instantaneous random velocity push to the robot base.

    Useful for training robustness to external disturbances.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if isinstance(env_ids, Sequence):
        env_ids = torch.tensor(list(env_ids), dtype=torch.int32, device=asset.device)

    n_ids = len(env_ids)

    # Random push in XY plane
    push_vel = sample_uniform(
        push_velocity_range[0],
        push_velocity_range[1],
        (n_ids, 3),
        device=asset.device,
    )
    push_vel[:, 2] = 0.0  # no vertical push

    # Add to current base velocity
    current_vel = asset.data.root_vel_w[env_ids]
    new_vel = current_vel + push_vel

    asset.write_root_velocity_to_sim(new_vel, env_ids)


# ------------------------------------------------------------------------------
# Convenience: Randomization Manager
# ------------------------------------------------------------------------------

class DomainRandomizationManager:
    """
    Orchestrates multiple randomization events with scheduling.

    Example::

        manager = DomainRandomizationManager(env)
        manager.add_event(RandomizeMassCfg(func=randomize_link_masses))
        manager.add_event(RandomizeGravityCfg(func=randomize_gravity))
        manager.reset(env_ids)
    """

    def __init__(self, env: ManagerBasedEnv):
        self.env = env
        self.events: list[RandomizationEventCfg] = []
        self._step_count: int = 0

    def add_event(self, cfg: RandomizationEventCfg) -> None:
        """Register a randomization event."""
        self.events.append(cfg)

    def reset(self, env_ids: torch.Tensor | Sequence[int]) -> None:
        """Trigger all reset-mode events."""
        for event in self.events:
            if event.mode == "reset":
                event.func(self.env, env_ids, **(event.params or {}))

    def step(self, env_ids: torch.Tensor | Sequence[int] | None = None) -> None:
        """Trigger step-mode events if interval has elapsed."""
        self._step_count += 1
        for event in self.events:
            if event.mode == "step" and self._step_count % event.interval == 0:
                if env_ids is None:
                    env_ids = torch.arange(self.env.num_envs, device=self.env.device)
                event.func(self.env, env_ids, **(event.params or {}))


# ------------------------------------------------------------------------------
# Pre-built randomization presets
# ------------------------------------------------------------------------------

def get_default_randomization_preset(env: ManagerBasedEnv) -> DomainRandomizationManager:
    """
    Return a manager with standard Moses humanoid randomizations.
    """
    manager = DomainRandomizationManager(env)

    manager.add_event(
        RandomizationEventCfg(
            func=randomize_link_masses,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "mass_range": (0.8, 1.2),
                "distribution": "uniform",
            },
        )
    )

    manager.add_event(
        RandomizationEventCfg(
            func=randomize_friction_coefficients,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "static_friction_range": (0.5, 1.25),
                "dynamic_friction_range": (0.3, 1.0),
                "restitution_range": (0.0, 0.5),
            },
        )
    )

    manager.add_event(
        RandomizationEventCfg(
            func=randomize_gravity,
            mode="reset",
            params={
                "gravity_range": (-10.5, -9.3),
                "distribution": "uniform",
            },
        )
    )

    manager.add_event(
        RandomizationEventCfg(
            func=randomize_joint_properties,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "stiffness_range": (0.9, 1.1),
                "damping_range": (0.9, 1.1),
                "armature_range": (0.9, 1.1),
            },
        )
    )

    manager.add_event(
        RandomizationEventCfg(
            func=apply_random_push,
            mode="step",
            interval=15,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "push_velocity_range": (-1.0, 1.0),
            },
        )
    )

    return manager
