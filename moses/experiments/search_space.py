"""
Search space definitions for Moses v5.0 experiments.

Provides hyperparameter, architecture, and environment search spaces
for PPO, SAC, GR00T algorithms with composable mixing.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import optuna
    from optuna.distributions import (
        FloatDistribution,
        IntDistribution,
        CategoricalDistribution,
    )
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ---------------------------------------------------------------------------
# Base Search Space
# ---------------------------------------------------------------------------

class SearchSpace(abc.ABC):
    """Abstract base for all search spaces."""

    @abc.abstractmethod
    def sample(self, trial: Any) -> Dict[str, Any]:
        """Sample parameters from this space using an Optuna trial."""
        ...

    @abc.abstractmethod
    def get_bounds(self) -> Dict[str, Tuple[Any, Any]]:
        """Return parameter bounds as (low, high) or list of choices."""
        ...

    def to_optuna(self, trial: Any, prefix: str = "") -> Dict[str, Any]:
        """Convert to Optuna suggestions."""
        return self.sample(trial)


# ---------------------------------------------------------------------------
# Hyperparameter Search Spaces
# ---------------------------------------------------------------------------

@dataclass
class PPOSearchSpace(SearchSpace):
    """Search space for Proximal Policy Optimization (PPO)."""

    # Learning rate
    lr_low: float = 1e-5
    lr_high: float = 1e-3
    lr_log: bool = True

    # Clip epsilon
    clip_eps_low: float = 0.1
    clip_eps_high: float = 0.3

    # Value function coefficient
    vf_coef_low: float = 0.1
    vf_coef_high: float = 1.0

    # Entropy coefficient
    ent_coef_low: float = 1e-4
    ent_coef_high: float = 0.1
    ent_coef_log: bool = True

    # GAE lambda
    gae_lambda_low: float = 0.9
    gae_lambda_high: float = 1.0

    # Number of epochs per update
    n_epochs_low: int = 3
    n_epochs_high: int = 15

    # Batch size (power of 2)
    batch_size_choices: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])

    # Number of steps per rollout
    n_steps_low: int = 512
    n_steps_high: int = 8192

    # Discount factor
    gamma_low: float = 0.95
    gamma_high: float = 0.999

    # Max gradient norm
    max_grad_norm_low: float = 0.1
    max_grad_norm_high: float = 1.0

    def sample(self, trial: Any) -> Dict[str, Any]:
        params = {
            "lr": trial.suggest_float("ppo_lr", self.lr_low, self.lr_high, log=self.lr_log),
            "clip_eps": trial.suggest_float("ppo_clip_eps", self.clip_eps_low, self.clip_eps_high),
            "vf_coef": trial.suggest_float("ppo_vf_coef", self.vf_coef_low, self.vf_coef_high),
            "ent_coef": trial.suggest_float("ppo_ent_coef", self.ent_coef_low, self.ent_coef_high, log=self.ent_coef_log),
            "gae_lambda": trial.suggest_float("ppo_gae_lambda", self.gae_lambda_low, self.gae_lambda_high),
            "n_epochs": trial.suggest_int("ppo_n_epochs", self.n_epochs_low, self.n_epochs_high),
            "batch_size": trial.suggest_categorical("ppo_batch_size", self.batch_size_choices),
            "n_steps": trial.suggest_int("ppo_n_steps", self.n_steps_low, self.n_steps_high, log=True),
            "gamma": trial.suggest_float("ppo_gamma", self.gamma_low, self.gamma_high),
            "max_grad_norm": trial.suggest_float("ppo_max_grad_norm", self.max_grad_norm_low, self.max_grad_norm_high),
        }
        return params

    def get_bounds(self) -> Dict[str, Tuple[Any, Any]]:
        return {
            "lr": (self.lr_low, self.lr_high),
            "clip_eps": (self.clip_eps_low, self.clip_eps_high),
            "vf_coef": (self.vf_coef_low, self.vf_coef_high),
            "ent_coef": (self.ent_coef_low, self.ent_coef_high),
            "gae_lambda": (self.gae_lambda_low, self.gae_lambda_high),
            "n_epochs": (self.n_epochs_low, self.n_epochs_high),
            "batch_size": self.batch_size_choices,
            "n_steps": (self.n_steps_low, self.n_steps_high),
            "gamma": (self.gamma_low, self.gamma_high),
            "max_grad_norm": (self.max_grad_norm_low, self.max_grad_norm_high),
        }


@dataclass
class SACSearchSpace(SearchSpace):
    """Search space for Soft Actor-Critic (SAC)."""

    # Learning rate for actor
    actor_lr_low: float = 1e-5
    actor_lr_high: float = 3e-4
    actor_lr_log: bool = True

    # Learning rate for critic
    critic_lr_low: float = 1e-5
    critic_lr_high: float = 3e-4
    critic_lr_log: bool = True

    # Temperature (alpha) learning rate
    alpha_lr_low: float = 1e-5
    alpha_lr_high: float = 1e-3
    alpha_lr_log: bool = True

    # Discount factor
    gamma_low: float = 0.95
    gamma_high: float = 0.999

    # Target network smoothing
    tau_low: float = 0.001
    tau_high: float = 0.05
    tau_log: bool = True

    # Replay buffer size
    buffer_size_choices: List[int] = field(default_factory=lambda: [100_000, 500_000, 1_000_000, 2_000_000])

    # Batch size
    batch_size_choices: List[int] = field(default_factory=lambda: [64, 128, 256, 512])

    # Target entropy (auto-tune or fixed)
    target_entropy_low: float = -10.0
    target_entropy_high: float = -1.0

    # Update frequency
    update_every_low: int = 1
    update_every_high: int = 4

    # Number of critic networks
    n_critics_choices: List[int] = field(default_factory=lambda: [1, 2])

    def sample(self, trial: Any) -> Dict[str, Any]:
        params = {
            "actor_lr": trial.suggest_float("sac_actor_lr", self.actor_lr_low, self.actor_lr_high, log=self.actor_lr_log),
            "critic_lr": trial.suggest_float("sac_critic_lr", self.critic_lr_low, self.critic_lr_high, log=self.critic_lr_log),
            "alpha_lr": trial.suggest_float("sac_alpha_lr", self.alpha_lr_low, self.alpha_lr_high, log=self.alpha_lr_log),
            "gamma": trial.suggest_float("sac_gamma", self.gamma_low, self.gamma_high),
            "tau": trial.suggest_float("sac_tau", self.tau_low, self.tau_high, log=self.tau_log),
            "buffer_size": trial.suggest_categorical("sac_buffer_size", self.buffer_size_choices),
            "batch_size": trial.suggest_categorical("sac_batch_size", self.batch_size_choices),
            "target_entropy": trial.suggest_float("sac_target_entropy", self.target_entropy_low, self.target_entropy_high),
            "update_every": trial.suggest_int("sac_update_every", self.update_every_low, self.update_every_high),
            "n_critics": trial.suggest_categorical("sac_n_critics", self.n_critics_choices),
        }
        return params

    def get_bounds(self) -> Dict[str, Tuple[Any, Any]]:
        return {
            "actor_lr": (self.actor_lr_low, self.actor_lr_high),
            "critic_lr": (self.critic_lr_low, self.critic_lr_high),
            "alpha_lr": (self.alpha_lr_low, self.alpha_lr_high),
            "gamma": (self.gamma_low, self.gamma_high),
            "tau": (self.tau_low, self.tau_high),
            "buffer_size": self.buffer_size_choices,
            "batch_size": self.batch_size_choices,
            "target_entropy": (self.target_entropy_low, self.target_entropy_high),
            "update_every": (self.update_every_low, self.update_every_high),
            "n_critics": self.n_critics_choices,
        }


@dataclass
class GR00TSearchSpace(SearchSpace):
    """Search space for GR00T (Generalist Robot with Object-Oriented Training)."""

    # Vision encoder learning rate
    vision_lr_low: float = 1e-6
    vision_lr_high: float = 1e-4
    vision_lr_log: bool = True

    # Policy head learning rate
    policy_lr_low: float = 1e-5
    policy_lr_high: float = 1e-3
    policy_lr_log: bool = True

    # Transformer layers
    n_layers_low: int = 4
    n_layers_high: int = 24

    # Attention heads
    n_heads_choices: List[int] = field(default_factory=lambda: [4, 8, 12, 16])

    # Embedding dimension
    embed_dim_choices: List[int] = field(default_factory=lambda: [256, 512, 768, 1024])

    # Dropout
    dropout_low: float = 0.0
    dropout_high: float = 0.3

    # Warmup steps
    warmup_steps_low: int = 100
    warmup_steps_high: int = 10_000
    warmup_steps_log: bool = True

    # Weight decay
    weight_decay_low: float = 1e-6
    weight_decay_high: float = 0.1
    weight_decay_log: bool = True

    # Action chunk size
    action_chunk_choices: List[int] = field(default_factory=lambda: [1, 4, 8, 16])

    # Diffusion steps (for diffusion policy)
    diffusion_steps_low: int = 10
    diffusion_steps_high: int = 100

    def sample(self, trial: Any) -> Dict[str, Any]:
        params = {
            "vision_lr": trial.suggest_float("gr00t_vision_lr", self.vision_lr_low, self.vision_lr_high, log=self.vision_lr_log),
            "policy_lr": trial.suggest_float("gr00t_policy_lr", self.policy_lr_low, self.policy_lr_high, log=self.policy_lr_log),
            "n_layers": trial.suggest_int("gr00t_n_layers", self.n_layers_low, self.n_layers_high),
            "n_heads": trial.suggest_categorical("gr00t_n_heads", self.n_heads_choices),
            "embed_dim": trial.suggest_categorical("gr00t_embed_dim", self.embed_dim_choices),
            "dropout": trial.suggest_float("gr00t_dropout", self.dropout_low, self.dropout_high),
            "warmup_steps": trial.suggest_int("gr00t_warmup_steps", self.warmup_steps_low, self.warmup_steps_high, log=self.warmup_steps_log),
            "weight_decay": trial.suggest_float("gr00t_weight_decay", self.weight_decay_low, self.weight_decay_high, log=self.weight_decay_log),
            "action_chunk_size": trial.suggest_categorical("gr00t_action_chunk_size", self.action_chunk_choices),
            "diffusion_steps": trial.suggest_int("gr00t_diffusion_steps", self.diffusion_steps_low, self.diffusion_steps_high),
        }
        return params

    def get_bounds(self) -> Dict[str, Tuple[Any, Any]]:
        return {
            "vision_lr": (self.vision_lr_low, self.vision_lr_high),
            "policy_lr": (self.policy_lr_low, self.policy_lr_high),
            "n_layers": (self.n_layers_low, self.n_layers_high),
            "n_heads": self.n_heads_choices,
            "embed_dim": self.embed_dim_choices,
            "dropout": (self.dropout_low, self.dropout_high),
            "warmup_steps": (self.warmup_steps_low, self.warmup_steps_high),
            "weight_decay": (self.weight_decay_low, self.weight_decay_high),
            "action_chunk_size": self.action_chunk_choices,
            "diffusion_steps": (self.diffusion_steps_low, self.diffusion_steps_high),
        }


# ---------------------------------------------------------------------------
# Architecture Search Spaces
# ---------------------------------------------------------------------------

@dataclass
class ArchitectureSearchSpace(SearchSpace):
    """Neural architecture search space for policy/value networks."""

    # Layer configurations
    min_layers: int = 2
    max_layers: int = 6

    # Units per layer
    min_units: int = 64
    max_units: int = 2048
    unit_step: int = 64

    # Activation functions
    activation_choices: List[str] = field(default_factory=lambda: ["relu", "elu", "swish", "gelu", "tanh"])

    # Use layer normalization
    use_layer_norm: bool = True

    # Use residual connections
    use_residual: bool = True

    # Network type
    network_type_choices: List[str] = field(default_factory=lambda: ["mlp", "transformer", "lstm", "gru"])

    # For transformers
    n_heads_choices: List[int] = field(default_factory=lambda: [4, 8, 12])
    ff_mult_choices: List[float] = field(default_factory=lambda: [2.0, 4.0])

    def sample(self, trial: Any) -> Dict[str, Any]:
        params = {
            "n_layers": trial.suggest_int("arch_n_layers", self.min_layers, self.max_layers),
            "units_per_layer": [
                trial.suggest_int(f"arch_units_l{i}", self.min_units, self.max_units, step=self.unit_step)
                for i in range(self.max_layers)
            ],
            "activation": trial.suggest_categorical("arch_activation", self.activation_choices),
            "network_type": trial.suggest_categorical("arch_network_type", self.network_type_choices),
        }

        if self.use_layer_norm:
            params["use_layer_norm"] = trial.suggest_categorical("arch_use_layer_norm", [True, False])
        if self.use_residual:
            params["use_residual"] = trial.suggest_categorical("arch_use_residual", [True, False])

        if params["network_type"] == "transformer":
            params["n_heads"] = trial.suggest_categorical("arch_n_heads", self.n_heads_choices)
            params["ff_multiplier"] = trial.suggest_categorical("arch_ff_mult", self.ff_mult_choices)

        # Only keep units for actual number of layers
        params["units_per_layer"] = params["units_per_layer"][:params["n_layers"]]

        return params

    def get_bounds(self) -> Dict[str, Tuple[Any, Any]]:
        bounds = {
            "n_layers": (self.min_layers, self.max_layers),
            "units_per_layer": (self.min_units, self.max_units),
            "activation": self.activation_choices,
            "network_type": self.network_type_choices,
        }
        if self.use_layer_norm:
            bounds["use_layer_norm"] = (True, False)
        if self.use_residual:
            bounds["use_residual"] = (True, False)
        return bounds


# ---------------------------------------------------------------------------
# Environment Search Spaces
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentSearchSpace(SearchSpace):
    """Domain randomization and environment parameter search space."""

    # Physics parameters
    gravity_low: float = 8.0
    gravity_high: float = 12.0

    # Mass randomization
    mass_scale_low: float = 0.8
    mass_scale_high: float = 1.2

    # Friction randomization
    friction_low: float = 0.5
    friction_high: float = 1.5

    # Joint damping
    damping_low: float = 0.0
    damping_high: float = 1.0

    # Action noise
    action_noise_low: float = 0.0
    action_noise_high: float = 0.1

    # Observation noise
    obs_noise_low: float = 0.0
    obs_noise_high: float = 0.05

    # Time step randomization
    dt_scale_low: float = 0.9
    dt_scale_high: float = 1.1

    # Camera randomization
    camera_pos_noise_low: float = 0.0
    camera_pos_noise_high: float = 0.05

    # Light randomization
    light_randomization: bool = True

    # Texture randomization
    texture_randomization: bool = True

    # Object count variation
    min_objects: int = 1
    max_objects: int = 10

    # Domain randomization frequency
    dr_freq_choices: List[str] = field(default_factory=lambda: ["episode", "step", "reset"])

    def sample(self, trial: Any) -> Dict[str, Any]:
        params = {
            "gravity": trial.suggest_float("env_gravity", self.gravity_low, self.gravity_high),
            "mass_scale": trial.suggest_float("env_mass_scale", self.mass_scale_low, self.mass_scale_high),
            "friction": trial.suggest_float("env_friction", self.friction_low, self.friction_high),
            "damping": trial.suggest_float("env_damping", self.damping_low, self.damping_high),
            "action_noise": trial.suggest_float("env_action_noise", self.action_noise_low, self.action_noise_high),
            "obs_noise": trial.suggest_float("env_obs_noise", self.obs_noise_low, self.obs_noise_high),
            "dt_scale": trial.suggest_float("env_dt_scale", self.dt_scale_low, self.dt_scale_high),
            "camera_pos_noise": trial.suggest_float("env_camera_noise", self.camera_pos_noise_low, self.camera_pos_noise_high),
            "n_objects": trial.suggest_int("env_n_objects", self.min_objects, self.max_objects),
            "dr_frequency": trial.suggest_categorical("env_dr_freq", self.dr_freq_choices),
        }

        if self.light_randomization:
            params["light_randomization"] = trial.suggest_categorical("env_light_rand", [True, False])
        if self.texture_randomization:
            params["texture_randomization"] = trial.suggest_categorical("env_texture_rand", [True, False])

        return params

    def get_bounds(self) -> Dict[str, Tuple[Any, Any]]:
        bounds = {
            "gravity": (self.gravity_low, self.gravity_high),
            "mass_scale": (self.mass_scale_low, self.mass_scale_high),
            "friction": (self.friction_low, self.friction_high),
            "damping": (self.damping_low, self.damping_high),
            "action_noise": (self.action_noise_low, self.action_noise_high),
            "obs_noise": (self.obs_noise_low, self.obs_noise_high),
            "dt_scale": (self.dt_scale_low, self.dt_scale_high),
            "camera_pos_noise": (self.camera_pos_noise_low, self.camera_pos_noise_high),
            "n_objects": (self.min_objects, self.max_objects),
            "dr_frequency": self.dr_freq_choices,
        }
        return bounds


# ---------------------------------------------------------------------------
# Composable Search Space
# ---------------------------------------------------------------------------

class ComposableSearchSpace(SearchSpace):
    """Mix and match multiple search spaces."""

    def __init__(self, spaces: Optional[Dict[str, SearchSpace]] = None):
        self.spaces: Dict[str, SearchSpace] = spaces or {}

    def add_space(self, name: str, space: SearchSpace) -> "ComposableSearchSpace":
        """Add a named search space. Returns self for chaining."""
        self.spaces[name] = space
        return self

    def remove_space(self, name: str) -> "ComposableSearchSpace":
        """Remove a named search space. Returns self for chaining."""
        self.spaces.pop(name, None)
        return self

    def sample(self, trial: Any) -> Dict[str, Any]:
        """Sample from all composed spaces with prefixed keys."""
        params = {}
        for name, space in self.spaces.items():
            sub_params = space.sample(trial)
            for key, value in sub_params.items():
                params[f"{name}/{key}"] = value
        return params

    def get_bounds(self) -> Dict[str, Tuple[Any, Any]]:
        bounds = {}
        for name, space in self.spaces.items():
            sub_bounds = space.get_bounds()
            for key, value in sub_bounds.items():
                bounds[f"{name}/{key}"] = value
        return bounds

    def get_space_params(self, name: str, full_params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters for a specific sub-space from full sampled params."""
        prefix = f"{name}/"
        return {
            key[len(prefix):]: value
            for key, value in full_params.items()
            if key.startswith(prefix)
        }

    @classmethod
    def from_algorithm(cls, algorithm: str) -> "ComposableSearchSpace":
        """Factory to create a standard composable space for an algorithm."""
        space = cls()

        if algorithm.lower() == "ppo":
            space.add_space("hyperparams", PPOSearchSpace())
        elif algorithm.lower() == "sac":
            space.add_space("hyperparams", SACSearchSpace())
        elif algorithm.lower() == "gr00t":
            space.add_space("hyperparams", GR00TSearchSpace())
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        space.add_space("architecture", ArchitectureSearchSpace())
        space.add_space("environment", EnvironmentSearchSpace())
        return space


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def grid_search_space(space: SearchSpace, n_samples: int = 10) -> List[Dict[str, Any]]:
    """Generate a grid of samples from a search space (for grid search fallback)."""
    bounds = space.get_bounds()
    samples = []

    for _ in range(n_samples):
        params = {}
        for key, bound in bounds.items():
            if isinstance(bound, tuple) and len(bound) == 2:
                low, high = bound
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = np.random.randint(low, high + 1)
                else:
                    params[key] = np.random.uniform(low, high)
            elif isinstance(bound, list):
                params[key] = np.random.choice(bound)
            else:
                params[key] = bound
        samples.append(params)

    return samples


def mutate_params(params: Dict[str, Any], space: SearchSpace, mutation_rate: float = 0.1) -> Dict[str, Any]:
    """Mutate a parameter dict within the search space bounds."""
    bounds = space.get_bounds()
    mutated = dict(params)

    for key, bound in bounds.items():
        if np.random.random() > mutation_rate:
            continue

        if isinstance(bound, tuple) and len(bound) == 2:
            low, high = bound
            if isinstance(low, int) and isinstance(high, int):
                mutated[key] = np.random.randint(low, high + 1)
            else:
                mutated[key] = np.random.uniform(low, high)
        elif isinstance(bound, list):
            mutated[key] = np.random.choice(bound)

    return mutated
