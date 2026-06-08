"""
Predictive World Model for Moses v4.0
======================================

Learns environment dynamics from experience.
Predicts next state given current state + action.
Uses for planning: model predictive control (MPC).
Reduces need for real environment interaction.
Uncertainty-aware predictions.

Mathematical Foundation
-----------------------
World model as a learned transition function:

    s_{t+1} ~ p_θ(s_{t+1} | s_t, a_t)

where θ are the model parameters. The model is trained to maximize:

    L(θ) = E_{(s_t, a_t, s_{t+1}) ~ D} [ log p_θ(s_{t+1} | s_t, a_t) ]

For deterministic models: s_{t+1} = f_θ(s_t, a_t)
For stochastic models: s_{t+1} = μ_θ(s_t, a_t) + σ_θ(s_t, a_t) · ε,  ε ~ N(0, I)

Model Predictive Control (MPC):
    At each timestep, plan H-step trajectory:
        a*_{t:t+H} = argmin_{a} Σ_{k=0}^{H-1} c(s̃_{t+k}, a_{t+k}) + V(s̃_{t+H})
    where s̃ are predicted states and c is the cost function.
    Execute first action, re-plan.

Uncertainty quantification via ensemble or Bayesian neural networks:
    Var[s_{t+1}] = E_θ[f_θ(s,a)²] - E_θ[f_θ(s,a)]²

References
----------
- Ha & Schmidhuber. "World Models." 2018. https://arxiv.org/abs/1803.10122
- Hafner et al. "Dream to Control: Learning Behaviors by Latent Imagination." ICLR 2020.
- Hafner et al. "Mastering Atari with Discrete World Models." ICLR 2021. (DreamerV2)
- Chua et al. "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models." NeurIPS 2018. (PETS)
- Nagabandi et al. "Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning." 2018.
- Deisenroth & Rasmussen. "PILCO: A Model-Based and Data-Efficient Approach to Policy Search." ICML 2011.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WorldModelConfig:
    """Configuration for the predictive world model."""
    state_dim: int = 64
    action_dim: int = 4
    latent_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 3
    ensemble_size: int = 5               # For uncertainty estimation
    predict_reward: bool = True
    predict_terminal: bool = True
    stochastic: bool = True
    min_std: float = 0.1
    max_std: float = 1.0
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    horizon: int = 15                    # MPC planning horizon
    num_candidates: int = 1000           # CEM candidate samples
    num_elites: int = 100                # CEM elite fraction
    num_cem_iterations: int = 5          # CEM optimization iterations
    # Uncertainty
    uncertainty_weight: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# State-Action Encoder
# ---------------------------------------------------------------------------

class StateActionEncoder(nn.Module):
    """
    Encodes (state, action) pairs into a joint representation.

    z = encoder([s, a])

    Uses layer normalization and residual connections for stable training
    of deep dynamics models.
    """

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, num_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(state_dim + action_dim, latent_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.SiLU(),
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + layer(x)  # residual
        return self.output_norm(x)


# ---------------------------------------------------------------------------
# Transition Model (Deterministic or Stochastic)
# ---------------------------------------------------------------------------

class TransitionModel(nn.Module):
    """
    Predicts next state (and optionally reward, terminal flag).

    Stochastic variant parameterizes a Gaussian:
        μ(s,a), σ(s,a) = network(s,a)
        s' ~ N(μ, diag(σ²))

    Deterministic variant predicts s' directly.

    Reference: Chua et al. "PETS" (NeurIPS 2018)
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.encoder = StateActionEncoder(
            config.state_dim, config.action_dim, config.latent_dim, config.num_layers
        )

        # Next state prediction
        self.next_state_mean = nn.Linear(config.latent_dim, config.state_dim)
        if config.stochastic:
            self.next_state_std = nn.Linear(config.latent_dim, config.state_dim)

        # Reward prediction
        if config.predict_reward:
            self.reward_head = nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, 1),
            )

        # Terminal prediction
        if config.predict_terminal:
            self.terminal_head = nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, 1),
            )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning predicted next state distribution and auxiliaries.

        Returns dict with keys:
            - next_state: predicted next state (mean if stochastic)
            - next_state_std: standard deviation (if stochastic)
            - reward: predicted reward (if enabled)
            - terminal_logit: terminal probability logit (if enabled)
        """
        z = self.encoder(state, action)

        result = {}
        mean = self.next_state_mean(z)

        if self.config.stochastic:
            # Bounded standard deviation
            raw_std = self.next_state_std(z)
            std = self.config.min_std + (self.config.max_std - self.config.min_std) * torch.sigmoid(raw_std)
            result["next_state"] = mean
            result["next_state_std"] = std
        else:
            result["next_state"] = mean

        if self.config.predict_reward:
            result["reward"] = self.reward_head(z).squeeze(-1)

        if self.config.predict_terminal:
            result["terminal_logit"] = self.terminal_head(z).squeeze(-1)

        return result

    def sample_next_state(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample a next state from the model.

        Returns (next_state, info_dict with mean/std for uncertainty).
        """
        pred = self.forward(state, action)
        if self.config.stochastic:
            dist = Independent(Normal(pred["next_state"], pred["next_state_std"]), 1)
            next_state = dist.rsample()
            return next_state, pred
        else:
            return pred["next_state"], pred


# ---------------------------------------------------------------------------
# Ensemble for Uncertainty Quantification
# ---------------------------------------------------------------------------

class EnsembleTransitionModel(nn.Module):
    """
    Ensemble of transition models for epistemic uncertainty estimation.

    The ensemble disagreement provides a proxy for model uncertainty:
        Var_epistemic[s'] = Var_{model}[ E[s' | model] ]

    This is crucial for safe MPC: uncertain regions should be avoided.

    Reference: Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty
    Estimation using Deep Ensembles." NeurIPS 2017.
    """

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        self.models = nn.ModuleList([
            TransitionModel(config) for _ in range(config.ensemble_size)
        ])

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Get predictions from all ensemble members."""
        return [model(state, action) for model in self.models]

    def predict_with_uncertainty(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate ensemble predictions with uncertainty estimates.

        Returns dict with:
            - mean: ensemble mean prediction
            - std: total predictive std (aleatoric + epistemic)
            - epistemic_std: std across ensemble means
            - aleatoric_std: mean of individual model stds
        """
        preds = self.forward(state, action)

        # Stack next state means
        means = torch.stack([p["next_state"] for p in preds], dim=0)  # (E, B, S)
        ensemble_mean = means.mean(dim=0)
        epistemic_var = means.var(dim=0)

        if self.config.stochastic:
            # Stack aleatoric variances
            aleatoric_vars = torch.stack([p["next_state_std"] ** 2 for p in preds], dim=0)
            aleatoric_var = aleatoric_vars.mean(dim=0)
        else:
            aleatoric_var = torch.zeros_like(epistemic_var)

        total_var = aleatoric_var + epistemic_var

        result = {
            "mean": ensemble_mean,
            "std": torch.sqrt(total_var + 1e-8),
            "epistemic_std": torch.sqrt(epistemic_var + 1e-8),
            "aleatoric_std": torch.sqrt(aleatoric_var + 1e-8),
        }

        # Aggregate rewards
        if self.config.predict_reward:
            rewards = torch.stack([p["reward"] for p in preds], dim=0)
            result["reward_mean"] = rewards.mean(dim=0)
            result["reward_std"] = rewards.std(dim=0)

        # Aggregate terminals
        if self.config.predict_terminal:
            terminals = torch.stack([torch.sigmoid(p["terminal_logit"]) for p in preds], dim=0)
            result["terminal_prob"] = terminals.mean(dim=0)

        return result

    def sample_next_state(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Sample from a randomly chosen ensemble member (TS-1 exploration)."""
        idx = np.random.randint(self.config.ensemble_size)
        return self.models[idx].sample_next_state(state, action)


# ---------------------------------------------------------------------------
# Latent Space Model (VAE-based, inspired by World Models / Dreamer)
# ---------------------------------------------------------------------------

class LatentWorldModel(nn.Module):
    """
    World model operating in a learned latent space.

    Architecture (from Ha & Schmidhuber 2018, Hafner et al. 2020):
        1. Vision / Observation Encoder: o_t → z_t
        2. Recurrent State Model: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        3. Transition Model: predicts z_t from h_{t-1}, a_{t-1}
        4. Reward Model: predicts r_t from h_t
        5. Observation Decoder: reconstructs o_t from h_t, z_t

    This enables imagination in latent space — much faster than pixel space.
    """

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Observation encoder (can be CNN for images)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # μ, logσ for VAE
        )

        # Recurrent dynamics
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Prior: p(z_t | h_t)
        self.prior = nn.Linear(hidden_dim, latent_dim * 2)

        # Posterior: q(z_t | h_t, o_t) — uses encoded observation
        self.posterior = nn.Linear(hidden_dim + latent_dim, latent_dim * 2)

        # Reward predictor
        self.reward_pred = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Observation decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def encode(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation to latent parameters."""
        params = self.encoder(obs)
        mu, log_std = params.chunk(2, dim=-1)
        return mu, torch.clamp(log_std, -10, 2)

    def reparameterize(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Single step of latent world model.

        Returns dict with predicted next latent, reward, reconstruction, etc.
        """
        # Encode current observation
        obs_mu, obs_log_std = self.encode(obs)
        z = self.reparameterize(obs_mu, obs_log_std)

        # Recurrent update
        gru_input = torch.cat([z, action], dim=-1)
        next_hidden = self.gru(gru_input, hidden)

        # Prior
        prior_params = self.prior(next_hidden)
        prior_mu, prior_log_std = prior_params.chunk(2, dim=-1)
        prior_log_std = torch.clamp(prior_log_std, -10, 2)

        # Posterior (uses next observation if available; here simplified)
        posterior_input = torch.cat([next_hidden, z], dim=-1)
        post_params = self.posterior(posterior_input)
        post_mu, post_log_std = post_params.chunk(2, dim=-1)
        post_log_std = torch.clamp(post_log_std, -10, 2)

        # Predictions
        reward = self.reward_pred(torch.cat([next_hidden, z], dim=-1)).squeeze(-1)
        recon = self.decoder(torch.cat([next_hidden, z], dim=-1))

        return {
            "hidden": next_hidden,
            "z": z,
            "prior_mu": prior_mu,
            "prior_log_std": prior_log_std,
            "post_mu": post_mu,
            "post_log_std": post_log_std,
            "reward_pred": reward,
            "reconstruction": recon,
            "obs_mu": obs_mu,
            "obs_log_std": obs_log_std,
        }

    def imagine(
        self, hidden: torch.Tensor, actions: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Rollout imagined trajectory given action sequence.

        Returns list of step predictions — no real environment needed.
        """
        trajectory = []
        h = hidden
        for t in range(actions.shape[0]):
            # Sample z from prior (no observation during imagination)
            prior_params = self.prior(h)
            prior_mu, prior_log_std = prior_params.chunk(2, dim=-1)
            prior_log_std = torch.clamp(prior_log_std, -10, 2)
            z = self.reparameterize(prior_mu, prior_log_std)

            # Predict next hidden
            gru_input = torch.cat([z, actions[t]], dim=-1)
            h = self.gru(gru_input, h)

            reward = self.reward_pred(torch.cat([h, z], dim=-1)).squeeze(-1)
            trajectory.append({
                "hidden": h,
                "z": z,
                "reward": reward,
                "prior_mu": prior_mu,
            })
        return trajectory


# ---------------------------------------------------------------------------
# Model Predictive Control (MPC) with CEM
# ---------------------------------------------------------------------------

class MPCPlanner:
    """
    Model Predictive Control using Cross-Entropy Method (CEM).

    At each timestep:
        1. Sample candidate action sequences
        2. Roll out world model for each candidate
        3. Evaluate cost / negative reward
        4. Refit distribution to elite candidates
        5. Repeat
        6. Execute first action of best sequence

    Reference: Chua et al. "PETS" (NeurIPS 2018)
    """

    def __init__(self, world_model: EnsembleTransitionModel, config: WorldModelConfig):
        self.world_model = world_model
        self.config = config

    def plan(
        self,
        current_state: torch.Tensor,
        cost_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Plan optimal action using CEM optimization.

        Args:
            current_state: (state_dim,) or (batch, state_dim)
            cost_fn: (state, action) → cost. If None, uses negative predicted reward.

        Returns:
            action: (action_dim,) optimal action
        """
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)

        batch_size = current_state.shape[0]
        action_dim = self.config.action_dim
        horizon = self.config.horizon
        num_candidates = self.config.num_candidates
        num_elites = self.config.num_elites
        num_iterations = self.config.num_cem_iterations

        # Initialize action distribution (Gaussian)
        mean = torch.zeros(horizon, action_dim, device=self.config.device)
        std = torch.ones(horizon, action_dim, device=self.config.device)

        for _ in range(num_iterations):
            # Sample candidate action sequences
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                num_candidates, horizon, action_dim, device=self.config.device
            )
            actions = torch.tanh(actions)  # bound actions

            # Roll out candidates
            states = current_state.unsqueeze(0).expand(num_candidates, -1, -1)
            costs = torch.zeros(num_candidates, device=self.config.device)

            for t in range(horizon):
                action_t = actions[:, t, :]  # (num_candidates, action_dim)
                # Expand states to match candidates
                state_t = states[:, 0, :] if states.dim() == 3 else states

                # Predict next state with uncertainty penalty
                pred = self.world_model.predict_with_uncertainty(state_t, action_t)
                next_state = pred["mean"]

                # Uncertainty penalty (exploration / safety)
                uncertainty_penalty = self.config.uncertainty_weight * pred["epistemic_std"].mean(dim=-1)

                # Cost
                if cost_fn is not None:
                    step_cost = cost_fn(next_state, action_t)
                elif "reward_mean" in pred:
                    step_cost = -pred["reward_mean"]  # minimize negative reward
                else:
                    step_cost = torch.zeros(num_candidates, device=self.config.device)

                costs += step_cost + uncertainty_penalty
                states = next_state.unsqueeze(1) if next_state.dim() == 2 else next_state

            # Select elites
            elite_indices = torch.argsort(costs)[:num_elites]
            elite_actions = actions[elite_indices]

            # Refit distribution
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6

        # Return first action of mean sequence
        return torch.tanh(mean[0])


# ---------------------------------------------------------------------------
# World Model Trainer
# ---------------------------------------------------------------------------

class WorldModelTrainer:
    """
    Trains the world model on environment interaction data.

    Data format: list of transitions (s_t, a_t, r_t, s_{t+1}, done_t)
    """

    def __init__(self, world_model: Union[TransitionModel, EnsembleTransitionModel, LatentWorldModel], config: WorldModelConfig):
        self.world_model = world_model
        self.config = config
        self.optimizer = torch.optim.Adam(world_model.parameters(), lr=config.learning_rate)
        self.buffer: List[Tuple[torch.Tensor, ...]] = []

    def add_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the replay buffer."""
        self.buffer.append((
            torch.tensor(state, dtype=torch.float32, device=self.config.device),
            torch.tensor(action, dtype=torch.float32, device=self.config.device),
            torch.tensor(reward, dtype=torch.float32, device=self.config.device),
            torch.tensor(next_state, dtype=torch.float32, device=self.config.device),
            torch.tensor(done, dtype=torch.float32, device=self.config.device),
        ))

    def train_step(self) -> Dict[str, float]:
        """Single gradient step on world model."""
        if len(self.buffer) < self.config.batch_size:
            return {"loss": 0.0}

        # Sample batch
        indices = np.random.choice(len(self.buffer), self.config.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.stack([b[0] for b in batch])
        actions = torch.stack([b[1] for b in batch])
        rewards = torch.stack([b[2] for b in batch])
        next_states = torch.stack([b[3] for b in batch])
        dones = torch.stack([b[4] for b in batch])

        self.optimizer.zero_grad()

        if isinstance(self.world_model, EnsembleTransitionModel):
            loss = self._ensemble_loss(states, actions, rewards, next_states, dones)
        elif isinstance(self.world_model, LatentWorldModel):
            loss = self._latent_loss(states, actions, rewards, next_states)
        else:
            loss = self._standard_loss(states, actions, rewards, next_states, dones)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item()}

    def _standard_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.world_model(states, actions)

        if self.config.stochastic:
            dist = Independent(Normal(pred["next_state"], pred["next_state_std"]), 1)
            loss = -dist.log_prob(next_states).mean()
        else:
            loss = F.mse_loss(pred["next_state"], next_states)

        if self.config.predict_reward:
            loss += F.mse_loss(pred["reward"], rewards)

        if self.config.predict_terminal:
            loss += F.binary_cross_entropy_with_logits(pred["terminal_logit"], dones)

        return loss

    def _ensemble_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Train each ensemble member on bootstrapped data."""
        total_loss = 0.0
        for model in self.world_model.models:
            pred = model(states, actions)
            if self.config.stochastic:
                dist = Independent(Normal(pred["next_state"], pred["next_state_std"]), 1)
                loss = -dist.log_prob(next_states).mean()
            else:
                loss = F.mse_loss(pred["next_state"], next_states)

            if self.config.predict_reward:
                loss += F.mse_loss(pred["reward"], rewards)
            total_loss += loss
        return total_loss / len(self.world_model.models)

    def _latent_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """Loss for latent world model (VAE + dynamics + reward)."""
        # Simplified: assume states are observations
        hidden = torch.zeros(states.shape[0], self.world_model.hidden_dim, device=states.device)
        result = self.world_model(states, actions, hidden)

        # Reconstruction loss
        recon_loss = F.mse_loss(result["reconstruction"], states)

        # KL divergence between posterior and prior
        prior_std = torch.exp(result["prior_log_std"])
        post_std = torch.exp(result["post_log_std"])
        kl = torch.log(prior_std / post_std) + (
            post_std ** 2 + (result["post_mu"] - result["prior_mu"]) ** 2
        ) / (2 * prior_std ** 2) - 0.5
        kl_loss = kl.sum(dim=-1).mean()

        # Reward prediction
        reward_loss = F.mse_loss(result["reward_pred"], rewards)

        return recon_loss + 0.5 * kl_loss + reward_loss


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "WorldModelConfig",
    "StateActionEncoder",
    "TransitionModel",
    "EnsembleTransitionModel",
    "LatentWorldModel",
    "MPCPlanner",
    "WorldModelTrainer",
]