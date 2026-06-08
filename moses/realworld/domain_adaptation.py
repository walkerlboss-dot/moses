"""
Domain Adaptation Module for Sim-to-Real Transfer
=================================================

Bridges the reality gap between simulation and physical deployment through:
- Domain randomization tuning from real data
- Adversarial domain adaptation (DANN, gradient reversal)
- Meta-learning for fast adaptation (MAML, Reptile)
- Sim-to-real gap quantification

References:
-----------
[1] Tobin et al., "Domain Randomization for Transferring Deep Neural Networks
    from Simulation to the Real World", IROS 2017.
[2] Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.
[3] Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks",
    ICML 2017.
[4] Tan et al. [2] for sim-to-real with randomization
[5] Chebotar et al., "Closing the Sim-to-Real Loop: Adapting Simulation Randomization
    with Real World Experience", ICRA 2019.
[6] Hwangbo et al. [3] for dynamics randomization
[7] Rudin et al. [4] for rapid adaptation
[8] Zhao et al. [1] for comprehensive survey

Author: Moses Team
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import copy

logger = logging.getLogger(__name__)


class AdaptationMethod(Enum):
    """Supported domain adaptation methods."""
    DOMAIN_RANDOMIZATION = "domain_randomization"
    ADVERSARIAL = "adversarial"  # DANN-style
    META_LEARNING = "meta_learning"  # MAML/Reptile
    SYSTEM_ID = "system_id"  # Direct parameter matching
    ENSEMBLE = "ensemble"


@dataclass
class DomainAdaptationConfig:
    """Configuration for domain adaptation."""
    method: AdaptationMethod = AdaptationMethod.DOMAIN_RANDOMIZATION
    
    # Domain randomization parameters
    randomize_mass: bool = True
    randomize_inertia: bool = True
    randomize_friction: bool = True
    randomize_motor: bool = True
    randomize_delay: bool = True
    randomize_sensor_noise: bool = True
    
    mass_range: Tuple[float, float] = (0.8, 1.2)
    inertia_range: Tuple[float, float] = (0.8, 1.2)
    friction_range: Tuple[float, float] = (0.5, 1.5)
    motor_gain_range: Tuple[float, float] = (0.8, 1.2)
    delay_range: Tuple[int, int] = (0, 3)  # timesteps
    sensor_noise_range: Tuple[float, float] = (0.0, 0.1)
    
    # Adversarial adaptation
    adversarial_lambda: float = 0.1
    discriminator_lr: float = 1e-4
    feature_extractor_lr: float = 1e-4
    
    # Meta-learning
    meta_lr: float = 1e-3
    inner_lr: float = 0.01
    inner_steps: int = 5
    meta_batch_size: int = 4
    
    # General
    batch_size: int = 256
    epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for Domain-Adversarial Neural Networks (DANN).
    
    From Ganin et al. [2]: Forward pass is identity, backward pass
    multiplies gradient by -lambda, encouraging domain-invariant features.
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial domain adaptation.
    
    Predicts whether features come from simulation (0) or real (1).
    """
    
    def __init__(self, feature_dim: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        layers = []
        prev_dim = feature_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.net(x))


class AdversarialDomainAdapter(nn.Module):
    """
    Adversarial domain adaptation for sim-to-real transfer.
    
    Implements DANN [2] with gradient reversal: trains a policy network
    whose features are domain-invariant (discriminator can't tell sim from real).
    """
    
    def __init__(self,
                 policy_network: nn.Module,
                 feature_dim: int,
                 config: DomainAdaptationConfig = None):
        super().__init__()
        self.config = config or DomainAdaptationConfig()
        self.policy = policy_network
        self.discriminator = DomainDiscriminator(feature_dim)
        self.lambda_ = config.adversarial_lambda if config else 0.1
        
        self.feature_extractor = None  # Set to policy's feature layers
        self._extract_features = False
        
    def set_feature_extractor(self, feature_layers: nn.Module):
        """Set which layers extract domain-invariant features."""
        self.feature_extractor = feature_layers
        
    def forward(self, obs: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both policy output and domain prediction.
        
        Args:
            obs: Observation tensor
            alpha: Gradient reversal coefficient (increases during training)
        
        Returns:
            (policy_output, domain_prediction)
        """
        # Extract features through policy
        if self.feature_extractor is not None:
            features = self.feature_extractor(obs)
        else:
            features = obs
            
        # Policy output
        policy_output = self.policy(obs)
        
        # Domain prediction with gradient reversal
        reversed_features = GradientReversalLayer.apply(features, alpha * self.lambda_)
        domain_pred = self.discriminator(reversed_features)
        
        return policy_output, domain_pred
    
    def adapt(self,
              sim_data: torch.Tensor,
              real_data: torch.Tensor,
              sim_labels: torch.Tensor,
              epochs: int = 100) -> Dict[str, List[float]]:
        """
        Train with adversarial domain adaptation.
        
        Args:
            sim_data: Simulated observations
            real_data: Real observations
            sim_labels: Simulated action labels / rewards
            epochs: Training epochs
        
        Returns:
            Training history
        """
        device = self.config.device
        self.to(device)
        
        sim_dataset = TensorDataset(sim_data, sim_labels)
        real_dataset = TensorDataset(real_data)
        
        sim_loader = DataLoader(sim_dataset, batch_size=self.config.batch_size, shuffle=True)
        real_loader = DataLoader(real_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.feature_extractor_lr
        )
        disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.discriminator_lr
        )
        
        history = {'policy_loss': [], 'disc_loss': [], 'domain_acc': []}
        
        for epoch in range(epochs):
            epoch_policy_loss = 0
            epoch_disc_loss = 0
            epoch_domain_acc = 0
            n_batches = 0
            
            # Progressively increase lambda
            p = float(epoch) / epochs
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            
            for (sim_batch, sim_labels_batch), (real_batch,) in zip(sim_loader, real_loader):
                sim_batch = sim_batch.to(device)
                real_batch = real_batch.to(device)
                sim_labels_batch = sim_labels_batch.to(device)
                
                batch_size = sim_batch.size(0)
                
                # Labels: 0 for sim, 1 for real
                sim_domain = torch.zeros(batch_size, 1).to(device)
                real_domain = torch.ones(batch_size, 1).to(device)
                
                # Forward pass
                sim_output, sim_domain_pred = self.forward(sim_batch, alpha)
                _, real_domain_pred = self.forward(real_batch, alpha)
                
                # Policy loss (only on sim data with labels)
                policy_loss = F.mse_loss(sim_output, sim_labels_batch)
                
                # Domain loss
                disc_loss_sim = F.binary_cross_entropy(sim_domain_pred, sim_domain)
                disc_loss_real = F.binary_cross_entropy(real_domain_pred, real_domain)
                disc_loss = disc_loss_sim + disc_loss_real
                
                # Update discriminator
                disc_optimizer.zero_grad()
                disc_loss.backward(retain_graph=True)
                disc_optimizer.step()
                
                # Update policy (adversarial)
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                
                # Metrics
                sim_pred_label = (sim_domain_pred < 0.5).float()
                real_pred_label = (real_domain_pred >= 0.5).float()
                domain_acc = (sim_pred_label.sum() + real_pred_label.sum()) / (2 * batch_size)
                
                epoch_policy_loss += policy_loss.item()
                epoch_disc_loss += disc_loss.item()
                epoch_domain_acc += domain_acc.item()
                n_batches += 1
                
            history['policy_loss'].append(epoch_policy_loss / n_batches)
            history['disc_loss'].append(epoch_disc_loss / n_batches)
            history['domain_acc'].append(epoch_domain_acc / n_batches)
            
            if self.config.verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: policy_loss={history['policy_loss'][-1]:.4f}, "
                           f"disc_loss={history['disc_loss'][-1]:.4f}, "
                           f"domain_acc={history['domain_acc'][-1]:.4f}")
                
        return history


class MetaLearningAdapter:
    """
    Model-Agnostic Meta-Learning (MAML) for fast sim-to-real adaptation.
    
    From Finn et al. [3]: Learns initial parameters that can adapt to real
    robot dynamics with few gradient steps.
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: DomainAdaptationConfig = None):
        self.config = config or DomainAdaptationConfig()
        self.model = model.to(self.config.device)
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.meta_lr
        )
        
    def inner_loop(self,
                   model: nn.Module,
                   task_data: torch.Tensor,
                   task_labels: torch.Tensor,
                   create_graph: bool = True) -> nn.Module:
        """
        Perform inner loop adaptation on a single task.
        
        Args:
            model: Model to adapt
            task_data: Task observations
            task_labels: Task labels
            create_graph: Whether to create computation graph for meta-gradients
        
        Returns:
            Adapted model
        """
        adapted_model = copy.deepcopy(model)
        adapted_params = list(adapted_model.parameters())
        
        for step in range(self.config.inner_steps):
            pred = adapted_model(task_data)
            loss = F.mse_loss(pred, task_labels)
            
            grads = torch.autograd.grad(
                loss, adapted_params,
                create_graph=create_graph,
                allow_unused=True
            )
            
            # Manual SGD update
            for param, grad in zip(adapted_params, grads):
                if grad is not None:
                    param.data = param.data - self.config.inner_lr * grad
                    
        return adapted_model
    
    def meta_train(self,
                   task_datasets: List[Tuple[torch.Tensor, torch.Tensor]],
                   epochs: int = 100) -> Dict[str, List[float]]:
        """
        Meta-train on distribution of tasks (different sim randomizations).
        
        Args:
            task_datasets: List of (data, labels) tuples, one per task
            epochs: Meta-training epochs
        
        Returns:
            Training history
        """
        device = self.config.device
        history = {'meta_loss': []}
        
        for epoch in range(epochs):
            meta_loss = 0
            
            # Sample meta-batch of tasks
            task_indices = np.random.choice(
                len(task_datasets),
                size=min(self.config.meta_batch_size, len(task_datasets)),
                replace=False
            )
            
            self.meta_optimizer.zero_grad()
            
            for task_idx in task_indices:
                task_data, task_labels = task_datasets[task_idx]
                task_data = task_data.to(device)
                task_labels = task_labels.to(device)
                
                # Split into support and query sets
                split = int(0.5 * len(task_data))
                support_data, query_data = task_data[:split], task_data[split:]
                support_labels, query_labels = task_labels[:split], task_labels[split:]
                
                # Inner loop on support set
                adapted_model = self.inner_loop(
                    self.model, support_data, support_labels, create_graph=True
                )
                
                # Evaluate on query set
                query_pred = adapted_model(query_data)
                task_loss = F.mse_loss(query_pred, query_labels)
                meta_loss += task_loss
                
            # Meta-update
            meta_loss = meta_loss / len(task_indices)
            meta_loss.backward()
            self.meta_optimizer.step()
            
            history['meta_loss'].append(meta_loss.item())
            
            if self.config.verbose and epoch % 10 == 0:
                logger.info(f"Meta-epoch {epoch}: loss={meta_loss.item():.4f}")
                
        return history
    
    def adapt_to_real(self,
                      real_data: torch.Tensor,
                      real_labels: torch.Tensor,
                      steps: int = None) -> nn.Module:
        """
        Fast adaptation to real robot with few gradient steps.
        
        Args:
            real_data: Real observations
            real_labels: Real labels (or proxy rewards)
            steps: Number of adaptation steps (default: config.inner_steps)
        
        Returns:
            Adapted model for real deployment
        """
        if steps is None:
            steps = self.config.inner_steps
            
        adapted_model = self.inner_loop(
            self.model, real_data, real_labels, create_graph=False
        )
        return adapted_model


class DynamicsRandomizer:
    """
    Physics parameter randomization for domain randomization.
    
    From Tobin et al. [1] and Tan et al. [2]: Randomize simulation parameters
    during training to create robust policies.
    """
    
    def __init__(self, config: DomainAdaptationConfig = None):
        self.config = config or DomainAdaptationConfig()
        
    def sample_randomization(self) -> Dict[str, np.ndarray]:
        """Sample a randomization configuration."""
        randomization = {}
        
        if self.config.randomize_mass:
            randomization['mass_scale'] = np.random.uniform(
                *self.config.mass_range
            )
        if self.config.randomize_inertia:
            randomization['inertia_scale'] = np.random.uniform(
                *self.config.inertia_range
            )
        if self.config.randomize_friction:
            randomization['friction_scale'] = np.random.uniform(
                *self.config.friction_range
            )
        if self.config.randomize_motor:
            randomization['motor_gain_scale'] = np.random.uniform(
                *self.config.motor_gain_range
            )
        if self.config.randomize_delay:
            randomization['actuator_delay'] = np.random.randint(
                *self.config.delay_range
            )
        if self.config.randomize_sensor_noise:
            randomization['sensor_noise_scale'] = np.random.uniform(
                *self.config.sensor_noise_range
            )
            
        return randomization
    
    def apply_randomization(self,
                           base_params: Dict[str, np.ndarray],
                           randomization: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Apply randomization to base parameters."""
        randomized = copy.deepcopy(base_params)
        
        if 'mass_scale' in randomization and 'mass' in randomized:
            randomized['mass'] *= randomization['mass_scale']
        if 'inertia_scale' in randomization and 'inertia' in randomized:
            randomized['inertia'] *= randomization['inertia_scale']
        if 'friction_scale' in randomization and 'friction' in randomized:
            randomized['friction'] *= randomization['friction_scale']
        if 'motor_gain_scale' in randomization and 'motor_gain' in randomized:
            randomized['motor_gain'] *= randomization['motor_gain_scale']
            
        return randomized


class DomainRandomizationTuner:
    """
    Tune domain randomization ranges using real robot data.
    
    From Chebotar et al. [5]: Adapt randomization distribution to match
    real-world data distribution, closing the sim-to-real loop.
    """
    
    def __init__(self, config: DomainAdaptationConfig = None):
        self.config = config or DomainAdaptationConfig()
        self.randomizer = DynamicsRandomizer(config)
        self.randomization_history = deque(maxlen=1000)
        
    def quantify_gap(self,
                     sim_trajectories: np.ndarray,
                     real_trajectories: np.ndarray) -> Dict[str, float]:
        """
        Quantify sim-to-real gap using trajectory statistics.
        
        Args:
            sim_trajectories: Simulated state trajectories (N, T, state_dim)
            real_trajectories: Real state trajectories (M, T, state_dim)
        
        Returns:
            Gap metrics
        """
        # Mean trajectory distance
        sim_mean = sim_trajectories.mean(axis=(0, 1))
        real_mean = real_trajectories.mean(axis=(0, 1))
        mean_shift = np.linalg.norm(sim_mean - real_mean)
        
        # Covariance distance (using Fréchet distance approximation)
        sim_std = sim_trajectories.std(axis=(0, 1))
        real_std = real_trajectories.std(axis=(0, 1))
        std_ratio = np.mean(np.abs(np.log(sim_std / (real_std + 1e-8) + 1e-8)))
        
        # Maximum mean discrepancy (simplified)
        sim_flat = sim_trajectories.reshape(-1, sim_trajectories.shape[-1])
        real_flat = real_trajectories.reshape(-1, real_trajectories.shape[-1])
        mmd = self._compute_mmd(sim_flat, real_flat)
        
        return {
            'mean_shift': mean_shift,
            'std_ratio': std_ratio,
            'mmd': mmd,
            'overall_gap': mean_shift + std_ratio + mmd,
        }
    
    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """Compute Maximum Mean Discrepancy with RBF kernel."""
        def rbf_kernel(X, Y, gamma):
            XX = np.sum(X**2, axis=1)[:, None]
            YY = np.sum(Y**2, axis=1)[None, :]
            XY = X @ Y.T
            dists = XX + YY - 2 * XY
            return np.exp(-gamma * dists)
        
        K_xx = rbf_kernel(X, X, gamma)
        K_yy = rbf_kernel(Y, Y, gamma)
        K_xy = rbf_kernel(X, Y, gamma)
        
        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return float(mmd)
    
    def tune_randomization(self,
                          sim_env,
                          real_data: np.ndarray,
                          n_iterations: int = 50) -> Dict:
        """
        Tune randomization ranges to minimize sim-to-real gap.
        
        Uses Bayesian optimization or grid search to find randomization
        parameters that make sim trajectories match real data.
        
        Args:
            sim_env: Simulation environment callable
            real_data: Real trajectory data
            n_iterations: Number of tuning iterations
        
        Returns:
            Best randomization parameters
        """
        best_gap = float('inf')
        best_params = None
        
        for iteration in range(n_iterations):
            # Sample randomization
            rand_params = self.randomizer.sample_randomization()
            
            # Run sim with randomization
            sim_traj = sim_env(rand_params)
            
            # Compute gap
            gap = self.quantify_gap(sim_traj, real_data)
            
            if gap['overall_gap'] < best_gap:
                best_gap = gap['overall_gap']
                best_params = rand_params
                
            self.randomization_history.append((rand_params, gap['overall_gap']))
            
            if self.config.verbose and iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: gap={gap['overall_gap']:.4f}, "
                           f"best={best_gap:.4f}")
                
        return {
            'best_params': best_params,
            'best_gap': best_gap,
            'history': list(self.randomization_history),
        }


class SimRealGapQuantifier:
    """
    Comprehensive quantification of sim-to-real gap.
    
    Measures discrepancies in dynamics, observations, rewards, and
    policy performance between sim and real.
    """
    
    def __init__(self):
        self.metrics = {}
        
    def compute_dynamics_gap(self,
                             sim_transitions: np.ndarray,
                             real_transitions: np.ndarray) -> Dict[str, float]:
        """
        Measure dynamics prediction error gap.
        
        Args:
            sim_transitions: (s, a, s') from sim
            real_transitions: (s, a, s') from real
        """
        sim_states, sim_actions, sim_next = sim_transitions
        real_states, real_actions, real_next = real_transitions
        
        # Fit dynamics models
        from sklearn.linear_model import Ridge
        
        sim_model = Ridge(alpha=0.1).fit(
            np.hstack([sim_states, sim_actions]),
            sim_next - sim_states
        )
        real_model = Ridge(alpha=0.1).fit(
            np.hstack([real_states, real_actions]),
            real_next - real_states
        )
        
        # Cross-prediction errors
        sim_on_real = sim_model.predict(np.hstack([real_states, real_actions]))
        real_on_sim = real_model.predict(np.hstack([sim_states, sim_actions]))
        
        sim_error = np.mean((sim_on_real - (real_next - real_states))**2)
        real_error = np.mean((real_on_sim - (sim_next - sim_states))**2)
        
        return {
            'sim_on_real_error': sim_error,
            'real_on_sim_error': real_error,
            'dynamics_gap': abs(sim_error - real_error),
        }
    
    def compute_policy_gap(self,
                           sim_policy,
                           real_policy,
                           test_states: np.ndarray) -> Dict[str, float]:
        """Measure policy output discrepancy."""
        with torch.no_grad():
            sim_actions = sim_policy(torch.FloatTensor(test_states)).numpy()
            real_actions = real_policy(torch.FloatTensor(test_states)).numpy()
            
        action_mse = np.mean((sim_actions - real_actions)**2)
        action_corr = np.corrcoef(sim_actions.flatten(), real_actions.flatten())[0, 1]
        
        return {
            'action_mse': action_mse,
            'action_correlation': action_corr,
            'policy_gap': action_mse,
        }
    
    def compute_all_gaps(self,
                         sim_data: Dict,
                         real_data: Dict) -> Dict[str, Dict[str, float]]:
        """Compute all gap metrics."""
        gaps = {}
        
        if 'transitions' in sim_data and 'transitions' in real_data:
            gaps['dynamics'] = self.compute_dynamics_gap(
                sim_data['transitions'], real_data['transitions']
            )
            
        if 'trajectories' in sim_data and 'trajectories' in real_data:
            tuner = DomainRandomizationTuner()
            gaps['trajectory'] = tuner.quantify_gap(
                sim_data['trajectories'], real_data['trajectories']
            )
            
        if 'policy' in sim_data and 'policy' in real_data and 'test_states' in sim_data:
            gaps['policy'] = self.compute_policy_gap(
                sim_data['policy'], real_data['policy'], sim_data['test_states']
            )
            
        self.metrics = gaps
        return gaps
