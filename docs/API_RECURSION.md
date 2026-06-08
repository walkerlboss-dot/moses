# Recursion API — Moses

> **Meta-meta-learning, self-healing, predictive world models, and training outcome predictors.**

---

## MetaMetaLearner

```python
from moses.recursion.meta_meta_learning import MetaMetaLearner
```

Learns how to design meta-learning strategies.

### Architecture

Three-level hierarchy:
1. **Task level:** Learn specific task
2. **Meta level:** Learn how to learn tasks
3. **Meta-meta level:** Learn how to design meta-learning algorithms

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `discover_algorithm()` | `task_distribution` | `AlgorithmConfig` | Discover new learning algorithm |
| `evaluate_algorithm()` | `algorithm`, `tasks` | `float` | Evaluate on task distribution |
| `meta_meta_train()` | `meta_tasks`, `epochs` | — | Train meta-meta learner |

### Algorithm Discovery Cell

Uses DARTS-style primitive composition:
- Primitives: SGD, Adam, RMSprop, momentum, etc.
- Combinations: Weighted sum of primitives
- Search: Differentiable architecture search

### Example

```python
learner = MetaMetaLearner()

# Discover new optimizer for humanoid tasks
algorithm = learner.discover_algorithm(
    task_distribution=humanoid_tasks,
)

# Evaluate
score = learner.evaluate_algorithm(algorithm, test_tasks)
# Returns: 0.92 (92% average success rate)
```

---

## SelfHealing

```python
from moses.recursion.self_healing import SelfHealing
```

Auto-detects and repairs system failures.

### Anomaly Detection

| Anomaly | Detection | Auto-Repair |
|---------|-----------|-------------|
| NaN/Inf loss | `torch.isnan()` | Gradient clipping |
| Gradient explosion | Norm > 100 | Clip + reduce LR |
| Gradient vanishing | Norm < 1e-6 | Increase LR, skip connection |
| Loss divergence | Increase > 50% | Rollback checkpoint |
| Dead neurons | Zero activation | Re-initialize layer |
| Memory leak | RAM growth > 10%/hr | Restart process |
| Training stall | No progress > 1hr | New seed, adjust batch |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `monitor()` | `training_state` | `dict` | Check health |
| `diagnose()` | `anomaly` | `str` | Root cause analysis |
| `repair()` | `diagnosis` | `bool` | Apply fix |
| `verify()` | — | `bool` | Confirm fix worked |

### Bayesian Root Cause Analysis

```python
# Likelihood table: anomaly → cause
likelihood = {
    "nan_loss": {"lr_too_high": 0.6, "bad_data": 0.3, "bug": 0.1},
    "grad_explosion": {"lr_too_high": 0.7, "bad_init": 0.2, "arch_issue": 0.1},
}

# Given anomaly, infer most likely cause
cause = self.diagnose(anomaly="nan_loss")
# Returns: "lr_too_high" (60% probability)
```

---

## WorldModel

```python
from moses.recursion.world_model import WorldModel
```

Predicts environment dynamics for planning.

### Variants

| Variant | Description | Use Case |
|---------|-------------|----------|
| **Deterministic** | Single next-state prediction | Fast planning |
| **Ensemble** | Multiple models + disagreement | Uncertainty estimation |
| **VAE** | Latent space dynamics | Complex environments |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `predict()` | `state`, `action` | `next_state` | Predict next state |
| `predict_uncertainty()` | `state`, `action` | `mean`, `std` | Predict with uncertainty |
| `plan()` | `goal`, `horizon` | `actions` | MPC planning |
| `train()` | `transitions` | — | Train on experience |

### MPC Planning (CEM)

```python
model = WorldModel(variant="ensemble")

# Plan action sequence
actions = model.plan(
    goal=target_position,
    horizon=16,
    n_samples=1000,
    n_elites=100,
    n_iterations=5,
)

# Execute first action, re-plan
robot.execute(actions[0])
```

---

## Predictor

```python
from moses.recursion.predictor import Predictor
```

Predicts training outcomes before running experiments.

### Surrogate Models

| Model | Type | Use Case |
|-------|------|----------|
| **Neural Ensemble** | Deep learning | Fast prediction |
| **Gaussian Process** | Bayesian | Small data, uncertainty |

### Acquisition Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| **EI** | Expected Improvement | Exploitation |
| **UCB** | Upper Confidence Bound | Exploration |
| **PI** | Probability of Improvement | Balanced |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `fit()` | `historical_results` | — | Train surrogate |
| `predict()` | `config` | `mean`, `std` | Predict outcome |
| `should_run()` | `config` | `bool` | Worth running? |
| `suggest_next()` | `n` | `list` | Suggest configs to try |

### Example

```python
predictor = Predictor(model_type="neural_ensemble")

# Train on historical data
predictor.fit(past_experiments)

# Predict new config
mean, std = predictor.predict(new_config)
# Returns: mean=45.2, std=3.1 (predicted reward)

# Filter unpromising configs
if predictor.should_run(new_config):
    run_experiment(new_config)
else:
    logger.info("Skipping unpromising config.")
```

---

*Recursive improvement: the system that improves the system that improves the system.*
