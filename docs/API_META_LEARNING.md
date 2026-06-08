# Meta-Learning API — Moses

> **Automated hyperparameter optimization, neural architecture search, and curriculum learning.**

---

## HyperparameterSearch

```python
from moses.meta_learning.hyperparameter_search import HyperparameterSearch
```

Optuna-based hyperparameter optimization.

### Constructor

```python
HyperparameterSearch(
    search_space: SearchSpace,
    objective: callable,
    n_trials: int = 100,
    sampler: str = "TPE",
    pruner: str = "Median",
    storage: str = "sqlite:///optuna.db",
)
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `search()` | — | `dict` | Run optimization, return best config |
| `get_best_config()` | — | `dict` | Get best configuration |
| `plot_history()` | — | `Figure` | Optimization history |
| `save_study()` | `path` | — | Save study for resumption |

### Search Spaces

| Algorithm | Parameters | Range |
|-----------|-----------|-------|
| PPO | learning_rate | 1e-5 to 1e-2 |
| | clip_epsilon | 0.1 to 0.3 |
| | entropy_coef | 0.001 to 0.1 |
| | gae_lambda | 0.9 to 0.99 |
| SAC | actor_lr | 1e-5 to 1e-2 |
| | critic_lr | 1e-5 to 1e-2 |
| | tau | 0.001 to 0.1 |
| | buffer_size | 1e5 to 1e7 |

### Example

```python
search = HyperparameterSearch(
    search_space=PPOSearchSpace(),
    objective=train_and_evaluate,
    n_trials=100,
)

best = search.search()
# Returns: {"learning_rate": 0.0003, "clip_epsilon": 0.2, ...}
```

---

## NeuralArchitectureSearch

```python
from moses.meta_learning.neural_architecture_search import NeuralArchitectureSearch
```

Efficient NAS with supernet weight sharing.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `search()` | `n_candidates`, `eval_fn` | `ArchitectureConfig` | Best architecture |
| `train_supernet()` | `epochs` | — | Train weight-sharing supernet |
| `evaluate_architecture()` | `arch` | `float` | Evaluate single architecture |

### Architecture Space

| Component | Options |
|-----------|---------|
| Network type | MLP, Transformer, LSTM, GRU |
| Layers | 2-8 |
| Width | 128-1024 |
| Activation | ReLU, ELU, GELU, Swish |
| Skip connections | True/False |
| Layer norm | True/False |
| Dropout | 0.0-0.5 |

---

## CurriculumScheduler

```python
from moses.meta_learning.curriculum_learning import CurriculumScheduler
```

Adaptive curriculum with promote/regress logic.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `update()` | `success_rate` | `str` | "promote", "regress", or None |
| `get_difficulty()` | — | `dict` | Current difficulty parameters |
| `get_stats()` | — | `dict` | Curriculum statistics |

### Difficulty Parameters

| Parameter | Easy | Hard | Control |
|-----------|------|------|---------|
| Slope | 0° | 15° | Sliding window |
| Obstacle density | 0% | 30% | EMA smoothing |
| Speed | 0.5 m/s | 2.0 m/s | Threshold + delta |
| Roughness | 0.0 | 0.1 | Cooldown guard |
| Step height | 0 cm | 15 cm | Non-linear easing |

### Example

```python
scheduler = CurriculumScheduler(
    success_threshold=0.8,
    regression_threshold=0.5,
    cooldown_steps=1000,
)

for episode in range(10000):
    success = train_episode(difficulty=scheduler.get_difficulty())
    action = scheduler.update(success)
    
    if action == "promote":
        logger.info("Promoting to next difficulty level!")
    elif action == "regress":
        logger.warning("Regressing to easier level.")
```

---

## RewardShaper

```python
from moses.meta_learning.reward_shaping import RewardShaper
```

Evolutionary reward design with anti-hacking detection.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `evolve()` | `n_generations` | `dict` | Evolve reward weights |
| `diagnose()` | `episode` | `dict` | Hacking risk assessment |
| `get_reward()` | `obs`, `action`, `info` | `float` | Compute shaped reward |

### Anti-Hacking Detection

| Check | Threshold | Penalty |
|-------|-----------|---------|
| Proxy mismatch | Return ↑, Success ↓ | -0.5 fitness |
| Exploit detection | Z-score > 3 | -0.3 fitness |
| Correlation collapse | Return/step > 10× | -0.4 fitness |

---

*See `docs/BENCHMARKS.md` for meta-learning performance.*
