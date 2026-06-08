# Experiments API — Moses

> **Automated experimentation, search spaces, and budget management.**

---

## ExperimentRunner

```python
from moses.experiments.runner import ExperimentRunner
```

Automated experiment orchestration.

### Constructor

```python
ExperimentRunner(
    config: ExperimentConfig,
    search_space: SearchSpace,
    budget: BudgetManager,
)
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `run_experiment()` | `name` | `ExperimentResult` | Single experiment |
| `run_experiments_parallel()` | `names` | `list` | Parallel experiments |
| `compare()` | `exp_a`, `exp_b` | `dict` | Statistical comparison |
| `promote_winner()` | `experiment` | `bool` | Promote to production |

### Experiment Config

```python
@dataclass
class ExperimentConfig:
    algorithm: str = "PPO"
    search_space: SearchSpace = PPOSearchSpace()
    budget: BudgetManager = BudgetManager()
    sampler: str = "TPE"
    pruner: str = "Median"
    auto_promote: bool = True
    promotion_threshold: float = 0.05  # 5% improvement
```

### Example

```python
runner = ExperimentRunner(
    config=ExperimentConfig(algorithm="PPO"),
    search_space=PPOSearchSpace(),
    budget=BudgetManager(max_gpu_hours=100),
)

# Run experiment
result = runner.run_experiment("ppo_walk_v2")

# Compare with baseline
comparison = runner.compare(result, baseline_experiment)
# Returns: {"winner": "v2", "p_value": 0.003, "effect_size": 0.45}

# Auto-promote if significant
if comparison["winner"] == "v2":
    runner.promote_winner(result)
```

---

## SearchSpace

```python
from moses.experiments.search_space import SearchSpace
```

Composable search space definitions.

### Pre-built Spaces

| Space | Parameters | Count |
|-------|-----------|-------|
| `PPOSearchSpace` | lr, clip, entropy, gae, gamma, etc. | 10 |
| `SACSearchSpace` | actor_lr, critic_lr, tau, buffer, etc. | 10 |
| `GR00TSearchSpace` | vision_lr, n_layers, n_heads, etc. | 10 |
| `ArchitectureSearchSpace` | layers, widths, activations, etc. | 6 |
| `EnvironmentSearchSpace` | gravity, mass, friction, etc. | 9 |

### Composable

```python
from moses.experiments.search_space import ComposableSearchSpace

# Combine spaces
space = ComposableSearchSpace()
space.add_space(PPOSearchSpace())
space.add_space(EnvironmentSearchSpace())
space.remove_space("SACSearchSpace")

# 19 parameters total
```

---

## BudgetManager

```python
from moses.experiments.budget import BudgetManager
```

Compute and experiment budget management.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `allocate()` | `experiment`, `gpu_hours` | `bool` | Allocate budget |
| `track_usage()` | `experiment`, `hours` | — | Track usage |
| `check_budget()` | `experiment` | `bool` | Within budget? |
| `get_report()` | — | `dict` | Budget report |

### GPU Cost Rates

| GPU | $/hour | TFLOPS |
|-----|--------|--------|
| A100 | $2.50 | 312 |
| H100 | $4.00 | 989 |
| V100 | $1.50 | 125 |
| T4 | $0.35 | 65 |
| A10 | $0.60 | 125 |

### Example

```python
budget = BudgetManager(
    max_gpu_hours_per_day=24,
    max_cost_per_week=500,
)

# Allocate
if budget.allocate("ppo_experiment", gpu_hours=10):
    run_experiment()
    budget.track_usage("ppo_experiment", hours=8.5)
else:
    logger.warning("Budget exceeded. Experiment queued.")
```

---

*The best experiment is the one you don't have to run.*
