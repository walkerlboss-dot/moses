# Self-Modification API — Moses

> **Safe code mutation, A/B testing, rollback, and evolutionary improvement.**

---

## CodeMutator

```python
from moses.self_modify.code_mutator import CodeMutator
```

Safe code mutation via AST parsing.

### Constructor

```python
CodeMutator(
    whitelist: list[str] = None,  # Allowed mutation types
    blacklist: list[str] = None,  # Forbidden patterns
)
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `mutate()` | `source_code`, `n_mutations` | `str` | Apply safe mutations |
| `generate_diff()` | `original`, `mutated` | `str` | Unified diff |
| `validate()` | `code` | `bool` | Check mutation safety |

### Safe Mutations

| Type | Description | Example |
|------|-------------|---------|
| `constant_change` | Change numeric constants | `lr=0.001` → `lr=0.002` |
| `swap_lines` | Swap independent statements | Reorder assignments |
| `add_early_return` | Add guard clause | `if x < 0: return 0` |
| `rename_variable` | Rename local variable | `x` → `value` |
| `extract_method` | Extract code block | Inline → function |
| `inline_method` | Inline function call | Function → inline |

### Forbidden Patterns

- Import statements
- Class definitions
- Safety-critical code
- Core infrastructure

### Example

```python
mutator = CodeMutator(whitelist=["constant_change", "swap_lines"])

original = """
def train_step(lr=0.001):
    loss = compute_loss()
    grad = compute_grad(loss)
    return grad * lr
"""

mutated = mutator.mutate(original, n_mutations=3)
diff = mutator.generate_diff(original, mutated)
print(diff)
# Shows: lr=0.001 → lr=0.002, line swaps, etc.
```

---

## ABTester

```python
from moses.self_modify.ab_tester import ABTester
```

A/B testing with statistical significance.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `run_experiment()` | `variant_a`, `variant_b`, `metric_fn` | `dict` | Run A/B test |
| `compare()` | `results_a`, `results_b` | `dict` | Statistical comparison |
| `is_significant()` | `p_value`, `alpha` | `bool` | Significance check |

### Statistical Tests

| Test | Use Case | Threshold |
|------|----------|-----------|
| Welch's t-test | Mean comparison | p < 0.05 |
| Bootstrap CI | Confidence interval | 95% CI |
| Bootstrap p-value | Non-parametric | p < 0.05 |
| Cohen's d | Effect size | d > 0.2 |

### Example

```python
tester = ABTester(n_samples=1000)

results = tester.run_experiment(
    variant_a=baseline_policy,
    variant_b=mutated_policy,
    metric_fn=evaluate_reward,
)

# Returns:
# {
#   "winner": "b",
#   "p_value": 0.003,
#   "effect_size": 0.45,
#   "confidence": 0.99,
#   "recommendation": "promote_variant_b"
# }
```

---

## RollbackManager

```python
from moses.self_modify.rollback import RollbackManager
```

Dual-layer versioning: file snapshots + git commits.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `snapshot()` | `files` | `str` | Create snapshot |
| `commit()` | `message` | `str` | Git commit |
| `rollback()` | `steps=1` | `bool` | Rollback to previous |
| `restore()` | `snapshot_id` | `bool` | Restore specific snapshot |
| `history()` | — | `list` | Full change history |

### Auto-Rollback Decorator

```python
@auto_rollback(metric="success_rate", threshold=0.95)
def deploy_new_policy(policy):
    # If success_rate drops below 95%, auto-rollback
    robot.load_policy(policy)
```

---

## EvolutionEngine

```python
from moses.self_modify.evolution import EvolutionEngine
```

Genetic algorithm for code evolution.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `evolve()` | `generations`, `population_size` | `dict` | Run evolution |
| `evaluate_fitness()` | `code` | `float` | Fitness function |
| `select_parents()` | `population` | `list` | Tournament selection |
| `crossover()` | `parent_a`, `parent_b` | `str` | AST-aware crossover |
| `mutate()` | `code` | `str` | Apply mutation |

### Fitness Function

```python
def fitness(code):
    # Compile and test
    results = run_tests(code)
    if results.failed > 0:
        return 0.0
    
    # Performance metrics
    speed = benchmark_speed(code)
    accuracy = benchmark_accuracy(code)
    
    return 0.4 * speed + 0.6 * accuracy
```

### Example

```python
engine = EvolutionEngine(
    population_size=20,
    generations=10,
    mutation_rate=0.1,
    crossover_rate=0.7,
    elitism=2,
)

result = engine.evolve(
    seed_code=baseline_policy_code,
    fitness_fn=fitness,
)

# Returns: {"best_code": "...", "best_fitness": 0.95, "generations": 10}
```

---

*Safety: All mutations are reversible. Core infrastructure is protected.*
