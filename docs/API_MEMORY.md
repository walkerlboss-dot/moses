# Memory API — Moses

> **Persistent knowledge accumulation, causal reasoning, and transfer learning.**

---

## ExperienceStore

```python
from moses.memory.experience_store import ExperienceStore
```

SQLite-backed experience replay with vector embeddings.

### Constructor

```python
ExperienceStore(
    db_path: str = "~/.moses/experiences.db",
    embedding_dim: int = 128,
)
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `store()` | `config`, `result`, `metadata` | `str` | Store experiment |
| `query()` | `config`, `k=5` | `list` | Find similar experiments |
| `get_best()` | `task`, `metric` | `dict` | Best config for task |
| `get_similar_tasks()` | `task` | `list` | Similar tasks |

### Query Examples

```python
store = ExperienceStore()

# Store experiment
store.store(
    config={"lr": 0.001, "batch_size": 256},
    result={"reward": 45.2, "success_rate": 0.92},
    metadata={"task": "humanoid_walk", "date": "2026-06-08"},
)

# Query similar configs
similar = store.query({"lr": 0.001}, k=5)
# Returns: [{"config": {...}, "result": {...}, "similarity": 0.95}, ...]

# Get best config for task
best = store.get_best("humanoid_walk", metric="reward")
# Returns: {"config": {...}, "result": {...}}
```

---

## CausalReasoner

```python
from moses.memory.causal_reasoning import CausalReasoner
```

Estimates causal effects of code changes.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `estimate_effect()` | `change`, `outcome` | `dict` | Causal effect estimate |
| `difference_in_differences()` | `treatment`, `control` | `float` | DiD estimate |
| `counterfactual()` | `config`, `intervention` | `float` | Counterfactual prediction |

### Example

```python
reasoner = CausalReasoner()

# Did increasing learning rate cause improvement?
effect = reasoner.estimate_effect(
    change="lr: 0.001 → 0.002",
    outcome="reward: 42 → 45",
)
# Returns: {"effect": 3.0, "confidence": 0.85, "method": "DiD"}

# What if we had used lr=0.003?
counterfactual = reasoner.counterfactual(
    config={"lr": 0.002},
    intervention={"lr": 0.003},
)
# Returns: 47.5 (predicted reward)
```

---

## KnowledgeGraph

```python
from moses.memory.knowledge_graph import KnowledgeGraph
```

Property graph for structured robotics knowledge.

### Schema

| Node Type | Properties |
|-----------|-----------|
| `Algorithm` | name, type, hyperparameters |
| `Task` | name, difficulty, success_rate |
| `Hyperparameter` | name, range, optimal_value |
| `Robot` | name, dof, mass, height |

| Edge Type | Meaning |
|-----------|---------|
| `IMPROVES` | Algorithm → Task |
| `DEGRADES` | Algorithm → Task |
| `COMPATIBLE_WITH` | Algorithm → Algorithm |
| `REQUIRES` | Task → Robot |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `add_node()` | `type`, `properties` | `str` | Add node |
| `add_edge()` | `from`, `to`, `type` | — | Add edge |
| `query()` | `pattern` | `list` | Pattern match |
| `traverse()` | `start`, `depth` | `list` | BFS traversal |

### Example

```python
kg = KnowledgeGraph()

# Add knowledge
kg.add_node("Algorithm", {"name": "PPO", "type": "RL"})
kg.add_node("Task", {"name": "humanoid_walk", "difficulty": 0.7})
kg.add_edge("PPO", "humanoid_walk", "IMPROVES")

# Query
results = kg.query("What learning rate works best for humanoid_walk?")
# Returns: [{"hyperparameter": "lr", "optimal": 0.0003, "confidence": 0.92}]
```

---

## TransferLearner

```python
from moses.memory.transfer_learning import TransferLearner
```

Extracts general principles for cross-task transfer.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `extract_principles()` | `source_task`, `target_task` | `list` | Transferable principles |
| `adapt_hyperparameters()` | `source`, `target` | `dict` | Adapted hyperparameters |
| `measure_transferability()` | `source`, `target` | `float` | Transfer score |

### Example

```python
transfer = TransferLearner()

# Extract principles from walking to running
principles = transfer.extract_principles("humanoid_walk", "humanoid_run")
# Returns: [
#   {"principle": "high_lr_for_fast_tasks", "confidence": 0.85},
#   {"principle": "larger_batch_for_dynamics", "confidence": 0.72},
# ]

# Adapt hyperparameters
adapted = transfer.adapt_hyperparameters(
    source={"lr": 0.0003, "batch_size": 4096},
    target="humanoid_run",
)
# Returns: {"lr": 0.0005, "batch_size": 8192}
```

---

*Knowledge is power. Persistent knowledge is perpetual power.*
