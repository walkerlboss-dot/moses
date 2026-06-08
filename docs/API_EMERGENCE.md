# Emergence API — Moses

> **Swarm intelligence, behavior arbitration, curiosity-driven exploration, and consciousness simulation.**

---

## SwarmIntelligence

```python
from moses.emergence.swarm_intelligence import SwarmIntelligence
```

Population-based collective learning.

### Constructor

```python
SwarmIntelligence(
    population_size: int = 10,
    communication_interval: int = 100,
    emergence_threshold: float = 0.15,  # 15% improvement
)
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `add_agent()` | `agent_config` | `str` | Add agent to swarm |
| `share_experience()` | `agent_id`, `experience` | — | Share experience |
| `vote()` | `topic` | `dict` | Collective decision |
| `detect_emergence()` | — | `list` | Detect emergent strategies |
| `get_consensus()` | `topic` | `dict` | Confidence-weighted vote |

### Experience Packets

```python
@dataclass
class ExperiencePacket:
    agent_id: str
    hyperparameters: dict
    architecture: dict
    performance: float
    task: str
    confidence: float  # 0-1
```

### Emergence Detection

Emergent strategy detected when:
- ≥3 agents independently achieve >15% improvement
- Strategy not present in initial population
- Reproducible across tasks

### Example

```python
swarm = SwarmIntelligence(population_size=10)

# Add agents with diverse configs
for i in range(10):
    swarm.add_agent({
        "learning_rate": 10 ** np.random.uniform(-4, -2),
        "entropy_coef": np.random.uniform(0.001, 0.1),
    })

# Run training
for step in range(10000):
    for agent in swarm.agents:
        experience = agent.train()
        swarm.share_experience(agent.id, experience)
    
    # Detect emergent strategies
    emergent = swarm.detect_emergence()
    if emergent:
        logger.info(f"Emergent strategy detected: {emergent}")
```

---

## BehaviorArbitrator

```python
from moses.emergence.behavior_arbitration import BehaviorArbitrator
```

Arbitrates between multiple competing controllers.

### Controllers

| Controller | Type | Strengths | Weaknesses |
|------------|------|-----------|------------|
| **GR00T** | Foundation model | Generalization | Compute cost |
| **PPO** | RL policy | Speed | Narrow tasks |
| **MPC** | Model-based | Safety | Planning delay |
| **Heuristic** | Rule-based | Reliability | Limited tasks |

### Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Winner-take-all** | Best controller executes | Clear winner |
| **Soft-blend** | Weighted average | Uncertain |
| **Hierarchical** | High-level plans, low-level executes | Complex tasks |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `register_controller()` | `name`, `controller`, `confidence_fn` | — | Add controller |
| `select_controller()` | `state`, `task` | `str` | Choose controller |
| `blend_actions()` | `actions`, `weights` | `action` | Weighted blend |
| `get_confidence()` | `controller`, `state` | `float` | Controller confidence |

### Safety Override

```python
# If stability < 0.5, override with MPC (safest)
if stability < 0.5:
    arbitrator.override("MPC")
```

---

## CuriosityEngine

```python
from moses.emergence.curiosity import CuriosityEngine
```

Intrinsic motivation for exploration.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `compute_reward()` | `state`, `next_state`, `action` | `float` | Curiosity reward |
| `update()` | `transition` | — | Update curiosity model |
| `get_novelty()` | `state` | `float` | State novelty score |

### Curiosity Types

| Type | Mechanism | Use Case |
|------|-----------|----------|
| **ICM** | Forward model prediction error | General exploration |
| **RND** | Random network distillation | Episodic novelty |
| **Episodic** | Count-based + embedding | Local exploration |

### Example

```python
curiosity = CuriosityEngine(methods=["icm", "rnd"])

for episode in range(1000):
    state = env.reset()
    
    for step in range(1000):
        action = policy.get_action(state)
        next_state, extrinsic_reward, done, _ = env.step(action)
        
        # Add curiosity bonus
        intrinsic_reward = curiosity.compute_reward(state, next_state, action)
        total_reward = extrinsic_reward + 0.1 * intrinsic_reward
        
        policy.update(state, action, total_reward)
        curiosity.update((state, action, next_state))
        
        state = next_state
```

---

## ConsciousnessSim

```python
from moses.emergence.consciousness_sim import ConsciousnessSim
```

Lightweight consciousness simulation for better decision-making.

### Components

| Component | Description |
|-----------|-------------|
| **Self-model** | Agent's model of itself |
| **Introspection** | Monitoring internal states |
| **Goal hierarchy** | High-level → mid-level → low-level |
| **Attention** | Focus on relevant inputs |

### States

| State | Trigger | Action |
|-------|---------|--------|
| **Focused** | Clear goal, low distraction | Execute plan |
| **Curious** | Novel state detected | Explore |
| **Stressed** | Multiple conflicting goals | Replan |
| **Bored** | Repetitive tasks | Seek novelty |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `update_self_model()` | `observation`, `action`, `outcome` | — | Update self-model |
| `introspect()` | — | `dict` | Current internal state |
| `set_goal()` | `goal`, `priority` | — | Set high-level goal |
| `get_attention()` | `inputs` | `weights` | Attention weights |
| `replan()` | — | `plan` | Replan based on state |

### Example

```python
consciousness = ConsciousnessSim()

# Set high-level goal
consciousness.set_goal("walk to door", priority=1.0)

# During execution
for step in range(1000):
    obs = robot.get_observation()
    
    # Introspect
    state = consciousness.introspect()
    if state["stress"] > 0.7:
        plan = consciousness.replan()
        logger.info("Replanning due to stress.")
    
    # Attention
    attention = consciousness.get_attention(obs)
    focused_obs = obs * attention
    
    # Act
    action = policy.get_action(focused_obs)
    robot.execute(action)
    
    # Update self-model
    consciousness.update_self_model(obs, action, robot.get_outcome())
```

---

*Emergence: complex behaviors from simple rules. Not magic — just math.*
