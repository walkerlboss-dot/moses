# Deployment API — Moses

> **Safe deployment for physical humanoid robots.**

---

## SafeDeploy

```python
from moses.deploy.safe_deploy import SafeDeploy
```

Staged rollout with A/B testing and auto-rollback.

### Constructor

```python
SafeDeploy(
    config: DeploymentConfig,
    robot_interface: RobotInterface,
    metrics_client: MetricsClient,
)
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `staged_rollout()` | `new_policy`, `stages`, `metrics` | `dict` | Deploy with staged rollout |
| `ab_test()` | `policy_a`, `policy_b`, `duration` | `dict` | A/B test two policies |
| `rollback()` | `steps=1` | `bool` | Rollback to previous policy |
| `get_status()` | — | `dict` | Current deployment status |

### Staged Rollout

```python
deploy = SafeDeploy(config, robot, metrics)

result = deploy.staged_rollout(
    new_policy="checkpoints/v2.pt",
    stages=[0.01, 0.10, 0.50, 1.00],  # 1% → 10% → 50% → 100%
    metrics=["success_rate", "energy_efficiency", "stability"],
    min_duration_per_stage=3600,  # 1 hour minimum
    rollback_threshold=0.95,  # Rollback if <95% of baseline
)
```

### A/B Testing

```python
result = deploy.ab_test(
    policy_a="checkpoints/baseline.pt",
    policy_b="checkpoints/v2.pt",
    duration=7200,  # 2 hours
    metrics=["success_rate", "episode_length"],
)
# Returns: {"winner": "b", "confidence": 0.99, "effect_size": 0.15}
```

---

## ShadowMode

```python
from moses.deploy.shadow import ShadowMode
```

Runs new policy in parallel without executing actions.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `enable()` | `new_policy` | — | Start shadow mode |
| `compare()` | — | `dict` | Compare shadow vs active |
| `divergence_detected()` | `threshold` | `bool` | Check for divergence |
| `promote()` | — | — | Promote shadow to active |

### Example

```python
shadow = ShadowMode(robot, metrics)
shadow.enable("checkpoints/v2.pt")

# Run for 1 hour
for _ in range(3600):
    active_action = robot.get_action()
    shadow_action = shadow.get_action()
    
    if shadow.divergence_detected(threshold=0.1):
        logger.warning("Divergence detected! Aborting shadow.")
        shadow.disable()
        break
    
    robot.execute(active_action)  # Only active policy executes
```

---

## ValidationPipeline

```python
from moses.deploy.validation import ValidationPipeline
```

Pre-deployment validation: sim → HITL → checklist.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `validate_sim()` | `policy`, `env` | `dict` | Validate in simulation |
| `validate_hitl()` | `policy`, `robot` | `dict` | Human-in-the-loop test |
| `run_checklist()` | — | `dict` | Pre-flight checklist |
| `get_validation_report()` | — | `dict` | Full validation report |

### Validation Stages

1. **Simulation:** 100 episodes, all metrics green
2. **HITL:** Human operator approves, torque clamps active
3. **Checklist:** Joint limits, sensors, e-stop, battery

---

*Safety invariants: Never deploy untested. Never jump to 100%. Rollback in one command.*
