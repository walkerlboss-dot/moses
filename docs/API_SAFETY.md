# Safety API — Moses

> **Defense-in-depth safety system for autonomous humanoid robots.**

---

## BoundsChecker

```python
from moses.safety.bounds_checker import BoundsChecker
```

Hard limits on compute, code changes, and performance.

### Constructor

```python
BoundsChecker(
    max_gpu_hours_per_day: float = 24.0,
    max_disk_usage_gb: float = 1000.0,
    max_code_changes_per_day: int = 100,
    max_files_touched_per_day: int = 20,
    performance_floor: float = 0.95,  # 95% of baseline
)
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `check_compute()` | `gpu_hours` | `bool` | Within compute budget? |
| `check_disk()` | `usage_gb` | `bool` | Within disk limit? |
| `check_code_changes()` | `n_changes` | `bool` | Within change limit? |
| `check_performance()` | `metric`, `baseline` | `bool` | Above performance floor? |
| `enforce()` | — | `bool` | Run all checks, return pass/fail |

### Auto-Shutdown

```python
checker = BoundsChecker(max_gpu_hours_per_day=24.0)

if not checker.enforce():
    logger.critical("Bounds exceeded! Initiating safe shutdown.")
    safe_shutdown()
```

---

## ApprovalGates

```python
from moses.safety.approval_gates import ApprovalGates
```

4-tier human oversight system.

### Tiers

| Tier | Scope | Auto-Approve | Human Notification |
|------|-------|--------------|-------------------|
| **T1** | Hyperparameter changes | ✅ Yes | Daily digest |
| **T2** | Code mutations | ✅ Yes | Immediate alert |
| **T3** | Core file changes | ❌ No | Telegram + wait for approval |
| **T4** | Self-architecture changes | ❌ No | Telegram + 24h wait |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `request_approval()` | `tier`, `change`, `impact` | `bool` | Request human approval |
| `approve()` | `request_id` | — | Approve change |
| `reject()` | `request_id`, `reason` | — | Reject change |
| `get_pending()` | — | `list` | List pending requests |

### Example

```python
gates = ApprovalGates()

# T1: Auto-approved
result = gates.request_approval(
    tier=1,
    change="learning_rate: 0.001 → 0.002",
    impact="Expected 2% improvement",
)
# Returns: True (auto-approved)

# T3: Requires human approval
result = gates.request_approval(
    tier=3,
    change="Modify training pipeline core logic",
    impact="Could affect all training jobs",
)
# Sends Telegram to Alex, waits for response
```

---

## DriftDetector (Safety)

```python
from moses.safety.drift_detector import SafetyDriftDetector
```

Detects undesirable drift in code quality.

### Metrics

| Metric | Threshold | Alert |
|--------|-----------|-------|
| Code complexity | >15 cyclomatic | Warning |
| Test coverage | <80% | Critical |
| Docstring coverage | <90% | Warning |
| File size | >500 lines | Warning |
| Circular dependencies | >0 | Critical |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `check_complexity()` | `file_path` | `dict` | Cyclomatic complexity |
| `check_coverage()` | — | `dict` | Test coverage |
| `check_docstrings()` | — | `dict` | Docstring coverage |
| `check_bloat()` | — | `dict` | File size trends |
| `run_all_checks()` | — | `dict` | Full drift report |

---

## IntegrityChecker

```python
from moses.safety.integrity_checker import IntegrityChecker
```

Verifies system integrity.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `verify_checksums()` | — | `bool` | Core file checksums |
| `validate_imports()` | — | `dict` | All imports resolve |
| `check_circular_deps()` | — | `list` | Circular dependencies |
| `verify_permissions()` | — | `dict` | File permissions |
| `full_check()` | — | `dict` | Complete integrity report |

---

*Safety is not an afterthought. It is a constraint that all other systems must satisfy.*
