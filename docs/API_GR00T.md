# GR00T API — Moses

> **NVIDIA Isaac GR00T integration for humanoid robot control.**

---

## Gr00TAdapter

```python
from moses.gr00t.adapter import Gr00TAdapter
```

Bridges Moses observation/action spaces with NVIDIA GR00T N1.7 policies.

### Constructor

```python
Gr00TAdapter(
    model_path: str,           # HuggingFace model ID or local path
    embodiment_tag: str,       # e.g. "UNITREE_G1_SONIC"
    device: str | int,         # "cuda:0", "cpu", or GPU index
    *,
    strict: bool = True,       # Validate obs/action shapes
    camera_key_map: dict | None = None,  # Moses→GR00T camera mapping
    state_key_map: dict | None = None,   # Moses→GR00T state mapping
    language_key: str = "annotation.human.task_description",
    action_horizon: int = 8,   # 1-16, default 8 for deployment
)
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `moses_obs_to_gr00t()` | `moses_obs`, `task_text`, `batch_size=1` | `dict` | Convert Moses obs to GR00T format |
| `gr00t_action_to_moses()` | `gr00t_action` | `dict` | Convert GR00T action to Moses format |
| `get_action()` | `moses_obs`, `task_text` | `dict` | End-to-end: obs → action |
| `reset()` | — | `dict` | Reset policy state between episodes |
| `prepare_finetune_dataset()` | `rollout_buffer`, `output_dir` | `Path` | Convert rollouts to LeRobot v2 format |

### Example

```python
adapter = Gr00TAdapter(
    model_path="nvidia/GR00T-N1.7-3B",
    embodiment_tag="UNITREE_G1_SONIC",
    device="cuda:0",
)

# Single-step inference
action = adapter.get_action(obs, task_text="walk forward")

# Batch inference
gr00t_obs = adapter.moses_obs_to_gr00t(obs, task_text, batch_size=4)
gr00t_action, _ = adapter.policy.get_action(gr00t_obs)
actions = adapter.gr00t_action_to_moses(gr00t_action)
```

---

## GR00TFineTuner

```python
from moses.gr00t.finetune import GR00TFineTuner
```

Fine-tunes GR00T on Moses-specific tasks.

### Constructor

```python
GR00TFineTuner(
    model_path: str,
    output_dir: str,
    config: GR00TFineTuneConfig,
)
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `load_dataset()` | `dataset_path` | `Dataset` | Load LeRobot v2 dataset |
| `setup_training()` | — | — | Configure LoRA, optimizer, scheduler |
| `train()` | `num_steps` | `dict` | Run fine-tuning |
| `evaluate()` | `eval_dataset` | `dict` | Evaluate on held-out data |
| `save_checkpoint()` | `step` | `Path` | Save checkpoint |

### Config

```python
@dataclass
class GR00TFineTuneConfig:
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_steps: int = 10000
    lora_rank: int = 16
    lora_alpha: int = 32
    warmup_steps: int = 500
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
```

---

## EmbodimentConfig

```python
from moses.gr00t.embodiment import EmbodimentConfig
```

Defines robot embodiment for GR00T.

### Unitree H2 Plus + Sharpa Wave

```python
config = EmbodimentConfig.from_preset("UNITREE_H2_SHARPA")
# 75 DOF: 31 body + 44 hands (22 per hand)
```

### Custom Embodiment

```python
config = EmbodimentConfig(
    body=BodyConfig(dof=28, height=1.75, mass=75.0),
    hands=[
        HandConfig(type="SHARPA_WAVE", dof=22, side="left"),
        HandConfig(type="SHARPA_WAVE", dof=22, side="right"),
    ],
    sensors=SensorConfig(
        cameras=["rgb_front", "rgb_wrist", "rgb_head"],
        imu=True,
        force_torque=True,
    ),
)
```

---

*See `docs/GR00T_INTEGRATION.md` for full research document.*
