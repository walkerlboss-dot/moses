# API Documentation — Moses

> **Reference for key classes, functions, and configuration options.**

---

## Core Modules

### `moses.envs.humanoid_env`

#### `MosesHumanoidEnvCfg`

Configuration class for the humanoid environment.

```python
from moses.envs.humanoid_env import MosesHumanoidEnvCfg

cfg = MosesHumanoidEnvCfg(
    num_envs=4096,
    env_spacing=4.0,
    episode_length_s=20.0,
    robot_cfg=RobotCfg(
        usd_path="assets/moses_humanoid.usd",
        init_pos=(0.0, 0.0, 1.05),
    ),
    rewards=RewardCfg(
        velocity_tracking_weight=1.0,
        energy_penalty_weight=-0.01,
        stability_bonus_weight=0.1,
    ),
    domain_randomization=DomainRandomizationCfg(
        randomize_mass=True,
        mass_range=(0.9, 1.1),
        randomize_friction=True,
        friction_range=(0.5, 1.2),
    ),
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_envs` | int | 4096 | Number of parallel environments |
| `env_spacing` | float | 4.0 | Distance between env origins |
| `episode_length_s` | float | 20.0 | Max episode duration |
| `robot_cfg` | RobotCfg | — | Robot asset configuration |
| `rewards` | RewardCfg | — | Reward function weights |
| `domain_randomization` | DomainRandomizationCfg | — | Randomization settings |

---

### `moses.utils.checkpoint`

#### `CheckpointManager`

Manages policy checkpoints with resume capability.

```python
from moses.utils.checkpoint import CheckpointManager

ckpt = CheckpointManager(
    save_dir="checkpoints",
    max_to_keep=10,
    save_interval=50,  # iterations
)

# Save checkpoint
ckpt.save(
    iteration=100,
    policy=policy,
    optimizer=optimizer,
    env_state=env_state,
)

# Load checkpoint
policy, optimizer, env_state = ckpt.load(
    checkpoint_path="checkpoints/best.pt"
)

# Resume training
policy, optimizer, start_iter = ckpt.resume(
    run_dir="checkpoints/experiment_1"
)
```

**Methods:**

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `save()` | iteration, policy, optimizer, env_state | Path | Save checkpoint |
| `load()` | checkpoint_path | (policy, optimizer, env_state) | Load checkpoint |
| `resume()` | run_dir | (policy, optimizer, start_iter) | Resume from latest |
| `list()` | run_dir | List[Path] | List available checkpoints |

---

### `moses.utils.domain_randomization`

#### `DomainRandomizer`

Applies domain randomization to simulation parameters.

```python
from moses.utils.domain_randomization import DomainRandomizer

randomizer = DomainRandomizer(
    randomize_mass=True,
    mass_range=(0.9, 1.1),
    randomize_friction=True,
    friction_range=(0.5, 1.2),
    randomize_gravity=True,
    gravity_range=(9.0, 10.0),
)

# Apply to environment
randomizer.randomize(env)
```

---

## Training Scripts

### `scripts/train_humanoid.py`

Main training script.

```bash
python scripts/train_humanoid.py [OPTIONS]
```

**Options:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | str | configs/train_ppo.yaml | Training config file |
| `--num_envs` | int | 4096 | Number of parallel environments |
| `--headless` | flag | False | Run in headless mode |
| `--distributed` | flag | False | Enable multi-GPU training |
| `--resume` | flag | False | Resume from checkpoint |
| `--checkpoint` | str | None | Checkpoint path to resume from |
| `--wandb_project` | str | moses-humanoid | W&B project name |
| `--experiment_name` | str | None | Experiment name |
| `--seed` | int | 42 | Random seed |
| `--mixed_precision` | flag | False | Enable AMP |
| `--cuda_graphs` | flag | False | Enable CUDA graphs |

---

### `scripts/eval_policy.py`

Policy evaluation script.

```bash
python scripts/eval_policy.py [OPTIONS]
```

**Options:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--checkpoint` | str | required | Policy checkpoint path |
| `--num_episodes` | int | 100 | Number of eval episodes |
| `--render` | flag | False | Enable rendering |
| `--save_video` | flag | False | Save episode videos |
| `--export_onnx` | flag | False | Export to ONNX |
| `--onnx_path` | str | None | ONNX output path |

---

### `scripts/export_tensorrt.py`

TensorRT optimization script.

```bash
python scripts/export_tensorrt.py [OPTIONS]
```

**Options:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--onnx` | str | required | ONNX model path |
| `--output` | str | required | Output .trt path |
| `--fp16` | flag | False | Use FP16 precision |
| `--int8` | flag | False | Use INT8 precision |
| `--batch_size` | int | 1 | Batch size |
| `--benchmark` | flag | False | Run benchmark |

---

## Monitoring

### `monitoring/monitor_dashboard.py`

Live terminal dashboard.

```bash
python monitoring/monitor_dashboard.py [OPTIONS]
```

**Options:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--refresh` | float | 5.0 | Refresh interval (seconds) |
| `--metrics_db` | str | ~/.moses/metrics.db | Metrics database path |

---

### `monitoring/health_check.py`

System health verification.

```bash
python monitoring/health_check.py [OPTIONS]
```

**Options:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--json` | flag | False | Output as JSON |
| `--verbose` | flag | False | Detailed output |

**Returns:**
```json
{
  "status": "healthy",
  "checks": {
    "cuda": {"status": "pass", "version": "12.3"},
    "isaac_sim": {"status": "pass", "version": "4.2.0"},
    "disk": {"status": "pass", "free_gb": 850},
    "memory": {"status": "pass", "free_gb": 120}
  }
}
```

---

## Configuration Files

### `configs/train_ppo.yaml`

See [configs/train_ppo.yaml](../configs/train_ppo.yaml) for full reference.

Key sections:
- `env`: Environment configuration
- `policy`: Network architecture
- `algorithm`: PPO hyperparameters
- `runner`: Training loop settings
- `distributed`: Multi-GPU config

---

*For more details, see the source code and inline docstrings.*
