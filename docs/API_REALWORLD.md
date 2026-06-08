# Real-World Bridge API — Moses

> **Sim-to-real transfer, system identification, domain adaptation, and calibration.**

---

## SystemID

```python
from moses.realworld.system_id import SystemID
```

Identifies physical parameters from real robot data.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `identify_mass()` | `trajectory` | `float` | Identify link masses |
| `identify_inertia()` | `trajectory` | `dict` | Identify inertia tensors |
| `identify_friction()` | `trajectory` | `dict` | Identify friction coefficients |
| `identify_motor_model()` | `trajectory` | `dict` | Motor torque constant, resistance |
| `identify_contact()` | `trajectory` | `dict` | Contact stiffness, damping |

### Example

```python
sysid = SystemID()

# Collect real robot data
trajectory = robot.collect_trajectory(duration=60.0)

# Identify parameters
mass = sysid.identify_mass(trajectory)
friction = sysid.identify_friction(trajectory)
motor = sysid.identify_motor_model(trajectory)

# Update simulation model
sim.set_mass(mass)
sim.set_friction(friction)
sim.set_motor_model(motor)
```

---

## DomainAdaptation

```python
from moses.realworld.domain_adaptation import DomainAdaptation
```

Bridges sim-to-real gap.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `quantify_gap()` | `sim_data`, `real_data` | `dict` | Measure sim-to-real gap |
| `randomize_from_real()` | `real_data` | `dict` | Tune domain randomization |
| `adapt_policy()` | `policy`, `real_data` | `Policy` | Adapt policy to real |
| `meta_adapt()` | `policy`, `tasks` | `Policy` | Meta-learn adaptation |

### Gap Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **State MSE** | Mean squared error in state prediction | < 0.1 |
| **Reward correlation** | Correlation between sim and real reward | > 0.8 |
| **Success rate gap** | Difference in task success | < 10% |
| **Transfer score** | Normalized performance on real | > 0.7 |

---

## Calibration

```python
from moses.realworld.calibration import Calibration
```

Calibrates sensors and kinematics.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `calibrate_kinematics()` | `measurements` | `dict` | DH parameter calibration |
| `calibrate_camera()` | `images`, `pattern` | `dict` | Intrinsic + extrinsic |
| `calibrate_force_sensor()` | `loads` | `dict` | Force/torque calibration |
| `hand_eye_calibration()` | `poses`, `images` | `dict` | Hand-eye transform |

### Camera Calibration

```python
calib = Calibration()

# Intrinsic calibration
intrinsic = calib.calibrate_camera(
    images=checkerboard_images,
    pattern=(9, 6),
    square_size=0.025,
)
# Returns: {"fx": 600, "fy": 600, "cx": 320, "cy": 240, "distortion": [...]}

# Extrinsic calibration
extrinsic = calib.calibrate_camera(
    images=robot_images,
    pattern=(9, 6),
    robot_poses=robot_poses,
)
# Returns: {"rotation": [...], "translation": [...]}
```

---

## RealWorldDeploy

```python
from moses.realworld.deployment import RealWorldDeploy
```

Deploys policy to physical robot with safety limits.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `load_policy()` | `policy_path` | — | Load policy |
| `set_safety_limits()` | `limits` | — | Set torque/velocity limits |
| `enable_emergency_stop()` | — | — | Enable e-stop |
| `run_episode()` | `max_steps` | `dict` | Run one episode |
| `log_performance()` | `metrics` | — | Log to monitoring |

### Safety Limits

| Limit | Default | Description |
|-------|---------|-------------|
| Max torque | 80% of rated | Prevent motor damage |
| Max velocity | 90% of rated | Prevent overspeed |
| Max position error | 5° | Detect encoder failure |
| Max force | 50 N | Prevent collision damage |

---

*Sim-to-real: the final boss of robotics.*
