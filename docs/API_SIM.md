# Advanced Simulation API — Moses

> **Multi-physics, contact-rich manipulation, deformable objects, and advanced sensors.**

---

## MultiPhysics

```python
from moses.sim.multiphysics import MultiPhysics
```

Coupled rigid body + soft body + fluid + thermal dynamics.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `add_rigid_body()` | `body` | — | Add rigid body |
| `add_soft_body()` | `body`, `youngs_modulus` | — | Add soft body (FEM) |
| `add_fluid()` | `volume`, `viscosity` | — | Add fluid interaction |
| `add_thermal()` | `body`, `conductivity` | — | Add thermal effects |
| `step()` | `dt` | — | Advance simulation |

### Physics Coupling

| Coupling | Description | Use Case |
|----------|-------------|----------|
| Rigid-Soft | Contact forces between rigid and deformable | Foot on soft ground |
| Rigid-Fluid | Buoyancy, drag | Robot in water |
| Thermal-Rigid | Motor heating, friction | Overheating detection |
| Electromagnetic | Actuator fields | Magnetic interference |

---

## Manipulation

```python
from moses.sim.manipulation import Manipulation
```

Contact-rich manipulation with grasp planning.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `plan_grasp()` | `object`, `hand` | `GraspConfig` | Analytical grasp plan |
| `plan_in_hand()` | `object`, `target_pose` | `list` | In-hand manipulation |
| `simulate_sliding()` | `object`, `force` | `Trajectory` | Sliding simulation |
| `simulate_rolling()` | `object`, `force` | `Trajectory` | Rolling simulation |
| `get_tactile_feedback()` | `contact_points` | `dict` | Simulated tactile |

### Grasp Types

| Type | Description | Example |
|------|-------------|---------|
| **Power** | Enveloping grasp | Holding a hammer |
| **Precision** | Fingertip grasp | Holding a pen |
| **Hook** | Hook with fingers | Carrying a bag |
| **Pinch** | Thumb + finger | Picking up a coin |
| **Tripod** | Thumb + 2 fingers | Holding a ball |

---

## Deformable

```python
from moses.sim.deformable import Deformable
```

Finite Element Method for soft objects.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `create_cloth()` | `size`, `resolution` | `Mesh` | Cloth mesh |
| `create_soft_body()` | `shape`, `youngs_modulus` | `Mesh` | Soft body |
| `simulate_cutting()` | `mesh`, `cut_line` | `Mesh` | Cutting simulation |
| `simulate_tearing()` | `mesh`, `force` | `Mesh` | Tearing simulation |
| `simulate_folding()` | `mesh`, `fold_line` | `Mesh` | Folding simulation |

### Material Properties

| Material | Young's Modulus (Pa) | Poisson's Ratio | Use Case |
|----------|---------------------|-----------------|----------|
| Rubber | 1e7 | 0.49 | Grippers |
| Foam | 1e5 | 0.3 | Cushioning |
| Gel | 1e4 | 0.45 | Tactile sensors |
| Cloth | 1e9 | 0.3 | Fabric |

---

## AdvancedSensors

```python
from moses.sim.sensors import AdvancedSensors
```

Advanced sensor simulation with noise models.

### Camera Simulation

| Type | Resolution | FPS | Noise Model |
|------|-----------|-----|-------------|
| RGB | 1920×1080 | 60 | Gaussian + Poisson |
| Depth | 640×480 | 30 | Structured light error |
| Stereo | 2×1280×720 | 30 | Discretization + calibration |
| Event | 1280×720 | — | Asynchronous, threshold-based |

### Tactile Simulation

| Sensor | Resolution | Range | Output |
|--------|-----------|-------|--------|
| Pressure | 10×10 | 0-1 MPa | Force map |
| Vibration | 100 Hz | 0-1 kHz | Acceleration |
| Temperature | 1 point | 0-100°C | Thermal |

### Force/Torque

| Sensor | DOF | Range | Accuracy |
|--------|-----|-------|----------|
| Wrist | 6 | ±200 N, ±10 Nm | 0.1% FS |
| Fingertip | 6 | ±50 N, ±2 Nm | 0.5% FS |
| Distributed | 3 | ±20 N | 1% FS |

### IMU Simulation

| Error Type | Model | Magnitude |
|------------|-------|-----------|
| Bias | Random walk | 0.01 °/s |
| Noise | White noise | 0.001 °/s/√Hz |
| Scale factor | Linear drift | 0.1% |
| Cross-axis | Coupling | 0.5% |

---

*Physics is the ultimate reality check.*
