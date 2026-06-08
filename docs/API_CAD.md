# CAD API — Moses

> **Parametric humanoid robot CAD generation.**

---

## ParametricHumanoid

```python
from moses.cad.parametric_humanoid import ParametricHumanoid
```

Generates humanoid robot CAD from parameters.

### Constructor

```python
ParametricHumanoid(
    height: float = 1.75,      # meters
    mass: float = 75.0,        # kg
    mass_distribution: dict | None = None,  # {"torso": 0.35, "legs": 0.40, ...}
    limb_ratios: dict | None = None,        # {"leg": 0.53, "arm": 0.44, ...}
    dof_config: dict | None = None,         # Joint configuration
)
```

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `generate()` | — | `dict` | Generate full kinematic chain |
| `export_urdf()` | `path` | `Path` | Export to URDF |
| `export_usd()` | `path` | `Path` | Export to USD |
| `export_step()` | `path` | `Path` | Export to STEP |
| `export_stl()` | `path` | `Path` | Export to STL |
| `get_bom()` | — | `DataFrame` | Bill of Materials |
| `get_mass_properties()` | — | `dict` | CoM, inertia tensor |

### Example

```python
humanoid = ParametricHumanoid(
    height=1.75,
    mass=75.0,
    dof_config={
        "legs": {"hip": 3, "knee": 1, "ankle": 2},
        "arms": {"shoulder": 3, "elbow": 1, "wrist": 2},
        "torso": {"waist": 1},
        "head": {"neck": 2},
    },
)

humanoid.generate()
humanoid.export_urdf("output/humanoid.urdf")
humanoid.export_step("output/humanoid.step")

bom = humanoid.get_bom()
# 43 items, 81.11 kg total
```

---

## MeshGenerator

```python
from moses.cad.mesh_generator import MeshGenerator
```

Generates 3D meshes for simulation and visualization.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `generate_collision_mesh()` | `link`, `simplify=True` | `Trimesh` | Convex hull or simplified mesh |
| `generate_visual_mesh()` | `link`, `detail="high"` | `Trimesh` | Detailed mesh with textures |
| `generate_lod()` | `mesh`, `levels=3` | `list` | Level-of-detail variants |

---

## AssemblyManager

```python
from moses.cad.assembly_manager import AssemblyManager
```

Manages assembly with BOM and instructions.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `generate_bom()` | — | `DataFrame` | Bill of materials |
| `generate_instructions()` | — | `list` | Step-by-step assembly |
| `check_interference()` | — | `list` | Interference report |
| `get_mass_properties()` | — | `dict` | CoM, inertia, total mass |
| `tolerance_analysis()` | `tolerance_spec` | `dict` | RSS tolerance stack-up |

---

*See `docs/EXAMPLES.md` for full usage examples.*
