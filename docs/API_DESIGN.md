# Design API — Moses

> **Structural analysis, biomechanics, weight optimization, and cost modeling.**

---

## StructuralAnalyzer

```python
from moses.design.structural_analysis import StructuralAnalyzer
```

FEA-style structural analysis for humanoid robots.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `analyze_stress()` | `link`, `loads` | `dict` | Von Mises stress |
| `analyze_deflection()` | `link`, `loads` | `dict` | Deflection analysis |
| `analyze_buckling()` | `link`, `loads` | `dict` | Euler buckling |
| `calculate_safety_factor()` | `stress`, `yield_strength` | `float` | Safety factor |
| `get_critical_load()` | `link`, `boundary_conditions` | `float` | Critical buckling load |

### Load Cases

| Case | Description | Multiplier |
|------|-------------|------------|
| Standing | Static weight | 1.0× |
| Walking | Dynamic gait | 1.8× |
| Running | High impact | 2.5× |
| Falling | Impact load | 5.0× |
| Lifting | External load | 2.0× |

### Safety Factors

| Component | Factor | Standard |
|-----------|--------|----------|
| Critical (joints, actuators) | 2.0 | NASA-STD-5003 |
| Non-critical (covers) | 1.5 | NASA-STD-5003 |
| Buckling | 3.0 | Euler |
| Fatigue | 2.0 | S-N curves |

### Example

```python
analyzer = StructuralAnalyzer()

# Analyze leg under walking load
result = analyzer.analyze_stress(
    link=robot.leg,
    loads={"axial": 500, "bending": 200, "torsional": 50},
)
# Returns: {"von_mises": 150, "safety_factor": 2.1, "pass": True}
```

---

## BiomechanicsDB

```python
from moses.design.biomechanics import BiomechanicsDB
```

Human biomechanics reference database.

### Joint ROM

| Joint | Flexion | Extension | Abduction | Rotation |
|-------|---------|-----------|-----------|----------|
| Hip | 120° | 30° | 45° | 45° |
| Knee | 140° | 0° | — | 15° |
| Ankle | 30° | 50° | 20° | — |
| Shoulder | 180° | 60° | 180° | 90° |
| Elbow | 150° | 0° | — | 90° |
| Wrist | 80° | 70° | 30° | — |

### Muscle Force Curves

| Muscle | Max Force (N) | Optimal Length | Contraction Velocity |
|--------|--------------|----------------|---------------------|
| Quadriceps | 3000 | 0.12 m | 0.5 m/s |
| Hamstrings | 2500 | 0.14 m | 0.4 m/s |
| Gastrocnemius | 2000 | 0.10 m | 0.3 m/s |
| Biceps | 800 | 0.08 m | 0.2 m/s |
| Triceps | 1000 | 0.09 m | 0.25 m/s |

### Gait Cycle

| Phase | Duration | Key Events |
|-------|----------|------------|
| Heel strike | 0% | Initial contact |
| Loading response | 0-10% | Weight acceptance |
| Mid stance | 10-30% | Single support |
| Terminal stance | 30-50% | Heel off |
| Pre-swing | 50-60% | Toe off |
| Initial swing | 60-73% | Limb advancement |
| Mid swing | 73-87% | Clearance |
| Terminal swing | 87-100% | Deceleration |

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `get_rom()` | `joint` | `dict` | Range of motion |
| `get_muscle_force()` | `muscle`, `length`, `velocity` | `float` | Force from length-velocity curve |
| `get_gait_phase()` | `time` | `str` | Gait phase at time |
| `scale_to_robot()` | `human_dimension`, `robot_height` | `float` | Scale human data to robot |

---

## WeightOptimizer

```python
from moses.design.weight_optimizer import WeightOptimizer
```

Topology optimization for mass minimization.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `optimize_topology()` | `design_space`, `loads`, `constraints` | `dict` | SIMP optimization |
| `optimize_tube()` | `loads`, `constraints` | `dict` | Tube sizing |
| `get_pareto_frontier()` | `designs` | `list` | Pareto-optimal designs |

### SIMP Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Penalty factor | 3.0 | SIMP penalty |
| Filter radius | 2.0 | Density filter |
| Volume fraction | 0.3 | Max material usage |
| Min density | 0.001 | Void density |

### Example

```python
optimizer = WeightOptimizer()

# Topology optimization
result = optimizer.optimize_topology(
    design_space=robot.leg_envelope,
    loads={"standing": 500, "walking": 900, "falling": 2500},
    constraints={"max_stress": 200, "max_deflection": 0.001},
)
# Returns: {"topology": [...], "mass": 2.3, "compliance": 0.05}
```

---

## CostModel

```python
from moses.design.cost_model import CostModel
```

Cost estimation for humanoid robot manufacturing.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `estimate_bom()` | `design` | `dict` | Bill of materials cost |
| `estimate_manufacturing()` | `processes` | `dict` | Manufacturing cost |
| `estimate_assembly()` | `steps` | `dict` | Assembly labor cost |
| `get_total_cost()` | — | `dict` | Total cost breakdown |

### Cost Breakdown

| Category | Cost | Percentage |
|----------|------|------------|
| Actuators | $8,500 | 23% |
| Sensors | $3,200 | 9% |
| Electronics | $2,800 | 8% |
| Structure | $4,500 | 12% |
| Fasteners | $800 | 2% |
| Bearings | $1,200 | 3% |
| **BOM Total** | **$21,000** | **58%** |
| Manufacturing | $8,500 | 23% |
| Assembly labor | $4,200 | 12% |
| Overhead | $2,100 | 6% |
| **Total** | **$35,800** | **100%** |
| **Sale price** | **$47,252** | **132%** |

### Example

```python
cost = CostModel()

# Estimate full cost
estimate = cost.get_total_cost(robot_design)
# Returns: {
#   "bom": 21000,
#   "manufacturing": 8500,
#   "assembly": 4200,
#   "overhead": 2100,
#   "total": 35800,
#   "sale_price": 47252,
# }
```

---

*Engineering is the art of balancing performance, cost, and safety.*
