# Manufacturing API — Moses

> **CNC, 3D printing, sheet metal, and carbon fiber manufacturing output.**

---

## CNCGenerator

```python
from moses.manufacturing.cnc_generator import CNCGenerator
```

Generates CNC G-code for milling, turning, drilling.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `generate_face_mill()` | `stock`, `tool`, `params` | `str` | Face milling G-code |
| `generate_profile_mill()` | `contour`, `tool`, `params` | `str` | Profile milling G-code |
| `generate_drill()` | `holes`, `tool`, `params` | `str` | Drilling G-code |
| `estimate_time()` | `gcode` | `float` | Cycle time (minutes) |
| `estimate_cost()` | `time`, `machine_rate` | `float` | Cost estimate |

### Supported Materials

| Material | Feed (mm/min) | Speed (RPM) | Notes |
|----------|---------------|-------------|-------|
| Al 6061-T6 | 1200 | 8000 | Standard |
| Al 7075-T6 | 1000 | 7000 | Harder, slower |
| Steel 4140 | 400 | 2500 | Requires coolant |
| Ti-6Al-4V | 200 | 1500 | Slow, expensive |

### Example

```python
gen = CNCGenerator()
gcode = gen.generate_face_mill(
    stock={"width": 100, "length": 150, "height": 25},
    tool={"diameter": 12, "flutes": 4, "material": "carbide"},
    params={"depth_of_cut": 2.0, "stepover": 0.6},
)
# ~15 min cycle, ~$28 cost
```

---

## PrintGenerator

```python
from moses.manufacturing.print_generator import PrintGenerator
```

Generates 3D print files with optimized orientation and supports.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `generate_gcode()` | `mesh`, `material`, `printer` | `str` | Sliced G-code |
| `optimize_orientation()` | `mesh`, `material` | `tuple` | Best orientation + angle |
| `generate_supports()` | `mesh`, `orientation` | `mesh` | Support structures |
| `estimate_time()` | `gcode` | `float` | Print time (hours) |
| `estimate_cost()` | `time`, `material` | `float` | Material + machine cost |

### Supported Materials

| Material | Strength (MPa) | Cost ($/kg) | Notes |
|----------|----------------|-------------|-------|
| PLA | 65 | 25 | Easy, biodegradable |
| PETG | 75 | 30 | Strong, durable |
| ABS | 80 | 28 | Impact resistant |
| Nylon | 85 | 45 | Flexible, tough |
| CF-Nylon | 120 | 80 | High strength |

---

## SheetMetal

```python
from moses.manufacturing.sheet_metal import SheetMetal
```

Designs sheet metal parts with bend allowance and flat patterns.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `calculate_bend_allowance()` | `angle`, `radius`, `thickness` | `float` | Bend allowance (mm) |
| `generate_flat_pattern()` | `part` | `mesh` | Flat pattern for cutting |
| `select_punch_die()` | `hole_diameter`, `material` | `dict` | Punch/die specs |
| `export_dxf()` | `flat_pattern`, `path` | `Path` | DXF for laser/waterjet |

---

## CarbonFiber

```python
from moses.manufacturing.carbon_fiber import CarbonFiber
```

Designs composite layups with mold design and curing cycles.

### Methods

| Method | Args | Returns | Description |
|--------|------|---------|-------------|
| `design_layup()` | `part`, `loads`, `constraints` | `dict` | Layup schedule |
| `optimize_orientation()` | `loads` | `list` | Fiber angles per ply |
| `design_mold()` | `part`, `type` | `dict` | Male/female mold |
| `specify_cure_cycle()` | `resin`, `thickness` | `list` | Temperature profile |
| `estimate_cost()` | `layup`, `mold` | `float` | Total cost |

---

*See `docs/EXAMPLES.md` for full usage examples.*
