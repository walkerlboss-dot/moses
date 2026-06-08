"""
moses/design/cost_model.py
Cost estimation for humanoid robot design and manufacturing.

Covers:
- Component BOM costs (motors, sensors, bearings, fasteners)
- Manufacturing costs (CNC, 3D print, labor)
- Assembly time estimation
- Total BOM with supplier-style quotes
- Cost vs performance trade-off analysis
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

try:
    from .structural_analysis import Material, MATERIALS
    from .weight_optimizer import ComponentDesign, RobotWeightBudget
except ImportError:
    from structural_analysis import Material, MATERIALS
    from weight_optimizer import ComponentDesign, RobotWeightBudget


# ---------------------------------------------------------------------------
# Component cost database (USD, representative 2025 market)
# ---------------------------------------------------------------------------

@dataclass
class ComponentQuote:
    """Supplier-style quote for a component."""
    description: str
    unit_cost: float       # USD each
    min_order_qty: int = 1
    lead_time_weeks: int = 4
    supplier: str = "generic"
    url: str = ""


# Actuators / motors
MOTOR_QUOTES: Dict[str, ComponentQuote] = {
    "harmonic_drive_20nm": ComponentQuote(
        "Harmonic drive servo, 20 N·m, 1.5 kg", 450.0,
        supplier="Harmonic Drive LLC", lead_time_weeks=8),
    "harmonic_drive_40nm": ComponentQuote(
        "Harmonic drive servo, 40 N·m, 2.2 kg", 680.0,
        supplier="Harmonic Drive LLC", lead_time_weeks=8),
    "harmonic_drive_80nm": ComponentQuote(
        "Harmonic drive servo, 80 N·m, 3.5 kg", 950.0,
        supplier="Harmonic Drive LLC", lead_time_weeks=10),
    "brushless_outrunner_5kw": ComponentQuote(
        "BLDC outrunner, 5 kW peak, 1.8 kg", 320.0,
        supplier="T-Motor / Maxon", lead_time_weeks=6),
    "planetary_gearbox_10nm": ComponentQuote(
        "Planetary gearbox, 10:1, 10 N·m cont.", 180.0,
        supplier="Neugart", lead_time_weeks=6),
}

# Sensors
SENSOR_QUOTES: Dict[str, ComponentQuote] = {
    "imu_9dof": ComponentQuote(
        "9-DOF IMU (BNO085 / ICM-20948)", 12.50,
        supplier="Bosch / TDK", lead_time_weeks=2),
    "force_torque_6axis": ComponentQuote(
        "6-axis F/T sensor, 500 N / 20 N·m", 850.0,
        supplier="ATI Industrial Automation", lead_time_weeks=6),
    "lidar_360": ComponentQuote(
        "360° LiDAR, 40 m range", 280.0,
        supplier="Slamtec / Livox", lead_time_weeks=4),
    "depth_camera": ComponentQuote(
        "RGB-D camera (RealSense D455)", 350.0,
        supplier="Intel", lead_time_weeks=3),
    "encoder_abs_19bit": ComponentQuote(
        "19-bit absolute magnetic encoder", 45.0,
        supplier="RLS / Renishaw", lead_time_weeks=4),
    "tactile_sensor_array": ComponentQuote(
        "Tactile sensor array, 8×8", 65.0,
        supplier="SynTouch / Pressure Profile", lead_time_weeks=4),
}

# Bearings
BEARING_QUOTES: Dict[str, ComponentQuote] = {
    "deep_groove_6204": ComponentQuote(
        "Deep groove ball bearing 6204", 4.50,
        supplier="SKF", lead_time_weeks=2),
    "angular_contact_7204": ComponentQuote(
        "Angular contact bearing 7204", 18.0,
        supplier="SKF", lead_time_weeks=3),
    "cross_roller_rb_2508": ComponentQuote(
        "Crossed roller bearing RB2508", 120.0,
        supplier="THK / IKO", lead_time_weeks=6),
    "needle_hk_2016": ComponentQuote(
        "Needle bearing HK2016", 3.20,
        supplier="INA", lead_time_weeks=2),
}

# Fasteners & hardware
FASTENER_QUOTES: Dict[str, ComponentQuote] = {
    "socket_head_m4x20": ComponentQuote(
        "M4×20 socket head cap screw, A2-70", 0.15,
        supplier="McMaster-Carr", lead_time_weeks=1),
    "socket_head_m6x30": ComponentQuote(
        "M6×30 socket head cap screw, A2-70", 0.25,
        supplier="McMaster-Carr", lead_time_weeks=1),
    "socket_head_m8x40": ComponentQuote(
        "M8×40 socket head cap screw, A2-70", 0.45,
        supplier="McMaster-Carr", lead_time_weeks=1),
    "hex_nut_m4": ComponentQuote(
        "M4 hex nut, A2-70", 0.05,
        supplier="McMaster-Carr", lead_time_weeks=1),
    "hex_nut_m6": ComponentQuote(
        "M6 hex nut, A2-70", 0.08,
        supplier="McMaster-Carr", lead_time_weeks=1),
    "washer_m4": ComponentQuote(
        "M4 flat washer, A2", 0.03,
        supplier="McMaster-Carr", lead_time_weeks=1),
    "threaded_insert_m4": ComponentQuote(
        "M4 threaded insert for 3D print", 0.35,
        supplier="McMaster-Carr", lead_time_weeks=1),
}

# Electronics
ELECTRONICS_QUOTES: Dict[str, ComponentQuote] = {
    "main_controller": ComponentQuote(
        "x86 mini-ITX SBC (Intel NUC class)", 450.0,
        supplier="Intel / ASUS", lead_time_weeks=3),
    "motor_driver_48v_30a": ComponentQuote(
        "48V 30A motor driver", 85.0,
        supplier="ODrive / Trinamic", lead_time_weeks=4),
    "power_distribution": ComponentQuote(
        "Power distribution board, 48V", 120.0,
        supplier="Custom PCB", lead_time_weeks=4),
    "battery_pack_48v_20ah": ComponentQuote(
        "48V 20Ah Li-ion battery pack", 650.0,
        supplier="LG / Samsung", lead_time_weeks=6),
    "wifi_module": ComponentQuote(
        "WiFi 6 + BT 5.2 module", 25.0,
        supplier="Intel / Realtek", lead_time_weeks=2),
}


# ---------------------------------------------------------------------------
# Manufacturing cost models
# ---------------------------------------------------------------------------

class ManufacturingMethod(Enum):
    CNC_MILLING = "cnc_milling"
    CNC_TURNING = "cnc_turning"
    FDM_3DPRINT = "fdm_3dprint"
    SLS_3DPRINT = "sls_3dprint"
    SHEET_METAL = "sheet_metal"
    WATERJET = "waterjet"
    CASTING = "casting"


@dataclass
class ManufacturingCost:
    """Cost to manufacture a single part."""
    method: ManufacturingMethod
    machine_time_hours: float
    setup_time_hours: float
    material_volume_m3: float
    material: Material
    surface_area_m2: float = 0.0
    finishing_required: bool = False

    # Rates (USD / hour or USD / kg)
    MACHINE_RATE: float = 75.0      # CNC / print machine rate
    LABOR_RATE: float = 55.0        # Technician / machinist
    SETUP_RATE: float = 65.0        # Setup labor rate
    FINISH_RATE: float = 45.0       # Post-processing

    def material_cost(self) -> float:
        mass = self.material_volume_m3 * self.material.rho
        return mass * self.material.cost_per_kg

    def machine_cost(self) -> float:
        return self.machine_time_hours * self.MACHINE_RATE

    def setup_cost(self) -> float:
        return self.setup_time_hours * self.SETUP_RATE

    def labor_cost(self) -> float:
        # Operator attention fraction
        attention_frac = 0.3 if self.method in (
            ManufacturingMethod.CNC_MILLING,
            ManufacturingMethod.CNC_TURNING,
        ) else 0.1
        return self.machine_time_hours * attention_frac * self.LABOR_RATE

    def finishing_cost(self) -> float:
        if not self.finishing_required:
            return 0.0
        # Anodizing, painting, etc.
        return self.surface_area_m2 * 25.0  # $/m²

    def total(self) -> float:
        return (
            self.material_cost() +
            self.machine_cost() +
            self.setup_cost() +
            self.labor_cost() +
            self.finishing_cost()
        )


def estimate_cnc_time(volume_m3: float, complexity: float = 1.0) -> float:
    """
    Rough CNC time estimate based on material removal volume.
    complexity: 1.0 simple, 2.0 moderate, 3.0 complex
    """
    # Typical aluminum removal rate ~ 10 cm³/min for roughing
    removal_rate_m3_s = 1.67e-7  # m³/s
    rough_time = volume_m3 / removal_rate_m3_s / 3600.0
    return rough_time * complexity


def estimate_print_time(volume_m3: float, layer_height_mm: float = 0.2,
                        speed_mm_s: float = 50.0) -> float:
    """Estimate FDM print time in hours."""
    # Simplified: total extrusion length / speed
    # Assume 0.4 mm nozzle, 0.2 mm layer, ~50% infill
    filament_dia = 1.75e-3  # m
    cross_area = np.pi * (filament_dia / 2) ** 2
    length = volume_m3 / cross_area
    time_s = length / (speed_mm_s * 1e-3)
    return time_s / 3600.0


# ---------------------------------------------------------------------------
# Assembly time estimation
# ---------------------------------------------------------------------------

@dataclass
class AssemblyStep:
    description: str
    time_minutes: float
    skill_level: int = 1   # 1=tech, 2=mechanic, 3=specialist


ASSEMBLY_RATES: Dict[int, float] = {
    1: 0.60,   # $/min for technician
    2: 0.90,   # mechanic
    3: 1.50,   # specialist
}


def estimate_assembly_time(n_joints: int = 25,
                           n_sensors: int = 12,
                           n_electrical_harnesses: int = 8) -> float:
    """
    Estimate total assembly time in hours.
    Based on MTM / MODAPTS style breakdown.
    """
    # Mechanical assembly per joint
    joint_time = n_joints * 45.0  # minutes

    # Sensor mounting & calibration
    sensor_time = n_sensors * 20.0

    # Electrical routing & connectors
    harness_time = n_electrical_harnesses * 30.0

    # System integration & test
    integration_time = 240.0  # 4 hours

    total_min = joint_time + sensor_time + harness_time + integration_time
    return total_min / 60.0  # hours


def assembly_cost(total_hours: float,
                  team_size: int = 2,
                  skill_mix: Tuple[float, float, float] = (0.5, 0.4, 0.1)) -> float:
    """
    Compute assembly labor cost.
    skill_mix: fractions of tech, mechanic, specialist
    """
    hourly_blended = (
        skill_mix[0] * 35.0 +
        skill_mix[1] * 55.0 +
        skill_mix[2] * 90.0
    )
    return total_hours * hourly_blended * team_size


# ---------------------------------------------------------------------------
# Full BOM and cost rollup
# ---------------------------------------------------------------------------

@dataclass
class BOMItem:
    part_number: str
    description: str
    quantity: int
    unit_cost: float
    extended_cost: float
    supplier: str
    lead_time_weeks: int


@dataclass
class RobotCostEstimate:
    """Complete cost estimate for one humanoid robot."""
    bom_items: List[BOMItem] = field(default_factory=list)
    manufacturing_cost: float = 0.0
    assembly_cost: float = 0.0
    overhead_rate: float = 0.25   # 25% overhead
    profit_margin: float = 0.30   # 30% margin

    def subtotal(self) -> float:
        bom = sum(i.extended_cost for i in self.bom_items)
        return bom + self.manufacturing_cost + self.assembly_cost

    def overhead(self) -> float:
        return self.subtotal() * self.overhead_rate

    def total_cost(self) -> float:
        return self.subtotal() + self.overhead()

    def sale_price(self) -> float:
        return self.total_cost() * (1.0 + self.profit_margin)

    def summary(self) -> Dict[str, float]:
        return {
            "bom_cost": sum(i.extended_cost for i in self.bom_items),
            "manufacturing": self.manufacturing_cost,
            "assembly": self.assembly_cost,
            "overhead": self.overhead(),
            "total_cost": self.total_cost(),
            "sale_price": self.sale_price(),
        }


def build_humanoid_bom(weight_budget: RobotWeightBudget) -> RobotCostEstimate:
    """
    Build a complete BOM for a humanoid robot.
    """
    estimate = RobotCostEstimate()
    items: List[BOMItem] = []

    # --- Structural components (from weight optimizer) ---
    for name, design in weight_budget.components.items():
        mat = MATERIALS[design.material]
        items.append(BOMItem(
            part_number=f"STR-{name.upper()}",
            description=f"{name} tube, {mat.name}, "
                        f"Ø{design.outer_d*1e3:.1f}×{design.wall*1e3:.1f} mm",
            quantity=1,
            unit_cost=design.cost,
            extended_cost=design.cost,
            supplier="In-house / Vendor",
            lead_time_weeks=4,
        ))

    # --- Motors / Actuators ---
    # Typical humanoid: 25 DOF
    motor_map = {
        "hip_yaw": "harmonic_drive_80nm",
        "hip_roll": "harmonic_drive_80nm",
        "hip_pitch": "harmonic_drive_80nm",
        "knee": "harmonic_drive_80nm",
        "ankle_pitch": "harmonic_drive_40nm",
        "ankle_roll": "harmonic_drive_40nm",
        "shoulder_pitch": "harmonic_drive_40nm",
        "shoulder_roll": "harmonic_drive_40nm",
        "shoulder_yaw": "harmonic_drive_20nm",
        "elbow": "harmonic_drive_40nm",
        "wrist": "planetary_gearbox_10nm",
        "neck_yaw": "planetary_gearbox_10nm",
        "neck_pitch": "planetary_gearbox_10nm",
    }

    # Count instances (2 legs, 2 arms, 1 neck, etc.)
    joint_counts = {
        "hip_yaw": 2, "hip_roll": 2, "hip_pitch": 2,
        "knee": 2, "ankle_pitch": 2, "ankle_roll": 2,
        "shoulder_pitch": 2, "shoulder_roll": 2, "shoulder_yaw": 2,
        "elbow": 2, "wrist": 2,
        "neck_yaw": 1, "neck_pitch": 1,
    }

    for joint, count in joint_counts.items():
        key = motor_map[joint]
        quote = MOTOR_QUOTES[key]
        items.append(BOMItem(
            part_number=f"MOT-{joint.upper()}",
            description=quote.description,
            quantity=count,
            unit_cost=quote.unit_cost,
            extended_cost=quote.unit_cost * count,
            supplier=quote.supplier,
            lead_time_weeks=quote.lead_time_weeks,
        ))

    # --- Sensors ---
    sensor_counts = {
        "imu_9dof": 2,           # torso + head
        "force_torque_6axis": 2, # ankles
        "lidar_360": 1,
        "depth_camera": 2,       # stereo
        "encoder_abs_19bit": 25, # per joint
        "tactile_sensor_array": 4, # hands/feet
    }
    for key, count in sensor_counts.items():
        quote = SENSOR_QUOTES[key]
        items.append(BOMItem(
            part_number=f"SEN-{key.upper()}",
            description=quote.description,
            quantity=count,
            unit_cost=quote.unit_cost,
            extended_cost=quote.unit_cost * count,
            supplier=quote.supplier,
            lead_time_weeks=quote.lead_time_weeks,
        ))

    # --- Bearings ---
    bearing_counts = {
        "cross_roller_rb_2508": 12, # main joints
        "deep_groove_6204": 20,
        "angular_contact_7204": 8,
    }
    for key, count in bearing_counts.items():
        quote = BEARING_QUOTES[key]
        items.append(BOMItem(
            part_number=f"BRG-{key.upper()}",
            description=quote.description,
            quantity=count,
            unit_cost=quote.unit_cost,
            extended_cost=quote.unit_cost * count,
            supplier=quote.supplier,
            lead_time_weeks=quote.lead_time_weeks,
        ))

    # --- Fasteners (estimated kit) ---
    fastener_kit_cost = 85.0
    items.append(BOMItem(
        part_number="HW-KIT-001",
        description="Fastener kit (M4/M6/M8 screws, nuts, washers, inserts)",
        quantity=1,
        unit_cost=fastener_kit_cost,
        extended_cost=fastener_kit_cost,
        supplier="McMaster-Carr",
        lead_time_weeks=1,
    ))

    # --- Electronics ---
    elec_counts = {
        "main_controller": 1,
        "motor_driver_48v_30a": 13,  # one per 2 motors + spare
        "power_distribution": 1,
        "battery_pack_48v_20ah": 2,
        "wifi_module": 1,
    }
    for key, count in elec_counts.items():
        quote = ELECTRONICS_QUOTES[key]
        items.append(BOMItem(
            part_number=f"ELEC-{key.upper()}",
            description=quote.description,
            quantity=count,
            unit_cost=quote.unit_cost,
            extended_cost=quote.unit_cost * count,
            supplier=quote.supplier,
            lead_time_weeks=quote.lead_time_weeks,
        ))

    estimate.bom_items = items

    # --- Manufacturing cost (structural parts) ---
    mfg_total = 0.0
    for name, design in weight_budget.components.items():
        mat = MATERIALS[design.material]
        vol = design.mass / mat.rho
        # Assume CNC for metal, FDM for polymer
        if "3dprint" in design.material:
            mfg = ManufacturingCost(
                method=ManufacturingMethod.FDM_3DPRINT,
                machine_time_hours=estimate_print_time(vol),
                setup_time_hours=0.5,
                material_volume_m3=vol,
                material=mat,
            )
        else:
            mfg = ManufacturingCost(
                method=ManufacturingMethod.CNC_MILLING,
                machine_time_hours=estimate_cnc_time(vol, complexity=1.5),
                setup_time_hours=1.0,
                material_volume_m3=vol * 2.5,  # rough stock
                material=mat,
                finishing_required=True,
                surface_area_m2=design.length * np.pi * design.outer_d,
            )
        mfg_total += mfg.total()

    estimate.manufacturing_cost = mfg_total

    # --- Assembly cost ---
    asm_hours = estimate_assembly_time(n_joints=25, n_sensors=12)
    estimate.assembly_cost = assembly_cost(asm_hours, team_size=2)

    return estimate


# ---------------------------------------------------------------------------
# Cost vs performance trade-off
# ---------------------------------------------------------------------------

@dataclass
class CostPerformancePoint:
    total_cost: float
    sale_price: float
    total_mass: float
    structural_mass: float
    actuator_mass: float
    material: str


def material_tradeoff_analysis(weight_budget_fn,
                               materials: List[str],
                               target_masses: List[float]) -> List[CostPerformancePoint]:
    """
    Explore cost vs mass trade-off by varying structural material.
    """
    points: List[CostPerformancePoint] = []
    for mat in materials:
        for mass in target_masses:
            budget = weight_budget_fn(target_total_mass=mass, materials=[mat])
            est = build_humanoid_bom(budget)
            points.append(CostPerformancePoint(
                total_cost=est.total_cost(),
                sale_price=est.sale_price(),
                total_mass=mass,
                structural_mass=budget.structural_mass,
                actuator_mass=budget.actuator_mass,
                material=mat,
            ))
    return points


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MOSES v4.0 — Cost Model Demo")
    print("=" * 60)

    from .weight_optimizer import optimize_humanoid_weight

    budget = optimize_humanoid_weight(target_total_mass=80.0, height=1.75)
    estimate = build_humanoid_bom(budget)

    print("\n--- BOM Summary ---")
    categories: Dict[str, float] = {}
    for item in estimate.bom_items:
        cat = item.part_number.split("-")[0]
        categories[cat] = categories.get(cat, 0.0) + item.extended_cost
    for cat, cost in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:10s}: ${cost:8,.2f}")

    print("\n--- Cost Rollup ---")
    summary = estimate.summary()
    for key, val in summary.items():
        print(f"  {key:20s}: ${val:10,.2f}")

    print("\n--- Material Trade-off (80 kg robot) ---")
    from .weight_optimizer import optimize_humanoid_weight
    points = material_tradeoff_analysis(
        optimize_humanoid_weight,
        ["aluminum_6061", "aluminum_7075", "titanium_6al4v", "carbon_fiber"],
        [80.0]
    )
    for p in points:
        print(f"  {p.material:18s}: cost=${p.total_cost:8,.0f}  "
              f"sale=${p.sale_price:8,.0f}  struct={p.structural_mass:.2f}kg")
