"""
Moses v4.0 Carbon Fiber / Composite Design Module
===================================================
Layup schedule design, fiber orientation optimization,
mold design (male/female), curing cycle specification,
and weight/strength ratio calculation.

Supports prepreg and wet layup processes.
All units are metric (mm, MPa, g) unless noted.

References
----------
- Hexcel HexPly datasheets (prepreg systems)
- Toray T300/T700/T800 data sheets
- ASTM D3039 (tensile), ASTM D7264 (flexural)
- Autoclave curing: Boeing BAC 5317, Airbus AIPS 03-02-003
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Fiber & Resin Database
# ---------------------------------------------------------------------------

class FiberType(Enum):
    T300 = auto()      # Standard modulus
    T700S = auto()     # Intermediate modulus, high strength
    T800H = auto()     # Intermediate modulus, aerospace
    T1000G = auto()    # Ultra-high strength
    M40J = auto()      # High modulus
    M55J = auto()      # Ultra-high modulus
    S_GLASS = auto()   # S-Glass (cost-effective high strength)
    KEVLAR_49 = auto() # Aramid (impact resistance)


class ResinType(Enum):
    EPOXY_3501 = auto()       # Hexcel 3501-6
    EPOXY_8552 = auto()       # Hexcel 8552 (toughened)
    EPOXY_977_3 = auto()      # Cytec 977-3 (high temp)
    BMI = auto()              # Bismaleimide (250°C+ service)
    CYANOACRYLATE = auto()    # Not for structural
    VINYLESTER = auto()       # Marine / corrosion


FIBER_DB: Dict[FiberType, dict] = {
    FiberType.T300: {
        "name": "Toray T300",
        "tensile_mpa": 3530,
        "tensile_modulus_gpa": 230,
        "elongation_pct": 1.5,
        "density_g_cm3": 1.76,
        "fiber_diameter_um": 7.0,
        "cost_per_kg_usd": 25.0,
    },
    FiberType.T700S: {
        "name": "Toray T700S",
        "tensile_mpa": 4900,
        "tensile_modulus_gpa": 230,
        "elongation_pct": 2.1,
        "density_g_cm3": 1.80,
        "fiber_diameter_um": 7.0,
        "cost_per_kg_usd": 35.0,
    },
    FiberType.T800H: {
        "name": "Toray T800H",
        "tensile_mpa": 5490,
        "tensile_modulus_gpa": 294,
        "elongation_pct": 1.9,
        "density_g_cm3": 1.81,
        "fiber_diameter_um": 5.0,
        "cost_per_kg_usd": 65.0,
    },
    FiberType.T1000G: {
        "name": "Toray T1000G",
        "tensile_mpa": 6370,
        "tensile_modulus_gpa": 294,
        "elongation_pct": 2.2,
        "density_g_cm3": 1.80,
        "fiber_diameter_um": 5.0,
        "cost_per_kg_usd": 120.0,
    },
    FiberType.M40J: {
        "name": "Toray M40J",
        "tensile_mpa": 4410,
        "tensile_modulus_gpa": 377,
        "elongation_pct": 1.2,
        "density_g_cm3": 1.77,
        "fiber_diameter_um": 5.0,
        "cost_per_kg_usd": 90.0,
    },
    FiberType.M55J: {
        "name": "Toray M55J",
        "tensile_mpa": 4020,
        "tensile_modulus_gpa": 540,
        "elongation_pct": 0.8,
        "density_g_cm3": 1.91,
        "fiber_diameter_um": 5.0,
        "cost_per_kg_usd": 250.0,
    },
    FiberType.S_GLASS: {
        "name": "S-Glass",
        "tensile_mpa": 4580,
        "tensile_modulus_gpa": 86,
        "elongation_pct": 5.4,
        "density_g_cm3": 2.49,
        "fiber_diameter_um": 9.0,
        "cost_per_kg_usd": 12.0,
    },
    FiberType.KEVLAR_49: {
        "name": "Kevlar 49",
        "tensile_mpa": 3000,
        "tensile_modulus_gpa": 112,
        "elongation_pct": 2.5,
        "density_g_cm3": 1.44,
        "fiber_diameter_um": 12.0,
        "cost_per_kg_usd": 45.0,
    },
}


RESIN_DB: Dict[ResinType, dict] = {
    ResinType.EPOXY_3501: {
        "name": "Hexcel 3501-6",
        "tensile_mpa": 90,
        "tensile_modulus_gpa": 3.5,
        "tg_c": 180,
        "density_g_cm3": 1.27,
        "cost_per_kg_usd": 18.0,
        "cure_temp_c": 177,
        "cure_time_hr": 2.0,
        "post_cure_temp_c": 177,
        "post_cure_time_hr": 4.0,
    },
    ResinType.EPOXY_8552: {
        "name": "Hexcel 8552",
        "tensile_mpa": 110,
        "tensile_modulus_gpa": 4.2,
        "tg_c": 200,
        "density_g_cm3": 1.30,
        "cost_per_kg_usd": 28.0,
        "cure_temp_c": 180,
        "cure_time_hr": 2.0,
        "post_cure_temp_c": 180,
        "post_cure_time_hr": 4.0,
    },
    ResinType.EPOXY_977_3: {
        "name": "Cytec 977-3",
        "tensile_mpa": 100,
        "tensile_modulus_gpa": 3.8,
        "tg_c": 220,
        "density_g_cm3": 1.28,
        "cost_per_kg_usd": 35.0,
        "cure_temp_c": 177,
        "cure_time_hr": 3.0,
        "post_cure_temp_c": 227,
        "post_cure_time_hr": 4.0,
    },
    ResinType.BMI: {
        "name": "Bismaleimide",
        "tensile_mpa": 120,
        "tensile_modulus_gpa": 4.5,
        "tg_c": 290,
        "density_g_cm3": 1.25,
        "cost_per_kg_usd": 55.0,
        "cure_temp_c": 190,
        "cure_time_hr": 4.0,
        "post_cure_temp_c": 250,
        "post_cure_time_hr": 6.0,
    },
    ResinType.VINYLESTER: {
        "name": "Vinyl Ester",
        "tensile_mpa": 80,
        "tensile_modulus_gpa": 3.2,
        "tg_c": 120,
        "density_g_cm3": 1.12,
        "cost_per_kg_usd": 8.0,
        "cure_temp_c": 25,  # RT cure possible
        "cure_time_hr": 24.0,
        "post_cure_temp_c": 80,
        "post_cure_time_hr": 4.0,
    },
}


# ---------------------------------------------------------------------------
# Ply Definition
# ---------------------------------------------------------------------------

@dataclass
class Ply:
    """A single ply (layer) of composite material."""

    fiber: FiberType
    resin: ResinType
    orientation_deg: float = 0.0  # 0 = warp (0°), 90 = fill (90°), ±45
    thickness_mm: float = 0.125   # Typical prepreg ply thickness
    fiber_volume_fraction: float = 0.60  # Vf
    areal_weight_g_m2: float = 200.0   # gsm

    def __post_init__(self):
        assert 0.3 <= self.fiber_volume_fraction <= 0.75

    @property
    def density_g_cm3(self) -> float:
        """Rule of mixtures for composite density."""
        vf = self.fiber_volume_fraction
        vm = 1.0 - vf
        rho_f = FIBER_DB[self.fiber]["density_g_cm3"]
        rho_m = RESIN_DB[self.resin]["density_g_cm3"]
        return vf * rho_f + vm * rho_m

    @property
    def tensile_modulus_gpa(self) -> float:
        """Longitudinal modulus (rule of mixtures, parallel)."""
        vf = self.fiber_volume_fraction
        vm = 1.0 - vf
        Ef = FIBER_DB[self.fiber]["tensile_modulus_gpa"]
        Em = RESIN_DB[self.resin]["tensile_modulus_gpa"]
        return vf * Ef + vm * Em

    @property
    def tensile_strength_mpa(self) -> float:
        """Longitudinal tensile strength (rule of mixtures)."""
        vf = self.fiber_volume_fraction
        vm = 1.0 - vf
        sigma_f = FIBER_DB[self.fiber]["tensile_mpa"]
        sigma_m = RESIN_DB[self.resin]["tensile_mpa"]
        return vf * sigma_f + vm * sigma_m

    def weight_g(self, area_m2: float) -> float:
        """Ply weight for a given area."""
        return self.areal_weight_g_m2 * area_m2


# ---------------------------------------------------------------------------
# Layup Schedule
# ---------------------------------------------------------------------------

@dataclass
class LayupSchedule:
    """Ordered stack of plies."""

    name: str = "default"
    plies: List[Ply] = field(default_factory=list)
    area_m2: float = 0.01  # 100 cm² default

    def add_ply(self, ply: Ply) -> "LayupSchedule":
        self.plies.append(ply)
        return self

    def add_stack(self, orientations: List[float], ply: Ply) -> "LayupSchedule":
        """Add multiple plies with given orientations."""
        for angle in orientations:
            p = Ply(
                fiber=ply.fiber,
                resin=ply.resin,
                orientation_deg=angle,
                thickness_mm=ply.thickness_mm,
                fiber_volume_fraction=ply.fiber_volume_fraction,
                areal_weight_g_m2=ply.areal_weight_g_m2,
            )
            self.plies.append(p)
        return self

    @property
    def total_thickness_mm(self) -> float:
        return sum(p.thickness_mm for p in self.plies)

    @property
    def total_weight_g(self) -> float:
        return sum(p.weight_g(self.area_m2) for p in self.plies)

    @property
    def laminate_density_g_cm3(self) -> float:
        """Average laminate density."""
        total_vol_cm3 = self.total_thickness_mm * self.area_m2 * 10_000 / 1000  # mm³ → cm³
        return self.total_weight_g / total_vol_cm3 if total_vol_cm3 > 0 else 0.0

    def balance_check(self) -> dict:
        """Check for symmetric/balanced layup."""
        angles = [p.orientation_deg for p in self.plies]
        # Symmetric: ply i angle == ply -i-1 angle
        symmetric = all(
            math.isclose(angles[i], angles[-(i + 1)], abs_tol=0.1)
            for i in range(len(angles) // 2)
        )
        # Balanced: for every +θ there is a -θ (excluding 0 and 90)
        non_ortho = [a for a in angles if not math.isclose(abs(a) % 90, 0, abs_tol=0.1)]
        balanced = all(
            any(math.isclose(a, -b, abs_tol=0.1) for b in non_ortho)
            for a in non_ortho
        )
        return {"symmetric": symmetric, "balanced": balanced, "angles": angles}

    def effective_properties(self) -> dict:
        """
        Classical Lamination Theory (CLT) simplified:
        A-matrix in-plane stiffnesses.
        Returns Ex, Ey, Gxy approximations.
        """
        n = len(self.plies)
        if n == 0:
            return {"Ex_gpa": 0, "Ey_gpa": 0, "Gxy_gpa": 0, "nu_xy": 0.3}

        # Simplified: average transformed modulus
        Ex_sum = 0.0
        Ey_sum = 0.0
        Gxy_sum = 0.0
        for ply in self.plies:
            E1 = ply.tensile_modulus_gpa
            E2 = E1 * 0.1  # Typical E2 ≈ 0.1 E1 for unidirectional
            G12 = E1 / 20.0  # Approximate
            theta = math.radians(ply.orientation_deg)
            c = math.cos(theta)
            s = math.sin(theta)
            c2, s2, c4, s4 = c**2, s**2, c**4, s**4
            Ex = 1.0 / (c4 / E1 + s4 / E2 + c2 * s2 * (1.0 / G12 - 2.0 * 0.3 / E1))
            Ey = 1.0 / (s4 / E1 + c4 / E2 + c2 * s2 * (1.0 / G12 - 2.0 * 0.3 / E1))
            Gxy = 1.0 / ((c2 - s2) ** 2 * (1.0 / E1 + 1.0 / E2 + 2.0 * 0.3 / E1) + c2 * s2 / G12)
            Ex_sum += Ex
            Ey_sum += Ey
            Gxy_sum += Gxy

        return {
            "Ex_gpa": round(Ex_sum / n, 2),
            "Ey_gpa": round(Ey_sum / n, 2),
            "Gxy_gpa": round(Gxy_sum / n, 2),
            "nu_xy": 0.3,
        }


# ---------------------------------------------------------------------------
# Fiber Orientation Optimizer
# ---------------------------------------------------------------------------

class OrientationOptimizer:
    """
    Optimize ply orientations for given load cases.
    Uses simplified failure criteria (Tsai-Hill approximations).
    """

    @staticmethod
    def optimize_for_tension(
        fiber: FiberType,
        resin: ResinType,
        load_direction_deg: float = 0.0,
        min_plies: int = 4,
        max_plies: int = 16,
    ) -> LayupSchedule:
        """
        Generate a near-optimal layup for uniaxial tension.
        Primary load direction gets 0° plies; ±45° for shear/torsion.
        """
        ply = Ply(fiber=fiber, resin=resin, orientation_deg=0.0)
        schedule = LayupSchedule(name="tension_optimized")

        # Standard quasi-isotropic base if no direction specified
        if load_direction_deg == 0.0:
            stack = [0, 45, -45, 90] * (max_plies // 4)
        else:
            # Bias toward load direction
            stack = [load_direction_deg] * (max_plies // 2)
            stack += [load_direction_deg + 45, load_direction_deg - 45] * (max_plies // 4)

        schedule.add_stack(stack[:max_plies], ply)
        return schedule

    @staticmethod
    def optimize_for_torsion(
        fiber: FiberType,
        resin: ResinType,
        min_plies: int = 8,
    ) -> LayupSchedule:
        """±45° dominant layup for torsional stiffness."""
        ply = Ply(fiber=fiber, resin=resin, orientation_deg=45.0)
        schedule = LayupSchedule(name="torsion_optimized")
        stack = [45, -45, 45, -45, 0, 90] * (min_plies // 6)
        schedule.add_stack(stack, ply)
        return schedule

    @staticmethod
    def optimize_for_bending(
        fiber: FiberType,
        resin: ResinType,
        min_plies: int = 8,
    ) -> LayupSchedule:
        """0° on outer surfaces for bending stiffness (I-beam analogy)."""
        ply = Ply(fiber=fiber, resin=resin, orientation_deg=0.0)
        schedule = LayupSchedule(name="bending_optimized")
        # Outer: 0°, core: ±45, inner: 90
        stack = [0, 0, 45, -45, 90, 90, -45, 45, 0, 0]
        schedule.add_stack(stack[:min_plies], ply)
        return schedule


# ---------------------------------------------------------------------------
# Mold Design
# ---------------------------------------------------------------------------

class MoldType(Enum):
    MALE = auto()
    FEMALE = auto()
    SPLIT_LINE = auto()
    BLADDER = auto()   # For hollow tubes (humanoid limbs)


@dataclass
class MoldDesign:
    """Tooling design for composite fabrication."""

    mold_type: MoldType
    part_length_mm: float = 300.0
    part_width_mm: float = 100.0
    part_depth_mm: float = 50.0
    draft_angle_deg: float = 2.0
    shrinkage_pct: float = 0.2  # Mold shrinkage compensation
    surface_finish: str = "SPI-C3"  # Medium gloss
    tool_material: str = "aluminum_6061"  # aluminum_6061, steel_p20, invar
    cavity_count: int = 1

    def mold_dimensions(self) -> Tuple[float, float, float]:
        """Mold block dimensions with shrinkage and machining allowance."""
        factor = 1.0 + self.shrinkage_pct / 100.0
        L = self.part_length_mm * factor + 40.0  # 20mm margin each side
        W = self.part_width_mm * factor + 40.0
        D = self.part_depth_mm * factor + 30.0
        return (L, W, D)

    def mold_volume_cm3(self) -> float:
        L, W, D = self.mold_dimensions()
        return (L * W * D) / 1000.0

    def mold_weight_kg(self) -> float:
        """Approximate mold weight."""
        rho = {"aluminum_6061": 2.70, "steel_p20": 7.85, "invar": 8.05}
        return self.mold_volume_cm3() * rho.get(self.tool_material, 2.70) / 1000.0

    def mold_cost_usd(self) -> float:
        """Rough mold cost estimate."""
        # Aluminum: $150/kg machined, Steel: $250/kg, Invar: $800/kg
        rates = {"aluminum_6061": 150.0, "steel_p20": 250.0, "invar": 800.0}
        rate = rates.get(self.tool_material, 150.0)
        base = self.mold_weight_kg() * rate
        # Complexity multiplier
        mult = 1.0 + (self.cavity_count - 1) * 0.3
        if self.mold_type == MoldType.SPLIT_LINE:
            mult *= 1.5
        elif self.mold_type == MoldType.BLADDER:
            mult *= 2.0
        return base * mult

    def generate_drawing_notes(self) -> List[str]:
        notes = [
            f"Mold Type: {self.mold_type.name}",
            f"Tool Material: {self.tool_material}",
            f"Draft Angle: {self.draft_angle_deg}° per side",
            f"Shrinkage Compensation: {self.shrinkage_pct}%",
            f"Surface Finish: {self.surface_finish}",
            f"Cavity Count: {self.cavity_count}",
            "Seal all joints for vacuum bagging.",
            "Apply release agent (Frekote 700-NC) before layup.",
        ]
        return notes


# ---------------------------------------------------------------------------
# Curing Cycle
# ---------------------------------------------------------------------------

@dataclass
class CuringCycle:
    """Thermal profile for composite cure."""

    resin: ResinType
    heat_rate_c_per_min: float = 1.5
    hold_temp_c: float = 177.0
    hold_time_hr: float = 2.0
    post_cure_temp_c: Optional[float] = None
    post_cure_time_hr: Optional[float] = None
    pressure_mpa: float = 0.69  # 100 psi autoclave / vacuum
    vacuum_inhg: float = 22.0   # Vacuum bag pressure

    @classmethod
    def from_resin(cls, resin: ResinType) -> "CuringCycle":
        db = RESIN_DB[resin]
        return cls(
            resin=resin,
            hold_temp_c=db["cure_temp_c"],
            hold_time_hr=db["cure_time_hr"],
            post_cure_temp_c=db.get("post_cure_temp_c"),
            post_cure_time_hr=db.get("post_cure_time_hr"),
        )

    def cycle_time_hr(self) -> float:
        """Total cycle time including ramp and post-cure."""
        ramp_up = (self.hold_temp_c - 25.0) / (self.heat_rate_c_per_min * 60.0)
        ramp_down = (self.hold_temp_c - 25.0) / (2.0 * 60.0)  # 2°C/min cool
        total = ramp_up + self.hold_time_hr + ramp_down
        if self.post_cure_temp_c and self.post_cure_time_hr:
            post_ramp = (self.post_cure_temp_c - 25.0) / (self.heat_rate_c_per_min * 60.0)
            total += post_ramp + self.post_cure_time_hr + post_ramp
        return total

    def to_program(self) -> List[Tuple[float, float]]:
        """Return list of (time_min, temp_c) setpoints."""
        points = []
        t = 0.0
        temp = 25.0
        points.append((t, temp))
        # Ramp to hold
        while temp < self.hold_temp_c:
            temp += self.heat_rate_c_per_min
            t += 1.0
            points.append((t, min(temp, self.hold_temp_c)))
        # Hold
        hold_min = self.hold_time_hr * 60.0
        points.append((t + hold_min, self.hold_temp_c))
        t += hold_min
        # Cool
        while temp > 25.0:
            temp -= 2.0
            t += 1.0
            points.append((t, max(temp, 25.0)))
        return points


# ---------------------------------------------------------------------------
# Weight / Strength Calculator
# ---------------------------------------------------------------------------

class WeightStrengthCalculator:
    """
    Calculate specific strength and specific modulus
    (strength-to-weight and stiffness-to-weight ratios).
    """

    @staticmethod
    def specific_strength(tensile_mpa: float, density_g_cm3: float) -> float:
        """MPa / (g/cm³) — higher is better."""
        return tensile_mpa / density_g_cm3

    @staticmethod
    def specific_modulus(modulus_gpa: float, density_g_cm3: float) -> float:
        """GPa / (g/cm³) — higher is better."""
        return modulus_gpa / density_g_cm3

    @classmethod
    def for_layup(cls, schedule: LayupSchedule) -> dict:
        props = schedule.effective_properties()
        rho = schedule.laminate_density_g_cm3
        Ex = props["Ex_gpa"]
        # Approximate tensile strength from 0° ply dominance
        sigma_0 = next((p.tensile_strength_mpa for p in schedule.plies if math.isclose(p.orientation_deg, 0, abs_tol=1)), 0)
        if sigma_0 == 0:
            sigma_0 = sum(p.tensile_strength_mpa for p in schedule.plies) / len(schedule.plies)

        return {
            "total_weight_g": round(schedule.total_weight_g, 2),
            "total_thickness_mm": round(schedule.total_thickness_mm, 3),
            "laminate_density_g_cm3": round(rho, 3),
            "Ex_gpa": Ex,
            "specific_strength_kn_m_kg": round(cls.specific_strength(sigma_0, rho) / 1000, 2),
            "specific_modulus_mn_m_kg": round(cls.specific_modulus(Ex, rho) * 1000, 2),
        }

    @classmethod
    def compare_materials(cls, materials: List[Tuple[FiberType, ResinType]]) -> List[dict]:
        """Compare specific properties across fiber/resin combinations."""
        results = []
        for fiber, resin in materials:
            ply = Ply(fiber=fiber, resin=resin)
            sigma = ply.tensile_strength_mpa
            E = ply.tensile_modulus_gpa
            rho = ply.density_g_cm3
            results.append({
                "fiber": FIBER_DB[fiber]["name"],
                "resin": RESIN_DB[resin]["name"],
                "density_g_cm3": round(rho, 3),
                "tensile_mpa": sigma,
                "modulus_gpa": E,
                "specific_strength": round(cls.specific_strength(sigma, rho), 1),
                "specific_modulus": round(cls.specific_modulus(E, rho), 1),
            })
        return results


# ---------------------------------------------------------------------------
# Composite Layup (High-level wrapper)
# ---------------------------------------------------------------------------

@dataclass
class CompositeLayup:
    """Full composite part definition."""

    name: str
    schedule: LayupSchedule
    mold: Optional[MoldDesign] = None
    cure: Optional[CuringCycle] = None
    labor_hours: float = 2.0
    labor_rate_usd: float = 45.0
    autoclave_rate_usd: float = 120.0  # per hour

    def estimate_cost(self) -> dict:
        """Full part cost estimate."""
        # Material cost
        fiber_cost = 0.0
        resin_cost = 0.0
        for ply in self.schedule.plies:
            gsm = ply.areal_weight_g_m2
            area = self.schedule.area_m2
            weight_g = gsm * area
            # Fiber fraction of weight
            vf = ply.fiber_volume_fraction
            rho_f = FIBER_DB[ply.fiber]["density_g_cm3"]
            rho_m = RESIN_DB[ply.resin]["density_g_cm3"]
            total_rho = vf * rho_f + (1 - vf) * rho_m
            fiber_mass_g = weight_g * (vf * rho_f / total_rho)
            resin_mass_g = weight_g - fiber_mass_g
            fiber_cost += (fiber_mass_g / 1000.0) * FIBER_DB[ply.fiber]["cost_per_kg_usd"]
            resin_cost += (resin_mass_g / 1000.0) * RESIN_DB[ply.resin]["cost_per_kg_usd"]

        material_cost = fiber_cost + resin_cost

        # Labor
        labor_cost = self.labor_hours * self.labor_rate_usd

        # Cure cycle
        cure_time_hr = self.cure.cycle_time_hr() if self.cure else 4.0
        cure_cost = cure_time_hr * self.autoclave_rate_usd

        # Mold amortization (assume 500 parts)
        mold_cost = self.mold.mold_cost_usd() if self.mold else 0.0
        mold_per_part = mold_cost / 500.0

        overhead = (labor_cost + cure_cost) * 0.25
        total = material_cost + labor_cost + cure_cost + mold_per_part + overhead

        return {
            "fiber_cost_usd": round(fiber_cost, 2),
            "resin_cost_usd": round(resin_cost, 2),
            "material_cost_usd": round(material_cost, 2),
            "labor_cost_usd": round(labor_cost, 2),
            "cure_cost_usd": round(cure_cost, 2),
            "mold_amort_usd": round(mold_per_part, 2),
            "overhead_usd": round(overhead, 2),
            "total_cost_usd": round(total, 2),
            "cure_time_hr": round(cure_time_hr, 2),
        }

    def mechanical_report(self) -> dict:
        """Full mechanical and weight report."""
        ws = WeightStrengthCalculator.for_layup(self.schedule)
        balance = self.schedule.balance_check()
        props = self.schedule.effective_properties()
        return {
            "name": self.name,
            "plies": len(self.schedule.plies),
            **ws,
            **props,
            **balance,
        }


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def quick_tube_layup(
    fiber: FiberType = FiberType.T700S,
    resin: ResinType = ResinType.EPOXY_8552,
    diameter_mm: float = 40.0,
    length_mm: float = 300.0,
    wall_thickness_mm: float = 2.0,
) -> CompositeLayup:
    """Create a standard quasi-isotropic tube layup."""
    ply_thick = 0.125
    n_plies = max(4, int(wall_thickness_mm / ply_thick))
    area_m2 = math.pi * (diameter_mm / 1000.0) * (length_mm / 1000.0)
    schedule = LayupSchedule(area_m2=area_m2)
    base_ply = Ply(fiber=fiber, resin=resin, thickness_mm=ply_thick)
    stack = [0, 45, -45, 90] * (n_plies // 4)
    if n_plies % 4:
        stack += [0] * (n_plies % 4)
    schedule.add_stack(stack, base_ply)

    mold = MoldDesign(
        mold_type=MoldType.BLADDER,
        part_length_mm=length_mm,
        part_width_mm=diameter_mm,
        part_depth_mm=wall_thickness_mm,
        tool_material="aluminum_6061",
    )
    cure = CuringCycle.from_resin(resin)
    return CompositeLayup(name="humanoid_tube", schedule=schedule, mold=mold, cure=cure)
