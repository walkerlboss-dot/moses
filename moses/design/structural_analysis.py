"""
moses/design/structural_analysis.py
FEA-style structural analysis for humanoid robot components.

Uses beam theory, Euler buckling, and safety-factor methods.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

# ---------------------------------------------------------------------------
# Material database — real engineering properties
# ---------------------------------------------------------------------------

class Material:
    """Isotropic material definition."""
    def __init__(self, name: str, E: float, rho: float, sigma_y: float,
                 sigma_u: float, cost_per_kg: float):
        """
        Args:
            name: Common name
            E: Young's modulus [Pa]
            rho: Density [kg/m³]
            sigma_y: Yield strength [Pa]
            sigma_u: Ultimate tensile strength [Pa]
            cost_per_kg: USD / kg
        """
        self.name = name
        self.E = E
        self.rho = rho
        self.sigma_y = sigma_y
        self.sigma_u = sigma_u
        self.cost_per_kg = cost_per_kg

MATERIALS: Dict[str, Material] = {
    "aluminum_6061": Material(
        "Aluminum 6061-T6", E=68.9e9, rho=2700.0,
        sigma_y=276e6, sigma_u=310e6, cost_per_kg=3.50
    ),
    "aluminum_7075": Material(
        "Aluminum 7075-T6", E=71.7e9, rho=2810.0,
        sigma_y=503e6, sigma_u=572e6, cost_per_kg=5.20
    ),
    "titanium_6al4v": Material(
        "Ti-6Al-4V", E=113.8e9, rho=4430.0,
        sigma_y=880e6, sigma_u=950e6, cost_per_kg=35.00
    ),
    "steel_4140": Material(
        "Steel 4140", E=205e9, rho=7850.0,
        sigma_y=655e6, sigma_u=745e6, cost_per_kg=2.80
    ),
    "carbon_fiber": Material(
        "Carbon Fiber (T700)", E=230e9, rho=1600.0,
        sigma_y=2550e6, sigma_u=2550e6, cost_per_kg=45.00
    ),
    "pla_3dprint": Material(
        "PLA (3D-printed)", E=3.5e9, rho=1250.0,
        sigma_y=50e6, sigma_u=55e6, cost_per_kg=25.00
    ),
    "abs_3dprint": Material(
        "ABS (3D-printed)", E=2.3e9, rho=1050.0,
        sigma_y=40e6, sigma_u=45e6, cost_per_kg=22.00
    ),
}


# ---------------------------------------------------------------------------
# Cross-section utilities
# ---------------------------------------------------------------------------

@dataclass
class CrossSection:
    """Generic cross-section with area and moments of inertia."""
    A: float          # Area [m²]
    Ix: float         # Second moment of area about x [m⁴]
    Iy: float         # Second moment of area about y [m⁴]
    J: float          # Polar moment of inertia [m⁴]

    @staticmethod
    def solid_circle(diameter: float) -> "CrossSection":
        r = diameter / 2.0
        A = np.pi * r ** 2
        I = np.pi * r ** 4 / 4.0
        J = np.pi * r ** 4 / 2.0
        return CrossSection(A=A, Ix=I, Iy=I, J=J)

    @staticmethod
    def hollow_circle(outer_d: float, inner_d: float) -> "CrossSection":
        ro, ri = outer_d / 2.0, inner_d / 2.0
        A = np.pi * (ro ** 2 - ri ** 2)
        I = np.pi * (ro ** 4 - ri ** 4) / 4.0
        J = np.pi * (ro ** 4 - ri ** 4) / 2.0
        return CrossSection(A=A, Ix=I, Iy=I, J=J)

    @staticmethod
    def rectangular(width: float, height: float) -> "CrossSection":
        A = width * height
        Ix = width * height ** 3 / 12.0
        Iy = height * width ** 3 / 12.0
        # Approximate torsion constant for rectangle
        a, b = max(width, height) / 2.0, min(width, height) / 2.0
        J = a * b ** 3 * (16.0 / 3.0 - 3.36 * b / a * (1 - b ** 4 / (12 * a ** 4)))
        return CrossSection(A=A, Ix=Ix, Iy=Iy, J=J)

    @staticmethod
    def box_section(outer_w: float, outer_h: float,
                    thickness: float) -> "CrossSection":
        """Thin-walled rectangular tube."""
        iw, ih = outer_w - 2 * thickness, outer_h - 2 * thickness
        A = outer_w * outer_h - iw * ih
        Ix = (outer_w * outer_h ** 3 - iw * ih ** 3) / 12.0
        Iy = (outer_h * outer_w ** 3 - ih * iw ** 3) / 12.0
        # Simplified torsion constant for thin-walled box
        J = 2 * thickness * (outer_w - thickness) ** 2 * (outer_h - thickness) ** 2 / (outer_w + outer_h - 2 * thickness)
        return CrossSection(A=A, Ix=Ix, Iy=Iy, J=J)


# ---------------------------------------------------------------------------
# Structural component
# ---------------------------------------------------------------------------

class LoadCase(Enum):
    STANDING = "standing"
    WALKING = "walking"
    FALLING = "falling"
    IMPACT = "impact"


@dataclass
class StructuralComponent:
    """A 1D beam-like structural member."""
    name: str
    length: float                # [m]
    material: Material
    section: CrossSection
    fixed_end: bool = True       # Cantilever / fixed-pinned factor

    def mass(self) -> float:
        return self.length * self.section.A * self.material.rho

    # ------------------------------------------------------------------
    # Beam theory — deflection under end load
    # ------------------------------------------------------------------
    def deflection_end_load(self, F: float) -> float:
        """End-loaded cantilever deflection [m]."""
        EI = self.material.E * self.section.Ix
        if self.fixed_end:
            return F * self.length ** 3 / (3.0 * EI)
        else:
            # Fixed-pinned
            return F * self.length ** 3 / (12.0 * EI)

    def deflection_distributed(self, w: float) -> float:
        """Uniformly distributed load w [N/m] — cantilever."""
        EI = self.material.E * self.section.Ix
        return w * self.length ** 4 / (8.0 * EI)

    # ------------------------------------------------------------------
    # Stress under bending + axial
    # ------------------------------------------------------------------
    def max_bending_stress(self, M: float, c: Optional[float] = None) -> float:
        """σ = M·c / I   [Pa]"""
        if c is None:
            # Solid circle approximation
            c = (self.section.Ix * 4.0 / np.pi) ** 0.25
        return M * c / self.section.Ix

    def axial_stress(self, F: float) -> float:
        """σ = F / A   [Pa]"""
        return F / self.section.A

    def von_mises_beam(self, M: float, F_axial: float,
                       T: float = 0.0, c: Optional[float] = None) -> float:
        """
        Simplified von Mises for beam with bending, axial, torsion.
        σ_vm = sqrt( (σ_axial+σ_bending)² + 3·τ² )
        """
        sigma_b = self.max_bending_stress(M, c)
        sigma_a = self.axial_stress(F_axial)
        if c is None:
            c = (self.section.Ix * 4.0 / np.pi) ** 0.25
        tau = T * c / self.section.J if self.section.J > 0 else 0.0
        return np.sqrt((sigma_a + sigma_b) ** 2 + 3.0 * tau ** 2)

    # ------------------------------------------------------------------
    # Buckling — Euler critical load
    # ------------------------------------------------------------------
    def euler_critical_load(self, K: float = 2.0) -> float:
        """
        P_cr = π²·E·I / (K·L)²
        K = 2.0  cantilever (fixed-free)
        K = 1.0  pinned-pinned
        K = 0.7  fixed-pinned
        K = 0.5  fixed-fixed
        """
        EI = self.material.E * min(self.section.Ix, self.section.Iy)
        return np.pi ** 2 * EI / (K * self.length) ** 2

    def slenderness_ratio(self, K: float = 2.0) -> float:
        """L/r where r = sqrt(I/A)."""
        r = np.sqrt(min(self.section.Ix, self.section.Iy) / self.section.A)
        return K * self.length / r

    # ------------------------------------------------------------------
    # Safety factors
    # ------------------------------------------------------------------
    def factor_of_safety(self, sigma_vm: float,
                         critical: bool = False) -> float:
        """
        FoS = σ_yield / σ_vm  (or ultimate if brittle).
        Required: 2.0 for critical, 1.5 for non-critical.
        """
        required = 2.0 if critical else 1.5
        fos = self.material.sigma_y / sigma_vm
        return fos, fos >= required

    def buckling_fos(self, applied_load: float,
                     K: float = 2.0, critical: bool = False) -> float:
        required = 2.0 if critical else 1.5
        pcr = self.euler_critical_load(K)
        fos = pcr / applied_load
        return fos, fos >= required


# ---------------------------------------------------------------------------
# Humanoid-specific load cases
# ---------------------------------------------------------------------------

HUMANOID_MASS_DEFAULT = 80.0          # kg
GRAVITY = 9.80665                     # m/s²


def standing_loads(mass: float = HUMANOID_MASS_DEFAULT) -> Dict[str, float]:
    """
    Static standing: each leg carries ~half body weight.
    Returns axial force per femur [N].
    """
    return {
        "femur_axial": mass * GRAVITY / 2.0,
        "tibia_axial": mass * GRAVITY / 2.0,
        "spine_axial": mass * GRAVITY,
    }


def walking_loads(mass: float = HUMANOID_MASS_DEFAULT,
                  dynamic_factor: float = 1.8) -> Dict[str, float]:
    """
    Walking: single-leg stance with dynamic amplification.
    Dynamic factor ~1.5–2.5 depending on gait speed.
    """
    F_static = mass * GRAVITY / 2.0
    F_dynamic = F_static * dynamic_factor
    # Bending moment at knee during mid-stance (simplified)
    # M ≈ F_dynamic * moment_arm; moment_arm ~ 0.08 m for knee
    return {
        "femur_axial": F_dynamic,
        "femur_bending_moment": F_dynamic * 0.08,
        "tibia_axial": F_dynamic,
        "tibia_bending_moment": F_dynamic * 0.06,
        "ankle_torque": F_dynamic * 0.05,
    }


def falling_loads(mass: float = HUMANOID_MASS_DEFAULT,
                  fall_height: float = 0.5,
                  impact_duration: float = 0.05) -> Dict[str, float]:
    """
    Impact load from falling: F = m·v / Δt  where v = sqrt(2·g·h)
    Distributed across two legs.
    """
    v = np.sqrt(2.0 * GRAVITY * fall_height)
    impulse = mass * v
    F_impact = impulse / impact_duration / 2.0  # per leg
    return {
        "impact_force_per_leg": F_impact,
        "femur_axial": F_impact,
        "tibia_axial": F_impact,
    }


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    component: str
    load_case: str
    sigma_vm: float          # Pa
    deflection: float        # m
    fos_stress: float
    fos_buckling: float
    buckling_ok: bool
    stress_ok: bool
    mass: float              # kg


def analyze_humanoid_leg(material_key: str = "aluminum_6061",
                         femur_len: float = 0.45,
                         tibia_len: float = 0.45,
                         outer_d: float = 0.035,
                         wall: float = 0.003,
                         mass: float = HUMANOID_MASS_DEFAULT,
                         critical: bool = True) -> List[AnalysisResult]:
    """
    Run full structural analysis on a simplified humanoid leg.
    """
    mat = MATERIALS[material_key]
    sec = CrossSection.hollow_circle(outer_d, outer_d - 2 * wall)

    femur = StructuralComponent("femur", femur_len, mat, sec, fixed_end=True)
    tibia = StructuralComponent("tibia", tibia_len, mat, sec, fixed_end=True)

    results: List[AnalysisResult] = []

    for case_name, loads in [
        ("standing", standing_loads(mass)),
        ("walking", walking_loads(mass)),
        ("falling", falling_loads(mass)),
    ]:
        # Femur
        F_axial = loads.get("femur_axial", 0.0)
        M = loads.get("femur_bending_moment", 0.0)
        sigma_vm = femur.von_mises_beam(M, F_axial)
        defl = femur.deflection_end_load(F_axial)
        fos_s, s_ok = femur.factor_of_safety(sigma_vm, critical)
        fos_b, b_ok = femur.buckling_fos(F_axial, K=2.0, critical=critical)
        results.append(AnalysisResult(
            component="femur", load_case=case_name,
            sigma_vm=sigma_vm, deflection=defl,
            fos_stress=fos_s, fos_buckling=fos_b,
            buckling_ok=b_ok, stress_ok=s_ok,
            mass=femur.mass()
        ))

        # Tibia
        F_axial = loads.get("tibia_axial", 0.0)
        M = loads.get("tibia_bending_moment", 0.0)
        sigma_vm = tibia.von_mises_beam(M, F_axial)
        defl = tibia.deflection_end_load(F_axial)
        fos_s, s_ok = tibia.factor_of_safety(sigma_vm, critical)
        fos_b, b_ok = tibia.buckling_fos(F_axial, K=2.0, critical=critical)
        results.append(AnalysisResult(
            component="tibia", load_case=case_name,
            sigma_vm=sigma_vm, deflection=defl,
            fos_stress=fos_s, fos_buckling=fos_b,
            buckling_ok=b_ok, stress_ok=s_ok,
            mass=tibia.mass()
        ))

    return results


# ---------------------------------------------------------------------------
# Example / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MOSES v4.0 — Structural Analysis Demo")
    print("=" * 60)

    for mat_key in ["aluminum_6061", "titanium_6al4v", "carbon_fiber"]:
        print(f"\n--- Material: {MATERIALS[mat_key].name} ---")
        results = analyze_humanoid_leg(material_key=mat_key)
        for r in results:
            status = "PASS" if (r.stress_ok and r.buckling_ok) else "FAIL"
            print(
                f"  {r.component:8s} | {r.load_case:8s} | "
                f"σ_vm={r.sigma_vm/1e6:6.1f} MPa | "
                f"δ={r.deflection*1e3:5.2f} mm | "
                f"FoS_s={r.fos_stress:4.2f} | FoS_b={r.fos_buckling:4.2f} | "
                f"mass={r.mass*1e3:6.1f} g | {status}"
            )
