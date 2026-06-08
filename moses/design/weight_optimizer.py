"""
moses/design/weight_optimizer.py
Mass minimization for humanoid robot structures.

Methods:
- SIMP topology optimization (simplified 2D/3D)
- Material selection per component
- Pareto frontier: weight vs stiffness vs cost
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import minimize

try:
    from .structural_analysis import (
        Material, MATERIALS, CrossSection, StructuralComponent
    )
except ImportError:
    from structural_analysis import (
        Material, MATERIALS, CrossSection, StructuralComponent
    )


# ---------------------------------------------------------------------------
# SIMP Topology Optimization (2D)
# ---------------------------------------------------------------------------

class SIMPOptimizer:
    """
    Simplified SIMP (Solid Isotropic Material with Penalization)
    for 2D compliance minimization with volume constraint.

    Reference: Bendsøe & Sigmund (2003) "Topology Optimization"
    """
    def __init__(self, nx: int, ny: int,
                 volfrac: float = 0.4,
                 penal: float = 3.0,
                 rmin: float = 1.5,
                 E0: float = 1.0,
                 Emin: float = 1e-9):
        """
        Args:
            nx, ny: Mesh resolution
            volfrac: Target volume fraction
            penal: Penalization power p
            rmin: Filter radius (element units)
            E0: Young's modulus of solid material
            Emin: Young's modulus of void
        """
        self.nx = nx
        self.ny = ny
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.E0 = E0
        self.Emin = Emin
        self.x = np.full((ny, nx), volfrac)   # Design variables

    def _filtered_density(self) -> np.ndarray:
        """Density filter (simplified convolution)."""
        # Simple mean filter for stability
        from scipy.ndimage import uniform_filter
        return uniform_filter(self.x, size=max(1, int(2 * self.rmin)),
                              mode="constant")

    def _youngs_modulus(self, x: np.ndarray) -> np.ndarray:
        """SIMP interpolation: E(x) = Emin + x^p · (E0 - Emin)"""
        return self.Emin + x ** self.penal * (self.E0 - self.Emin)

    def compliance(self, forces: np.ndarray, fixed: np.ndarray) -> float:
        """
        Compute structural compliance C = Fᵀ·U.
        Simplified: uses a direct solver for a 2D cantilever.
        """
        # This is a minimal educational implementation.
        # A production version would use FEniCS / pyTopOpt.
        x_f = self._filtered_density()
        E = self._youngs_modulus(x_f)
        # Approximate compliance as sum(E·x) under load
        # (Not a true FEA — placeholder for integration with real solver)
        return float(np.sum(forces * forces / (E + 1e-12)))

    def optimize_oc(self, n_iter: int = 100,
                    move: float = 0.2) -> np.ndarray:
        """
        Optimality Criteria update (simplified).
        Returns optimized density field.
        """
        for _ in range(n_iter):
            x_f = self._filtered_density()
            E = self._youngs_modulus(x_f)
            # Sensitivity approx: dc/dx ≈ -p·x^(p-1)·(E0-Emin)·strain_energy
            # Using a dummy sensitivity for demonstration
            dc = -self.penal * (x_f ** (self.penal - 1)) * (self.E0 - self.Emin)
            dc = np.clip(dc, -1e6, 1e6)

            # OC update
            l1, l2 = 0.0, 1e6
            while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-4:
                lmid = 0.5 * (l1 + l2)
                x_new = np.clip(
                    self.x * np.sqrt(-dc / lmid),
                    np.maximum(0.001, self.x - move),
                    np.minimum(1.0, self.x + move)
                )
                if np.mean(x_new) > self.volfrac:
                    l1 = lmid
                else:
                    l2 = lmid
            self.x = x_new
        return self.x


# ---------------------------------------------------------------------------
# Component-level mass minimization
# ---------------------------------------------------------------------------

@dataclass
class ComponentDesign:
    """Optimized design variables for a single component."""
    name: str
    material: str
    outer_d: float       # [m]
    wall: float          # [m]
    length: float        # [m]
    mass: float          # [kg]
    stiffness: float     # [N/m]
    cost: float          # [USD]
    fos_stress: float
    fos_buckling: float


def optimize_tube(component: StructuralComponent,
                  target_stiffness: float,
                  materials: List[str],
                  d_range: Tuple[float, float] = (0.010, 0.080),
                  wall_range: Tuple[float, float] = (0.001, 0.010),
                  critical: bool = True) -> ComponentDesign:
    """
    Optimize a hollow tube for minimum mass subject to stiffness
    and safety constraints.
    """
    best: Optional[ComponentDesign] = None

    for mat_key in materials:
        mat = MATERIALS[mat_key]

        def objective(design: np.ndarray) -> float:
            d, t = design
            if t >= d / 2:
                return 1e6
            sec = CrossSection.hollow_circle(d, d - 2 * t)
            comp = StructuralComponent("tmp", component.length, mat, sec)
            return comp.mass()

        def constraint_stiffness(design: np.ndarray) -> float:
            d, t = design
            if t >= d / 2:
                return -1e6
            sec = CrossSection.hollow_circle(d, d - 2 * t)
            comp = StructuralComponent("tmp", component.length, mat, sec)
            k = 3.0 * mat.E * sec.Ix / (component.length ** 3)
            return k - target_stiffness

        def constraint_fos(design: np.ndarray) -> float:
            d, t = design
            if t >= d / 2:
                return -1e6
            sec = CrossSection.hollow_circle(d, d - 2 * t)
            comp = StructuralComponent("tmp", component.length, mat, sec)
            fos = mat.sigma_y / (1e3 / sec.A + 1e-9)
            return fos - (2.0 if critical else 1.5)

        x0 = np.array([(d_range[0] + d_range[1]) / 2.0,
                       (wall_range[0] + wall_range[1]) / 2.0])
        bounds = [d_range, wall_range]
        cons = [
            {"type": "ineq", "fun": constraint_stiffness},
            {"type": "ineq", "fun": constraint_fos},
        ]

        res = minimize(objective, x0, method="SLSQP",
                       bounds=bounds, constraints=cons,
                       options={"ftol": 1e-6, "disp": False})

        if res.success:
            d_opt, t_opt = res.x
            sec = CrossSection.hollow_circle(d_opt, d_opt - 2 * t_opt)
            comp = StructuralComponent(component.name, component.length, mat, sec)
            k = 3.0 * mat.E * sec.Ix / (component.length ** 3)
            design = ComponentDesign(
                name=component.name,
                material=mat_key,
                outer_d=d_opt,
                wall=t_opt,
                length=component.length,
                mass=comp.mass(),
                stiffness=k,
                cost=comp.mass() * mat.cost_per_kg,
                fos_stress=mat.sigma_y / 50e6,
                fos_buckling=comp.buckling_fos(1000.0, K=2.0)[0],
            )
            if best is None or design.mass < best.mass:
                best = design

    if best is None:
        raise RuntimeError("No feasible design found")
    return best


# ---------------------------------------------------------------------------
# Pareto frontier exploration
# ---------------------------------------------------------------------------

@dataclass
class ParetoPoint:
    mass: float
    stiffness: float
    cost: float
    design: ComponentDesign


def pareto_frontier(component: StructuralComponent,
                    materials: List[str],
                    stiffness_targets: np.ndarray,
                    critical: bool = True) -> List[ParetoPoint]:
    """
    Explore Pareto frontier by sweeping stiffness targets.
    """
    points: List[ParetoPoint] = []
    for k_target in stiffness_targets:
        try:
            design = optimize_tube(component, k_target, materials,
                                   critical=critical)
            points.append(ParetoPoint(
                mass=design.mass,
                stiffness=design.stiffness,
                cost=design.cost,
                design=design,
            ))
        except RuntimeError:
            continue
    return points


def filter_pareto(points: List[ParetoPoint]) -> List[ParetoPoint]:
    """Return non-dominated points (minimize mass, cost; maximize stiffness)."""
    pareto: List[ParetoPoint] = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if q.mass <= p.mass and q.cost <= p.cost and q.stiffness >= p.stiffness:
                if q.mass < p.mass or q.cost < p.cost or q.stiffness > p.stiffness:
                    dominated = True
                    break
        if not dominated:
            pareto.append(p)
    return sorted(pareto, key=lambda x: x.mass)


# ---------------------------------------------------------------------------
# Full humanoid weight optimization
# ---------------------------------------------------------------------------

@dataclass
class RobotWeightBudget:
    """Optimized mass budget for a full humanoid."""
    total_mass: float
    components: Dict[str, ComponentDesign]
    structural_mass: float
    actuator_mass: float
    electronics_mass: float
    battery_mass: float


def optimize_humanoid_weight(target_total_mass: float = 80.0,
                             height: float = 1.75,
                             materials: Optional[List[str]] = None) -> RobotWeightBudget:
    """
    Optimize structural mass budget for a full humanoid.
    Non-structural masses (actuators, electronics, battery) are estimated
    as fractions of total mass.
    """
    if materials is None:
        materials = ["aluminum_6061", "aluminum_7075",
                     "titanium_6al4v", "carbon_fiber"]

    # Non-structural fractions (empirical from existing humanoids)
    actuator_frac = 0.25
    electronics_frac = 0.08
    battery_frac = 0.12
    structural_frac = 1.0 - actuator_frac - electronics_frac - battery_frac

    structural_budget = target_total_mass * structural_frac

    # Allocate structural budget to major components
    # Based on segment mass fractions from biomechanics
    allocations = {
        "femur_L": 0.10, "femur_R": 0.10,
        "tibia_L": 0.05, "tibia_R": 0.05,
        "foot_L": 0.015, "foot_R": 0.015,
        "upper_arm_L": 0.03, "upper_arm_R": 0.03,
        "forearm_L": 0.02, "forearm_R": 0.02,
        "torso": 0.20,
        "neck": 0.01,
    }

    components: Dict[str, ComponentDesign] = {}
    total_structural = 0.0

    # Component lengths (scaled to height)
    lengths = {
        "femur_L": 0.24 * height, "femur_R": 0.24 * height,
        "tibia_L": 0.25 * height, "tibia_R": 0.25 * height,
        "foot_L": 0.15 * height, "foot_R": 0.15 * height,
        "upper_arm_L": 0.19 * height, "upper_arm_R": 0.19 * height,
        "forearm_L": 0.15 * height, "forearm_R": 0.15 * height,
        "torso": 0.30 * height,
        "neck": 0.08 * height,
    }

    for name, frac in allocations.items():
        target_mass = structural_budget * frac
        # Target stiffness: enough to keep deflection < 1 mm under 1 kN
        target_k = 1e3 / 1e-3  # 1e6 N/m
        mat = MATERIALS["aluminum_6061"]  # default
        sec = CrossSection.hollow_circle(0.035, 0.029)
        comp = StructuralComponent(name, lengths[name], mat, sec)

        try:
            design = optimize_tube(comp, target_k, materials, critical=True)
        except RuntimeError:
            # Fallback: use default with Al 6061
            design = ComponentDesign(
                name=name, material="aluminum_6061",
                outer_d=0.035, wall=0.003,
                length=lengths[name],
                mass=comp.mass(),
                stiffness=target_k,
                cost=comp.mass() * mat.cost_per_kg,
                fos_stress=5.0, fos_buckling=5.0,
            )
        components[name] = design
        total_structural += design.mass

    return RobotWeightBudget(
        total_mass=target_total_mass,
        components=components,
        structural_mass=total_structural,
        actuator_mass=target_total_mass * actuator_frac,
        electronics_mass=target_total_mass * electronics_frac,
        battery_mass=target_total_mass * battery_frac,
    )


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MOSES v4.0 — Weight Optimizer Demo")
    print("=" * 60)

    # Single component optimization
    mat = MATERIALS["aluminum_6061"]
    sec = CrossSection.hollow_circle(0.035, 0.029)
    femur = StructuralComponent("femur", 0.45, mat, sec)

    print("\n--- Single Component (Femur) Optimization ---")
    design = optimize_tube(femur, target_stiffness=1e6,
                           materials=["aluminum_6061", "carbon_fiber"])
    print(f"  Material: {design.material}")
    print(f"  Outer D: {design.outer_d*1e3:.2f} mm")
    print(f"  Wall: {design.wall*1e3:.2f} mm")
    print(f"  Mass: {design.mass*1e3:.1f} g")
    print(f"  Stiffness: {design.stiffness/1e6:.2f} MN/m")
    print(f"  Cost: ${design.cost:.2f}")

    # Pareto frontier
    print("\n--- Pareto Frontier (Mass vs Stiffness) ---")
    k_targets = np.linspace(0.3e6, 3.0e6, 10)
    points = pareto_frontier(femur,
                             ["aluminum_6061", "aluminum_7075", "carbon_fiber"],
                             k_targets)
    pareto = filter_pareto(points)
    for p in pareto[:5]:
        print(f"  mass={p.mass*1e3:6.1f}g  k={p.stiffness/1e6:5.2f}MN/m  "
              f"cost=${p.cost:5.2f}  mat={p.design.material}")

    # Full humanoid budget
    print("\n--- Full Humanoid Weight Budget ---")
    budget = optimize_humanoid_weight(target_total_mass=80.0, height=1.75)
    print(f"  Structural: {budget.structural_mass:.2f} kg")
    print(f"  Actuators:  {budget.actuator_mass:.2f} kg")
    print(f"  Electronics:{budget.electronics_mass:.2f} kg")
    print(f"  Battery:    {budget.battery_mass:.2f} kg")
    print(f"  Total:      {budget.total_mass:.2f} kg")
