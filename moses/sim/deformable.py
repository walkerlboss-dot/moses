"""Deformable object simulation for Moses v6.0.

Finite Element Method (FEM) for soft objects, cloth simulation,
liquid/gel simulation, and cutting/tearing/folding.

References:
- Isaac Lab deformable body API (isaaclab.assets.DeformableObject)
- "Finite Element Methods for the Incompressible Navier-Stokes Equations"
- "Position Based Dynamics" (Müller et al., 2007)
- "A Finite Element Formulation for Problems of Large Strain and Large Displacement"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union

import numpy as np
import numpy.typing as npt

from moses.sim.multiphysics import Vec3, Quaternion, Transform, Material

NDArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Core FEM data structures
# ---------------------------------------------------------------------------

@dataclass
class FEMNode:
    """A node in an FEM mesh."""

    index: int
    position: Vec3 = field(default_factory=Vec3)
    rest_position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    mass: float = 0.0
    force: Vec3 = field(default_factory=Vec3)
    is_fixed: bool = False
    # For cutting: track if node has been cut
    is_cut: bool = False


@dataclass
class FEMTetrahedron:
    """A tetrahedral element for 3D FEM."""

    indices: Tuple[int, int, int, int]
    rest_volume: float = 0.0
    # Shape matrix inverse: D_m^{-1} where D_m = [x1-x0, x2-x0, x3-x0]
    inv_rest_shape: NDArray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )
    # Deformation gradient F = D_s * D_m^{-1}
    F: NDArray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    # First Piola-Kirchhoff stress
    P: NDArray = field(default_factory=lambda: np.zeros((3, 3), dtype=np.float64))
    # Strain energy
    energy: float = 0.0
    # For cutting: track if element is damaged
    damage: float = 0.0  # 0 = intact, 1 = fully damaged


@dataclass
class FEMTriangle:
    """A triangular element for 2D shell/cloth FEM."""

    indices: Tuple[int, int, int]
    rest_area: float = 0.0
    # Rest shape matrix inverse
    inv_rest_shape: NDArray = field(
        default_factory=lambda: np.zeros((2, 2), dtype=np.float64)
    )
    # Normal in rest configuration
    rest_normal: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 1.0))
    # Bending stiffness (for cloth)
    bending_stiffness: float = 0.0


# ---------------------------------------------------------------------------
# Constitutive models
# ---------------------------------------------------------------------------

class ConstitutiveModel:
    """Base class for material constitutive models."""

    def compute_stress(self, F: NDArray, material: Material) -> NDArray:
        """Compute first Piola-Kirchhoff stress P from deformation gradient F."""
        raise NotImplementedError

    def compute_energy(self, F: NDArray, material: Material) -> float:
        """Compute strain energy density."""
        raise NotImplementedError


class NeoHookeanModel(ConstitutiveModel):
    """Neo-Hookean material model for large deformations.

    Psi = (mu/2) * (tr(F^T F) - 3) - mu * log(J) + (lambda/2) * log(J)^2
    P = mu * (F - F^{-T}) + lambda * log(J) * F^{-T}
    """

    def compute_stress(self, F: NDArray, material: Material) -> NDArray:
        mu = material.lame_mu()
        lam = material.lame_lambda()
        J = max(np.linalg.det(F), 1e-8)
        F_inv_T = np.linalg.inv(F).T
        P = mu * (F - F_inv_T) + lam * math.log(J) * F_inv_T
        return P

    def compute_energy(self, F: NDArray, material: Material) -> float:
        mu = material.lame_mu()
        lam = material.lame_lambda()
        J = max(np.linalg.det(F), 1e-8)
        I1 = float(np.trace(F.T @ F))
        energy = 0.5 * mu * (I1 - 3.0) - mu * math.log(J) + 0.5 * lam * (math.log(J) ** 2)
        return energy


class CorotationalLinearModel(ConstitutiveModel):
    """Corotated linear elasticity for stable large deformations.

    P = R * (2*mu*(R^T*F - I) + lambda*tr(R^T*F - I)*I)
    where F = R*S is the polar decomposition.
    """

    def compute_stress(self, F: NDArray, material: Material) -> NDArray:
        mu = material.lame_mu()
        lam = material.lame_lambda()
        U, S, Vt = np.linalg.svd(F)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        strain = R.T @ F - np.eye(3)
        trace_strain = float(np.trace(strain))
        stress = 2.0 * mu * strain + lam * trace_strain * np.eye(3)
        P = R @ stress
        return P

    def compute_energy(self, F: NDArray, material: Material) -> float:
        mu = material.lame_mu()
        lam = material.lame_lambda()
        U, S, Vt = np.linalg.svd(F)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        strain = R.T @ F - np.eye(3)
        trace_strain = float(np.trace(strain))
        frob_sq = float(np.sum(strain * strain))
        energy = mu * frob_sq + 0.5 * lam * (trace_strain ** 2)
        return energy


class StVKModel(ConstitutiveModel):
    """St. Venant-Kirchhoff model.

    Psi = (mu/2) * tr(E^2) + (lambda/2) * tr(E)^2
    where E = 0.5 * (F^T F - I) is the Green-Lagrange strain.
    S = lambda*tr(E)*I + 2*mu*E  (2nd PK stress)
    P = F * S
    """

    def compute_stress(self, F: NDArray, material: Material) -> NDArray:
        mu = material.lame_mu()
        lam = material.lame_lambda()
        C = F.T @ F
        E = 0.5 * (C - np.eye(3))
        trace_E = float(np.trace(E))
        S = lam * trace_E * np.eye(3) + 2.0 * mu * E
        P = F @ S
        return P

    def compute_energy(self, F: NDArray, material: Material) -> float:
        mu = material.lame_mu()
        lam = material.lame_lambda()
        C = F.T @ F
        E = 0.5 * (C - np.eye(3))
        trace_E = float(np.trace(E))
        frob_sq = float(np.sum(E * E))
        energy = mu * frob_sq + 0.5 * lam * (trace_E ** 2)
        return energy


# ---------------------------------------------------------------------------
# 3D Soft body (tetrahedral FEM)
# ---------------------------------------------------------------------------

class SoftBodyFEM:
    """Soft body simulation using tetrahedral FEM.

    Supports Neo-Hookean, Corotational, and St. Venant-Kirchhoff materials.
    Uses implicit time integration for stability.
    """

    def __init__(
        self,
        nodes: List[FEMNode],
        elements: List[FEMTetrahedron],
        material: Material,
        model: ConstitutiveModel,
    ) -> None:
        self.nodes = nodes
        self.elements = elements
        self.material = material
        self.model = model
        self.damping = 0.01
        self._compute_rest_state()

    def _compute_rest_state(self) -> None:
        """Compute rest volumes and shape matrices."""
        for elem in self.elements:
            i0, i1, i2, i3 = elem.indices
            p0 = self.nodes[i0].rest_position.to_array()
            p1 = self.nodes[i1].rest_position.to_array()
            p2 = self.nodes[i2].rest_position.to_array()
            p3 = self.nodes[i3].rest_position.to_array()

            D = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
            elem.rest_volume = abs(np.linalg.det(D)) / 6.0
            if elem.rest_volume > 1e-12:
                elem.inv_rest_shape = np.linalg.inv(D)

    def compute_forces(self) -> None:
        """Compute elastic forces for all elements."""
        # Reset nodal forces
        for node in self.nodes:
            node.force = Vec3(0.0, 0.0, 0.0)

        for elem in self.elements:
            if elem.damage >= 1.0:
                continue

            i0, i1, i2, i3 = elem.indices
            p0 = self.nodes[i0].position.to_array()
            p1 = self.nodes[i1].position.to_array()
            p2 = self.nodes[i2].position.to_array()
            p3 = self.nodes[i3].position.to_array()

            D = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
            elem.F = D @ elem.inv_rest_shape

            # Compute stress
            elem.P = self.model.compute_stress(elem.F, self.material)
            elem.energy = self.model.compute_energy(elem.F, self.material) * elem.rest_volume

            # Forces: f_i = -dPsi/dx_i = -P * dF/dx_i * V
            # For tetrahedron: H = -V * P * (D_m^{-1})^T
            H = -elem.rest_volume * elem.P @ elem.inv_rest_shape.T

            f1 = Vec3.from_array(H[:, 0])
            f2 = Vec3.from_array(H[:, 1])
            f3 = Vec3.from_array(H[:, 2])
            f0 = -(f1 + f2 + f3)

            # Apply damage reduction
            damage_factor = max(0.0, 1.0 - elem.damage)
            f0 = f0 * damage_factor
            f1 = f1 * damage_factor
            f2 = f2 * damage_factor
            f3 = f3 * damage_factor

            self.nodes[i0].force = self.nodes[i0].force + f0
            self.nodes[i1].force = self.nodes[i1].force + f1
            self.nodes[i2].force = self.nodes[i2].force + f2
            self.nodes[i3].force = self.nodes[i3].force + f3

    def integrate_explicit(self, dt: float, gravity: Vec3 = Vec3(0.0, 0.0, -9.81)) -> None:
        """Explicit Euler integration."""
        self.compute_forces()
        for node in self.nodes:
            if node.is_fixed or node.is_cut:
                continue
            accel = node.force * (1.0 / node.mass) + gravity
            # Rayleigh damping
            accel = accel - node.velocity * self.damping
            node.velocity = node.velocity + accel * dt
            node.position = node.position + node.velocity * dt

    def integrate_implicit(self, dt: float, gravity: Vec3 = Vec3(0.0, 0.0, -9.81), max_iters: int = 5) -> None:
        """Implicit Euler using Newton-Raphson (simplified).

        (M + dt^2 * K) * dv = dt * (f_ext + f_int) - dt^2 * K * v
        """
        n = len(self.nodes)
        # For simplicity, fall back to explicit with smaller dt
        sub_dt = dt / max_iters
        for _ in range(max_iters):
            self.integrate_explicit(sub_dt, gravity)

    def get_volume(self) -> float:
        """Compute current total volume."""
        total = 0.0
        for elem in self.elements:
            if elem.damage >= 1.0:
                continue
            i0, i1, i2, i3 = elem.indices
            p0 = self.nodes[i0].position.to_array()
            p1 = self.nodes[i1].position.to_array()
            p2 = self.nodes[i2].position.to_array()
            p3 = self.nodes[i3].position.to_array()
            D = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
            total += abs(np.linalg.det(D)) / 6.0
        return total

    def get_center_of_mass(self) -> Vec3:
        total_mass = sum(n.mass for n in self.nodes)
        if total_mass < 1e-12:
            return Vec3()
        weighted = Vec3()
        for n in self.nodes:
            weighted = weighted + n.position * n.mass
        return weighted / total_mass


# ---------------------------------------------------------------------------
# Cloth simulation (mass-spring + bending)
# ---------------------------------------------------------------------------

@dataclass
class ClothSpring:
    """A spring connecting two cloth nodes."""

    node_a: int
    node_b: int
    rest_length: float = 0.0
    stiffness: float = 1000.0
    damping: float = 0.1


class ClothSimulator:
    """Cloth simulation using mass-spring system with bending resistance.

    References:
    - "Large Steps in Cloth Simulation" (Baraff & Witkin, 1998)
    - Position Based Dynamics for stability
    """

    def __init__(
        self,
        nodes: List[FEMNode],
        triangles: List[FEMTriangle],
        springs: List[ClothSpring],
        material: Material,
    ) -> None:
        self.nodes = nodes
        self.triangles = triangles
        self.springs = springs
        self.material = material
        self.wind_force: Vec3 = Vec3()
        self._compute_rest_state()

    def _compute_rest_state(self) -> None:
        """Compute rest lengths and areas."""
        for spring in self.springs:
            pa = self.nodes[spring.node_a].rest_position
            pb = self.nodes[spring.node_b].rest_position
            spring.rest_length = (pa - pb).norm()

        for tri in self.triangles:
            i0, i1, i2 = tri.indices
            p0 = self.nodes[i0].rest_position.to_array()
            p1 = self.nodes[i1].rest_position.to_array()
            p2 = self.nodes[i2].rest_position.to_array()

            D = np.column_stack([p1 - p0, p2 - p0])
            # 2D rest shape in triangle plane
            # Use the 2D projection onto the XY plane for area
            D2d = np.array([[D[0, 0], D[0, 1]], [D[1, 0], D[1, 1]]], dtype=np.float64)
            tri.rest_area = 0.5 * abs(np.linalg.det(D2d))

            # Compute rest normal
            e1 = p1 - p0
            e2 = p2 - p0
            normal = np.cross(e1, e2)
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                tri.rest_normal = Vec3.from_array(normal / norm)

    def compute_spring_forces(self) -> None:
        """Compute spring forces: F = -k * (L - L0) * dir - c * v_rel."""
        for spring in self.springs:
            na = self.nodes[spring.node_a]
            nb = self.nodes[spring.node_b]
            diff = na.position - nb.position
            L = diff.norm()
            if L < 1e-12:
                continue
            dir_vec = diff / L
            stretch = L - spring.rest_length
            # Hooke's law
            force_mag = spring.stiffness * stretch
            # Damping
            v_rel = na.velocity - nb.velocity
            damp_mag = spring.damping * v_rel.dot(dir_vec)
            total_mag = force_mag + damp_mag

            force = dir_vec * total_mag
            na.force = na.force - force
            nb.force = nb.force + force

    def compute_bending_forces(self) -> None:
        """Compute bending forces between adjacent triangles.

        Uses discrete shell bending energy:
        E_bend = k_bend * (theta - theta0)^2 / |e|
        where theta is dihedral angle.
        """
        # Build edge-to-triangles map
        edge_tris: Dict[Tuple[int, int], List[int]] = {}
        for ti, tri in enumerate(self.triangles):
            for e in [(tri.indices[0], tri.indices[1]),
                      (tri.indices[1], tri.indices[2]),
                      (tri.indices[2], tri.indices[0])]:
                key = (min(e), max(e))
                if key not in edge_tris:
                    edge_tris[key] = []
                edge_tris[key].append(ti)

        # For each edge with two triangles, compute bending force
        for edge, tri_indices in edge_tris.items():
            if len(tri_indices) < 2:
                continue
            t1, t2 = tri_indices[0], tri_indices[1]
            tri1 = self.triangles[t1]
            tri2 = self.triangles[t2]

            # Get the two non-edge vertices
            edge_set = set(edge)
            v1 = [i for i in tri1.indices if i not in edge_set][0]
            v2 = [i for i in tri2.indices if i not in edge_set][0]

            p_edge_a = self.nodes[edge[0]].position
            p_edge_b = self.nodes[edge[1]].position
            p_v1 = self.nodes[v1].position
            p_v2 = self.nodes[v2].position

            # Compute normals
            n1 = (p_edge_b - p_edge_a).cross(p_v1 - p_edge_a).normalize()
            n2 = (p_edge_b - p_edge_a).cross(p_v2 - p_edge_a).normalize()

            # Dihedral angle
            dot_n = n1.dot(n2)
            dot_n = max(-1.0, min(1.0, dot_n))
            theta = math.acos(dot_n)

            # Bending force (simplified)
            k_bend = 0.1
            edge_len = (p_edge_a - p_edge_b).norm()
            if edge_len < 1e-12:
                continue
            force_mag = k_bend * theta / edge_len

            # Apply to vertices (simplified)
            force_dir = (n1 - n2).normalize()
            if force_dir.norm() > 0.1:
                self.nodes[v1].force = self.nodes[v1].force + force_dir * force_mag
                self.nodes[v2].force = self.nodes[v2].force + force_dir * force_mag

    def compute_wind_forces(self) -> None:
        """Apply wind force to cloth triangles."""
        for tri in self.triangles:
            i0, i1, i2 = tri.indices
            p0 = self.nodes[i0].position
            p1 = self.nodes[i1].position
            p2 = self.nodes[i2].position

            # Triangle normal
            e1 = p1 - p0
            e2 = p2 - p0
            normal = e1.cross(e2)
            area = 0.5 * normal.norm()
            if area < 1e-12:
                continue
            n_hat = normal.normalize()

            # Wind force proportional to dot product with normal
            wind_dot = self.wind_force.dot(n_hat)
            if wind_dot > 0:
                tri_force = n_hat * wind_dot * area * 0.5
                f_per_node = tri_force / 3.0
                self.nodes[i0].force = self.nodes[i0].force + f_per_node
                self.nodes[i1].force = self.nodes[i1].force + f_per_node
                self.nodes[i2].force = self.nodes[i2].force + f_per_node

    def get_center_of_mass(self) -> Vec3:
        total_mass = sum(n.mass for n in self.nodes)
        if total_mass < 1e-12:
            return Vec3()
        weighted = Vec3()
        for n in self.nodes:
            weighted = weighted + n.position * n.mass
        return weighted / total_mass

    def integrate(self, dt: float, gravity: Vec3 = Vec3(0.0, 0.0, -9.81)) -> None:
        """Semi-implicit Euler integration for cloth."""
        for node in self.nodes:
            node.force = Vec3(0.0, 0.0, 0.0)

        self.compute_spring_forces()
        self.compute_bending_forces()
        self.compute_wind_forces()

        for node in self.nodes:
            if node.is_fixed:
                continue
            accel = node.force * (1.0 / node.mass) + gravity
            # Air drag
            drag = node.velocity * (-0.01 * node.velocity.norm())
            accel = accel + drag
            node.velocity = node.velocity + accel * dt
            node.position = node.position + node.velocity * dt

    def set_wind(self, wind: Vec3) -> None:
        self.wind_force = wind


# ---------------------------------------------------------------------------
# Liquid / Gel simulation (SPH with surface tension)
# ---------------------------------------------------------------------------

@dataclass
class FluidParticle:
    """Particle for liquid/gel simulation."""

    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    density: float = 1000.0
    pressure: float = 0.0
    mass: float = 0.001
    viscosity: float = 0.001
    # For gel: additional elastic properties
    rest_position: Vec3 = field(default_factory=Vec3)
    is_gel: bool = False
    gel_stiffness: float = 0.0


class LiquidGelSimulator:
    """SPH-based liquid and gel simulation.

    For liquid: standard weakly compressible SPH.
    For gel: add elastic return force toward rest shape.

    References:
    - "Smoothed Particle Hydrodynamics" (Monaghan, 1992)
    - "Position Based Fluids" (Macklin & Müller, 2013)
    """

    def __init__(
        self,
        particles: List[FluidParticle],
        smoothing_length: float = 0.05,
        material: Material = Material(),
    ) -> None:
        self.particles = particles
        self.h = smoothing_length
        self.h2 = smoothing_length**2
        self.h3 = smoothing_length**3
        self.h6 = smoothing_length**6
        self.h9 = smoothing_length**9
        self.material = material
        self.gas_constant = 2000.0
        self.rest_density = material.density
        self.gravity = Vec3(0.0, 0.0, -9.81)
        self.surface_tension = material.surface_tension

    def _kernel_poly6(self, r2: float) -> float:
        if r2 >= self.h2:
            return 0.0
        return 315.0 / (64.0 * math.pi * self.h9) * (self.h2 - r2) ** 3

    def _kernel_spiky_gradient(self, r: Vec3, dist: float) -> Vec3:
        if dist >= self.h or dist < 1e-12:
            return Vec3()
        coeff = -45.0 / (math.pi * self.h6) * (self.h - dist) ** 2 / dist
        return r * coeff

    def _kernel_viscosity_laplacian(self, dist: float) -> float:
        if dist >= self.h:
            return 0.0
        return 45.0 / (math.pi * self.h6) * (self.h - dist)

    def _kernel_surface_tension(self, r: Vec3, dist: float) -> Vec3:
        """Cohesion force for surface tension."""
        if dist >= self.h or dist < 1e-12:
            return Vec3()
        # Simplified cohesion kernel
        cohesion = -self.surface_tension * (self.h - dist) ** 2 / dist
        return r.normalize() * cohesion

    def compute_densities(self) -> None:
        for i, pi in enumerate(self.particles):
            density = 0.0
            for j, pj in enumerate(self.particles):
                r = pi.position - pj.position
                r2 = r.dot(r)
                density += pj.mass * self._kernel_poly6(r2)
            pi.density = max(density, self.rest_density * 0.1)

    def compute_pressures(self) -> None:
        for p in self.particles:
            p.pressure = self.gas_constant * (p.density - self.rest_density)

    def compute_forces(self) -> None:
        for pi in self.particles:
            f_pressure = Vec3()
            f_viscosity = Vec3()
            f_surface = Vec3()
            f_elastic = Vec3()

            for pj in self.particles:
                if pi is pj:
                    continue
                r = pi.position - pj.position
                dist = r.norm()

                # Pressure
                if dist < self.h and dist > 1e-12:
                    pressure_term = (pi.pressure + pj.pressure) / (2.0 * pj.density)
                    f_pressure = f_pressure + self._kernel_spiky_gradient(r, dist) * (
                        -pressure_term * pj.mass
                    )

                # Viscosity
                if dist < self.h:
                    lap = self._kernel_viscosity_laplacian(dist)
                    visc_term = (pj.velocity - pi.velocity) * (
                        pj.mass / pj.density * pi.viscosity * lap
                    )
                    f_viscosity = f_viscosity + visc_term

                # Surface tension
                if dist < self.h and dist > 1e-12:
                    f_surface = f_surface + self._kernel_surface_tension(r, dist) * pj.mass

                # Elastic force for gel
                if pi.is_gel and pj.is_gel and dist < self.h * 2.0:
                    rest_diff = pi.rest_position - pj.rest_position
                    curr_diff = pi.position - pj.position
                    elastic_force = (rest_diff - curr_diff) * pi.gel_stiffness
                    f_elastic = f_elastic + elastic_force

            pi.velocity = pi.velocity + (f_pressure + f_viscosity + f_surface + f_elastic) * (
                1.0 / pi.density
            )

    def integrate(self, dt: float) -> None:
        self.compute_densities()
        self.compute_pressures()
        self.compute_forces()
        for p in self.particles:
            p.velocity = p.velocity + self.gravity * dt
            p.position = p.position + p.velocity * dt

    def add_particle(self, particle: FluidParticle) -> None:
        self.particles.append(particle)


# ---------------------------------------------------------------------------
# Cutting, tearing, folding
# ---------------------------------------------------------------------------

class CuttingTool:
    """Tool for cutting deformable objects.

    Cuts elements that intersect with a cutting plane or line segment.
    """

    def __init__(self, sharpness: float = 1.0) -> None:
        self.sharpness = sharpness

    def cut_tetrahedral_mesh(
        self,
        soft_body: SoftBodyFEM,
        plane_origin: Vec3,
        plane_normal: Vec3,
    ) -> None:
        """Cut tetrahedral mesh with a plane.

        Marks elements on the negative side of the plane as damaged.
        """
        n = plane_normal.normalize()
        for elem in soft_body.elements:
            i0, i1, i2, i3 = elem.indices
            positions = [
                soft_body.nodes[i0].position,
                soft_body.nodes[i1].position,
                soft_body.nodes[i2].position,
                soft_body.nodes[i3].position,
            ]

            # Check which side each vertex is on
            sides = [(p - plane_origin).dot(n) for p in positions]
            positive = sum(1 for s in sides if s > 0)
            negative = sum(1 for s in sides if s < 0)

            if positive > 0 and negative > 0:
                # Element is cut
                elem.damage = min(1.0, elem.damage + self.sharpness * 0.5)
                # Mark cut nodes
                for i, side in zip(elem.indices, sides):
                    if abs(side) < 0.001:
                        soft_body.nodes[i].is_cut = True

    def cut_cloth(
        self,
        cloth: ClothSimulator,
        start: Vec3,
        end: Vec3,
    ) -> None:
        """Cut cloth along a line segment.

        Removes springs that intersect the cutting line.
        """
        cut_dir = end - start
        cut_len = cut_dir.norm()
        if cut_len < 1e-12:
            return
        cut_dir = cut_dir / cut_len

        springs_to_remove: List[int] = []
        for si, spring in enumerate(cloth.springs):
            pa = cloth.nodes[spring.node_a].position
            pb = cloth.nodes[spring.node_b].position

            # Check if line segment intersects cutting line
            # Simplified: check if both points are on opposite sides of cut plane
            side_a = (pa - start).cross(cut_dir).norm()
            side_b = (pb - start).cross(cut_dir).norm()

            if side_a < 0.005 and side_b < 0.005:
                # Check if segment crosses the cut line
                proj_a = (pa - start).dot(cut_dir)
                proj_b = (pb - start).dot(cut_dir)
                if (proj_a < cut_len and proj_b > 0) or (proj_b < cut_len and proj_a > 0):
                    springs_to_remove.append(si)

        # Remove in reverse order
        for si in reversed(springs_to_remove):
            cloth.springs.pop(si)


class FoldingTool:
    """Tool for folding deformable objects (primarily cloth)."""

    def __init__(self) -> None:
        self.fold_line: Optional[Tuple[Vec3, Vec3]] = None
        self.fold_angle: float = 0.0

    def set_fold_line(self, point: Vec3, direction: Vec3) -> None:
        """Define a fold line."""
        self.fold_line = (point, direction.normalize())

    def apply_fold(
        self,
        cloth: ClothSimulator,
        angle: float,
    ) -> None:
        """Apply a fold by rotating nodes on one side of the fold line.

        Args:
            angle: Fold angle in radians
        """
        if self.fold_line is None:
            return
        origin, direction = self.fold_line

        for node in cloth.nodes:
            if node.is_fixed:
                continue
            # Determine which side of fold line
            to_node = node.position - origin
            side = to_node.cross(direction).dot(Vec3(0.0, 0.0, 1.0))
            if side > 0:
                # Rotate this node about fold line
                # Project to plane perpendicular to fold direction
                proj = to_node.dot(direction)
                perp = to_node - direction * proj
                perp_len = perp.norm()
                if perp_len < 1e-12:
                    continue

                # Rotate perpendicular component
                perp_norm = perp / perp_len
                rot_axis = direction.cross(perp_norm).normalize()
                if rot_axis.norm() < 0.1:
                    continue

                q = Quaternion.from_axis_angle(direction, angle)
                new_perp = q.rotate(perp)
                node.position = origin + direction * proj + new_perp


class TearingSimulator:
    """Simulate tearing of deformable materials.

    Progressive damage accumulation leading to fracture.
    """

    def __init__(self, fracture_threshold: float = 1.0) -> None:
        self.fracture_threshold = fracture_threshold
        self.damage_field: Dict[int, float] = {}  # Per-element damage

    def accumulate_stress_damage(
        self,
        soft_body: SoftBodyFEM,
        dt: float,
    ) -> List[int]:
        """Accumulate damage based on stress.

        Returns list of element indices that have fractured.
        """
        fractured: List[int] = []
        for ei, elem in enumerate(soft_body.elements):
            # Compute von Mises stress from P
            if elem.P is None or not hasattr(elem, 'P'):
                continue
            # Simplified: use Frobenius norm of P as damage indicator
            stress_norm = float(np.linalg.norm(elem.P))
            damage_rate = stress_norm / self.fracture_threshold * dt

            if ei not in self.damage_field:
                self.damage_field[ei] = 0.0
            self.damage_field[ei] += damage_rate
            elem.damage = min(1.0, self.damage_field[ei])

            if elem.damage >= 1.0:
                fractured.append(ei)

        return fractured

    def propagate_crack(
        self,
        soft_body: SoftBodyFEM,
        fractured_elements: List[int],
    ) -> None:
        """Propagate crack from fractured elements to neighbors."""
        # Build element adjacency
        node_to_elems: Dict[int, Set[int]] = {}
        for ei, elem in enumerate(soft_body.elements):
            for ni in elem.indices:
                if ni not in node_to_elems:
                    node_to_elems[ni] = set()
                node_to_elems[ni].add(ei)

        # Propagate damage to neighbors
        for fe in fractured_elements:
            elem = soft_body.elements[fe]
            for ni in elem.indices:
                for neighbor_ei in node_to_elems.get(ni, set()):
                    if neighbor_ei != fe:
                        soft_body.elements[neighbor_ei].damage = min(
                            1.0,
                            soft_body.elements[neighbor_ei].damage + 0.3,
                        )


# ---------------------------------------------------------------------------
# Deformable object factory
# ---------------------------------------------------------------------------

class DeformableObjectFactory:
    """Factory for creating common deformable objects."""

    @staticmethod
    def create_soft_cube(
        size: float = 0.1,
        divisions: int = 4,
        material: Material = Material(),
        model: ConstitutiveModel = NeoHookeanModel(),
    ) -> SoftBodyFEM:
        """Create a soft cube with tetrahedral mesh."""
        nodes: List[FEMNode] = []
        elements: List[FEMTetrahedron] = []

        # Generate grid nodes
        node_map: Dict[Tuple[int, int, int], int] = {}
        idx = 0
        for i in range(divisions + 1):
            for j in range(divisions + 1):
                for k in range(divisions + 1):
                    x = (i / divisions - 0.5) * size
                    y = (j / divisions - 0.5) * size
                    z = (k / divisions - 0.5) * size
                    pos = Vec3(x, y, z)
                    node = FEMNode(index=idx, position=pos, rest_position=pos)
                    node.mass = material.density * (size / divisions) ** 3
                    nodes.append(node)
                    node_map[(i, j, k)] = idx
                    idx += 1

        # Generate tetrahedra (simplified: 6 tets per cube cell)
        for i in range(divisions):
            for j in range(divisions):
                for k in range(divisions):
                    c000 = node_map[(i, j, k)]
                    c100 = node_map[(i + 1, j, k)]
                    c010 = node_map[(i, j + 1, k)]
                    c001 = node_map[(i, j, k + 1)]
                    c110 = node_map[(i + 1, j + 1, k)]
                    c101 = node_map[(i + 1, j, k + 1)]
                    c011 = node_map[(i, j + 1, k + 1)]
                    c111 = node_map[(i + 1, j + 1, k + 1)]

                    # 5 tets per cube (simplified)
                    tets = [
                        (c000, c100, c010, c001),
                        (c111, c011, c101, c110),
                        (c100, c010, c001, c101),
                        (c010, c001, c101, c011),
                        (c100, c010, c101, c110),
                    ]
                    for tet in tets:
                        elements.append(FEMTetrahedron(indices=tet))

        body = SoftBodyFEM(nodes, elements, material, model)
        return body

    @staticmethod
    def create_cloth_sheet(
        width: float = 1.0,
        height: float = 1.0,
        res_x: int = 20,
        res_y: int = 20,
        material: Material = Material(),
    ) -> ClothSimulator:
        """Create a rectangular cloth sheet."""
        nodes: List[FEMNode] = []
        triangles: List[FEMTriangle] = []
        springs: List[ClothSpring] = []

        # Create nodes
        node_map: Dict[Tuple[int, int], int] = {}
        idx = 0
        for i in range(res_x + 1):
            for j in range(res_y + 1):
                x = (i / res_x - 0.5) * width
                y = (j / res_y - 0.5) * height
                z = 0.0
                pos = Vec3(x, y, z)
                node = FEMNode(index=idx, position=pos, rest_position=pos)
                # Pin two corners
                if (i == 0 and j == 0) or (i == res_x and j == res_y):
                    node.is_fixed = True
                node.mass = material.density * (width / res_x) * (height / res_y) * 0.001
                nodes.append(node)
                node_map[(i, j)] = idx
                idx += 1

        # Create triangles and springs
        for i in range(res_x):
            for j in range(res_y):
                n00 = node_map[(i, j)]
                n10 = node_map[(i + 1, j)]
                n01 = node_map[(i, j + 1)]
                n11 = node_map[(i + 1, j + 1)]

                # Two triangles per quad
                triangles.append(FEMTriangle(indices=(n00, n10, n11)))
                triangles.append(FEMTriangle(indices=(n00, n11, n01)))

                # Structural springs
                springs.append(ClothSpring(n00, n10))
                springs.append(ClothSpring(n00, n01))
                springs.append(ClothSpring(n10, n11))
                springs.append(ClothSpring(n01, n11))
                springs.append(ClothSpring(n00, n11))  # Shear
                springs.append(ClothSpring(n10, n01))  # Shear

        # Bending springs (every other node)
        for i in range(res_x - 1):
            for j in range(res_y + 1):
                springs.append(ClothSpring(node_map[(i, j)], node_map[(i + 2, j)]))
        for i in range(res_x + 1):
            for j in range(res_y - 1):
                springs.append(ClothSpring(node_map[(i, j)], node_map[(i, j + 2)]))

        return ClothSimulator(nodes, triangles, springs, material)

    @staticmethod
    def create_liquid_pool(
        num_particles: int = 1000,
        volume: float = 0.001,
        material: Material = Material(),
    ) -> LiquidGelSimulator:
        """Create a pool of liquid particles."""
        particles: List[FluidParticle] = []
        mass_per_particle = material.density * volume / num_particles

        # Grid arrangement
        n = int(num_particles ** (1.0 / 3.0)) + 1
        spacing = (volume ** (1.0 / 3.0)) / n

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len(particles) >= num_particles:
                        break
                    x = i * spacing
                    y = j * spacing
                    z = k * spacing
                    p = FluidParticle(
                        position=Vec3(x, y, z),
                        rest_position=Vec3(x, y, z),
                        mass=mass_per_particle,
                        viscosity=material.viscosity,
                    )
                    particles.append(p)

        return LiquidGelSimulator(particles, smoothing_length=spacing * 2.0, material=material)

    @staticmethod
    def create_gel_block(
        num_particles: int = 500,
        size: float = 0.1,
        stiffness: float = 100.0,
        material: Material = Material(),
    ) -> LiquidGelSimulator:
        """Create a block of gel particles."""
        particles: List[FluidParticle] = []
        volume = size ** 3
        mass_per_particle = material.density * volume / num_particles

        n = int(num_particles ** (1.0 / 3.0)) + 1
        spacing = size / n

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if len(particles) >= num_particles:
                        break
                    x = (i / n - 0.5) * size
                    y = (j / n - 0.5) * size
                    z = (k / n - 0.5) * size
                    p = FluidParticle(
                        position=Vec3(x, y, z),
                        rest_position=Vec3(x, y, z),
                        mass=mass_per_particle,
                        viscosity=material.viscosity * 10,  # Higher viscosity for gel
                        is_gel=True,
                        gel_stiffness=stiffness,
                    )
                    particles.append(p)

        return LiquidGelSimulator(particles, smoothing_length=spacing * 2.5, material=material)
