"""Multi-physics simulation for Moses v6.0.

Coupled rigid body, soft body, fluid, thermal, and electromagnetic dynamics.
References Isaac Lab sensor APIs and NVIDIA PhysX 5.x multi-physics features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class Vec3:
    """3D vector with physics operations."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, s: float) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s: float) -> Vec3:
        return self.__mul__(s)

    def __truediv__(self, s: float) -> Vec3:
        return Vec3(self.x / s, self.y / s, self.z / s)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> Vec3:
        n = self.norm()
        if n < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        return self / n

    def to_array(self) -> NDArray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray) -> Vec3:
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass
class Quaternion:
    """Unit quaternion for 3D rotations."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __mul__(self, other: Quaternion) -> Quaternion:
        return Quaternion(
            w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def conjugate(self) -> Quaternion:
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def rotate(self, v: Vec3) -> Vec3:
        """Rotate a vector by this quaternion: q * v * q^-1."""
        qv = Quaternion(0.0, v.x, v.y, v.z)
        q_conj = self.conjugate()
        result = self * qv * q_conj
        return Vec3(result.x, result.y, result.z)

    def to_rotation_matrix(self) -> NDArray:
        """Convert to 3x3 rotation matrix."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_axis_angle(cls, axis: Vec3, angle: float) -> Quaternion:
        """Create quaternion from axis-angle (right-hand rule)."""
        half = angle * 0.5
        s = math.sin(half)
        n = axis.normalize()
        return cls(math.cos(half), n.x * s, n.y * s, n.z * s)


@dataclass
class Transform:
    """Rigid body pose: position + orientation."""

    position: Vec3 = field(default_factory=Vec3)
    orientation: Quaternion = field(default_factory=Quaternion)

    def apply(self, local_point: Vec3) -> Vec3:
        return self.position + self.orientation.rotate(local_point)

    def inverse(self) -> Transform:
        q_inv = self.orientation.conjugate()
        return Transform(
            position=q_inv.rotate(Vec3(-self.position.x, -self.position.y, -self.position.z)),
            orientation=q_inv,
        )


# ---------------------------------------------------------------------------
# Material properties
# ---------------------------------------------------------------------------

@dataclass
class Material:
    """Physical material properties for multi-physics coupling."""

    name: str = "default"
    density: float = 1000.0  # kg/m^3
    youngs_modulus: float = 1e9  # Pa
    poisson_ratio: float = 0.3
    friction_coefficient: float = 0.5
    restitution: float = 0.1
    thermal_conductivity: float = 1.0  # W/(m·K)
    specific_heat: float = 1000.0  # J/(kg·K)
    thermal_expansion: float = 1e-5  # 1/K
    electrical_conductivity: float = 1e7  # S/m
    magnetic_permeability: float = 1.25663706e-6  # H/m (mu_0 for non-magnetic)
    viscosity: float = 0.001  # Pa·s (for fluid-like behavior)
    surface_tension: float = 0.072  # N/m (water-air at 20C)

    def bulk_modulus(self) -> float:
        """K = E / (3 * (1 - 2*nu))."""
        return self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))

    def shear_modulus(self) -> float:
        """G = E / (2 * (1 + nu))."""
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def lame_lambda(self) -> float:
        """First Lamé parameter."""
        nu = self.poisson_ratio
        E = self.youngs_modulus
        return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def lame_mu(self) -> float:
        """Second Lamé parameter (shear modulus)."""
        return self.shear_modulus()


# ---------------------------------------------------------------------------
# Rigid body dynamics
# ---------------------------------------------------------------------------

@dataclass
class RigidBody:
    """6-DOF rigid body with mass and inertia tensor."""

    body_id: int
    transform: Transform = field(default_factory=Transform)
    linear_velocity: Vec3 = field(default_factory=Vec3)
    angular_velocity: Vec3 = field(default_factory=Vec3)
    mass: float = 1.0
    inertia_local: NDArray = field(
        default_factory=lambda: np.eye(3, dtype=np.float64)
    )
    force: Vec3 = field(default_factory=Vec3)
    torque: Vec3 = field(default_factory=Vec3)
    material: Material = field(default_factory=Material)
    is_kinematic: bool = False
    is_sleeping: bool = False

    def inertia_world(self) -> NDArray:
        """Rotate inertia tensor to world frame: I_world = R * I_local * R^T."""
        R = self.transform.orientation.to_rotation_matrix()
        return R @ self.inertia_local @ R.T

    def kinetic_energy(self) -> float:
        """T = 0.5 * m * v^2 + 0.5 * omega^T * I * omega."""
        v = self.linear_velocity
        trans = 0.5 * self.mass * v.dot(v)
        I = self.inertia_world()
        omega = self.angular_velocity.to_array()
        rot = 0.5 * float(omega.T @ I @ omega)
        return trans + rot

    def apply_impulse(self, impulse: Vec3, world_point: Vec3) -> None:
        """Apply impulse at a world-space point."""
        if self.is_kinematic or self.is_sleeping:
            return
        self.linear_velocity = self.linear_velocity + impulse / self.mass
        r = world_point - self.transform.position
        I_inv = np.linalg.inv(self.inertia_world())
        delta_omega = I_inv @ r.cross(impulse).to_array()
        self.angular_velocity = self.angular_velocity + Vec3.from_array(delta_omega)

    def integrate(self, dt: float, gravity: Vec3 = Vec3(0.0, 0.0, -9.81)) -> None:
        """Semi-implicit Euler integration."""
        if self.is_kinematic or self.is_sleeping:
            return
        # Linear
        accel = self.force * (1.0 / self.mass) + gravity
        self.linear_velocity = self.linear_velocity + accel * dt
        self.transform.position = self.transform.position + self.linear_velocity * dt

        # Angular
        I = self.inertia_world()
        omega = self.angular_velocity.to_array()
        torque_arr = self.torque.to_array()
        # Euler's rotation equation: I * domega/dt + omega x (I * omega) = tau
        I_omega = I @ omega
        gyroscopic = np.cross(omega, I_omega)
        alpha = np.linalg.solve(I, torque_arr - gyroscopic)
        self.angular_velocity = self.angular_velocity + Vec3.from_array(alpha) * dt

        # Update orientation: q_new = q + 0.5 * dt * [0, omega] * q
        q = self.transform.orientation
        half_dt = 0.5 * dt
        dq = Quaternion(
            0.0,
            self.angular_velocity.x * half_dt,
            self.angular_velocity.y * half_dt,
            self.angular_velocity.z * half_dt,
        ) * q
        q.w += dq.w
        q.x += dq.x
        q.y += dq.y
        q.z += dq.z
        # Renormalize
        norm = math.sqrt(q.w**2 + q.x**2 + q.y**2 + q.z**2)
        if norm > 1e-12:
            q.w /= norm
            q.x /= norm
            q.y /= norm
            q.z /= norm


# ---------------------------------------------------------------------------
# Soft body dynamics (mass-spring + corotational linear FEM)
# ---------------------------------------------------------------------------

@dataclass
class SoftBodyNode:
    """A node (vertex) in a soft body mesh."""

    index: int
    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    mass: float = 0.0
    force: Vec3 = field(default_factory=Vec3)
    is_fixed: bool = False
    temperature: float = 293.15  # K (20C)


@dataclass
class SoftBodyElement:
    """A tetrahedral element for FEM soft body simulation."""

    indices: Tuple[int, int, int, int]
    rest_volume: float = 0.0
    inv_rest_shape: NDArray = field(
        default_factory=lambda: np.zeros((3, 3), dtype=np.float64)
    )


class SoftBody:
    """Deformable body using corotational linear FEM.

    Based on "Robust Simulation of Deformable Objects" (Müller et al.)
    and Isaac Lab's deformable body API.
    """

    def __init__(self, nodes: List[SoftBodyNode], elements: List[SoftBodyElement], material: Material) -> None:
        self.nodes = nodes
        self.elements = elements
        self.material = material
        self._compute_rest_state()

    def _compute_rest_state(self) -> None:
        """Compute rest volumes and shape matrices for each tetrahedron."""
        for elem in self.elements:
            i0, i1, i2, i3 = elem.indices
            p0 = self.nodes[i0].position.to_array()
            p1 = self.nodes[i1].position.to_array()
            p2 = self.nodes[i2].position.to_array()
            p3 = self.nodes[i3].position.to_array()

            # Shape matrix D = [p1-p0, p2-p0, p3-p0]
            D = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
            elem.rest_volume = abs(np.linalg.det(D)) / 6.0
            if elem.rest_volume > 1e-12:
                elem.inv_rest_shape = np.linalg.inv(D)

    def compute_forces(self, dt: float) -> None:
        """Compute corotational linear FEM elastic forces."""
        mu = self.material.lame_mu()
        lam = self.material.lame_lambda()

        for node in self.nodes:
            node.force = Vec3(0.0, 0.0, 0.0)

        for elem in self.elements:
            i0, i1, i2, i3 = elem.indices
            p0 = self.nodes[i0].position.to_array()
            p1 = self.nodes[i1].position.to_array()
            p2 = self.nodes[i2].position.to_array()
            p3 = self.nodes[i3].position.to_array()

            D = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
            F = D @ elem.inv_rest_shape  # Deformation gradient

            # Polar decomposition for corotational formulation
            U, S, Vt = np.linalg.svd(F)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt

            # Corotated linear strain: epsilon = R^T * F - I
            strain = R.T @ F - np.eye(3)

            # Stress: sigma = 2*mu*strain + lambda*tr(strain)*I
            trace_strain = np.trace(strain)
            stress = 2.0 * mu * strain + lam * trace_strain * np.eye(3)

            # Force density: P = R * sigma
            P = R @ stress

            # Forces on nodes (neglecting volume scaling for simplicity)
            # f_i = -dPsi/dx_i, where Psi is Neo-Hookean-like energy
            H = -P @ elem.inv_rest_shape.T * elem.rest_volume

            f1 = Vec3.from_array(H[:, 0])
            f2 = Vec3.from_array(H[:, 1])
            f3 = Vec3.from_array(H[:, 2])
            f0 = -(f1 + f2 + f3)

            self.nodes[i0].force = self.nodes[i0].force + f0
            self.nodes[i1].force = self.nodes[i1].force + f1
            self.nodes[i2].force = self.nodes[i2].force + f2
            self.nodes[i3].force = self.nodes[i3].force + f3

    def integrate(self, dt: float, gravity: Vec3 = Vec3(0.0, 0.0, -9.81)) -> None:
        """Semi-implicit Euler for soft body nodes."""
        self.compute_forces(dt)
        for node in self.nodes:
            if node.is_fixed:
                continue
            accel = node.force * (1.0 / node.mass) + gravity
            # Damping
            damping = 0.01
            accel = accel - node.velocity * damping
            node.velocity = node.velocity + accel * dt
            node.position = node.position + node.velocity * dt

    def get_center_of_mass(self) -> Vec3:
        total_mass = sum(n.mass for n in self.nodes)
        if total_mass < 1e-12:
            return Vec3()
        weighted = sum((n.position * n.mass for n in self.nodes), Vec3())
        return weighted / total_mass


# ---------------------------------------------------------------------------
# Fluid dynamics (SPH for free-surface, drag for immersed bodies)
# ---------------------------------------------------------------------------

@dataclass
class SPHParticle:
    """Smoothed Particle Hydrodynamics particle."""

    position: Vec3 = field(default_factory=Vec3)
    velocity: Vec3 = field(default_factory=Vec3)
    density: float = 1000.0  # kg/m^3
    pressure: float = 0.0
    mass: float = 0.001  # kg
    viscosity: float = 0.001  # Pa·s


class FluidSimulator:
    """SPH fluid simulator for water/air interaction.

    Based on Müller et al. "Particle-Based Fluid Simulation for Interactive Applications" (2003)
    and Isaac Lab's particle system API.
    """

    def __init__(self, particles: List[SPHParticle], smoothing_length: float = 0.05) -> None:
        self.particles = particles
        self.h = smoothing_length
        self.h2 = smoothing_length**2
        self.h3 = smoothing_length**3
        self.h6 = smoothing_length**6
        self.h9 = smoothing_length**9
        self.gas_constant: float = 2000.0  # k in p = k*(rho - rho0)
        self.rest_density: float = 1000.0
        self.gravity: Vec3 = Vec3(0.0, 0.0, -9.81)

    def _kernel_poly6(self, r2: float) -> float:
        """Poly6 kernel for density estimation."""
        if r2 >= self.h2:
            return 0.0
        return 315.0 / (64.0 * math.pi * self.h9) * (self.h2 - r2) ** 3

    def _kernel_spiky_gradient(self, r: Vec3, dist: float) -> Vec3:
        """Spiky kernel gradient for pressure force."""
        if dist >= self.h or dist < 1e-12:
            return Vec3()
        coeff = -45.0 / (math.pi * self.h6) * (self.h - dist) ** 2 / dist
        return r * coeff

    def _kernel_viscosity_laplacian(self, dist: float) -> float:
        """Viscosity kernel Laplacian."""
        if dist >= self.h:
            return 0.0
        return 45.0 / (math.pi * self.h6) * (self.h - dist)

    def compute_densities(self) -> None:
        """Compute particle densities using SPH summation."""
        for i, pi in enumerate(self.particles):
            density = 0.0
            for j, pj in enumerate(self.particles):
                r = pi.position - pj.position
                r2 = r.dot(r)
                density += pj.mass * self._kernel_poly6(r2)
            pi.density = max(density, self.rest_density * 0.1)

    def compute_pressures(self) -> None:
        """Equation of state: p = k * (rho - rho0)."""
        for p in self.particles:
            p.pressure = self.gas_constant * (p.density - self.rest_density)

    def compute_forces(self) -> None:
        """Compute pressure and viscosity forces."""
        for pi in self.particles:
            f_pressure = Vec3()
            f_viscosity = Vec3()
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
                    viscosity_term = (pj.velocity - pi.velocity) * (
                        pj.mass / pj.density * pi.viscosity * lap
                    )
                    f_viscosity = f_viscosity + viscosity_term
            pi.velocity = pi.velocity + (f_pressure + f_viscosity) * (1.0 / pi.density)

    def integrate(self, dt: float) -> None:
        """Explicit Euler integration for SPH."""
        self.compute_densities()
        self.compute_pressures()
        self.compute_forces()
        for p in self.particles:
            p.velocity = p.velocity + self.gravity * dt
            p.position = p.position + p.velocity * dt

    def compute_drag_on_body(
        self, body_velocity: Vec3, body_area: float, body_position: Vec3, drag_coeff: float = 0.47
    ) -> Vec3:
        """Compute fluid drag force on a rigid body using relative velocity.

        F_drag = 0.5 * rho * v_rel^2 * C_d * A * (-v_rel_hat)
        """
        # Approximate local fluid velocity as average of nearby particles
        nearby_vel = Vec3()
        count = 0
        for p in self.particles:
            dist = (p.position - body_position).norm()
            if dist < self.h * 2.0:
                nearby_vel = nearby_vel + p.velocity
                count += 1
        if count > 0:
            nearby_vel = nearby_vel / count

        v_rel = body_velocity - nearby_vel
        v_mag = v_rel.norm()
        if v_mag < 1e-6:
            return Vec3()

        # Approximate local density
        local_rho = self.rest_density
        for p in self.particles:
            if (p.position - body_position).norm() < self.h:
                local_rho = p.density
                break

        drag_mag = 0.5 * local_rho * v_mag**2 * drag_coeff * body_area
        return v_rel.normalize() * (-drag_mag)


# ---------------------------------------------------------------------------
# Thermal dynamics
# ---------------------------------------------------------------------------

@dataclass
class ThermalState:
    """Thermal state for a body or region."""

    temperature: float = 293.15  # K
    heat_capacity: float = 1000.0  # J/K
    heat_generation: float = 0.0  # W (e.g., motor heating)


class ThermalSimulator:
    """Thermal effects: motor heating, friction heating, conduction, convection.

    References Isaac Lab's thermal sensor API and NVIDIA PhysX thermal features.
    """

    def __init__(self, ambient_temp: float = 293.15) -> None:
        self.ambient_temp = ambient_temp
        self.bodies: Dict[int, ThermalState] = {}
        self.conduction_links: List[Tuple[int, int, float]] = []  # (id1, id2, conductance)

    def register_body(self, body_id: int, state: ThermalState) -> None:
        self.bodies[body_id] = state

    def add_conduction_link(self, id1: int, id2: int, conductance: float) -> None:
        """Add thermal conduction link with conductance G (W/K)."""
        self.conduction_links.append((id1, id2, conductance))

    def compute_friction_heat(
        self, friction_force: Vec3, slip_velocity: Vec3, dt: float
    ) -> float:
        """Q = F_friction * v_slip * dt (Joules)."""
        power = friction_force.norm() * slip_velocity.norm()
        return power * dt

    def compute_motor_heat(
        self, current: float, resistance: float, dt: float
    ) -> float:
        """Joule heating: Q = I^2 * R * dt."""
        return current**2 * resistance * dt

    def step(self, dt: float) -> None:
        """Explicit thermal integration."""
        delta_temps: Dict[int, float] = {bid: 0.0 for bid in self.bodies}

        for bid, state in self.bodies.items():
            # Heat generation
            delta_temps[bid] += state.heat_generation * dt / state.heat_capacity

            # Convection to ambient (simplified: h*A lumped into a coefficient)
            h_conv = 10.0  # W/(m^2*K) approximate
            area = 0.01  # m^2 approximate
            q_conv = h_conv * area * (self.ambient_temp - state.temperature)
            delta_temps[bid] += q_conv * dt / state.heat_capacity

        # Conduction between linked bodies
        for id1, id2, G in self.conduction_links:
            if id1 not in self.bodies or id2 not in self.bodies:
                continue
            t1 = self.bodies[id1].temperature
            t2 = self.bodies[id2].temperature
            q_cond = G * (t2 - t1)
            delta_temps[id1] += q_cond * dt / self.bodies[id1].heat_capacity
            delta_temps[id2] -= q_cond * dt / self.bodies[id2].heat_capacity

        for bid, dtemp in delta_temps.items():
            self.bodies[bid].temperature += dtemp


# ---------------------------------------------------------------------------
# Electromagnetic dynamics
# ---------------------------------------------------------------------------

@dataclass
class ElectromagneticState:
    """Electromagnetic state for actuators/sensors."""

    charge: float = 0.0  # C
    current: float = 0.0  # A
    voltage: float = 0.0  # V
    magnetic_moment: Vec3 = field(default_factory=Vec3)  # A·m^2


class ElectromagneticSimulator:
    """Electromagnetic effects: actuator fields, Lorentz force, induction.

    References Isaac Lab's electromagnet sensor API.
    """

    def __init__(self) -> None:
        self.mu_0 = 4.0 * math.pi * 1e-7  # H/m
        self.epsilon_0 = 8.854187817e-12  # F/m
        self.states: Dict[int, ElectromagneticState] = {}
        self.coils: Dict[int, Tuple[Vec3, Vec3, float]] = {}  # (position, axis, turns)

    def register_body(self, body_id: int, state: ElectromagneticState) -> None:
        self.states[body_id] = state

    def register_coil(self, coil_id: int, position: Vec3, axis: Vec3, turns: float) -> None:
        self.coils[coil_id] = (position, axis.normalize(), turns)

    def magnetic_field_from_coil(
        self, coil_id: int, current: float, eval_point: Vec3
    ) -> Vec3:
        """Approximate B-field from a circular coil using dipole approximation.

        B = (mu_0 / 4*pi) * (3*(m·r̂)*r̂ - m) / r^3
        where m = N * I * A * n̂ (magnetic moment)
        """
        pos, axis, turns = self.coils[coil_id]
        r_vec = eval_point - pos
        r = r_vec.norm()
        if r < 1e-6:
            return Vec3()

        # Assume unit area for simplicity; in practice A = pi * R_coil^2
        area = 0.001  # m^2
        m = axis * (turns * current * area)
        r_hat = r_vec / r

        prefactor = self.mu_0 / (4.0 * math.pi * r**3)
        dot = m.dot(r_hat)
        B = r_hat * (3.0 * dot) - m
        return B * prefactor

    def lorentz_force(
        self, body_id: int, velocity: Vec3, B_field: Vec3
    ) -> Vec3:
        """F = q * (E + v × B).  For current-carrying wire: dF = I * dl × B."""
        state = self.states.get(body_id)
        if state is None:
            return Vec3()
        # Simplified: F = q * v × B for charged particle
        v_cross_B = velocity.cross(B_field)
        return v_cross_B * state.charge

    def compute_motor_torque(
        self, current: float, motor_constant: float, gear_ratio: float = 1.0
    ) -> float:
        """tau = K_t * I * N_gear.

        K_t is motor torque constant (N·m/A).
        """
        return motor_constant * current * gear_ratio

    def compute_back_emf(
        self, angular_velocity: float, motor_constant: float
    ) -> float:
        """V_back = K_e * omega.  For DC motors, K_e ≈ K_t (SI units)."""
        return motor_constant * angular_velocity


# ---------------------------------------------------------------------------
# Multi-physics coupling engine
# ---------------------------------------------------------------------------

class MultiPhysicsEngine:
    """Coupled multi-physics simulation engine.

    Integrates rigid body, soft body, fluid, thermal, and electromagnetic
    subsystems with two-way coupling.
    """

    def __init__(self, dt: float = 1.0 / 60.0) -> None:
        self.dt = dt
        self.rigid_bodies: Dict[int, RigidBody] = {}
        self.soft_bodies: Dict[int, SoftBody] = {}
        self.fluid: Optional[FluidSimulator] = None
        self.thermal = ThermalSimulator()
        self.em = ElectromagneticSimulator()
        self.gravity: Vec3 = Vec3(0.0, 0.0, -9.81)
        self.substeps: int = 4

    def add_rigid_body(self, body: RigidBody) -> None:
        self.rigid_bodies[body.body_id] = body

    def add_soft_body(self, sb_id: int, body: SoftBody) -> None:
        self.soft_bodies[sb_id] = body

    def set_fluid(self, fluid: FluidSimulator) -> None:
        self.fluid = fluid

    def _couple_fluid_rigid(self) -> None:
        """Apply fluid drag and buoyancy to rigid bodies."""
        if self.fluid is None:
            return
        for body in self.rigid_bodies.values():
            # Approximate submerged volume and area
            # For a sphere: V = 4/3 * pi * r^3, A = pi * r^2
            # Use mass and density to estimate radius
            if body.material.density > 0:
                volume = body.mass / body.material.density
                radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
                area = math.pi * radius**2
                drag = self.fluid.compute_drag_on_body(
                    body.linear_velocity, area, body.transform.position
                )
                body.force = body.force + drag

                # Buoyancy: F_b = -rho_fluid * V_sub * g
                # Approximate fully submerged for simplicity
                F_buoyancy = Vec3(0.0, 0.0, 1.0) * (
                    self.fluid.rest_density * volume * 9.81
                )
                body.force = body.force + F_buoyancy

    def _couple_thermal_rigid(self) -> None:
        """Thermal expansion affects geometry (simplified)."""
        for body in self.rigid_bodies.values():
            state = self.thermal.bodies.get(body.body_id)
            if state is None:
                continue
            # Thermal expansion changes effective size
            # delta_L = alpha * L0 * delta_T
            # For simulation, we scale mass properties (simplified)
            dT = state.temperature - 293.15
            if abs(dT) > 1.0:
                # Add thermal stress as a fictitious force
                thermal_force = Vec3(0.0, 0.0, body.material.thermal_expansion * dT * 1e3)
                body.force = body.force + thermal_force

    def _couple_em_rigid(self) -> None:
        """Apply electromagnetic forces to rigid bodies."""
        for body in self.rigid_bodies.values():
            state = self.em.states.get(body.body_id)
            if state is None:
                continue
            # Sum B-fields from all coils
            total_B = Vec3()
            for coil_id in self.em.coils:
                B = self.em.magnetic_field_from_coil(
                    coil_id, state.current, body.transform.position
                )
                total_B = total_B + B

            F_em = self.em.lorentz_force(
                body.body_id, body.linear_velocity, total_B
            )
            body.force = body.force + F_em

    def step(self) -> None:
        """Advance simulation by one time step with coupling."""
        sub_dt = self.dt / self.substeps

        for _ in range(self.substeps):
            # Reset forces
            for body in self.rigid_bodies.values():
                body.force = Vec3()
                body.torque = Vec3()

            # Coupling
            self._couple_fluid_rigid()
            self._couple_thermal_rigid()
            self._couple_em_rigid()

            # Integrate rigid bodies
            for body in self.rigid_bodies.values():
                body.integrate(sub_dt, self.gravity)

            # Integrate soft bodies
            for sb in self.soft_bodies.values():
                sb.integrate(sub_dt, self.gravity)

            # Integrate fluid
            if self.fluid is not None:
                self.fluid.integrate(sub_dt)

            # Integrate thermal
            self.thermal.step(sub_dt)

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a summary of the simulation state."""
        return {
            "rigid_bodies": len(self.rigid_bodies),
            "soft_bodies": len(self.soft_bodies),
            "fluid_particles": len(self.fluid.particles) if self.fluid else 0,
            "thermal_bodies": len(self.thermal.bodies),
            "em_bodies": len(self.em.states),
            "dt": self.dt,
            "substeps": self.substeps,
        }
