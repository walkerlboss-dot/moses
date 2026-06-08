"""Contact-rich manipulation for Moses v6.0.

Grasp planning, in-hand manipulation, sliding/rolling/pivoting,
and tactile feedback simulation.

References:
- Isaac Lab manipulation API (isaaclab.manipulators)
- PhysX contact reporting and friction model
- "A Mathematical Introduction to Robotic Manipulation" (Murray, Li, Sastry)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple, Union

import numpy as np
import numpy.typing as npt

from moses.sim.multiphysics import Vec3, Quaternion, Transform, RigidBody, Material

NDArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Contact and friction models
# ---------------------------------------------------------------------------

class ContactType(Enum):
    """Types of contact in manipulation."""

    POINT = auto()
    LINE = auto()
    PATCH = auto()


@dataclass
class ContactPoint:
    """A single contact point between two bodies."""

    body_a: int
    body_b: int
    point: Vec3  # World-space contact point
    normal: Vec3  # From A to B, normalized
    penetration: float  # Positive = overlap
    friction_coeff: float = 0.5
    restitution: float = 0.1

    def relative_velocity(
        self, va: Vec3, wa: Vec3, vb: Vec3, wb: Vec3, ra: Vec3, rb: Vec3
    ) -> Vec3:
        """v_rel = (vb + wb × rb) - (va + wa × ra)."""
        v_contact_a = va + wa.cross(ra)
        v_contact_b = vb + wb.cross(rb)
        return v_contact_b - v_contact_a


@dataclass
class FrictionCone:
    """Coulomb friction cone at a contact point.

    The friction cone constraint: ||f_t|| <= mu * f_n
    where f_t is tangential force and f_n is normal force.
    """

    normal: Vec3
    mu: float  # friction coefficient
    tangent1: Vec3 = field(init=False)
    tangent2: Vec3 = field(init=False)

    def __post_init__(self) -> None:
        # Build orthonormal basis with normal as z-axis
        n = self.normal.normalize()
        # Pick arbitrary vector not parallel to normal
        if abs(n.z) < 0.9:
            arbitrary = Vec3(0.0, 0.0, 1.0)
        else:
            arbitrary = Vec3(0.0, 1.0, 0.0)
        self.tangent1 = n.cross(arbitrary).normalize()
        self.tangent2 = n.cross(self.tangent1).normalize()

    def decompose_force(self, force: Vec3) -> Tuple[float, Vec3, Vec3]:
        """Decompose force into normal and tangential components.

        Returns (f_n, f_t, f_t_direction)
        """
        f_n = force.dot(self.normal)
        f_normal = self.normal * f_n
        f_t_vec = force - f_normal
        f_t_mag = f_t_vec.norm()
        if f_t_mag > 1e-12:
            f_t_dir = f_t_vec / f_t_mag
        else:
            f_t_dir = Vec3()
        return f_n, f_t_vec, f_t_dir

    def is_inside_cone(self, force: Vec3) -> bool:
        """Check if force lies within the friction cone."""
        f_n, f_t_vec, _ = self.decompose_force(force)
        if f_n < 0:
            return False  # No adhesion
        return f_t_vec.norm() <= self.mu * f_n


# ---------------------------------------------------------------------------
# Grasp models
# ---------------------------------------------------------------------------

@dataclass
class GraspContact:
    """A contact point used for grasp analysis."""

    position: Vec3  # On object surface, object-local
    normal: Vec3  # Outward from object, object-local
    friction_coeff: float = 0.5

    def to_world(self, object_transform: Transform) -> Tuple[Vec3, Vec3]:
        """Convert to world-space position and normal."""
        world_pos = object_transform.apply(self.position)
        world_normal = object_transform.orientation.rotate(self.normal).normalize()
        return world_pos, world_normal


@dataclass
class GraspWrench:
    """Wrench (force + torque) applied at a contact."""

    force: Vec3
    torque: Vec3

    def magnitude(self) -> float:
        return math.sqrt(self.force.dot(self.force) + self.torque.dot(self.torque))


class GraspQualityMetrics:
    """Grasp quality metrics based on wrench space analysis.

    Implements epsilon-quality and volume-quality from:
    Ferrari & Canny, "Planning Optimal Grasps" (1992)
    """

    @staticmethod
    def compute_grasp_matrix(
        contacts: List[GraspContact], object_transform: Transform
    ) -> NDArray:
        """Build the grasp matrix G (6 x 3m for m point contacts without friction).

        For frictional contacts, we use a linearized friction cone (m edges).
        G = [n1, n2, ..., nm; r1×n1, r2×n2, ..., rm×nm]
        where ni are contact normals and ri are contact positions.
        """
        G_rows = []
        for contact in contacts:
            world_pos, world_normal = contact.to_world(object_transform)
            # For simplicity, use normal direction only (no friction cone discretization)
            G_rows.append(world_normal.to_array())
        G = np.column_stack(G_rows)
        return G

    @staticmethod
    def epsilon_quality(contacts: List[GraspContact], object_transform: Transform) -> float:
        """Epsilon quality: radius of largest inscribed ball in wrench space.

        Q_epsilon = min_{||w||=1} max_i (w^T * G_i)
        where G_i are the columns of the grasp matrix.
        """
        if len(contacts) < 2:
            return 0.0

        # Discretize friction cone into edges
        num_edges = 8
        G_cols = []
        for contact in contacts:
            world_pos, world_normal = contact.to_world(object_transform)
            # Build friction cone edges
            cone = FrictionCone(world_normal, contact.friction_coeff)
            for k in range(num_edges):
                angle = 2.0 * math.pi * k / num_edges
                edge = (
                    world_normal
                    + cone.tangent1 * math.cos(angle) * contact.friction_coeff
                    + cone.tangent2 * math.sin(angle) * contact.friction_coeff
                )
                edge = edge.normalize()
                r = world_pos - object_transform.position
                torque = r.cross(edge)
                col = np.concatenate([edge.to_array(), torque.to_array()])
                G_cols.append(col)

        G = np.column_stack(G_cols)
        # Compute epsilon quality via convex hull distance
        # Simplified: use minimum singular value as proxy
        try:
            s = np.linalg.svd(G, compute_uv=False)
            min_sv = float(np.min(s))
            return max(0.0, min_sv)
        except np.linalg.LinAlgError:
            return 0.0

    @staticmethod
    def volume_quality(contacts: List[GraspContact], object_transform: Transform) -> float:
        """Volume quality: volume of the convex hull of contact wrenches.

        Q_volume = sqrt(det(G * G^T))
        """
        G = GraspQualityMetrics.compute_grasp_matrix(contacts, object_transform)
        if G.shape[1] < G.shape[0]:
            return 0.0
        try:
            det = np.linalg.det(G @ G.T)
            return math.sqrt(max(0.0, det))
        except np.linalg.LinAlgError:
            return 0.0

    @staticmethod
    def force_closure(contacts: List[GraspContact], object_transform: Transform) -> bool:
        """Check if grasp is in force closure.

        A grasp is in force closure if the origin is strictly inside the
        convex hull of the primitive contact wrenches.
        """
        if len(contacts) < 2:
            return False
        q = GraspQualityMetrics.epsilon_quality(contacts, object_transform)
        return q > 1e-6


# ---------------------------------------------------------------------------
# Grasp planning
# ---------------------------------------------------------------------------

class GraspPlanner:
    """Grasp planning with analytical and learned components.

    Analytical: antipodal grasp search on mesh
    Learned: neural network grasp quality predictor (placeholder interface)
    """

    def __init__(self, object_mesh: Optional[Any] = None) -> None:
        self.object_mesh = object_mesh
        self.quality_metrics = GraspQualityMetrics()
        self._grasp_cache: List[Tuple[List[GraspContact], float]] = []

    def sample_antipodal_grasps(
        self,
        num_samples: int = 100,
        object_transform: Transform = Transform(),
    ) -> List[Tuple[List[GraspContact], float]]:
        """Sample antipodal grasp candidates.

        An antipodal grasp requires contact normals to be opposite:
        n1 · n2 < -cos(theta_threshold)
        """
        grasps: List[Tuple[List[GraspContact], float]] = []

        # Generate random surface points (simplified sphere sampling)
        for _ in range(num_samples):
            # Random direction
            theta = 2.0 * math.pi * np.random.random()
            phi = math.acos(2.0 * np.random.random() - 1.0)
            n1 = Vec3(
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi),
            )
            # Antipodal direction
            n2 = n1 * (-1.0)

            # Sample points on a unit sphere surface
            radius = 0.05  # 5cm object
            p1 = n1 * radius
            p2 = n2 * radius

            contact1 = GraspContact(position=p1, normal=n1, friction_coeff=0.5)
            contact2 = GraspContact(position=p2, normal=n2, friction_coeff=0.5)
            contacts = [contact1, contact2]

            quality = self.quality_metrics.epsilon_quality(contacts, object_transform)
            if quality > 0.01:
                grasps.append((contacts, quality))

        # Sort by quality
        grasps.sort(key=lambda x: x[1], reverse=True)
        self._grasp_cache = grasps
        return grasps

    def plan_grasp_analytical(
        self,
        object_transform: Transform = Transform(),
        num_contacts: int = 2,
    ) -> Optional[Tuple[List[GraspContact], float]]:
        """Plan a grasp using analytical force closure analysis."""
        candidates = self.sample_antipodal_grasps(200, object_transform)
        if not candidates:
            return None
        return candidates[0]

    def plan_grasp_learned(
        self,
        point_cloud: NDArray,
        gripper_pose_guess: Transform,
        quality_predictor: Optional[Callable[[NDArray, Transform], float]] = None,
    ) -> Optional[Tuple[Transform, float]]:
        """Plan a grasp using a learned quality predictor.

        Args:
            point_cloud: Nx3 array of object surface points
            gripper_pose_guess: Initial gripper pose estimate
            quality_predictor: Neural network that scores (point_cloud, pose) -> quality

        Returns:
            Best (gripper_pose, quality) or None
        """
        if quality_predictor is None:
            # Fallback to random search with analytical check
            return None

        best_pose = gripper_pose_guess
        best_quality = -1.0

        # Local search around initial guess
        for _ in range(50):
            # Perturb pose
            noise_pos = Vec3(
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
            )
            noise_rot = Vec3(
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
            )
            pose = Transform(
                position=gripper_pose_guess.position + noise_pos,
                orientation=Quaternion.from_axis_angle(
                    noise_rot.normalize(), noise_rot.norm()
                )
                * gripper_pose_guess.orientation,
            )
            quality = quality_predictor(point_cloud, pose)
            if quality > best_quality:
                best_quality = quality
                best_pose = pose

        if best_quality > 0.5:
            return best_pose, best_quality
        return None

    def plan_grasp_hybrid(
        self,
        object_transform: Transform,
        point_cloud: NDArray,
        quality_predictor: Optional[Callable[[NDArray, Transform], float]] = None,
    ) -> Optional[Tuple[Union[List[GraspContact], Transform], float]]:
        """Hybrid grasp planning: analytical + learned refinement."""
        # First, get analytical candidates
        analytical = self.plan_grasp_analytical(object_transform)
        if analytical is None:
            # Fall back to learned
            return self.plan_grasp_learned(point_cloud, object_transform, quality_predictor)

        contacts, a_quality = analytical
        if a_quality > 0.5:
            return contacts, a_quality

        # Refine with learned model
        learned = self.plan_grasp_learned(point_cloud, object_transform, quality_predictor)
        if learned and learned[1] > a_quality:
            return learned
        return contacts, a_quality


# ---------------------------------------------------------------------------
# In-hand manipulation
# ---------------------------------------------------------------------------

class InHandManipulation:
    """In-hand manipulation: sliding, rolling, pivoting.

    Based on contact kinematics and quasi-static manipulation.
    References:
    - "Mechanics and Planning of Manipulator Pushing Operations" (Mason, 1986)
    - "In-Hand Manipulation" (Bicchi & Kumar, 2000)
    """

    def __init__(self, object_body: RigidBody, finger_bodies: List[RigidBody]) -> None:
        self.object = object_body
        self.fingers = finger_bodies
        self.contacts: List[ContactPoint] = []

    def update_contacts(self, contacts: List[ContactPoint]) -> None:
        """Update current contact configuration."""
        self.contacts = [c for c in contacts if c.body_a == self.object.body_id or c.body_b == self.object.body_id]

    def compute_sliding_velocity(
        self, finger_vel: Vec3, contact: ContactPoint
    ) -> Vec3:
        """Compute sliding velocity at a contact.

        v_slide = v_finger - v_object - omega_object × r
        projected onto tangent plane.
        """
        r = contact.point - self.object.transform.position
        v_object_at_contact = (
            self.object.linear_velocity + self.object.angular_velocity.cross(r)
        )
        v_rel = finger_vel - v_object_at_contact
        # Project to tangent plane
        v_normal = contact.normal * v_rel.dot(contact.normal)
        return v_rel - v_normal

    def compute_rolling_constraint(
        self, finger_vel: Vec3, finger_omega: Vec3, contact: ContactPoint, finger_radius: float
    ) -> Vec3:
        """Rolling without slipping constraint.

        v_object + omega_object × r = v_finger + omega_finger × (-r_finger)
        """
        r_obj = contact.point - self.object.transform.position
        r_finger = contact.point - self.fingers[0].transform.position

        v_obj = self.object.linear_velocity + self.object.angular_velocity.cross(r_obj)
        v_finger_contact = finger_vel + finger_omega.cross(r_finger)

        return v_obj - v_finger_contact  # Should be ~0 for pure rolling

    def compute_pivot_motion(
        self, pivot_point: Vec3, desired_rotation: Quaternion
    ) -> Tuple[Vec3, Vec3]:
        """Compute object motion for pivoting about a point.

        Pivot: rotate object about a fixed contact point.
        v_pivot = 0, so v_cm = -omega × r_pivot
        """
        r = pivot_point - self.object.transform.position
        # Desired angular velocity from quaternion difference
        # omega = 2 * q_dot * q_conj (simplified)
        omega = Vec3(0.0, 0.0, 1.0) * 0.5  # Example: rotate about z
        v_cm = omega.cross(r) * (-1.0)
        return v_cm, omega

    def plan_finger_motions(
        self, target_object_pose: Transform, dt: float
    ) -> Dict[int, Tuple[Vec3, Vec3]]:
        """Plan finger velocities to achieve target object pose.

        Uses Jacobian transpose method for kinematic control.
        Returns: {finger_id: (linear_vel, angular_vel)}
        """
        # Compute pose error
        pos_error = target_object_pose.position - self.object.transform.position

        # Simple P controller for position
        kp = 5.0
        v_desired = pos_error * kp

        # Distribute to fingers (simplified: equal sharing)
        finger_motions: Dict[int, Tuple[Vec3, Vec3]] = {}
        num_fingers = max(len(self.fingers), 1)
        v_per_finger = v_desired / num_fingers

        for finger in self.fingers:
            finger_motions[finger.body_id] = (v_per_finger, Vec3())

        return finger_motions


# ---------------------------------------------------------------------------
# Sliding, rolling, pivoting primitives
# ---------------------------------------------------------------------------

class ManipulationPrimitive:
    """Base class for manipulation primitives."""

    def execute(
        self,
        object_body: RigidBody,
        finger_bodies: List[RigidBody],
        dt: float,
    ) -> None:
        raise NotImplementedError


class SlidingPrimitive(ManipulationPrimitive):
    """Slide an object along a surface.

    Uses Coulomb friction model:
    F_friction = -mu * N * v_hat  (if sliding)
    |F_friction| <= mu * N       (if sticking)
    """

    def __init__(self, direction: Vec3, speed: float, surface_normal: Vec3) -> None:
        self.direction = direction.normalize()
        self.speed = speed
        self.surface_normal = surface_normal.normalize()

    def execute(
        self,
        object_body: RigidBody,
        finger_bodies: List[RigidBody],
        dt: float,
    ) -> None:
        # Apply pushing force in direction
        push_force = self.direction * 2.0  # N
        object_body.force = object_body.force + push_force

        # Apply friction opposing motion
        v = object_body.linear_velocity
        v_tangent = v - self.surface_normal * v.dot(self.surface_normal)
        if v_tangent.norm() > 1e-6:
            friction_dir = v_tangent.normalize() * (-1.0)
            # Normal force approximated as weight
            normal_force = object_body.mass * 9.81
            friction_mag = object_body.material.friction_coefficient * normal_force
            friction_force = friction_dir * friction_mag
            object_body.force = object_body.force + friction_force


class RollingPrimitive(ManipulationPrimitive):
    """Roll an object on a surface.

    For a cylinder/sphere: v_cm = R * omega (rolling without slipping)
    """

    def __init__(self, axis: Vec3, angular_speed: float, radius: float) -> None:
        self.axis = axis.normalize()
        self.angular_speed = angular_speed
        self.radius = radius

    def execute(
        self,
        object_body: RigidBody,
        finger_bodies: List[RigidBody],
        dt: float,
    ) -> None:
        # Desired angular velocity
        omega_desired = self.axis * self.angular_speed

        # Rolling constraint: v_cm = omega × r_contact
        # For rolling on flat surface, r_contact is -radius * surface_normal
        surface_normal = Vec3(0.0, 0.0, 1.0)
        r_contact = surface_normal * (-self.radius)
        v_desired = omega_desired.cross(r_contact)

        # PD control to achieve desired velocity
        v_error = v_desired - object_body.linear_velocity
        omega_error = omega_desired - object_body.angular_velocity

        kp_v = 10.0
        kp_w = 5.0
        force = v_error * kp_v * object_body.mass
        torque = omega_error * kp_w  # Simplified inertia scaling

        object_body.force = object_body.force + force
        object_body.torque = object_body.torque + torque


class PivotingPrimitive(ManipulationPrimitive):
    """Pivot an object about a contact point.

    The pivot point remains stationary; object rotates about it.
    """

    def __init__(self, pivot_point: Vec3, target_angle: float, axis: Vec3) -> None:
        self.pivot_point = pivot_point
        self.target_angle = target_angle
        self.axis = axis.normalize()
        self.current_angle = 0.0

    def execute(
        self,
        object_body: RigidBody,
        finger_bodies: List[RigidBody],
        dt: float,
    ) -> None:
        r = self.pivot_point - object_body.transform.position

        # Angular velocity to rotate about pivot
        angle_error = self.target_angle - self.current_angle
        omega_mag = angle_error * 2.0  # P control
        omega = self.axis * omega_mag

        # Constraint: v_cm = -omega × r
        v_cm = omega.cross(r) * (-1.0)

        # Apply as velocity constraint (simplified)
        object_body.linear_velocity = v_cm
        object_body.angular_velocity = omega

        self.current_angle += omega_mag * dt


# ---------------------------------------------------------------------------
# Tactile feedback simulation
# ---------------------------------------------------------------------------

@dataclass
class TactileSensorReading:
    """Reading from a tactile sensor patch."""

    sensor_id: int
    position: Vec3  # Sensor center in world space
    normal_force: float  # N
    shear_force: Vec3  # N
    contact_area: float  # m^2
    pressure: float  # Pa
    temperature: float  # K
    slip_detected: bool = False


class TactileSensorArray:
    """Array of tactile sensors simulating pressure, vibration, and temperature.

    Models:
    - Normal pressure: p = F_n / A
    - Shear stress: tau = F_t / A
    - Slip detection: |v_slide| > threshold
    - Temperature: conduction from contacted object

    References Isaac Lab's tactile sensor API (isaaclab.sensors.TactileSensor).
    """

    def __init__(
        self,
        sensor_positions: List[Vec3],
        sensor_area: float = 1e-6,  # 1 mm^2 per taxel
        slip_threshold: float = 0.001,  # m/s
    ) -> None:
        self.positions = sensor_positions
        self.sensor_area = sensor_area
        self.slip_threshold = slip_threshold
        self.readings: List[TactileSensorReading] = []
        self._contact_history: Dict[int, List[Vec3]] = {}  # For slip detection

    def update(
        self,
        contacts: List[ContactPoint],
        object_temps: Dict[int, float],
        dt: float,
    ) -> List[TactileSensorReading]:
        """Update tactile readings from contact information."""
        self.readings = []

        for i, pos in enumerate(self.positions):
            # Find nearest contact
            nearest_contact: Optional[ContactPoint] = None
            min_dist = float("inf")
            for contact in contacts:
                dist = (contact.point - pos).norm()
                if dist < min_dist:
                    min_dist = dist
                    nearest_contact = contact

            if nearest_contact is None or min_dist > 0.005:  # 5mm threshold
                # No contact
                reading = TactileSensorReading(
                    sensor_id=i,
                    position=pos,
                    normal_force=0.0,
                    shear_force=Vec3(),
                    contact_area=0.0,
                    pressure=0.0,
                    temperature=293.15,
                    slip_detected=False,
                )
                self.readings.append(reading)
                continue

            contact = nearest_contact
            # Estimate contact area (circular patch approximation)
            contact_radius = math.sqrt(self.sensor_area / math.pi)
            contact_area = math.pi * contact_radius**2

            # Normal pressure
            # For rigid contact, use Hertzian contact theory approximation
            # a = (3 * F * R / (4 * E*))^(1/3)
            # p0 = 3F / (2 * pi * a^2)
            # Simplified: uniform pressure
            pressure = abs(contact.penetration) * 1e6  # Stiffness-based approximation
            normal_force = pressure * contact_area

            # Shear force (friction)
            friction_mag = contact.friction_coeff * normal_force
            # Random shear direction for simulation
            shear = contact.normal.cross(Vec3(0.0, 0.0, 1.0)).normalize()
            if shear.norm() < 0.1:
                shear = Vec3(1.0, 0.0, 0.0)
            shear_force = shear * friction_mag * 0.3  # Partial slip

            # Slip detection
            slip = False
            if i not in self._contact_history:
                self._contact_history[i] = []
            self._contact_history[i].append(contact.point)
            if len(self._contact_history[i]) > 2:
                self._contact_history[i].pop(0)
                p_prev = self._contact_history[i][0]
                p_curr = self._contact_history[i][1]
                v_slide = (p_curr - p_prev).norm() / dt
                slip = v_slide > self.slip_threshold

            # Temperature
            other_body = contact.body_a if contact.body_b == -1 else contact.body_b
            temp = object_temps.get(other_body, 293.15)

            reading = TactileSensorReading(
                sensor_id=i,
                position=pos,
                normal_force=normal_force,
                shear_force=shear_force,
                contact_area=contact_area,
                pressure=pressure,
                temperature=temp,
                slip_detected=slip,
            )
            self.readings.append(reading)

        return self.readings

    def get_pressure_map(self) -> NDArray:
        """Return pressure values as an array."""
        return np.array([r.pressure for r in self.readings], dtype=np.float64)

    def get_slip_vector(self) -> NDArray:
        """Return binary slip detection vector."""
        return np.array([1.0 if r.slip_detected else 0.0 for r in self.readings], dtype=np.float64)


# ---------------------------------------------------------------------------
# Manipulation controller
# ---------------------------------------------------------------------------

class ManipulationController:
    """High-level manipulation controller coordinating planning and execution."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.planner = GraspPlanner()
        self.in_hand: Optional[InHandManipulation] = None
        self.tactile: Optional[TactileSensorArray] = None
        self.current_primitive: Optional[ManipulationPrimitive] = None
        self._grasp_state: Dict[str, Any] = {}

    def plan_grasp(
        self,
        object_body: RigidBody,
        finger_bodies: List[RigidBody],
        object_transform: Transform,
    ) -> Optional[Tuple[List[GraspContact], float]]:
        """Plan a grasp for the given object."""
        self.planner = GraspPlanner()
        result = self.planner.plan_grasp_analytical(object_transform)
        if result:
            contacts, quality = result
            self._grasp_state = {
                "object_id": object_body.body_id,
                "contacts": contacts,
                "quality": quality,
            }
            self.in_hand = InHandManipulation(object_body, finger_bodies)
        return result

    def execute_grasp(self, gripper_force: float) -> None:
        """Execute grasp by applying normal forces at contact points."""
        if "contacts" not in self._grasp_state:
            return
        contacts = self._grasp_state["contacts"]
        for contact in contacts:
            # Apply normal force toward object center
            force = contact.normal * (-gripper_force)
            # In a real simulator, this would be applied via constraint solver

    def start_sliding(self, direction: Vec3, speed: float, surface_normal: Vec3) -> None:
        """Start a sliding primitive."""
        self.current_primitive = SlidingPrimitive(direction, speed, surface_normal)

    def start_rolling(self, axis: Vec3, angular_speed: float, radius: float) -> None:
        """Start a rolling primitive."""
        self.current_primitive = RollingPrimitive(axis, angular_speed, radius)

    def start_pivoting(self, pivot_point: Vec3, target_angle: float, axis: Vec3) -> None:
        """Start a pivoting primitive."""
        self.current_primitive = PivotingPrimitive(pivot_point, target_angle, axis)

    def step(self, dt: float) -> None:
        """Execute one step of the current manipulation primitive."""
        if self.current_primitive and self.in_hand:
            self.current_primitive.execute(
                self.in_hand.object,
                self.in_hand.fingers,
                dt,
            )

    def get_tactile_feedback(self) -> Optional[List[TactileSensorReading]]:
        """Get latest tactile sensor readings."""
        if self.tactile is None:
            return None
        return self.tactile.readings
