"""Advanced sensor simulation for Moses v6.0.

Camera (RGB, depth, stereo, event), tactile, force/torque, IMU, and LiDAR sensors.

References:
- Isaac Lab sensor APIs (isaaclab.sensors)
- NVIDIA Isaac Sim sensor implementations
- Real sensor datasheets and noise models
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import numpy.typing as npt

from moses.sim.multiphysics import Vec3, Quaternion, Transform, RigidBody

NDArray = npt.NDArray[np.float64]


# ---------------------------------------------------------------------------
# Base sensor classes
# ---------------------------------------------------------------------------

@dataclass
class SensorConfig:
    """Base configuration for all sensors."""

    name: str = "sensor"
    update_rate: float = 60.0  # Hz
    offset: Transform = field(default_factory=Transform)


class BaseSensor:
    """Base class for all sensors."""

    def __init__(self, config: SensorConfig) -> None:
        self.config = config
        self._last_update_time: float = 0.0
        self._data: Any = None

    def update(self, time: float, **kwargs: Any) -> Any:
        """Update sensor reading."""
        dt = time - self._last_update_time
        if dt < 1.0 / self.config.update_rate:
            return self._data
        self._last_update_time = time
        self._data = self._sample(**kwargs)
        return self._data

    def _sample(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def get_data(self) -> Any:
        return self._data


# ---------------------------------------------------------------------------
# Camera sensors (RGB, depth, stereo, event)
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig(SensorConfig):
    """Configuration for camera sensors."""

    resolution: Tuple[int, int] = (640, 480)
    fov_horizontal: float = 60.0  # degrees
    fov_vertical: float = 45.0  # degrees
    near_clip: float = 0.01
    far_clip: float = 100.0
    # Noise parameters
    noise_std: float = 0.01
    noise_mean: float = 0.0
    # Depth-specific
    depth_accuracy: float = 0.001  # meters at 1m
    depth_noise_scale: float = 0.001
    # Event-specific
    event_threshold: float = 0.1  # intensity change threshold
    refractory_period: float = 1e-6  # seconds


class RGBCamera(BaseSensor):
    """RGB camera sensor simulation.

    Simulates perspective projection with lens distortion and noise.
    References Isaac Lab's Camera sensor (isaaclab.sensors.Camera).
    """

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self.config = config
        self._image: NDArray = np.zeros(
            (config.resolution[1], config.resolution[0], 3), dtype=np.uint8
        )
        self._intrinsics = self._compute_intrinsics()

    def _compute_intrinsics(self) -> NDArray:
        """Compute camera intrinsic matrix K.

        K = [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
        """
        w, h = self.config.resolution
        fx = (w / 2.0) / math.tan(math.radians(self.config.fov_horizontal) / 2.0)
        fy = (h / 2.0) / math.tan(math.radians(self.config.fov_vertical) / 2.0)
        cx = w / 2.0
        cy = h / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def project_point(self, point_3d: Vec3, camera_transform: Transform) -> Optional[Tuple[float, float]]:
        """Project a 3D world point to 2D image coordinates.

        Uses perspective projection: u = fx * X/Z + cx, v = fy * Y/Z + cy
        """
        # Transform to camera frame
        local = camera_transform.inverse().apply(point_3d)
        if local.z < self.config.near_clip or local.z > self.config.far_clip:
            return None

        # Project
        K = self._intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = fx * local.x / local.z + cx
        v = fy * local.y / local.z + cy

        # Check bounds
        w, h = self.config.resolution
        if u < 0 or u >= w or v < 0 or v >= h:
            return None
        return (u, v)

    def _sample(self, **kwargs: Any) -> NDArray:
        """Generate synthetic RGB image.

        In a real implementation, this would render from the simulation.
        Here we simulate the noise model.
        """
        h, w = self.config.resolution[1], self.config.resolution[0]
        # Base image (placeholder: gradient)
        image = np.zeros((h, w, 3), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                image[i, j, 0] = (j / w) * 255  # R gradient
                image[i, j, 1] = (i / h) * 255  # G gradient
                image[i, j, 2] = 128  # B constant

        # Add Gaussian noise
        noise = np.random.normal(
            self.config.noise_mean, self.config.noise_std * 255, (h, w, 3)
        )
        image = image + noise
        image = np.clip(image, 0, 255).astype(np.uint8)
        self._image = image
        return image

    def get_intrinsics(self) -> NDArray:
        return self._intrinsics.copy()

    def get_extrinsics(self, camera_transform: Transform) -> NDArray:
        """Get 4x4 extrinsic matrix [R|t]."""
        R = camera_transform.orientation.to_rotation_matrix()
        t = camera_transform.position.to_array().reshape(3, 1)
        Rt = np.hstack([R, t])
        bottom = np.array([[0, 0, 0, 1]], dtype=np.float64)
        return np.vstack([Rt, bottom])


class DepthCamera(BaseSensor):
    """Depth camera sensor simulation.

    Simulates depth measurement with accuracy falloff and noise.
    References Isaac Lab's Camera with depth output.
    """

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self.config = config
        self._depth: NDArray = np.zeros(config.resolution[::-1], dtype=np.float32)
        self._intrinsics = self._compute_intrinsics()

    def _compute_intrinsics(self) -> NDArray:
        w, h = self.config.resolution
        fx = (w / 2.0) / math.tan(math.radians(self.config.fov_horizontal) / 2.0)
        fy = (h / 2.0) / math.tan(math.radians(self.config.fov_vertical) / 2.0)
        cx = w / 2.0
        cy = h / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def _sample(self, **kwargs: Any) -> NDArray:
        """Generate synthetic depth image with realistic noise.

        Depth noise model: sigma = a * z^2 + b * z + c
        where z is true depth.
        """
        h, w = self.config.resolution[1], self.config.resolution[0]
        depth = np.ones((h, w), dtype=np.float64) * self.config.far_clip

        # Placeholder: create a depth ramp
        for i in range(h):
            for j in range(w):
                # Simple depth field: plane at z=1.0
                z_true = 1.0 + 0.5 * math.sin(j * 0.01) * math.cos(i * 0.01)
                # Noise increases with depth
                a, b = self.config.depth_noise_scale, self.config.depth_accuracy
                sigma = a * z_true**2 + b * z_true
                noise = np.random.normal(0, sigma)
                depth[i, j] = max(self.config.near_clip, z_true + noise)

        self._depth = depth.astype(np.float32)
        return self._depth

    def depth_to_pointcloud(
        self, depth_image: NDArray, camera_transform: Transform
    ) -> NDArray:
        """Convert depth image to 3D point cloud.

        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth
        """
        h, w = depth_image.shape
        K = self._intrinsics
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        points = []
        for v in range(h):
            for u in range(w):
                z = float(depth_image[v, u])
                if z >= self.config.far_clip * 0.99:
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                # Transform to world
                local = Vec3(x, y, z)
                world = camera_transform.apply(local)
                points.append(world.to_array())

        return np.array(points, dtype=np.float64)


class StereoCamera(BaseSensor):
    """Stereo camera pair for disparity-based depth estimation.

    Depth from disparity: Z = f * B / d
    where f is focal length, B is baseline, d is disparity.
    """

    def __init__(self, config: CameraConfig, baseline: float = 0.12) -> None:
        super().__init__(config)
        self.config = config
        self.baseline = baseline  # meters
        self.left_camera = RGBCamera(config)
        self.right_camera = RGBCamera(config)
        self._disparity: NDArray = np.zeros(config.resolution[::-1], dtype=np.float32)
        self._depth: NDArray = np.zeros(config.resolution[::-1], dtype=np.float32)

    def _sample(self, **kwargs: Any) -> Tuple[NDArray, NDArray, NDArray]:
        """Sample left image, right image, and computed depth."""
        left_img = self.left_camera._sample(**kwargs)
        right_img = self.right_camera._sample(**kwargs)

        # Compute disparity (simplified block matching)
        h, w = self.config.resolution[1], self.config.resolution[0]
        disparity = np.zeros((h, w), dtype=np.float64)
        block_size = 5
        max_disp = 64

        for v in range(block_size, h - block_size):
            for u in range(block_size + max_disp, w - block_size):
                best_disp = 0
                best_score = float("inf")
                left_block = left_img[
                    v - block_size : v + block_size,
                    u - block_size : u + block_size,
                ]
                for d in range(max_disp):
                    if u - d - block_size < 0:
                        break
                    right_block = right_img[
                        v - block_size : v + block_size,
                        u - d - block_size : u - d + block_size,
                    ]
                    score = float(np.sum(np.abs(left_block.astype(np.float64) - right_block.astype(np.float64))))
                    if score < best_score:
                        best_score = score
                        best_disp = d
                disparity[v, u] = best_disp

        # Convert disparity to depth
        K = self.left_camera.get_intrinsics()
        fx = K[0, 0]
        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = fx * self.baseline / disparity[valid]
        depth = np.clip(depth, self.config.near_clip, self.config.far_clip)

        self._disparity = disparity.astype(np.float32)
        self._depth = depth.astype(np.float32)
        return left_img, right_img, self._depth

    def get_depth(self) -> NDArray:
        return self._depth

    def get_disparity(self) -> NDArray:
        return self._disparity


class EventCamera(BaseSensor):
    """Event camera (Dynamic Vision Sensor) simulation.

    Event cameras output asynchronous events when pixel brightness changes exceed threshold.
    Event: (x, y, t, p) where p = +1 (ON) or -1 (OFF)

    References:
    - "Event-based Vision: A Survey" (Gallego et al., 2020)
    - Isaac Lab event camera sensor API
    """

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self.config = config
        self._last_intensity: NDArray = np.zeros(config.resolution[::-1], dtype=np.float64)
        self._events: List[Tuple[int, int, float, int]] = []  # (x, y, t, polarity)
        self._refractory: NDArray = np.zeros(config.resolution[::-1], dtype=np.float64)

    def _sample(self, **kwargs: Any) -> List[Tuple[int, int, float, int]]:
        """Generate events from intensity change.

        Event generation: log(I(t)) - log(I(t-dt)) > C_pos -> ON event
                          log(I(t)) - log(I(t-dt)) < C_neg -> OFF event
        """
        time = kwargs.get("time", 0.0)
        current_intensity = kwargs.get("intensity", self._last_intensity)

        h, w = self.config.resolution[1], self.config.resolution[0]
        events = []
        threshold = self.config.event_threshold
        refractory = self.config.refractory_period

        for v in range(h):
            for u in range(w):
                # Check refractory period
                if time - self._refractory[v, u] < refractory:
                    continue

                prev = self._last_intensity[v, u]
                curr = current_intensity[v, u]
                if prev < 1e-6 or curr < 1e-6:
                    continue

                delta_log = math.log(curr) - math.log(prev)
                if delta_log > threshold:
                    events.append((u, v, time, 1))
                    self._refractory[v, u] = time
                elif delta_log < -threshold:
                    events.append((u, v, time, -1))
                    self._refractory[v, u] = time

        self._last_intensity = current_intensity.copy()
        self._events = events
        return events

    def get_events(self) -> List[Tuple[int, int, float, int]]:
        return self._events

    def events_to_frame(self, events: List[Tuple[int, int, float, int]]) -> NDArray:
        """Convert events to a frame (for visualization)."""
        h, w = self.config.resolution[1], self.config.resolution[0]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for x, y, t, p in events:
            if 0 <= x < w and 0 <= y < h:
                if p > 0:
                    frame[y, x] = [255, 0, 0]  # ON = red
                else:
                    frame[y, x] = [0, 0, 255]  # OFF = blue
        return frame


# ---------------------------------------------------------------------------
# Tactile sensors
# ---------------------------------------------------------------------------

@dataclass
class TactileConfig(SensorConfig):
    """Configuration for tactile sensors."""

    num_taxels: int = 100
    taxel_area: float = 1e-6  # m^2 per taxel
    pressure_range: Tuple[float, float] = (0.0, 1e6)  # Pa
    temperature_range: Tuple[float, float] = (273.15, 373.15)  # K
    vibration_bandwidth: float = 1000.0  # Hz
    noise_std_pressure: float = 100.0  # Pa
    noise_std_temp: float = 0.1  # K


@dataclass
class TactileReading:
    """Reading from a single tactile taxel."""

    taxel_id: int
    position: Vec3
    normal_force: float  # N
    shear_force: Vec3  # N
    pressure: float  # Pa
    temperature: float  # K
    vibration: float  # m/s amplitude
    slip: bool


class TactileSensor(BaseSensor):
    """Tactile sensor array simulating pressure, vibration, and temperature.

    Models:
    - Pressure: p = F_n / A_taxel
    - Shear: tau = F_t / A_taxel
    - Temperature: thermal conduction from contact
    - Vibration: high-frequency force fluctuations
    - Slip: relative tangential velocity detection

    References Isaac Lab's ContactSensor and TactileSensor.
    """

    def __init__(self, config: TactileConfig) -> None:
        super().__init__(config)
        self.config = config
        self.taxel_positions: List[Vec3] = []
        self._readings: List[TactileReading] = []
        self._prev_contact_points: Dict[int, Vec3] = {}

    def set_taxel_positions(self, positions: List[Vec3]) -> None:
        self.taxel_positions = positions

    def _sample(self, **kwargs: Any) -> List[TactileReading]:
        """Sample tactile sensor array.

        kwargs:
            contacts: List of ContactPoint
            body_temps: Dict[int, float] mapping body_id to temperature
            dt: time step
        """
        contacts = kwargs.get("contacts", [])
        body_temps = kwargs.get("body_temps", {})
        dt = kwargs.get("dt", 1.0 / 60.0)
        time = kwargs.get("time", 0.0)

        readings = []
        for i, pos in enumerate(self.taxel_positions):
            # Find nearest contact
            nearest = None
            min_dist = float("inf")
            for contact in contacts:
                dist = (contact.point - pos).norm()
                if dist < min_dist:
                    min_dist = dist
                    nearest = contact

            if nearest is None or min_dist > 0.005:
                # No contact
                readings.append(
                    TactileReading(
                        taxel_id=i,
                        position=pos,
                        normal_force=0.0,
                        shear_force=Vec3(),
                        pressure=0.0,
                        temperature=293.15,
                        vibration=0.0,
                        slip=False,
                    )
                )
                continue

            # Compute pressure from penetration (Hertzian approximation)
            # p0 = E* * sqrt(delta/R) where E* is effective modulus
            E_eff = 1e6  # Pa (soft sensor)
            R_eff = 0.001  # m
            delta = abs(nearest.penetration)
            p0 = E_eff * math.sqrt(delta / R_eff) if delta > 0 else 0.0
            pressure = min(p0, self.config.pressure_range[1])

            # Normal force
            normal_force = pressure * self.config.taxel_area

            # Shear force (Coulomb friction)
            mu = nearest.friction_coeff
            shear_mag = mu * normal_force * 0.3  # Partial slip
            # Random shear direction
            shear_dir = nearest.normal.cross(Vec3(0.0, 0.0, 1.0)).normalize()
            if shear_dir.norm() < 0.1:
                shear_dir = Vec3(1.0, 0.0, 0.0)
            shear_force = shear_dir * shear_mag

            # Temperature
            other_body = nearest.body_a  # Simplified
            temp = body_temps.get(other_body, 293.15)

            # Vibration (simulated high-frequency noise)
            vibration = abs(np.random.normal(0, 1e-6))

            # Slip detection
            slip = False
            if i in self._prev_contact_points:
                prev_pos = self._prev_contact_points[i]
                v_slide = (nearest.point - prev_pos).norm() / dt
                slip = v_slide > 0.001
            self._prev_contact_points[i] = nearest.point

            # Add noise
            pressure += np.random.normal(0, self.config.noise_std_pressure)
            temp += np.random.normal(0, self.config.noise_std_temp)

            readings.append(
                TactileReading(
                    taxel_id=i,
                    position=pos,
                    normal_force=normal_force,
                    shear_force=shear_force,
                    pressure=max(0.0, pressure),
                    temperature=temp,
                    vibration=vibration,
                    slip=slip,
                )
            )

        self._readings = readings
        return readings

    def get_pressure_map(self) -> NDArray:
        return np.array([r.pressure for r in self._readings], dtype=np.float64)

    def get_temperature_map(self) -> NDArray:
        return np.array([r.temperature for r in self._readings], dtype=np.float64)

    def get_shear_magnitude_map(self) -> NDArray:
        return np.array([r.shear_force.norm() for r in self._readings], dtype=np.float64)


# ---------------------------------------------------------------------------
# Force/Torque sensors
# ---------------------------------------------------------------------------

@dataclass
class ForceTorqueConfig(SensorConfig):
    """Configuration for force/torque sensors."""

    max_force: float = 1000.0  # N
    max_torque: float = 100.0  # N·m
    noise_std_force: float = 0.1  # N
    noise_std_torque: float = 0.01  # N·m
    crosstalk: float = 0.01  # Force-to-torque coupling
    # For distributed sensing
    num_sensing_points: int = 1


@dataclass
class Wrench6D:
    """6-axis force/torque reading."""

    force: Vec3
    torque: Vec3
    timestamp: float = 0.0
    sensor_id: int = 0

    def to_array(self) -> NDArray:
        return np.concatenate([self.force.to_array(), self.torque.to_array()])


class ForceTorqueSensor(BaseSensor):
    """6-axis force/torque sensor (wrench sensor).

    Simulates strain gauge-based F/T sensor with crosstalk and noise.
    References:
    - ATI Nano series datasheets
    - Isaac Lab's ContactSensor with force reporting
    """

    def __init__(self, config: ForceTorqueConfig) -> None:
        super().__init__(config)
        self.config = config
        self._wrench = Wrench6D(Vec3(), Vec3())
        self._calibration_matrix: Optional[NDArray] = None

    def set_calibration_matrix(self, C: NDArray) -> None:
        """Set calibration matrix (6x6) mapping raw strains to wrench."""
        self._calibration_matrix = C

    def _sample(self, **kwargs: Any) -> Wrench6D:
        """Sample force/torque.

        kwargs:
            true_force: Vec3
            true_torque: Vec3
            time: float
        """
        true_force = kwargs.get("true_force", Vec3())
        true_torque = kwargs.get("true_torque", Vec3())
        time = kwargs.get("time", 0.0)

        # Add noise
        noise_f = Vec3(
            np.random.normal(0, self.config.noise_std_force),
            np.random.normal(0, self.config.noise_std_force),
            np.random.normal(0, self.config.noise_std_force),
        )
        noise_t = Vec3(
            np.random.normal(0, self.config.noise_std_torque),
            np.random.normal(0, self.config.noise_std_torque),
            np.random.normal(0, self.config.noise_std_torque),
        )

        # Crosstalk: force induces apparent torque
        crosstalk_torque = true_force * self.config.crosstalk

        measured_force = true_force + noise_f
        measured_torque = true_torque + noise_t + crosstalk_torque

        # Apply calibration if available
        if self._calibration_matrix is not None:
            raw = np.concatenate([measured_force.to_array(), measured_torque.to_array()])
            calibrated = self._calibration_matrix @ raw
            measured_force = Vec3.from_array(calibrated[:3])
            measured_torque = Vec3.from_array(calibrated[3:])

        self._wrench = Wrench6D(measured_force, measured_torque, time)
        return self._wrench

    def get_wrench(self) -> Wrench6D:
        return self._wrench


class DistributedForceSensor(BaseSensor):
    """Distributed force sensing across a surface.

    Array of mini F/T sensors for pressure distribution measurement.
    """

    def __init__(self, config: ForceTorqueConfig, sensor_positions: List[Vec3]) -> None:
        super().__init__(config)
        self.config = config
        self.positions = sensor_positions
        self._sensors = [ForceTorqueSensor(config) for _ in sensor_positions]
        self._readings: List[Wrench6D] = []

    def _sample(self, **kwargs: Any) -> List[Wrench6D]:
        forces = kwargs.get("forces", [Vec3() for _ in self.positions])
        torques = kwargs.get("torques", [Vec3() for _ in self.positions])
        time = kwargs.get("time", 0.0)

        self._readings = []
        for i, sensor in enumerate(self._sensors):
            wrench = sensor._sample(
                true_force=forces[i],
                true_torque=torques[i],
                time=time,
            )
            self._readings.append(wrench)

        return self._readings

    def get_total_force(self) -> Vec3:
        total = Vec3()
        for r in self._readings:
            total = total + r.force
        return total

    def get_total_torque(self, about_point: Vec3 = Vec3()) -> Vec3:
        total = Vec3()
        for i, r in enumerate(self._readings):
            r_vec = self.positions[i] - about_point
            total = total + r.torque + r.force.cross(r_vec)
        return total


# ---------------------------------------------------------------------------
# IMU sensors
# ---------------------------------------------------------------------------

@dataclass
class IMUConfig(SensorConfig):
    """Configuration for IMU sensors."""

    # Accelerometer
    accel_noise_density: float = 0.001  # m/s^2 / sqrt(Hz)
    accel_random_walk: float = 0.0001  # m/s^2 / sqrt(Hz)
    accel_bias_instability: float = 0.0001  # m/s^2
    # Gyroscope
    gyro_noise_density: float = 0.0001  # rad/s / sqrt(Hz)
    gyro_random_walk: float = 0.00001  # rad/s / sqrt(Hz)
    gyro_bias_instability: float = 0.00001  # rad/s
    # Update rate
    update_rate: float = 1000.0  # Hz typical for IMU


@dataclass
class IMUReading:
    """IMU sensor reading."""

    linear_acceleration: Vec3  # m/s^2 in sensor frame
    angular_velocity: Vec3  # rad/s in sensor frame
    orientation: Quaternion  # Estimated orientation
    timestamp: float
    temperature: float = 293.15  # K


class IMUSensor(BaseSensor):
    """Inertial Measurement Unit sensor simulation.

    Models:
    - Accelerometer: a_measured = R^T * (a - g) + bias + noise
    - Gyroscope: omega_measured = R^T * omega + bias + noise
    - Bias random walk (1/f noise)
    - Temperature-dependent bias drift

    References:
    - IEEE Std 952-1997 (Inertial Sensor Terminology)
    - Isaac Lab's ImuSensor
    - "Aided Navigation: GPS with High Rate Sensors" (Farrell)
    """

    def __init__(self, config: IMUConfig) -> None:
        super().__init__(config)
        self.config = config
        self._accel_bias = Vec3()
        self._gyro_bias = Vec3()
        self._accel_bias_walk = Vec3()
        self._gyro_bias_walk = Vec3()
        self._last_time: float = 0.0
        self._reading = IMUReading(
            linear_acceleration=Vec3(),
            angular_velocity=Vec3(),
            orientation=Quaternion(),
            timestamp=0.0,
        )

    def _sample(self, **kwargs: Any) -> IMUReading:
        """Sample IMU.

        kwargs:
            true_accel: Vec3 (world frame)
            true_omega: Vec3 (world frame)
            orientation: Quaternion (body-to-world)
            time: float
            temperature: float
        """
        true_accel = kwargs.get("true_accel", Vec3())
        true_omega = kwargs.get("true_omega", Vec3())
        orientation = kwargs.get("orientation", Quaternion())
        time = kwargs.get("time", 0.0)
        temperature = kwargs.get("temperature", 293.15)

        dt = time - self._last_time
        self._last_time = time
        sqrt_dt = math.sqrt(max(dt, 1e-6))

        # Transform to body frame
        R_inv = orientation.conjugate()
        accel_body = R_inv.rotate(true_accel)
        omega_body = R_inv.rotate(true_omega)

        # Add gravity in body frame
        g_world = Vec3(0.0, 0.0, -9.81)
        g_body = R_inv.rotate(g_world)
        accel_body = accel_body + g_body

        # Bias random walk
        self._accel_bias_walk = Vec3(
            self._accel_bias_walk.x + np.random.normal(0, self.config.accel_random_walk * sqrt_dt),
            self._accel_bias_walk.y + np.random.normal(0, self.config.accel_random_walk * sqrt_dt),
            self._accel_bias_walk.z + np.random.normal(0, self.config.accel_random_walk * sqrt_dt),
        )
        self._gyro_bias_walk = Vec3(
            self._gyro_bias_walk.x + np.random.normal(0, self.config.gyro_random_walk * sqrt_dt),
            self._gyro_bias_walk.y + np.random.normal(0, self.config.gyro_random_walk * sqrt_dt),
            self._gyro_bias_walk.z + np.random.normal(0, self.config.gyro_random_walk * sqrt_dt),
        )

        # Temperature-dependent bias
        temp_drift = (temperature - 293.15) * 0.00001

        # White noise
        noise_accel = Vec3(
            np.random.normal(0, self.config.accel_noise_density / sqrt_dt),
            np.random.normal(0, self.config.accel_noise_density / sqrt_dt),
            np.random.normal(0, self.config.accel_noise_density / sqrt_dt),
        )
        noise_gyro = Vec3(
            np.random.normal(0, self.config.gyro_noise_density / sqrt_dt),
            np.random.normal(0, self.config.gyro_noise_density / sqrt_dt),
            np.random.normal(0, self.config.gyro_noise_density / sqrt_dt),
        )

        measured_accel = accel_body + self._accel_bias + self._accel_bias_walk + noise_accel
        measured_accel = Vec3(
            measured_accel.x + temp_drift,
            measured_accel.y + temp_drift,
            measured_accel.z + temp_drift,
        )

        measured_gyro = omega_body + self._gyro_bias + self._gyro_bias_walk + noise_gyro
        measured_gyro = Vec3(
            measured_gyro.x + temp_drift * 0.1,
            measured_gyro.y + temp_drift * 0.1,
            measured_gyro.z + temp_drift * 0.1,
        )

        self._reading = IMUReading(
            linear_acceleration=measured_accel,
            angular_velocity=measured_gyro,
            orientation=orientation,
            timestamp=time,
            temperature=temperature,
        )
        return self._reading

    def get_reading(self) -> IMUReading:
        return self._reading

    def reset_bias(self) -> None:
        """Reset bias estimates (e.g., during calibration)."""
        self._accel_bias = Vec3()
        self._gyro_bias = Vec3()
        self._accel_bias_walk = Vec3()
        self._gyro_bias_walk = Vec3()


# ---------------------------------------------------------------------------
# LiDAR sensors
# ---------------------------------------------------------------------------

@dataclass
class LiDARConfig(SensorConfig):
    """Configuration for LiDAR sensors."""

    num_rays_horizontal: int = 360
    num_rays_vertical: int = 16
    horizontal_fov: Tuple[float, float] = (0.0, 360.0)  # degrees
    vertical_fov: Tuple[float, float] = (-15.0, 15.0)  # degrees
    min_range: float = 0.1
    max_range: float = 100.0
    # Noise
    range_noise_std: float = 0.01  # m
    angular_noise_std: float = 0.001  # degrees
    # Dropout
    dropout_rate: float = 0.001  # Probability of no return
    # Intensity
    intensity_enabled: bool = True


@dataclass
class LiDARPoint:
    """A single LiDAR point."""

    x: float
    y: float
    z: float
    intensity: float
    ring: int  # Vertical ring index
    timestamp: float


class LiDARSensor(BaseSensor):
    """LiDAR sensor simulation with ray casting.

    Simulates rotating/multi-beam LiDAR (e.g., Velodyne, Ouster).
    Uses spherical ray casting with noise and dropout.

    References:
    - Velodyne VLP-16 / HDL-64E datasheets
    - Isaac Lab's RayCaster sensor
    """

    def __init__(self, config: LiDARConfig) -> None:
        super().__init__(config)
        self.config = config
        self._point_cloud: List[LiDARPoint] = []
        self._ranges: NDArray = np.zeros(
            (config.num_rays_vertical, config.num_rays_horizontal), dtype=np.float32
        )
        self._intensities: NDArray = np.zeros(
            (config.num_rays_vertical, config.num_rays_horizontal), dtype=np.float32
        )

    def _compute_ray_directions(self) -> List[Tuple[int, int, Vec3]]:
        """Compute ray directions for all beams."""
        directions = []
        h_start, h_end = self.config.horizontal_fov
        v_start, v_end = self.config.vertical_fov
        nh = self.config.num_rays_horizontal
        nv = self.config.num_rays_vertical

        for vi in range(nv):
            v_angle = math.radians(v_start + (v_end - v_start) * vi / max(nv - 1, 1))
            for hi in range(nh):
                h_angle = math.radians(h_start + (h_end - h_start) * hi / max(nh - 1, 1))
                # Spherical coordinates
                x = math.cos(v_angle) * math.cos(h_angle)
                y = math.cos(v_angle) * math.sin(h_angle)
                z = math.sin(v_angle)
                directions.append((vi, hi, Vec3(x, y, z)))
        return directions

    def _cast_ray(
        self,
        origin: Vec3,
        direction: Vec3,
        scene_objects: List[Any],
    ) -> Optional[Tuple[float, float]]:
        """Cast a single ray against scene objects.

        Returns (distance, intensity) or None if no hit.
        """
        closest_dist = self.config.max_range
        hit_intensity = 0.0

        for obj in scene_objects:
            # Simplified: sphere intersection test
            if hasattr(obj, "transform") and hasattr(obj, "radius"):
                center = obj.transform.position
                radius = obj.radius
                oc = origin - center
                a = direction.dot(direction)
                b = 2.0 * oc.dot(direction)
                c = oc.dot(oc) - radius**2
                discriminant = b**2 - 4.0 * a * c
                if discriminant >= 0:
                    t = (-b - math.sqrt(discriminant)) / (2.0 * a)
                    if 0 < t < closest_dist:
                        closest_dist = t
                        # Intensity based on surface reflectivity
                        hit_intensity = getattr(obj, "reflectivity", 0.5)

            # Plane intersection (ground)
            elif hasattr(obj, "normal") and hasattr(obj, "distance"):
                n = obj.normal
                d = obj.distance
                denom = direction.dot(n)
                if abs(denom) > 1e-6:
                    t = -(origin.dot(n) + d) / denom
                    if 0 < t < closest_dist:
                        closest_dist = t
                        hit_intensity = getattr(obj, "reflectivity", 0.3)

        if closest_dist >= self.config.max_range * 0.99:
            return None
        return closest_dist, hit_intensity

    def _sample(self, **kwargs: Any) -> List[LiDARPoint]:
        """Sample LiDAR point cloud.

        kwargs:
            sensor_transform: Transform
            scene_objects: List of objects with intersection methods
            time: float
        """
        sensor_transform = kwargs.get("sensor_transform", Transform())
        scene_objects = kwargs.get("scene_objects", [])
        time = kwargs.get("time", 0.0)

        origin = sensor_transform.position
        directions = self._compute_ray_directions()
        points = []

        for vi, hi, dir_local in directions:
            # Transform direction to world frame
            dir_world = sensor_transform.orientation.rotate(dir_local).normalize()

            # Add angular noise
            noise_h = np.random.normal(0, math.radians(self.config.angular_noise_std))
            noise_v = np.random.normal(0, math.radians(self.config.angular_noise_std))
            # Apply small rotation (simplified)
            dir_world = Vec3(
                dir_world.x + noise_h,
                dir_world.y + noise_v,
                dir_world.z,
            ).normalize()

            # Cast ray
            result = self._cast_ray(origin, dir_world, scene_objects)

            # Dropout
            if result is None or np.random.random() < self.config.dropout_rate:
                self._ranges[vi, hi] = 0.0
                self._intensities[vi, hi] = 0.0
                continue

            distance, intensity = result

            # Add range noise
            distance += np.random.normal(0, self.config.range_noise_std)
            distance = max(self.config.min_range, min(distance, self.config.max_range))

            # Compute point
            point = origin + dir_world * distance

            self._ranges[vi, hi] = distance
            self._intensities[vi, hi] = intensity

            points.append(
                LiDARPoint(
                    x=point.x,
                    y=point.y,
                    z=point.z,
                    intensity=intensity,
                    ring=vi,
                    timestamp=time,
                )
            )

        self._point_cloud = points
        return points

    def get_point_cloud(self) -> List[LiDARPoint]:
        return self._point_cloud

    def get_point_cloud_array(self) -> NDArray:
        """Return point cloud as Nx4 array [x, y, z, intensity]."""
        return np.array(
            [[p.x, p.y, p.z, p.intensity] for p in self._point_cloud],
            dtype=np.float64,
        )

    def get_range_image(self) -> NDArray:
        return self._ranges.copy()

    def get_intensity_image(self) -> NDArray:
        return self._intensities.copy()

    def get_occupancy_grid(
        self, resolution: float = 0.1, size: Tuple[float, float] = (20.0, 20.0)
    ) -> NDArray:
        """Convert point cloud to 2D occupancy grid (bird's eye view)."""
        grid_w = int(size[0] / resolution)
        grid_h = int(size[1] / resolution)
        grid = np.zeros((grid_h, grid_w), dtype=np.float64)

        for p in self._point_cloud:
            gx = int((p.x + size[0] / 2) / resolution)
            gy = int((p.y + size[1] / 2) / resolution)
            if 0 <= gx < grid_w and 0 <= gy < grid_h:
                grid[gy, gx] = 1.0

        return grid


# ---------------------------------------------------------------------------
# Sensor manager
# ---------------------------------------------------------------------------

class SensorManager:
    """Manager for all sensors in a simulation scene."""

    def __init__(self) -> None:
        self.sensors: Dict[str, BaseSensor] = {}
        self._sensor_data: Dict[str, Any] = {}

    def add_sensor(self, name: str, sensor: BaseSensor) -> None:
        self.sensors[name] = sensor

    def remove_sensor(self, name: str) -> None:
        if name in self.sensors:
            del self.sensors[name]

    def update_all(self, time: float, **kwargs: Any) -> Dict[str, Any]:
        """Update all sensors and return their data."""
        for name, sensor in self.sensors.items():
            self._sensor_data[name] = sensor.update(time, **kwargs)
        return self._sensor_data

    def get_sensor_data(self, name: str) -> Any:
        return self._sensor_data.get(name)

    def get_camera_data(self, name: str) -> Optional[NDArray]:
        sensor = self.sensors.get(name)
        if isinstance(sensor, (RGBCamera, DepthCamera)):
            return sensor.get_data()
        return None

    def get_lidar_pointcloud(self, name: str) -> Optional[List[LiDARPoint]]:
        sensor = self.sensors.get(name)
        if isinstance(sensor, LiDARSensor):
            return sensor.get_point_cloud()
        return None

    def get_imu_reading(self, name: str) -> Optional[IMUReading]:
        sensor = self.sensors.get(name)
        if isinstance(sensor, IMUSensor):
            return sensor.get_reading()
        return None

    def get_ft_reading(self, name: str) -> Optional[Wrench6D]:
        sensor = self.sensors.get(name)
        if isinstance(sensor, ForceTorqueSensor):
            return sensor.get_wrench()
        return None

    def get_tactile_readings(self, name: str) -> Optional[List[TactileReading]]:
        sensor = self.sensors.get(name)
        if isinstance(sensor, TactileSensor):
            return sensor.get_data()
        return None
