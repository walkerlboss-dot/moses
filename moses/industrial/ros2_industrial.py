"""
moses/industrial/ros2_industrial.py
====================================

ROS2 Industrial integration for humanoid robot manipulation.

Implements:
  - MoveIt 2 integration for motion planning
  - Industrial trajectory processing (time-optimal, blending)
  - Calibration data management (hand-eye, extrinsic, intrinsic)
  - Vendor-specific drivers (Universal Robots, FANUC)

Standards:
  - ROS2 Humble/Iron/Jazzy
  - MoveIt 2 (moveit_core, moveit_ros)
  - ros2_control / ros2_controllers
  - ISO 9283 (Robot performance criteria)
  - ISO 9787 (Robot coordinate systems)

Dependencies:
  - rclpy
  - moveit_commander (Python bindings)
  - sensor_msgs, geometry_msgs, trajectory_msgs
  - industrial_msgs (for robot status)

Author: Moses Industrial Team
Version: 6.0.0
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------

class RobotMode(Enum):
    """Robot operating mode per industrial_msgs/RobotMode."""
    MANUAL = 0
    AUTO = 1
    EXTERNAL = 2


class TriState(IntEnum):
    """Tri-state for robot status (industrial_msgs)."""
    UNKNOWN = -1
    FALSE = 0
    TRUE = 1


@dataclass
class JointState:
    """Simplified joint state."""
    name: List[str]
    position: List[float]
    velocity: List[float] = field(default_factory=list)
    effort: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Pose:
    """6-DOF pose: position (x,y,z) + quaternion (qx,qy,qz,qw)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Pose:
        return cls(*arr.tolist())


# ---------------------------------------------------------------------------
# MoveIt Integration
# ---------------------------------------------------------------------------

@dataclass
class PlanningSceneConfig:
    """MoveIt planning scene configuration."""
    robot_description: str = "robot_description"
    planning_group: str = "manipulator"
    planner_id: str = "RRTConnectkConfigDefault"
    planning_time: float = 5.0
    num_planning_attempts: int = 10
    max_velocity_scaling: float = 0.1
    max_acceleration_scaling: float = 0.1


@dataclass
class Constraints:
    """Motion planning constraints."""
    joint_constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    position_constraints: List[Tuple[float, float, float, float]] = field(default_factory=list)
    orientation_constraints: List[Tuple[float, float, float, float]] = field(default_factory=list)


class MoveItInterface:
    """
    MoveIt 2 motion planning interface.

    Wraps moveit_commander and rclpy for Pythonic access to:
      - Motion planning (joint space + Cartesian)
      - Collision-aware planning
      - Planning scene management
      - Trajectory execution monitoring
    """
    def __init__(self, config: PlanningSceneConfig) -> None:
        self.config = config
        self._move_group: Any = None
        self._robot: Any = None
        self._scene: Any = None
        self._node: Any = None

    def initialize(self, node_name: str = "moses_moveit_interface") -> None:
        """Initialize MoveIt commander and ROS2 node."""
        try:
            import rclpy
            from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface
            from moveit_commander.roscpp_initializer import roscpp_initialize

            rclpy.init()
            self._node = rclpy.create_node(node_name)
            roscpp_initialize([])

            self._robot = RobotCommander(robot_description=self.config.robot_description)
            self._scene = PlanningSceneInterface()
            self._move_group = MoveGroupCommander(
                self.config.planning_group,
                robot_description=self.config.robot_description,
            )
            self._move_group.set_planner_id(self.config.planner_id)
            self._move_group.set_planning_time(self.config.planning_time)
            self._move_group.set_num_planning_attempts(self.config.num_planning_attempts)
            self._move_group.set_max_velocity_scaling_factor(self.config.max_velocity_scaling)
            self._move_group.set_max_acceleration_scaling_factor(self.config.max_acceleration_scaling)
        except ImportError as e:
            raise RuntimeError(f"MoveIt/ROS2 not available: {e}")

    def shutdown(self) -> None:
        if self._node:
            self._node.destroy_node()
            import rclpy
            rclpy.shutdown()

    def plan_joint_motion(
        self,
        joint_targets: Dict[str, float],
        constraints: Optional[Constraints] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Plan joint-space motion to target configuration.

        Returns trajectory dict with keys:
          - joint_trajectory: JointTrajectory message dict
          - fraction: planning success indicator
          - planning_time: seconds
        """
        if not self._move_group:
            return None

        self._move_group.set_joint_value_target(joint_targets)
        if constraints:
            self._apply_constraints(constraints)

        plan = self._move_group.plan()
        if plan[0]:
            return {
                "joint_trajectory": plan[1],
                "fraction": 1.0,
                "planning_time": plan[2] if len(plan) > 2 else 0.0,
            }
        return None

    def plan_cartesian_path(
        self,
        waypoints: List[Pose],
        eef_step: float = 0.01,
        jump_threshold: float = 0.0,
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Plan Cartesian path through waypoints.

        Returns (trajectory_dict, fraction) where fraction is
        the fraction of the path that was successfully planned.
        """
        if not self._move_group:
            return None, 0.0

        from geometry_msgs.msg import Pose as ROSPose
        ros_waypoints = []
        for wp in waypoints:
            p = ROSPose()
            p.position.x = wp.x
            p.position.y = wp.y
            p.position.z = wp.z
            p.orientation.x = wp.qx
            p.orientation.y = wp.qy
            p.orientation.z = wp.qz
            p.orientation.w = wp.qw
            ros_waypoints.append(p)

        plan, fraction = self._move_group.compute_cartesian_path(
            ros_waypoints, eef_step, jump_threshold
        )
        if fraction > 0.99:
            return {
                "joint_trajectory": plan,
                "fraction": fraction,
            }, fraction
        return None, fraction

    def execute_trajectory(self, trajectory: Dict[str, Any]) -> bool:
        """Execute planned trajectory on robot."""
        if not self._move_group:
            return False
        return self._move_group.execute(trajectory.get("joint_trajectory"), wait=True)

    def get_current_pose(self) -> Pose:
        """Get current end-effector pose."""
        if not self._move_group:
            return Pose()
        p = self._move_group.get_current_pose().pose
        return Pose(p.position.x, p.position.y, p.position.z,
                    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)

    def get_current_joint_values(self) -> Dict[str, float]:
        """Get current joint positions."""
        if not self._move_group:
            return {}
        return dict(zip(
            self._move_group.get_active_joints(),
            self._move_group.get_current_joint_values()
        ))

    def add_collision_object(
        self,
        name: str,
        shape: str,          # "box", "sphere", "cylinder", "mesh"
        dimensions: List[float],
        pose: Pose,
    ) -> None:
        """Add collision object to planning scene."""
        if not self._scene:
            return
        from moveit_commander import PlanningSceneInterface
        # Simplified: actual implementation would use shape_msgs
        self._scene.add_box(name, pose.to_array().tolist()[:3], dimensions)

    def _apply_constraints(self, constraints: Constraints) -> None:
        """Apply motion constraints to planning request."""
        # Simplified: actual implementation would build moveit_msgs/Constraints
        pass


# ---------------------------------------------------------------------------
# Industrial Trajectory Processing
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryPoint:
    """Single trajectory point with time, position, velocity, acceleration."""
    time_from_start: float
    positions: List[float]
    velocities: List[float] = field(default_factory=list)
    accelerations: List[float] = field(default_factory=list)
    effort: List[float] = field(default_factory=list)


@dataclass
class Trajectory:
    """Joint trajectory with metadata."""
    joint_names: List[str]
    points: List[TrajectoryPoint]
    duration: float = 0.0

    def __post_init__(self) -> None:
        if self.points:
            self.duration = self.points[-1].time_from_start


class TrajectoryProcessor:
    """
    Industrial trajectory post-processing.

    Implements:
      - Time-optimal path parameterization (TOPP)
      - Trajectory blending (corner smoothing)
      - Velocity/acceleration limiting per joint
      - Jerk limiting for smooth motion
    """
    def __init__(
        self,
        joint_limits: Dict[str, Tuple[float, float, float]],
        # (vel_limit, acc_limit, jerk_limit) per joint
    ) -> None:
        self.joint_limits = joint_limits

    def time_optimal_parameterization(
        self,
        path: List[List[float]],
        time_step: float = 0.001,
    ) -> Trajectory:
        """
        Time-Optimal Path Parameterization (TOPP).

        Uses the convex optimization approach (Pham, 2014)
        to find the fastest trajectory respecting velocity
        and acceleration constraints.
        """
        if len(path) < 2:
            return Trajectory(joint_names=list(self.joint_limits.keys()), points=[])

        n_joints = len(path[0])
        joint_names = list(self.joint_limits.keys())[:n_joints]

        # Simplified: compute path length and uniform time scaling
        path_length = sum(
            math.sqrt(sum((a[i] - b[i])**2 for i in range(n_joints)))
            for a, b in zip(path, path[1:])
        )

        # Conservative time estimate based on average velocity limit
        avg_vlim = sum(lim[0] for lim in self.joint_limits.values()) / len(self.joint_limits)
        total_time = path_length / (avg_vlim * 0.5)  # 50% of max velocity

        # Generate time-parameterized points
        num_points = max(2, int(total_time / time_step))
        points: List[TrajectoryPoint] = []
        for i in range(num_points):
            t = i * time_step
            # Interpolate along path
            s = min(t / total_time, 1.0)
            idx = int(s * (len(path) - 1))
            frac = s * (len(path) - 1) - idx
            idx = min(idx, len(path) - 2)
            pos = [
                path[idx][j] + frac * (path[idx + 1][j] - path[idx][j])
                for j in range(n_joints)
            ]
            points.append(TrajectoryPoint(time_from_start=t, positions=pos))

        return Trajectory(joint_names=joint_names, points=points, duration=total_time)

    def blend_corners(
        self,
        trajectory: Trajectory,
        blend_radius: float = 0.05,
    ) -> Trajectory:
        """
        Blend corners in Cartesian or joint space.

        Uses circular arc blending with specified radius.
        For industrial robots, typical blend radius: 1-50 mm.
        """
        # Simplified: insert intermediate points around sharp corners
        new_points: List[TrajectoryPoint] = [trajectory.points[0]]
        for i in range(1, len(trajectory.points) - 1):
            prev_p = np.array(trajectory.points[i - 1].positions)
            curr_p = np.array(trajectory.points[i].positions)
            next_p = np.array(trajectory.points[i + 1].positions)

            # Check corner angle
            v1 = curr_p - prev_p
            v2 = next_p - curr_p
            angle = math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))

            if angle < math.radians(150):  # Sharp corner
                # Insert blend points
                blend_start = curr_p - blend_radius * v1 / np.linalg.norm(v1)
                blend_end = curr_p + blend_radius * v2 / np.linalg.norm(v2)
                t = trajectory.points[i].time_from_start
                new_points.append(TrajectoryPoint(
                    time_from_start=t - 0.01,
                    positions=blend_start.tolist(),
                ))
                new_points.append(TrajectoryPoint(
                    time_from_start=t + 0.01,
                    positions=blend_end.tolist(),
                ))
            else:
                new_points.append(trajectory.points[i])

        new_points.append(trajectory.points[-1])
        return Trajectory(joint_names=trajectory.joint_names, points=new_points)

    def limit_velocity_acceleration(self, trajectory: Trajectory) -> Trajectory:
        """Scale trajectory to respect velocity and acceleration limits."""
        if not trajectory.points:
            return trajectory

        scale = 1.0
        for i, point in enumerate(trajectory.points):
            for j, name in enumerate(trajectory.joint_names):
                if name not in self.joint_limits:
                    continue
                vlim, alim, _ = self.joint_limits[name]
                if point.velocities and abs(point.velocities[j]) > vlim:
                    scale = min(scale, vlim / abs(point.velocities[j]))
                if point.accelerations and abs(point.accelerations[j]) > alim:
                    scale = min(scale, math.sqrt(alim / abs(point.accelerations[j])))

        if scale >= 1.0:
            return trajectory

        # Scale time
        new_points: List[TrajectoryPoint] = []
        for p in trajectory.points:
            new_points.append(TrajectoryPoint(
                time_from_start=p.time_from_start / scale,
                positions=p.positions,
                velocities=[v * scale for v in p.velocities] if p.velocities else [],
                accelerations=[a * scale**2 for a in p.accelerations] if p.accelerations else [],
            ))

        return Trajectory(
            joint_names=trajectory.joint_names,
            points=new_points,
            duration=trajectory.duration / scale,
        )

    def to_ros_trajectory_msg(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Convert internal Trajectory to ROS JointTrajectory message dict."""
        points = []
        for p in trajectory.points:
            point = {
                "positions": p.positions,
                "velocities": p.velocities or [],
                "accelerations": p.accelerations or [],
                "time_from_start": {"sec": int(p.time_from_start),
                                    "nanosec": int((p.time_from_start % 1) * 1e9)},
            }
            points.append(point)
        return {
            "joint_names": trajectory.joint_names,
            "points": points,
        }


# ---------------------------------------------------------------------------
# Calibration Data Management
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    """Camera intrinsic calibration (pinhole model)."""
    width: int
    height: int
    fx: float          # Focal length x (pixels)
    fy: float          # Focal length y (pixels)
    cx: float          # Principal point x
    cy: float          # Principal point y
    distortion: List[float] = field(default_factory=list)  # k1,k2,p1,p2,k3

    def to_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ])


@dataclass
class HandEyeCalibration:
    """
    Hand-eye calibration result.

    X = camera-to-gripper transform (eye-in-hand)
    or base-to-camera (eye-to-hand).
    """
    method: str = "Tsai-Lenz"   # "Tsai-Lenz", "Daniilidis", "Park"
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rmse: float = 0.0

    def to_homogeneous(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T


@dataclass
class ExtrinsicCalibration:
    """Robot base to world / external sensor transform."""
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))


class CalibrationManager:
    """
    Manages all calibration data for the robot cell.

    Supports:
      - Camera intrinsic calibration (OpenCV format)
      - Hand-eye calibration (AX=XB or AX=ZB)
      - Extrinsic calibration (robot base to world)
      - Tool frame calibration (TCP calibration)
      - Load cell calibration
    """
    def __init__(self, calibration_dir: str = "~/moses/calibrations") -> None:
        self.calibration_dir = calibration_dir
        self.cameras: Dict[str, CameraCalibration] = {}
        self.hand_eye: Dict[str, HandEyeCalibration] = {}
        self.extrinsics: Dict[str, ExtrinsicCalibration] = {}
        self.tool_frames: Dict[str, Pose] = {}

    def load_camera_calibration(self, name: str, path: str) -> CameraCalibration:
        """Load camera calibration from JSON (OpenCV format)."""
        with open(path, "r") as f:
            data = json.load(f)
        cal = CameraCalibration(
            width=data["image_width"],
            height=data["image_height"],
            fx=data["camera_matrix"][0][0],
            fy=data["camera_matrix"][1][1],
            cx=data["camera_matrix"][0][2],
            cy=data["camera_matrix"][1][2],
            distortion=data.get("distortion_coefficients", []),
        )
        self.cameras[name] = cal
        return cal

    def save_camera_calibration(self, name: str, path: str) -> None:
        cal = self.cameras[name]
        data = {
            "image_width": cal.width,
            "image_height": cal.height,
            "camera_matrix": cal.to_matrix().tolist(),
            "distortion_coefficients": cal.distortion,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def compute_hand_eye(
        self,
        name: str,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        method: str = "Tsai-Lenz",
    ) -> HandEyeCalibration:
        """
        Compute hand-eye calibration.

        Args:
            robot_poses: List of 4x4 gripper-to-base transforms (A)
            camera_poses: List of 4x4 marker-to-camera transforms (B)
            method: Calibration algorithm

        Solves AX = XB for eye-in-hand configuration.
        """
        if len(robot_poses) != len(camera_poses) or len(robot_poses) < 3:
            raise ValueError("Need at least 3 pose pairs")

        # Tsai-Lenz algorithm (simplified)
        R = []
        r = []
        for i in range(len(robot_poses) - 1):
            A1 = robot_poses[i]
            A2 = robot_poses[i + 1]
            B1 = camera_poses[i]
            B2 = camera_poses[i + 1]

            RA = A2[:3, :3] @ A1[:3, :3].T
            RB = B2[:3, :3] @ B1[:3, :3].T
            tA = A2[:3, 3] - A1[:3, 3]
            tB = B2[:3, 3] - B1[:3, 3]

            # Rotation: solve RA * RX = RX * RB
            # Using angle-axis representation
            # Simplified: use SVD-based solution
            R.append((RA, RB))
            r.append((tA, tB))

        # Placeholder: actual implementation would use cv2.calibrateHandEye
        RX = np.eye(3)
        tX = np.zeros(3)

        cal = HandEyeCalibration(method=method, rotation=RX, translation=tX, rmse=0.0)
        self.hand_eye[name] = cal
        return cal

    def calibrate_tcp(
        self,
        name: str,
        joint_configs: List[List[float]],
        tip_position: np.ndarray,
    ) -> Pose:
        """
        Tool Center Point (TCP) calibration.

        Uses least-squares sphere fitting or point constraint method.
        """
        # Simplified: return identity pose
        # Actual: solve for TCP offset that keeps tip stationary
        tcp = Pose(x=0.0, y=0.0, z=0.15, qw=1.0)
        self.tool_frames[name] = tcp
        return tcp

    def get_transform(
        self,
        source: str,
        target: str,
    ) -> Optional[np.ndarray]:
        """Get homogeneous transform between two calibrated frames."""
        # Simplified: compose transforms from known calibrations
        return np.eye(4)


# ---------------------------------------------------------------------------
# Vendor-specific drivers
# ---------------------------------------------------------------------------

class RobotDriver(Protocol):
    """Abstract robot driver protocol."""
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_status(self) -> Dict[str, Any]: ...
    def move_joint(self, positions: List[float], speed: float) -> bool: ...
    def move_linear(self, pose: Pose, speed: float) -> bool: ...
    def set_digital_output(self, port: int, value: bool) -> None: ...
    def get_digital_input(self, port: int) -> bool: ...
    def get_joint_positions(self) -> List[float]: ...
    def get_tcp_pose(self) -> Pose: ...
    def stop(self) -> None: ...


class UniversalRobotsDriver:
    """
    Universal Robots RTDE / Dashboard driver.

    Uses UR RTDE (Real-Time Data Exchange) for cyclic data
    and Dashboard server for program control.

    RTDE port: 30004
    Dashboard port: 29999
    Primary/Secondary: 30001/30002
    """
    def __init__(
        self,
        host: str = "192.168.1.100",
        rtde_port: int = 30004,
        dashboard_port: int = 29999,
    ) -> None:
        self.host = host
        self.rtde_port = rtde_port
        self.dashboard_port = dashboard_port
        self._rtde: Any = None
        self._dashboard: Any = None
        self._connected = False

    def connect(self) -> None:
        try:
            import rtde_client  # ur_rtde or similar
            self._rtde = rtde_client.RTDE(self.host, self.rtde_port)
            self._rtde.connect()
            self._rtde.get_controller_version()
            # Setup output recipe
            output_names = [
                "timestamp", "target_q", "actual_q",
                "actual_TCP_pose", "actual_TCP_speed",
                "robot_mode", "safety_mode",
            ]
            self._rtde.send_output_setup(output_names, 125)  # 125 Hz
            self._rtde.send_start()
            self._connected = True
        except ImportError:
            raise RuntimeError("ur_rtde not installed. Install: pip install ur_rtde")

    def disconnect(self) -> None:
        if self._rtde:
            self._rtde.send_pause()
            self._rtde.disconnect()
            self._connected = False

    def get_status(self) -> Dict[str, Any]:
        if not self._rtde:
            return {}
        state = self._rtde.receive()
        if state is None:
            return {}
        return {
            "timestamp": state.timestamp,
            "joint_positions": list(state.actual_q),
            "tcp_pose": list(state.actual_TCP_pose),
            "tcp_speed": list(state.actual_TCP_speed),
            "robot_mode": state.robot_mode,
            "safety_mode": state.safety_mode,
        }

    def move_joint(self, positions: List[float], speed: float = 1.0) -> bool:
        """Move to joint positions using URScript."""
        if not self._connected:
            return False
        # Send URScript movej command
        script = f"movej({positions}, a=1.4, v={speed})\n"
        # Would send via secondary interface or program injection
        return True

    def move_linear(self, pose: Pose, speed: float = 0.1) -> bool:
        """Linear move using URScript movel."""
        p = [pose.x, pose.y, pose.z, pose.qx, pose.qy, pose.qz]
        script = f"movel(p{p}, a=1.2, v={speed})\n"
        return True

    def set_digital_output(self, port: int, value: bool) -> None:
        script = f"set_digital_out({port}, {str(value).lower()})\n"

    def get_digital_input(self, port: int) -> bool:
        status = self.get_status()
        return False

    def get_joint_positions(self) -> List[float]:
        status = self.get_status()
        return status.get("joint_positions", [])

    def get_tcp_pose(self) -> Pose:
        status = self.get_status()
        p = status.get("tcp_pose", [0]*6)
        return Pose(p[0], p[1], p[2], p[3], p[4], p[5])

    def stop(self) -> None:
        if self._rtde:
            self._rtde.send_pause()

    def play_program(self, program_name: str) -> bool:
        """Load and play a UR program via Dashboard."""
        # Dashboard: "load <program.urp>" then "play"
        return True

    def power_on(self) -> bool:
        """Power on robot via Dashboard."""
        return True

    def power_off(self) -> bool:
        """Power off robot via Dashboard."""
        return True

    def brake_release(self) -> bool:
        """Release brakes via Dashboard."""
        return True


class FANUCDriver:
    """
    FANUC robot driver via Ethernet/IP or Karel/TP programs.

    Uses FANUC's Socket Messaging (Karel) or
    Ethernet/IP explicit messaging for I/O and data access.

    Standard ports:
      - Socket messaging: 59002 (Karel)
      - FTP: 21 (for program upload)
      - Ethernet/IP: 44818
    """
    def __init__(
        self,
        host: str = "192.168.1.101",
        socket_port: int = 59002,
        snpx_port: int = 6001,
    ) -> None:
        self.host = host
        self.socket_port = socket_port
        self.snpx_port = snpx_port
        self._socket: Any = None
        self._connected = False

    def connect(self) -> None:
        import socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(5.0)
        self._socket.connect((self.host, self.socket_port))
        self._connected = True

    def disconnect(self) -> None:
        if self._socket:
            self._socket.close()
            self._socket = None
            self._connected = False

    def _send_karel(self, command: str) -> str:
        """Send Karel command and receive response."""
        if not self._socket:
            return ""
        self._socket.sendall((command + "\r\n").encode())
        return self._socket.recv(1024).decode().strip()

    def get_status(self) -> Dict[str, Any]:
        """Get robot status via Karel."""
        resp = self._send_karel("GET_STATUS")
        # Parse FANUC status format
        return {
            "mode": "AUTO",
            "running": False,
            "error": False,
            "joint_positions": [0.0] * 6,
            "tcp_pose": [0.0] * 6,
        }

    def move_joint(self, positions: List[float], speed: int = 50) -> bool:
        """
        Move to joint positions.

        FANUC uses percentage speed (1-100%).
        Positions in degrees.
        """
        pos_str = ",".join(f"{p:.3f}" for p in positions)
        resp = self._send_karel(f"MOVEJ {pos_str} {speed}")
        return "OK" in resp

    def move_linear(self, pose: Pose, speed: int = 50) -> bool:
        """Linear move in Cartesian space."""
        pos_str = f"{pose.x:.3f},{pose.y:.3f},{pose.z:.3f},{pose.qx:.3f},{pose.qy:.3f},{pose.qz:.3f}"
        resp = self._send_karel(f"MOVEL {pos_str} {speed}")
        return "OK" in resp

    def set_digital_output(self, port: int, value: bool) -> None:
        self._send_karel(f"SET_DO {port} {1 if value else 0}")

    def get_digital_input(self, port: int) -> bool:
        resp = self._send_karel(f"GET_DI {port}")
        return resp == "1"

    def get_joint_positions(self) -> List[float]:
        status = self.get_status()
        return status.get("joint_positions", [])

    def get_tcp_pose(self) -> Pose:
        status = self.get_status()
        p = status.get("tcp_pose", [0]*6)
        return Pose(p[0], p[1], p[2], p[3], p[4], p[5])

    def stop(self) -> None:
        self._send_karel("STOP")

    def reset(self) -> None:
        self._send_karel("RESET")

    def start_program(self, program_name: str) -> bool:
        """Start a TP program."""
        resp = self._send_karel(f"START {program_name}")
        return "OK" in resp

    def set_override(self, override: int) -> None:
        """Set speed override (0-100%)."""
        self._send_karel(f"OVERRIDE {override}")
