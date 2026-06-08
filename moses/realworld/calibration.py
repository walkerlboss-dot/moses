"""
Robot Calibration Module
========================

Comprehensive calibration for kinematics, force/torque sensors, cameras,
and hand-eye coordination.

References:
-----------
[1] Tsai & Lenz, "A New Technique for Fully Autonomous and Efficient 3D Robotics
    Hand/Eye Calibration", IEEE T-RA, 1989.
[2] Zhang, "A Flexible New Technique for Camera Calibration", IEEE T-PAMI, 2000.
[3] He et al., "A New Method for Camera Calibration with Lens Distortion",
    ICRA 2013.
[4] Gautier et al. [2] for kinematic calibration
[5] Focchi et al. [4] for force sensor calibration
[6] Hartley & Zisserman, "Multiple View Geometry in Computer Vision", 2004.

Author: Moses Team
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation as R
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Container for calibration results."""
    parameters: Dict
    residuals: np.ndarray
    rmse: float
    covariance: Optional[np.ndarray] = None
    confidence_95: Optional[Dict] = None


class KinematicCalibrator:
    """
    Kinematic calibration using modified Denavit-Hartenberg (DH) parameters.
    
    Calibrates link lengths, joint offsets, and twist angles to minimize
    end-effector position errors.
    
    References:
    [1] Gautier et al. [2] for geometric parameter identification
    [2] Tsai & Lenz [1] for hand-eye calibration
    """
    
    def __init__(self, num_joints: int = 12):
        self.num_joints = num_joints
        # Nominal DH parameters: [alpha, a, d, theta_offset] per joint
        self.nominal_dh = np.zeros((num_joints, 4))
        
    def set_nominal_dh(self, dh_params: np.ndarray):
        """Set nominal DH parameters."""
        self.nominal_dh = dh_params.copy()
        
    def dh_transform(self, alpha: float, a: float, d: float, theta: float) -> np.ndarray:
        """Compute DH transformation matrix."""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
    
    def forward_kinematics(self,
                           q: np.ndarray,
                           dh_params: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics with given DH parameters.
        
        Args:
            q: Joint angles (n_joints,)
            dh_params: DH parameters (n_joints, 4)
        
        Returns:
            End-effector pose (4, 4)
        """
        T = np.eye(4)
        for i in range(self.num_joints):
            alpha, a, d, theta_offset = dh_params[i]
            theta = q[i] + theta_offset
            T_i = self.dh_transform(alpha, a, d, theta)
            T = T @ T_i
        return T
    
    def calibrate(self,
                  joint_angles: np.ndarray,
                  measured_poses: np.ndarray,
                  calibrate_mask: np.ndarray = None) -> CalibrationResult:
        """
        Calibrate DH parameters from measured end-effector poses.
        
        Args:
            joint_angles: (N, n_joints) measured joint angles
            measured_poses: (N, 4, 4) measured end-effector poses
            calibrate_mask: (n_joints, 4) boolean mask for parameters to calibrate
        
        Returns:
            CalibrationResult with optimized parameters
        """
        if calibrate_mask is None:
            # Calibrate all parameters by default
            calibrate_mask = np.ones((self.num_joints, 4), dtype=bool)
            
        # Flatten parameters to optimize
        nominal_flat = self.nominal_dh.flatten()
        mask_flat = calibrate_mask.flatten()
        x0 = nominal_flat[mask_flat]
        
        def residuals(x):
            """Compute position residuals."""
            params = nominal_flat.copy()
            params[mask_flat] = x
            dh = params.reshape(self.num_joints, 4)
            
            res = []
            for q, T_meas in zip(joint_angles, measured_poses):
                T_pred = self.forward_kinematics(q, dh)
                pos_pred = T_pred[:3, 3]
                pos_meas = T_meas[:3, 3]
                res.extend(pos_pred - pos_meas)
                
                # Orientation error (axis-angle)
                R_pred = T_pred[:3, :3]
                R_meas = T_meas[:3, :3]
                R_err = R_pred.T @ R_meas
                angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
                res.append(angle)
                
            return np.array(res)
        
        result = least_squares(
            residuals, x0,
            method='lm',
            max_nfev=10000
        )
        
        # Reconstruct calibrated parameters
        calibrated = nominal_flat.copy()
        calibrated[mask_flat] = result.x
        dh_calibrated = calibrated.reshape(self.num_joints, 4)
        
        rmse = np.sqrt(np.mean(result.fun**2))
        
        return CalibrationResult(
            parameters={'dh_params': dh_calibrated},
            residuals=result.fun,
            rmse=rmse,
            covariance=result.cost if hasattr(result, 'cost') else None
        )
    
    def validate(self,
                 joint_angles: np.ndarray,
                 measured_poses: np.ndarray,
                 dh_params: np.ndarray) -> float:
        """Validate calibration on test data."""
        errors = []
        for q, T_meas in zip(joint_angles, measured_poses):
            T_pred = self.forward_kinematics(q, dh_params)
            pos_error = np.linalg.norm(T_pred[:3, 3] - T_meas[:3, 3])
            errors.append(pos_error)
        return np.mean(errors)


class ForceTorqueCalibrator:
    """
    Calibrate force/torque sensors for bias, scale, and cross-talk.
    
    References:
    [1] Focchi et al. [4] for force sensor calibration
    [2] Gautier et al. [2] for load identification
    """
    
    def __init__(self):
        self.bias = np.zeros(6)
        self.scale = np.eye(6)
        self.crosstalk = np.zeros((6, 6))
        self.is_calibrated = False
        
    def calibrate_bias(self, zero_load_data: np.ndarray) -> np.ndarray:
        """
        Calibrate zero-load bias.
        
        Args:
            zero_load_data: (N, 6) sensor readings at zero load
        
        Returns:
            Bias vector (6,)
        """
        self.bias = zero_load_data.mean(axis=0)
        self.is_calibrated = True
        return self.bias
    
    def calibrate_scale_crosstalk(self,
                                   known_loads: np.ndarray,
                                   sensor_readings: np.ndarray) -> Dict:
        """
        Calibrate scale factors and cross-talk matrix.
        
        Model: F_meas = C * S * (F_true + b)
        where C is crosstalk matrix, S is diagonal scale, b is bias.
        
        Args:
            known_loads: (N, 6) ground truth forces/torques
            sensor_readings: (N, 6) raw sensor readings
        
        Returns:
            Calibration parameters
        """
        # Remove bias
        readings_centered = sensor_readings - self.bias
        
        # Solve for combined transformation matrix M = C * S
        M, residuals, rank, s = np.linalg.lstsq(
            known_loads, readings_centered, rcond=None
        )
        
        # Decompose into scale and crosstalk
        # M should be close to diagonal for well-designed sensor
        scale_diag = np.diag(M)
        self.scale = np.diag(scale_diag)
        self.crosstalk = M / scale_diag[:, None] - np.eye(6)
        
        # Ensure diagonal of crosstalk is zero
        np.fill_diagonal(self.crosstalk, 0)
        
        rmse = np.sqrt(np.mean(residuals)) if len(residuals) > 0 else 0
        
        return {
            'scale': self.scale,
            'crosstalk': self.crosstalk,
            'full_matrix': M,
            'rmse': rmse,
        }
    
    def apply_calibration(self, raw_reading: np.ndarray) -> np.ndarray:
        """Apply calibration to raw sensor reading."""
        if not self.is_calibrated:
            return raw_reading
            
        centered = raw_reading - self.bias
        # Invert crosstalk and scale
        M = (np.eye(6) + self.crosstalk) @ self.scale
        calibrated = np.linalg.solve(M.T, centered)
        return calibrated
    
    def calibrate_from_gravity(self,
                               robot_poses: np.ndarray,
                               sensor_readings: np.ndarray,
                               end_effector_mass: float) -> CalibrationResult:
        """
        Calibrate using known gravity load at different orientations.
        
        Args:
            robot_poses: (N, 4, 4) end-effector poses
            sensor_readings: (N, 6) sensor readings
            end_effector_mass: Known mass of end-effector (kg)
        
        Returns:
            CalibrationResult
        """
        g = 9.81
        
        # Expected forces in sensor frame
        expected_forces = []
        for T in robot_poses:
            R = T[:3, :3]
            # Gravity in world frame
            F_world = np.array([0, 0, -end_effector_mass * g])
            # Transform to sensor frame (assuming sensor aligned with EE)
            F_sensor = R.T @ F_world
            expected_forces.append(F_sensor)
            
        expected_forces = np.array(expected_forces)
        
        # Use only force components for calibration (first 3)
        force_readings = sensor_readings[:, :3]
        
        # Calibrate
        cal = self.calibrate_scale_crosstalk(expected_forces, force_readings)
        
        return CalibrationResult(
            parameters=cal,
            residuals=force_readings - expected_forces,
            rmse=cal['rmse']
        )


class CameraCalibrator:
    """
    Camera calibration for intrinsic and extrinsic parameters.
    
    Uses OpenCV's calibration routines with support for:
    - Pinhole and fisheye models
    - Radial and tangential distortion
    - Stereo calibration
    
    References:
    [1] Zhang [2] for camera calibration
    [2] He et al. [3] for distortion model
    [3] Hartley & Zisserman [6] for multi-view geometry
    """
    
    def __init__(self, pattern_size: Tuple[int, int] = (9, 6),
                 square_size: float = 0.025):
        """
        Args:
            pattern_size: (width, height) of checkerboard corners
            square_size: Size of checkerboard squares in meters
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.objp = self._create_object_points()
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
    def _create_object_points(self) -> np.ndarray:
        """Create 3D object points for checkerboard."""
        objp = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0],
                               0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def detect_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect checkerboard corners in image.
        
        Args:
            image: Grayscale or color image
        
        Returns:
            Corner points (N, 1, 2) or None if not found
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        ret, corners = cv2.findChessboardCorners(
            gray, self.pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                  cv2.CALIB_CB_FAST_CHECK
        )
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners
        return None
    
    def calibrate_intrinsic(self,
                           images: List[np.ndarray],
                           image_size: Tuple[int, int] = None) -> CalibrationResult:
        """
        Calibrate camera intrinsics from checkerboard images.
        
        Args:
            images: List of calibration images
            image_size: (width, height) if known
        
        Returns:
            CalibrationResult with camera_matrix, dist_coeffs
        """
        objpoints = []
        imgpoints = []
        
        for img in images:
            corners = self.detect_corners(img)
            if corners is not None:
                objpoints.append(self.objp)
                imgpoints.append(corners)
                
        if len(objpoints) < 3:
            raise ValueError(f"Need at least 3 valid images, found {len(objpoints)}")
            
        # Determine image size
        if image_size is None:
            image_size = (images[0].shape[1], images[0].shape[0])
            
        # Calibrate
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None,
            flags=cv2.CALIB_RATIONAL_MODEL
        )
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        # Compute reprojection error
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
            
        rmse = total_error / len(objpoints)
        
        return CalibrationResult(
            parameters={
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'fx': camera_matrix[0, 0],
                'fy': camera_matrix[1, 1],
                'cx': camera_matrix[0, 2],
                'cy': camera_matrix[1, 2],
            },
            residuals=np.array([]),
            rmse=rmse
        )
    
    def calibrate_extrinsic(self,
                           image: np.ndarray,
                           known_pattern_pose: np.ndarray) -> np.ndarray:
        """
        Calibrate extrinsic parameters (camera pose relative to pattern).
        
        Args:
            image: Image with visible checkerboard
            known_pattern_pose: (4, 4) pattern pose in world frame
        
        Returns:
            Camera pose in world frame (4, 4)
        """
        corners = self.detect_corners(image)
        if corners is None:
            raise ValueError("Could not detect checkerboard corners")
            
        if self.camera_matrix is None:
            raise ValueError("Must calibrate intrinsics first")
            
        ret, rvec, tvec = cv2.solvePnP(
            self.objp, corners, self.camera_matrix, self.dist_coeffs
        )
        
        if not ret:
            raise ValueError("PnP solve failed")
            
        # Convert to transformation matrix
        R_cam, _ = cv2.Rodrigues(rvec)
        T_cam_pattern = np.eye(4)
        T_cam_pattern[:3, :3] = R_cam
        T_cam_pattern[:3, 3] = tvec.flatten()
        
        # Camera pose in world frame
        T_pattern_world = known_pattern_pose
        T_cam_world = T_pattern_world @ np.linalg.inv(T_cam_pattern)
        
        return T_cam_world
    
    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Undistort image using calibrated parameters."""
        if self.camera_matrix is None:
            return image
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
    
    def project_points(self,
                       points_3d: np.ndarray,
                       rvec: np.ndarray = None,
                       tvec: np.ndarray = None) -> np.ndarray:
        """
        Project 3D points to image plane.
        
        Args:
            points_3d: (N, 3) 3D points
            rvec: Rotation vector (optional)
            tvec: Translation vector (optional)
        
        Returns:
            (N, 2) projected points
        """
        if rvec is None:
            rvec = np.zeros(3)
        if tvec is None:
            tvec = np.zeros(3)
            
        points_2d, _ = cv2.projectPoints(
            points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        return points_2d.reshape(-1, 2)


class HandEyeCalibrator:
    """
    Hand-eye calibration: find transformation between robot end-effector
    and camera.
    
    Supports both eye-in-hand and eye-to-hand configurations.
    
    References:
    [1] Tsai & Lenz [1] for classical hand-eye calibration
    [2] Daniilidis, "Hand-Eye Calibration Using Dual Quaternions", IJRR 1999.
    """
    
    def __init__(self, configuration: str = "eye_in_hand"):
        """
        Args:
            configuration: 'eye_in_hand' or 'eye_to_hand'
        """
        if configuration not in ["eye_in_hand", "eye_to_hand"]:
            raise ValueError("Configuration must be 'eye_in_hand' or 'eye_to_hand'")
        self.configuration = configuration
        self.T_hand_eye = None
        
    def calibrate_tsai(self,
                       robot_poses: List[np.ndarray],
                       camera_poses: List[np.ndarray]) -> np.ndarray:
        """
        Tsai-Lenz hand-eye calibration.
        
        Solves AX = XB where:
        - eye_in_hand: A = robot motion, B = camera motion, X = hand-to-eye
        - eye_to_hand: A = robot motion, B = camera motion, X = hand-to-camera
        
        Args:
            robot_poses: List of (4, 4) robot end-effector poses
            camera_poses: List of (4, 4) camera poses (from calibration target)
        
        Returns:
            T_hand_eye: (4, 4) transformation
        """
        if len(robot_poses) < 2:
            raise ValueError("Need at least 2 poses")
            
        n = len(robot_poses)
        
        # Build relative motions
        A_list = []
        B_list = []
        
        for i in range(n - 1):
            # Robot motion: T_i^{-1} * T_{i+1}
            A = np.linalg.inv(robot_poses[i]) @ robot_poses[i + 1]
            # Camera motion
            B = np.linalg.inv(camera_poses[i]) @ camera_poses[i + 1]
            A_list.append(A)
            B_list.append(B)
            
        # Solve for rotation using least squares
        # From Tsai-Lenz: skew(Ra_i) * R = skew(Rb_i)
        M = np.zeros((3, 3))
        
        for A, B in zip(A_list, B_list):
            Ra = A[:3, :3]
            Rb = B[:3, :3]
            
            # Axis-angle representation
            theta_a = np.arccos(np.clip((np.trace(Ra) - 1) / 2, -1, 1))
            theta_b = np.arccos(np.clip((np.trace(Rb) - 1) / 2, -1, 1))
            
            if abs(theta_a) < 1e-6 or abs(theta_b) < 1e-6:
                continue
                
            # Rotation axis
            k_a = np.array([Ra[2, 1] - Ra[1, 2],
                           Ra[0, 2] - Ra[2, 0],
                           Ra[1, 0] - Ra[0, 1]]) / (2 * np.sin(theta_a))
            k_b = np.array([Rb[2, 1] - Rb[1, 2],
                           Rb[0, 2] - Rb[2, 0],
                           Rb[1, 0] - Rb[0, 1]]) / (2 * np.sin(theta_b))
            
            # Build equation
            M += np.outer(k_a, k_b)
            
        # SVD for rotation
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = U @ Vt
            
        # Solve for translation
        C = np.zeros((3 * len(A_list), 3))
        d = np.zeros(3 * len(A_list))
        
        for i, (A, B) in enumerate(zip(A_list, B_list)):
            Ra = A[:3, :3]
            ta = A[:3, 3]
            tb = B[:3, 3]
            
            C[3*i:3*i+3] = Ra - np.eye(3)
            d[3*i:3*i+3] = R @ tb - ta
            
        t, _, _, _ = np.linalg.lstsq(C, d, rcond=None)
        
        # Construct transformation
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        self.T_hand_eye = T
        return T
    
    def calibrate_daniilidis(self,
                             robot_poses: List[np.ndarray],
                             camera_poses: List[np.ndarray]) -> np.ndarray:
        """
        Dual quaternion hand-eye calibration (more robust for small motions).
        
        From Daniilidis [2]: Uses dual quaternions to solve AX = XB
        simultaneously for rotation and translation.
        """
        n = len(robot_poses)
        
        # Build system of equations using dual quaternions
        T = np.zeros((6, 8))
        
        for i in range(n - 1):
            A = np.linalg.inv(robot_poses[i]) @ robot_poses[i + 1]
            B = np.linalg.inv(camera_poses[i]) @ camera_poses[i + 1]
            
            # Extract rotation and translation
            Ra = A[:3, :3]
            ta = A[:3, 3]
            Rb = B[:3, :3]
            tb = B[:3, 3]
            
            # Quaternion representation
            qa = R.from_matrix(Ra).as_quat()  # [x, y, z, w]
            qb = R.from_matrix(Rb).as_quat()
            
            # Build constraint matrix (simplified)
            # Full dual quaternion formulation is more involved
            # This is a placeholder for the complete implementation
            
        # Solve using SVD
        _, _, Vt = np.linalg.svd(T)
        solution = Vt[-1]
        
        # Extract rotation and translation from dual quaternion
        q_rot = solution[:4]
        q_dual = solution[4:]
        
        R_mat = R.from_quat(q_rot).as_matrix()
        t = 2 * np.array([
            q_dual[0] * q_rot[3] - q_dual[3] * q_rot[0] + q_dual[2] * q_rot[1] - q_dual[1] * q_rot[2],
            q_dual[1] * q_rot[3] - q_dual[2] * q_rot[0] - q_dual[3] * q_rot[1] + q_dual[0] * q_rot[2],
            q_dual[2] * q_rot[3] + q_dual[1] * q_rot[0] - q_dual[0] * q_rot[1] - q_dual[3] * q_rot[2],
        ])
        
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        
        self.T_hand_eye = T
        return T
    
    def validate(self,
                 robot_poses: List[np.ndarray],
                 camera_poses: List[np.ndarray]) -> float:
        """Validate calibration by computing reprojection error."""
        if self.T_hand_eye is None:
            raise ValueError("Must calibrate first")
            
        errors = []
        for T_robot, T_camera in zip(robot_poses, camera_poses):
            if self.configuration == "eye_in_hand":
                T_pred = T_robot @ self.T_hand_eye
            else:
                T_pred = T_robot @ np.linalg.inv(self.T_hand_eye)
                
            error = np.linalg.norm(T_pred[:3, 3] - T_camera[:3, 3])
            errors.append(error)
            
        return np.mean(errors)


class CalibrationDataCollector:
    """
    Automated data collection for calibration procedures.
    
    Guides robot through predefined poses for calibration data collection.
    """
    
    def __init__(self, robot_interface=None):
        self.robot = robot_interface
        self.data = {
            'joint_angles': [],
            'end_effector_poses': [],
            'images': [],
            'force_readings': [],
        }
        
    def generate_calibration_poses(self,
                                   n_poses: int = 20,
                                   joint_limits: np.ndarray = None) -> List[np.ndarray]:
        """
        Generate diverse calibration poses.
        
        Uses Halton sequence for uniform coverage of joint space.
        
        Args:
            n_poses: Number of poses to generate
            joint_limits: (n_dof, 2) joint limits
        
        Returns:
            List of joint angle configurations
        """
        if joint_limits is None:
            # Default symmetric limits
            joint_limits = np.array([[-np.pi, np.pi]] * 12)
            
        n_dof = len(joint_limits)
        poses = []
        
        # Halton sequence for uniform sampling
        def halton(n, base):
            result = []
            for i in range(n):
                f = 1
                r = 0
                idx = i + 1
                while idx > 0:
                    f = f / base
                    r = r + f * (idx % base)
                    idx = idx // base
                result.append(r)
            return result
        
        # Use first n_dof prime bases
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        sequences = [halton(n_poses, primes[i]) for i in range(n_dof)]
        
        for i in range(n_poses):
            q = np.array([
                joint_limits[j, 0] + sequences[j][i] * (joint_limits[j, 1] - joint_limits[j, 0])
                for j in range(n_dof)
            ])
            poses.append(q)
            
        return poses
    
    def collect_kinematic_data(self,
                               poses: List[np.ndarray],
                               measurement_system) -> Dict:
        """
        Collect data for kinematic calibration.
        
        Args:
            poses: List of joint configurations
            measurement_system: External measurement system (e.g., motion capture)
        
        Returns:
            Collected data dictionary
        """
        for pose in poses:
            if self.robot is not None:
                self.robot.set_joint_positions(pose)
                # Wait for settling
                import time
                time.sleep(0.5)
                
            q = self.robot.get_joint_positions() if self.robot else pose
            T = measurement_system.get_end_effector_pose()
            
            self.data['joint_angles'].append(q)
            self.data['end_effector_poses'].append(T)
            
        return {
            'joint_angles': np.array(self.data['joint_angles']),
            'end_effector_poses': np.array(self.data['end_effector_poses']),
        }
    
    def save_calibration_data(self, filepath: str):
        """Save collected calibration data to file."""
        np.savez(filepath, **self.data)
        logger.info(f"Calibration data saved to {filepath}")
        
    def load_calibration_data(self, filepath: str):
        """Load calibration data from file."""
        loaded = np.load(filepath)
        self.data = {key: loaded[key] for key in loaded.files}
        logger.info(f"Calibration data loaded from {filepath}")
