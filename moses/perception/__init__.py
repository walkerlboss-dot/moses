"""
moses.perception — Moses v6.0 Advanced Perception Stack

Modules:
    vision3d          — Stereo depth, point clouds, 3D detection, visual SLAM
    tactile           — Tactile sensors, slip detection, texture, grasp stability
    force_estimation  — External force estimation, collision detection, admittance/impedance control
    fusion            — Multi-modal Kalman filtering, attention-based sensor fusion

Example:
    >>> from moses.perception import StereoDepthEstimator, PointCloud, SlipDetector
    >>> from moses.perception import MultiModalFusion, KalmanFilter
"""

from .vision3d import (
    CameraIntrinsics,
    SE3,
    PointCloud,
    StereoDepthEstimator,
    PointNetDetector,
    BoundingBox3D,
    SceneUnderstanding,
    ORBSLAMStyle,
)

from .tactile import (
    TactileReading,
    GelSightModel,
    BioTacModel,
    TactileImageProcessor,
    SlipDetector,
    TextureClassifier,
    GraspStabilityEstimator,
)

from .force_estimation import (
    RobotState,
    ExternalWrench,
    InverseDynamics,
    ExternalForceEstimator,
    ContactForceDistribution,
    CollisionDetector,
    AdmittanceController,
    ImpedanceController,
)

from .fusion import (
    GaussianState,
    SensorObservation,
    KalmanFilter,
    ExtendedKalmanFilter,
    CovarianceIntersection,
    UncertaintyWeightedFusion,
    SensorAttention,
    GatingFusion,
    MultiModalFusion,
)

__all__ = [
    # vision3d
    "CameraIntrinsics",
    "SE3",
    "PointCloud",
    "StereoDepthEstimator",
    "PointNetDetector",
    "BoundingBox3D",
    "SceneUnderstanding",
    "ORBSLAMStyle",
    # tactile
    "TactileReading",
    "GelSightModel",
    "BioTacModel",
    "TactileImageProcessor",
    "SlipDetector",
    "TextureClassifier",
    "GraspStabilityEstimator",
    # force_estimation
    "RobotState",
    "ExternalWrench",
    "InverseDynamics",
    "ExternalForceEstimator",
    "ContactForceDistribution",
    "CollisionDetector",
    "AdmittanceController",
    "ImpedanceController",
    # fusion
    "GaussianState",
    "SensorObservation",
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "CovarianceIntersection",
    "UncertaintyWeightedFusion",
    "SensorAttention",
    "GatingFusion",
    "MultiModalFusion",
]

__version__ = "6.0.0"
