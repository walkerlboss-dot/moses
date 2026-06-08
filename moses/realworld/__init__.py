"""
Moses RealWorld-Bridge: Sim-to-Real Deployment Module
======================================================

Provides system identification, domain adaptation, calibration, and safe
deployment for transferring simulation-trained policies to physical robots.

References:
-----------
[1] Zhao et al., "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics:
    A Survey", IEEE T-CDS, 2023.
[2] Tan et al., "Sim-to-Real: Learning Agile Locomotion For Quadruped Robots",
    RSS 2018.
[3] Hwangbo et al., "Learning Agile and Dynamic Motor Skills for Legged Robots",
    Science Robotics, 2019.
[4] Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL",
    IROS 2022.
[5] Margolis et al., "Rapid Locomotion via Reinforcement Learning", RSS 2022.

Modules:
--------
- system_id: Identify physical parameters from robot data
- domain_adaptation: Bridge sim-to-real gap
- calibration: Sensor and kinematic calibration
- deployment: Safe policy deployment

Author: Moses Team
Version: 6.0.0
"""

__version__ = "6.0.0"
__author__ = "Moses Team"

from .system_id import (
    SystemIdentifier,
    MotorModelIdentifier,
    SensorCalibrator,
    ContactModelIdentifier,
    InertiaEstimator,
)

from .domain_adaptation import (
    DomainRandomizationTuner,
    AdversarialDomainAdapter,
    MetaLearningAdapter,
    SimRealGapQuantifier,
    DynamicsRandomizer,
)

from .calibration import (
    KinematicCalibrator,
    ForceTorqueCalibrator,
    CameraCalibrator,
    HandEyeCalibrator,
    CalibrationDataCollector,
)

from .deployment import (
    PolicyDeployer,
    SafetyMonitor,
    EmergencyStop,
    PerformanceLogger,
    DeploymentConfig,
)

__all__ = [
    # System Identification
    "SystemIdentifier",
    "MotorModelIdentifier",
    "SensorCalibrator",
    "ContactModelIdentifier",
    "InertiaEstimator",
    # Domain Adaptation
    "DomainRandomizationTuner",
    "AdversarialDomainAdapter",
    "MetaLearningAdapter",
    "SimRealGapQuantifier",
    "DynamicsRandomizer",
    # Calibration
    "KinematicCalibrator",
    "ForceTorqueCalibrator",
    "CameraCalibrator",
    "HandEyeCalibrator",
    "CalibrationDataCollector",
    # Deployment
    "PolicyDeployer",
    "SafetyMonitor",
    "EmergencyStop",
    "PerformanceLogger",
    "DeploymentConfig",
]
