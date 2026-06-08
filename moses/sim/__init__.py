"""Moses v6.0 Advanced Simulation Package.

Provides multi-physics, contact-rich manipulation, deformable object simulation,
and advanced sensor simulation capabilities.

Modules:
    multiphysics: Coupled rigid body, soft body, fluid, thermal, and EM dynamics
    manipulation: Grasp planning, in-hand manipulation, tactile feedback
    deformable: FEM soft bodies, cloth, liquid/gel, cutting/tearing/folding
    sensors: Camera, tactile, force/torque, IMU, LiDAR sensor simulation

Example:
    >>> from moses.sim import MultiPhysicsEngine, GraspPlanner, SoftBodyFEM, RGBCamera
    >>> engine = MultiPhysicsEngine(dt=1.0/60.0)
    >>> # Add bodies, set up sensors, run simulation
    >>> engine.step()
"""

__version__ = "6.0.0"
__author__ = "Moses Team"

# Multi-physics
from moses.sim.multiphysics import (
    Vec3,
    Quaternion,
    Transform,
    Material,
    RigidBody,
    SoftBody,
    SoftBodyNode,
    SoftBodyElement,
    SPHParticle,
    FluidSimulator,
    ThermalState,
    ThermalSimulator,
    ElectromagneticState,
    ElectromagneticSimulator,
    MultiPhysicsEngine,
)

# Manipulation
from moses.sim.manipulation import (
    ContactPoint,
    FrictionCone,
    GraspContact,
    GraspWrench,
    GraspQualityMetrics,
    GraspPlanner,
    InHandManipulation,
    TactileSensorReading,
    TactileSensorArray,
    SlidingPrimitive,
    RollingPrimitive,
    PivotingPrimitive,
    ManipulationController,
)

# Deformable
from moses.sim.deformable import (
    FEMNode,
    FEMTetrahedron,
    FEMTriangle,
    ConstitutiveModel,
    NeoHookeanModel,
    CorotationalLinearModel,
    StVKModel,
    SoftBodyFEM,
    ClothSpring,
    ClothSimulator,
    FluidParticle,
    LiquidGelSimulator,
    CuttingTool,
    FoldingTool,
    TearingSimulator,
    DeformableObjectFactory,
)

# Sensors
from moses.sim.sensors import (
    SensorConfig,
    BaseSensor,
    CameraConfig,
    RGBCamera,
    DepthCamera,
    StereoCamera,
    EventCamera,
    TactileConfig,
    TactileReading,
    TactileSensor,
    ForceTorqueConfig,
    Wrench6D,
    ForceTorqueSensor,
    DistributedForceSensor,
    IMUConfig,
    IMUReading,
    IMUSensor,
    LiDARConfig,
    LiDARPoint,
    LiDARSensor,
    SensorManager,
)

__all__ = [
    # Version
    "__version__",
    # Multi-physics
    "Vec3",
    "Quaternion",
    "Transform",
    "Material",
    "RigidBody",
    "SoftBody",
    "SoftBodyNode",
    "SoftBodyElement",
    "SPHParticle",
    "FluidSimulator",
    "ThermalState",
    "ThermalSimulator",
    "ElectromagneticState",
    "ElectromagneticSimulator",
    "MultiPhysicsEngine",
    # Manipulation
    "ContactPoint",
    "FrictionCone",
    "GraspContact",
    "GraspWrench",
    "GraspQualityMetrics",
    "GraspPlanner",
    "InHandManipulation",
    "TactileSensorReading",
    "TactileSensorArray",
    "SlidingPrimitive",
    "RollingPrimitive",
    "PivotingPrimitive",
    "ManipulationController",
    # Deformable
    "FEMNode",
    "FEMTetrahedron",
    "FEMTriangle",
    "ConstitutiveModel",
    "NeoHookeanModel",
    "CorotationalLinearModel",
    "StVKModel",
    "SoftBodyFEM",
    "ClothSpring",
    "ClothSimulator",
    "FluidParticle",
    "LiquidGelSimulator",
    "CuttingTool",
    "FoldingTool",
    "TearingSimulator",
    "DeformableObjectFactory",
    # Sensors
    "SensorConfig",
    "BaseSensor",
    "CameraConfig",
    "RGBCamera",
    "DepthCamera",
    "StereoCamera",
    "EventCamera",
    "TactileConfig",
    "TactileReading",
    "TactileSensor",
    "ForceTorqueConfig",
    "Wrench6D",
    "ForceTorqueSensor",
    "DistributedForceSensor",
    "IMUConfig",
    "IMUReading",
    "IMUSensor",
    "LiDARConfig",
    "LiDARPoint",
    "LiDARSensor",
    "SensorManager",
]
