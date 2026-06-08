"""
Moses Industrial Integration Module
====================================

Industrial-grade integration for humanoid robots:
  - PLC communication (Modbus, OPC-UA, EtherNet/IP)
  - EtherCAT real-time fieldbus
  - ROS2 Industrial / MoveIt
  - Functional safety (ISO 12100, ISO 10218, ISO/TS 15066)

Standards referenced:
  - IEC 61131-3 (PLC programming)
  - IEC 61784 (Industrial communication)
  - ETG.1000.1 (EtherCAT Technology Group)
  - CiA 402 (CANopen device profile for drives)
  - ISO 12100 (Risk assessment)
  - ISO 10218-1/2 (Robot safety)
  - ISO/TS 15066 (Collaborative robots)
  - IEC 62061 / ISO 13849-1 (Safety of machinery)

Version: 6.0.0
"""

__version__ = "6.0.0"
__all__ = [
    "plc",
    "ethercat",
    "ros2_industrial",
    "safety",
]

# Re-export key classes for convenience
from .plc import (
    PLCInterface,
    ModbusClient,
    OPCUAClient,
    EtherNetIPClient,
    SafetyIOMap,
    DigitalIOMap,
)

from .ethercat import (
    EtherCATMaster,
    EtherCATSlave,
    PDORemapping,
    DistributedClocks,
    CiA402DriveProfile,
)

from .ros2_industrial import (
    MoveItInterface,
    TrajectoryProcessor,
    CalibrationManager,
    UniversalRobotsDriver,
    FANUCDriver,
)

from .safety import (
    RiskAssessment,
    SISTEMACalculator,
    PerformanceLevel,
    SafetyPLCInterface,
    CollaborativeSafety,
)
