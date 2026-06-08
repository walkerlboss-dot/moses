"""
Moses v4.0 Manufacturing Engine
===============================
Deep manufacturing capabilities for metal, CNC, 3D printing,
sheet metal, and carbon fiber composite fabrication of humanoid robot parts.

Modules
-------
cnc_generator    : G-code generation for milling, turning, drilling
print_generator  : 3D print preparation and slicer integration
sheet_metal      : Bend allowance, flat patterns, DXF export
carbon_fiber     : Composite layup, mold design, curing cycles

Example
-------
>>> from moses.manufacturing import CNCJob, PrintJob, SheetMetalPart, CompositeLayup
>>> job = CNCJob(controller='haas', material='aluminum_6061')
>>> job.add_operation(FaceMill(...))
>>> gcode = job.generate_gcode()
"""

from .cnc_generator import (
    CNCJob,
    Tool,
    Toolpath,
    FaceMill,
    PocketMill,
    DrillCycle,
    TurnOperation,
    FeedSpeedCalculator,
    SurfaceFinishPredictor,
    GCodeExporter,
)

from .print_generator import (
    PrintJob,
    SlicerProfile,
    SupportGenerator,
    OrientationOptimizer,
    PrintEstimator,
)

from .sheet_metal import (
    SheetMetalPart,
    Bend,
    BendAllowanceCalculator,
    FlatPatternGenerator,
    PunchDieSelector,
    DXFExporter,
)

from .carbon_fiber import (
    CompositeLayup,
    Ply,
    LayupSchedule,
    MoldDesign,
    CuringCycle,
    WeightStrengthCalculator,
    quick_tube_layup,
)

__version__ = "4.0.0"
__all__ = [
    # CNC
    "CNCJob",
    "Tool",
    "Toolpath",
    "FaceMill",
    "PocketMill",
    "DrillCycle",
    "TurnOperation",
    "FeedSpeedCalculator",
    "SurfaceFinishPredictor",
    "GCodeExporter",
    # 3D Print
    "PrintJob",
    "SlicerProfile",
    "SupportGenerator",
    "OrientationOptimizer",
    "PrintEstimator",
    # Sheet Metal
    "SheetMetalPart",
    "Bend",
    "BendAllowanceCalculator",
    "FlatPatternGenerator",
    "PunchDieSelector",
    "DXFExporter",
    # Carbon Fiber
    "CompositeLayup",
    "Ply",
    "LayupSchedule",
    "MoldDesign",
    "CuringCycle",
    "WeightStrengthCalculator",
    "quick_tube_layup",
]
