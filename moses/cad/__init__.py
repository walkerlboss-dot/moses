"""
Moses v4.0 — CAD & Manufacturing Package
=========================================

Professional-grade CAD and manufacturing capabilities for humanoid robot design.

Modules
-------
parametric_humanoid :
    Parametric humanoid CAD generator with kinematic chains, joints,
    actuators, bearings, and multi-format export (URDF, USD, STEP, STL).

mesh_generator :
    3D mesh generation pipeline: visual meshes, collision meshes,
    LOD chains, and export to OBJ, STL, DAE, glTF.

assembly_manager :
    BOM generation, assembly instructions, interference detection,
    mass property calculation, and tolerance analysis.

Quick Start
-----------
>>> from moses.cad import create_humanoid, MeshGenerator, AssemblyManager
>>> humanoid = create_humanoid(height=1.75, mass=75.0)
>>> humanoid.export("moses.urdf", ExportFormat.URDF)
>>> gen = MeshGenerator()
>>> lod = gen.generate_lod_chain("torso", {"type": "box", "size": (0.3, 0.2, 0.4)})
>>> mgr = AssemblyManager()
>>> mgr.generate_bom_from_humanoid(humanoid)
>>> mgr.export_bom_csv("bom.csv")

Dependencies
------------
- trimesh (required for mesh operations and most exports)
- cadquery (optional, for STEP export)
- build123d (optional, for STEP export)
- numpy (required)
- scipy (optional, for advanced spatial queries)
"""

from __future__ import annotations

__version__ = "4.0.0"
__author__ = "Moses CAD Team"

# ---------------------------------------------------------------------------
# Core imports — gracefully degrade if modules have missing deps
# ---------------------------------------------------------------------------

try:
    from .parametric_humanoid import (
        ParametricHumanoid,
        HumanoidParameters,
        JointDef,
        LinkDef,
        JointType,
        ExportFormat,
        ActuatorDef,
        BearingDef,
        FastenerDef,
        JointLimits,
        GeometryBuilder,
        TrimeshFactory,
        create_humanoid,
        BIOMECH_RATIOS,
        MASS_RATIOS,
    )
except Exception as _exc:
    import logging
    logging.getLogger(__name__).warning(
        "parametric_humanoid import failed: %s", _exc
    )
    ParametricHumanoid = None  # type: ignore
    HumanoidParameters = None  # type: ignore
    JointDef = None  # type: ignore
    LinkDef = None  # type: ignore
    JointType = None  # type: ignore
    ExportFormat = None  # type: ignore
    ActuatorDef = None  # type: ignore
    BearingDef = None  # type: ignore
    FastenerDef = None  # type: ignore
    JointLimits = None  # type: ignore
    GeometryBuilder = None  # type: ignore
    TrimeshFactory = None  # type: ignore
    create_humanoid = None  # type: ignore
    BIOMECH_RATIOS = {}
    MASS_RATIOS = {}

try:
    from .mesh_generator import (
        MeshGenerator,
        MeshSpec,
        MeshResult,
        MeshPurpose,
        LODLevel,
        CollisionMeshFactory,
        VisualMeshFactory,
        generate_collision_mesh,
        generate_visual_mesh,
        generate_lod_chain,
    )
except Exception as _exc:
    import logging
    logging.getLogger(__name__).warning(
        "mesh_generator import failed: %s", _exc
    )
    MeshGenerator = None  # type: ignore
    MeshSpec = None  # type: ignore
    MeshResult = None  # type: ignore
    MeshPurpose = None  # type: ignore
    LODLevel = None  # type: ignore
    CollisionMeshFactory = None  # type: ignore
    VisualMeshFactory = None  # type: ignore
    generate_collision_mesh = None  # type: ignore
    generate_visual_mesh = None  # type: ignore
    generate_lod_chain = None  # type: ignore

try:
    from .assembly_manager import (
        AssemblyManager,
        BOMItem,
        BOMCategory,
        AssemblyStep,
        MassProperties,
        InterferenceReport,
        ToleranceSpec,
        ToleranceAnalysis,
        compute_center_of_mass,
        parallel_axis_theorem,
    )
except Exception as _exc:
    import logging
    logging.getLogger(__name__).warning(
        "assembly_manager import failed: %s", _exc
    )
    AssemblyManager = None  # type: ignore
    BOMItem = None  # type: ignore
    BOMCategory = None  # type: ignore
    AssemblyStep = None  # type: ignore
    MassProperties = None  # type: ignore
    InterferenceReport = None  # type: ignore
    ToleranceSpec = None  # type: ignore
    ToleranceAnalysis = None  # type: ignore
    compute_center_of_mass = None  # type: ignore
    parallel_axis_theorem = None  # type: ignore


# ---------------------------------------------------------------------------
# Package-level convenience
# ---------------------------------------------------------------------------

def get_version() -> str:
    """Return package version string."""
    return __version__


def check_dependencies() -> dict:
    """Check which CAD backends are available."""
    result = {}
    try:
        import trimesh
        result["trimesh"] = trimesh.__version__
    except Exception:
        result["trimesh"] = False
    try:
        import cadquery
        result["cadquery"] = True
    except Exception:
        result["cadquery"] = False
    try:
        import build123d
        result["build123d"] = True
    except Exception:
        result["build123d"] = False
    try:
        import scipy
        result["scipy"] = scipy.__version__
    except Exception:
        result["scipy"] = False
    try:
        import numpy
        result["numpy"] = numpy.__version__
    except Exception:
        result["numpy"] = False
    return result


__all__ = [
    # Version
    "__version__",
    "get_version",
    "check_dependencies",
    # parametric_humanoid
    "ParametricHumanoid",
    "HumanoidParameters",
    "JointDef",
    "LinkDef",
    "JointType",
    "ExportFormat",
    "ActuatorDef",
    "BearingDef",
    "FastenerDef",
    "JointLimits",
    "GeometryBuilder",
    "TrimeshFactory",
    "create_humanoid",
    "BIOMECH_RATIOS",
    "MASS_RATIOS",
    # mesh_generator
    "MeshGenerator",
    "MeshSpec",
    "MeshResult",
    "MeshPurpose",
    "LODLevel",
    "CollisionMeshFactory",
    "VisualMeshFactory",
    "generate_collision_mesh",
    "generate_visual_mesh",
    "generate_lod_chain",
    # assembly_manager
    "AssemblyManager",
    "BOMItem",
    "BOMCategory",
    "AssemblyStep",
    "MassProperties",
    "InterferenceReport",
    "ToleranceSpec",
    "ToleranceAnalysis",
    "compute_center_of_mass",
    "parallel_axis_theorem",
]
