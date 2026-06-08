"""
Moses v4.0 — 3D Mesh Generator
==============================

Collision-mesh and visual-mesh pipeline with LOD support.
Exports: OBJ, STL, DAE, glTF

Libraries: trimesh (primary), numpy
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import numpy.typing as npt

try:
    import trimesh
    from trimesh import Trimesh
    from trimesh.exchange import gltf, obj, stl, dae
    HAS_TRIMESH = True
except Exception as exc:
    HAS_TRIMESH = False
    logging.warning("trimesh not available: %s", exc)

# Optional: pygltflib for stricter glTF validation
try:
    import pygltflib
    HAS_PYGLTFLIB = True
except Exception:
    HAS_PYGLTFLIB = False


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class MeshPurpose(Enum):
    """Purpose of the mesh in the pipeline."""
    VISUAL = auto()
    COLLISION = auto()
    SIMPLIFIED = auto()
    PROXY = auto()


class LODLevel(Enum):
    """Level-of-detail tiers."""
    LOD0 = 0  # Production / highest detail
    LOD1 = 1  # Game / real-time
    LOD2 = 2  # Preview / editor
    LOD3 = 3  # Icon / thumbnail


@dataclass
class MeshSpec:
    """Specification for a mesh to be generated."""
    name: str
    purpose: MeshPurpose
    lod: LODLevel = LODLevel.LOD0
    source_geometry: Optional[Dict[str, Any]] = None
    target_triangles: Optional[int] = None
    convex_hull: bool = False
    voxel_size: Optional[float] = None
    smooth_shading: bool = True
    generate_uvs: bool = False
    texture_resolution: Optional[Tuple[int, int]] = None


@dataclass
class MeshResult:
    """Result of mesh generation."""
    name: str
    mesh: Any  # Trimesh object
    purpose: MeshPurpose
    lod: LODLevel
    triangle_count: int
    vertex_count: int
    bounds: Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    generation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mesh Generator
# ---------------------------------------------------------------------------

class MeshGenerator:
    """
    Generate visual and collision meshes for humanoid links.

    Supports:
    - Convex hull collision meshes
    - Simplified geometry (decimation, voxelization)
    - Multi-level-of-detail (LOD) chains
    - UV generation and material assignment
    """

    # Default triangle budgets per LOD
    DEFAULT_LOD_BUDGETS: Dict[LODLevel, int] = {
        LODLevel.LOD0: 50_000,
        LODLevel.LOD1: 10_000,
        LODLevel.LOD2: 2_500,
        LODLevel.LOD3: 500,
    }

    def __init__(self, default_lod_budgets: Optional[Dict[LODLevel, int]] = None):
        if not HAS_TRIMESH:
            raise RuntimeError("trimesh is required for MeshGenerator")
        self.budgets = default_lod_budgets or self.DEFAULT_LOD_BUDGETS.copy()
        self._cache: Dict[str, MeshResult] = {}

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self, spec: MeshSpec) -> MeshResult:
        """Generate a mesh according to specification."""
        import time
        t0 = time.perf_counter()

        # Build base mesh from geometry description
        base_mesh = self._build_base(spec.source_geometry or {})

        # Apply purpose-specific processing
        if spec.purpose == MeshPurpose.COLLISION:
            mesh = self._make_collision_mesh(base_mesh, spec)
        elif spec.purpose == MeshPurpose.SIMPLIFIED:
            mesh = self._make_simplified_mesh(base_mesh, spec)
        elif spec.purpose == MeshPurpose.PROXY:
            mesh = self._make_proxy_mesh(base_mesh, spec)
        else:
            mesh = self._make_visual_mesh(base_mesh, spec)

        # Apply LOD decimation if target specified
        if spec.target_triangles is not None:
            mesh = self._decimate(mesh, spec.target_triangles)
        elif spec.lod in self.budgets:
            mesh = self._decimate(mesh, self.budgets[spec.lod])

        # Cache and return
        elapsed = (time.perf_counter() - t0) * 1000.0
        result = MeshResult(
            name=spec.name,
            mesh=mesh,
            purpose=spec.purpose,
            lod=spec.lod,
            triangle_count=len(mesh.faces),
            vertex_count=len(mesh.vertices),
            bounds=(mesh.bounds[0], mesh.bounds[1]),
            generation_time_ms=elapsed,
            metadata={
                "is_watertight": mesh.is_watertight,
                "volume": float(mesh.volume) if mesh.is_watertight else None,
                "surface_area": float(mesh.area),
            },
        )
        self._cache[f"{spec.name}_{spec.purpose.name}_{spec.lod.name}"] = result
        return result

    def generate_lod_chain(
        self,
        name: str,
        source_geometry: Dict[str, Any],
        purpose: MeshPurpose = MeshPurpose.VISUAL,
    ) -> Dict[LODLevel, MeshResult]:
        """Generate a full LOD chain for a single link."""
        results: Dict[LODLevel, MeshResult] = {}
        for lod in LODLevel:
            spec = MeshSpec(
                name=name,
                purpose=purpose,
                lod=lod,
                source_geometry=source_geometry,
                target_triangles=self.budgets[lod],
            )
            results[lod] = self.generate(spec)
        return results

    # ------------------------------------------------------------------
    # Base mesh builders
    # ------------------------------------------------------------------

    def _build_base(self, geom: Dict[str, Any]) -> Trimesh:
        """Create a base trimesh from geometry description."""
        gtype = geom.get("type", "box")
        if gtype == "box":
            return self._box(geom.get("size", (0.1, 0.1, 0.1)))
        elif gtype == "cylinder":
            return self._cylinder(geom.get("radius", 0.05), geom.get("height", 0.2))
        elif gtype == "sphere":
            return self._sphere(geom.get("radius", 0.05))
        elif gtype == "ellipsoid":
            return self._ellipsoid(geom.get("size", (0.1, 0.1, 0.12)))
        elif gtype == "cone":
            return self._cone(geom.get("radius", 0.05), geom.get("height", 0.1))
        elif gtype == "capsule":
            return self._capsule(geom.get("radius", 0.03), geom.get("height", 0.1))
        elif gtype == "tapered_cylinder":
            return self._tapered_cylinder(
                geom.get("height", 0.2),
                geom.get("radius_top", 0.05),
                geom.get("radius_bottom", 0.04),
            )
        else:
            logger.warning("Unknown geometry type '%s', falling back to box", gtype)
            return self._box((0.1, 0.1, 0.1))

    def _box(self, size: Tuple[float, float, float]) -> Trimesh:
        return trimesh.creation.box(extents=size)

    def _cylinder(self, radius: float, height: float) -> Trimesh:
        return trimesh.creation.cylinder(radius=radius, height=height, sections=32)

    def _sphere(self, radius: float) -> Trimesh:
        return trimesh.creation.icosphere(subdivisions=2, radius=radius)

    def _ellipsoid(self, size: Tuple[float, float, float]) -> Trimesh:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        sphere.apply_scale(size)
        return sphere

    def _cone(self, radius: float, height: float) -> Trimesh:
        return trimesh.creation.cone(radius=radius, height=height, sections=32)

    def _capsule(self, radius: float, height: float) -> Trimesh:
        return trimesh.creation.capsule(radius=radius, height=height, count=[16, 16])

    def _tapered_cylinder(
        self, height: float, r_top: float, r_bottom: float
    ) -> Trimesh:
        if abs(r_top - r_bottom) < 1e-6:
            return self._cylinder(r_top, height)
        # Approximate frustum with cone + scale
        cone = trimesh.creation.cone(radius=r_bottom, height=height, sections=32)
        # Scale top to match r_top by non-uniform scaling (approximate)
        # Better: create a proper frustum via extrusion
        return cone

    # ------------------------------------------------------------------
    # Purpose-specific processors
    # ------------------------------------------------------------------

    def _make_visual_mesh(self, mesh: Trimesh, spec: MeshSpec) -> Trimesh:
        """Prepare mesh for visual rendering."""
        mesh = mesh.copy()
        if spec.smooth_shading:
            # Smooth normals if mesh is watertight
            if mesh.is_watertight:
                mesh.fix_normals()
        if spec.generate_uvs:
            mesh = self._generate_uvs(mesh)
        return mesh

    def _make_collision_mesh(self, mesh: Trimesh, spec: MeshSpec) -> Trimesh:
        """Create a collision mesh — convex hull or simplified."""
        mesh = mesh.copy()
        if spec.convex_hull:
            mesh = mesh.convex_hull
        else:
            # Voxel-based simplification
            if spec.voxel_size is not None:
                mesh = mesh.voxelized(spec.voxel_size).marching_cubes
            else:
                # Default: convex hull for reliable collision
                mesh = mesh.convex_hull
        mesh.fix_normals()
        return mesh

    def _make_simplified_mesh(self, mesh: Trimesh, spec: MeshSpec) -> Trimesh:
        """Create a simplified mesh for proxy rendering."""
        mesh = mesh.copy()
        target = spec.target_triangles or self.budgets.get(spec.lod, 1000)
        mesh = self._decimate(mesh, target)
        return mesh

    def _make_proxy_mesh(self, mesh: Trimesh, spec: MeshSpec) -> Trimesh:
        """Create an extreme low-poly proxy (bounding box or oriented bbox)."""
        # Return oriented bounding box as mesh
        obb = mesh.bounding_box_oriented
        return obb.to_mesh()

    # ------------------------------------------------------------------
    # Mesh operations
    # ------------------------------------------------------------------

    def _decimate(self, mesh: Trimesh, target_faces: int) -> Trimesh:
        """Reduce face count to target using quadratic decimation."""
        if len(mesh.faces) <= target_faces:
            return mesh
        try:
            # trimesh simplified wrapper
            return mesh.simplify_quadric_decimation(face_count=target_faces)
        except Exception as exc:
            logger.warning("Quadric decimation failed (%s), using convex hull fallback", exc)
            # Fallback: iterative collapse or convex hull
            return mesh.convex_hull

    def _generate_uvs(self, mesh: Trimesh) -> Trimesh:
        """Generate planar UV coordinates."""
        # Simple planar projection on XY
        verts = mesh.vertices
        uv = verts[:, :2].copy()
        # Normalize to 0-1
        uv_min, uv_max = uv.min(axis=0), uv.max(axis=0)
        span = uv_max - uv_min
        span[span == 0] = 1.0
        uv = (uv - uv_min) / span
        mesh.visual.uv = uv
        return mesh

    def merge_meshes(
        self,
        results: List[MeshResult],
        name: str = "merged",
    ) -> MeshResult:
        """Merge multiple mesh results into a single mesh."""
        meshes = [r.mesh for r in results]
        merged = trimesh.util.concatenate(meshes)
        return MeshResult(
            name=name,
            mesh=merged,
            purpose=MeshPurpose.VISUAL,
            lod=LODLevel.LOD0,
            triangle_count=len(merged.faces),
            vertex_count=len(merged.vertices),
            bounds=(merged.bounds[0], merged.bounds[1]),
            generation_time_ms=0.0,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        result: MeshResult,
        path: Union[str, Path],
        fmt: Optional[str] = None,
    ) -> Path:
        """Export a mesh result to file."""
        path = Path(path)
        if fmt is None:
            fmt = path.suffix.lstrip(".").lower()

        mesh = result.mesh
        if fmt in ("stl", "stlb"):
            mesh.export(str(path))
        elif fmt == "obj":
            mesh.export(str(path))
        elif fmt == "dae":
            mesh.export(str(path))
        elif fmt in ("gltf", "glb"):
            self._export_gltf(mesh, path)
        elif fmt == "ply":
            mesh.export(str(path))
        else:
            raise ValueError(f"Unsupported export format: {fmt}")
        return path

    def _export_gltf(self, mesh: Trimesh, path: Path) -> None:
        """Export to glTF 2.0 (binary or JSON)."""
        # trimesh has built-in gltf export
        mesh.export(str(path))

    def export_lod_chain(
        self,
        lod_results: Dict[LODLevel, MeshResult],
        directory: Union[str, Path],
        basename: str,
        fmt: str = "glb",
    ) -> Dict[LODLevel, Path]:
        """Export a full LOD chain to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths: Dict[LODLevel, Path] = {}
        for lod, result in lod_results.items():
            filename = f"{basename}_{lod.name}.{fmt}"
            out_path = directory / filename
            self.export(result, out_path, fmt=fmt)
            paths[lod] = out_path
        return paths

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_cache_key(
        self, name: str, purpose: MeshPurpose, lod: LODLevel
    ) -> str:
        return f"{name}_{purpose.name}_{lod.name}"

    def clear_cache(self) -> None:
        self._cache.clear()

    def cache_stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._cache),
            "total_triangles": sum(r.triangle_count for r in self._cache.values()),
            "keys": list(self._cache.keys()),
        }


# ---------------------------------------------------------------------------
# Collision Mesh Factory
# ---------------------------------------------------------------------------

class CollisionMeshFactory:
    """
    Specialized factory for physics collision meshes.
    Produces convex hulls, decomposition, and voxel proxies.
    """

    def __init__(self):
        if not HAS_TRIMESH:
            raise RuntimeError("trimesh required")

    def convex_hull(self, mesh: Trimesh) -> Trimesh:
        """Compute convex hull of a mesh."""
        return mesh.convex_hull

    def convex_decomposition(
        self,
        mesh: Trimesh,
        max_hulls: int = 10,
        max_vertices_per_hull: int = 64,
    ) -> List[Trimesh]:
        """
        Approximate convex decomposition.
        Falls back to single hull if VHACD not available.
        """
        try:
            # trimesh VHACD wrapper if available
            decomp = mesh.convex_decomposition(
                maxhulls=max_hulls,
                resolution=100_000,
            )
            if isinstance(decomp, list):
                return decomp
            return [decomp]
        except Exception as exc:
            logger.warning("Convex decomposition failed (%s), returning single hull", exc)
            return [mesh.convex_hull]

    def voxel_proxy(
        self, mesh: Trimesh, voxel_size: float
    ) -> Trimesh:
        """Create a voxelized proxy mesh."""
        voxelized = mesh.voxelized(voxel_size)
        return voxelized.marching_cubes

    def oriented_bounding_box(self, mesh: Trimesh) -> Trimesh:
        """Return oriented bounding box as a mesh."""
        return mesh.bounding_box_oriented.to_mesh()

    def sphere_approximation(
        self, mesh: Trimesh, num_spheres: int = 5
    ) -> List[Tuple[npt.NDArray[np.float64], float]]:
        """
        Approximate mesh with a set of bounding spheres.
        Returns list of (center, radius).
        """
        # Simple k-means-like sphere packing via vertex clustering
        from scipy.cluster.vq import kmeans2
        verts = mesh.vertices
        centers, _ = kmeans2(verts, num_spheres, minit="points")
        spheres = []
        for c in centers:
            dists = np.linalg.norm(verts - c, axis=1)
            spheres.append((c, float(dists.max())))
        return spheres


# ---------------------------------------------------------------------------
# Visual Mesh Factory
# ---------------------------------------------------------------------------

class VisualMeshFactory:
    """
    Specialized factory for high-fidelity visual meshes.
    Handles smoothing, UVs, materials, and detail enhancement.
    """

    def __init__(self):
        if not HAS_TRIMESH:
            raise RuntimeError("trimesh required")

    def subdivide(self, mesh: Trimesh, iterations: int = 1) -> Trimesh:
        """Subdivide mesh faces for smoother appearance."""
        for _ in range(iterations):
            mesh = mesh.subdivide()
        return mesh

    def smooth_laplacian(
        self, mesh: Trimesh, iterations: int = 5, lamb: float = 0.5
    ) -> Trimesh:
        """Apply Laplacian smoothing."""
        # trimesh smoothing filter
        mesh = mesh.copy()
        for _ in range(iterations):
            trimesh.smoothing.filter_laplacian(mesh, lamb=lamb)
        return mesh

    def smooth_taubin(
        self, mesh: Trimesh, iterations: int = 5
    ) -> Trimesh:
        """Apply Taubin smoothing (preserves volume better)."""
        mesh = mesh.copy()
        trimesh.smoothing.filter_taubin(mesh, iterations=iterations)
        return mesh

    def apply_color(
        self, mesh: Trimesh, rgba: Tuple[float, float, float, float]
    ) -> Trimesh:
        """Apply vertex color to mesh."""
        mesh = mesh.copy()
        color_uint8 = (np.array(rgba) * 255).astype(np.uint8)
        mesh.visual.vertex_colors = color_uint8
        return mesh

    def bake_ambient_occlusion(
        self, mesh: Trimesh, rays_per_sample: int = 64
    ) -> Trimesh:
        """Approximate ambient occlusion as vertex colors."""
        # Simplified: use ray-mesh occlusion
        mesh = mesh.copy()
        # Placeholder: trimesh has ray tracing for this
        # For now, return mesh unchanged
        return mesh


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def generate_collision_mesh(
    geometry: Dict[str, Any],
    convex_hull: bool = True,
    name: str = "collision",
) -> MeshResult:
    """One-shot collision mesh generation."""
    gen = MeshGenerator()
    spec = MeshSpec(
        name=name,
        purpose=MeshPurpose.COLLISION,
        source_geometry=geometry,
        convex_hull=convex_hull,
    )
    return gen.generate(spec)


def generate_visual_mesh(
    geometry: Dict[str, Any],
    lod: LODLevel = LODLevel.LOD0,
    smooth: bool = True,
    name: str = "visual",
) -> MeshResult:
    """One-shot visual mesh generation."""
    gen = MeshGenerator()
    spec = MeshSpec(
        name=name,
        purpose=MeshPurpose.VISUAL,
        lod=lod,
        source_geometry=geometry,
        smooth_shading=smooth,
    )
    return gen.generate(spec)


def generate_lod_chain(
    geometry: Dict[str, Any],
    name: str = "model",
    purpose: MeshPurpose = MeshPurpose.VISUAL,
) -> Dict[LODLevel, MeshResult]:
    """One-shot LOD chain generation."""
    gen = MeshGenerator()
    return gen.generate_lod_chain(name, geometry, purpose=purpose)


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test: generate a box with LOD chain
    geom = {"type": "box", "size": (0.2, 0.1, 0.3)}
    lod = generate_lod_chain(geom, name="test_box")
    for level, result in lod.items():
        print(f"{level.name}: {result.triangle_count} triangles, "
              f"{result.vertex_count} vertices, "
              f"{result.generation_time_ms:.2f} ms")
