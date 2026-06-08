"""
moses/perception/vision3d.py
3D Vision Module — Moses v6.0 Perception Stack

Provides stereo depth estimation, point cloud processing, 3D object detection,
scene understanding, and visual SLAM.

References:
- Scharstein & Szeliski, "A Taxonomy and Evaluation of Dense Two-Frame Stereo
  Correspondence Algorithms", IJCV 2002.
- Mur-Artal et al., "ORB-SLAM2: An Open-Source SLAM System for Monocular,
  Stereo, and RGB-D Cameras", IEEE Trans. Robotics 2017.
- Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and
  Segmentation", CVPR 2017.
- Engel et al., "Direct Sparse Odometry", IEEE Trans. PAMI 2018.
- Rusu & Cousins, "3D is here: Point Cloud Library (PCL)", ICRA 2011.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
from collections import deque

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants & Types
# ---------------------------------------------------------------------------

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    baseline: float = 0.12  # Stereo baseline in metres

    def K(self) -> FloatArray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass
class SE3:
    """Rigid-body transform (rotation + translation)."""
    R: FloatArray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    t: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def matrix(self) -> FloatArray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    @staticmethod
    def from_matrix(T: FloatArray) -> "SE3":
        return SE3(R=T[:3, :3].copy(), t=T[:3, 3].copy())

    def inverse(self) -> "SE3":
        Rinv = self.R.T
        return SE3(R=Rinv, t=-Rinv @ self.t)

    def __mul__(self, other: "SE3") -> "SE3":
        return SE3(R=self.R @ other.R, t=self.R @ other.t + self.t)


@dataclass
class PointCloud:
    points: FloatArray      # (N, 3)
    colors: Optional[FloatArray] = None   # (N, 3) in [0,1]
    normals: Optional[FloatArray] = None  # (N, 3)

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must be (N,3)")

    def downsample_voxel(self, voxel_size: float) -> "PointCloud":
        """Voxel-grid downsampling (Rusu & Cousins, ICRA 2011)."""
        if len(self.points) == 0:
            return PointCloud(points=self.points.copy())
        voxel_indices = np.floor(self.points / voxel_size).astype(np.int64)
        # Hash each voxel coordinate
        keys = voxel_indices[:, 0] * 73856093 + voxel_indices[:, 1] * 19349663 + voxel_indices[:, 2] * 83492791
        _, unique_inv = np.unique(keys, return_inverse=True)
        # Centroid per voxel
        new_pts = np.zeros((np.max(unique_inv) + 1, 3), dtype=np.float64)
        np.add.at(new_pts, unique_inv, self.points)
        counts = np.bincount(unique_inv, minlength=new_pts.shape[0]).reshape(-1, 1)
        new_pts /= np.maximum(counts, 1)
        return PointCloud(points=new_pts)

    def estimate_normals(self, k: int = 10) -> "PointCloud":
        """Normal estimation via local PCA (Hoppe et al., SIGGRAPH 1992)."""
        from scipy.spatial import KDTree
        tree = KDTree(self.points)
        normals = np.zeros_like(self.points)
        for i, p in enumerate(self.points):
            dists, idx = tree.query(p, k=k + 1)
            neighbours = self.points[idx[1:]]
            cov = np.cov(neighbours.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normals[i] = eigvecs[:, 0]  # smallest eigenvalue
        return PointCloud(points=self.points, colors=self.colors, normals=normals)


# ---------------------------------------------------------------------------
# 1. Stereo Depth Estimation
# ---------------------------------------------------------------------------

class StereoDepthEstimator:
    """
    Semi-global block matching (SGBM) style stereo depth.
    Based on Hirschmüller, "Stereo Processing by Semi-Global Matching
    and Mutual Information", IEEE Trans. PAMI 2008.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        num_disparities: int = 128,
        block_size: int = 5,
        uniqueness_ratio: float = 10.0,
        p1: float = 8.0,
        p2: float = 32.0,
    ):
        self.K = intrinsics
        self.num_disp = num_disparities
        self.block = block_size
        self.uniq = uniqueness_ratio
        self.p1 = p1
        self.p2 = p2

    def _census_transform(self, img: FloatArray) -> IntArray:
        """9×7 Census transform for robust matching (Zabih & Woodfill, ICCV 1994)."""
        h, w = img.shape
        census = np.zeros((h, w), dtype=np.uint64)
        win_h, win_w = 3, 3  # 7×7 is heavy; 3×3 for real-time
        for dy in range(-win_h, win_h + 1):
            for dx in range(-win_w, win_w + 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
                bit = (img > shifted).astype(np.uint64)
                census = (census << 1) | bit
        return census

    def _hamming(self, a: IntArray, b: IntArray) -> IntArray:
        return np.bitwise_xor(a, b)

    def compute_disparity(self, left: FloatArray, right: FloatArray) -> FloatArray:
        """
        Compute disparity map D(x,y) such that:
            Z = (f * baseline) / D
        Returns float disparity in pixels.
        """
        if left.shape != right.shape:
            raise ValueError("Left/right images must match")
        h, w = left.shape
        # Simple block-matching with SAD for speed; census optional upgrade
        disp = np.zeros((h, w), dtype=np.float32)
        half = self.block // 2
        for y in range(half, h - half):
            for x in range(half, w - half):
                best_d = 0
                best_cost = float('inf')
                second_best = float('inf')
                patch_l = left[y - half:y + half + 1, x - half:x + half + 1]
                for d in range(self.num_disp):
                    if x - d < half:
                        break
                    patch_r = right[y - half:y + half + 1, x - d - half:x - d + half + 1]
                    cost = float(np.sum(np.abs(patch_l - patch_r)))
                    if cost < best_cost:
                        second_best = best_cost
                        best_cost = cost
                        best_d = d
                    elif cost < second_best:
                        second_best = cost
                # Uniqueness check
                if second_best > 0 and best_cost / second_best < 1.0 - 1.0 / self.uniq:
                    disp[y, x] = best_d
                else:
                    disp[y, x] = -1.0  # invalid
        # Sub-pixel refinement via parabola fit
        # (omitted for brevity; standard quadratic interpolation)
        return disp

    def disparity_to_depth(self, disparity: FloatArray) -> FloatArray:
        """
        Z = f * B / D  (Scharstein & Szeliski, IJCV 2002)
        Invalid disparities become NaN.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = (self.K.fx * self.K.baseline) / disparity
        depth[disparity <= 0] = np.nan
        return depth

    def depth_to_pointcloud(self, depth: FloatArray, rgb: Optional[FloatArray] = None) -> PointCloud:
        """Back-project depth map to 3D points."""
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        x = (u - self.K.cx) * z / self.K.fx
        y = (v - self.K.cy) * z / self.K.fy
        pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        valid = np.isfinite(pts[:, 2])
        pts = pts[valid]
        colors = None
        if rgb is not None:
            colors = rgb.reshape(-1, 3)[valid] / 255.0
        return PointCloud(points=pts, colors=colors)


# ---------------------------------------------------------------------------
# 2. 3D Object Detection
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox3D:
    centre: FloatArray          # (3,)
    extents: FloatArray         # (3,) half-sizes
    R: FloatArray               # (3, 3) orientation
    label: str = "object"
    score: float = 1.0

    def corners(self) -> FloatArray:
        """8 corners in world frame."""
        cx, cy, cz = np.diag(self.extents)
        local = np.array([
            [cx, cy, cz], [cx, cy, -cz], [cx, -cy, cz], [cx, -cy, -cz],
            [-cx, cy, cz], [-cx, cy, -cz], [-cx, -cy, cz], [-cx, -cy, -cz]
        ], dtype=np.float64)
        return (self.R @ local.T).T + self.centre


class PointNetDetector:
    """
    Lightweight PointNet-style 3D detector (Qi et al., CVPR 2017).
    Uses shared MLP + max-pooling for permutation-invariant features.
    """

    def __init__(self, num_classes: int = 10, feat_dim: int = 128):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # Simplified: random initialisation; in production load trained weights
        self.W1 = np.random.randn(3, 64).astype(np.float64) * 0.01
        self.W2 = np.random.randn(64, 128).astype(np.float64) * 0.01
        self.W3 = np.random.randn(128, feat_dim).astype(np.float64) * 0.01
        self.cls_W = np.random.randn(feat_dim, num_classes).astype(np.float64) * 0.01

    def _mlp(self, x: FloatArray, W: FloatArray) -> FloatArray:
        return np.maximum(x @ W, 0)  # ReLU

    def forward(self, pc: PointCloud) -> Tuple[FloatArray, FloatArray]:
        """
        Returns global feature vector (feat_dim,) and classification logits.
        """
        pts = pc.points  # (N, 3)
        h1 = self._mlp(pts, self.W1)          # (N, 64)
        h2 = self._mlp(h1, self.W2)           # (N, 128)
        h3 = self._mlp(h2, self.W3)           # (N, feat_dim)
        global_feat = np.max(h3, axis=0)      # symmetric function
        logits = global_feat @ self.cls_W
        return global_feat, logits

    def detect(self, pc: PointCloud, threshold: float = 0.5) -> List[BoundingBox3D]:
        """
        Naïve clustering-based detection: segment by Euclidean clustering,
        then fit oriented bounding boxes.
        """
        from scipy.spatial import KDTree
        labels = self._euclidean_cluster(pc.points, cluster_tolerance=0.05, min_size=50)
        unique_labels = set(labels) - {-1}
        detections: List[BoundingBox3D] = []
        for lbl in unique_labels:
            mask = labels == lbl
            cluster = pc.points[mask]
            centre = cluster.mean(axis=0)
            # PCA for orientation
            cov = np.cov(cluster.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            R = eigvecs[:, ::-1]  # sort descending
            local = (R.T @ (cluster - centre).T).T
            extents = np.abs(local).max(axis=0)
            detections.append(BoundingBox3D(centre=centre, extents=extents, R=R))
        return detections

    @staticmethod
    def _euclidean_cluster(pts: FloatArray, cluster_tolerance: float, min_size: int) -> IntArray:
        from scipy.spatial import KDTree
        tree = KDTree(pts)
        n = len(pts)
        visited = np.zeros(n, dtype=bool)
        labels = np.full(n, -1, dtype=np.int64)
        label = 0
        for i in range(n):
            if visited[i]:
                continue
            queue = deque([i])
            cluster = []
            visited[i] = True
            while queue:
                idx = queue.popleft()
                cluster.append(idx)
                neighbours = tree.query_ball_point(pts[idx], cluster_tolerance)
                for nb in neighbours:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            if len(cluster) >= min_size:
                labels[cluster] = label
                label += 1
        return labels


# ---------------------------------------------------------------------------
# 3. Scene Understanding
# ---------------------------------------------------------------------------

class SceneUnderstanding:
    """
    RANSAC plane extraction + free-space estimation.
    Fischler & Bolles, "Random Sample Consensus", CACM 1981.
    """

    def __init__(self, pc: PointCloud):
        self.pc = pc

    def extract_planes(self, distance_threshold: float = 0.02, max_planes: int = 5) -> List[Tuple[FloatArray, float]]:
        """
        Returns list of (normal, d) for plane equation n·x + d = 0.
        """
        pts = self.pc.points.copy()
        normals = self.pc.normals if self.pc.normals is not None else self.pc.estimate_normals().normals
        planes: List[Tuple[FloatArray, float]] = []
        remaining = np.ones(len(pts), dtype=bool)
        for _ in range(max_planes):
            if remaining.sum() < 100:
                break
            idx = np.random.choice(np.where(remaining)[0], size=3, replace=False)
            p0, p1, p2 = pts[idx]
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n /= norm
            d = -n @ p0
            dists = np.abs(pts @ n + d)
            inliers = (dists < distance_threshold) & remaining
            if inliers.sum() < 100:
                continue
            # Refit with all inliers
            A = np.column_stack([pts[inliers], np.ones(inliers.sum())])
            _, _, Vt = np.linalg.svd(A)
            plane = Vt[-1, :]
            n_refined = plane[:3]
            n_refined /= np.linalg.norm(n_refined)
            d_refined = plane[3] / np.linalg.norm(plane[:3])
            planes.append((n_refined, d_refined))
            remaining[inliers] = False
        return planes

    def free_space_map(self, grid_res: float = 0.05, max_range: float = 3.0) -> FloatArray:
        """
        2.5D occupancy grid: 0 = free, 1 = occupied.
        """
        size = int(max_range / grid_res)
        grid = np.zeros((size, size), dtype=np.float32)
        for p in self.pc.points:
            if p[2] <= 0 or p[2] > max_range:
                continue
            ix = int((p[0] + max_range / 2) / grid_res)
            iy = int((p[1] + max_range / 2) / grid_res)
            if 0 <= ix < size and 0 <= iy < size:
                grid[iy, ix] = 1.0
        return grid


# ---------------------------------------------------------------------------
# 4. Visual SLAM
# ---------------------------------------------------------------------------

class ORBSLAMStyle:
    """
    Simplified keyframe-based visual SLAM inspired by ORB-SLAM2
    (Mur-Artal et al., IEEE Trans. Robotics 2017).
    Uses FAST corners + 8-point essential matrix for pose estimation.
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        self.K = intrinsics.K()
        self.Kinv = np.linalg.inv(self.K)
        self.keyframes: List[Dict] = []
        self.current_pose = SE3()
        self.map_points: List[FloatArray] = []

    def _detect_keypoints(self, gray: FloatArray) -> Tuple[FloatArray, IntArray]:
        """FAST corners (Rosten & Drummond, ECCV 2006)."""
        from scipy.ndimage import maximum_filter
        # Approximate FAST with local maxima of Harris response for brevity
        dx = np.gradient(gray, axis=1)
        dy = np.gradient(gray, axis=0)
        Ixx = dx * dx
        Ixy = dx * dy
        Iyy = dy * dy
        # Box filter approx
        k = 0.04
        det = Ixx * Iyy - Ixy ** 2
        trace = Ixx + Iyy
        harris = det - k * trace ** 2
        local_max = maximum_filter(harris, size=7) == harris
        threshold = np.percentile(harris[local_max], 95)
        coords = np.argwhere((harris > threshold) & local_max)
        responses = harris[coords[:, 0], coords[:, 1]]
        # Keep top 500
        if len(coords) > 500:
            idx = np.argsort(responses)[-500:]
            coords = coords[idx]
        return coords.astype(np.float32), responses

    def _match_descriptors(self, desc1: FloatArray, desc2: FloatArray) -> List[Tuple[int, int]]:
        """Brute-force Hamming on binary descriptors (simplified to L2 for demo)."""
        matches: List[Tuple[int, int]] = []
        for i, d1 in enumerate(desc1):
            dists = np.linalg.norm(desc2 - d1, axis=1)
            best = int(np.argmin(dists))
            if dists[best] < 50:  # threshold
                matches.append((i, best))
        return matches

    def _estimate_pose_8point(self, pts1: FloatArray, pts2: FloatArray) -> SE3:
        """
        Normalized 8-point algorithm (Hartley & Zisserman, MVG 2004).
        """
        # Normalise
        pts1n = (self.Kinv @ np.column_stack([pts1, np.ones(len(pts1))]).T).T[:, :2]
        pts2n = (self.Kinv @ np.column_stack([pts2, np.ones(len(pts2))]).T).T[:, :2]
        # Build constraint matrix
        A = np.column_stack([
            pts2n[:, 0] * pts1n[:, 0],
            pts2n[:, 0] * pts1n[:, 1],
            pts2n[:, 0],
            pts2n[:, 1] * pts1n[:, 0],
            pts2n[:, 1] * pts1n[:, 1],
            pts2n[:, 1],
            pts1n[:, 0],
            pts1n[:, 1],
            np.ones(len(pts1n))
        ])
        _, _, Vt = np.linalg.svd(A)
        E = Vt[-1].reshape(3, 3)
        # Enforce rank-2
        U, S, Vt = np.linalg.svd(E)
        S[2] = 0
        E = U @ np.diag(S) @ Vt
        # Extract R, t (four solutions; pick by cheirality)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        t = U[:, 2]
        if np.linalg.det(R1) < 0:
            R1 = -R1
        if np.linalg.det(R2) < 0:
            R2 = -R2
        # Simplified: return first valid; real system tests all 4
        return SE3(R=R1, t=t)

    def process_frame(self, gray: FloatArray, depth: Optional[FloatArray] = None) -> SE3:
        """
        Track current frame against last keyframe.
        Returns updated camera pose in world frame.
        """
        kpts, _ = self._detect_keypoints(gray)
        if len(self.keyframes) == 0:
            self.keyframes.append({"kpts": kpts, "pose": SE3(), "gray": gray.copy()})
            return SE3()
        last = self.keyframes[-1]
        # Simplified matching by nearest 2D distance (replace with real descriptors)
        # For demo we assume small motion and match by proximity
        matches = []
        for i, p1 in enumerate(kpts):
            dists = np.linalg.norm(last["kpts"] - p1, axis=1)
            best = int(np.argmin(dists))
            if dists[best] < 20:
                matches.append((i, best))
        if len(matches) < 8:
            # Insufficient matches; insert keyframe
            self.keyframes.append({"kpts": kpts, "pose": self.current_pose, "gray": gray.copy()})
            return self.current_pose
        pts_cur = kpts[[m[0] for m in matches]]
        pts_last = last["kpts"[[m[1] for m in matches]]]
        T_rel = self._estimate_pose_8point(pts_cur, pts_last)
        self.current_pose = last["pose"] * T_rel.inverse()
        # Keyframe criterion: motion magnitude
        motion = np.linalg.norm(T_rel.t)
        if motion > 0.1 or len(matches) < 50:
            self.keyframes.append({"kpts": kpts, "pose": self.current_pose, "gray": gray.copy()})
        return self.current_pose

    def get_trajectory(self) -> FloatArray:
        """Camera centres of all keyframes."""
        return np.array([kf["pose"].t for kf in self.keyframes], dtype=np.float64)
