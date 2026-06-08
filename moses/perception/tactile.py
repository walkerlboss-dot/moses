"""
moses/perception/tactile.py
Tactile Sensing Module — Moses v6.0 Perception Stack

Provides tactile sensor models, tactile image processing, slip detection,
texture classification, and grasp stability estimation.

References:
- Lee & Nicholls, "Tactile sensing for mechatronics", IEEE Trans. Mechatronics 1999.
- Li et al., "Slip detection with combined tactile and visual information",
  ICRA 2018.
- Yuan et al., "GelSight: High-Resolution Robot Tactile Sensors", RSS 2017.
- Romero et al., "Human-to-robot mapping of grasp synergies", T-RO 2014.
- Lepora et al., "Exploratory tactile servoing with active touch", RAM 2017.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from collections import deque

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# 1. Tactile Sensor Models
# ---------------------------------------------------------------------------

@dataclass
class TactileReading:
    """Raw output from a tactile sensor array."""
    pressure: FloatArray       # (H, W) normal force
    shear_x: Optional[FloatArray] = None   # (H, W) x-shear
    shear_y: Optional[FloatArray] = None   # (H, W) y-shear
    vibration: Optional[FloatArray] = None  # (H, W) high-freq AC component
    timestamp: float = 0.0
    temperature: Optional[FloatArray] = None


class GelSightModel:
    """
    Physics-based model of GelSight-style optical tactile sensor.
    (Yuan et al., RSS 2017; Donlon et al., "GelSight Wedge", ICRA 2018)

    Elastomer deformation under contact is approximated via
    linear elasticity (Hooke's law) plus photometric stereo for
    surface normal recovery from RGB gel image.
    """

    def __init__(self, grid_shape: Tuple[int, int] = (240, 320), elastomer_thickness: float = 0.004):
        self.H, self.W = grid_shape
        self.thickness = elastomer_thickness  # 4 mm typical
        self.E = 3e5  # Young's modulus, Pa (soft silicone)
        self.nu = 0.48  # Poisson ratio (near incompressible)

    def photometric_stereo(self, images: List[FloatArray], light_dirs: FloatArray) -> FloatArray:
        """
        Recover surface normals from N images under known lighting.
        Woodham, "Photometric method for determining surface orientation
        from multiple images", Optical Engineering 1980.

        Args:
            images: list of (H, W) grayscale under different lights
            light_dirs: (N, 3) normalised light directions
        Returns:
            normals: (H, W, 3)
        """
        N = len(images)
        I = np.stack(images, axis=-1)  # (H, W, N)
        # Solve L · n = I  (least squares)
        L = light_dirs  # (N, 3)
        LtL_inv = np.linalg.inv(L.T @ L)
        n = I @ L @ LtL_inv  # (H, W, 3)
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        return n / norm

    def depth_from_normals(self, normals: FloatArray) -> FloatArray:
        """
        Integrate normal field to height map via Frankot-Chellappa
        integration (Frankot & Chellappa, IEEE Trans. PAMI 1988).
        """
        nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
        # Avoid division by zero
        p = -nx / np.where(nz == 0, 1e-6, nz)
        q = -ny / np.where(nz == 0, 1e-6, nz)
        # FFT-based integration
        P = np.fft.fft2(p)
        Q = np.fft.fft2(q)
        H, W = p.shape
        u = (np.arange(W) - W // 2) * 2 * np.pi / W
        v = (np.arange(H) - H // 2) * 2 * np.pi / H
        U, V = np.meshgrid(u, v)
        denom = (1j * U) ** 2 + (1j * V) ** 2
        denom[denom == 0] = 1e-12
        Z_fft = (1j * U * P + 1j * V * Q) / denom
        Z = np.fft.ifft2(np.fft.fftshift(Z_fft)).real
        return Z

    def simulate_contact(self, object_shape: FloatArray, force: float) -> TactileReading:
        """
        Simulate elastomer deformation given object height map and total force.
        Uses Boussinesq-Cerruti point-load solution for half-space.
        (Johnson, "Contact Mechanics", Cambridge 1985)
        """
        # Simplified: displacement proportional to pressure
        pressure = force * np.exp(-((np.arange(self.H)[:, None] - self.H / 2) ** 2 +
                                    (np.arange(self.W)[None, :] - self.W / 2) ** 2) / (2 * (self.H / 8) ** 2))
        pressure = np.clip(pressure, 0, None)
        return TactileReading(pressure=pressure)


class BioTacModel:
    """
    Syntouch BioTac-style impedance sensor model.
    (Wettels et al., "Multi-Modal Synergy", ICRA 2009)
    """

    def __init__(self, n_electrodes: int = 19):
        self.n_electrodes = n_electrodes
        self.impedance_map = np.zeros(n_electrodes, dtype=np.float64)

    def update(self, deformation: FloatArray, fluid_pressure: float) -> TactileReading:
        """
        Map local deformation to electrode impedance changes.
        """
        # Linear sensitivity model
        sensitivity = np.random.rand(self.n_electrodes) * 0.1 + 0.9
        self.impedance_map += sensitivity * np.linalg.norm(deformation) - 0.001 * self.impedance_map
        pressure_grid = self.impedance_map[:16].reshape(4, 4)  # simplified mapping
        return TactileReading(pressure=pressure_grid, vibration=np.zeros_like(pressure_grid))


# ---------------------------------------------------------------------------
# 2. Tactile Image Processing
# ---------------------------------------------------------------------------

class TactileImageProcessor:
    """
    Process raw tactile arrays: filtering, contact segmentation,
    force distribution extraction.
    """

    def __init__(self, grid_shape: Tuple[int, int] = (240, 320)):
        self.H, self.W = grid_shape

    def gaussian_filter(self, img: FloatArray, sigma: float = 1.0) -> FloatArray:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(img, sigma=sigma)

    def contact_mask(self, pressure: FloatArray, threshold: float = 0.05) -> FloatArray:
        """Binary mask of contact region."""
        return (pressure > threshold * pressure.max()).astype(np.float64)

    def force_distribution(self, reading: TactileReading) -> Dict[str, float]:
        """
        Compute aggregated force metrics.
        """
        p = reading.pressure
        total_force = float(p.sum())
        max_pressure = float(p.max())
        centroid = np.array([np.average(np.arange(p.shape[0]), weights=p.mean(axis=1)),
                             np.average(np.arange(p.shape[1]), weights=p.mean(axis=0))])
        # Moment of pressure distribution (spread)
        yy, xx = np.mgrid[:p.shape[0], :p.shape[1]]
        spread = float(np.sqrt(np.average((yy - centroid[0]) ** 2 + (xx - centroid[1]) ** 2, weights=p)))
        return {
            "total_force": total_force,
            "max_pressure": max_pressure,
            "centroid": centroid,
            "spread": spread,
        }

    def shear_magnitude(self, reading: TactileReading) -> FloatArray:
        if reading.shear_x is None or reading.shear_y is None:
            return np.zeros_like(reading.pressure)
        return np.sqrt(reading.shear_x ** 2 + reading.shear_y ** 2)


# ---------------------------------------------------------------------------
# 3. Slip Detection
# ---------------------------------------------------------------------------

class SlipDetector:
    """
    Detect incipient slip from tactile and vibration cues.
    Based on Li et al., ICRA 2018; Tremblay & Cutkosky, ICRA 1993.
    """

    def __init__(self, history_len: int = 30, fs: float = 1000.0):
        self.history: deque[TactileReading] = deque(maxlen=history_len)
        self.fs = fs  # sampling rate Hz
        self.slip_threshold = 0.3
        self.vib_threshold = 2.0

    def update(self, reading: TactileReading) -> Dict[str, float]:
        self.history.append(reading)
        if len(self.history) < 3:
            return {"slip_probability": 0.0, "vib_energy": 0.0, "shear_rate": 0.0}

        # Tactile flow: temporal derivative of pressure centroid
        proc = TactileImageProcessor()
        dists = [proc.force_distribution(r) for r in self.history]
        centroids = np.stack([d["centroid"] for d in dists])
        shear_rate = float(np.linalg.norm(np.diff(centroids, axis=0).mean(axis=0))) * self.fs

        # Vibration energy in high-frequency band
        vib_energy = 0.0
        if reading.vibration is not None:
            vib = reading.vibration.flatten()
            # Welch-style PSD estimate (simplified)
            vib_energy = float(np.var(vib))

        # Slip likelihood via logistic fusion
        z = 0.5 * (shear_rate / 50.0) + 0.5 * (vib_energy / self.vib_threshold) - 1.0
        slip_prob = 1.0 / (1.0 + np.exp(-z))

        return {
            "slip_probability": float(np.clip(slip_prob, 0.0, 1.0)),
            "vib_energy": vib_energy,
            "shear_rate": shear_rate,
        }


# ---------------------------------------------------------------------------
# 4. Texture Classification
# ---------------------------------------------------------------------------

class TextureClassifier:
    """
    Classify surface texture from tactile images using Local Binary Patterns
    (Ojala et al., IEEE Trans. PAMI 2002) + SVM.
    """

    def __init__(self, n_classes: int = 5, p: int = 8, r: int = 1):
        self.n_classes = n_classes
        self.P = p
        self.R = r
        # Simplified linear classifier weights (random init; train on data)
        self.W = np.random.randn(256, n_classes).astype(np.float64) * 0.01
        self.b = np.zeros(n_classes, dtype=np.float64)

    def _lbp(self, img: FloatArray) -> FloatArray:
        """Uniform LBP histogram."""
        h, w = img.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        for i, (dy, dx) in enumerate([
            (-1, -1), (-1, 0), (-1, 1), (0, 1),
            (1, 1), (1, 0), (1, -1), (0, -1)
        ]):
            shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            lbp += ((shifted > img).astype(np.uint8) << i)
        # Uniform patterns only (reduce to 59 bins); here simplified to 256
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256), density=True)
        return hist.astype(np.float64)

    def extract_features(self, reading: TactileReading) -> FloatArray:
        """LBP histogram over pressure image."""
        return self._lbp(reading.pressure)

    def predict(self, reading: TactileReading) -> Tuple[int, FloatArray]:
        feat = self.extract_features(reading)
        logits = feat @ self.W + self.b
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return int(np.argmax(probs)), probs


# ---------------------------------------------------------------------------
# 5. Grasp Stability from Tactile Feedback
# ---------------------------------------------------------------------------

class GraspStabilityEstimator:
    """
    Estimate grasp stability from distributed tactile readings.
    Based on contact wrench space analysis (Li & Sastry, 1988)
    and learning-based stability (Calandra et al., ICRA 2015).
    """

    def __init__(self, n_fingers: int = 2):
        self.n_fingers = n_fingers
        self.slip_detectors = [SlipDetector() for _ in range(n_fingers)]

    def compute_wrench(self, readings: List[TactileReading], contact_normals: List[FloatArray]) -> FloatArray:
        """
        Compute resultant wrench (force + torque) from contact patches.
        Assumes each reading corresponds to one finger/contact patch.
        """
        force = np.zeros(3, dtype=np.float64)
        torque = np.zeros(3, dtype=np.float64)
        for r, n in zip(readings, contact_normals):
            proc = TactileImageProcessor()
            f_mag = proc.force_distribution(r)["total_force"]
            f_vec = f_mag * n / np.linalg.norm(n)
            force += f_vec
            # Simplified torque about origin; real system needs contact positions
            torque += np.cross(np.zeros(3), f_vec)
        return np.concatenate([force, torque])

    def stability_score(self, readings: List[TactileReading]) -> float:
        """
        Scalar stability metric in [0, 1].
        Combines: total normal force, symmetry, absence of slip.
        """
        if len(readings) != self.n_fingers:
            raise ValueError("Expected one reading per finger")
        total_force = sum(r.pressure.sum() for r in readings)
        # Symmetry: variance of forces across fingers
        forces = np.array([r.pressure.sum() for r in readings])
        symmetry = 1.0 - np.std(forces) / (np.mean(forces) + 1e-6)
        # Slip
        slip_probs = [det.update(r)["slip_probability"] for det, r in zip(self.slip_detectors, readings)]
        max_slip = max(slip_probs)
        score = 0.4 * np.tanh(total_force / 10.0) + 0.3 * symmetry + 0.3 * (1.0 - max_slip)
        return float(np.clip(score, 0.0, 1.0))
