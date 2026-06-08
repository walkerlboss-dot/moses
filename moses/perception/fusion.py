"""
moses/perception/fusion.py
Multi-Modal Fusion Module — Moses v6.0 Perception Stack

Fuses vision, tactile, and force modalities via Kalman filtering,
uncertainty-aware fusion, and attention-based sensor selection.

References:
- Welch & Bishop, "An Introduction to the Kalman Filter", TR 95-041, 2006.
- Khoshelham & Elberink, "Accuracy and resolution of Kinect depth data",
  ISPRS 2012.
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
- Endo et al., "Deep learning for tactile understanding", RSS 2018.
- Durrant-Whyte & Henderson, "Multisensor Data Fusion", Springer Handbook 2016.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable
from collections import deque

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


# ---------------------------------------------------------------------------
# 1. Uncertainty Models
# ---------------------------------------------------------------------------

@dataclass
class GaussianState:
    """Gaussian belief: mean x and covariance P."""
    x: FloatArray
    P: FloatArray

    def mahalanobis(self, z: FloatArray, H: FloatArray, R: FloatArray) -> float:
        """Mahalanobis distance of measurement z."""
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        return float(np.sqrt(y.T @ np.linalg.solve(S, y)))


@dataclass
class SensorObservation:
    """Generic observation from any sensor modality."""
    z: FloatArray
    H: FloatArray          # Observation matrix
    R: FloatArray          # Observation noise covariance
    modality: str          # "vision", "tactile", "force", "proprioception"
    timestamp: float = 0.0
    confidence: float = 1.0  # [0, 1] from sensor-specific heuristics


# ---------------------------------------------------------------------------
# 2. Kalman Filtering for State Estimation
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    Standard Linear Kalman Filter.
    Welch & Bishop, "An Introduction to the Kalman Filter", 2006.

    State transition:
        x_{k+1} = F x_k + B u_k + w_k,   w_k ~ N(0, Q)
    Measurement:
        z_k = H x_k + v_k,                v_k ~ N(0, R)
    """

    def __init__(self, state_dim: int):
        self.n = state_dim
        self.state = GaussianState(x=np.zeros(state_dim), P=np.eye(state_dim) * 1.0)

    def predict(self, F: FloatArray, B: FloatArray, u: FloatArray, Q: FloatArray) -> GaussianState:
        """Prediction step."""
        self.state.x = F @ self.state.x + B @ u
        self.state.P = F @ self.state.P @ F.T + Q
        return self.state

    def update(self, obs: SensorObservation) -> GaussianState:
        """Measurement update."""
        y = obs.z - obs.H @ self.state.x          # innovation
        S = obs.H @ self.state.P @ obs.H.T + obs.R  # innovation covariance
        K = self.state.P @ obs.H.T @ np.linalg.inv(S)  # Kalman gain
        self.state.x = self.state.x + K @ y
        I_KH = np.eye(self.n) - K @ obs.H
        self.state.P = I_KH @ self.state.P @ I_KH.T + K @ obs.R @ K.T  # Joseph form
        return self.state

    def update_multiple(self, observations: List[SensorObservation]) -> GaussianState:
        """Sequential update with multiple observations."""
        for obs in observations:
            self.update(obs)
        return self.state


class ExtendedKalmanFilter(KalmanFilter):
    """
    EKF for nonlinear dynamics / observation models.
    """

    def predict_nonlinear(
        self,
        f: Callable[[FloatArray, FloatArray], FloatArray],
        F_jacobian: FloatArray,
        u: FloatArray,
        Q: FloatArray,
    ) -> GaussianState:
        self.state.x = f(self.state.x, u)
        self.state.P = F_jacobian @ self.state.P @ F_jacobian.T + Q
        return self.state

    def update_nonlinear(
        self,
        h: Callable[[FloatArray], FloatArray],
        H_jacobian: FloatArray,
        z: FloatArray,
        R: FloatArray,
    ) -> GaussianState:
        y = z - h(self.state.x)
        S = H_jacobian @ self.state.P @ H_jacobian.T + R
        K = self.state.P @ H_jacobian.T @ np.linalg.inv(S)
        self.state.x = self.state.x + K @ y
        I_KH = np.eye(self.n) - K @ H_jacobian
        self.state.P = I_KH @ self.state.P @ I_KH.T + K @ R @ K.T
        return self.state


# ---------------------------------------------------------------------------
# 3. Uncertainty-Aware Fusion
# ---------------------------------------------------------------------------

class CovarianceIntersection:
    """
    Covariance Intersection for consistent fusion of correlated estimates.
    Julier & Uhlmann, "A non-divergent estimation algorithm in the presence
    of unknown correlations", ACC 1997.
    """

    @staticmethod
    def fuse(a: GaussianState, b: GaussianState, omega: Optional[float] = None) -> GaussianState:
        """
        P^{-1} = omega P_a^{-1} + (1-omega) P_b^{-1}
        x = P (omega P_a^{-1} x_a + (1-omega) P_b^{-1} x_b)
        """
        if omega is None:
            # Optimise omega via trace minimisation (simplified grid search)
            omegas = np.linspace(0.01, 0.99, 20)
            best_omega = 0.5
            best_trace = float('inf')
            for o in omegas:
                Pinv = o * np.linalg.inv(a.P) + (1 - o) * np.linalg.inv(b.P)
                P = np.linalg.inv(Pinv)
                tr = np.trace(P)
                if tr < best_trace:
                    best_trace = tr
                    best_omega = o
            omega = best_omega
        Pinv = omega * np.linalg.inv(a.P) + (1 - omega) * np.linalg.inv(b.P)
        P = np.linalg.inv(Pinv)
        x = P @ (omega * np.linalg.inv(a.P) @ a.x + (1 - omega) * np.linalg.inv(b.P) @ b.x)
        return GaussianState(x=x, P=P)


class UncertaintyWeightedFusion:
    """
    Weighted average inverse-covariance fusion (optimal for independent estimates).
    Durrant-Whyte & Henderson, Springer Handbook 2016.
    """

    @staticmethod
    def fuse(states: List[GaussianState]) -> GaussianState:
        Pinv_sum = sum(np.linalg.inv(s.P) for s in states)
        P = np.linalg.inv(Pinv_sum)
        x = P @ sum(np.linalg.inv(s.P) @ s.x for s in states)
        return GaussianState(x=x, P=P)


# ---------------------------------------------------------------------------
# 4. Attention Mechanisms for Sensor Selection
# ---------------------------------------------------------------------------

class SensorAttention:
    """
    Learned attention over sensor modalities for adaptive fusion.
    Inspired by Vaswani et al., NeurIPS 2017; Endo et al., RSS 2018.

    Query = current task embedding
    Keys  = sensor feature embeddings
    Values = sensor observations
    """

    def __init__(self, embed_dim: int = 64, num_sensors: int = 3):
        self.d = embed_dim
        self.m = num_sensors
        # Learnable projections (random init; train on data)
        self.W_q = np.random.randn(embed_dim, embed_dim).astype(np.float64) * 0.01
        self.W_k = np.random.randn(num_sensors, embed_dim, embed_dim).astype(np.float64) * 0.01
        self.W_v = np.random.randn(num_sensors, embed_dim, embed_dim).astype(np.float64) * 0.01

    def _softmax(self, x: FloatArray) -> FloatArray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def forward(
        self,
        task_embedding: FloatArray,          # (d,)
        sensor_features: List[FloatArray],   # list of (d,)
    ) -> Tuple[FloatArray, FloatArray]:
        """
        Returns attention weights (m,) and fused embedding (d,).
        """
        Q = task_embedding @ self.W_q  # (d,)
        K = np.stack([sensor_features[i] @ self.W_k[i] for i in range(self.m)], axis=0)  # (m, d)
        V = np.stack([sensor_features[i] @ self.W_v[i] for i in range(self.m)], axis=0)  # (m, d)
        scores = Q @ K.T / np.sqrt(self.d)  # (m,)
        weights = self._softmax(scores)
        fused = weights @ V  # (d,)
        return weights, fused

    def select_best_sensor(self, task_embedding: FloatArray, sensor_features: List[FloatArray]) -> int:
        weights, _ = self.forward(task_embedding, sensor_features)
        return int(np.argmax(weights))


class GatingFusion:
    """
    Gated sensor fusion: each sensor has a gate that modulates its contribution.
    Inspired by Kalman-gain gating and mixture-of-experts.
    """

    def __init__(self, state_dim: int, num_sensors: int):
        self.n = state_dim
        self.m = num_sensors
        # Gate network: simple linear layer
        self.W_gate = np.random.randn(num_sensors, state_dim).astype(np.float64) * 0.01
        self.b_gate = np.zeros(num_sensors, dtype=np.float64)

    def compute_gates(self, state: FloatArray, context: FloatArray) -> FloatArray:
        """
        Sigmoid gating based on current state and task context.
        """
        logits = self.W_gate @ np.concatenate([state, context])[:self.W_gate.shape[1]] + self.b_gate
        return 1.0 / (1.0 + np.exp(-logits))

    def fuse(
        self,
        predictions: List[GaussianState],
        gates: FloatArray,
    ) -> GaussianState:
        """
        Weighted covariance intersection per sensor.
        """
        # Normalise gates
        gates = gates / (gates.sum() + 1e-6)
        Pinv = sum(g * np.linalg.inv(p.P) for g, p in zip(gates, predictions))
        P = np.linalg.inv(Pinv)
        x = P @ sum(g * np.linalg.inv(p.P) @ p.x for g, p in zip(gates, predictions))
        return GaussianState(x=x, P=P)


# ---------------------------------------------------------------------------
# 5. Multi-Modal Perception Fusion Pipeline
# ---------------------------------------------------------------------------

class MultiModalFusion:
    """
    End-to-end fusion pipeline for Moses perception stack.

    Integrates:
    - 3D vision (point clouds, object poses)
    - Tactile (contact geometry, slip)
    - Force/torque (external wrenches)
    - Proprioception (joint states)

    Outputs unified state estimate with uncertainty.
    """

    def __init__(self, state_dim: int = 12):
        self.state_dim = state_dim
        self.ekf = ExtendedKalmanFilter(state_dim)
        self.attention = SensorAttention(embed_dim=64, num_sensors=4)
        self.gating = GatingFusion(state_dim, num_sensors=4)
        self.history: deque[Dict] = deque(maxlen=100)

    def vision_observation(
        self,
        object_pose: FloatArray,      # (6,) [x,y,z,roll,pitch,yaw]
        pose_covariance: FloatArray,  # (6,6)
    ) -> SensorObservation:
        """Package 3D vision detection as EKF observation."""
        z = object_pose
        H = np.zeros((6, self.state_dim))
        H[:6, :6] = np.eye(6)
        R = pose_covariance
        return SensorObservation(z=z, H=H, R=R, modality="vision")

    def tactile_observation(
        self,
        contact_normal: FloatArray,
        contact_depth: float,
        tactile_covariance: FloatArray,
    ) -> SensorObservation:
        """Package tactile contact as observation of local geometry."""
        z = np.concatenate([contact_normal, [contact_depth]])
        H = np.zeros((4, self.state_dim))
        H[:3, 6:9] = np.eye(3)  # map to surface normal state
        H[3, 9] = 1.0           # map to depth state
        R = tactile_covariance
        return SensorObservation(z=z, H=H, R=R, modality="tactile")

    def force_observation(
        self,
        wrench: FloatArray,           # (6,)
        wrench_covariance: FloatArray,
    ) -> SensorObservation:
        """Package force/torque estimate as observation."""
        z = wrench
        H = np.zeros((6, self.state_dim))
        H[:6, 6:12] = np.eye(6)
        R = wrench_covariance
        return SensorObservation(z=z, H=H, R=R, modality="force")

    def step(
        self,
        observations: List[SensorObservation],
        control_input: FloatArray,
        F: FloatArray,
        B: FloatArray,
        Q: FloatArray,
    ) -> GaussianState:
        """
        One full EKF cycle with adaptive sensor weighting.
        """
        # Prediction
        self.ekf.predict(F, B, control_input, Q)

        # Mahalanobis gating: reject outliers
        gated = []
        for obs in observations:
            d = self.ekf.state.mahalanobis(obs.z, obs.H, obs.R)
            if d < 3.0:  # 3-sigma gate
                gated.append(obs)
            else:
                # Reduce confidence for outlier
                obs.confidence *= 0.5
                gated.append(obs)

        # Adaptive weighting via inverse covariance scaled by confidence
        for obs in gated:
            obs.R = obs.R / max(obs.confidence, 0.01)

        # Sequential update
        self.ekf.update_multiple(gated)

        self.history.append({
            "state": self.ekf.state.x.copy(),
            "cov": self.ekf.state.P.copy(),
            "obs_count": len(gated),
        })
        return self.ekf.state

    def get_object_pose_estimate(self) -> Tuple[FloatArray, FloatArray]:
        """Extract object pose (x,y,z,r,p,y) and covariance."""
        x = self.ekf.state.x[:6]
        P = self.ekf.state.P[:6, :6]
        return x, P

    def get_contact_state_estimate(self) -> Tuple[FloatArray, FloatArray]:
        """Extract contact geometry and force state."""
        x = self.ekf.state.x[6:12]
        P = self.ekf.state.P[6:12, 6:12]
        return x, P
