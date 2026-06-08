"""
test_rewards.py — Unit tests for humanoid reward functions.

Verifies:
  - Velocity tracking reward shape and range
  - Energy penalty sign and magnitude
  - Stability bonus activation conditions
  - Termination condition triggers
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Velocity Tracking Reward
# ---------------------------------------------------------------------------

class TestVelocityTracking:
    """Tests for velocity tracking reward."""

    def test_perfect_tracking(self, reward_functions):
        """
        When desired and actual velocities match exactly, reward should be 1.0.
        """
        desired = np.array([1.0, 0.0, 0.5])
        actual = np.array([1.0, 0.0, 0.5])
        r = reward_functions.velocity_tracking(desired, actual, sigma=0.5)
        assert np.isclose(r, 1.0, atol=1e-6)

    def test_zero_tracking_large_error(self, reward_functions):
        """
        With a very large velocity error, the Gaussian reward should approach 0.
        """
        desired = np.array([0.0, 0.0, 0.0])
        actual = np.array([10.0, 10.0, 10.0])
        r = reward_functions.velocity_tracking(desired, actual, sigma=0.5)
        assert r < 1e-6

    def test_reward_range(self, reward_functions):
        """
        Velocity tracking reward must always be in [0, 1].
        """
        for _ in range(50):
            desired = np.random.randn(3) * 2.0
            actual = np.random.randn(3) * 2.0
            r = reward_functions.velocity_tracking(desired, actual, sigma=0.5)
            assert 0.0 <= r <= 1.0, f"Reward {r} out of bounds"

    def test_sigma_scaling(self, reward_functions):
        """
        Larger sigma should yield higher reward for the same error.
        """
        desired = np.array([1.0, 0.0, 0.0])
        actual = np.array([0.5, 0.0, 0.0])
        r_narrow = reward_functions.velocity_tracking(desired, actual, sigma=0.1)
        r_wide = reward_functions.velocity_tracking(desired, actual, sigma=1.0)
        assert r_wide > r_narrow


# ---------------------------------------------------------------------------
# Energy Penalty
# ---------------------------------------------------------------------------

class TestEnergyPenalty:
    """Tests for energy consumption penalty."""

    def test_zero_penalty_at_rest(self, reward_functions):
        """
        When torques and velocities are zero, the penalty must be zero.
        """
        torques = np.zeros(12)
        velocities = np.zeros(12)
        p = reward_functions.energy_penalty(torques, velocities)
        assert np.isclose(p, 0.0, atol=1e-12)

    def test_negative_penalty(self, reward_functions):
        """
        Energy penalty must always be non-positive (it subtracts from total reward).
        """
        torques = np.ones(12)
        velocities = np.ones(12)
        p = reward_functions.energy_penalty(torques, velocities)
        assert p <= 0.0

    def test_penalty_scales_with_power(self, reward_functions):
        """
        Doubling both torque and velocity should quadruple the absolute penalty.
        """
        torques = np.ones(12)
        velocities = np.ones(12)
        p1 = abs(reward_functions.energy_penalty(torques, velocities))
        p2 = abs(reward_functions.energy_penalty(torques * 2, velocities * 2))
        assert np.isclose(p2, 4.0 * p1, rtol=1e-6)

    def test_penalty_independence(self, reward_functions):
        """
        Penalty should be the sum of independent joint contributions.
        """
        torques = np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        velocities = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        p = reward_functions.energy_penalty(torques, velocities)
        expected = -0.01 * (1.0 * 1.0 + 2.0 * 1.0)
        assert np.isclose(p, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Stability Bonus
# ---------------------------------------------------------------------------

class TestStabilityBonus:
    """Tests for torso height stability bonus."""

    def test_full_bonus_at_target(self, reward_functions):
        """
        When base_z exactly equals target_z, the full bonus of 1.0 is awarded.
        """
        r = reward_functions.stability_bonus(base_z=0.85, target_z=0.85, tolerance=0.05)
        assert np.isclose(r, 1.0, atol=1e-12)

    def test_full_bonus_within_tolerance(self, reward_functions):
        """
        Base z within the tolerance band should still yield full bonus.
        """
        r = reward_functions.stability_bonus(base_z=0.87, target_z=0.85, tolerance=0.05)
        assert np.isclose(r, 1.0, atol=1e-12)

    def test_zero_bonus_outside_tolerance(self, reward_functions):
        """
        Base z outside the tolerance band yields zero bonus.
        """
        r = reward_functions.stability_bonus(base_z=0.5, target_z=0.85, tolerance=0.05)
        assert np.isclose(r, 0.0, atol=1e-12)

    def test_bonus_boundary(self, reward_functions):
        """
        Exactly at tolerance boundary should still give full bonus.
        """
        r = reward_functions.stability_bonus(base_z=0.80, target_z=0.85, tolerance=0.05)
        assert np.isclose(r, 1.0, atol=1e-12)

    def test_bonus_just_outside(self, reward_functions):
        """
        Just beyond tolerance boundary should give zero bonus.
        """
        r = reward_functions.stability_bonus(base_z=0.799, target_z=0.85, tolerance=0.05)
        assert np.isclose(r, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Termination Conditions
# ---------------------------------------------------------------------------

class TestTerminationConditions:
    """Tests for episode termination logic."""

    def test_terminate_on_fall(self, reward_functions):
        """
        When base_z drops below min_height, episode must terminate with reason 'fallen'.
        """
        terminated, reason = reward_functions.termination_conditions(
            base_z=0.1,
            base_orientation=np.array([0.0, 0.0, 0.0]),
            min_height=0.3,
        )
        assert terminated is True
        assert reason == "fallen"

    def test_terminate_on_excessive_tilt(self, reward_functions):
        """
        When roll/pitch exceeds max_tilt_rad, episode must terminate with reason 'excessive_tilt'.
        """
        terminated, reason = reward_functions.termination_conditions(
            base_z=0.85,
            base_orientation=np.array([1.5, 0.0, 0.0]),  # ~85° roll
            max_tilt_rad=np.pi / 3,
        )
        assert terminated is True
        assert reason == "excessive_tilt"

    def test_no_termination_normal(self, reward_functions):
        """
        Normal upright standing should not trigger termination.
        """
        terminated, reason = reward_functions.termination_conditions(
            base_z=0.85,
            base_orientation=np.array([0.05, 0.05, 0.0]),
        )
        assert terminated is False
        assert reason == ""

    def test_terminate_priority_fall_over_tilt(self, reward_functions):
        """
        If both fall and tilt conditions are violated, 'fallen' should be reported
        (or at least a valid termination reason).
        """
        terminated, reason = reward_functions.termination_conditions(
            base_z=0.1,
            base_orientation=np.array([1.5, 0.0, 0.0]),
            min_height=0.3,
            max_tilt_rad=np.pi / 3,
        )
        assert terminated is True
        assert reason in ("fallen", "excessive_tilt")

    def test_terminate_with_quaternion(self, reward_functions):
        """
        Termination should work when orientation is passed as a quaternion (size >= 4).
        The first two elements are interpreted as roll/pitch proxy.
        """
        terminated, reason = reward_functions.termination_conditions(
            base_z=0.85,
            base_orientation=np.array([0.0, 1.5, 0.0, 1.0]),
            max_tilt_rad=np.pi / 3,
        )
        assert terminated is True
        assert reason == "excessive_tilt"
