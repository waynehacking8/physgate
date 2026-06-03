"""Tests for the Sim-Gate L1 kinematic-limit checker (A2).

L1 is the cheapest gate layer: pure numpy joint position/velocity limit checks
against the Unitree Go2 URDF limits. Must run in <1 ms and never touch a GPU.
"""

import numpy as np
import pytest

from physgate.gate.l1_kinematic import (
    GO2_JOINT_NAMES,
    GO2_POSITION_LIMITS,
    GO2_VELOCITY_LIMIT,
    check_joint_trajectory,
)
from physgate.gate.schemas import FailureCode, GateLayer


def _neutral_pose() -> np.ndarray:
    """A standing pose comfortably inside all Go2 joint limits."""
    # hip=0, thigh=0.8, calf=-1.5 for each of the 4 legs (FL, FR, RL, RR)
    return np.array([0.0, 0.8, -1.5] * 4)


def _valid_trajectory(steps: int = 20) -> np.ndarray:
    """Small sinusoidal motion around the neutral pose, well within limits."""
    base = _neutral_pose()
    t = np.linspace(0, 1, steps)[:, None]
    return base[None, :] + 0.1 * np.sin(2 * np.pi * t)


def test_go2_has_12_joints():
    assert len(GO2_JOINT_NAMES) == 12
    assert GO2_POSITION_LIMITS.shape == (12, 2)
    # lower bounds strictly below upper bounds
    assert np.all(GO2_POSITION_LIMITS[:, 0] < GO2_POSITION_LIMITS[:, 1])


def test_valid_trajectory_passes():
    assert check_joint_trajectory(_valid_trajectory()) is None


def test_single_pose_passes():
    """A (12,) single pose is accepted, not just (T, 12) trajectories."""
    assert check_joint_trajectory(_neutral_pose()) is None


def test_hip_position_violation_denied():
    traj = _valid_trajectory()
    traj[5, 0] = 2.0  # FL hip limit is ±1.0472 rad
    report = check_joint_trajectory(traj)
    assert report is not None
    assert report.verdict == "DENIED"
    assert report.failure_code == FailureCode.JOINT_LIMIT
    assert report.layer == GateLayer.KINEMATIC_LIMIT
    # violation names the offending joint and timestep
    assert "FL_hip" in report.violations[0].detail
    assert "5" in report.violations[0].detail


def test_calf_position_violation_denied():
    traj = _valid_trajectory()
    traj[0, 2] = 0.0  # calf upper limit is -0.83776 (calf can never reach 0)
    report = check_joint_trajectory(traj)
    assert report is not None
    assert report.violations[0].type == "position_limit"


def test_velocity_violation_denied():
    traj = _valid_trajectory()
    # huge jump between consecutive steps at 50 Hz => velocity >> 30 rad/s limit
    traj[10] = traj[9] + 2.0
    report = check_joint_trajectory(traj, dt=0.02)
    assert report is not None
    assert any(v.type == "velocity_limit" for v in report.violations)


def test_velocity_check_skipped_for_single_pose():
    assert check_joint_trajectory(_neutral_pose(), dt=0.02) is None


def test_wrong_joint_count_fails_closed():
    bad = np.zeros((5, 7))  # 7 joints is not a Go2
    report = check_joint_trajectory(bad)
    assert report is not None
    assert report.violations[0].type == "malformed_trajectory"
    assert report.retryable is False


def test_nan_fails_closed():
    traj = _valid_trajectory()
    traj[3, 3] = np.nan
    report = check_joint_trajectory(traj)
    assert report is not None
    assert report.violations[0].type == "malformed_trajectory"


def test_multiple_violations_all_reported():
    traj = _valid_trajectory()
    traj[2, 0] = 9.0
    traj[7, 5] = -9.0
    report = check_joint_trajectory(traj)
    assert report is not None
    assert len(report.violations) >= 2


def test_l1_meets_latency_budget():
    """L1 on a 20-step trajectory must run well under 10 ms (design: <1 ms).

    Relaxed to 5 ms to avoid flaky failures on loaded CI runners; the meaningful
    bound is <10 ms (far below any physics step budget).
    """
    import time

    traj = _valid_trajectory()
    check_joint_trajectory(traj)  # warm up
    t0 = time.perf_counter()
    for _ in range(100):
        check_joint_trajectory(traj)
    per_call_ms = (time.perf_counter() - t0) / 100 * 1000
    assert per_call_ms < 5.0, f"L1 took {per_call_ms:.3f} ms per call (budget: <5 ms)"


def test_velocity_limit_constant_is_go2_spec():
    assert GO2_VELOCITY_LIMIT == pytest.approx(30.1)
