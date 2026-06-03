"""Sim-Gate Layer 1: kinematic joint-limit check (Unitree Go2).

Cheapest deterministic gate layer. Validates a joint-space trajectory against
the Go2 URDF position limits and velocity limit using pure numpy — no GPU, no
simulator, <1 ms per plan (design budget, enforced by test).

Joint limits source: Unitree go2_description URDF (unitree_ros), 12 actuated
joints in Isaac Lab ordering: [FL, FR, RL, RR] x [hip, thigh, calf].

Design reference: docs/design/architecture.md section 5 (validation gate).
"""

from __future__ import annotations

import numpy as np

from physgate.gate.schemas import (
    FailureCode,
    FailureReport,
    GateLayer,
    Violation,
)

#: Go2 actuated joints, [leg][joint] order: FL, FR, RL, RR x hip, thigh, calf.
GO2_JOINT_NAMES: tuple[str, ...] = (
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
)

# Per-joint (lower, upper) position limits in radians, from the Go2 URDF.
_HIP_LIMITS = (-1.0472, 1.0472)
_FRONT_THIGH_LIMITS = (-1.5708, 3.4907)
_REAR_THIGH_LIMITS = (-0.5236, 4.5379)
_CALF_LIMITS = (-2.7227, -0.83776)

GO2_POSITION_LIMITS: np.ndarray = np.array(
    [
        _HIP_LIMITS, _FRONT_THIGH_LIMITS, _CALF_LIMITS,   # FL
        _HIP_LIMITS, _FRONT_THIGH_LIMITS, _CALF_LIMITS,   # FR
        _HIP_LIMITS, _REAR_THIGH_LIMITS, _CALF_LIMITS,    # RL
        _HIP_LIMITS, _REAR_THIGH_LIMITS, _CALF_LIMITS,    # RR
    ]
)

#: Go2 joint velocity limit (rad/s), uniform across joints per URDF.
GO2_VELOCITY_LIMIT: float = 30.1


def _malformed(detail: str) -> FailureReport:
    """Fail-closed report for trajectories we cannot even check."""
    return FailureReport(
        failure_code=FailureCode.JOINT_LIMIT,
        layer=GateLayer.KINEMATIC_LIMIT,
        violations=[Violation(type="malformed_trajectory", detail=detail)],
        remediation_hint="emit a (T, 12) finite joint trajectory in Go2 joint order",
        retryable=False,
    )


def check_joint_trajectory(
    trajectory: np.ndarray,
    dt: float | None = None,
    step_id: int | None = None,
) -> FailureReport | None:
    """Check a Go2 joint trajectory against position (and optional velocity) limits.

    Args:
        trajectory: joint positions in radians, shape ``(T, 12)`` or ``(12,)``
            for a single pose, in :data:`GO2_JOINT_NAMES` order.
        dt: timestep in seconds between trajectory rows. If given and the
            trajectory has >=2 rows, finite-difference velocities are checked
            against :data:`GO2_VELOCITY_LIMIT`.
        step_id: plan step this trajectory belongs to (for the failure report).

    Returns:
        ``None`` if every sample is within limits (PASS), else a
        :class:`FailureReport` listing every violation (DENIED).
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim == 1:
        traj = traj[None, :]

    if traj.ndim != 2 or traj.shape[1] != len(GO2_JOINT_NAMES):
        return _malformed(
            f"expected shape (T, {len(GO2_JOINT_NAMES)}), got {np.asarray(trajectory).shape}"
        )
    if not np.all(np.isfinite(traj)):
        return _malformed("trajectory contains NaN or Inf values")

    violations: list[Violation] = []

    # Position limits — vectorized over all (timestep, joint) pairs.
    lower, upper = GO2_POSITION_LIMITS[:, 0], GO2_POSITION_LIMITS[:, 1]
    out_of_bounds = (traj < lower) | (traj > upper)
    for t_idx, j_idx in zip(*np.nonzero(out_of_bounds)):
        violations.append(
            Violation(
                type="position_limit",
                detail=(
                    f"{GO2_JOINT_NAMES[j_idx]} = {traj[t_idx, j_idx]:.4f} rad at step {t_idx} "
                    f"outside [{lower[j_idx]:.4f}, {upper[j_idx]:.4f}]"
                ),
            )
        )

    # Velocity limits — finite difference between consecutive rows.
    if dt is not None and dt > 0 and traj.shape[0] >= 2:
        velocities = np.abs(np.diff(traj, axis=0)) / dt
        too_fast = velocities > GO2_VELOCITY_LIMIT
        for t_idx, j_idx in zip(*np.nonzero(too_fast)):
            violations.append(
                Violation(
                    type="velocity_limit",
                    detail=(
                        f"{GO2_JOINT_NAMES[j_idx]} velocity {velocities[t_idx, j_idx]:.2f} rad/s "
                        f"between steps {t_idx}->{t_idx + 1} exceeds {GO2_VELOCITY_LIMIT} rad/s"
                    ),
                )
            )

    if not violations:
        return None

    return FailureReport(
        failed_step_id=step_id,
        failure_code=FailureCode.JOINT_LIMIT,
        layer=GateLayer.KINEMATIC_LIMIT,
        violations=violations,
        remediation_hint="re-plan with joint targets inside Go2 URDF limits",
        retryable=True,
    )
