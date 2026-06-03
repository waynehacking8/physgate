"""Go2 locomotion: trained rsl_rl policy + waypoint navigation (#1).

Replaces kinematic base driving with real walking:

* :class:`Go2PolicyController` — wraps the TorchScript policy exported by
  rsl_rl (``play.py`` → ``exported/policy.pt``). Reproduces the EXACT
  observation layout of ``Isaac-Velocity-Flat-Unitree-Go2-v0`` (48-dim, no
  height scan) and applies actions the same way (joint position targets,
  scale 0.25, default offset).
* :class:`WaypointNavigator` — high-level P-controller turning a goal position
  into velocity commands (vx, vy, wz) within the policy's training ranges.

The controller is batched: one call computes actions for all N envs at once.

IMPORTANT: import only after SimulationApp launch (uses isaaclab assets).
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from physgate.nav.path_planner import WAYPOINT_TRACKING_TOLERANCE

#: Go2 velocity-task action scale (rough_env_cfg override, inherited by flat).
ACTION_SCALE = 0.25
#: Policy control frequency: one policy step every N physics steps (decimation).
POLICY_DECIMATION = 4
#: Physics timestep the policy was trained with.
POLICY_PHYSICS_DT = 0.005
#: Velocity command limits (the policy saw commands in these ranges in training).
MAX_LIN_VEL = 1.0
MAX_ANG_VEL = 1.0
#: Forward speed commanded while turning toward a goal outside the heading
#: deadband. The trained policy tracks rotation poorly from a standstill —
#: pure rotation (vx=0, wz=0.9) yields only ~10% of the commanded turn (a
#: "keep standing" fixed point), while the same wz with vx=0.2 tracks ~95%
#: (benchmarks/rebuild/diag_turn_in_place.py). Never command pure rotation:
#: the resulting ~0.22 m turning arc stays inside the path planner's
#: WAYPOINT_TRACKING_TOLERANCE clearance inflation.
TURN_CREEP_SPEED = 0.2

#: Default location of the trained policy (latest run's export).
DEFAULT_POLICY_DIR = Path.home() / "IsaacLab" / "logs" / "rsl_rl" / "unitree_go2_flat"


def find_exported_policy(log_dir: Path | str = DEFAULT_POLICY_DIR) -> Path | None:
    """Locate the most recent exported TorchScript policy, if any."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None
    candidates = sorted(log_dir.glob("*/exported/policy.pt"))
    return candidates[-1] if candidates else None


class Go2PolicyController:
    """Batched inference wrapper for the exported Go2 flat-velocity policy."""

    NUM_OBS = 48
    NUM_ACTIONS = 12

    def __init__(self, policy_path: str | Path, num_envs: int, device: str = "cuda:0"):
        self.policy = torch.jit.load(str(policy_path), map_location=device)
        self.policy.eval()
        self.device = device
        self.num_envs = num_envs
        self.last_actions = torch.zeros(num_envs, self.NUM_ACTIONS, device=device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear action history (call after env reset)."""
        if env_ids is None:
            self.last_actions.zero_()
        else:
            self.last_actions[env_ids] = 0.0

    def compute_observations(self, robot, commands: torch.Tensor) -> torch.Tensor:
        """Build the 48-dim observation vector for every env.

        Term order must match ``ObservationsCfg.PolicyCfg`` of the velocity env
        (flat variant — no height scan):
            base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3) |
            velocity_commands(3) | joint_pos_rel(12) | joint_vel_rel(12) | last_action(12)
        """
        return torch.cat(
            [
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
                robot.data.projected_gravity_b,
                commands,
                robot.data.joint_pos - robot.data.default_joint_pos,
                robot.data.joint_vel - robot.data.default_joint_vel,
                self.last_actions,
            ],
            dim=-1,
        )

    @torch.no_grad()
    def joint_position_targets(self, robot, commands: torch.Tensor) -> torch.Tensor:
        """Run the policy and return absolute joint position targets (N, 12)."""
        obs = self.compute_observations(robot, commands)
        actions = self.policy(obs)
        self.last_actions = actions.clone()
        return robot.data.default_joint_pos + ACTION_SCALE * actions


class WaypointNavigator:
    """Per-env P-controller: goal positions -> velocity commands for the policy."""

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda:0",
        # must match the clearance inflation the path planner applies to
        # policy-followed routes (nav/path_planner.WAYPOINT_TRACKING_TOLERANCE)
        arrival_tolerance: float = WAYPOINT_TRACKING_TOLERANCE,
        heading_gain: float = 1.5,
        heading_deadband: float = 0.6,
    ):
        self.num_envs = num_envs
        self.device = device
        self.arrival_tolerance = arrival_tolerance
        self.heading_gain = heading_gain
        #: only walk forward when roughly facing the goal (rad)
        self.heading_deadband = heading_deadband

    def velocity_commands(
        self,
        positions: torch.Tensor,  # (N, 3) current base positions (env-local)
        yaws: torch.Tensor,  # (N,) current headings
        goals: torch.Tensor,  # (N, 3) goal positions (env-local)
        speeds: torch.Tensor,  # (N,) commanded forward speeds
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (N, 3) velocity commands [vx, vy, wz] and (N,) arrived flags."""
        delta = goals[:, :2] - positions[:, :2]
        distance = torch.norm(delta, dim=-1)
        arrived = distance < self.arrival_tolerance

        heading_to_goal = torch.atan2(delta[:, 1], delta[:, 0])
        heading_error = torch.atan2(
            torch.sin(heading_to_goal - yaws), torch.cos(heading_to_goal - yaws)
        )

        wz = torch.clamp(self.heading_gain * heading_error, -MAX_ANG_VEL, MAX_ANG_VEL)
        # walk at full speed only when roughly facing the goal; slow down near it.
        # Outside the deadband, keep a creep speed instead of stopping: the policy
        # cannot turn in place from a standstill (see TURN_CREEP_SPEED) and a
        # zero-vx turn command deadlocks the robot at sharp path corners.
        facing = heading_error.abs() < self.heading_deadband
        vx = torch.where(
            facing,
            torch.clamp(torch.minimum(speeds, distance * 2.0), 0.0, MAX_LIN_VEL),
            torch.full_like(speeds, TURN_CREEP_SPEED),
        )
        commands = torch.stack([vx, torch.zeros_like(vx), wz], dim=-1)
        # arrived envs get zero commands (stand still)
        commands[arrived] = 0.0
        return commands, arrived


def base_yaws(robot) -> torch.Tensor:
    """Extract base yaw angles (N,) from the robot's world-frame quaternions (wxyz)."""
    quat = robot.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_heading_error(yaw: float, target_yaw: float) -> float:
    """Smallest signed angle from yaw to target_yaw."""
    return math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
