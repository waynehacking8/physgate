"""Diagnostic: can the Go2 policy turn in place? (REBUILD Phase 2 stall investigation)

The policy rollout deadlocks when the waypoint navigator commands pure rotation
(vx=0, wz~0.9) at path corners whose heading error exceeds the 0.6 rad deadband.
This script measures how the trained policy responds to rotation commands with
varying forward-velocity components, to determine the correct navigator fix.

Run inside the Isaac venv:
    python benchmarks/rebuild/diag_turn_in_place.py
"""

from __future__ import annotations

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch  # noqa: E402

from physgate.gate.reset_workaround import reset_scene_to_identical_state  # noqa: E402
from physgate.world.fetch_scene import FetchSimWorld  # noqa: E402
from physgate.world.locomotion import (  # noqa: E402
    POLICY_DECIMATION,
    Go2PolicyController,
    base_yaws,
    find_exported_policy,
)

NUM_ENVS = 4
TRIAL_SECONDS = 6.0

# one command per env, applied constantly for TRIAL_SECONDS:
#   env 0: pure rotation            (what the navigator commands at corners today)
#   env 1: creep forward + rotate   (tight arc)
#   env 2: slow forward + rotate    (arc)
#   env 3: envelope forward + rotate (wide arc)
COMMANDS = [
    (0.0, 0.0, 0.9),
    (0.1, 0.0, 0.9),
    (0.2, 0.0, 0.9),
    (0.4, 0.0, 0.9),
]


def main() -> None:
    world = FetchSimWorld(num_envs=NUM_ENVS)
    policy_path = find_exported_policy()
    if policy_path is None:
        raise SystemExit("no exported Go2 policy found")

    reset_scene_to_identical_state(world.scene, world.sim)
    controller = Go2PolicyController(policy_path, num_envs=NUM_ENVS, device=world.device)

    commands = torch.tensor(COMMANDS, dtype=torch.float32, device=world.device)
    control_dt = world.dt * POLICY_DECIMATION
    steps = int(TRIAL_SECONDS / control_dt)

    start_pos = (world.robot.data.root_pos_w - world.env_origins).clone()
    start_yaw = base_yaws(world.robot).clone()
    yaw_unwrapped = start_yaw.clone()
    prev_yaw = start_yaw.clone()

    for _ in range(steps):
        targets = controller.joint_position_targets(world.robot, commands)
        world.apply_joint_targets(targets)
        for _ in range(POLICY_DECIMATION):
            world.step()
        # unwrap yaw so multi-revolution turns are measured correctly
        yaw = base_yaws(world.robot)
        delta = torch.atan2(torch.sin(yaw - prev_yaw), torch.cos(yaw - prev_yaw))
        yaw_unwrapped += delta
        prev_yaw = yaw

    end_pos = world.robot.data.root_pos_w - world.env_origins
    total_turn = yaw_unwrapped - start_yaw
    displacement = torch.norm(end_pos[:, :2] - start_pos[:, :2], dim=-1)
    upright = world.robot.data.projected_gravity_b[:, 2] < -0.6

    print("\n=== turn-in-place diagnostic ===")
    print(f"trial: {TRIAL_SECONDS:.0f} s per command, commanded wz = 0.9 rad/s")
    print(f"expected turn if tracking perfectly: {0.9 * TRIAL_SECONDS:.2f} rad")
    for i, cmd in enumerate(COMMANDS):
        print(
            f"  cmd(vx={cmd[0]:.1f}, wz={cmd[2]:.1f}): "
            f"turned {float(total_turn[i]):+.2f} rad "
            f"({float(total_turn[i]) / (0.9 * TRIAL_SECONDS) * 100:.0f}% of commanded), "
            f"displaced {float(displacement[i]):.2f} m, "
            f"upright={bool(upright[i])}"
        )
    print("================================\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
