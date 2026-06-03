#!/usr/bin/env python3
"""Record dynamic proof that plan validation runs real robotic control in Isaac Sim.

Produces the README's "watch it run" media from ONE real policy-validation
rollout (the same code path the Sim-Gate uses — rollout_plans_with_policy):

    docs/media/isaac_rollout.gif      — Isaac Sim camera capture: the Go2 walks
                                        the A* route, picks the box, carries it
                                        around the pillar, places it on the shelf
    docs/media/isaac_rollout.mp4      — same capture, H.264 (higher quality)
    docs/media/topdown_trajectory.gif — top-down animation: planned A* route vs
                                        the robot's actually-walked trail
    docs/media/rollout_trajectory.json — raw per-step trajectory log (the data
                                        sanity check runs against this)

Data plausibility checks (run automatically, fail the script if violated):
    * the walked trail never enters the obstacle footprint
    * the box ends up resting on the shelf (success = physics ground truth)
    * the robot's average speed stays within the locomotion envelope

Run inside the Isaac venv:
    python benchmarks/rebuild/record_rollout.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MEDIA_DIR = REPO_ROOT / "docs" / "media"
TASK = "put the fallen box back on shelf A"

# capture cadence: control loop runs at 50 Hz; every 4th step -> 12.5 fps
CAPTURE_EVERY_N_STEPS = 4
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        default="direct",
        choices=["direct", "scan_first", "cautious"],
        help="which mock plan variant to record (default: direct)",
    )
    parser.add_argument("--speed", type=float, default=0.5, help="commanded speed (m/s)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ---- Isaac app with offscreen rendering for the camera ----
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app  # noqa: F841

    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.sensors import Camera, CameraCfg

    from physgate.gate.l2_physics import rollout_plans_with_policy
    from physgate.gate.trajectory import compile_mission
    from physgate.planner.planner import MockPlanner
    from physgate.viz.encode import encode_frames_to_gif, encode_frames_to_mp4
    from physgate.viz.topdown import render_rollout_animation
    from physgate.world.fetch_scene import FetchSimWorld
    from physgate.world.layout import SCENE_LAYOUT, SHELF_TOP_Z
    from physgate.world.locomotion import find_exported_policy

    policy = find_exported_policy()
    if policy is None:
        print("ERROR: no exported Go2 policy found", file=sys.stderr)
        return 1

    # ---- world + recording camera ----
    world = FetchSimWorld(num_envs=1)

    camera = Camera(
        CameraCfg(
            prim_path="/World/viz_camera",
            update_period=0.0,
            height=CAMERA_HEIGHT,
            width=CAMERA_WIDTH,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
        )
    )
    # re-reset so the freshly created camera sensor initializes
    world.sim.reset()

    # 3/4 view across the whole scene (env 0)
    origin = world.env_origins[0].cpu().numpy()
    eye = origin + [1.6, -4.8, 3.2]
    target = origin + [1.6, -0.3, 0.2]
    camera.set_world_poses_from_view(
        torch.tensor([eye], dtype=torch.float32, device=world.device),
        torch.tensor([target], dtype=torch.float32, device=world.device),
    )

    # ---- the plan to record (mock planner, critic-passing variant) ----
    from physgate.examples_lib.fetch_and_place import build_demo_scene

    scene = build_demo_scene()
    plans = MockPlanner()(TASK, scene, 8, None)
    plan = next(p for p in plans if args.plan in p.plan_id)
    print(f"recording plan: {plan.plan_id}")

    # the planned A* route (for the top-down animation)
    mission = compile_mission(plan, SCENE_LAYOUT)
    planned_path = [(0.0, 0.0)] + [
        (float(event[1][0]), float(event[1][1])) for event in mission if event[0] == "goto"
    ]

    # ---- recording hooks ----
    camera_frames: list = []
    samples: list[dict] = []

    def on_control_step(control_step: int, w, state: dict) -> None:
        if control_step % CAPTURE_EVERY_N_STEPS != 0:
            return
        # trajectory sample (env 0)
        robot_local = (w.robot.data.root_pos_w[0] - w.env_origins[0]).cpu().numpy()
        box_local = (w.box.data.root_pos_w[0] - w.env_origins[0]).cpu().numpy()
        samples.append(
            {
                "t": round(control_step * w.dt * 4, 3),
                "robot_xy": [round(float(robot_local[0]), 3), round(float(robot_local[1]), 3)],
                "box_xy": [round(float(box_local[0]), 3), round(float(box_local[1]), 3)],
                "box_z": round(float(box_local[2]), 3),
                "carrying": bool(state["carrying"][0]),
                "mission_index": int(state["mission_index"][0]),
            }
        )
        # camera frame
        camera.update(dt=w.dt)
        rgb = camera.data.output["rgb"][0]
        camera_frames.append(rgb.cpu().numpy().astype("uint8"))

    # ---- run the REAL validation rollout with recording attached ----
    results = rollout_plans_with_policy(world, [plan], policy, on_control_step=on_control_step)
    result = results[0]
    print(
        f"rollout: success={result.success} time={result.completion_time_s}s "
        f"collisions={result.collision_count} frames={len(camera_frames)} samples={len(samples)}"
    )

    # ================= data plausibility checks =================
    failures: list[str] = []

    # 1. physics ground truth: the rollout must succeed (box on shelf)
    if not result.success:
        detail = result.failure.violations[0].detail if result.failure else "?"
        failures.append(f"rollout failed: {detail}")

    # 2. the walked trail never enters the obstacle footprint
    from physgate.world.layout import OBSTACLE_SIZE

    ox, oy = SCENE_LAYOUT["obstacle_P"][:2]
    for s in samples:
        if (
            abs(s["robot_xy"][0] - ox) <= OBSTACLE_SIZE[0] / 2
            and abs(s["robot_xy"][1] - oy) <= OBSTACLE_SIZE[1] / 2
        ):
            failures.append(f"trail point {s['robot_xy']} inside obstacle footprint at t={s['t']}")
            break

    # 3. robot average moving speed stays in the locomotion envelope (with tolerance)
    moving = [
        (samples[i], samples[i + 1])
        for i in range(len(samples) - 1)
        if not math.isclose(samples[i]["robot_xy"][0], samples[i + 1]["robot_xy"][0], abs_tol=1e-4)
        or not math.isclose(samples[i]["robot_xy"][1], samples[i + 1]["robot_xy"][1], abs_tol=1e-4)
    ]
    if moving:
        speeds = [
            math.hypot(b["robot_xy"][0] - a["robot_xy"][0], b["robot_xy"][1] - a["robot_xy"][1])
            / (b["t"] - a["t"])
            for a, b in moving
            if b["t"] > a["t"]
        ]
        mean_speed = sum(speeds) / len(speeds)
        if not 0.1 <= mean_speed <= 1.0:
            failures.append(f"mean moving speed {mean_speed:.2f} m/s outside plausible range")
        print(f"mean moving speed: {mean_speed:.2f} m/s (envelope [0.4, 0.6] commanded)")

    # 4. the box ends up at shelf height (resting on the shelf top)
    if samples and result.success:
        final_box_z = samples[-1]["box_z"]
        if not (SHELF_TOP_Z - 0.05) <= final_box_z <= (SHELF_TOP_Z + 0.3):
            failures.append(
                f"final box height {final_box_z:.2f} m inconsistent with shelf top {SHELF_TOP_Z} m"
            )

    if failures:
        print("\nDATA PLAUSIBILITY CHECK FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  ✗ {f}", file=sys.stderr)
        return 2
    print("data plausibility checks: all passed")

    # ================= encode the recordings =================
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    # Isaac camera capture
    encode_frames_to_mp4(camera_frames, MEDIA_DIR / "isaac_rollout.mp4", fps=12)
    encode_frames_to_gif(camera_frames, MEDIA_DIR / "isaac_rollout.gif", fps=12)

    # top-down trajectory animation (with collision checking on)
    topdown_frames = render_rollout_animation(
        layout=SCENE_LAYOUT,
        planned_path=planned_path,
        samples=samples,
        hold_last_frames=18,
        check_collisions=True,
    )
    encode_frames_to_gif(topdown_frames, MEDIA_DIR / "topdown_trajectory.gif", fps=12)

    # raw trajectory log (the evidence the checks ran against)
    log = {
        "plan_id": plan.plan_id,
        "task": TASK,
        "rollout_success": result.success,
        "completion_time_s": result.completion_time_s,
        "collision_count": result.collision_count,
        "planned_path": planned_path,
        "samples": samples,
    }
    (MEDIA_DIR / "rollout_trajectory.json").write_text(json.dumps(log, indent=1))

    sizes = {
        p.name: f"{p.stat().st_size / 1e6:.1f} MB"
        for p in sorted(MEDIA_DIR.iterdir())
        if p.suffix in (".gif", ".mp4", ".json")
    }
    print(f"media written -> {MEDIA_DIR}: {sizes}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.exit(code)
