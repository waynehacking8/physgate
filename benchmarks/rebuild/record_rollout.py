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

import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MEDIA_DIR = REPO_ROOT / "docs" / "media"
TASK = "put the fallen box back on shelf A"

# capture cadence: control loop runs at 50 Hz; every 2nd step -> 25 fps
CAPTURE_EVERY_N_STEPS = 2
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720


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
    # directional key light (USD API — reliable spawn): angled sunlight gives the
    # scene shadows and depth; the default dome light alone renders flat
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    stage = omni.usd.get_context().get_stage()
    key_light = UsdLux.DistantLight.Define(stage, "/World/key_light")
    key_light.CreateIntensityAttr(4000.0)
    key_light.CreateAngleAttr(1.0)
    key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
    UsdGeom.Xformable(key_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 25.0, 0.0))
    # soft fill from the opposite side so shadows aren't pitch black
    fill_light = UsdLux.DistantLight.Define(stage, "/World/fill_light")
    fill_light.CreateIntensityAttr(800.0)
    fill_light.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))
    UsdGeom.Xformable(fill_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, -120.0, 0.0))

    # re-reset so the freshly created camera sensor initializes
    world.sim.reset()

    origin = world.env_origins[0].cpu().numpy()

    # scene centroid (box / shelf / pillar) keeps the task context in frame
    scene_center = np.mean(
        [SCENE_LAYOUT["box_03"][:2], SCENE_LAYOUT["shelf_A"][:2], SCENE_LAYOUT["obstacle_P"][:2]],
        axis=0,
    )

    def aim_camera_at(robot_xy: tuple[float, float]) -> None:
        """Tracking camera: 3/4 view following the robot, framed so the task
        objects stay visible (target = weighted robot/scene-center blend)."""
        tx = 0.6 * robot_xy[0] + 0.4 * scene_center[0]
        ty = 0.6 * robot_xy[1] + 0.4 * scene_center[1]
        eye = origin + [tx - 1.0, ty - 3.6, 2.6]
        target = origin + [tx + 0.3, ty + 0.3, 0.2]
        camera.set_world_poses_from_view(
            torch.tensor([eye], dtype=torch.float32, device=world.device),
            torch.tensor([target], dtype=torch.float32, device=world.device),
        )

    aim_camera_at((0.0, 0.0))

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

    # mission phases for the HUD (derived from the plan structure)
    n_mission_events = len(mission)

    def phase_label(state: dict, carrying: bool) -> str:
        idx = int(state["mission_index"][0])
        if bool(state["completed"][0]):
            return "DONE - box placed on shelf"
        if carrying:
            return "CARRY  (kinematic attach, D-018)"
        if idx >= n_mission_events:
            return "DONE"
        kind = mission[idx][0] if idx < n_mission_events else "?"
        return {
            "goto": "NAVIGATE  (A* route, walking policy)",
            "pick": "PICK",
            "place": "PLACE  (momentum release)",
            "wait": "SCAN",
        }.get(kind, kind.upper())

    from PIL import Image, ImageDraw, ImageFont

    _FONT_DIR = "/usr/share/fonts/truetype/dejavu"
    font_title = ImageFont.truetype(f"{_FONT_DIR}/DejaVuSans-Bold.ttf", 26)
    font_phase = ImageFont.truetype(f"{_FONT_DIR}/DejaVuSans-Bold.ttf", 30)
    font_mono = ImageFont.truetype(f"{_FONT_DIR}/DejaVuSansMono.ttf", 22)

    def draw_hud(frame, state: dict, t: float, robot_xy, carrying: bool):
        """Annotate the frame: phase, sim time, telemetry (paper-video style HUD)."""
        img = Image.fromarray(frame[:, :, :3])
        draw = ImageDraw.Draw(img, "RGBA")
        # top banner
        draw.rectangle([(0, 0), (img.width, 96)], fill=(8, 10, 16, 215))
        draw.text(
            (28, 12),
            "physgate  |  Sim-Gate validation rollout  (Isaac Lab PhysX, Go2 rsl_rl walking policy)",
            font=font_title,
            fill=(225, 228, 235),
        )
        draw.text(
            (28, 50), phase_label(state, carrying), font=font_phase, fill=(110, 225, 140)
        )
        # bottom telemetry bar
        telem = (
            f"sim t = {t:5.1f} s   robot xy = ({robot_xy[0]:+.2f}, {robot_xy[1]:+.2f}) m   "
            f"commanded speed envelope [0.4, 0.6] m/s"
        )
        draw.rectangle([(0, img.height - 44), (img.width, img.height)], fill=(8, 10, 16, 215))
        draw.text((28, img.height - 36), telem, font=font_mono, fill=(225, 228, 235))
        return np.asarray(img)

    def on_control_step(control_step: int, w, state: dict) -> None:
        if control_step % CAPTURE_EVERY_N_STEPS != 0:
            return
        # trajectory sample (env 0)
        robot_local = (w.robot.data.root_pos_w[0] - w.env_origins[0]).cpu().numpy()
        box_local = (w.box.data.root_pos_w[0] - w.env_origins[0]).cpu().numpy()
        carrying = bool(state["carrying"][0])
        t = round(control_step * w.dt * 4, 3)
        samples.append(
            {
                "t": t,
                "robot_xy": [round(float(robot_local[0]), 3), round(float(robot_local[1]), 3)],
                "box_xy": [round(float(box_local[0]), 3), round(float(box_local[1]), 3)],
                "box_z": round(float(box_local[2]), 3),
                "carrying": carrying,
                "mission_index": int(state["mission_index"][0]),
            }
        )
        # tracking camera + frame capture + HUD
        aim_camera_at((float(robot_local[0]), float(robot_local[1])))
        camera.update(dt=w.dt)
        rgb = camera.data.output["rgb"][0].cpu().numpy().astype("uint8")
        camera_frames.append(draw_hud(rgb, state, t, robot_local, carrying))

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

    # Isaac camera capture: full-quality MP4 + README-sized GIF
    encode_frames_to_mp4(camera_frames, MEDIA_DIR / "isaac_rollout.mp4", fps=25)
    # README GIF: 512 px wide, 10 fps (GitHub-friendly size)
    gif_frames = [f[::2, ::2] for f in camera_frames[::3]]
    encode_frames_to_gif(gif_frames, MEDIA_DIR / "isaac_rollout.gif", fps=8)

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
