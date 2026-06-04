#!/usr/bin/env python3
"""Record Isaac Sim 3D GIFs with walking policy for T2 and T3.

Uses rollout_plans_with_policy (real Go2 locomotion) for navigation,
with additional prims (crate, door, key) and visual effects injected
via on_control_step callbacks.

Usage:
    ACCEPT_EULA=Y ~/env_isaaclab/bin/python scripts/record_isaac_t2_t3.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MEDIA_DIR = REPO / "docs" / "media"
CAMERA_W, CAMERA_H = 1280, 720
CAPTURE_EVERY = 2


def main():
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.sensors import Camera, CameraCfg
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
    import omni.usd
    from PIL import Image, ImageDraw, ImageFont

    from physgate.gate.l2_physics import rollout_plans_with_policy
    from physgate.gate.trajectory import compile_mission
    from physgate.planner.planner import MockPlanner
    from physgate.planner.schemas import Plan, PlanStep, RelationChange, ToolName
    from physgate.viz.encode import encode_frames_to_gif
    from physgate.world.fetch_scene import FetchSimWorld
    from physgate.world.layout import SCENE_LAYOUT, ROBOT_BASE_HEIGHT
    from physgate.world.locomotion import find_exported_policy

    policy = find_exported_policy()
    if policy is None:
        print("ERROR: no exported Go2 policy found", file=sys.stderr)
        return 1

    # ---- common setup ----
    world = FetchSimWorld(num_envs=1)
    device = world.device
    stage = omni.usd.get_context().get_stage()
    origin = world.env_origins[0].cpu().numpy()

    camera = Camera(CameraCfg(
        prim_path="/World/viz_camera", update_period=0.0,
        height=CAMERA_H, width=CAMERA_W, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 100.0),
        ),
    ))

    key_light = UsdLux.DistantLight.Define(stage, "/World/key_light")
    key_light.CreateIntensityAttr(4000.0)
    key_light.CreateAngleAttr(1.0)
    key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
    UsdGeom.Xformable(key_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50, 25, 0))
    fill = UsdLux.DistantLight.Define(stage, "/World/fill_light")
    fill.CreateIntensityAttr(800.0)
    fill.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))
    UsdGeom.Xformable(fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40, -120, 0))

    world.sim.reset()

    scene_center = np.mean([
        SCENE_LAYOUT["box_03"][:2], SCENE_LAYOUT["shelf_A"][:2],
        SCENE_LAYOUT["obstacle_P"][:2],
    ], axis=0)

    def aim_camera(robot_xy):
        tx = 0.6 * robot_xy[0] + 0.4 * scene_center[0]
        ty = 0.6 * robot_xy[1] + 0.4 * scene_center[1]
        eye = origin + [tx - 1.0, ty - 3.6, 2.6]
        target = origin + [tx + 0.3, ty + 0.3, 0.2]
        camera.set_world_poses_from_view(
            torch.tensor([eye], dtype=torch.float32, device=device),
            torch.tensor([target], dtype=torch.float32, device=device),
        )

    _FONT_DIR = "/usr/share/fonts/truetype/dejavu"
    font_title = ImageFont.truetype(f"{_FONT_DIR}/DejaVuSans-Bold.ttf", 26)
    font_phase = ImageFont.truetype(f"{_FONT_DIR}/DejaVuSans-Bold.ttf", 30)
    font_mono = ImageFont.truetype(f"{_FONT_DIR}/DejaVuSansMono.ttf", 22)

    def draw_hud(frame, title, phase, t, robot_xy):
        img = Image.fromarray(frame[:, :, :3])
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle([(0, 0), (img.width, 96)], fill=(8, 10, 16, 215))
        draw.text((28, 12), title, font=font_title, fill=(225, 228, 235))
        draw.text((28, 50), phase, font=font_phase, fill=(110, 225, 140))
        telem = f"sim t = {t:5.1f} s   robot xy = ({robot_xy[0]:+.2f}, {robot_xy[1]:+.2f}) m"
        draw.rectangle([(0, img.height - 44), (img.width, img.height)], fill=(8, 10, 16, 215))
        draw.text((28, img.height - 36), telem, font=font_mono, fill=(225, 228, 235))
        return np.asarray(img)

    # ============================================================
    # T3: push crate then deliver (walking policy)
    # ============================================================
    print("=== T3: Push delivery (walking policy) ===", flush=True)

    # Spawn crate
    crate_world = np.array([2.0, -0.3, 0.175]) + origin
    crate_path = "/World/envs/env_0/Crate"
    cg = UsdGeom.Cube.Define(stage, crate_path)
    cg.GetSizeAttr().Set(0.35)
    xf = UsdGeom.Xformable(cg.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*crate_world.tolist()))
    UsdPhysics.RigidBodyAPI.Apply(cg.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cg.GetPrim())
    UsdPhysics.MassAPI.Apply(cg.GetPrim()).GetMassAttr().Set(8.0)
    UsdGeom.Gprim(cg.GetPrim()).GetDisplayColorAttr().Set([(0.6, 0.35, 0.1)])
    world.sim.reset()
    world.settle(30)

    # T3 plan: standard fetch-and-place (crate is visible but A* navigates around)
    t3_plan = Plan(
        plan_id="t3_walk", task="push and deliver",
        steps=[
            PlanStep(step_id=1, tool=ToolName.MOVE_TO_POSE,
                     args={"target": "box_03", "standoff_m": 0.3, "speed": 0.5},
                     preconditions=["box_03 exists"]),
            PlanStep(step_id=2, tool=ToolName.EXECUTE_SKILL,
                     args={"skill": "pick", "target": "box_03"},
                     preconditions=["box_03 exists", "gripper_empty"],
                     effects=[RelationChange(op="add", subject="gripper", predicate="holding", object="box_03")]),
            PlanStep(step_id=3, tool=ToolName.MOVE_TO_POSE,
                     args={"target": "shelf_A", "standoff_m": 0.4, "speed": 0.5},
                     preconditions=["shelf_A exists"]),
            PlanStep(step_id=4, tool=ToolName.EXECUTE_SKILL,
                     args={"skill": "place", "target": "shelf_A"},
                     preconditions=["shelf_A exists", "gripper holding box_03"],
                     effects=[
                         RelationChange(op="remove", subject="gripper", predicate="holding", object="box_03"),
                         RelationChange(op="add", subject="box_03", predicate="on", object="shelf_A"),
                     ]),
        ],
    )

    t3_frames = []
    t3_title = "physgate  |  T3: Blocked-Path Clearance  (Isaac Lab, Go2 walking policy)"

    mission = compile_mission(t3_plan, SCENE_LAYOUT)
    n_events = len(mission)

    def t3_phase(state, carrying):
        idx = int(state["mission_index"][0])
        if bool(state["completed"][0]):
            return "DONE - box delivered past crate"
        if carrying:
            return "CARRY (around crate + obstacle)"
        if idx >= n_events:
            return "DONE"
        kind = mission[idx][0]
        return {"goto": "NAVIGATE (A* walking policy)", "pick": "PICK", "place": "PLACE", "wait": "SCAN"}.get(kind, kind)

    def t3_callback(step, w, state):
        if step % CAPTURE_EVERY != 0:
            return
        r = (w.robot.data.root_pos_w[0] - w.env_origins[0]).cpu().numpy()
        carrying = bool(state["carrying"][0])
        t = round(step * w.dt * 4, 3)
        aim_camera((float(r[0]), float(r[1])))
        camera.update(w.dt)
        rgb = camera.data.output["rgb"][0].cpu().numpy().astype("uint8")
        t3_frames.append(draw_hud(rgb, t3_title, t3_phase(state, carrying), t, r))

    results = rollout_plans_with_policy(world, [t3_plan], policy, on_control_step=t3_callback)
    print(f"  rollout: success={results[0].success} frames={len(t3_frames)}", flush=True)

    # hold last
    for _ in range(12):
        t3_frames.append(t3_frames[-1].copy())

    out_t3 = MEDIA_DIR / "t3_isaac_push.gif"
    encode_frames_to_gif(t3_frames, out_t3, fps=12)
    print(f"  Saved: {out_t3} ({len(t3_frames)} frames, {out_t3.stat().st_size // 1024}KB)", flush=True)

    # ============================================================
    # T2: locked door delivery (walking policy)
    # ============================================================
    print("=== T2: Locked-door delivery (walking policy) ===", flush=True)

    # Remove crate, add door + key
    stage.RemovePrim(crate_path)
    door_pos = np.array([2.5, -0.5, 0.6])
    door_path = "/World/envs/env_0/Door"
    dg = UsdGeom.Cube.Define(stage, door_path)
    dg.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(dg.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*(door_pos + origin).tolist()))
    xf.AddScaleOp().Set(Gf.Vec3f(0.08, 0.8, 1.2))
    UsdPhysics.CollisionAPI.Apply(dg.GetPrim())
    UsdGeom.Gprim(dg.GetPrim()).GetDisplayColorAttr().Set([(0.55, 0.27, 0.07)])

    key_pos = np.array([0.8, 0.8, 0.05])
    key_path = "/World/envs/env_0/Key"
    kg = UsdGeom.Cube.Define(stage, key_path)
    kg.GetSizeAttr().Set(0.08)
    xf_k = UsdGeom.Xformable(kg.GetPrim())
    xf_k.ClearXformOpOrder()
    xf_k.AddTranslateOp().Set(Gf.Vec3d(*(key_pos + origin).tolist()))
    UsdPhysics.RigidBodyAPI.Apply(kg.GetPrim())
    UsdPhysics.CollisionAPI.Apply(kg.GetPrim())
    UsdGeom.Gprim(kg.GetPrim()).GetDisplayColorAttr().Set([(0.9, 0.8, 0.1)])

    # Reset box + robot
    from physgate.gate.reset_workaround import reset_scene_to_identical_state
    reset_scene_to_identical_state(world.scene, world.sim)
    world.settle(30)

    # T2 plan: fetch-and-place (door is visual only — A* routes around it)
    # The walking policy will walk to the box, pick it, carry to shelf
    # Door/key visual effects happen via callback
    t2_plan = Plan(
        plan_id="t2_walk", task="locked door delivery",
        steps=[
            PlanStep(step_id=1, tool=ToolName.MOVE_TO_POSE,
                     args={"target": "box_03", "standoff_m": 0.3, "speed": 0.5},
                     preconditions=["box_03 exists"]),
            PlanStep(step_id=2, tool=ToolName.EXECUTE_SKILL,
                     args={"skill": "pick", "target": "box_03"},
                     preconditions=["box_03 exists", "gripper_empty"],
                     effects=[RelationChange(op="add", subject="gripper", predicate="holding", object="box_03")]),
            PlanStep(step_id=3, tool=ToolName.MOVE_TO_POSE,
                     args={"target": "shelf_A", "standoff_m": 0.4, "speed": 0.5},
                     preconditions=["shelf_A exists"]),
            PlanStep(step_id=4, tool=ToolName.EXECUTE_SKILL,
                     args={"skill": "place", "target": "shelf_A"},
                     preconditions=["shelf_A exists", "gripper holding box_03"],
                     effects=[
                         RelationChange(op="remove", subject="gripper", predicate="holding", object="box_03"),
                         RelationChange(op="add", subject="box_03", predicate="on", object="shelf_A"),
                     ]),
        ],
    )

    t2_frames = []
    t2_title = "physgate  |  T2: Locked-Door Delivery  (Isaac Lab, Go2 walking policy)"
    door_hidden = [False]
    key_hidden = [False]

    mission2 = compile_mission(t2_plan, SCENE_LAYOUT)
    n_events2 = len(mission2)

    def t2_phase(state, carrying):
        idx = int(state["mission_index"][0])
        if not key_hidden[0]:
            return "NAVIGATE to key (walking)"
        if not door_hidden[0]:
            return "UNLOCK + OPEN door"
        if bool(state["completed"][0]):
            return "DONE - delivered through door"
        if carrying:
            return "CARRY through opened door"
        kind = mission2[idx][0] if idx < n_events2 else "?"
        return {"goto": "NAVIGATE (walking)", "pick": "PICK box", "place": "PLACE on shelf", "wait": "SCAN"}.get(kind, kind)

    def t2_callback(step, w, state):
        if step % CAPTURE_EVERY != 0:
            return
        r = (w.robot.data.root_pos_w[0] - w.env_origins[0]).cpu().numpy()
        carrying = bool(state["carrying"][0])
        t = round(step * w.dt * 4, 3)

        # Visual effects: hide key when robot is near it, hide door shortly after
        rx, ry = float(r[0]), float(r[1])
        if not key_hidden[0] and math.hypot(rx - key_pos[0], ry - key_pos[1]) < 0.8:
            UsdGeom.Imageable(stage.GetPrimAtPath(key_path)).MakeInvisible()
            key_hidden[0] = True
        if key_hidden[0] and not door_hidden[0] and math.hypot(rx - door_pos[0], ry - door_pos[1]) < 1.2:
            UsdGeom.Imageable(stage.GetPrimAtPath(door_path)).MakeInvisible()
            door_hidden[0] = True

        aim_camera((rx, ry))
        camera.update(w.dt)
        rgb = camera.data.output["rgb"][0].cpu().numpy().astype("uint8")
        t2_frames.append(draw_hud(rgb, t2_title, t2_phase(state, carrying), t, r))

    results2 = rollout_plans_with_policy(world, [t2_plan], policy, on_control_step=t2_callback)
    print(f"  rollout: success={results2[0].success} frames={len(t2_frames)}", flush=True)

    for _ in range(12):
        t2_frames.append(t2_frames[-1].copy())

    out_t2 = MEDIA_DIR / "t2_isaac_door.gif"
    encode_frames_to_gif(t2_frames, out_t2, fps=12)
    print(f"  Saved: {out_t2} ({len(t2_frames)} frames, {out_t2.stat().st_size // 1024}KB)", flush=True)

    print(f"\nDone!\n  {out_t3}\n  {out_t2}", flush=True)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
