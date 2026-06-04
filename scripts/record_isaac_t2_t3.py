#!/usr/bin/env python3
"""Record Isaac Sim 3D GIFs for T2 (locked-door) and T3 (push crate).

Uses isaaclab.sensors.Camera (same approach as record_rollout.py) for
proven headless rendering. Robot is driven kinematically; extra prims
(crate, door, key) are spawned into the base scene.

Usage:
    ACCEPT_EULA=Y ~/env_isaaclab/bin/python scripts/record_isaac_t2_t3.py

Output: docs/media/t3_isaac_push.gif, docs/media/t2_isaac_door.gif
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


def main():
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.sensors import Camera, CameraCfg
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
    import omni.usd

    from physgate.gate.trajectory import yaw_to_quat
    from physgate.viz.encode import encode_frames_to_gif
    from physgate.world.fetch_scene import FetchSimWorld
    from physgate.world.layout import SCENE_LAYOUT, ROBOT_BASE_HEIGHT

    # ---- build world + camera ----
    world = FetchSimWorld(num_envs=1)
    device = world.device
    origin = world.env_origins[0].cpu().numpy()
    stage = omni.usd.get_context().get_stage()

    camera = Camera(CameraCfg(
        prim_path="/World/viz_camera",
        update_period=0.0,
        height=CAMERA_H,
        width=CAMERA_W,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 100.0),
        ),
    ))

    # lighting
    key_light = UsdLux.DistantLight.Define(stage, "/World/key_light")
    key_light.CreateIntensityAttr(4000.0)
    key_light.CreateAngleAttr(1.0)
    key_light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
    UsdGeom.Xformable(key_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50, 25, 0))
    fill_light = UsdLux.DistantLight.Define(stage, "/World/fill_light")
    fill_light.CreateIntensityAttr(800.0)
    fill_light.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))
    UsdGeom.Xformable(fill_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40, -120, 0))

    world.sim.reset()

    scene_center = np.mean([
        SCENE_LAYOUT["box_03"][:2],
        SCENE_LAYOUT["shelf_A"][:2],
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

    def capture(robot_xy):
        aim_camera(robot_xy)
        world.scene.write_data_to_sim()
        world.sim.step()
        world.scene.update(world.dt)
        camera.update(world.dt)
        rgb = camera.data.output["rgb"][0].cpu().numpy()
        return rgb[:, :, :3].copy()

    def drive_to(start, end, n=40, carry_box=False):
        """Drive robot, optionally carry box, capture every 2nd frame."""
        frames = []
        yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        for i in range(n):
            t = (i + 1) / n
            pos = start * (1 - t) + end * t
            wp = pos + origin
            all_pos = world.robot.data.root_pos_w.clone()
            all_quat = world.robot.data.root_quat_w.clone()
            all_pos[0] = torch.tensor(wp, dtype=torch.float32, device=device)
            all_quat[0] = torch.tensor(yaw_to_quat(yaw), dtype=torch.float32, device=device)
            world.write_robot_poses(all_pos, all_quat)
            if carry_box:
                bw = pos + np.array([0.0, 0.0, 0.6]) + origin
                world.write_box_poses(
                    torch.tensor([bw], dtype=torch.float32, device=device),
                    env_ids=torch.tensor([0], dtype=torch.long, device=device),
                )
            world.scene.write_data_to_sim()
            world.sim.step()
            world.scene.update(world.dt)
            if i % 2 == 0:
                camera.update(world.dt)
                rgb = camera.data.output["rgb"][0].cpu().numpy()
                frames.append(rgb[:, :, :3].copy())
        return frames, end

    # ================ T3: push crate ================
    print("=== Recording T3: Push delivery (Isaac Sim 3D) ===", flush=True)

    # spawn crate
    crate_pos = np.array([2.0, -0.3, 0.175])
    crate_path = "/World/envs/env_0/Crate"
    crate_geom = UsdGeom.Cube.Define(stage, crate_path)
    crate_geom.GetSizeAttr().Set(0.35)
    xf = UsdGeom.Xformable(crate_geom.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*(crate_pos + origin)))
    UsdPhysics.RigidBodyAPI.Apply(crate_geom.GetPrim())
    UsdPhysics.CollisionAPI.Apply(crate_geom.GetPrim())
    UsdPhysics.MassAPI.Apply(crate_geom.GetPrim()).GetMassAttr().Set(8.0)
    UsdGeom.Gprim(crate_geom.GetPrim()).GetDisplayColorAttr().Set([(0.6, 0.35, 0.1)])
    world.sim.reset()
    world.settle(10)

    t3_frames = []
    robot_pos = np.array([0.0, 0.0, ROBOT_BASE_HEIGHT])

    print("  Step 1: navigate to crate", flush=True)
    end = np.array([crate_pos[0] - 0.5, crate_pos[1], ROBOT_BASE_HEIGHT])
    fr, robot_pos = drive_to(robot_pos, end, n=40)
    t3_frames.extend(fr)

    print("  Step 2: push crate (drive through)", flush=True)
    end = np.array([crate_pos[0] + 0.6, crate_pos[1], ROBOT_BASE_HEIGHT])
    fr, robot_pos = drive_to(robot_pos, end, n=40)
    t3_frames.extend(fr)

    print("  Step 3: navigate to box", flush=True)
    box_xy = np.array(SCENE_LAYOUT["box_03"])
    end = np.array([box_xy[0] - 0.3, box_xy[1], ROBOT_BASE_HEIGHT])
    fr, robot_pos = drive_to(robot_pos, end, n=30)
    t3_frames.extend(fr)

    print("  Step 4: pick box", flush=True)
    for _ in range(6):
        bw = robot_pos + np.array([0.0, 0.0, 0.6]) + origin
        world.write_box_poses(
            torch.tensor([bw], dtype=torch.float32, device=device),
            env_ids=torch.tensor([0], dtype=torch.long, device=device),
        )
        t3_frames.append(capture(robot_pos[:2]))

    print("  Step 5: carry to shelf", flush=True)
    shelf_xy = np.array(SCENE_LAYOUT["shelf_A"])
    end = np.array([shelf_xy[0] - 0.4, shelf_xy[1], ROBOT_BASE_HEIGHT])
    fr, robot_pos = drive_to(robot_pos, end, n=40, carry_box=True)
    t3_frames.extend(fr)

    print("  Step 6: place", flush=True)
    place = np.array([shelf_xy[0], shelf_xy[1], 0.6]) + origin
    world.write_box_poses(
        torch.tensor([place], dtype=torch.float32, device=device),
        env_ids=torch.tensor([0], dtype=torch.long, device=device),
    )
    world.box.write_root_velocity_to_sim(
        torch.zeros(1, 6, device=device),
        env_ids=torch.tensor([0], dtype=torch.long, device=device),
    )
    for _ in range(30):
        world.step()
    for _ in range(8):
        t3_frames.append(capture(robot_pos[:2]))
    for _ in range(10):
        t3_frames.append(t3_frames[-1].copy())

    out_t3 = MEDIA_DIR / "t3_isaac_push.gif"
    encode_frames_to_gif(t3_frames, out_t3, fps=10)
    print(f"  Saved: {out_t3} ({len(t3_frames)} frames, {out_t3.stat().st_size // 1024}KB)", flush=True)

    # ================ T2: locked door ================
    print("=== Recording T2: Locked-door delivery (Isaac Sim 3D) ===", flush=True)

    # remove crate, spawn door + key
    stage.RemovePrim(crate_path)
    door_pos = np.array([2.5, -0.5, 0.6])
    door_path = "/World/envs/env_0/Door"
    dg = UsdGeom.Cube.Define(stage, door_path)
    dg.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(dg.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*(door_pos + origin)))
    xf.AddScaleOp().Set(Gf.Vec3f(0.08, 0.8, 1.2))
    UsdPhysics.CollisionAPI.Apply(dg.GetPrim())
    UsdGeom.Gprim(dg.GetPrim()).GetDisplayColorAttr().Set([(0.55, 0.27, 0.07)])

    key_pos = np.array([0.8, 0.8, 0.05])
    key_path = "/World/envs/env_0/Key"
    kg = UsdGeom.Cube.Define(stage, key_path)
    kg.GetSizeAttr().Set(0.08)
    xf_k = UsdGeom.Xformable(kg.GetPrim())
    xf_k.ClearXformOpOrder()
    xf_k.AddTranslateOp().Set(Gf.Vec3d(*(key_pos + origin)))
    UsdPhysics.RigidBodyAPI.Apply(kg.GetPrim())
    UsdPhysics.CollisionAPI.Apply(kg.GetPrim())
    UsdGeom.Gprim(kg.GetPrim()).GetDisplayColorAttr().Set([(0.9, 0.8, 0.1)])

    # reset box to original position
    box_orig = np.array(SCENE_LAYOUT["box_03"]) + origin
    world.write_box_poses(
        torch.tensor([box_orig], dtype=torch.float32, device=device),
        env_ids=torch.tensor([0], dtype=torch.long, device=device),
    )
    world.sim.reset()
    world.settle(10)

    t2_frames = []
    robot_pos = np.array([0.0, 0.0, ROBOT_BASE_HEIGHT])

    print("  Step 1: navigate to key", flush=True)
    end = np.array([key_pos[0] - 0.3, key_pos[1], ROBOT_BASE_HEIGHT])
    fr, robot_pos = drive_to(robot_pos, end, n=25)
    t2_frames.extend(fr)

    print("  Step 2: pick key (hide)", flush=True)
    UsdGeom.Imageable(stage.GetPrimAtPath(key_path)).MakeInvisible()
    for _ in range(5):
        t2_frames.append(capture(robot_pos[:2]))

    print("  Step 3: navigate to door", flush=True)
    end = np.array([door_pos[0] - 0.4, door_pos[1], ROBOT_BASE_HEIGHT])
    fr, robot_pos = drive_to(robot_pos, end, n=30)
    t2_frames.extend(fr)

    print("  Step 4: unlock + open (hide door)", flush=True)
    UsdGeom.Imageable(stage.GetPrimAtPath(door_path)).MakeInvisible()
    for _ in range(6):
        t2_frames.append(capture(robot_pos[:2]))

    print("  Step 5: navigate to box", flush=True)
    end = np.array([box_xy[0] - 0.3, box_xy[1], ROBOT_BASE_HEIGHT])
    fr, robot_pos = drive_to(robot_pos, end, n=25)
    t2_frames.extend(fr)

    print("  Step 6: pick box", flush=True)
    for _ in range(5):
        bw = robot_pos + np.array([0.0, 0.0, 0.6]) + origin
        world.write_box_poses(
            torch.tensor([bw], dtype=torch.float32, device=device),
            env_ids=torch.tensor([0], dtype=torch.long, device=device),
        )
        t2_frames.append(capture(robot_pos[:2]))

    print("  Step 7: carry to shelf", flush=True)
    end = np.array([shelf_xy[0] - 0.4, shelf_xy[1], ROBOT_BASE_HEIGHT])
    fr, robot_pos = drive_to(robot_pos, end, n=35, carry_box=True)
    t2_frames.extend(fr)

    print("  Step 8: place", flush=True)
    world.write_box_poses(
        torch.tensor([place], dtype=torch.float32, device=device),
        env_ids=torch.tensor([0], dtype=torch.long, device=device),
    )
    world.box.write_root_velocity_to_sim(
        torch.zeros(1, 6, device=device),
        env_ids=torch.tensor([0], dtype=torch.long, device=device),
    )
    for _ in range(30):
        world.step()
    for _ in range(8):
        t2_frames.append(capture(robot_pos[:2]))
    for _ in range(10):
        t2_frames.append(t2_frames[-1].copy())

    out_t2 = MEDIA_DIR / "t2_isaac_door.gif"
    encode_frames_to_gif(t2_frames, out_t2, fps=10)
    print(f"  Saved: {out_t2} ({len(t2_frames)} frames, {out_t2.stat().st_size // 1024}KB)", flush=True)

    print(f"\nDone!\n  {out_t3}\n  {out_t2}", flush=True)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
