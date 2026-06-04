"""Extended Isaac Lab scenes for T2-T5 task types.

Builds on the base FetchSimWorld infrastructure, adding:
- Pushable crate (T3: blocked-path clearance)
- Door with hinge joint + key object (T2: locked-room delivery)
- Elevator platform (T5: cross-floor delivery)

Each scene is a standalone function that returns a configured SimWorld
with the additional prims spawned after the base scene is built.

IMPORTANT: import only after SimulationApp launch.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, AssetBaseCfg

from physgate.world.fetch_scene import FetchSimWorld, FetchSceneCfg, get_shared_world
from physgate.world.layout import (
    BOX_SIZE,
    OBSTACLE_SIZE,
    ROBOT_BASE_HEIGHT,
    SCENE_LAYOUT,
    SHELF_SIZE,
)

# ── T3: Blocked-path scene with pushable crate ─────────────────────────

CRATE_SIZE = (0.35, 0.35, 0.35)
CRATE_MASS_KG = 8.0
CRATE_POS = (2.0, -0.3, 0.175)  # between box and shelf, on the floor

# ── T2: Door scene with key ────────────────────────────────────────────

DOOR_SIZE = (0.1, 1.0, 1.2)  # thin panel
DOOR_POS = (2.5, -0.5, 0.6)  # blocks path to shelf
KEY_SIZE = (0.08, 0.04, 0.02)
KEY_POS = (0.8, 0.8, 0.10)  # on the floor near start


def spawn_crate(world: FetchSimWorld, env_idx: int = 0) -> None:
    """Spawn a pushable crate into env 0 of an existing FetchSimWorld."""
    from pxr import UsdGeom, UsdPhysics, Gf

    stage = world.sim.stage
    origin = world.env_origins[env_idx].cpu().numpy()

    prim_path = f"/World/envs/env_{env_idx}/Crate"
    cube_geom = UsdGeom.Cube.Define(stage, prim_path)
    cube_geom.GetSizeAttr().Set(1.0)
    xform = UsdGeom.Xformable(cube_geom.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(
        float(CRATE_POS[0] + origin[0]),
        float(CRATE_POS[1] + origin[1]),
        float(CRATE_POS[2] + origin[2]),
    ))
    xform.AddScaleOp().Set(Gf.Vec3f(
        float(CRATE_SIZE[0]),
        float(CRATE_SIZE[1]),
        float(CRATE_SIZE[2]),
    ))

    prim = cube_geom.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.GetMassAttr().Set(float(CRATE_MASS_KG))

    # visual: brown/orange crate color
    mat_path = f"{prim_path}/CrateMaterial"
    mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.35, 0.1))
    mat.func(mat_path, mat)
    UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([(0.6, 0.35, 0.1)])


def spawn_door_and_key(world: FetchSimWorld, env_idx: int = 0) -> None:
    """Spawn a door panel and a key object into env 0."""
    from pxr import UsdGeom, UsdPhysics, Gf

    stage = world.sim.stage
    origin = world.env_origins[env_idx].cpu().numpy()

    # Door: static collision panel (locked = immovable until "unlocked")
    door_path = f"/World/envs/env_{env_idx}/Door"
    door_geom = UsdGeom.Cube.Define(stage, door_path)
    door_geom.GetSizeAttr().Set(1.0)
    xform = UsdGeom.Xformable(door_geom.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(
        float(DOOR_POS[0] + origin[0]),
        float(DOOR_POS[1] + origin[1]),
        float(DOOR_POS[2] + origin[2]),
    ))
    xform.AddScaleOp().Set(Gf.Vec3f(
        float(DOOR_SIZE[0]),
        float(DOOR_SIZE[1]),
        float(DOOR_SIZE[2]),
    ))
    UsdPhysics.CollisionAPI.Apply(door_geom.GetPrim())
    UsdGeom.Gprim(door_geom.GetPrim()).GetDisplayColorAttr().Set([(0.4, 0.25, 0.1)])

    # Key: small dynamic rigid body
    key_path = f"/World/envs/env_{env_idx}/Key"
    key_geom = UsdGeom.Cube.Define(stage, key_path)
    key_geom.GetSizeAttr().Set(1.0)
    xform_k = UsdGeom.Xformable(key_geom.GetPrim())
    xform_k.ClearXformOpOrder()
    xform_k.AddTranslateOp().Set(Gf.Vec3d(
        float(KEY_POS[0] + origin[0]),
        float(KEY_POS[1] + origin[1]),
        float(KEY_POS[2] + origin[2]),
    ))
    xform_k.AddScaleOp().Set(Gf.Vec3f(
        float(KEY_SIZE[0]),
        float(KEY_SIZE[1]),
        float(KEY_SIZE[2]),
    ))
    UsdPhysics.RigidBodyAPI.Apply(key_geom.GetPrim())
    UsdPhysics.CollisionAPI.Apply(key_geom.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(key_geom.GetPrim())
    mass_api.GetMassAttr().Set(0.1)
    UsdGeom.Gprim(key_geom.GetPrim()).GetDisplayColorAttr().Set([(0.9, 0.8, 0.1)])


def remove_prim(world: FetchSimWorld, prim_path: str) -> None:
    """Remove a prim from the stage (used for door opening animation)."""
    stage = world.sim.stage
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)
