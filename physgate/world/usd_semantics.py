"""Build a symbolic Scene from the Isaac Sim USD stage (C13).

Reads prims + their semantic labels (UsdSemantics.LabelsAPI, applied by
FetchSimWorld at scene creation) from one env's namespace and reconstructs the
same flat object-list Scene the planner and L3 gate consume.

This closes the perception loop for the sim side of the dual-backend design:
the symbolic world state is *derived from* the simulation, not hand-authored.
(On the real robot the same schema would be produced by GroundingDINO + depth —
architecture doc §5.)

IMPORTANT: import only after SimulationApp launch.
"""

from __future__ import annotations

import numpy as np

from physgate.gate.schemas import Scene, SceneObject
from physgate.world.fetch_scene import BOX_SIZE, SHELF_TOP_Z

#: semantic label -> affordances (the planner needs these to choose actions)
LABEL_AFFORDANCES: dict[str, list[str]] = {
    "cardboard_box": ["graspable"],
    "shelf": ["placeable"],
    "pillar": [],
    "waypoint": [],
}

#: semantic label -> symbolic object id used by plans / the demo scene
LABEL_TO_OBJECT_ID: dict[str, str] = {
    "cardboard_box": "box_03",
    "shelf": "shelf_A",
    "pillar": "obstacle_P",
    "waypoint": "waypoint_W",
}


def _read_prim_labels(prim) -> list[str]:
    """Read semantic labels from a prim, tolerating both Isaac 5.x and legacy APIs."""
    labels: list[str] = []
    # modern API (UsdSemantics.LabelsAPI) via Isaac Lab helper
    try:
        from isaaclab.sim.utils.semantics import get_labels

        for label_list in get_labels(prim).values():
            labels.extend(label_list)
    except Exception:  # noqa: BLE001 — fall through to legacy scan
        pass
    if labels:
        return labels
    # legacy Semantics API: attributes named semantic:<instance>:params:semanticData
    for attr in prim.GetAttributes():
        if attr.GetName().endswith("params:semanticData"):
            value = attr.Get()
            if value:
                labels.append(str(value))
    return labels


def _world_position(prim) -> np.ndarray:
    """World-space translation of a prim."""
    from pxr import UsdGeom

    xform = UsdGeom.Xformable(prim)
    matrix = xform.ComputeLocalToWorldTransform(0.0)
    translation = matrix.ExtractTranslation()
    return np.array([translation[0], translation[1], translation[2]])


def scene_from_stage(
    stage=None,
    env_index: int = 0,
    gripper_empty: bool = True,
) -> Scene:
    """Reconstruct the symbolic Scene from the USD stage of one env.

    Args:
        stage: USD stage (default: the current Isaac stage).
        env_index: which env clone's namespace to read (default env 0).
        gripper_empty: gripper state (the stage does not encode it).
    """
    if stage is None:
        from isaaclab.sim.utils.stage import get_current_stage

        stage = get_current_stage()

    env_ns = f"/World/envs/env_{env_index}"
    env_prim = stage.GetPrimAtPath(env_ns)
    if not env_prim or not env_prim.IsValid():
        raise ValueError(f"no env prim found at {env_ns}")

    env_origin = _world_position(env_prim)

    objects: list[SceneObject] = [
        SceneObject(id="floor_01", label="floor"),
        SceneObject(id="go2", label="robot"),
    ]
    relations: list[tuple[str, str, str]] = []
    positions: dict[str, np.ndarray] = {}

    # walk the env's children; any prim with a semantic label becomes a SceneObject
    from pxr import Usd

    # fallback when semantic labels are unavailable (e.g. Replicator API not
    # enabled and the legacy Semantics module is absent): our scene prims have
    # semantic names by construction, so prim names map to labels directly
    PRIM_NAME_FALLBACK_LABELS = {
        "Box": "cardboard_box",
        "Shelf": "shelf",
        "Obstacle": "pillar",
        "Waypoint": "waypoint",
    }

    for prim in Usd.PrimRange(env_prim):
        labels = _read_prim_labels(prim)
        if not labels and prim.GetName() in PRIM_NAME_FALLBACK_LABELS:
            labels = [PRIM_NAME_FALLBACK_LABELS[prim.GetName()]]
        if not labels:
            continue
        label = labels[0]
        object_id = LABEL_TO_OBJECT_ID.get(label, prim.GetName().lower())
        if any(o.id == object_id for o in objects):
            continue
        local_pos = _world_position(prim) - env_origin
        positions[object_id] = local_pos
        objects.append(
            SceneObject(
                id=object_id,
                label=label,
                affordances=LABEL_AFFORDANCES.get(label, []),
                # the box is anomalous when it is NOT on the shelf
                is_anomaly=(
                    label == "cardboard_box"
                    and abs(local_pos[2] - (SHELF_TOP_Z + BOX_SIZE[2] / 2)) > 0.1
                ),
            )
        )

    # derive "on" relations from geometry
    if "box_03" in positions:
        box_pos = positions["box_03"]
        if abs(box_pos[2] - (SHELF_TOP_Z + BOX_SIZE[2] / 2)) <= 0.1:
            relations.append(("box_03", "on", "shelf_A"))
        elif box_pos[2] <= BOX_SIZE[2] / 2 + 0.1:
            relations.append(("box_03", "on", "floor_01"))
    if "shelf_A" in positions and ("box_03", "on", "shelf_A") not in relations:
        relations.append(("shelf_A", "unoccupied", "shelf_A"))

    return Scene(objects=objects, relations=relations, gripper_empty=gripper_empty)
