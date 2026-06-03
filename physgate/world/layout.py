"""Physical scene layout for the fetch-and-place task (pure logic — no Isaac).

Single source of truth for entity positions and sizes, shared by:

* the navigation planner (``physgate/nav/path_planner.py``) — obstacle footprints,
* trajectory/mission compilation (``physgate/gate/trajectory.py``),
* the Isaac scene definition (``physgate/world/fetch_scene.py``) — prim spawn poses,
* the executor (``physgate/executor/sim_backend.py``).

This module is importable WITHOUT Isaac Sim so that navigation and trajectory
logic can be unit-tested in the pure-logic venv.

REBUILD.md Phase 1 note: the former ``waypoint_W`` entry is gone. It was a
physics hack masquerading as a semantic object — obstacle avoidance now lives
in the deterministic navigation layer, not in the scene.
"""

from __future__ import annotations

#: World-frame positions (per env, relative to env origin) for every scene entity.
SCENE_LAYOUT: dict[str, tuple[float, float, float]] = {
    "go2": (0.0, 0.0, 0.40),
    "box_03": (1.5, 0.5, 0.10),
    "shelf_A": (3.0, -1.0, 0.25),
    "obstacle_P": (2.2, -0.2, 0.40),
    "floor_01": (0.0, 0.0, 0.0),
}

BOX_SIZE = (0.2, 0.2, 0.2)
SHELF_SIZE = (0.8, 0.4, 0.5)        # top surface at z = 0.5
OBSTACLE_SIZE = (0.3, 0.3, 0.8)
ROBOT_BASE_HEIGHT = 0.40            # Go2 standing base height
#: Carried box position relative to the robot base. The carry is BOOKKEEPING,
#: not physics: there is no gripper articulation in the MVP, so the box is
#: kinematically written each step. It must therefore stay clear of ANY possible
#: robot posture — a kinematically written box that contacts the dynamic robot
#: acts as an immovable obstacle and crushes/stalls it (D-018). 0.6 m overhead
#: clears the trunk/head at full gait pitch and also clears the shelf (0.5 m)
#: and pillar (0.8 m) tops during transport. What IS physical about carrying:
#: the box leaves its original location, and is released WITH momentum at
#: placement (the physics that decides placement success).
CARRY_OFFSET = (0.0, 0.0, 0.60)
SHELF_TOP_Z = SCENE_LAYOUT["shelf_A"][2] + SHELF_SIZE[2] / 2

#: Go2 half-width + margin: used for swept-path checks AND navigation clearance.
ROBOT_COLLISION_RADIUS = 0.30

#: Static scene geometry the robot must navigate around (2D footprints).
#: This INCLUDES large manipulation targets (the shelf): the robot routes around
#: them and stops at a standoff outside their footprint. Only small, dynamic,
#: graspable objects (the box) are excluded — the robot must get right next to
#: those to manipulate them.
STATIC_FOOTPRINTS: dict[str, tuple[float, float, float]] = {
    "obstacle_P": OBSTACLE_SIZE,
    "shelf_A": SHELF_SIZE,
}

#: Backwards-compatible alias (pre-rebuild name).
OBSTACLE_FOOTPRINTS = STATIC_FOOTPRINTS

#: Semantic labels applied to prims (read back by world/usd_semantics.py).
SEMANTIC_LABELS: dict[str, str] = {
    "box_03": "cardboard_box",
    "shelf_A": "shelf",
    "obstacle_P": "pillar",
}


def navigation_obstacles(
    layout: dict[str, tuple[float, float, float]] | None = None,
) -> list:
    """Build the navigation Obstacle list from the scene layout.

    Returns:
        ``list[physgate.nav.path_planner.Obstacle]`` for every entry in
        :data:`STATIC_FOOTPRINTS` present in the layout.
    """
    from physgate.nav.path_planner import Obstacle

    lay = layout or SCENE_LAYOUT
    return [
        Obstacle(
            center_xy=(lay[entity_id][0], lay[entity_id][1]),
            half_extents_xy=(size[0] / 2, size[1] / 2),
        )
        for entity_id, size in STATIC_FOOTPRINTS.items()
        if entity_id in lay
    ]


def target_half_extents(target_id: str) -> tuple[float, float] | None:
    """2D footprint half-extents of a navigation target, or None for point targets.

    Footprint targets (the shelf) get their standoff measured from the footprint
    EDGE; point targets (the box) from their center.
    """
    size = STATIC_FOOTPRINTS.get(target_id)
    return (size[0] / 2, size[1] / 2) if size else None
