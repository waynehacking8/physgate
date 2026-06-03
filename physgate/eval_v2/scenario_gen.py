"""E3 procedural scenario generation with certified ground-truth labels.

Layouts are sampled inside the workspace and labeled BY CONSTRUCTION
(PlanBench's approach — no manual annotation):

* feasible instances — the A* planner certifies that a collision-free route
  exists robot -> box AND box -> shelf,
* infeasible instances — the box is placed inside the pillar's inflated
  footprint, and the A* planner certifies that NO standoff route exists.

The generator never trusts its own sampling: every emitted instance is
re-checked through the same navigation code the gate itself uses. If sampling
cannot satisfy the constraints, generation fails loudly instead of silently
emitting unlabeled instances.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from physgate.gate.schemas import Scene, SceneObject
from physgate.nav.path_planner import PathPlannerError, plan_standoff_route
from physgate.world.layout import (
    ROBOT_COLLISION_RADIUS,
    navigation_obstacles,
    target_half_extents,
)

#: Minimum center-to-center separation between scene entities (m).
MIN_SEPARATION_M = 1.2
#: Default workspace half-extents the entities are sampled in (m).
DEFAULT_WORKSPACE = (4.0, 3.0)
#: Sampling attempts per instance before giving up.
MAX_ATTEMPTS = 500


@dataclass(frozen=True)
class TaskInstance:
    """One generated task instance with its certified ground-truth label."""

    instance_id: str
    layout: dict[str, tuple[float, float, float]]
    scene: Scene
    #: True = the task is achievable (certified by A*); False = certified unreachable.
    feasible: bool


# ------------------------------------------------------------------ sampling


def _route_exists(layout: dict, frm: str, to: str, standoff: float) -> bool:
    try:
        plan_standoff_route(
            tuple(layout[frm][:2]),
            tuple(layout[to][:2]),
            standoff=standoff,
            obstacles=navigation_obstacles(layout),
            robot_radius=ROBOT_COLLISION_RADIUS,
            target_half_extents=target_half_extents(to),
        )
        return True
    except PathPlannerError:
        return False


def _min_pairwise_distance(layout: dict, entities: list[str]) -> float:
    best = math.inf
    for i, a in enumerate(entities):
        for b in entities[i + 1 :]:
            ax, ay = layout[a][:2]
            bx, by = layout[b][:2]
            best = min(best, math.hypot(ax - bx, ay - by))
    return best


def _sample_layout(rng: random.Random, workspace: tuple[float, float]) -> dict:
    """One random layout: robot at origin, entities sampled in the workspace."""
    width, height = workspace

    def _sample_xy() -> tuple[float, float]:
        # keep entities away from the robot spawn (origin) by at least 1 m
        while True:
            x = rng.uniform(0.8, width)
            y = rng.uniform(-height / 2, height / 2)
            if math.hypot(x, y) >= 1.0:
                return (round(x, 2), round(y, 2))

    box = _sample_xy()
    shelf = _sample_xy()
    obstacle = _sample_xy()
    return {
        "go2": (0.0, 0.0, 0.40),
        "box_03": (*box, 0.10),
        "shelf_A": (*shelf, 0.25),
        "obstacle_P": (*obstacle, 0.40),
        "floor_01": (0.0, 0.0, 0.0),
    }


def _build_scene(feasible: bool) -> Scene:
    """The symbolic scene matching a generated layout (what perception reports)."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="obstacle_P", label="pillar"),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[("box_03", "on", "floor_01"), ("shelf_A", "unoccupied", "shelf_A")],
        gripper_empty=True,
    )


# ----------------------------------------------------------------- instances


def _generate_feasible(rng: random.Random, workspace: tuple[float, float], index: int) -> TaskInstance:
    entities = ["box_03", "shelf_A", "obstacle_P"]
    for _ in range(MAX_ATTEMPTS):
        layout = _sample_layout(rng, workspace)
        if _min_pairwise_distance(layout, entities) < MIN_SEPARATION_M:
            continue
        # certification: both legs of the task must be solvable
        if not _route_exists(layout, "go2", "box_03", standoff=0.3):
            continue
        if not _route_exists(layout, "box_03", "shelf_A", standoff=0.4):
            continue
        return TaskInstance(
            instance_id=f"gen_feasible_{index:03d}",
            layout=layout,
            scene=_build_scene(feasible=True),
            feasible=True,
        )
    raise ValueError(
        f"could not sample a feasible layout in {MAX_ATTEMPTS} attempts — "
        f"workspace {workspace} too small for separation {MIN_SEPARATION_M} m"
    )


def _generate_infeasible(
    rng: random.Random, workspace: tuple[float, float], index: int
) -> TaskInstance:
    """The box sits inside the pillar's inflated footprint -> no reachable standoff."""
    for _ in range(MAX_ATTEMPTS):
        layout = _sample_layout(rng, workspace)
        # co-locate the box with the obstacle (inside its inflated clearance)
        ox, oy, _ = layout["obstacle_P"]
        layout = {**layout, "box_03": (ox, oy, 0.10)}
        # keep the shelf clear of the obstacle so ONLY the box leg is impossible
        if _min_pairwise_distance(layout, ["shelf_A", "obstacle_P"]) < MIN_SEPARATION_M:
            continue
        # certification: the box must be UNREACHABLE
        if _route_exists(layout, "go2", "box_03", standoff=0.3):
            continue
        return TaskInstance(
            instance_id=f"gen_infeasible_{index:03d}",
            layout=layout,
            scene=_build_scene(feasible=False),
            feasible=False,
        )
    raise ValueError(
        f"could not construct an infeasible layout in {MAX_ATTEMPTS} attempts "
        f"(workspace {workspace})"
    )


def generate_instances(
    n_feasible: int,
    n_infeasible: int,
    seed: int,
    workspace: tuple[float, float] = DEFAULT_WORKSPACE,
) -> list[TaskInstance]:
    """Generate a labeled instance set: ``n_feasible`` solvable + ``n_infeasible``
    certified-unreachable task layouts.

    Deterministic per seed (same seed -> identical instances).
    """
    if workspace[0] < MIN_SEPARATION_M or workspace[1] < MIN_SEPARATION_M:
        raise ValueError(
            f"workspace {workspace} is smaller than the minimum entity separation "
            f"({MIN_SEPARATION_M} m) — no layout can satisfy the constraints"
        )

    rng = random.Random(seed)
    instances: list[TaskInstance] = []
    for i in range(n_feasible):
        instances.append(_generate_feasible(rng, workspace, i))
    for i in range(n_infeasible):
        instances.append(_generate_infeasible(rng, workspace, i))
    return instances
