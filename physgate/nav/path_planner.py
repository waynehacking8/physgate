"""Deterministic A* occupancy-grid path planner (the Nav2 boundary).

REBUILD.md Phase 1: obstacle avoidance belongs in the LOW (deterministic) level,
not in the LLM. Every ``move_to_pose`` is expanded through :func:`plan_path`, so
navigation ALWAYS routes around obstacles — reproducibly, independent of what
the planner LLM happened to emit.

Interface contract (kept Nav2-compatible on purpose):

    plan_path(start_xy, goal_xy, obstacles) -> list[(x, y)]

mirrors a ROS 2 Nav2 global planner (``ComputePathToPose`` over a costmap). A
Nav2-backed implementation can replace this module behind the same signature
when the ROS 2 executor lands (see DECISIONS.md D-017 for the trade-off).

Clearance model (mirrors a Nav2 costmap):

* hard clearance  = robot radius + safety margin — never violated anywhere,
* tracking inflation = extra clearance for paths followed by an imperfect
  waypoint controller (the walking policy cuts corners by up to its arrival
  tolerance), applied everywhere EXCEPT a small exemption zone around the
  start and goal (so targets standing close to obstacles stay reachable —
  Nav2's goal-tolerance behaviour).

Pure logic: stdlib only — no GPU, no Isaac, no ROS.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable

#: Default grid resolution (m per cell). Fine enough that grid quantization
#: cannot eat into the clearance margin.
DEFAULT_RESOLUTION = 0.05
#: Default robot clearance radius (Go2 half-width + margin) — must match
#: layout.ROBOT_COLLISION_RADIUS so planned paths pass the swept-path check.
DEFAULT_ROBOT_RADIUS = 0.30
#: Extra clearance beyond the robot radius, absorbing grid quantization.
SAFETY_MARGIN = 0.10
#: Free space padded around the bounding box of start/goal/obstacles.
WORKSPACE_PADDING = 1.5
#: How far a waypoint-following controller may deviate from the planned path
#: (the WaypointNavigator's arrival tolerance). Paths followed by the walking
#: policy are planned with this extra clearance so corner-cutting between
#: waypoints can never push the robot into an obstacle (Nav2 costmap-inflation
#: equivalent).
WAYPOINT_TRACKING_TOLERANCE = 0.35


class PathPlannerError(Exception):
    """No collision-free path exists between start and goal."""


@dataclass(frozen=True)
class Obstacle:
    """Axis-aligned rectangular obstacle footprint in the ground plane."""

    center_xy: tuple[float, float]
    half_extents_xy: tuple[float, float]

    def clearance_to(self, point_xy: tuple[float, float]) -> float:
        """Chebyshev-style clearance: how far outside the footprint the point is.

        Positive = outside (distance to the nearest face along the worst axis),
        negative = inside.
        """
        dx = abs(point_xy[0] - self.center_xy[0]) - self.half_extents_xy[0]
        dy = abs(point_xy[1] - self.center_xy[1]) - self.half_extents_xy[1]
        return max(dx, dy)


#: A point-occupancy predicate: True when the point is traversable.
ClearFn = Callable[[tuple[float, float]], bool]


# ----------------------------------------------------------------- primitives


def _point_clear(point_xy: tuple[float, float], obstacles: list[Obstacle], clearance: float) -> bool:
    return all(o.clearance_to(point_xy) >= clearance for o in obstacles)


def _segment_clear_fn(
    a_xy: tuple[float, float],
    b_xy: tuple[float, float],
    clear_fn: ClearFn,
    sample_step: float = 0.02,
) -> bool:
    """True when every densely-sampled point of segment a->b is traversable."""
    dist = math.hypot(b_xy[0] - a_xy[0], b_xy[1] - a_xy[1])
    n = max(int(dist / sample_step), 1)
    for k in range(n + 1):
        t = k / n
        point = (a_xy[0] + (b_xy[0] - a_xy[0]) * t, a_xy[1] + (b_xy[1] - a_xy[1]) * t)
        if not clear_fn(point):
            return False
    return True


def segment_clear(
    a_xy: tuple[float, float],
    b_xy: tuple[float, float],
    obstacles: list[Obstacle],
    clearance: float,
    sample_step: float = 0.02,
) -> bool:
    """True when the straight segment a->b keeps ``clearance`` from every obstacle.

    Uses dense sampling (deterministic) rather than exact segment-AABB algebra:
    at 2 cm steps the worst-case error is far below the SAFETY_MARGIN the
    planner already adds on top of the robot radius.
    """
    return _segment_clear_fn(
        a_xy, b_xy, lambda p: _point_clear(p, obstacles, clearance), sample_step
    )


def _make_clear_fn(
    start: tuple[float, float],
    goal: tuple[float, float],
    obstacles: list[Obstacle],
    robot_radius: float,
    tracking_error: float,
) -> ClearFn:
    """Build the occupancy predicate encoding the two-tier clearance model.

    * everywhere:                 robot_radius + SAFETY_MARGIN + tracking_error
    * near the start or goal:     robot_radius + SAFETY_MARGIN  (hard minimum)

    The exemption zone keeps goals that stand close to obstacles reachable; the
    final approach is short and ends at a stationary point, so corner-cutting
    cannot occur there.
    """
    hard = robot_radius + SAFETY_MARGIN
    full = hard + tracking_error
    # the exemption zone must be large enough that a point leaving it (walking
    # radially away from the obstacle) has already gained full clearance
    exempt_radius = tracking_error + 2 * SAFETY_MARGIN

    def clear_fn(point: tuple[float, float]) -> bool:
        """Return True if the point satisfies the two-tier clearance model."""
        required = full
        if tracking_error > 0.0 and (
            math.hypot(point[0] - start[0], point[1] - start[1]) <= exempt_radius
            or math.hypot(point[0] - goal[0], point[1] - goal[1]) <= exempt_radius
        ):
            required = hard
        return _point_clear(point, obstacles, required)

    return clear_fn


# ------------------------------------------------------------------------- A*


def _astar_grid(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    obstacles: list[Obstacle],
    clear_fn: ClearFn,
    resolution: float,
) -> list[tuple[float, float]]:
    """8-connected A* over an occupancy grid. Returns the raw cell-center path."""
    # workspace bounds: bounding box of everything involved, padded
    xs = [start_xy[0], goal_xy[0]] + [
        o.center_xy[0] + s * o.half_extents_xy[0] for o in obstacles for s in (-1, 1)
    ]
    ys = [start_xy[1], goal_xy[1]] + [
        o.center_xy[1] + s * o.half_extents_xy[1] for o in obstacles for s in (-1, 1)
    ]
    min_x, max_x = min(xs) - WORKSPACE_PADDING, max(xs) + WORKSPACE_PADDING
    min_y, max_y = min(ys) - WORKSPACE_PADDING, max(ys) + WORKSPACE_PADDING

    cols = int((max_x - min_x) / resolution) + 1
    rows = int((max_y - min_y) / resolution) + 1

    def to_cell(p: tuple[float, float]) -> tuple[int, int]:
        """Convert a world-space point to its nearest grid cell."""
        return (
            int(round((p[0] - min_x) / resolution)),
            int(round((p[1] - min_y) / resolution)),
        )

    def to_point(c: tuple[int, int]) -> tuple[float, float]:
        """Convert a grid cell back to world-space coordinates."""
        return (min_x + c[0] * resolution, min_y + c[1] * resolution)

    def cell_free(c: tuple[int, int]) -> bool:
        """Return True if the cell is within bounds and traversable."""
        if not (0 <= c[0] < cols and 0 <= c[1] < rows):
            return False
        return clear_fn(to_point(c))

    start_cell, goal_cell = to_cell(start_xy), to_cell(goal_xy)
    snap_cells = int(DEFAULT_ROBOT_RADIUS / resolution) + 2
    if not cell_free(goal_cell):
        goal_cell = _nearest_free_cell(goal_cell, cell_free, max_cells=snap_cells)
        if goal_cell is None:
            raise PathPlannerError(
                f"goal {goal_xy} has no free grid cell (inside an inflated obstacle?)"
            )
    if not cell_free(start_cell):
        start_cell = _nearest_free_cell(start_cell, cell_free, max_cells=snap_cells)
        if start_cell is None:
            raise PathPlannerError(f"start {start_xy} is inside an inflated obstacle")

    # 8-connected moves with octile costs; fixed order keeps tie-breaking deterministic
    moves = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
    ]

    def heuristic(c: tuple[int, int]) -> float:
        """Compute the octile distance heuristic from cell to goal."""
        dx, dy = abs(c[0] - goal_cell[0]), abs(c[1] - goal_cell[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    # (f, tie_counter, cell): the counter makes heap order fully deterministic
    open_heap: list[tuple[float, int, tuple[int, int]]] = [(heuristic(start_cell), 0, start_cell)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}
    counter = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal_cell:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            cells.reverse()
            return [to_point(c) for c in cells]

        for dx, dy, cost in moves:
            neighbor = (current[0] + dx, current[1] + dy)
            if not cell_free(neighbor):
                continue
            tentative = g_score[current] + cost
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                counter += 1
                heapq.heappush(open_heap, (tentative + heuristic(neighbor), counter, neighbor))

    raise PathPlannerError(f"no path from {start_xy} to {goal_xy} (goal unreachable)")


def _nearest_free_cell(cell, cell_free, max_cells: int):
    """Spiral outward (deterministic order) to find the nearest free cell."""
    for radius in range(1, max_cells + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                candidate = (cell[0] + dx, cell[1] + dy)
                if cell_free(candidate):
                    return candidate
    return None


def _simplify(
    path: list[tuple[float, float]], clear_fn: ClearFn
) -> list[tuple[float, float]]:
    """Greedy line-of-sight simplification (string pulling): keep only the
    waypoints where the path must actually turn to stay traversable."""
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    anchor_idx = 0
    while anchor_idx < len(path) - 1:
        # furthest point reachable in a straight traversable line from the anchor
        next_idx = anchor_idx + 1
        for candidate_idx in range(len(path) - 1, anchor_idx, -1):
            if _segment_clear_fn(path[anchor_idx], path[candidate_idx], clear_fn):
                next_idx = candidate_idx
                break
        simplified.append(path[next_idx])
        anchor_idx = next_idx
    return simplified


def _merge_close_waypoints(
    path: list[tuple[float, float]], min_spacing: float
) -> list[tuple[float, float]]:
    """Drop intermediate waypoints closer than ``min_spacing`` to their predecessor.

    String pulling can leave corner clusters only centimetres apart where the
    path hugs a clearance boundary. A waypoint follower whose arrival tolerance
    is >= min_spacing cannot distinguish them — it overshoots one, turns back
    for the next, and oscillates without net progress. Merging them cuts corners
    by < min_spacing, which the tracking-error clearance inflation already
    absorbs, so the merged path is still safe.

    The first and last points are always kept exactly.
    """
    if len(path) <= 2 or min_spacing <= 0.0:
        return path
    merged = [path[0]]
    for point in path[1:-1]:
        if math.hypot(point[0] - merged[-1][0], point[1] - merged[-1][1]) >= min_spacing:
            merged.append(point)
    # the final goal is exact; drop a second-to-last corner that crowds it
    last = path[-1]
    if len(merged) > 1 and math.hypot(last[0] - merged[-1][0], last[1] - merged[-1][1]) < min_spacing:
        merged.pop()
    merged.append(last)
    return merged


# ------------------------------------------------------------------ public API


def plan_path(
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    obstacles: list[Obstacle],
    robot_radius: float = DEFAULT_ROBOT_RADIUS,
    resolution: float = DEFAULT_RESOLUTION,
    tracking_error: float = 0.0,
) -> list[tuple[float, float]]:
    """Plan a collision-free path from start to goal around the obstacles.

    Args:
        start_xy: starting position in the ground plane.
        goal_xy: goal position.
        obstacles: static obstacle footprints.
        robot_radius: clearance kept from every obstacle (plus a safety margin).
        resolution: occupancy-grid cell size.
        tracking_error: extra clearance for imperfect path followers (e.g. the
            walking policy's waypoint arrival tolerance). Kinematic followers
            use 0. The region right around start/goal is exempted so targets
            near obstacles stay reachable.

    Returns:
        Waypoints from start to goal (both included). A clear straight line
        gives exactly ``[start, goal]``; detours insert intermediate waypoints.

    Raises:
        PathPlannerError: when no collision-free path exists.
    """
    start = (float(start_xy[0]), float(start_xy[1]))
    goal = (float(goal_xy[0]), float(goal_xy[1]))

    if math.hypot(goal[0] - start[0], goal[1] - start[1]) < 1e-9:
        return [start]

    clear_fn = _make_clear_fn(start, goal, obstacles, robot_radius, tracking_error)

    # fast path: the straight segment is already traversable
    if _segment_clear_fn(start, goal, clear_fn):
        return [start, goal]

    cells = _astar_grid(start, goal, obstacles, clear_fn, resolution)
    # snap the endpoints back to the exact requested positions
    cells[0] = start
    cells[-1] = goal
    simplified = _simplify(cells, clear_fn)
    # corner clusters closer than the follower's tracking tolerance cause
    # overshoot oscillation — merge them (the clearance budget absorbs the cut)
    return _merge_close_waypoints(simplified, min_spacing=tracking_error)


def plan_standoff_route(
    start_xy: tuple[float, float],
    target_xy: tuple[float, float],
    standoff: float,
    obstacles: list[Obstacle],
    robot_radius: float = DEFAULT_ROBOT_RADIUS,
    resolution: float = DEFAULT_RESOLUTION,
    tracking_error: float = 0.0,
    target_half_extents: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    """Plan a route that stops ``standoff`` meters short of the target.

    Point targets (``target_half_extents is None``, e.g. a small box): the
    standoff is measured ALONG the planned path from the target center.

    Footprint targets (e.g. a shelf): the goal is a point on the target's
    footprint inflated by ``max(standoff, hard clearance)`` — the robot stops at
    a standoff outside the FOOTPRINT, on whichever face gives the shortest
    collision-free path (Nav2 goal-selection behaviour). The target itself must
    also be in ``obstacles`` so routes never cut through it.

    Returns:
        Waypoints from start to the standoff point. If the start is already
        within the standoff distance, returns ``[start]`` (no motion).
    """
    start = (float(start_xy[0]), float(start_xy[1]))
    target = (float(target_xy[0]), float(target_xy[1]))

    if target_half_extents is None:
        if math.hypot(target[0] - start[0], target[1] - start[1]) <= standoff + 1e-9:
            return [start]
        path = plan_path(
            start,
            target,
            obstacles,
            robot_radius=robot_radius,
            resolution=resolution,
            tracking_error=tracking_error,
        )
        return _trim_path_end(path, standoff)

    return _plan_footprint_standoff(
        start,
        target,
        standoff,
        target_half_extents,
        obstacles,
        robot_radius,
        resolution,
        tracking_error,
    )


def _plan_footprint_standoff(
    start: tuple[float, float],
    target: tuple[float, float],
    standoff: float,
    half_extents: tuple[float, float],
    obstacles: list[Obstacle],
    robot_radius: float,
    resolution: float,
    tracking_error: float,
) -> list[tuple[float, float]]:
    """Route to a standoff point on a footprint target's inflated boundary.

    Candidate goals = closest boundary point to the start + the 4 face
    midpoints; the candidate with the shortest collision-free path wins
    (deterministic: fixed candidate order, strict-improvement comparison).
    """
    hard = robot_radius + SAFETY_MARGIN
    # the goal must sit outside the target's own hard-clearance zone
    inflation = max(standoff, hard + 0.02)
    half_x = half_extents[0] + inflation
    half_y = half_extents[1] + inflation

    # already standing at a valid standoff?
    dx = abs(start[0] - target[0]) - half_x
    dy = abs(start[1] - target[1]) - half_y
    if max(dx, dy) >= -1e-9 and max(dx, dy) <= tracking_error + 1e-9:
        return [start]

    closest = (
        min(max(start[0], target[0] - half_x), target[0] + half_x),
        min(max(start[1], target[1] - half_y), target[1] + half_y),
    )
    candidates = [
        closest,
        (target[0], target[1] + half_y),  # north face
        (target[0], target[1] - half_y),  # south face
        (target[0] + half_x, target[1]),  # east face
        (target[0] - half_x, target[1]),  # west face
    ]

    best_path: list[tuple[float, float]] | None = None
    best_length = float("inf")
    for goal in candidates:
        if not _point_clear(goal, obstacles, hard):
            continue
        try:
            path = plan_path(
                start,
                goal,
                obstacles,
                robot_radius=robot_radius,
                resolution=resolution,
                tracking_error=tracking_error,
            )
        except PathPlannerError:
            continue
        length = sum(
            math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:])
        )
        if length < best_length - 1e-9:
            best_length = length
            best_path = path

    if best_path is None:
        raise PathPlannerError(
            f"no reachable standoff point around target at {target} "
            f"(footprint {half_extents}, standoff {standoff})"
        )
    return best_path


def _trim_path_end(path: list[tuple[float, float]], trim: float) -> list[tuple[float, float]]:
    """Walk back ``trim`` meters along the polyline from its end."""
    if trim <= 1e-9 or len(path) < 2:
        return path
    remaining = trim
    trimmed = list(path)
    while len(trimmed) >= 2:
        a, b = trimmed[-2], trimmed[-1]
        seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg_len >= remaining:
            t = (seg_len - remaining) / seg_len
            trimmed[-1] = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            return trimmed
        remaining -= seg_len
        trimmed.pop()
    return trimmed
