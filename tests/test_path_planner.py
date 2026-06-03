"""Tests for the deterministic A* occupancy-grid path planner (REBUILD.md Phase 1).

The path planner is the LOW level of the dual-system architecture: the agent
emits semantic skills (move_to_pose(<object id>)); this module guarantees
obstacle avoidance deterministically. Its interface mirrors a ROS 2 Nav2 global
planner (compute_path_to_pose) so Nav2 can drop in later.

Pure logic — no GPU, no Isaac.
"""

from __future__ import annotations

import math

import pytest

from physgate.nav.path_planner import (
    Obstacle,
    PathPlannerError,
    plan_path,
    plan_standoff_route,
    segment_clear,
)
from physgate.world.layout import OBSTACLE_SIZE, SCENE_LAYOUT, navigation_obstacles

#: The demo scene's pillar, as the planner sees it.
PILLAR = Obstacle(
    center_xy=(SCENE_LAYOUT["obstacle_P"][0], SCENE_LAYOUT["obstacle_P"][1]),
    half_extents_xy=(OBSTACLE_SIZE[0] / 2, OBSTACLE_SIZE[1] / 2),
)

BOX_XY = SCENE_LAYOUT["box_03"][:2]
SHELF_XY = SCENE_LAYOUT["shelf_A"][:2]
ROBOT_RADIUS = 0.30


def _point_clear_of(point_xy, obstacle: Obstacle, clearance: float) -> bool:
    """True when the point is at least `clearance` outside the obstacle footprint."""
    dx = abs(point_xy[0] - obstacle.center_xy[0]) - obstacle.half_extents_xy[0]
    dy = abs(point_xy[1] - obstacle.center_xy[1]) - obstacle.half_extents_xy[1]
    return max(dx, dy) >= clearance


def _densify(path: list[tuple[float, float]], step: float = 0.02) -> list[tuple[float, float]]:
    """Interpolate a polyline densely (mimics the kinematic trajectory sweep)."""
    points: list[tuple[float, float]] = []
    for a, b in zip(path[:-1], path[1:]):
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(int(dist / step), 1)
        for k in range(n):
            t = k / n
            points.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    points.append(path[-1])
    return points


# ----------------------------------------------------------------- basic paths


def test_no_obstacle_gives_straight_path():
    path = plan_path((0.0, 0.0), (2.0, 0.0), obstacles=[])
    assert path[0] == pytest.approx((0.0, 0.0))
    assert path[-1] == pytest.approx((2.0, 0.0))
    # nothing to avoid -> direct segment, no intermediate detour points
    assert len(path) == 2


def test_clear_segment_stays_straight_even_with_far_obstacle():
    far_obstacle = Obstacle(center_xy=(10.0, 10.0), half_extents_xy=(0.5, 0.5))
    path = plan_path((0.0, 0.0), (2.0, 0.0), obstacles=[far_obstacle])
    assert len(path) == 2


def test_start_equals_goal():
    path = plan_path((1.0, 1.0), (1.0, 1.0), obstacles=[PILLAR])
    assert path == [pytest.approx((1.0, 1.0))]


# ------------------------------------------------------------ obstacle routing


def test_path_routes_around_obstacle():
    """The box -> shelf straight line passes exactly through the pillar; the
    planned path must detour around it with full robot clearance."""
    path = plan_path(BOX_XY, SHELF_XY, obstacles=[PILLAR], robot_radius=ROBOT_RADIUS)

    assert path[0] == pytest.approx(BOX_XY)
    assert path[-1] == pytest.approx(SHELF_XY)
    assert len(path) > 2  # a detour needs at least one intermediate waypoint

    # every densely-interpolated point keeps robot-radius clearance
    for point in _densify(path):
        assert _point_clear_of(point, PILLAR, ROBOT_RADIUS), (
            f"path point {point} is within robot radius of the pillar"
        )


def test_path_is_deterministic():
    a = plan_path(BOX_XY, SHELF_XY, obstacles=[PILLAR], robot_radius=ROBOT_RADIUS)
    b = plan_path(BOX_XY, SHELF_XY, obstacles=[PILLAR], robot_radius=ROBOT_RADIUS)
    assert a == b


def test_path_avoids_multiple_obstacles():
    obstacles = [
        PILLAR,
        Obstacle(center_xy=(2.2, 1.0), half_extents_xy=(0.15, 0.15)),
    ]
    path = plan_path((0.0, 0.0), (4.0, 0.0), obstacles=obstacles, robot_radius=ROBOT_RADIUS)
    for point in _densify(path):
        for obstacle in obstacles:
            assert _point_clear_of(point, obstacle, ROBOT_RADIUS)


def test_unreachable_goal_raises():
    """A goal fully enclosed by obstacles must raise, not loop forever."""
    walls = [
        Obstacle(center_xy=(2.0, 0.0), half_extents_xy=(0.1, 2.0)),
        Obstacle(center_xy=(4.0, 0.0), half_extents_xy=(0.1, 2.0)),
        Obstacle(center_xy=(3.0, 1.9), half_extents_xy=(1.0, 0.1)),
        Obstacle(center_xy=(3.0, -1.9), half_extents_xy=(1.0, 0.1)),
    ]
    with pytest.raises(PathPlannerError):
        plan_path((0.0, 0.0), (3.0, 0.0), obstacles=walls, robot_radius=ROBOT_RADIUS)


# ------------------------------------------------------------- standoff routes


def test_standoff_route_stops_short_of_target():
    route = plan_standoff_route(
        (0.0, 0.0), BOX_XY, standoff=0.3, obstacles=[PILLAR], robot_radius=ROBOT_RADIUS
    )
    end = route[-1]
    dist_to_target = math.hypot(end[0] - BOX_XY[0], end[1] - BOX_XY[1])
    assert dist_to_target == pytest.approx(0.3, abs=0.05)


def test_standoff_route_around_obstacle_is_clear():
    """Robot near the box, navigating to a standoff near the shelf: the route
    must clear the pillar — this is the exact geometry the old straight-line
    driver failed on."""
    route = plan_standoff_route(
        BOX_XY, SHELF_XY, standoff=0.4, obstacles=[PILLAR], robot_radius=ROBOT_RADIUS
    )
    for point in _densify(route):
        assert _point_clear_of(point, PILLAR, ROBOT_RADIUS)
    # and it still ends near the shelf
    end = route[-1]
    assert math.hypot(end[0] - SHELF_XY[0], end[1] - SHELF_XY[1]) == pytest.approx(0.4, abs=0.05)


def test_standoff_larger_than_distance_returns_start_only():
    """Already within standoff distance of the target -> no motion needed."""
    near_box = (BOX_XY[0] - 0.2, BOX_XY[1])
    route = plan_standoff_route(
        near_box, BOX_XY, standoff=0.3, obstacles=[PILLAR], robot_radius=ROBOT_RADIUS
    )
    assert len(route) == 1
    assert route[0] == pytest.approx(near_box)


# --------------------------------------------------- tracking-error inflation


def test_tracking_error_keeps_wider_clearance():
    """Paths for imperfect followers (the walking policy) must keep extra
    clearance so waypoint corner-cutting can never reach the obstacle."""
    tracking = 0.35
    path = plan_path(
        BOX_XY, SHELF_XY, obstacles=[PILLAR], robot_radius=ROBOT_RADIUS, tracking_error=tracking
    )
    # every waypoint CORNER (where cutting happens) keeps robot+tracking clearance;
    # endpoints are exempt (they are stationary arrival points)
    for corner in path[1:-1]:
        assert _point_clear_of(corner, PILLAR, ROBOT_RADIUS + tracking), (
            f"corner {corner} too close to the pillar for a corner-cutting follower"
        )


def test_tracking_error_goal_near_obstacle_still_reachable():
    """A goal close to an obstacle (e.g. a shelf standoff pose) must stay
    reachable even with tracking inflation — the exemption zone handles it."""
    # goal 0.5 m from the pillar face: inside the inflated zone, outside the hard zone
    goal = (
        PILLAR.center_xy[0] + PILLAR.half_extents_xy[0] + 0.5,
        PILLAR.center_xy[1],
    )
    path = plan_path(
        (0.0, 0.0), goal, obstacles=[PILLAR], robot_radius=ROBOT_RADIUS, tracking_error=0.35
    )
    assert path[-1] == pytest.approx(goal)


def test_zero_tracking_error_unchanged_behaviour():
    """Kinematic followers (tracking_error=0) keep the tighter routes."""
    a = plan_path(BOX_XY, SHELF_XY, obstacles=[PILLAR], robot_radius=ROBOT_RADIUS)
    b = plan_path(
        BOX_XY, SHELF_XY, obstacles=[PILLAR], robot_radius=ROBOT_RADIUS, tracking_error=0.0
    )
    assert a == b


# ----------------------------------------------------------------- primitives


def test_segment_clear_detects_blocked_segment():
    assert not segment_clear(BOX_XY, SHELF_XY, [PILLAR], clearance=ROBOT_RADIUS)


def test_segment_clear_accepts_clear_segment():
    assert segment_clear((0.0, 0.0), BOX_XY, [PILLAR], clearance=ROBOT_RADIUS)


# ----------------------------------------------------- scene obstacle adapter


def test_navigation_obstacles_are_static_geometry():
    """Static geometry (pillar AND shelf) is navigation obstacles; only small
    dynamic graspable objects (the box) are excluded — the robot must get right
    next to those to manipulate them."""
    obstacles = navigation_obstacles()
    centers = {o.center_xy for o in obstacles}
    assert (SCENE_LAYOUT["obstacle_P"][0], SCENE_LAYOUT["obstacle_P"][1]) in centers
    assert (SCENE_LAYOUT["shelf_A"][0], SCENE_LAYOUT["shelf_A"][1]) in centers
    assert (SCENE_LAYOUT["box_03"][0], SCENE_LAYOUT["box_03"][1]) not in centers


def test_footprint_target_standoff_stays_outside_footprint():
    """Navigating to a footprint target (the shelf) must end OUTSIDE its
    footprint at the requested standoff — never inside it (the pre-rebuild bug:
    standoff from the CENTER put the goal inside the 0.8x0.4 shelf)."""
    from physgate.world.layout import SHELF_SIZE

    shelf_half = (SHELF_SIZE[0] / 2, SHELF_SIZE[1] / 2)
    route = plan_standoff_route(
        BOX_XY,
        SHELF_XY,
        standoff=0.4,
        obstacles=navigation_obstacles(),
        robot_radius=ROBOT_RADIUS,
        target_half_extents=shelf_half,
    )
    end = route[-1]
    edge_clearance = max(
        abs(end[0] - SHELF_XY[0]) - shelf_half[0],
        abs(end[1] - SHELF_XY[1]) - shelf_half[1],
    )
    assert edge_clearance == pytest.approx(0.4, abs=0.1)
    # and the whole route clears both the pillar and the shelf footprint
    for point in _densify(route):
        assert _point_clear_of(point, PILLAR, ROBOT_RADIUS)


def test_scene_layout_has_no_waypoint():
    """The waypoint_W physics hack is gone (REBUILD.md Phase 1)."""
    assert "waypoint_W" not in SCENE_LAYOUT
