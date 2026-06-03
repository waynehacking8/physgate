"""Tests for physgate.viz.topdown — top-down trajectory animation frames.

Pure logic (numpy only): renders the scene layout, the A* planned route, and
the robot/box trajectory logged from an Isaac rollout into RGB frames. The
frames prove the robot physically walked the planned route around the obstacle.
"""

from __future__ import annotations

import numpy as np
import pytest

from physgate.world.layout import OBSTACLE_SIZE, SCENE_LAYOUT


def _make_renderer():
    from physgate.viz.topdown import TopDownRenderer

    return TopDownRenderer(
        x_range=(-1.0, 4.5),
        y_range=(-2.5, 2.0),
        pixels_per_meter=60,
    )


def test_world_to_pixel_transform_is_consistent():
    renderer = _make_renderer()
    # x_range spans 5.5 m -> 330 px wide; y_range spans 4.5 m -> 270 px tall
    assert renderer.width == 330
    assert renderer.height == 270
    # world origin maps inside the image
    px, py = renderer.world_to_pixel(0.0, 0.0)
    assert 0 <= px < renderer.width
    assert 0 <= py < renderer.height
    # +y in world is "up" -> smaller pixel row
    _, py_up = renderer.world_to_pixel(0.0, 1.0)
    assert py_up < py


def test_frame_draws_obstacle_at_its_world_position():
    renderer = _make_renderer()
    frame = renderer.render_frame(
        layout=SCENE_LAYOUT,
        planned_path=[],
        robot_trail=[],
        robot_xy=(0.0, 0.0),
        box_xy=(1.5, 0.5),
        carrying=False,
    )
    assert frame.dtype == np.uint8 and frame.shape == (renderer.height, renderer.width, 3)
    # the obstacle footprint center pixel must differ from the background
    ox, oy = SCENE_LAYOUT["obstacle_P"][:2]
    px, py = renderer.world_to_pixel(ox, oy)
    background = frame[2, 2]
    assert not np.array_equal(frame[py, px], background), "obstacle not drawn"
    # a point far outside every footprint stays background
    fx, fy = renderer.world_to_pixel(-0.8, 1.8)
    assert np.array_equal(frame[fy, fx], background)


def test_animation_produces_one_frame_per_sample_plus_hold():
    from physgate.viz.topdown import render_rollout_animation

    samples = [
        {"t": i * 0.2, "robot_xy": [0.0 + 0.05 * i, 0.0], "box_xy": [1.5, 0.5], "carrying": False}
        for i in range(10)
    ]
    frames = render_rollout_animation(
        layout=SCENE_LAYOUT,
        planned_path=[(0.0, 0.0), (1.22, 0.41)],
        samples=samples,
        hold_last_frames=5,
    )
    assert len(frames) == 15  # 10 samples + 5 hold frames
    assert all(f.shape == frames[0].shape and f.dtype == np.uint8 for f in frames)


def test_animation_trail_grows_over_time():
    """Later frames contain the robot's past positions (the trail) — earlier
    frames must not contain future positions."""
    from physgate.viz.topdown import TopDownRenderer, render_rollout_animation

    # keep the trail off the 1 m grid lines so background comparison is clean
    samples = [
        {"t": i * 0.2, "robot_xy": [0.3 * i + 0.1, 0.3], "box_xy": [1.5, 0.5], "carrying": False}
        for i in range(5)
    ]
    frames = render_rollout_animation(
        layout=SCENE_LAYOUT, planned_path=[], samples=samples, hold_last_frames=0
    )
    renderer = TopDownRenderer()
    # the robot's final position (1.3, 0.3) is drawn in the last frame...
    px, py = renderer.world_to_pixel(1.3, 0.3)
    background = frames[0][2, 2]
    assert not np.array_equal(frames[-1][py, px], background)
    # ...but not in the first frame (robot hasn't been there yet)
    assert np.array_equal(frames[0][py, px], background)


def test_obstacle_pixels_never_overlap_robot_trail():
    """Sanity: the robot trail drawn from real rollout data must never enter the
    obstacle footprint (data plausibility check rendered visually)."""
    from physgate.viz.topdown import TopDownRenderer

    renderer = TopDownRenderer()
    ox, oy, _ = SCENE_LAYOUT["obstacle_P"]
    half_w, half_h = OBSTACLE_SIZE[0] / 2, OBSTACLE_SIZE[1] / 2
    # a trail point inside the obstacle is a violation the renderer must flag
    with pytest.raises(ValueError, match="inside the obstacle"):
        renderer.render_frame(
            layout=SCENE_LAYOUT,
            planned_path=[],
            robot_trail=[(ox, oy)],
            robot_xy=(ox, oy),
            box_xy=(1.5, 0.5),
            carrying=False,
            check_collisions=True,
        )
    # a trail point at clearance distance is fine
    frame = renderer.render_frame(
        layout=SCENE_LAYOUT,
        planned_path=[],
        robot_trail=[(ox - half_w - 0.5, oy - half_h - 0.5)],
        robot_xy=(0.0, 0.0),
        box_xy=(1.5, 0.5),
        carrying=False,
        check_collisions=True,
    )
    assert frame is not None
