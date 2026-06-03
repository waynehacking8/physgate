"""Tests for the publication-quality matplotlib top-down trajectory animation.

Replaces the bare numpy renderer's frames with proper figures: titled, with a
legend, metric axes, the obstacle/shelf footprints, the planned A* route, and
the growing walked trail.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from physgate.world.layout import SCENE_LAYOUT  # noqa: E402


def _samples(n=12):
    return [
        {
            "t": round(i * 0.2, 2),
            "robot_xy": [0.1 * i, 0.0],
            "box_xy": [1.5, 0.5],
            "carrying": i > 6,
        }
        for i in range(n)
    ]


def test_topdown_mpl_frame_is_high_resolution(tmp_path):
    from PIL import Image

    from physgate.viz.topdown_mpl import render_topdown_frame

    frame = render_topdown_frame(
        layout=SCENE_LAYOUT,
        planned_path=[(0.0, 0.0), (1.2, 0.4)],
        trail=[(0.0, 0.0), (0.5, 0.1)],
        robot_xy=(0.5, 0.1),
        box_xy=(1.5, 0.5),
        carrying=False,
        t=1.0,
    )
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3
    # 150 DPI at >=6.4 inches -> >=960 px
    img = Image.fromarray(frame)
    assert img.width >= 900


def test_topdown_mpl_frames_have_consistent_shape():
    from physgate.viz.topdown_mpl import render_topdown_animation

    frames = render_topdown_animation(
        layout=SCENE_LAYOUT,
        planned_path=[(0.0, 0.0), (1.2, 0.4)],
        samples=_samples(10),
        hold_last_frames=4,
    )
    assert len(frames) == 14
    assert all(f.shape == frames[0].shape for f in frames)


def test_topdown_mpl_trail_grows_monotonically():
    """Frame k must reflect k trail points (the trail accumulates)."""
    from physgate.viz.topdown_mpl import render_topdown_animation

    frames = render_topdown_animation(
        layout=SCENE_LAYOUT, planned_path=[], samples=_samples(8), hold_last_frames=0
    )
    # later frames have more non-background ink than the first (trail grew)
    def ink(frame):
        return int((frame.sum(axis=2) < 720).sum())

    assert ink(frames[-1]) > ink(frames[0])


def test_topdown_mpl_rejects_trail_inside_obstacle():
    from physgate.viz.topdown_mpl import render_topdown_animation

    ox, oy, _ = SCENE_LAYOUT["obstacle_P"]
    bad = [{"t": 0.0, "robot_xy": [ox, oy], "box_xy": [1.5, 0.5], "carrying": False}]
    with pytest.raises(ValueError, match="obstacle"):
        render_topdown_animation(
            layout=SCENE_LAYOUT, planned_path=[], samples=bad, check_collisions=True
        )
