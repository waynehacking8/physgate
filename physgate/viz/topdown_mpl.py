"""Publication-quality top-down trajectory animation (matplotlib).

Renders each frame as a proper figure — titled, legended, metric axes in meters,
obstacle/shelf footprints, the planned A* route, and the growing walked trail —
then returns RGB arrays the encoder turns into a GIF/MP4. This replaces the bare
numpy renderer (physgate.viz.topdown) for the README media; the numpy renderer
is kept for fast unit tests.

Design notes (camera-ready):
* the view is cropped tight to where the action happens, not to the whole arena,
  so the robot is never a lonely dot in a sea of white;
* the robot and the walked trail use distinct colors (a green trail leading a
  navy robot puck), and the robot carries a heading wedge so motion is legible
  in a single still frame;
* when carrying, the box rides on the robot puck so the payload stays visible.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from physgate.world.layout import BOX_SIZE, OBSTACLE_SIZE, SHELF_SIZE

DPI = 150
FIG_W, FIG_H = 8.0, 5.2
#: cropped to the action: start (0,0) → box (1.5,0.5) → around pillar → shelf (3,-1)
X_RANGE = (-0.6, 3.9)
Y_RANGE = (-1.55, 0.95)

# colors — robot and trail are deliberately different so motion reads in a still
C_OBSTACLE = "#d1495b"   # pillar
C_SHELF = "#8d99ae"      # goal shelf
C_BOX = "#e09f3e"        # box / payload
C_ROBOT = "#1d3557"      # navy robot puck (distinct from the green trail)
C_TRAIL = "#2a9d8f"      # teal-green walked trail
C_PLAN = "#f4a000"       # amber planned A* route
C_START = "#6c757d"      # neutral start marker


def _rect(ax, center_xy, size_xy, color, label=None, alpha=0.9, zorder=2):
    x = center_xy[0] - size_xy[0] / 2
    y = center_xy[1] - size_xy[1] / 2
    ax.add_patch(
        mpatches.Rectangle(
            (x, y), size_xy[0], size_xy[1], facecolor=color, edgecolor="black",
            linewidth=0.8, alpha=alpha, label=label, zorder=zorder,
        )
    )


def _inside_obstacle(layout, xy) -> bool:
    ox, oy = layout["obstacle_P"][:2]
    return abs(xy[0] - ox) <= OBSTACLE_SIZE[0] / 2 and abs(xy[1] - oy) <= OBSTACLE_SIZE[1] / 2


def _fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return buf.reshape(h, w, 4)[:, :, :3].copy()


def _heading(trail: list) -> float | None:
    """Heading (radians) from the last meaningful trail segment, or None."""
    if len(trail) < 2:
        return None
    x1, y1 = trail[-1]
    for prev in reversed(trail[:-1]):
        dx, dy = x1 - prev[0], y1 - prev[1]
        if dx * dx + dy * dy > 1e-4:  # >1 cm of motion
            return math.atan2(dy, dx)
    return None


def render_topdown_frame(
    layout: dict,
    planned_path: list,
    trail: list,
    robot_xy: tuple,
    box_xy: tuple,
    carrying: bool,
    t: float,
    progress: float | None = None,
) -> np.ndarray:
    """Render one annotated top-down frame -> RGB uint8 array."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI, layout="tight")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=10)
    ax.set_ylabel("y (m)", fontsize=10)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # static scene
    _rect(ax, layout["shelf_A"][:2], SHELF_SIZE[:2], C_SHELF, label="shelf (goal)")
    _rect(ax, layout["obstacle_P"][:2], OBSTACLE_SIZE[:2], C_OBSTACLE, label="pillar (obstacle)")

    # goal marker on the shelf
    sx, sy = layout["shelf_A"][:2]
    ax.scatter([sx], [sy], marker="*", s=180, color="white", edgecolor="black",
               linewidth=0.8, zorder=3)

    # start marker
    ax.scatter([0.0], [0.0], marker="P", s=80, color=C_START, edgecolor="black",
               linewidth=0.6, zorder=3, label="start")

    # planned A* route
    if planned_path:
        px, py = zip(*planned_path)
        ax.plot(px, py, "--", color=C_PLAN, linewidth=2.2, label="planned A* route",
                zorder=4, dash_capstyle="round")

    # walked trail
    if len(trail) >= 2:
        tx, ty = zip(*trail)
        ax.plot(tx, ty, "-", color=C_TRAIL, linewidth=3.0, label="walked trail",
                zorder=5, solid_capstyle="round")

    # box: at its own location, or riding the robot when carried
    box_draw = (robot_xy[0], robot_xy[1] + 0.34) if carrying else box_xy
    _rect(ax, box_draw, BOX_SIZE[:2], C_BOX,
          label="box (payload)" if carrying else "box", zorder=6)

    # robot puck + heading wedge
    head = _heading(trail)
    if head is not None:
        ax.add_patch(mpatches.FancyArrow(
            robot_xy[0], robot_xy[1],
            0.26 * math.cos(head), 0.26 * math.sin(head),
            width=0.04, head_width=0.16, head_length=0.14,
            length_includes_head=True, color=C_ROBOT, zorder=7, alpha=0.9,
        ))
    ax.add_patch(plt.Circle(robot_xy, 0.17, facecolor=C_ROBOT, edgecolor="white",
                            linewidth=1.4, zorder=8, label="Go2 robot"))
    if carrying:
        ax.add_patch(plt.Circle(robot_xy, 0.23, facecolor="none", edgecolor=C_BOX,
                                linewidth=2.0, linestyle=(0, (2, 1)), zorder=8))

    phase = "carrying box → shelf" if carrying else "navigating to box"
    ax.set_title(
        "physgate — Sim-Gate rollout, top-down trajectory (Isaac Lab PhysX)",
        fontsize=12, fontweight="bold", pad=10,
    )
    # phase / time as an in-axes caption (keeps the title one clean line)
    ax.text(0.015, 0.97, f"t = {t:4.1f} s   ·   {phase}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "#cccccc", "alpha": 0.9})

    # progress bar along the bottom of the axes
    if progress is not None:
        p = max(0.0, min(1.0, progress))
        ax.add_patch(mpatches.Rectangle((0.015, 0.03), 0.30, 0.022,
                     transform=ax.transAxes, facecolor="#e9ecef",
                     edgecolor="#adb5bd", linewidth=0.6, zorder=9))
        ax.add_patch(mpatches.Rectangle((0.015, 0.03), 0.30 * p, 0.022,
                     transform=ax.transAxes, facecolor=C_TRAIL, zorder=10))

    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92,
              edgecolor="#cccccc", ncols=1)
    frame = _fig_to_rgb(fig)
    plt.close(fig)
    return frame


def render_topdown_animation(
    layout: dict,
    planned_path: list,
    samples: list[dict],
    hold_last_frames: int = 16,
    check_collisions: bool = False,
) -> list[np.ndarray]:
    """Render the full annotated animation from logged trajectory samples."""
    if check_collisions:
        for s in samples:
            if _inside_obstacle(layout, s["robot_xy"]):
                raise ValueError(
                    f"trail point {s['robot_xy']} inside the obstacle footprint — "
                    "rollout data is implausible"
                )

    frames: list[np.ndarray] = []
    trail: list[tuple] = []
    n = len(samples)
    for i, s in enumerate(samples):
        trail.append(tuple(s["robot_xy"]))
        frames.append(
            render_topdown_frame(
                layout=layout,
                planned_path=planned_path,
                trail=list(trail),
                robot_xy=tuple(s["robot_xy"]),
                box_xy=tuple(s["box_xy"]),
                carrying=bool(s.get("carrying", False)),
                t=float(s.get("t", 0.0)),
                progress=(i + 1) / n if n > 1 else 1.0,
            )
        )
    if frames and hold_last_frames > 0:
        frames.extend([frames[-1].copy() for _ in range(hold_last_frames)])
    return frames
