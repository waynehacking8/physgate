"""Publication-quality top-down trajectory animation (matplotlib).

Renders each frame as a proper figure — titled, legended, metric axes in meters,
obstacle/shelf footprints, the planned A* route, and the growing walked trail —
then returns RGB arrays the encoder turns into a GIF/MP4. This replaces the bare
numpy renderer (physgate.viz.topdown) for the README media; the numpy renderer
is kept for fast unit tests.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from physgate.world.layout import BOX_SIZE, OBSTACLE_SIZE, SHELF_SIZE

DPI = 150
FIG_W, FIG_H = 7.2, 6.0
X_RANGE = (-1.0, 4.2)
Y_RANGE = (-2.2, 1.6)

# colors consistent with the Isaac scene materials
C_OBSTACLE = "#cc3333"
C_SHELF = "#6666a0"
C_BOX = "#b27f33"
C_ROBOT = "#2ca05a"
C_TRAIL = "#2e8b57"
C_PATH = "#f0c020"


def _rect(ax, center_xy, size_xy, color, label=None, alpha=0.85):
    x = center_xy[0] - size_xy[0] / 2
    y = center_xy[1] - size_xy[1] / 2
    ax.add_patch(
        mpatches.Rectangle(
            (x, y), size_xy[0], size_xy[1], facecolor=color, edgecolor="black",
            linewidth=0.8, alpha=alpha, label=label, zorder=2,
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


def render_topdown_frame(
    layout: dict,
    planned_path: list,
    trail: list,
    robot_xy: tuple,
    box_xy: tuple,
    carrying: bool,
    t: float,
) -> np.ndarray:
    """Render one annotated top-down frame -> RGB uint8 array."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=10)
    ax.set_ylabel("y (m)", fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    # static scene
    _rect(ax, layout["shelf_A"][:2], SHELF_SIZE[:2], C_SHELF, label="shelf (goal)")
    _rect(ax, layout["obstacle_P"][:2], OBSTACLE_SIZE[:2], C_OBSTACLE, label="pillar (obstacle)")

    # planned A* route
    if planned_path:
        px, py = zip(*planned_path)
        ax.plot(px, py, "--", color=C_PATH, linewidth=2.4, label="planned A* route", zorder=3)

    # walked trail
    if len(trail) >= 2:
        tx, ty = zip(*trail)
        ax.plot(tx, ty, "-", color=C_TRAIL, linewidth=2.8, label="walked trail", zorder=4)

    # box
    _rect(ax, box_xy, BOX_SIZE[:2], C_BOX, label="box")

    # robot (ring when carrying)
    if carrying:
        ax.add_patch(plt.Circle(robot_xy, 0.26, facecolor="none", edgecolor="white",
                                linewidth=2.0, zorder=5))
    ax.add_patch(plt.Circle(robot_xy, 0.20, facecolor=C_ROBOT, edgecolor="black",
                            linewidth=0.8, zorder=6, label="Go2 robot"))

    phase = "carrying box → shelf" if carrying else "navigating / manipulating"
    ax.set_title(
        f"physgate — top-down trajectory (Isaac Sim rollout)\n"
        f"t = {t:.1f} s   |   {phase}",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncols=1)
    fig.tight_layout()
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
    for s in samples:
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
            )
        )
    if frames and hold_last_frames > 0:
        frames.extend([frames[-1].copy() for _ in range(hold_last_frames)])
    return frames
