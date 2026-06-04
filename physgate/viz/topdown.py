"""Top-down trajectory animation: scene layout + planned route + walked trail.

Pure numpy rendering (no matplotlib, no Isaac) so it is unit-testable in the
pure-logic venv. Visualizes what the deterministic navigation layer + walking
policy actually did in Isaac Sim: the A* route around the obstacle, the robot's
real walked trail, and the carried box — the dynamic proof that obstacle
avoidance lives in the low level, not in the LLM.

Colors follow the Isaac scene materials (box: tan, shelf: blue-grey, pillar: red).
"""

from __future__ import annotations

import numpy as np

from physgate.world.layout import BOX_SIZE, OBSTACLE_SIZE, SHELF_SIZE

# ------------------------------------------------------------------- palette

BACKGROUND = np.array([24, 26, 32], dtype=np.uint8)
GRID = np.array([40, 44, 52], dtype=np.uint8)
OBSTACLE_COLOR = np.array([204, 51, 51], dtype=np.uint8)  # pillar: red
SHELF_COLOR = np.array([102, 102, 153], dtype=np.uint8)  # shelf: blue-grey
BOX_COLOR = np.array([178, 127, 51], dtype=np.uint8)  # box: tan
ROBOT_COLOR = np.array([64, 200, 96], dtype=np.uint8)  # robot: green
TRAIL_COLOR = np.array([46, 139, 87], dtype=np.uint8)  # walked trail
PATH_COLOR = np.array([240, 200, 60], dtype=np.uint8)  # planned A* route: yellow
CARRY_RING = np.array([255, 255, 255], dtype=np.uint8)  # ring when carrying


class TopDownRenderer:
    """World-coordinate (meters) -> pixel renderer for the fetch scene."""

    def __init__(
        self,
        x_range: tuple[float, float] = (-1.0, 4.5),
        y_range: tuple[float, float] = (-2.5, 2.0),
        pixels_per_meter: int = 60,
    ):
        """Initialize with world coordinate bounds and pixel resolution."""
        self.x_range = x_range
        self.y_range = y_range
        self.ppm = pixels_per_meter
        self.width = int(round((x_range[1] - x_range[0]) * pixels_per_meter))
        self.height = int(round((y_range[1] - y_range[0]) * pixels_per_meter))

    # ----- transforms -----

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """World (x, y) in meters -> (col, row) pixel. +y world is up (smaller row)."""
        px = int(round((x - self.x_range[0]) * self.ppm))
        py = int(round((self.y_range[1] - y) * self.ppm))
        return (
            int(np.clip(px, 0, self.width - 1)),
            int(np.clip(py, 0, self.height - 1)),
        )

    # ----- drawing primitives (in-place on a frame array) -----

    def _blank(self) -> np.ndarray:
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = BACKGROUND
        # 1 m grid lines
        for gx in range(int(np.ceil(self.x_range[0])), int(np.floor(self.x_range[1])) + 1):
            px, _ = self.world_to_pixel(gx, 0)
            frame[:, px] = GRID
        for gy in range(int(np.ceil(self.y_range[0])), int(np.floor(self.y_range[1])) + 1):
            _, py = self.world_to_pixel(0, gy)
            frame[py, :] = GRID
        return frame

    def _fill_rect(
        self, frame: np.ndarray, center_xy: tuple[float, float], size_xy: tuple[float, float], color
    ) -> None:
        x0, y0 = center_xy[0] - size_xy[0] / 2, center_xy[1] - size_xy[1] / 2
        x1, y1 = center_xy[0] + size_xy[0] / 2, center_xy[1] + size_xy[1] / 2
        px0, py1 = self.world_to_pixel(x0, y0)
        px1, py0 = self.world_to_pixel(x1, y1)
        frame[py0 : py1 + 1, px0 : px1 + 1] = color

    def _fill_disc(self, frame: np.ndarray, center_xy: tuple[float, float], radius_m: float, color) -> None:
        cx, cy = self.world_to_pixel(*center_xy)
        r = max(1, int(round(radius_m * self.ppm)))
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        mask = xx * xx + yy * yy <= r * r
        y0, y1 = max(0, cy - r), min(self.height, cy + r + 1)
        x0, x1 = max(0, cx - r), min(self.width, cx + r + 1)
        frame_region = frame[y0:y1, x0:x1]
        mask_region = mask[(y0 - (cy - r)) : (y1 - (cy - r)), (x0 - (cx - r)) : (x1 - (cx - r))]
        frame_region[mask_region] = color

    def _draw_polyline(
        self, frame: np.ndarray, points: list[tuple[float, float]], color, thickness_m: float = 0.04
    ) -> None:
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            length = float(np.hypot(x1 - x0, y1 - y0))
            n = max(2, int(length * self.ppm))
            for t in np.linspace(0.0, 1.0, n):
                self._fill_disc(
                    frame, (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t), thickness_m, color
                )

    # ----- scene frame -----

    @staticmethod
    def _inside_obstacle(layout: dict, point_xy: tuple[float, float]) -> bool:
        ox, oy = layout["obstacle_P"][:2]
        return (
            abs(point_xy[0] - ox) <= OBSTACLE_SIZE[0] / 2
            and abs(point_xy[1] - oy) <= OBSTACLE_SIZE[1] / 2
        )

    def render_frame(
        self,
        layout: dict,
        planned_path: list[tuple[float, float]],
        robot_trail: list[tuple[float, float]],
        robot_xy: tuple[float, float],
        box_xy: tuple[float, float],
        carrying: bool,
        check_collisions: bool = False,
    ) -> np.ndarray:
        """Render one top-down frame of the scene state."""
        if check_collisions:
            for point in robot_trail:
                if self._inside_obstacle(layout, point):
                    raise ValueError(
                        f"robot trail point {point} is inside the obstacle footprint — "
                        "rollout data is implausible (navigation must route around it)"
                    )

        frame = self._blank()
        # static scene
        self._fill_rect(frame, layout["shelf_A"][:2], SHELF_SIZE[:2], SHELF_COLOR)
        self._fill_rect(frame, layout["obstacle_P"][:2], OBSTACLE_SIZE[:2], OBSTACLE_COLOR)
        # planned route (under the trail)
        if planned_path:
            self._draw_polyline(frame, planned_path, PATH_COLOR, thickness_m=0.03)
        # walked trail
        if robot_trail:
            self._draw_polyline(frame, robot_trail, TRAIL_COLOR, thickness_m=0.05)
        # box
        self._fill_rect(frame, box_xy, BOX_SIZE[:2], BOX_COLOR)
        # robot (ring when carrying)
        if carrying:
            self._fill_disc(frame, robot_xy, 0.22, CARRY_RING)
        self._fill_disc(frame, robot_xy, 0.18, ROBOT_COLOR)
        return frame


def render_rollout_animation(
    layout: dict,
    planned_path: list[tuple[float, float]],
    samples: list[dict],
    renderer: TopDownRenderer | None = None,
    hold_last_frames: int = 12,
    check_collisions: bool = False,
) -> list[np.ndarray]:
    """Render a full rollout animation from logged trajectory samples.

    Args:
        layout: SCENE_LAYOUT-style dict of entity positions.
        planned_path: the A* waypoint route (drawn in yellow, static).
        samples: per-control-step logs: {"t", "robot_xy", "box_xy", "carrying"}.
        renderer: optional pre-configured renderer.
        hold_last_frames: repeat the final frame so the GIF pauses on the result.
        check_collisions: raise if the walked trail enters the obstacle footprint.
    """
    renderer = renderer or TopDownRenderer()
    frames: list[np.ndarray] = []
    trail: list[tuple[float, float]] = []
    for sample in samples:
        robot_xy = tuple(sample["robot_xy"])
        trail.append(robot_xy)
        frames.append(
            renderer.render_frame(
                layout=layout,
                planned_path=planned_path,
                robot_trail=list(trail),
                robot_xy=robot_xy,
                box_xy=tuple(sample["box_xy"]),
                carrying=bool(sample.get("carrying", False)),
                check_collisions=check_collisions,
            )
        )
    if frames and hold_last_frames > 0:
        frames.extend([frames[-1].copy() for _ in range(hold_last_frames)])
    return frames
