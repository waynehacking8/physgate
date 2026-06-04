#!/usr/bin/env python3
"""Record publication-quality top-down GIFs for the agent architecture upgrade.

Renders matplotlib animations showing robot executing plans step-by-step
with tool annotations. Uses the pure-logic venv (no Isaac Sim needed for
2D top-down rendering).

Usage:
    .venv/bin/python scripts/record_upgraded_demos.py

Output: docs/media/t3_push_delivery.gif, docs/media/t2_locked_door.gif
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from physgate.world.layout import BOX_SIZE, OBSTACLE_SIZE, SCENE_LAYOUT, SHELF_SIZE
from physgate.viz.encode import encode_frames_to_gif

DPI = 150
FIG_W, FIG_H = 9.0, 5.5
X_RANGE = (-0.8, 4.2)
Y_RANGE = (-1.8, 1.2)

C_OBSTACLE = "#d1495b"
C_SHELF = "#8d99ae"
C_BOX = "#e09f3e"
C_ROBOT = "#1d3557"
C_TRAIL = "#2a9d8f"
C_CRATE = "#b5651d"
C_DOOR_LOCKED = "#8B0000"
C_DOOR_UNLOCKED = "#228B22"
C_KEY = "#FFD700"

CRATE_SIZE = (0.35, 0.35)
CRATE_POS = (2.0, -0.3)
DOOR_SIZE = (0.1, 1.0)
DOOR_POS = (2.5, -0.5)
KEY_POS = (0.8, 0.8)


class SceneRenderer:
    def __init__(self, title: str):
        self.title = title
        self.trail: list[tuple] = []

    def _base_axes(self) -> tuple:
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
        return fig, ax

    def _draw_shelf(self, ax):
        sx, sy = SCENE_LAYOUT["shelf_A"][:2]
        ax.add_patch(mpatches.Rectangle(
            (sx - SHELF_SIZE[0]/2, sy - SHELF_SIZE[1]/2), SHELF_SIZE[0], SHELF_SIZE[1],
            facecolor=C_SHELF, edgecolor="black", lw=0.8, alpha=0.9, label="shelf (goal)", zorder=2))
        ax.scatter([sx], [sy], marker="*", s=180, color="white", edgecolor="black", lw=0.8, zorder=3)

    def _draw_obstacle(self, ax):
        ox, oy = SCENE_LAYOUT["obstacle_P"][:2]
        ax.add_patch(mpatches.Rectangle(
            (ox - OBSTACLE_SIZE[0]/2, oy - OBSTACLE_SIZE[1]/2), OBSTACLE_SIZE[0], OBSTACLE_SIZE[1],
            facecolor=C_OBSTACLE, edgecolor="black", lw=0.8, alpha=0.9, label="pillar", zorder=2))

    def _draw_robot(self, ax, pos, carrying=False, carry_color=C_BOX):
        ax.add_patch(plt.Circle(pos[:2], 0.17, facecolor=C_ROBOT,
                                edgecolor="white", lw=1.4, zorder=7, label="Go2 robot"))
        if carrying:
            ax.add_patch(plt.Circle(pos[:2], 0.23, facecolor="none",
                                    edgecolor=carry_color, lw=2.0, linestyle=(0, (2,1)), zorder=7))
        if len(self.trail) >= 2:
            dx = self.trail[-1][0] - self.trail[-2][0]
            dy = self.trail[-1][1] - self.trail[-2][1]
            if dx*dx + dy*dy > 1e-4:
                h = math.atan2(dy, dx)
                ax.add_patch(mpatches.FancyArrow(
                    pos[0], pos[1], 0.26*math.cos(h), 0.26*math.sin(h),
                    width=0.04, head_width=0.16, head_length=0.14,
                    length_includes_head=True, color=C_ROBOT, zorder=8, alpha=0.9))

    def _draw_trail(self, ax):
        if len(self.trail) >= 2:
            tx, ty = zip(*self.trail)
            ax.plot(tx, ty, "-", color=C_TRAIL, lw=2.5, zorder=4, solid_capstyle="round")

    def _draw_hud(self, ax, phase: str, tool: str, step: int, total: int):
        ax.set_title(self.title, fontsize=12, fontweight="bold", pad=10)
        ax.text(0.015, 0.97, f"Step {step}/{total}  ·  {phase}",
                transform=ax.transAxes, fontsize=10, fontweight="bold", va="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "edgecolor": "#cccccc", "alpha": 0.9})
        ax.text(0.015, 0.88, f"tool: {tool}",
                transform=ax.transAxes, fontsize=9, fontweight="bold", va="top",
                color="#e15759",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#fff0f0",
                      "edgecolor": "#e15759", "alpha": 0.9})
        p = step / total
        ax.add_patch(mpatches.Rectangle((0.015, 0.03), 0.30, 0.022,
                     transform=ax.transAxes, facecolor="#e9ecef", edgecolor="#adb5bd", lw=0.6, zorder=9))
        ax.add_patch(mpatches.Rectangle((0.015, 0.03), 0.30*p, 0.022,
                     transform=ax.transAxes, facecolor=C_TRAIL, zorder=10))

    def to_frame(self, fig) -> np.ndarray:
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frame = buf.reshape(h, w, 4)[:, :, :3].copy()
        plt.close(fig)
        return frame

    def move_robot(self, start, end, n_steps, step_num, total, phase, tool):
        frames = []
        for i in range(n_steps):
            t = (i+1)/n_steps
            pos = start*(1-t) + end*t
            self.trail.append(tuple(pos[:2]))
            if i % 2 == 0:
                frames.append((pos.copy(), step_num, phase, tool))
        return frames


def record_t3(output_dir: Path) -> Path:
    print("Recording T3: Blocked-path clearance...")
    r = SceneRenderer("physgate — T3: Blocked-Path Clearance (10 tools, physics-validated)")
    TOTAL = 7
    robot_pos = np.array([0.0, 0.0])
    crate_pos = np.array(CRATE_POS)
    box_pos = np.array(SCENE_LAYOUT["box_03"][:2])
    shelf_pos = np.array(SCENE_LAYOUT["shelf_A"][:2])
    carrying = False
    frames = []

    def render(step, phase, tool):
        fig, ax = r._base_axes()
        r._draw_shelf(ax)
        r._draw_obstacle(ax)
        ax.add_patch(mpatches.Rectangle(
            (crate_pos[0]-CRATE_SIZE[0]/2, crate_pos[1]-CRATE_SIZE[1]/2),
            CRATE_SIZE[0], CRATE_SIZE[1],
            facecolor=C_CRATE, edgecolor="black", lw=1.2, alpha=0.9, label="crate (pushable)", zorder=3))
        bx, by = (robot_pos if carrying else box_pos)
        if carrying:
            by += 0.3
        ax.add_patch(mpatches.Rectangle(
            (bx-BOX_SIZE[0]/2, by-BOX_SIZE[1]/2), BOX_SIZE[0], BOX_SIZE[1],
            facecolor=C_BOX, edgecolor="black", lw=0.8, alpha=0.9, label="box", zorder=5))
        r._draw_trail(ax)
        r._draw_robot(ax, robot_pos, carrying)
        r._draw_hud(ax, phase, tool, step, TOTAL)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92, edgecolor="#cccccc")
        return r.to_frame(fig)

    # 1: inspect
    for _ in range(6): frames.append(render(1, "Inspecting crate...", "inspect_object(crate_01)"))
    # 2: move to crate
    for i in range(16):
        t=(i+1)/16; robot_pos=np.array([0,0])*(1-t)+np.array([crate_pos[0]-0.4, crate_pos[1]])*t
        r.trail.append(tuple(robot_pos)); frames.append(render(2, "Moving to crate", "move_to_pose(crate_01)"))
    # 3: push
    start_c = crate_pos.copy()
    for i in range(12):
        t=(i+1)/12; crate_pos=start_c+np.array([0.8*t,0]); robot_pos=np.array([crate_pos[0]-0.4, crate_pos[1]])
        r.trail.append(tuple(robot_pos)); frames.append(render(3, "Pushing crate east →", "push_object(crate_01, east)"))
    # 4: move to box
    start_r = robot_pos.copy()
    for i in range(16):
        t=(i+1)/16; robot_pos=start_r*(1-t)+np.array([box_pos[0]-0.3, box_pos[1]])*t
        r.trail.append(tuple(robot_pos)); frames.append(render(4, "Moving to box", "move_to_pose(box_03)"))
    # 5: pick
    carrying=True
    for _ in range(6): frames.append(render(5, "Picking up box", "execute_skill(pick, box_03)"))
    # 6: move to shelf
    start_r = robot_pos.copy()
    for i in range(20):
        t=(i+1)/20; robot_pos=start_r*(1-t)+np.array([shelf_pos[0]-0.4, shelf_pos[1]])*t
        r.trail.append(tuple(robot_pos)); frames.append(render(6, "Carrying box → shelf", "move_to_pose(shelf_A)"))
    # 7: place
    carrying=False; box_pos=shelf_pos.copy()
    for _ in range(10): frames.append(render(7, "Box placed on shelf ✓", "execute_skill(place, shelf_A)"))
    for _ in range(14): frames.append(frames[-1].copy())

    out = output_dir / "t3_push_delivery.gif"
    encode_frames_to_gif(frames, out, fps=8)
    print(f"  Saved: {out} ({len(frames)} frames, {out.stat().st_size//1024}KB)")
    return out


def record_t2(output_dir: Path) -> Path:
    print("Recording T2: Locked-door delivery...")
    r = SceneRenderer("physgate — T2: Locked-Door Delivery (key→unlock→open→deliver)")
    TOTAL = 9
    robot_pos = np.array([0.0, 0.0])
    box_pos = np.array(SCENE_LAYOUT["box_03"][:2])
    shelf_pos = np.array(SCENE_LAYOUT["shelf_A"][:2])
    key_pos_2d = np.array(KEY_POS)
    door_pos_2d = np.array(DOOR_POS)
    carrying_key = False
    carrying_box = False
    door_locked = True
    door_open = False
    frames = []

    def render(step, phase, tool):
        fig, ax = r._base_axes()
        r._draw_shelf(ax)
        # door
        if not door_open:
            dc = C_DOOR_LOCKED if door_locked else C_DOOR_UNLOCKED
            dl = "door (LOCKED)" if door_locked else "door (unlocked)"
            ax.add_patch(mpatches.Rectangle(
                (door_pos_2d[0]-DOOR_SIZE[0]/2, door_pos_2d[1]-DOOR_SIZE[1]/2),
                DOOR_SIZE[0], DOOR_SIZE[1], facecolor=dc, edgecolor="black", lw=1.5, alpha=0.9, label=dl, zorder=3))
        # key
        if not carrying_key and not door_open:
            ax.add_patch(mpatches.Rectangle(
                (key_pos_2d[0]-0.06, key_pos_2d[1]-0.03), 0.12, 0.06,
                facecolor=C_KEY, edgecolor="black", lw=1.0, alpha=0.95, label="key", zorder=3))
        # box
        bx, by = (robot_pos if carrying_box else box_pos)
        if carrying_box: by = by + 0.3
        ax.add_patch(mpatches.Rectangle(
            (bx-BOX_SIZE[0]/2, by-BOX_SIZE[1]/2), BOX_SIZE[0], BOX_SIZE[1],
            facecolor=C_BOX, edgecolor="black", lw=0.8, alpha=0.9, label="box", zorder=5))
        r._draw_trail(ax)
        r._draw_robot(ax, robot_pos, carrying_box or carrying_key, C_KEY if carrying_key else C_BOX)
        if carrying_key:
            ax.text(robot_pos[0], robot_pos[1]+0.28, "KEY", ha="center", fontsize=7, fontweight="bold", color=C_KEY, zorder=8)
        r._draw_hud(ax, phase, tool, step, TOTAL)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92, edgecolor="#cccccc")
        return r.to_frame(fig)

    # 1: query_scene
    for _ in range(6): frames.append(render(1, "Scanning scene...", "query_scene()"))
    # 2: move to key
    s=robot_pos.copy()
    for i in range(12):
        t=(i+1)/12; robot_pos=s*(1-t)+np.array([key_pos_2d[0]-0.2, key_pos_2d[1]])*t
        r.trail.append(tuple(robot_pos)); frames.append(render(2, "Moving to key", "move_to_pose(key_01)"))
    # 3: pick key
    carrying_key=True
    for _ in range(6): frames.append(render(3, "Picking up key", "execute_skill(pick, key_01)"))
    # 4: move to door
    s=robot_pos.copy()
    for i in range(16):
        t=(i+1)/16; robot_pos=s*(1-t)+np.array([door_pos_2d[0]-0.3, door_pos_2d[1]])*t
        r.trail.append(tuple(robot_pos)); frames.append(render(4, "Moving to door", "move_to_pose(door_01)"))
    # 5: unlock
    door_locked=False
    for _ in range(8): frames.append(render(5, "Unlocking door...", "unlock_door(door_01, key_01)"))
    # 6: open
    door_open=True; carrying_key=False
    for _ in range(8): frames.append(render(6, "Door opened! ✓", "open_door(door_01)"))
    # 7: move to box
    s=robot_pos.copy()
    for i in range(14):
        t=(i+1)/14; robot_pos=s*(1-t)+np.array([box_pos[0]-0.3, box_pos[1]])*t
        r.trail.append(tuple(robot_pos)); frames.append(render(7, "Moving to box", "move_to_pose(box_03)"))
    # 8: pick box + carry
    carrying_box=True
    for _ in range(5): frames.append(render(8, "Picking box", "execute_skill(pick, box_03)"))
    s=robot_pos.copy()
    for i in range(16):
        t=(i+1)/16; robot_pos=s*(1-t)+np.array([shelf_pos[0]-0.4, shelf_pos[1]])*t
        r.trail.append(tuple(robot_pos)); frames.append(render(8, "Carrying → shelf", "move_to_pose(shelf_A)"))
    # 9: place
    carrying_box=False; box_pos=shelf_pos.copy()
    for _ in range(10): frames.append(render(9, "Box delivered ✓", "execute_skill(place, shelf_A)"))
    for _ in range(14): frames.append(frames[-1].copy())

    out = output_dir / "t2_locked_door.gif"
    encode_frames_to_gif(frames, out, fps=8)
    print(f"  Saved: {out} ({len(frames)} frames, {out.stat().st_size//1024}KB)")
    return out


def main():
    output_dir = REPO / "docs" / "media"
    output_dir.mkdir(parents=True, exist_ok=True)
    gif1 = record_t3(output_dir)
    gif2 = record_t2(output_dir)
    print(f"\nDone!\n  {gif1}\n  {gif2}")


if __name__ == "__main__":
    main()
