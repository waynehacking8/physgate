"""Isaac Sim execution backend (E17).

SimBackend implements the same WorldBackend protocol as MockWorldBackend, but
every action is physically executed in Isaac Sim (env 0 of the shared
FetchSimWorld):

* ``move_to_pose``  — the Go2 base is driven along the path; the swept path is
  checked against static obstacles.
* ``execute_skill('pick')``   — the box is attached (kinematic carry) after
  verifying the robot is physically close enough to reach it.
* ``execute_skill('place')``  — the box is released above the shelf with the
  approach momentum; physics settles it; success is read back from the box's
  actual resting position. A sloppy (fast) placement can physically fail here.

The symbolic scene graph is updated in lockstep so L3 precondition rechecks
(plan_executor) and query_scene stay consistent with the simulation.

IMPORTANT: import only after SimulationApp launch.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from physgate.executor.backend import MockWorldBackend
from physgate.gate.reset_workaround import reset_scene_to_identical_state
from physgate.gate.schemas import Scene
from physgate.gate.trajectory import (
    PLACE_DROP_HEIGHT,
    SKILL_HOLD_STEPS,
    box_on_shelf,
    count_path_collisions,
    yaw_to_quat,
)
from physgate.nav.path_planner import PathPlannerError, plan_standoff_route
from physgate.world.fetch_scene import FetchSimWorld, get_shared_world
from physgate.world.layout import (
    BOX_SIZE,
    CARRY_OFFSET,
    ROBOT_BASE_HEIGHT,
    ROBOT_COLLISION_RADIUS,
    SCENE_LAYOUT,
    SHELF_TOP_Z,
    navigation_obstacles,
    target_half_extents,
)

# how close (m, horizontal) the robot base must be to pick an object
PICK_REACH_M = 0.9
EXEC_ENV = 0  # the executor always uses env 0 of the shared world


class SimBackend:
    """WorldBackend executing against Isaac Sim (env 0), tracking symbolic state."""

    def __init__(self, scene: Scene, world: FetchSimWorld | None = None):
        """Initialize with symbolic state and an Isaac Sim world for env 0."""
        self._symbolic = MockWorldBackend(scene)
        self._world = world or get_shared_world()
        self._device = self._world.device
        # physical state for env 0
        reset_scene_to_identical_state(self._world.scene, self._world.sim)
        self._robot_pos = np.array(SCENE_LAYOUT["go2"], dtype=np.float64)
        self._robot_pos[2] = ROBOT_BASE_HEIGHT
        self._yaw = 0.0
        self._carrying: str | None = None
        self._last_speed = 0.0

    # ----- WorldBackend protocol -----

    def get_scene(self) -> Scene:
        """Return the current symbolic scene from the inner backend."""
        return self._symbolic.get_scene()

    def move_to_pose(self, target: str, standoff_m: float = 0.3, **kwargs: Any) -> dict[str, Any]:
        """Drive the robot along an A*-planned route to a standoff pose near the target."""
        symbolic = self._symbolic.move_to_pose(target=target, standoff_m=standoff_m, **kwargs)
        if not symbolic["success"]:
            return symbolic
        if target not in SCENE_LAYOUT:
            return {"success": False, "error": f"'{target}' has no physical location in the scene"}

        speed = max(float(kwargs.get("speed", 0.5)), 0.05)
        target_pos = np.array(SCENE_LAYOUT[target], dtype=np.float64)

        # deterministic navigation: route around obstacles to a standoff pose
        # near the target (REBUILD.md Phase 1 — no straight-line driving)
        try:
            route = plan_standoff_route(
                (float(self._robot_pos[0]), float(self._robot_pos[1])),
                (float(target_pos[0]), float(target_pos[1])),
                standoff=standoff_m,
                obstacles=navigation_obstacles(),
                robot_radius=ROBOT_COLLISION_RADIUS,
                target_half_extents=target_half_extents(target),
            )
        except PathPlannerError as exc:
            # navigation infeasibility is a failed action the orchestrator can
            # react to (replan / escalate), never a crash mid-execution
            return {"success": False, "error": f"no collision-free route to '{target}': {exc}"}

        # drive the base along every route segment, carrying the box if held
        path_points: list[np.ndarray] = []
        travel = 0.0
        for waypoint in route[1:]:
            goal = np.array([waypoint[0], waypoint[1], ROBOT_BASE_HEIGHT])
            direction = goal[:2] - self._robot_pos[:2]
            seg_len = float(np.linalg.norm(direction))
            if seg_len < 1e-9:
                continue
            self._yaw = math.atan2(direction[1], direction[0])
            n_steps = max(int(seg_len / (speed * self._world.dt)), 1)
            start = self._robot_pos.copy()
            for k in range(n_steps):
                point = start + (goal - start) * ((k + 1) / n_steps)
                path_points.append(point)
                self._write_robot(point)
                if self._carrying:
                    self._write_carried_box(point)
                self._world.step()
            self._robot_pos = goal
            travel += seg_len
        self._last_speed = speed

        path = np.array(path_points) if path_points else self._robot_pos.reshape(1, 3)
        collisions = count_path_collisions(path)
        return {
            **symbolic,
            "physical": True,
            "path_length_m": round(travel, 3),
            "route_waypoints": len(route) - 1,
            "swept_collisions": collisions,
            "sim_steps": len(path_points),
        }

    def execute_skill(self, skill: str, target: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a pick or place skill physically in Isaac Sim."""
        if skill == "pick":
            return self._pick(target, **kwargs)
        if skill == "place":
            return self._place(target, **kwargs)
        return {"success": False, "error": f"unknown skill '{skill}' (available: pick, place)"}

    # ----- skills -----

    def _pick(self, target: str, **kwargs: Any) -> dict[str, Any]:
        # physical reach check BEFORE the symbolic update
        box_pos = self._world.box_positions()[EXEC_ENV].cpu().numpy()
        reach = float(np.linalg.norm(box_pos[:2] - self._robot_pos[:2]))
        if reach > PICK_REACH_M:
            return {
                "success": False,
                "error": f"physical reach check failed: box is {reach:.2f} m away (max {PICK_REACH_M} m)",
            }
        symbolic = self._symbolic.execute_skill(skill="pick", target=target, **kwargs)
        if not symbolic["success"]:
            return symbolic

        # attach: carry the box kinematically from now on
        self._carrying = target
        for _ in range(SKILL_HOLD_STEPS):
            self._write_robot(self._robot_pos)
            self._write_carried_box(self._robot_pos)
            self._world.step()
        return {**symbolic, "physical": True, "reach_m": round(reach, 3)}

    def _place(self, target: str, **kwargs: Any) -> dict[str, Any]:
        symbolic = self._symbolic.execute_skill(skill="place", target=target, **kwargs)
        if not symbolic["success"]:
            return symbolic

        # release the box above the shelf with the approach momentum
        shelf = np.array(SCENE_LAYOUT.get(target, SCENE_LAYOUT["shelf_A"]), dtype=np.float64)
        release_pos = np.array(
            [shelf[0], shelf[1], SHELF_TOP_Z + BOX_SIZE[2] / 2 + PLACE_DROP_HEIGHT]
        )
        forward = np.array([math.cos(self._yaw), math.sin(self._yaw), 0.0])
        release_vel = forward * self._last_speed * 2.0

        origin = self._world.env_origins[EXEC_ENV].cpu().numpy()
        env_ids = torch.tensor([EXEC_ENV], dtype=torch.long, device=self._device)
        self._world.write_box_poses(
            torch.tensor(np.array([release_pos + origin]), dtype=torch.float32, device=self._device),
            env_ids=env_ids,
        )
        self._world.release_boxes(
            torch.tensor(np.array([release_vel]), dtype=torch.float32, device=self._device),
            env_ids=env_ids,
        )
        self._carrying = None

        # physics settles the box; then read where it ACTUALLY ended up
        for _ in range(240):
            self._write_robot(self._robot_pos)
            self._world.step()
        final_box = self._world.box_positions()[EXEC_ENV].cpu().numpy()
        physically_placed = box_on_shelf(final_box)

        if not physically_placed:
            return {
                "success": False,
                "error": (
                    f"physics: box did not stay on the shelf "
                    f"(ended at z={final_box[2]:.2f}, shelf top z={SHELF_TOP_Z:.2f}); "
                    "approach was probably too fast"
                ),
                "physical": True,
                "box_final_position": [round(float(v), 3) for v in final_box],
            }
        return {
            **symbolic,
            "physical": True,
            "box_final_position": [round(float(v), 3) for v in final_box],
        }

    # ----- sim helpers -----

    def _write_robot(self, position: np.ndarray) -> None:
        """Write env-0 robot pose (other envs hold their current pose)."""
        origins = self._world.env_origins.cpu().numpy()
        all_pos = self._world.robot.data.root_pos_w.clone()
        all_quat = self._world.robot.data.root_quat_w.clone()
        all_pos[EXEC_ENV] = torch.tensor(
            position + origins[EXEC_ENV], dtype=torch.float32, device=self._device
        )
        all_quat[EXEC_ENV] = torch.tensor(
            yaw_to_quat(self._yaw), dtype=torch.float32, device=self._device
        )
        self._world.write_robot_poses(all_pos, all_quat)

    # ----- new tools: symbolic fallback in Isaac env (H1) -----

    def open_door(self, door_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._symbolic.open_door(door_id=door_id, **kwargs)

    def unlock_door(self, door_id: str, key_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._symbolic.unlock_door(door_id=door_id, key_id=key_id, **kwargs)

    def press_button(self, button_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._symbolic.press_button(button_id=button_id, **kwargs)

    def call_elevator(self, elevator_id: str, target_floor: int, **kwargs: Any) -> dict[str, Any]:
        return self._symbolic.call_elevator(elevator_id=elevator_id, target_floor=target_floor, **kwargs)

    def push_object(self, object_id: str, direction: str, **kwargs: Any) -> dict[str, Any]:
        return self._symbolic.push_object(object_id=object_id, direction=direction, **kwargs)

    def inspect_object(self, object_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._symbolic.inspect_object(object_id=object_id, **kwargs)

    def request_assistance(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self._symbolic.request_assistance(message=message, **kwargs)

    # ----- sim helpers -----

    def _write_carried_box(self, robot_position: np.ndarray) -> None:
        forward = np.array([math.cos(self._yaw), math.sin(self._yaw), 0.0])
        carry_pos = (
            robot_position
            + forward * CARRY_OFFSET[0]
            + np.array([0.0, 0.0, CARRY_OFFSET[2]])
            + self._world.env_origins[EXEC_ENV].cpu().numpy()
        )
        self._world.write_box_poses(
            torch.tensor(np.array([carry_pos]), dtype=torch.float32, device=self._device),
            env_ids=torch.tensor([EXEC_ENV], dtype=torch.long, device=self._device),
        )


# --------------------------------------------------------- policy-driven executor


class PolicySimExecutor:
    """ExecutorFn: physically execute the selected plan with the walking policy.

    Unlike :class:`SimBackend` (kinematic, tool-by-tool), this executor runs the
    whole plan as one locomotion mission via
    :func:`physgate.gate.l2_physics.rollout_plans_with_policy` on env 0 — the
    robot actually walks, the obstacle actually blocks, the placement actually
    settles. The orchestrator consumes its structured result dict directly.
    """

    def __init__(self, world: FetchSimWorld, policy_path):
        """Initialize with a shared sim world and trained policy checkpoint."""
        self._world = world
        self._policy_path = policy_path

    def __call__(self, plan, scene: Scene) -> dict[str, Any]:
        """Execute the plan with the walking policy and return structured results."""
        from physgate.gate.l2_physics import rollout_plans_with_policy

        result = rollout_plans_with_policy(self._world, [plan], self._policy_path)[0]
        return {
            "success": result.success,
            "steps_completed": len(plan.steps) if result.success else 0,
            "steps_total": len(plan.steps),
            "error": (
                result.failure.violations[0].detail
                if (result.failure and result.failure.violations)
                else None
            ),
            "physics": {
                "completion_time_s": result.completion_time_s,
                "energy_j": result.energy_j,
                "collision_count": result.collision_count,
            },
        }
