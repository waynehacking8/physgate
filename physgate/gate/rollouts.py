"""Rollout functions: execute plans in Isaac Lab parallel physics envs.

Extracted from l2_physics.py for file-size hygiene. Two rollout modes:

* :func:`rollout_plans` — kinematic: the robot's base is teleported along the
  A*-routed trajectory each sim step. Fast, deterministic, no policy needed.
* :func:`rollout_plans_with_policy` — policy-driven: the Go2 locomotion policy
  tracks velocity commands from a waypoint navigator. Fully dynamic base,
  real joint torques, physics decides every outcome.

Both return ``list[PhysicsResult]`` — one per plan.
"""

from __future__ import annotations

import logging
import math
import os

import numpy as np
import torch

from physgate.gate.reset_workaround import reset_scene_to_identical_state
from physgate.gate.schemas import FailureCode, FailureReport, GateLayer, Violation
from physgate.gate.scoring import PhysicsResult
from physgate.gate.trajectory import (
    PLACE_DROP_HEIGHT,
    PlanTrajectory,
    RELEASE_VELOCITY_GAIN,
    ROBOT_MASS_KG,
    SETTLE_STEPS,
    TrajectoryEvent,
    UnknownTargetError,
    box_on_shelf,
    compile_mission,
    count_path_collisions,
    synthesize_base_trajectory,
    yaw_to_quat,
)
from physgate.nav.path_planner import PathPlannerError
from physgate.planner.schemas import Plan
from physgate.world.fetch_scene import FetchSimWorld
from physgate.world.layout import (
    BOX_SIZE,
    CARRY_OFFSET,
    OBSTACLE_SIZE,
    ROBOT_COLLISION_RADIUS,
    SCENE_LAYOUT,
    SHELF_SIZE,
    SHELF_TOP_Z,
)

logger = logging.getLogger(__name__)

PLACE_REACH_M = 1.0


# -------------------------------------------------------------------- kinematic


def rollout_plans(world: FetchSimWorld, plans: list[Plan]) -> list[PhysicsResult]:
    """Run N plans in N parallel Isaac Lab envs from an identical initial state."""
    if len(plans) > world.num_envs:
        raise ValueError(f"{len(plans)} plans but only {world.num_envs} parallel envs")

    device = world.device
    num_active = len(plans)

    reset_scene_to_identical_state(world.scene, world.sim)

    trajectories: list[PlanTrajectory | None] = []
    compile_errors: dict[int, tuple[str, str]] = {}
    for idx, p in enumerate(plans):
        try:
            trajectories.append(synthesize_base_trajectory(p, world.dt))
        except UnknownTargetError as exc:
            trajectories.append(None)
            compile_errors[idx] = ("unknown_object", str(exc))
        except PathPlannerError as exc:
            trajectories.append(None)
            compile_errors[idx] = ("infeasible_navigation", str(exc))
    max_steps = max((t.duration_steps for t in trajectories if t is not None), default=0)

    def pose_at(traj: PlanTrajectory, idx: int) -> tuple[np.ndarray, float]:
        """Return the position and yaw at a trajectory step, clamped to bounds."""
        i = min(idx, traj.duration_steps - 1)
        return traj.positions[i], traj.yaws[i]

    event_map: list[dict[int, list[TrajectoryEvent]]] = []
    for traj in trajectories:
        m: dict[int, list[TrajectoryEvent]] = {}
        for ev in traj.events if traj is not None else []:
            m.setdefault(ev.step_index, []).append(ev)
        event_map.append(m)

    env_origins = world.env_origins.cpu().numpy()
    carrying = [False] * num_active

    for step_idx in range(max_steps):
        robot_pos = np.zeros((world.num_envs, 3))
        robot_quat = np.zeros((world.num_envs, 4))
        robot_quat[:, 0] = 1.0

        for env_idx in range(world.num_envs):
            if env_idx < num_active and trajectories[env_idx] is not None:
                pos, yaw = pose_at(trajectories[env_idx], step_idx)
                robot_pos[env_idx] = pos + env_origins[env_idx]
                robot_quat[env_idx] = yaw_to_quat(yaw)
            else:
                robot_pos[env_idx] = np.array(SCENE_LAYOUT["go2"]) + env_origins[env_idx]

        world.write_robot_poses(
            torch.tensor(robot_pos, dtype=torch.float32, device=device),
            torch.tensor(robot_quat, dtype=torch.float32, device=device),
        )

        carry_ids, carry_positions = [], []
        release_ids, release_positions, release_velocities = [], [], []
        for env_idx in range(num_active):
            for ev in event_map[env_idx].get(step_idx, []):
                if ev.kind == "attach":
                    carrying[env_idx] = True
                elif ev.kind == "release":
                    carrying[env_idx] = False
                    release_ids.append(env_idx)
                    release_positions.append(np.array(ev.release_position) + env_origins[env_idx])
                    release_velocities.append(np.array(ev.release_velocity))
            if carrying[env_idx]:
                pos, yaw = pose_at(trajectories[env_idx], step_idx)
                forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
                carry_pos = (
                    pos
                    + forward * CARRY_OFFSET[0]
                    + np.array([0.0, 0.0, CARRY_OFFSET[2]])
                    + env_origins[env_idx]
                )
                carry_ids.append(env_idx)
                carry_positions.append(carry_pos)

        if carry_ids:
            world.write_box_poses(
                torch.tensor(np.array(carry_positions), dtype=torch.float32, device=device),
                env_ids=torch.tensor(carry_ids, dtype=torch.long, device=device),
            )
        if release_ids:
            world.write_box_poses(
                torch.tensor(np.array(release_positions), dtype=torch.float32, device=device),
                env_ids=torch.tensor(release_ids, dtype=torch.long, device=device),
            )
            world.release_boxes(
                torch.tensor(np.array(release_velocities), dtype=torch.float32, device=device),
                env_ids=torch.tensor(release_ids, dtype=torch.long, device=device),
            )

        world.step()

    final_box_positions = world.box_positions().cpu().numpy()

    results: list[PhysicsResult] = []
    for env_idx, (plan, traj) in enumerate(zip(plans, trajectories)):
        if traj is None:
            vtype, detail = compile_errors[env_idx]
            results.append(
                PhysicsResult(
                    plan_id=plan.plan_id,
                    success=False,
                    collision_count=0,
                    completion_time_s=0.0,
                    energy_j=0.0,
                    failure=FailureReport(
                        failure_code=FailureCode.PRECONDITION_VIOLATION,
                        layer=GateLayer.PHYSICS,
                        violations=[Violation(type=vtype, detail=detail)],
                        remediation_hint=(
                            "only reference object ids that exist in the scene"
                            if vtype == "unknown_object"
                            else "no collision-free route exists to this target — escalate"
                        ),
                        retryable=vtype == "unknown_object",
                    ),
                )
            )
            continue
        placed = any(ev.kind == "release" for ev in traj.events)
        success = placed and box_on_shelf(final_box_positions[env_idx])
        collisions = count_path_collisions(traj.positions)

        active_steps = traj.duration_steps - SETTLE_STEPS
        completion_time = max(active_steps, 0) * world.dt
        energy = float(0.5 * ROBOT_MASS_KG * np.sum(traj.motion_speeds**2) * world.dt)

        failure: FailureReport | None = None
        if not success:
            box_pos = final_box_positions[env_idx]
            detail = (
                f"box ended at ({box_pos[0]:.2f}, {box_pos[1]:.2f}, {box_pos[2]:.2f}); "
                f"shelf surface is at z={SHELF_TOP_Z:.2f}"
            )
            failure = FailureReport(
                failure_code=FailureCode.GRASP_FAILURE
                if placed
                else FailureCode.PRECONDITION_VIOLATION,
                layer=GateLayer.PHYSICS,
                violations=[
                    Violation(
                        type="placement_failed" if placed else "task_incomplete",
                        detail=detail if placed else "plan never placed the box",
                    )
                ],
                remediation_hint=(
                    "approach the shelf more slowly before placing"
                    if placed
                    else "the plan must navigate to the box, pick it, navigate to the "
                    "shelf, and place it — in that order"
                ),
            )

        results.append(
            PhysicsResult(
                plan_id=plan.plan_id,
                success=success,
                collision_count=collisions,
                completion_time_s=round(completion_time, 2),
                energy_j=round(energy, 2),
                failure=failure,
            )
        )
    return results


# ------------------------------------------------------ policy-driven rollout


def rollout_plans_with_policy(
    world: FetchSimWorld,
    plans: list[Plan],
    policy_path,
    max_sim_time_s: float = 90.0,
    on_control_step=None,
) -> list[PhysicsResult]:
    """Run N plans in N parallel envs with the trained Go2 locomotion policy.

    Unlike :func:`rollout_plans` (kinematic), the robots WALK: the policy tracks
    velocity commands from a waypoint navigator following the A*-planned route,
    the base is fully dynamic, and physics decides every outcome.

    Args:
        on_control_step: optional hook ``(control_step, world, state) -> None``
            invoked after each control step (state = {"carrying", "mission_index",
            "completed"} lists). Used by the recording pipeline
            (benchmarks/rebuild/record_rollout.py) to capture camera frames and
            trajectory samples without duplicating this control loop.
    """
    from physgate.world.locomotion import (
        POLICY_DECIMATION,
        Go2PolicyController,
        WaypointNavigator,
        base_yaws,
    )

    if len(plans) > world.num_envs:
        raise ValueError(f"{len(plans)} plans but only {world.num_envs} parallel envs")

    device = world.device
    num_active = len(plans)
    num_envs = world.num_envs

    reset_scene_to_identical_state(world.scene, world.sim)

    controller = Go2PolicyController(policy_path, num_envs=num_envs, device=device)
    navigator = WaypointNavigator(num_envs=num_envs, device=device)

    missions: list[list[tuple[str, object, float]]] = []
    compile_errors: dict[int, tuple[str, str]] = {}
    for idx, p in enumerate(plans):
        try:
            missions.append(compile_mission(p, SCENE_LAYOUT))
        except UnknownTargetError as exc:
            missions.append([])
            compile_errors[idx] = ("unknown_object", str(exc))
        except PathPlannerError as exc:
            missions.append([])
            compile_errors[idx] = ("infeasible_navigation", str(exc))
    mission_index = [0] * num_active
    carrying = [False] * num_active
    wait_counters = [0] * num_active
    completed = [False] * num_active
    fell_over = [False] * num_active
    stuck = [False] * num_active
    completion_time = [None] * num_active
    last_goto_speed = [0.0] * num_active
    energy = torch.zeros(num_envs, device=device)

    for idx in compile_errors:
        completed[idx] = True
        completion_time[idx] = 0.0

    env_origins = world.env_origins
    obstacle_center = torch.tensor(
        SCENE_LAYOUT["obstacle_P"][:2], dtype=torch.float32, device=device
    )
    obstacle_half = torch.tensor(
        [
            OBSTACLE_SIZE[0] / 2 + ROBOT_COLLISION_RADIUS,
            OBSTACLE_SIZE[1] / 2 + ROBOT_COLLISION_RADIUS,
        ],
        dtype=torch.float32,
        device=device,
    )

    control_dt = world.dt * POLICY_DECIMATION
    max_control_steps = int(max_sim_time_s / control_dt)
    proximity_history: list[torch.Tensor] = []

    stuck_window_steps = int(10.0 / control_dt)
    last_progress_pos = (world.robot.data.root_pos_w - env_origins).clone()

    debug_l2 = os.environ.get("PHYSGATE_DEBUG_L2") == "1"
    if debug_l2:
        logger.debug(
            "[L2] %d plans, control_dt=%s, max_control_steps=%d",
            num_active, control_dt, max_control_steps,
        )
        for i, m in enumerate(missions):
            logger.debug(
                "[L2] mission %d (%s): %s",
                i, plans[i].plan_id,
                [(e[0], e[1] if isinstance(e[1], str) else np.round(e[1], 2).tolist()) for e in m],
            )

    for control_step in range(max_control_steps):
        robot_pos_local = world.robot.data.root_pos_w - env_origins
        yaws = base_yaws(world.robot)

        if debug_l2 and control_step < 3:
            logger.debug(
                "[L2] step %d: robot_pos_local[:num_active,:2]=%s mission_index=%s",
                control_step,
                np.round(robot_pos_local[:num_active, :2].cpu().numpy(), 2).tolist(),
                mission_index,
            )

        if control_step > 0 and control_step % stuck_window_steps == 0:
            moved = torch.norm(robot_pos_local[:, :2] - last_progress_pos[:, :2], dim=-1)
            for i in range(num_active):
                if completed[i] or fell_over[i] or stuck[i]:
                    continue
                if (
                    mission_index[i] < len(missions[i])
                    and missions[i][mission_index[i]][0] == "goto"
                ):
                    if float(moved[i]) < 0.15:
                        stuck[i] = True
                        if debug_l2:
                            goal = missions[i][mission_index[i]][1]
                            logger.debug(
                                "[L2] STUCK env %d (%s) at t=%.1fs: "
                                "pos=%s goal=%s mission_index=%d moved=%.3f "
                                "yaw=%.2f carrying=%s upright_z=%.2f",
                                i, plans[i].plan_id, control_step * control_dt,
                                np.round(robot_pos_local[i, :2].cpu().numpy(), 2).tolist(),
                                np.round(np.asarray(goal, dtype=float), 2).tolist(),
                                mission_index[i], float(moved[i]),
                                float(yaws[i]), carrying[i],
                                float(world.robot.data.projected_gravity_b[i, 2]),
                            )
            if debug_l2:
                logger.debug(
                    "[L2] t=%.0fs mission_index=%s moved=%s pos=%s",
                    control_step * control_dt, mission_index,
                    [round(float(moved[i]), 2) for i in range(num_active)],
                    np.round(robot_pos_local[:num_active, :2].cpu().numpy(), 2).tolist(),
                )
            last_progress_pos = robot_pos_local.clone()

        goals = robot_pos_local.clone()
        speeds = torch.zeros(num_envs, device=device)
        navigating = [False] * num_active
        for i in range(num_active):
            if completed[i] or fell_over[i] or stuck[i] or mission_index[i] >= len(missions[i]):
                continue
            kind, payload, speed = missions[i][mission_index[i]]
            if kind == "goto":
                goals[i, 0], goals[i, 1] = float(payload[0]), float(payload[1])
                speeds[i] = speed
                last_goto_speed[i] = speed
                navigating[i] = True
            elif kind == "wait":
                wait_counters[i] += 1
                if wait_counters[i] >= int(payload):
                    wait_counters[i] = 0
                    mission_index[i] += 1
            elif kind == "pick":
                box_pos = (world.box.data.root_pos_w[i] - env_origins[i])[:2]
                reach = torch.norm(box_pos - robot_pos_local[i, :2]).item()
                if reach < 1.0:
                    carrying[i] = True
                mission_index[i] += 1
            elif kind == "place":
                if carrying[i]:
                    carrying[i] = False
                    shelf_xy = torch.tensor(
                        SCENE_LAYOUT["shelf_A"][:2], dtype=torch.float32, device=device
                    )
                    shelf_half = torch.tensor(
                        [SHELF_SIZE[0] / 2, SHELF_SIZE[1] / 2], dtype=torch.float32, device=device
                    )
                    edge_clearance = (
                        ((robot_pos_local[i, :2] - shelf_xy).abs() - shelf_half).max().item()
                    )
                    env_id_tensor = torch.tensor([i], dtype=torch.long, device=device)
                    if edge_clearance < PLACE_REACH_M:
                        release_pos = torch.tensor(
                            [
                                SCENE_LAYOUT["shelf_A"][0],
                                SCENE_LAYOUT["shelf_A"][1],
                                SHELF_TOP_Z + BOX_SIZE[2] / 2 + PLACE_DROP_HEIGHT,
                            ],
                            dtype=torch.float32,
                            device=device,
                        )
                        world.write_box_poses(
                            (release_pos + env_origins[i]).unsqueeze(0),
                            env_ids=env_id_tensor,
                        )

                    yaw_i = float(yaws[i])
                    release_vel = torch.tensor(
                        [
                            math.cos(yaw_i) * last_goto_speed[i] * RELEASE_VELOCITY_GAIN,
                            math.sin(yaw_i) * last_goto_speed[i] * RELEASE_VELOCITY_GAIN,
                            0.0,
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    world.release_boxes(release_vel.unsqueeze(0), env_ids=env_id_tensor)
                mission_index[i] += 1
            if mission_index[i] >= len(missions[i]):
                completed[i] = True
                completion_time[i] = control_step * control_dt

        commands, arrived = navigator.velocity_commands(robot_pos_local, yaws, goals, speeds)
        for i in range(num_active):
            if not navigating[i]:
                continue
            if completed[i] or fell_over[i] or stuck[i] or mission_index[i] >= len(missions[i]):
                continue
            if bool(arrived[i]):
                mission_index[i] += 1
                if mission_index[i] >= len(missions[i]):
                    completed[i] = True
                    completion_time[i] = control_step * control_dt

        targets = controller.joint_position_targets(world.robot, commands)
        world.apply_joint_targets(targets)

        carry_ids = [i for i in range(num_active) if carrying[i]]
        if carry_ids:
            ids = torch.tensor(carry_ids, dtype=torch.long, device=device)
            carry_yaws = yaws[ids]
            forward = torch.stack(
                [
                    torch.cos(carry_yaws),
                    torch.sin(carry_yaws),
                    torch.zeros_like(carry_yaws),
                ],
                dim=-1,
            )
            carry_pos = world.robot.data.root_pos_w[ids] + forward * CARRY_OFFSET[0]
            carry_pos[:, 2] += CARRY_OFFSET[2]
            world.write_box_poses(carry_pos, env_ids=ids)

        for _ in range(POLICY_DECIMATION):
            world.step()

        torque = world.robot.data.applied_torque
        joint_vel = world.robot.data.joint_vel
        energy += (torque * joint_vel).abs().sum(dim=-1) * control_dt

        upright = world.robot.data.projected_gravity_b[:, 2] < -0.6
        for i in range(num_active):
            if not bool(upright[i]) and not completed[i]:
                fell_over[i] = True

        pos_local = world.robot.data.root_pos_w - env_origins
        inside = ((pos_local[:, :2] - obstacle_center).abs() < obstacle_half).all(dim=-1)
        proximity_history.append(inside.clone())

        if on_control_step is not None:
            on_control_step(
                control_step,
                world,
                {
                    "carrying": list(carrying),
                    "mission_index": list(mission_index),
                    "completed": list(completed),
                },
            )

        if all(completed[i] or fell_over[i] or stuck[i] for i in range(num_active)):
            break

    final_box_positions = (world.box.data.root_pos_w - env_origins).cpu().numpy()
    proximity = torch.stack(proximity_history, dim=0).cpu().numpy()

    results: list[PhysicsResult] = []
    for i, plan in enumerate(plans):
        if i in compile_errors:
            vtype, detail = compile_errors[i]
            hint = (
                "only reference object ids that exist in the scene; "
                "do not invent waypoints or locations"
                if vtype == "unknown_object"
                else "no collision-free route exists to this target; "
                "the task is infeasible as posed — escalate"
            )
            results.append(
                PhysicsResult(
                    plan_id=plan.plan_id,
                    success=False,
                    collision_count=0,
                    completion_time_s=0.0,
                    energy_j=0.0,
                    failure=FailureReport(
                        failure_code=FailureCode.PRECONDITION_VIOLATION,
                        layer=GateLayer.PHYSICS,
                        violations=[Violation(type=vtype, detail=detail)],
                        remediation_hint=hint,
                        retryable=vtype == "unknown_object",
                    ),
                )
            )
            continue

        on_shelf = box_on_shelf(final_box_positions[i])
        success = completed[i] and on_shelf and not fell_over[i]

        inside = proximity[:, i]
        entries = int(np.sum(inside[1:] & ~inside[:-1]) + int(inside[0]))

        time_s = completion_time[i] if completion_time[i] is not None else max_sim_time_s

        failure: FailureReport | None = None
        if not success:
            if fell_over[i]:
                detail, vtype = "robot fell over during execution", "robot_fell"
            elif stuck[i]:
                detail, vtype = (
                    "robot made no progress for 10 s while navigating",
                    "blocked",
                )
            elif not completed[i]:
                detail, vtype = "mission timed out", "timeout"
            else:
                box = final_box_positions[i]
                detail = (
                    f"box ended at ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}), not on the shelf"
                )
                vtype = "placement_failed"
            failure = FailureReport(
                failure_code=FailureCode.TIMEOUT if not completed[i] else FailureCode.GRASP_FAILURE,
                layer=GateLayer.PHYSICS,
                violations=[Violation(type=vtype, detail=detail)],
                remediation_hint=(
                    "the plan must pick the box before navigating to the shelf, and "
                    "place it only after arriving — check step ordering and preconditions"
                ),
            )

        results.append(
            PhysicsResult(
                plan_id=plan.plan_id,
                success=success,
                collision_count=entries,
                completion_time_s=round(float(time_s), 2),
                energy_j=round(float(energy[i].item()), 2),
                failure=failure,
            )
        )
    return results
