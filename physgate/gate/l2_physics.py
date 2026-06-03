"""Sim-Gate Layer 2: parallel physics validation in Isaac Lab (C11).

Given N candidate plans, runs each one in its own Isaac Lab env (all envs reset
to an IDENTICAL initial state via gate/reset_workaround.py), simultaneously:

1. each plan is compiled to a navigation-routed trajectory/mission
   (``gate/trajectory.py`` — every move_to_pose goes through the deterministic
   A* path planner, so routes ALWAYS avoid obstacles),
2. all N robots execute their plan while the scene objects (the box) simulate
   dynamically on the GPU,
3. physics outcomes are read back per env:
     - task success  = the box physically ended up resting on the shelf
       (after the placement release, dynamics decide whether it stays or
       slides/bounces off — fast, sloppy plans drop it),
     - collisions    = path segments sweeping through static obstacles
       (kinematic mode) / obstacle-proximity events on the walked path (policy mode),
     - time / energy = from the executed trajectory.

What L2 validates (REBUILD.md §2): the AGENT's plan quality — decomposition
correctness, step ordering, preconditions, and physical outcome (placement
stability). It does NOT validate route geometry; the navigation layer
guarantees that for every plan.

IMPORTANT: import only after SimulationApp launch.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

from physgate.gate.reset_workaround import reset_scene_to_identical_state
from physgate.gate.schemas import FailureCode, FailureReport, GateLayer, Scene, Violation
from physgate.gate.scoring import PhysicsResult
from physgate.gate.trajectory import (  # noqa: F401  (re-exports for existing callers)
    PLACE_DROP_HEIGHT,
    RELEASE_VELOCITY_GAIN,
    ROBOT_MASS_KG,
    SCAN_HOLD_STEPS,
    SETTLE_STEPS,
    SKILL_HOLD_STEPS,
    PlanTrajectory,
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


#: How close (m) the robot must be to the shelf FOOTPRINT edge to place onto it:
#: max plan standoff (0.5) + waypoint arrival tolerance (0.35) + slack.
PLACE_REACH_M = 1.0


# -------------------------------------------------------------------- rollout


def rollout_plans(world: FetchSimWorld, plans: list[Plan]) -> list[PhysicsResult]:
    """Run N plans in N parallel Isaac Lab envs from an identical initial state."""
    if len(plans) > world.num_envs:
        raise ValueError(f"{len(plans)} plans but only {world.num_envs} parallel envs")

    device = world.device
    num_active = len(plans)

    # 1. identical initial state across every env (bug #2133 workaround)
    reset_scene_to_identical_state(world.scene, world.sim)

    # 2. compile plans to navigation-routed trajectories; plans whose navigation
    # is impossible (unknown target / unreachable goal) fail explicitly instead of
    # crashing the gate (REBUILD.md Phase 3 "goal navigation cannot reach")
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

    # pad: robots that finish early hold their final pose
    def pose_at(traj: PlanTrajectory, idx: int) -> tuple[np.ndarray, float]:
        i = min(idx, traj.duration_steps - 1)
        return traj.positions[i], traj.yaws[i]

    # event lookup per env: step_index -> [events]
    event_map: list[dict[int, list[TrajectoryEvent]]] = []
    for traj in trajectories:
        m: dict[int, list[TrajectoryEvent]] = {}
        for ev in traj.events if traj is not None else []:
            m.setdefault(ev.step_index, []).append(ev)
        event_map.append(m)

    env_origins = world.env_origins.cpu().numpy()
    carrying = [False] * num_active

    # 3. step all envs in lockstep
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
                # inactive envs hold the spawn pose
                robot_pos[env_idx] = np.array(SCENE_LAYOUT["go2"]) + env_origins[env_idx]

        world.write_robot_poses(
            torch.tensor(robot_pos, dtype=torch.float32, device=device),
            torch.tensor(robot_quat, dtype=torch.float32, device=device),
        )

        # carried boxes ride with their robot; releases hand the box to physics
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
            # teleport box to the release point, then give it the release velocity
            world.write_box_poses(
                torch.tensor(np.array(release_positions), dtype=torch.float32, device=device),
                env_ids=torch.tensor(release_ids, dtype=torch.long, device=device),
            )
            world.release_boxes(
                torch.tensor(np.array(release_velocities), dtype=torch.float32, device=device),
                env_ids=torch.tensor(release_ids, dtype=torch.long, device=device),
            )

        world.step()

    # 4. read back physics outcomes
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

        # time: trajectory length minus the settle tail
        active_steps = traj.duration_steps - SETTLE_STEPS
        completion_time = max(active_steps, 0) * world.dt
        # energy proxy: 1/2 m v^2 integrated over motion steps
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
    the base is fully dynamic, and physics decides every outcome:

        success    = mission completed AND box physically resting on the shelf
                     AND the robot never fell over
        collisions = proximity events between the robot's ACTUAL walked path and
                     the obstacle footprint
        time       = simulated seconds until mission completion (or timeout)
        energy     = sum |joint torque * joint velocity| * dt (real actuation energy)

    Args:
        on_control_step: optional hook ``(control_step, world, state) -> None``
            invoked after each control step (state = {"carrying", "mission_index",
            "completed"} lists). Used by the recording pipeline
            (benchmarks/rebuild/record_rollout.py) to capture camera frames and
            trajectory samples without duplicating this control loop.
    """
    from physgate.gate.reset_workaround import reset_scene_to_identical_state
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

    # compile plans -> missions; plans referencing unknown objects get an
    # explicit failure instead of a silently degraded mission (D-016)
    missions: list[list[tuple[str, object, float]]] = []
    #: env index -> (violation type, detail) for plans that cannot be compiled:
    #: unknown objects (hallucinated ids) or navigation-unreachable goals
    compile_errors: dict[int, tuple[str, str]] = {}
    for idx, p in enumerate(plans):
        try:
            missions.append(compile_mission(p, SCENE_LAYOUT))
        except UnknownTargetError as exc:
            missions.append([])
            compile_errors[idx] = ("unknown_object", str(exc))
        except PathPlannerError as exc:
            # REBUILD.md Phase 3 "goal navigation cannot reach": a clean
            # infeasible verdict, not a crash — the orchestrator escalates
            missions.append([])
            compile_errors[idx] = ("infeasible_navigation", str(exc))
    mission_index = [0] * num_active
    carrying = [False] * num_active
    wait_counters = [0] * num_active
    completed = [False] * num_active
    fell_over = [False] * num_active
    stuck = [False] * num_active
    completion_time = [None] * num_active
    # the speed the plan commanded for the most recent goto: at placement, the
    # box inherits this approach speed (momentum transfer, REBUILD.md Phase 2)
    last_goto_speed = [0.0] * num_active
    energy = torch.zeros(num_envs, device=device)

    # plans that failed mission compilation never simulate: mark them done so
    # the control loop does not wait 90 s for envs that will never move
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

    # stuck detection: an env that makes no positional progress for this many
    # seconds while it still has goto work left is physically blocked
    stuck_window_steps = int(10.0 / control_dt)
    last_progress_pos = (world.robot.data.root_pos_w - env_origins).clone()

    # diagnostic tracing for the first control steps (PHYSGATE_DEBUG_L2=1)
    debug_l2 = os.environ.get("PHYSGATE_DEBUG_L2") == "1"
    if debug_l2:
        print(
            f"[L2 debug] {num_active} plans, control_dt={control_dt}, "
            f"max_control_steps={max_control_steps}"
        )
        for i, m in enumerate(missions):
            print(
                f"[L2 debug] mission {i} ({plans[i].plan_id}): "
                f"{[(e[0], e[1] if isinstance(e[1], str) else np.round(e[1], 2).tolist()) for e in m]}"
            )

    for control_step in range(max_control_steps):
        robot_pos_local = world.robot.data.root_pos_w - env_origins
        yaws = base_yaws(world.robot)

        if debug_l2 and control_step < 3:
            print(
                f"[L2 debug] step {control_step}: robot_pos_local[:num_active,:2]="
                f"{np.round(robot_pos_local[:num_active, :2].cpu().numpy(), 2).tolist()} "
                f"mission_index={mission_index}"
            )

        # ---- stuck detection (every 10 s of sim time) ----
        if control_step > 0 and control_step % stuck_window_steps == 0:
            moved = torch.norm(robot_pos_local[:, :2] - last_progress_pos[:, :2], dim=-1)
            for i in range(num_active):
                if completed[i] or fell_over[i] or stuck[i]:
                    continue
                # only goto steps require movement
                if (
                    mission_index[i] < len(missions[i])
                    and missions[i][mission_index[i]][0] == "goto"
                ):
                    if float(moved[i]) < 0.15:
                        stuck[i] = True
                        if debug_l2:
                            goal = missions[i][mission_index[i]][1]
                            print(
                                f"[L2 debug] STUCK env {i} ({plans[i].plan_id}) at "
                                f"t={control_step * control_dt:.1f}s: "
                                f"pos={np.round(robot_pos_local[i, :2].cpu().numpy(), 2).tolist()} "
                                f"goal={np.round(np.asarray(goal, dtype=float), 2).tolist()} "
                                f"mission_index={mission_index[i]} moved={float(moved[i]):.3f} "
                                f"yaw={float(yaws[i]):.2f} carrying={carrying[i]} "
                                f"upright_z={float(world.robot.data.projected_gravity_b[i, 2]):.2f}"
                            )
            if debug_l2:
                print(
                    f"[L2 debug] t={control_step * control_dt:.0f}s "
                    f"mission_index={mission_index} "
                    f"moved={[round(float(moved[i]), 2) for i in range(num_active)]} "
                    f"pos={np.round(robot_pos_local[:num_active, :2].cpu().numpy(), 2).tolist()}"
                )
            last_progress_pos = robot_pos_local.clone()

        # ---- per-env mission state machine -> goals/speeds ----
        goals = robot_pos_local.clone()
        speeds = torch.zeros(num_envs, device=device)
        # which envs actually set a navigation goal THIS control step — only
        # these may advance their mission via the arrival check below (an env
        # whose pick/place just advanced to a goto would otherwise "arrive"
        # instantly at its own position and skip the goto entirely)
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
                mission_index[i] += 1  # advance regardless; failed pick -> no box -> task fails
            elif kind == "place":
                if carrying[i]:
                    carrying[i] = False
                    # release above the shelf only if the robot actually got there:
                    # within placing reach of the shelf FOOTPRINT edge (standoffs
                    # are measured from the footprint since the nav rebuild)
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
                    # else: box is dropped where the robot stands -> physics -> fail

                    # momentum transfer (REBUILD.md Phase 2): the box inherits the
                    # plan's commanded approach speed along the robot's heading,
                    # matching the kinematic rollout — write_box_poses zeroes the
                    # velocity, so WITHOUT this release the box always dropped
                    # dead and "fast, sloppy placements fail" never held.
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
            # mission finished?
            if mission_index[i] >= len(missions[i]):
                completed[i] = True
                completion_time[i] = control_step * control_dt

        # ---- navigation + policy ----
        commands, arrived = navigator.velocity_commands(robot_pos_local, yaws, goals, speeds)
        # only envs that were actually navigating this step may advance on arrival
        for i in range(num_active):
            if not navigating[i]:
                continue
            if completed[i] or fell_over[i] or stuck[i] or mission_index[i] >= len(missions[i]):
                continue
            if bool(arrived[i]):
                mission_index[i] += 1
                # mission finished on a final goto?
                if mission_index[i] >= len(missions[i]):
                    completed[i] = True
                    completion_time[i] = control_step * control_dt

        targets = controller.joint_position_targets(world.robot, commands)
        world.apply_joint_targets(targets)

        # carried boxes ride ahead of and above their robot — same formula as
        # the kinematic rollout and SimBackend. Carrying directly overhead puts
        # the box where the pitching trunk contacts it; a kinematically-written
        # box acts as an immovable obstacle and crushes/stalls the robot (D-018).
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

        # ---- step physics (decimation) ----
        for _ in range(POLICY_DECIMATION):
            world.step()

        # ---- monitoring ----
        # energy: |tau . qdot| * dt
        torque = world.robot.data.applied_torque
        joint_vel = world.robot.data.joint_vel
        energy += (torque * joint_vel).abs().sum(dim=-1) * control_dt

        # falls: projected gravity z close to -1 means upright
        upright = world.robot.data.projected_gravity_b[:, 2] < -0.6
        for i in range(num_active):
            if not bool(upright[i]) and not completed[i]:
                fell_over[i] = True

        # obstacle proximity on the actual path
        pos_local = world.robot.data.root_pos_w - env_origins
        inside = ((pos_local[:, :2] - obstacle_center).abs() < obstacle_half).all(dim=-1)
        proximity_history.append(inside.clone())

        # recording / inspection hook (benchmarks/rebuild/record_rollout.py)
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

    # ---- outcomes ----
    final_box_positions = (world.box.data.root_pos_w - env_origins).cpu().numpy()
    proximity = torch.stack(proximity_history, dim=0).cpu().numpy()  # (T, num_envs)

    results: list[PhysicsResult] = []
    for i, plan in enumerate(plans):
        # plans rejected at mission compilation: explicit failure (unknown
        # object or navigation-unreachable goal), never a crash
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

        # count proximity entry events on the walked path
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


# ------------------------------------------------------------------ L2Fn glue


class IsaacL2Gate:
    """L2Fn-compatible callable backed by Isaac Lab parallel physics (kinematic)."""

    def __init__(self, world: FetchSimWorld):
        self._world = world

    def __call__(self, plans: list[Plan], scene: Scene) -> list[PhysicsResult]:
        return rollout_plans(self._world, plans)

    # keep a friendly name for the demo report
    __name__ = "isaac_l2"


class PolicyL2Gate:
    """L2Fn-compatible callable: parallel physics with the trained walking policy."""

    def __init__(self, world: FetchSimWorld, policy_path):
        self._world = world
        self._policy_path = policy_path

    def __call__(self, plans: list[Plan], scene: Scene) -> list[PhysicsResult]:
        return rollout_plans_with_policy(self._world, plans, self._policy_path)

    __name__ = "isaac_l2_policy"
