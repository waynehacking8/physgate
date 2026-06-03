"""Sim-Gate Layer 2: parallel physics validation in Isaac Lab (C11).

Given N candidate plans, runs each one in its own Isaac Lab env (all envs reset
to an IDENTICAL initial state via gate/reset_workaround.py), simultaneously:

1. each plan is compiled to a base-motion trajectory + carry/release events
   (``synthesize_base_trajectory``),
2. all N robots are kinematically driven along their plan's trajectory while
   the scene objects (the box) simulate dynamically on the GPU,
3. physics outcomes are read back per env:
     - task success  = the box physically ended up resting on the shelf
       (after the placement release, dynamics decide whether it stays or
       slides/bounces off — fast, sloppy plans drop it),
     - collisions    = path segments sweeping through static obstacles,
     - time / energy = from the executed trajectory.

MVP scope note (see DECISIONS.md): the robot base is kinematically driven
(no learned locomotion policy yet); the box, contacts, and placement are
fully dynamic. Plan quality discrimination comes from real physics
(placement stability) + swept-path collision checks.

IMPORTANT: import only after SimulationApp launch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from physgate.gate.reset_workaround import reset_scene_to_identical_state
from physgate.gate.schemas import FailureCode, FailureReport, GateLayer, Scene, Violation
from physgate.gate.scoring import PhysicsResult
from physgate.planner.schemas import Plan, ToolName
from physgate.world.fetch_scene import (
    BOX_SIZE,
    CARRY_OFFSET,
    OBSTACLE_SIZE,
    ROBOT_BASE_HEIGHT,
    SCENE_LAYOUT,
    SHELF_SIZE,
    SHELF_TOP_Z,
    FetchSimWorld,
)

# tuning constants
SKILL_HOLD_STEPS = 60          # sim steps the robot pauses for a pick/place
SCAN_HOLD_STEPS = 30           # sim steps for a query_scene pause
SETTLE_STEPS = 240             # sim steps after the last plan step (2 s at 120 Hz)
RELEASE_VELOCITY_GAIN = 2.0    # released box inherits gain * last motion speed
ROBOT_COLLISION_RADIUS = 0.30  # Go2 half-width + margin for swept-path checks
PLACE_DROP_HEIGHT = 0.06       # box released this high above the shelf surface
ROBOT_MASS_KG = 15.0           # Go2 mass, for the energy proxy


# ------------------------------------------------------------------ synthesis


@dataclass
class TrajectoryEvent:
    """Carry/release event at a specific step of a base trajectory."""

    step_index: int
    kind: str                     # "attach" | "release"
    object_id: str
    release_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    release_position: tuple[float, float, float] | None = None


@dataclass
class PlanTrajectory:
    """A plan compiled to a kinematic base trajectory."""

    plan_id: str
    positions: np.ndarray         # (T, 3) base positions, env-local frame
    yaws: np.ndarray              # (T,) base headings
    events: list[TrajectoryEvent] = field(default_factory=list)
    motion_speeds: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (T,) commanded speed

    @property
    def duration_steps(self) -> int:
        return len(self.positions)


def _yaw_to_quat(yaw: float) -> np.ndarray:
    """Heading angle -> wxyz quaternion (rotation about +z)."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def synthesize_base_trajectory(
    plan: Plan,
    dt: float,
    layout: dict[str, tuple[float, float, float]] | None = None,
) -> PlanTrajectory:
    """Compile a Plan into a kinematic base trajectory + carry/release events."""
    lay = layout or SCENE_LAYOUT
    positions: list[np.ndarray] = []
    yaws: list[float] = []
    speeds: list[float] = []
    events: list[TrajectoryEvent] = []

    current = np.array(lay["go2"], dtype=np.float64)
    current[2] = ROBOT_BASE_HEIGHT
    yaw = 0.0
    held_object: str | None = None
    last_speed = 0.0

    def hold(steps: int) -> None:
        for _ in range(steps):
            positions.append(current.copy())
            yaws.append(yaw)
            speeds.append(0.0)

    for step in plan.steps:
        if step.tool == ToolName.QUERY_SCENE:
            hold(SCAN_HOLD_STEPS)

        elif step.tool == ToolName.MOVE_TO_POSE:
            target_id = step.args["target"]
            if target_id not in lay:
                # unknown target: hold in place (L3 should have caught this)
                hold(SCAN_HOLD_STEPS)
                continue
            target = np.array(lay[target_id], dtype=np.float64)
            standoff = float(step.args.get("standoff_m", 0.3))
            speed = max(float(step.args.get("speed", 0.5)), 0.05)

            direction = target[:2] - current[:2]
            distance = float(np.linalg.norm(direction))
            if distance > 1e-6:
                direction = direction / distance
            else:
                direction = np.array([1.0, 0.0])
            goal_xy = target[:2] - direction * standoff
            goal = np.array([goal_xy[0], goal_xy[1], ROBOT_BASE_HEIGHT])
            yaw = math.atan2(direction[1], direction[0])

            travel = float(np.linalg.norm(goal[:2] - current[:2]))
            n_steps = max(int(travel / (speed * dt)), 1)
            for k in range(1, n_steps + 1):
                positions.append(current + (goal - current) * (k / n_steps))
                yaws.append(yaw)
                speeds.append(speed)
            current = goal
            last_speed = speed

        elif step.tool == ToolName.EXECUTE_SKILL:
            skill = step.args.get("skill")
            target_id = step.args.get("target", "")
            hold(SKILL_HOLD_STEPS)
            if skill == "pick":
                events.append(
                    TrajectoryEvent(step_index=len(positions) - 1, kind="attach", object_id=target_id)
                )
                held_object = target_id
            elif skill == "place":
                # release the carried box above the shelf surface, inheriting
                # momentum from the approach speed — physics decides if it stays
                shelf = np.array(lay.get(target_id, lay["shelf_A"]), dtype=np.float64)
                release_pos = (
                    float(shelf[0]),
                    float(shelf[1]),
                    SHELF_TOP_Z + BOX_SIZE[2] / 2 + PLACE_DROP_HEIGHT,
                )
                forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
                release_vel = forward * last_speed * RELEASE_VELOCITY_GAIN
                events.append(
                    TrajectoryEvent(
                        step_index=len(positions) - 1,
                        kind="release",
                        object_id=held_object or target_id,
                        release_velocity=tuple(release_vel.tolist()),
                        release_position=release_pos,
                    )
                )
                held_object = None

    # tail settle so released objects come to rest before readback
    hold(SETTLE_STEPS)

    return PlanTrajectory(
        plan_id=plan.plan_id,
        positions=np.array(positions),
        yaws=np.array(yaws),
        events=events,
        motion_speeds=np.array(speeds),
    )


# ----------------------------------------------------------- geometric checks


def count_path_collisions(
    positions: np.ndarray,
    layout: dict[str, tuple[float, float, float]] | None = None,
    robot_radius: float = ROBOT_COLLISION_RADIUS,
) -> int:
    """Count how many times a base path sweeps through a static obstacle.

    The base is kinematically driven, so obstacle penetration is a *geometric*
    check (a kinematic body does not generate contact responses).

    Note: manipulation targets (the shelf) are NOT obstacles — the robot must
    approach them to act on them. Interaction quality with targets is judged by
    the placement dynamics instead (box release physics).
    """
    lay = layout or SCENE_LAYOUT
    obstacles = {"obstacle_P": OBSTACLE_SIZE}
    collisions = 0
    for obstacle_id, size in obstacles.items():
        center = np.array(lay[obstacle_id][:2])
        half = np.array(size[:2]) / 2 + robot_radius
        inside = np.all(np.abs(positions[:, :2] - center) < half, axis=1)
        # count entries (rising edges) into the collision volume
        entries = np.sum(inside[1:] & ~inside[:-1]) + int(inside[0])
        collisions += int(entries)
    return collisions


def _box_on_shelf(box_pos: np.ndarray, layout: dict | None = None, tol_xy: float = 0.05) -> bool:
    """Did the box physically end up resting on the shelf surface?"""
    lay = layout or SCENE_LAYOUT
    shelf_center = np.array(lay["shelf_A"])
    half_x, half_y = SHELF_SIZE[0] / 2 + tol_xy, SHELF_SIZE[1] / 2 + tol_xy
    expected_z = SHELF_TOP_Z + BOX_SIZE[2] / 2
    return (
        abs(box_pos[0] - shelf_center[0]) <= half_x
        and abs(box_pos[1] - shelf_center[1]) <= half_y
        and abs(box_pos[2] - expected_z) <= 0.08
    )


# -------------------------------------------------------------------- rollout


def rollout_plans(world: FetchSimWorld, plans: list[Plan]) -> list[PhysicsResult]:
    """Run N plans in N parallel Isaac Lab envs from an identical initial state."""
    if len(plans) > world.num_envs:
        raise ValueError(f"{len(plans)} plans but only {world.num_envs} parallel envs")

    device = world.device
    num_active = len(plans)

    # 1. identical initial state across every env (bug #2133 workaround)
    reset_scene_to_identical_state(world.scene, world.sim)

    # 2. compile plans to trajectories
    trajectories = [synthesize_base_trajectory(p, world.dt) for p in plans]
    max_steps = max(t.duration_steps for t in trajectories)

    # pad: robots that finish early hold their final pose
    def pose_at(traj: PlanTrajectory, idx: int) -> tuple[np.ndarray, float]:
        i = min(idx, traj.duration_steps - 1)
        return traj.positions[i], traj.yaws[i]

    # event lookup per env: step_index -> [events]
    event_map: list[dict[int, list[TrajectoryEvent]]] = []
    for traj in trajectories:
        m: dict[int, list[TrajectoryEvent]] = {}
        for ev in traj.events:
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
            if env_idx < num_active:
                pos, yaw = pose_at(trajectories[env_idx], step_idx)
                robot_pos[env_idx] = pos + env_origins[env_idx]
                robot_quat[env_idx] = _yaw_to_quat(yaw)
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
        placed = any(ev.kind == "release" for ev in traj.events)
        success = placed and _box_on_shelf(final_box_positions[env_idx])
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
                failure_code=FailureCode.GRASP_FAILURE if placed else FailureCode.PRECONDITION_VIOLATION,
                layer=GateLayer.PHYSICS,
                violations=[
                    Violation(
                        type="placement_failed" if placed else "task_incomplete",
                        detail=detail if placed else "plan never placed the box",
                    )
                ],
                remediation_hint="approach the shelf more slowly before placing",
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


def _compile_mission(
    plan: Plan, lay: dict[str, tuple[float, float, float]]
) -> list[tuple[str, object, float]]:
    """Compile a Plan into a waypoint mission for the walking robot.

    Mission entries: ("goto", goal_xy_position, speed) | ("pick", object_id, 0)
    | ("place", object_id, 0) | ("wait", duration_steps, 0).
    """
    mission: list[tuple[str, object, float]] = []
    current = np.array(lay["go2"][:2], dtype=np.float64)

    for step in plan.steps:
        if step.tool == ToolName.QUERY_SCENE:
            mission.append(("wait", 25, 0.0))  # ~0.5 s at 50 Hz control
        elif step.tool == ToolName.MOVE_TO_POSE:
            target_id = step.args["target"]
            if target_id not in lay:
                continue
            target = np.array(lay[target_id][:2], dtype=np.float64)
            standoff = float(step.args.get("standoff_m", 0.3))
            # floor the commanded speed at 0.4 m/s: the locomotion policy tracks
            # very low velocity commands poorly (it creeps or stalls); plan
            # "caution" is expressed by the route, not by sub-0.4 m/s speeds
            speed = float(np.clip(float(step.args.get("speed", 0.5)), 0.4, 1.0))
            direction = target - current
            dist = float(np.linalg.norm(direction))
            direction = direction / dist if dist > 1e-6 else np.array([1.0, 0.0])
            goal = target - direction * standoff
            mission.append(("goto", goal, speed))
            current = goal
        elif step.tool == ToolName.EXECUTE_SKILL:
            skill = step.args.get("skill")
            target_id = step.args.get("target", "")
            if skill == "pick":
                mission.append(("pick", target_id, 0.0))
            elif skill == "place":
                mission.append(("place", target_id, 0.0))
    return mission


def rollout_plans_with_policy(
    world: FetchSimWorld,
    plans: list[Plan],
    policy_path,
    max_sim_time_s: float = 90.0,
) -> list[PhysicsResult]:
    """Run N plans in N parallel envs with the trained Go2 locomotion policy.

    Unlike :func:`rollout_plans` (kinematic), the robots WALK: the policy tracks
    velocity commands from a waypoint navigator, the base is fully dynamic, and
    the obstacle physically blocks paths that go through it. Outcomes:

        success    = mission completed AND box physically resting on the shelf
                     AND the robot never fell over
        collisions = proximity events between the robot's ACTUAL walked path and
                     the obstacle footprint
        time       = simulated seconds until mission completion (or timeout)
        energy     = sum |joint torque * joint velocity| * dt (real actuation energy)
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

    missions = [_compile_mission(p, SCENE_LAYOUT) for p in plans]
    mission_index = [0] * num_active
    carrying = [False] * num_active
    wait_counters = [0] * num_active
    completed = [False] * num_active
    fell_over = [False] * num_active
    stuck = [False] * num_active
    completion_time = [None] * num_active
    energy = torch.zeros(num_envs, device=device)

    env_origins = world.env_origins
    obstacle_center = torch.tensor(
        SCENE_LAYOUT["obstacle_P"][:2], dtype=torch.float32, device=device
    )
    obstacle_half = torch.tensor(
        [OBSTACLE_SIZE[0] / 2 + ROBOT_COLLISION_RADIUS, OBSTACLE_SIZE[1] / 2 + ROBOT_COLLISION_RADIUS],
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

    for control_step in range(max_control_steps):
        robot_pos_local = world.robot.data.root_pos_w - env_origins
        yaws = base_yaws(world.robot)

        # ---- stuck detection (every 10 s of sim time) ----
        if control_step > 0 and control_step % stuck_window_steps == 0:
            moved = torch.norm(
                robot_pos_local[:, :2] - last_progress_pos[:, :2], dim=-1
            )
            for i in range(num_active):
                if completed[i] or fell_over[i] or stuck[i]:
                    continue
                # only goto steps require movement
                if mission_index[i] < len(missions[i]) and missions[i][mission_index[i]][0] == "goto":
                    if float(moved[i]) < 0.15:
                        stuck[i] = True
            last_progress_pos = robot_pos_local.clone()

        # ---- per-env mission state machine -> goals/speeds ----
        goals = robot_pos_local.clone()
        speeds = torch.zeros(num_envs, device=device)
        for i in range(num_active):
            if completed[i] or fell_over[i] or stuck[i] or mission_index[i] >= len(missions[i]):
                continue
            kind, payload, speed = missions[i][mission_index[i]]
            if kind == "goto":
                goals[i, 0], goals[i, 1] = float(payload[0]), float(payload[1])
                speeds[i] = speed
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
                    # release above the shelf only if the robot actually got there
                    shelf_xy = torch.tensor(
                        SCENE_LAYOUT["shelf_A"][:2], dtype=torch.float32, device=device
                    )
                    dist_to_shelf = torch.norm(robot_pos_local[i, :2] - shelf_xy).item()
                    if dist_to_shelf < 1.2:
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
                            env_ids=torch.tensor([i], dtype=torch.long, device=device),
                        )
                    # else: box is dropped where the robot stands -> physics -> fail
                mission_index[i] += 1
            # mission finished?
            if mission_index[i] >= len(missions[i]):
                completed[i] = True
                completion_time[i] = control_step * control_dt

        # ---- navigation + policy ----
        commands, arrived = navigator.velocity_commands(robot_pos_local, yaws, goals, speeds)
        # envs in goto state that arrived advance their mission
        for i in range(num_active):
            if completed[i] or fell_over[i] or stuck[i] or mission_index[i] >= len(missions[i]):
                continue
            kind, _, _ = missions[i][mission_index[i]]
            if kind == "goto" and bool(arrived[i]):
                mission_index[i] += 1

        targets = controller.joint_position_targets(world.robot, commands)
        world.apply_joint_targets(targets)

        # carried boxes ride above their robot
        carry_ids = [i for i in range(num_active) if carrying[i]]
        if carry_ids:
            ids = torch.tensor(carry_ids, dtype=torch.long, device=device)
            carry_pos = world.robot.data.root_pos_w[ids].clone()
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
        inside = (
            (pos_local[:, :2] - obstacle_center).abs() < obstacle_half
        ).all(dim=-1)
        proximity_history.append(inside.clone())

        if all(completed[i] or fell_over[i] or stuck[i] for i in range(num_active)):
            break

    # ---- outcomes ----
    final_box_positions = (world.box.data.root_pos_w - env_origins).cpu().numpy()
    proximity = torch.stack(proximity_history, dim=0).cpu().numpy()  # (T, num_envs)

    results: list[PhysicsResult] = []
    for i, plan in enumerate(plans):
        on_shelf = _box_on_shelf(final_box_positions[i])
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
                    "robot made no progress for 10 s — physically blocked (obstacle in path)",
                    "blocked",
                )
            elif not completed[i]:
                detail, vtype = "mission timed out (likely blocked by obstacle)", "timeout"
            else:
                box = final_box_positions[i]
                detail = f"box ended at ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}), not on the shelf"
                vtype = "placement_failed"
            failure = FailureReport(
                failure_code=FailureCode.TIMEOUT if not completed[i] else FailureCode.GRASP_FAILURE,
                layer=GateLayer.PHYSICS,
                violations=[Violation(type=vtype, detail=detail)],
                remediation_hint="choose a route that avoids the obstacle and approach the shelf closely",
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
