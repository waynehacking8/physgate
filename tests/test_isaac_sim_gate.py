"""Isaac Sim integration tests: C10 reset workaround, C11 L2 physics, C13 USD
semantics, E17 SimBackend.

These tests launch a headless SimulationApp (once per module) and share one
FetchSimWorld with 8 parallel envs. They are skipped automatically outside the
Isaac venv. Expect a few minutes of wall-clock for the full module.

Run:  source ~/env_isaaclab/bin/activate && pytest tests/test_isaac_sim_gate.py -v
"""

import numpy as np
import pytest

# skip the whole module when Isaac Sim is not available (pure-logic venv)
pytest.importorskip("isaacsim", reason="Isaac Sim not installed in this venv")

pytestmark = pytest.mark.isaac

# ---------------------------------------------------------------------------
# Launch the Omniverse app at module level via AppLauncher (the Isaac Lab test
# pattern). Raw SimulationApp inside a fixture silently exits pytest because
# kit tries to parse pytest's CLI args; AppLauncher with explicit kwargs does
# not touch sys.argv.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

simulation_app = AppLauncher(headless=True).app

NUM_ENVS = 8
TASK = "put the fallen box back on shelf A"


@pytest.fixture(scope="module")
def sim_world():
    """Build the shared 8-env fetch scene (one per test module / process)."""
    from physgate.world.fetch_scene import FetchSimWorld

    return FetchSimWorld(num_envs=NUM_ENVS)


@pytest.fixture()
def demo_scene():
    from physgate.examples_lib.fetch_and_place import build_demo_scene

    return build_demo_scene()


# ------------------------------------------------------------------ C10 reset


def test_reset_workaround_makes_all_envs_identical(sim_world):
    """Bug #2133 workaround: after reset, all 8 envs are in the same state."""
    from physgate.gate.reset_workaround import verify_identical_reset

    ok, deviation = verify_identical_reset(sim_world.scene, sim_world.sim, tolerance=1e-3)
    assert ok, f"envs diverged after identical reset: max deviation {deviation:.6f} m"


def test_reset_workaround_is_repeatable(sim_world):
    """Resetting twice gives the same state both times (determinism)."""
    from physgate.gate.reset_workaround import (
        max_state_deviation,
        reset_scene_to_identical_state,
    )

    reset_scene_to_identical_state(sim_world.scene, sim_world.sim)
    box_first = sim_world.box_positions().cpu().numpy().copy()

    reset_scene_to_identical_state(sim_world.scene, sim_world.sim)
    box_second = sim_world.box_positions().cpu().numpy().copy()

    assert np.allclose(box_first, box_second, atol=1e-3)
    assert max_state_deviation(sim_world.scene) <= 1e-3


# ------------------------------------------------------------- C11 L2 physics


def test_l2_rollout_returns_result_per_plan(sim_world, demo_scene):
    from physgate.gate.l2_physics import rollout_plans
    from physgate.planner.planner import MockPlanner

    plans = MockPlanner()(TASK, demo_scene, 4, None)
    results = rollout_plans(sim_world, plans)

    assert len(results) == 4
    assert {r.plan_id for r in results} == {p.plan_id for p in plans}
    # every result carries physics metrics
    for result in results:
        assert result.completion_time_s >= 0
        assert result.energy_j >= 0


def test_l2_navigation_routes_every_plan_cleanly(sim_world, demo_scene):
    """REBUILD.md Phase 1: obstacle avoidance is the navigation layer's job, not
    the plan's. EVERY plan's compiled trajectory must be collision-free — route
    feasibility is no longer something physics needs to discriminate."""
    from physgate.gate.l2_physics import rollout_plans
    from physgate.planner.planner import MockPlanner

    plans = MockPlanner()(TASK, demo_scene, 8, None)
    results = rollout_plans(sim_world, plans)

    assert all(r.collision_count == 0 for r in results), (
        "navigation must route every plan around the obstacle: "
        + str([(r.plan_id, r.collision_count) for r in results])
    )


def test_l2_at_least_one_plan_physically_succeeds(sim_world, demo_scene):
    """At least one candidate must complete the task in real physics
    (box resting on the shelf after placement)."""
    from physgate.gate.l2_physics import rollout_plans
    from physgate.planner.planner import MockPlanner

    plans = MockPlanner()(TASK, demo_scene, 8, None)
    results = rollout_plans(sim_world, plans)
    successes = [r for r in results if r.success]
    assert successes, "no plan physically placed the box on the shelf; outcomes: " + str(
        [(r.plan_id, r.failure.violations[0].detail if r.failure else "") for r in results]
    )


def test_full_gate_with_isaac_l2_selects_clean_plan(sim_world, demo_scene):
    """End-to-end gate (L1 -> L3 -> Isaac L2 -> scoring): the selected plan must
    be physically successful and collision-free."""
    from physgate.gate.l2_physics import IsaacL2Gate
    from physgate.gate.parallel import run_gate
    from physgate.planner.critic import MockCritic
    from physgate.planner.planner import MockPlanner

    plans = MockPlanner()(TASK, demo_scene, 8, None)
    survivors = MockCritic()(plans, demo_scene)
    selection = run_gate(survivors, demo_scene, l2_fn=IsaacL2Gate(sim_world))

    assert selection.any_feasible, f"gate rejected everything: {selection.rationale}"
    best = next(r for r in selection.ranked if r.plan_id == selection.best_plan_id)
    assert best.success
    assert best.collision_count == 0, "gate must prefer collision-free plans"


# ---------------------------------------------------------- C13 USD semantics


def test_scene_from_stage_reconstructs_objects(sim_world):
    from physgate.world.usd_semantics import scene_from_stage

    scene = scene_from_stage(env_index=0)
    ids = {o.id for o in scene.objects}
    assert {"box_03", "shelf_A", "obstacle_P", "go2", "floor_01"}.issubset(ids)

    box = next(o for o in scene.objects if o.id == "box_03")
    assert box.label == "cardboard_box"
    assert "graspable" in box.affordances


def test_scene_from_stage_derives_on_floor_relation(sim_world):
    from physgate.gate.reset_workaround import reset_scene_to_identical_state
    from physgate.world.usd_semantics import scene_from_stage

    reset_scene_to_identical_state(sim_world.scene, sim_world.sim)
    scene = scene_from_stage(env_index=0)
    assert scene.has_relation("box_03", "on", "floor_01")
    # the fallen box is the anomaly the task is about
    box = next(o for o in scene.objects if o.id == "box_03")
    assert box.is_anomaly


# --------------------------------------------------- policy locomotion (goal #1)


def _policy_path():
    from physgate.world.locomotion import find_exported_policy

    return find_exported_policy()


def test_navigator_never_commands_pure_rotation():
    """D-020: outside the heading deadband the navigator must command a creep
    forward speed, never vx=0 — the trained policy cannot turn in place from a
    standstill (a "keep standing" fixed point that deadlocks sharp corners)."""
    import torch

    from physgate.world.locomotion import TURN_CREEP_SPEED, WaypointNavigator

    nav = WaypointNavigator(num_envs=1, device="cpu")
    pos = torch.zeros(1, 3)
    yaw = torch.zeros(1)  # facing +x
    goal = torch.tensor([[0.0, 2.0, 0.0]])  # 90 deg to the left -> outside deadband
    speed = torch.tensor([0.5])

    commands, arrived = nav.velocity_commands(pos, yaw, goal, speed)
    assert not bool(arrived[0])
    # creep forward while turning, never a pure-rotation command
    assert abs(float(commands[0, 0]) - TURN_CREEP_SPEED) < 1e-6
    assert float(commands[0, 2]) > 0.5  # strong left turn command


def test_policy_robot_walks_to_goal(sim_world):
    """The trained policy makes the Go2 physically walk to a nearby goal."""
    import torch

    from physgate.gate.reset_workaround import reset_scene_to_identical_state
    from physgate.world.locomotion import Go2PolicyController, WaypointNavigator, base_yaws

    policy = _policy_path()
    if policy is None:
        pytest.skip("no exported Go2 policy (run rsl_rl play.py first)")

    reset_scene_to_identical_state(sim_world.scene, sim_world.sim)
    controller = Go2PolicyController(policy, num_envs=NUM_ENVS, device=sim_world.device)
    navigator = WaypointNavigator(num_envs=NUM_ENVS, device=sim_world.device)

    goal = torch.tensor([1.5, 0.0, 0.0], device=sim_world.device).repeat(NUM_ENVS, 1)
    speeds = torch.full((NUM_ENVS,), 0.5, device=sim_world.device)

    # walk for up to 15 simulated seconds (50 Hz control)
    arrived_any = False
    for _ in range(750):
        pos = sim_world.robot.data.root_pos_w - sim_world.env_origins
        yaw = base_yaws(sim_world.robot)
        commands, arrived = navigator.velocity_commands(pos, yaw, goal, speeds)
        if bool(arrived.all()):
            arrived_any = True
            break
        targets = controller.joint_position_targets(sim_world.robot, commands)
        sim_world.apply_joint_targets(targets)
        for _ in range(4):
            sim_world.step()

    final_pos = (sim_world.robot.data.root_pos_w - sim_world.env_origins).cpu().numpy()
    # robots walked forward (started at x=0) and stayed upright
    assert arrived_any or final_pos[:, 0].mean() > 1.0, (
        f"robots did not walk: mean x = {final_pos[:, 0].mean():.2f}"
    )
    upright = sim_world.robot.data.projected_gravity_b[:, 2] < -0.6
    assert bool(upright.all()), "some robots fell over while walking"


def test_policy_rollout_feasibility_near_100_percent(sim_world, demo_scene):
    """REBUILD.md Phase 1 acceptance: with navigation in the right layer, every
    well-formed plan (correct pick -> carry -> place decomposition) must be
    physically feasible — feasibility is ~100%, not a route lottery.

    Only the critic-surviving plans count: the reckless variant (no navigation,
    no preconditions) is SUPPOSED to fail — that is an orchestration error, the
    thing the gate still exists to catch."""
    from physgate.gate.l2_physics import rollout_plans_with_policy
    from physgate.planner.critic import MockCritic
    from physgate.planner.planner import MockPlanner

    policy = _policy_path()
    if policy is None:
        pytest.skip("no exported Go2 policy (run rsl_rl play.py first)")

    plans = MockPlanner()(TASK, demo_scene, 8, None)
    survivors = MockCritic()(plans, demo_scene)
    assert len(survivors) >= 4

    results = rollout_plans_with_policy(sim_world, survivors, policy)
    feasible = [r for r in results if r.success]

    feasibility = len(feasible) / len(results)
    assert feasibility >= 0.8, (
        f"feasibility should be ~100% after the rebuild, got {feasibility:.0%}: "
        + str(
            [
                (r.plan_id, r.failure.violations[0].detail if r.failure else "ok")
                for r in results
                if not r.success
            ]
        )
    )


def _fetch_plan(plan_id: str, speed: float):
    """A well-formed fetch-and-place plan with a uniform commanded speed."""
    from physgate.planner.schemas import Plan, PlanStep, ToolName

    return Plan(
        plan_id=plan_id,
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "box_03", "standoff_m": 0.3, "speed": speed},
                preconditions=["box_03 exists"],
            ),
            PlanStep(
                step_id=2,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=["box_03 exists", "gripper_empty"],
            ),
            PlanStep(
                step_id=3,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "shelf_A", "standoff_m": 0.4, "speed": speed},
                preconditions=["shelf_A exists"],
            ),
            PlanStep(
                step_id=4,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "place", "target": "shelf_A"},
                preconditions=[],
            ),
        ],
    )


def test_policy_place_calls_release_with_momentum(sim_world):
    """REBUILD.md Phase 2: the place handler must call release_boxes with the
    approach momentum. Pre-rebuild bug: it never called release_boxes at all
    (write_box_poses zeroes velocity), so the box always dropped dead and
    'fast, sloppy placements fail' did not hold in policy mode."""
    from physgate.gate.l2_physics import rollout_plans_with_policy

    policy = _policy_path()
    if policy is None:
        pytest.skip("no exported Go2 policy (run rsl_rl play.py first)")

    release_velocities = []
    original_release = sim_world.release_boxes

    def spy_release(velocities, env_ids):
        release_velocities.append(velocities.clone())
        return original_release(velocities, env_ids)

    sim_world.release_boxes = spy_release
    try:
        results = rollout_plans_with_policy(sim_world, [_fetch_plan("momentum_test", 0.5)], policy)
    finally:
        sim_world.release_boxes = original_release

    assert results[0].success, (
        f"baseline plan failed: "
        f"{results[0].failure.violations[0].detail if results[0].failure else '?'}"
    )
    # the placement release must exist and carry the commanded approach momentum
    assert release_velocities, "place never called release_boxes — the Phase 2 bug is back"
    horizontal_speed = float(release_velocities[-1][0, :2].norm())
    assert horizontal_speed > 0.5, (
        f"box released with only {horizontal_speed:.2f} m/s — momentum transfer missing"
    )


def test_policy_rollout_handles_out_of_range_speeds(sim_world):
    """Plans may command any speed; mission compilation clamps to the envelope
    the locomotion stack reliably executes (D-018) — out-of-range commands must
    not produce physically failing missions."""
    from physgate.gate.l2_physics import rollout_plans_with_policy

    policy = _policy_path()
    if policy is None:
        pytest.skip("no exported Go2 policy (run rsl_rl play.py first)")

    plans = [
        _fetch_plan("speed_min", 0.1),  # below the floor -> clamped up to 0.4
        _fetch_plan("speed_mid", 0.5),
        _fetch_plan("speed_max", 2.0),  # above the ceiling -> clamped down to 0.6
    ]
    results = rollout_plans_with_policy(sim_world, plans, policy)
    failures = [
        (r.plan_id, r.failure.violations[0].detail if r.failure else "")
        for r in results
        if not r.success
    ]
    assert not failures, f"clamped-speed plans failed: {failures}"


def test_policy_rollout_marks_unreachable_goal_infeasible(sim_world, monkeypatch):
    """REBUILD.md Phase 3 'goal navigation cannot reach': the gate must return a
    clean infeasible verdict (so the orchestrator can escalate), not crash with
    an uncaught PathPlannerError."""
    import physgate.gate.l2_physics as l2
    from physgate.nav.path_planner import PathPlannerError

    policy = _policy_path()
    if policy is None:
        pytest.skip("no exported Go2 policy (run rsl_rl play.py first)")

    def unreachable_compile(plan, layout):
        raise PathPlannerError("no path from (0.0, 0.0) to goal (goal unreachable)")

    monkeypatch.setattr(l2, "compile_mission", unreachable_compile)
    results = l2.rollout_plans_with_policy(sim_world, [_fetch_plan("unreachable", 0.5)], policy)

    assert len(results) == 1
    assert results[0].success is False
    violation = results[0].failure.violations[0]
    assert violation.type == "infeasible_navigation"
    assert "unreachable" in violation.detail


def test_kinematic_rollout_marks_unreachable_goal_infeasible(sim_world, demo_scene, monkeypatch):
    """Same contract for the kinematic rollout: navigation infeasibility is a
    verdict, not a crash."""
    import physgate.gate.l2_physics as l2
    from physgate.nav.path_planner import PathPlannerError
    from physgate.planner.planner import MockPlanner

    def unreachable_synth(plan, dt):
        raise PathPlannerError("no path (goal unreachable)")

    monkeypatch.setattr(l2, "synthesize_base_trajectory", unreachable_synth)
    plans = MockPlanner()(TASK, demo_scene, 2, None)
    results = l2.rollout_plans(sim_world, plans)

    assert len(results) == 2
    assert all(r.success is False for r in results)
    assert all(r.failure.violations[0].type == "infeasible_navigation" for r in results)


def test_sim_backend_reports_unreachable_target(sim_world, demo_scene, monkeypatch):
    """The executor must surface navigation infeasibility as a failed action
    (so the orchestrator replans/escalates), not crash mid-execution."""
    import physgate.executor.sim_backend as sb
    from physgate.nav.path_planner import PathPlannerError

    def unreachable_route(*args, **kwargs):
        raise PathPlannerError("no path (goal unreachable)")

    monkeypatch.setattr(sb, "plan_standoff_route", unreachable_route)
    backend = sb.SimBackend(demo_scene, world=sim_world)
    result = backend.move_to_pose("box_03", standoff_m=0.3)

    assert result["success"] is False
    assert "unreachable" in result["error"]


def test_policy_rollout_invokes_control_step_callback(sim_world):
    """The rollout exposes an on_control_step hook (used by the recording
    pipeline to capture camera frames and trajectory samples without
    duplicating the control loop)."""
    from physgate.gate.l2_physics import rollout_plans_with_policy

    policy = _policy_path()
    if policy is None:
        pytest.skip("no exported Go2 policy (run rsl_rl play.py first)")

    steps_seen = []
    states_seen = []

    def on_step(control_step, world, state):
        steps_seen.append(control_step)
        states_seen.append(state)

    results = rollout_plans_with_policy(
        sim_world, [_fetch_plan("callback_test", 0.5)], policy, on_control_step=on_step
    )

    assert results[0].success
    assert steps_seen, "callback never invoked"
    assert steps_seen == sorted(steps_seen), "callback must be invoked in step order"
    # the state dict exposes what the recorder needs
    assert all({"carrying", "mission_index", "completed"} <= set(s) for s in states_seen)
    # the carry flag must have been True at some point (the box was picked up)
    assert any(s["carrying"][0] for s in states_seen), "carry state never reported"


# -------------------------------------------------------------- E17 SimBackend


def test_sim_backend_executes_winning_plan(sim_world, demo_scene):
    """The executor physically runs a plan in Isaac Sim and the box really ends
    up on the shelf."""
    from physgate.executor.plan_executor import execute_plan
    from physgate.executor.sim_backend import SimBackend
    from physgate.planner.planner import MockPlanner

    plans = MockPlanner()(TASK, demo_scene, 8, None)
    cautious = next(p for p in plans if "cautious" in p.plan_id)

    backend = SimBackend(demo_scene, world=sim_world)
    result = execute_plan(cautious, backend)

    assert result["success"], f"execution failed: {result.get('error')}"
    # physics ground truth: the box is resting on the shelf in the simulation
    final_box = sim_world.box_positions()[0].cpu().numpy()
    from physgate.gate.l2_physics import _box_on_shelf

    assert _box_on_shelf(final_box), f"box ended at {final_box}, not on the shelf"
    # the symbolic scene agrees with the physics
    assert backend.get_scene().has_relation("box_03", "on", "shelf_A")


def test_rollout_rejects_hallucinated_target_plan(sim_world):
    """A plan that moves to an invented object id gets an explicit unknown_object
    failure — not a misleading 'box not on shelf' 0.0s result (D-016)."""
    from physgate.gate.l2_physics import rollout_plans_with_policy
    from physgate.planner.schemas import Plan, PlanStep, ToolName

    policy = _policy_path()
    if policy is None:
        pytest.skip("no exported Go2 policy (run rsl_rl play.py first)")

    plan = Plan(
        plan_id="hallucinated",
        task=TASK,
        rationale="moves via an invented waypoint",
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "waypoint_imaginary"},
                preconditions=[],
            ),
            PlanStep(
                step_id=2,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=["box_03 exists"],
            ),
        ],
    )
    results = rollout_plans_with_policy(sim_world, [plan], policy)

    assert results[0].success is False
    assert results[0].failure is not None
    violations = results[0].failure.violations
    assert any(v.type == "unknown_object" for v in violations)
    assert "waypoint_imaginary" in violations[0].detail
