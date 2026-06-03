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
