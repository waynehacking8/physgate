"""Tests for the full Sim-Gate pipeline: L1 -> L3 -> L2 -> scoring (C12).

The gate chains its layers cheapest-first:
  L1 (kinematic, numpy)  ->  L3 (symbolic preconditions/rollout)  ->  L2 (physics)
Plans rejected by a cheap layer never reach the expensive one. L2 is pluggable:
the symbolic L2 (mock backend rollout) is used here; the Isaac Lab L2
(gate/l2_physics.py) plugs in the same way.
"""

import numpy as np

from physgate.gate.parallel import run_gate, symbolic_l2
from physgate.gate.scoring import PhysicsResult, SelectionResult
from physgate.gate.schemas import Scene, SceneObject
from physgate.planner.planner import MockPlanner
from physgate.planner.schemas import Plan, PlanStep, RelationChange, ToolName

TASK = "put the fallen box back on shelf A"


def _scene() -> Scene:
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[("box_03", "on", "floor_01")],
        gripper_empty=True,
    )


def _feasible_plan(plan_id: str = "feasible") -> Plan:
    return Plan(
        plan_id=plan_id,
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "box_03", "standoff_m": 0.3, "speed": 0.5},
                preconditions=["box_03 exists"],
            ),
            PlanStep(
                step_id=2,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=["box_03 exists", "gripper_empty"],
                effects=[
                    RelationChange(
                        op="add", subject="gripper", predicate="holding", object="box_03"
                    )
                ],
            ),
            PlanStep(
                step_id=3,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "shelf_A", "standoff_m": 0.4, "speed": 0.5},
                preconditions=["shelf_A exists"],
            ),
            PlanStep(
                step_id=4,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "place", "target": "shelf_A"},
                preconditions=["gripper holding box_03"],
            ),
        ],
    )


def _l3_violating_plan() -> Plan:
    """References an object that does not exist — L3 must reject before L2."""
    return Plan(
        plan_id="ghost",
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_99"},
                preconditions=["box_99 exists", "gripper_empty"],
            )
        ],
    )


def _l1_violating_plan() -> Plan:
    """Carries an explicit joint trajectory outside Go2 limits — L1 must reject."""
    bad_trajectory = np.zeros((10, 12))
    bad_trajectory[:, 0] = 3.0  # hip limit is +-1.0472
    return Plan(
        plan_id="dislocator",
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.EXECUTE_SKILL,
                args={
                    "skill": "pick",
                    "target": "box_03",
                    "joint_trajectory": bad_trajectory.tolist(),
                },
                preconditions=["box_03 exists", "gripper_empty"],
            )
        ],
    )


# ------------------------------------------------------------------- run_gate


def test_feasible_plan_passes_full_gate():
    selection = run_gate([_feasible_plan()], _scene())
    assert isinstance(selection, SelectionResult)
    assert selection.any_feasible is True
    assert selection.best_plan_id == "feasible"


def test_l3_rejection_blocks_plan():
    selection = run_gate([_l3_violating_plan()], _scene())
    assert selection.any_feasible is False


def test_l1_rejection_blocks_plan():
    selection = run_gate([_l1_violating_plan()], _scene())
    assert selection.any_feasible is False


def test_rejected_plans_never_reach_l2():
    l2_calls = []

    def spy_l2(plans, scene):
        l2_calls.append([p.plan_id for p in plans])
        return symbolic_l2(plans, scene)

    run_gate([_feasible_plan(), _l3_violating_plan(), _l1_violating_plan()], _scene(), l2_fn=spy_l2)
    assert l2_calls == [["feasible"]]


def test_best_of_n_selects_most_feasible():
    """Among multiple feasible plans, the one with the best physics score wins."""
    fast = _feasible_plan("fast")
    for step in fast.steps:
        step.args["speed"] = 1.0

    slow = _feasible_plan("slow")
    for step in slow.steps:
        step.args["speed"] = 0.1

    selection = run_gate([slow, fast], _scene())
    assert selection.any_feasible is True
    # symbolic L2 estimates completion time from speed: faster plan scores higher
    assert selection.best_plan_id == "fast"
    assert selection.scores["fast"] > selection.scores["slow"]


def test_all_plans_rejected_gives_feedback_rationale():
    selection = run_gate([_l3_violating_plan()], _scene())
    assert selection.any_feasible is False
    assert selection.rationale != ""


def test_gate_with_mock_planner_candidates():
    """Integration: MockPlanner candidates flow through the full gate; the
    reckless (no-precondition) variants are feasible at L3 but the gate still
    picks a best plan."""
    plans = MockPlanner()(TASK, _scene(), 8, None)
    selection = run_gate(plans, _scene())
    assert selection.any_feasible is True
    assert selection.best_plan_id is not None
    assert len(selection.scores) == 8


def test_empty_plan_list_returns_infeasible():
    selection = run_gate([], _scene())
    assert selection.any_feasible is False
    assert selection.best_plan_id is None


# ---------------------------------------------------------------- symbolic_l2


def test_symbolic_l2_produces_physics_results():
    results = symbolic_l2([_feasible_plan()], _scene())
    assert len(results) == 1
    assert isinstance(results[0], PhysicsResult)
    assert results[0].success is True
    assert results[0].completion_time_s > 0
    assert results[0].energy_j > 0


def test_symbolic_l2_detects_symbolic_failure():
    """A plan that fails symbolic rollout (pick non-graspable) fails L2."""
    grab_shelf = Plan(
        plan_id="grab_shelf",
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "shelf_A"},
                preconditions=["shelf_A exists"],
            ),
            PlanStep(
                step_id=2,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "shelf_A"},
                preconditions=["shelf_A exists", "gripper_empty"],
            ),
        ],
    )
    results = symbolic_l2([grab_shelf], _scene())
    assert results[0].success is False
    assert results[0].failure is not None


def test_plan_with_hallucinated_target_rejected_at_l3():
    """The full gate pipeline must reject plans that move to invented objects
    BEFORE physics — they get a failed result with an unknown_object violation."""
    scene = _scene()
    plan = Plan(
        plan_id="hallucinated_waypoint",
        task=TASK,
        rationale="route via a waypoint that does not exist",
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
                preconditions=["box_03 exists", "gripper_empty"],
            ),
        ],
    )
    selection = run_gate([plan], scene, l2_fn=symbolic_l2)
    assert selection.any_feasible is False
    result = selection.ranked[0]
    assert not result.success
    assert result.failure is not None
    assert any(v.type == "unknown_object" for v in result.failure.violations)
    # it must never have reached L2 physics
    from physgate.gate.schemas import GateLayer

    assert result.failure.layer == GateLayer.SEMANTIC_PRECONDITION
