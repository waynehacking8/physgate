"""Tests for the backend-agnostic plan executor (E17, logic part).

The executor walks a validated Plan step by step, dispatching each step to the
WorldBackend through the MCP tool functions, checking L3 preconditions before
every step (defense in depth — the gate already validated, but state may have
drifted), and reporting structured results.
"""

from physgate.executor.backend import MockWorldBackend
from physgate.executor.plan_executor import execute_plan
from physgate.gate.schemas import Scene, SceneObject
from physgate.planner.planner import MockPlanner
from physgate.planner.schemas import Plan, PlanStep, RelationChange, ToolName

TASK = "put the fallen box back on shelf A"


def _scene() -> Scene:
    return Scene(
        objects=[
            SceneObject(id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[("box_03", "on", "floor_01")],
        gripper_empty=True,
    )


def _good_plan() -> Plan:
    return Plan(
        plan_id="good",
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "box_03", "standoff_m": 0.3},
                preconditions=["box_03 exists"],
            ),
            PlanStep(
                step_id=2,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=["box_03 exists", "gripper_empty"],
                effects=[
                    RelationChange(op="add", subject="gripper", predicate="holding", object="box_03")
                ],
            ),
            PlanStep(
                step_id=3,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "shelf_A", "standoff_m": 0.4},
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


def test_good_plan_executes_fully():
    backend = MockWorldBackend(_scene())
    result = execute_plan(_good_plan(), backend)
    assert result["success"] is True
    assert result["steps_completed"] == 4
    assert result["steps_total"] == 4
    # the world reflects the completed task
    final_scene = backend.get_scene()
    assert final_scene.has_relation("box_03", "on", "shelf_A")


def test_execution_stops_at_first_failure():
    plan = _good_plan()
    # sabotage step 2: pick a non-graspable object
    plan = plan.model_copy(deep=True)
    plan.steps[1].args["target"] = "shelf_A"
    plan.steps[1].preconditions = ["shelf_A exists", "gripper_empty"]

    backend = MockWorldBackend(_scene())
    result = execute_plan(plan, backend)
    assert result["success"] is False
    assert result["steps_completed"] == 1
    assert result["failed_step_id"] == 2
    assert "error" in result


def test_precondition_recheck_before_each_step():
    """L3 preconditions are re-checked at execution time (not just at gate time)."""
    plan = Plan(
        plan_id="stale",
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_99"},  # object does not exist
                preconditions=["box_99 exists"],
            )
        ],
    )
    backend = MockWorldBackend(_scene())
    result = execute_plan(plan, backend)
    assert result["success"] is False
    assert result["failure_report"] is not None
    assert result["failure_report"]["failure_code"] == "precondition_violation"


def test_query_scene_steps_succeed_and_capture_payload():
    plan = Plan(
        plan_id="scan",
        task=TASK,
        steps=[PlanStep(step_id=1, tool=ToolName.QUERY_SCENE, args={})],
    )
    backend = MockWorldBackend(_scene())
    result = execute_plan(plan, backend)
    assert result["success"] is True
    assert result["step_results"][0]["payload"]["gripper_empty"] is True


def test_mock_planner_direct_plan_executes_on_mock_backend():
    """Integration: MockPlanner's direct variant runs cleanly end to end."""
    plans = MockPlanner()(TASK, _scene(), 8, None)
    direct = next(p for p in plans if "direct" in p.plan_id)
    backend = MockWorldBackend(_scene())
    result = execute_plan(direct, backend)
    assert result["success"] is True


def test_step_results_record_per_step_outcome():
    backend = MockWorldBackend(_scene())
    result = execute_plan(_good_plan(), backend)
    assert len(result["step_results"]) == 4
    assert all(r["success"] for r in result["step_results"])
    assert [r["step_id"] for r in result["step_results"]] == [1, 2, 3, 4]
