"""Tests for the Sim-Gate L3 scene-graph precondition checker.

Written test-first: this layer is pure Python and needs no GPU, so it is the
first thing built (Phase 1 in the build order). The fetch-and-place MVP task is
the source of these cases.
"""

from physgate.gate.l3_scene import check_preconditions
from physgate.gate.schemas import FailureCode, GateLayer, Scene, SceneObject


def _fetch_scene() -> Scene:
    """A box on the floor near a shelf; gripper empty; shelf unoccupied."""
    return Scene(
        objects=[
            SceneObject(id="box_03", label="cardboard_box", affordances=["graspable"]),
            SceneObject(id="shelf_A", label="shelf"),
            SceneObject(id="floor_01", label="floor"),
        ],
        relations=[("box_03", "on", "floor_01"), ("shelf_A", "unoccupied", "shelf_A")],
        gripper_empty=True,
    )


def test_all_preconditions_met_returns_none():
    scene = _fetch_scene()
    preconds = ["box_03 exists", "gripper_empty", "shelf_A unoccupied shelf_A"]
    assert check_preconditions(preconds, scene, step_id=4) is None


def test_missing_object_is_denied():
    scene = _fetch_scene()
    report = check_preconditions(["box_99 exists"], scene, step_id=2)
    assert report is not None
    assert report.verdict == "DENIED"
    assert report.failure_code == FailureCode.PRECONDITION_VIOLATION
    assert report.layer == GateLayer.SEMANTIC_PRECONDITION
    assert report.failed_step_id == 2
    assert report.violations[0].type == "missing_object"


def test_gripper_not_empty_is_denied():
    scene = _fetch_scene().model_copy(update={"gripper_empty": False})
    report = check_preconditions(["gripper_empty"], scene)
    assert report is not None
    assert report.violations[0].type == "gripper_not_empty"


def test_unmet_relation_is_denied():
    scene = _fetch_scene()
    # box is on the floor, not on the shelf yet
    report = check_preconditions(["box_03 on shelf_A"], scene)
    assert report is not None
    assert report.violations[0].type == "unmet_relation"


def test_multiple_violations_all_reported():
    scene = _fetch_scene().model_copy(update={"gripper_empty": False})
    report = check_preconditions(["box_99 exists", "gripper_empty"], scene)
    assert report is not None
    assert len(report.violations) == 2


def test_unparseable_precondition_fails_closed():
    scene = _fetch_scene()
    report = check_preconditions(["this is not a valid precondition form"], scene)
    assert report is not None
    assert report.violations[0].type == "unparseable_precondition"


# ------------------------------------------- step-target validation (anti-hallucination)


def test_step_targets_all_known_returns_none():
    """Plans whose step targets all exist in the scene pass the target check."""
    from physgate.gate.l3_scene import check_step_targets
    from physgate.planner.schemas import Plan, PlanStep, ToolName

    plan = Plan(
        plan_id="ok",
        task="t",
        rationale="r",
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "box_03"},
                preconditions=["box_03 exists"],
            ),
            PlanStep(
                step_id=2,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=["gripper_empty"],
            ),
        ],
    )
    assert check_step_targets(plan, _fetch_scene()) is None


def test_step_target_referencing_unknown_object_is_denied():
    """An LLM planner can hallucinate object ids (e.g. invented waypoints).

    Those plans must be rejected at L3 with a clear violation — NOT silently
    degraded into do-nothing missions at L2 (the failure mode behind the
    2026-06-03 real-LLM demo escalation)."""
    from physgate.gate.l3_scene import check_step_targets
    from physgate.planner.schemas import Plan, PlanStep, ToolName

    plan = Plan(
        plan_id="hallucinated",
        task="t",
        rationale="r",
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "waypoint_2"},  # does not exist
                preconditions=[],
            ),
        ],
    )
    report = check_step_targets(plan, _fetch_scene())
    assert report is not None
    assert report.failure_code == FailureCode.PRECONDITION_VIOLATION
    assert report.layer == GateLayer.SEMANTIC_PRECONDITION
    assert any("waypoint_2" in v.detail for v in report.violations)
    assert any(v.type == "unknown_object" for v in report.violations)


def test_step_target_check_reports_every_unknown_id():
    """All hallucinated ids are reported, not just the first."""
    from physgate.gate.l3_scene import check_step_targets
    from physgate.planner.schemas import Plan, PlanStep, ToolName

    plan = Plan(
        plan_id="double_hallucination",
        task="t",
        rationale="r",
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "staging_area"},
                preconditions=[],
            ),
            PlanStep(
                step_id=2,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "waypoint_9"},
                preconditions=[],
            ),
        ],
    )
    report = check_step_targets(plan, _fetch_scene())
    assert report is not None
    assert len(report.violations) == 2


def test_step_without_target_arg_is_fine():
    """query_scene steps have no target; that is not a violation."""
    from physgate.gate.l3_scene import check_step_targets
    from physgate.planner.schemas import Plan, PlanStep, ToolName

    plan = Plan(
        plan_id="scan",
        task="t",
        rationale="r",
        steps=[PlanStep(step_id=1, tool=ToolName.QUERY_SCENE, args={}, preconditions=[])],
    )
    assert check_step_targets(plan, _fetch_scene()) is None
