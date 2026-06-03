"""Tests for planner plan/step data contracts (A1).

Written test-first. The plan JSON format is the artifact that crosses the
planner → gate → executor boundary (dual-envelope: data only, never code), so
round-trip fidelity and strict validation are the core requirements.
"""

import pytest
from pydantic import ValidationError

from physgate.planner.schemas import (
    DEFAULT_NUM_CANDIDATES,
    Plan,
    PlanStep,
    RelationChange,
    ToolName,
)


def _fetch_and_place_plan() -> Plan:
    """A realistic fetch-and-place candidate plan (the MVP task)."""
    return Plan(
        plan_id="plan_001",
        task="put the fallen box back on shelf A",
        rationale="walk to box, pick it, carry to shelf, place it",
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "box_03", "standoff_m": 0.3},
                preconditions=["box_03 exists"],
                effects=[
                    RelationChange(op="add", subject="robot", predicate="near", object="box_03")
                ],
            ),
            PlanStep(
                step_id=2,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=["box_03 exists", "gripper_empty", "robot near box_03"],
                effects=[
                    RelationChange(op="remove", subject="box_03", predicate="on", object="floor_01"),
                    RelationChange(op="add", subject="gripper", predicate="holding", object="box_03"),
                ],
            ),
            PlanStep(
                step_id=3,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "shelf_A", "standoff_m": 0.4},
                preconditions=["shelf_A exists"],
                effects=[
                    RelationChange(op="add", subject="robot", predicate="near", object="shelf_A")
                ],
            ),
            PlanStep(
                step_id=4,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "place", "target": "shelf_A"},
                preconditions=["gripper holding box_03", "robot near shelf_A"],
                effects=[
                    RelationChange(op="remove", subject="gripper", predicate="holding", object="box_03"),
                    RelationChange(op="add", subject="box_03", predicate="on", object="shelf_A"),
                ],
            ),
        ],
    )


def test_plan_round_trips_through_json():
    plan = _fetch_and_place_plan()
    restored = Plan.model_validate_json(plan.model_dump_json())
    assert restored == plan
    assert restored.steps[1].tool == ToolName.EXECUTE_SKILL
    assert restored.steps[1].effects[1].predicate == "holding"


def test_plan_parses_from_raw_llm_json():
    """The exact dict shape an LLM is prompted to emit must validate."""
    raw = {
        "plan_id": "candidate_3",
        "task": "put the fallen box back on shelf A",
        "rationale": "direct route",
        "steps": [
            {
                "step_id": 1,
                "tool": "execute_skill",
                "args": {"skill": "pick", "target": "box_03"},
                "preconditions": ["box_03 exists", "gripper_empty"],
                "effects": [
                    {"op": "add", "subject": "gripper", "predicate": "holding", "object": "box_03"}
                ],
            }
        ],
    }
    plan = Plan.model_validate(raw)
    assert plan.steps[0].tool == ToolName.EXECUTE_SKILL


def test_invalid_tool_name_rejected():
    with pytest.raises(ValidationError):
        PlanStep(step_id=1, tool="launch_missiles", args={})


def test_plan_requires_at_least_one_step():
    with pytest.raises(ValidationError):
        Plan(plan_id="empty", task="do nothing", steps=[])


def test_duplicate_step_ids_rejected():
    step = {"step_id": 1, "tool": "query_scene", "args": {}}
    with pytest.raises(ValidationError):
        Plan(plan_id="dup", task="t", steps=[step, step])


def test_invalid_relation_change_op_rejected():
    with pytest.raises(ValidationError):
        RelationChange(op="mutate", subject="a", predicate="on", object="b")


def test_step_defaults_are_empty_not_none():
    step = PlanStep(step_id=1, tool=ToolName.QUERY_SCENE, args={})
    assert step.preconditions == []
    assert step.effects == []


def test_default_num_candidates_is_eight():
    assert DEFAULT_NUM_CANDIDATES == 8
