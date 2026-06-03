"""Tests for the Claude planner + mock fallback (D14).

The planner turns a natural-language task + scene into N candidate Plans.
ClaudePlanner is tested against a fake Anthropic client (no network, no key);
MockPlanner is the deterministic fallback used when ANTHROPIC_API_KEY is unset.
"""

import json

import pytest

from physgate.gate.schemas import Scene, SceneObject
from physgate.planner.planner import ClaudePlanner, MockPlanner, make_planner
from physgate.planner.schemas import Plan, ToolName


def _scene() -> Scene:
    return Scene(
        objects=[
            SceneObject(id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
        ],
        relations=[("box_03", "on", "floor_01")],
        gripper_empty=True,
    )


TASK = "put the fallen box back on shelf A"


# ---------------------------------------------------------------- MockPlanner


def test_mock_planner_returns_n_valid_plans():
    plans = MockPlanner()(TASK, _scene(), 8, None)
    assert len(plans) == 8
    assert all(isinstance(p, Plan) for p in plans)
    assert len({p.plan_id for p in plans}) == 8  # unique ids


def test_mock_planner_plans_reference_scene_objects():
    plans = MockPlanner()(TASK, _scene(), 4, None)
    for plan in plans:
        for step in plan.steps:
            target = step.args.get("target")
            if target:
                assert target in {"box_03", "shelf_A", "floor_01"}


def test_mock_planner_is_deterministic():
    a = MockPlanner()(TASK, _scene(), 8, None)
    b = MockPlanner()(TASK, _scene(), 8, None)
    assert [p.model_dump() for p in a] == [p.model_dump() for p in b]


def test_mock_planner_includes_diverse_candidates():
    """Mock candidates must differ (different step counts / orderings) so the
    gate has something meaningful to select between."""
    plans = MockPlanner()(TASK, _scene(), 8, None)
    step_signatures = {tuple((s.tool, s.args.get("skill", "")) for s in p.steps) for p in plans}
    assert len(step_signatures) > 1


def test_mock_planner_feedback_changes_rationale():
    plans = MockPlanner()(TASK, _scene(), 2, "previous plans collided with the shelf")
    assert any("replan" in p.rationale.lower() or "feedback" in p.rationale.lower() for p in plans)


# -------------------------------------------------------------- ClaudePlanner


class _FakeContent:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeContent(text)]


class _FakeMessages:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(self._response_text)


class _FakeAnthropicClient:
    def __init__(self, response_text: str):
        self.messages = _FakeMessages(response_text)


def _llm_plan_json(n: int = 2) -> str:
    plans = []
    for i in range(n):
        plans.append(
            {
                "plan_id": f"candidate_{i}",
                "task": TASK,
                "rationale": f"route variant {i}",
                "steps": [
                    {
                        "step_id": 1,
                        "tool": "move_to_pose",
                        "args": {"target": "box_03"},
                        "preconditions": ["box_03 exists"],
                        "effects": [],
                    },
                    {
                        "step_id": 2,
                        "tool": "execute_skill",
                        "args": {"skill": "pick", "target": "box_03"},
                        "preconditions": ["gripper_empty"],
                        "effects": [
                            {"op": "add", "subject": "gripper", "predicate": "holding", "object": "box_03"}
                        ],
                    },
                ],
            }
        )
    return json.dumps(plans)


def test_claude_planner_parses_llm_response():
    client = _FakeAnthropicClient(_llm_plan_json(3))
    planner = ClaudePlanner(client=client)
    plans = planner(TASK, _scene(), 3, None)
    assert len(plans) == 3
    assert plans[0].steps[1].tool == ToolName.EXECUTE_SKILL


def test_claude_planner_prompt_contains_task_scene_and_n():
    client = _FakeAnthropicClient(_llm_plan_json(1))
    ClaudePlanner(client=client)(TASK, _scene(), 8, None)
    call = client.messages.calls[0]
    prompt_text = json.dumps(call)
    assert TASK in prompt_text
    assert "box_03" in prompt_text
    assert "8" in prompt_text


def test_claude_planner_passes_failure_feedback():
    client = _FakeAnthropicClient(_llm_plan_json(1))
    ClaudePlanner(client=client)(TASK, _scene(), 1, "all plans collided with shelf_A")
    prompt_text = json.dumps(client.messages.calls[0])
    assert "collided with shelf_A" in prompt_text


def test_claude_planner_handles_markdown_fenced_json():
    fenced = "```json\n" + _llm_plan_json(2) + "\n```"
    client = _FakeAnthropicClient(fenced)
    plans = ClaudePlanner(client=client)(TASK, _scene(), 2, None)
    assert len(plans) == 2


def test_claude_planner_drops_invalid_plans_keeps_valid():
    valid = json.loads(_llm_plan_json(1))
    invalid = {"plan_id": "broken", "task": TASK, "steps": []}  # empty steps invalid
    client = _FakeAnthropicClient(json.dumps(valid + [invalid]))
    plans = ClaudePlanner(client=client)(TASK, _scene(), 2, None)
    assert len(plans) == 1


def test_claude_planner_raises_on_unparseable_response():
    client = _FakeAnthropicClient("I cannot help with that.")
    with pytest.raises(ValueError, match="parse"):
        ClaudePlanner(client=client)(TASK, _scene(), 2, None)


# --------------------------------------------------------------- make_planner


def test_make_planner_returns_mock_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    planner = make_planner()
    assert isinstance(planner, MockPlanner)


def test_make_planner_returns_claude_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    planner = make_planner()
    assert isinstance(planner, ClaudePlanner)
