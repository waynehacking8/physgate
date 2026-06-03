"""Tests for the safety critic + mock fallback (D15).

The critic adversarially reviews candidate plans against safety contracts and
returns only the survivors (SAFER pattern). MockCritic enforces deterministic
structural contracts; ClaudeCritic delegates judgement to the LLM.
"""

import json

from physgate.gate.schemas import Scene, SceneObject
from physgate.planner.critic import SAFETY_CONTRACTS, ClaudeCritic, MockCritic, make_critic
from physgate.planner.planner import ClaudePlanner, MockPlanner
from physgate.planner.schemas import Plan, PlanStep, RelationChange, ToolName

TASK = "put the fallen box back on shelf A"


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


def _good_plan(plan_id: str = "good") -> Plan:
    return Plan(
        plan_id=plan_id,
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=["box_03 exists", "gripper_empty"],
                effects=[
                    RelationChange(op="add", subject="gripper", predicate="holding", object="box_03")
                ],
            )
        ],
    )


# ----------------------------------------------------------------- MockCritic


def test_mock_critic_passes_good_plans():
    survivors = MockCritic()([_good_plan("a"), _good_plan("b")], _scene())
    assert [p.plan_id for p in survivors] == ["a", "b"]


def test_mock_critic_rejects_unknown_object_reference():
    bad = Plan(
        plan_id="ghost_grabber",
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_99"},  # not in scene
                preconditions=["box_99 exists"],
            )
        ],
    )
    survivors = MockCritic()([_good_plan(), bad], _scene())
    assert [p.plan_id for p in survivors] == ["good"]


def test_mock_critic_rejects_unguarded_manipulation():
    """Manipulation steps with zero preconditions violate the 'no unchecked
    physical action' safety contract."""
    reckless = Plan(
        plan_id="reckless",
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=[],  # nothing checked before acting
            )
        ],
    )
    survivors = MockCritic()([reckless, _good_plan()], _scene())
    assert [p.plan_id for p in survivors] == ["good"]


def test_mock_critic_rejects_non_graspable_pick_target():
    grab_shelf = Plan(
        plan_id="grab_shelf",
        task=TASK,
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "shelf_A"},  # shelf is not graspable
                preconditions=["shelf_A exists", "gripper_empty"],
            )
        ],
    )
    survivors = MockCritic()([grab_shelf], _scene())
    assert survivors == []


def test_mock_critic_pipeline_with_mock_planner():
    """MockPlanner output must largely survive MockCritic (pipeline sanity)."""
    plans = MockPlanner()(TASK, _scene(), 8, None)
    survivors = MockCritic()(plans, _scene())
    assert len(survivors) >= 1


def test_safety_contracts_documented():
    assert len(SAFETY_CONTRACTS) >= 3
    assert all(isinstance(c, str) for c in SAFETY_CONTRACTS)


# --------------------------------------------------------------- ClaudeCritic


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


def test_claude_critic_keeps_approved_ids():
    verdict = json.dumps({"approved_plan_ids": ["a"], "rejections": {"b": "collides with human zone"}})
    client = _FakeAnthropicClient(verdict)
    survivors = ClaudeCritic(client=client)([_good_plan("a"), _good_plan("b")], _scene())
    assert [p.plan_id for p in survivors] == ["a"]


def test_claude_critic_prompt_contains_contracts_and_plans():
    verdict = json.dumps({"approved_plan_ids": [], "rejections": {}})
    client = _FakeAnthropicClient(verdict)
    ClaudeCritic(client=client)([_good_plan("a")], _scene())
    prompt_text = json.dumps(client.messages.calls[0])
    assert SAFETY_CONTRACTS[0][:30] in prompt_text
    assert '"a"' in prompt_text or "plan_id" in prompt_text


def test_claude_critic_fails_closed_on_unparseable_response():
    """If the critic LLM response cannot be parsed, NO plan survives (fail closed)."""
    client = _FakeAnthropicClient("everything looks fine to me!")
    survivors = ClaudeCritic(client=client)([_good_plan("a")], _scene())
    assert survivors == []


# ----------------------------------------------------------------- make_critic


def test_make_critic_mock_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(make_critic(), MockCritic)


def test_make_critic_claude_with_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    assert isinstance(make_critic(), ClaudeCritic)
