"""Failure analyst: structured diagnosis of plan execution failures.

The failure analyst receives a failed plan and the specific step failure,
then produces a structured FailureReport with root cause analysis and a
suggested minimal fix. This replaces blind regeneration with targeted repair.

Two implementations:
* :class:`ClaudeFailureAnalyst` — lightweight LLM call (~500 tokens)
* :class:`MockFailureAnalyst` — deterministic rule-based analysis for tests

Design reference: AGENT_UPGRADE.md §3.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from physgate.gate.schemas import Scene
from physgate.planner.schemas import Plan, PlanStep


class FailureType(str, Enum):
    """Classification of what went wrong during plan execution."""

    PRECONDITION_UNMET = "precondition_unmet"
    TOOL_ERROR = "tool_error"
    PHYSICS_REJECT = "physics_reject"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"


class FailureReport(BaseModel):
    """Structured failure diagnosis for the planner's targeted repair."""

    failed_step: PlanStep | None = None
    failed_step_index: int = -1
    failure_type: FailureType
    root_cause: str
    affected_steps: list[int] = Field(default_factory=list)
    suggested_fix: str
    prefix_valid_through: int = -1


class MockFailureAnalyst:
    """Deterministic rule-based failure analysis (no LLM)."""

    def __call__(
        self,
        plan: Plan,
        execution_result: dict[str, Any],
        scene: Scene,
    ) -> FailureReport:
        """Analyze a failed execution and produce a structured diagnosis."""
        failed_step_id = execution_result.get("failed_step_id")
        error = execution_result.get("error", "")
        failure_report_raw = execution_result.get("failure_report")

        failed_step = None
        failed_index = -1
        if failed_step_id is not None:
            for i, step in enumerate(plan.steps):
                if step.step_id == failed_step_id:
                    failed_step = step
                    failed_index = i
                    break

        failure_type = self._classify_failure(error, failure_report_raw)
        root_cause = self._diagnose_root_cause(error, failed_step, scene)
        affected = self._find_affected_steps(plan, failed_index)
        suggested_fix = self._suggest_fix(failure_type, root_cause, failed_step, plan, scene)
        prefix_valid = max(0, failed_index) if failed_index >= 0 else 0

        return FailureReport(
            failed_step=failed_step,
            failed_step_index=failed_index,
            failure_type=failure_type,
            root_cause=root_cause,
            affected_steps=affected,
            suggested_fix=suggested_fix,
            prefix_valid_through=prefix_valid,
        )

    @staticmethod
    def _classify_failure(error: str, failure_report: dict | None) -> FailureType:
        error_lower = error.lower()
        if failure_report is not None:
            return FailureType.PRECONDITION_UNMET
        if "locked" in error_lower:
            return FailureType.PRECONDITION_UNMET
        if "not near" in error_lower or "not holding" in error_lower:
            return FailureType.PRECONDITION_UNMET
        if "not pushable" in error_lower or "too heavy" in error_lower:
            return FailureType.INFEASIBLE
        if "timeout" in error_lower:
            return FailureType.TIMEOUT
        if "physics" in error_lower or "collision" in error_lower:
            return FailureType.PHYSICS_REJECT
        return FailureType.TOOL_ERROR

    @staticmethod
    def _diagnose_root_cause(error: str, step: PlanStep | None, scene: Scene) -> str:
        if step is None:
            return f"execution failed with: {error}"
        error_lower = error.lower()
        if "locked" in error_lower:
            return f"door is locked at step {step.step_id} — plan needs unlock before open"
        if "not near" in error_lower:
            return f"robot was not near target at step {step.step_id} — missing move_to_pose"
        if "not holding" in error_lower:
            return f"robot not holding required item at step {step.step_id} — missing pick"
        if "gripper" in error_lower and "holding" in error_lower:
            return f"gripper occupied at step {step.step_id} — must place held item first"
        if "not graspable" in error_lower or "not pushable" in error_lower:
            return f"object cannot be manipulated at step {step.step_id}"
        return f"step {step.step_id} failed: {error}"

    @staticmethod
    def _find_affected_steps(plan: Plan, failed_index: int) -> list[int]:
        if failed_index < 0:
            return [s.step_id for s in plan.steps]
        return [s.step_id for s in plan.steps[failed_index:]]

    @staticmethod
    def _suggest_fix(
        failure_type: FailureType,
        root_cause: str,
        step: PlanStep | None,
        plan: Plan,
        scene: Scene,
    ) -> str:
        if step is None:
            return "regenerate the entire plan"
        if "locked" in root_cause:
            return f"insert unlock_door + open_door before step {step.step_id}"
        if "not near" in root_cause:
            target = step.args.get("target") or step.args.get("door_id") or step.args.get("object_id")
            return f"insert move_to_pose(target={target}) before step {step.step_id}"
        if "not holding" in root_cause:
            return f"insert pick step for the required item before step {step.step_id}"
        if "gripper occupied" in root_cause:
            return f"insert place step to free gripper before step {step.step_id}"
        if failure_type == FailureType.INFEASIBLE:
            return "task is infeasible — call request_assistance"
        return f"repair steps from {step.step_id} onward"


class ClaudeFailureAnalyst:
    """LLM-based failure analysis (~500 token lightweight call)."""

    _SYSTEM_PROMPT = """\
You are a failure analyst for robot execution plans. Given a failed plan and
the specific error, diagnose the root cause and suggest a MINIMAL fix.

Output ONLY JSON:
{
  "failure_type": "precondition_unmet" | "tool_error" | "physics_reject" | "timeout" | "infeasible",
  "root_cause": "<one sentence explaining why the step failed>",
  "affected_steps": [<step_ids that need to change>],
  "suggested_fix": "<specific fix: insert/replace/reorder which steps>",
  "prefix_valid_through": <last step_id that executed correctly>
}
"""

    def __init__(self, client=None, model: str = "claude-haiku-4-5-20251001", max_tokens: int = 512):
        if client is None:
            from physgate.planner.planner import make_anthropic_client
            client = make_anthropic_client()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def __call__(
        self,
        plan: Plan,
        execution_result: dict[str, Any],
        scene: Scene,
    ) -> FailureReport:
        from physgate.world.scene_graph import to_query_scene_payload

        user_prompt = (
            f"Failed plan:\n{json.dumps(plan.model_dump(mode='json'), indent=2)}\n\n"
            f"Execution result:\n{json.dumps(execution_result, indent=2)}\n\n"
            f"Current scene:\n{json.dumps(to_query_scene_payload(scene), indent=2)}"
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return self._parse_response(text, plan, execution_result)

    @staticmethod
    def _parse_response(
        text: str, plan: Plan, execution_result: dict[str, Any]
    ) -> FailureReport:
        cleaned = text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start == -1 or end <= start:
                return _fallback_report(plan, execution_result)
            try:
                data = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return _fallback_report(plan, execution_result)

        failed_step_id = execution_result.get("failed_step_id")
        failed_step = None
        failed_index = -1
        if failed_step_id is not None:
            for i, s in enumerate(plan.steps):
                if s.step_id == failed_step_id:
                    failed_step = s
                    failed_index = i
                    break

        try:
            ft = FailureType(data.get("failure_type", "tool_error"))
        except ValueError:
            ft = FailureType.TOOL_ERROR

        return FailureReport(
            failed_step=failed_step,
            failed_step_index=failed_index,
            failure_type=ft,
            root_cause=data.get("root_cause", "unknown"),
            affected_steps=data.get("affected_steps", []),
            suggested_fix=data.get("suggested_fix", "regenerate the plan"),
            prefix_valid_through=data.get("prefix_valid_through", 0),
        )


def _fallback_report(plan: Plan, execution_result: dict[str, Any]) -> FailureReport:
    """Fallback when LLM response is unparseable."""
    failed_step_id = execution_result.get("failed_step_id")
    failed_step = None
    failed_index = -1
    if failed_step_id is not None:
        for i, s in enumerate(plan.steps):
            if s.step_id == failed_step_id:
                failed_step = s
                failed_index = i
                break
    return FailureReport(
        failed_step=failed_step,
        failed_step_index=failed_index,
        failure_type=FailureType.TOOL_ERROR,
        root_cause=execution_result.get("error", "unknown failure"),
        affected_steps=[s.step_id for s in plan.steps[max(0, failed_index):]],
        suggested_fix="regenerate the entire plan",
        prefix_valid_through=max(0, failed_index),
    )


def make_failure_analyst(client=None) -> ClaudeFailureAnalyst | MockFailureAnalyst:
    """Return ClaudeFailureAnalyst if LLM credentials available, else MockFailureAnalyst."""
    from physgate.planner.planner import llm_credentials_available
    if client is not None or llm_credentials_available():
        return ClaudeFailureAnalyst(client=client)
    return MockFailureAnalyst()
