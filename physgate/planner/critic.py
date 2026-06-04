"""Safety critic: adversarially prune candidate plans (SAFER pattern).

Two implementations behind the same callable interface
``critic(plans, scene) -> list[Plan]`` (returns the survivors):

* :class:`ClaudeCritic` — Claude API, judges each plan against the safety
  contracts; **fails closed** (rejects everything) if its response cannot be
  parsed.
* :class:`MockCritic` — deterministic structural contract checks, used in
  tests and when no API key is set.

Design reference: architecture doc §1 (Critic node).
"""

from __future__ import annotations

import json
import re
from physgate.gate.schemas import Scene
from physgate.planner.planner import DEFAULT_PLANNER_MODEL, AnthropicClientProtocol
from physgate.planner.schemas import Plan, ToolName
from physgate.world.scene_graph import objects_with_affordance, to_query_scene_payload

#: The safety contracts every plan is judged against.
SAFETY_CONTRACTS: tuple[str, ...] = (
    "Every manipulation step (execute_skill) must declare at least one precondition; "
    "no unchecked physical action is allowed.",
    "Plans may only reference objects that exist in the current scene.",
    "Only objects with the 'graspable' affordance may be targets of a pick skill.",
    "The robot must never carry more than one object at a time "
    "(no pick while gripper is holding something).",
    "Plans must not command motion toward humans or into regions outside the scene.",
)

_CRITIC_SYSTEM_PROMPT = """\
You are an adversarial safety critic for robot plans (SAFER pattern). Your job
is to REJECT any candidate plan that violates a safety contract. Be skeptical;
when in doubt, reject.

Safety contracts:
{contracts}

Respond ONLY with JSON:
{{
  "approved_plan_ids": ["<id>", ...],
  "rejections": {{"<id>": "<which contract it violates and why>", ...}}
}}
"""


class MockCritic:
    """Deterministic structural safety checks (no LLM)."""

    def __call__(self, plans: list[Plan], scene: Scene) -> list[Plan]:
        """Filter plans by deterministic structural safety checks."""
        return [p for p in plans if self._is_safe(p, scene)]

    @staticmethod
    def _is_safe(plan: Plan, scene: Scene) -> bool:
        scene_ids = {o.id for o in scene.objects}
        graspable_ids = {o.id for o in objects_with_affordance(scene, "graspable")}

        for step in plan.steps:
            target = step.args.get("target")

            # Contract: only reference objects present in the scene.
            if target is not None and target not in scene_ids:
                return False

            if step.tool == ToolName.EXECUTE_SKILL:
                # Contract: no unchecked physical action.
                if not step.preconditions:
                    return False
                # Contract: pick only graspable objects.
                if step.args.get("skill") == "pick" and target not in graspable_ids:
                    return False
        return True


class ClaudeCritic:
    """Claude API safety critic. Fails closed on unparseable responses."""

    def __init__(
        self,
        model: str = DEFAULT_PLANNER_MODEL,
        api_key: str | None = None,
        client: AnthropicClientProtocol | None = None,
        max_tokens: int = 4096,
    ):
        """Initialize with an Anthropic client, model name, and token budget."""
        if client is None:
            if api_key is not None:
                import anthropic

                client = anthropic.Anthropic(api_key=api_key)
            else:
                from physgate.planner.planner import make_anthropic_client

                client = make_anthropic_client()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def __call__(self, plans: list[Plan], scene: Scene) -> list[Plan]:
        """Judge plans against safety contracts via the Claude API."""
        if not plans:
            return []  # nothing to review — skip the LLM round-trip
        contracts = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(SAFETY_CONTRACTS))
        user_prompt = (
            f"Current scene:\n{json.dumps(to_query_scene_payload(scene), indent=2)}\n\n"
            f"Candidate plans to review:\n"
            f"{json.dumps([p.model_dump(mode='json') for p in plans], indent=2)}"
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_CRITIC_SYSTEM_PROMPT.format(contracts=contracts),
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")

        approved_ids = self._parse_approved_ids(text)
        if approved_ids is None:
            # Fail closed: an unreadable critic verdict approves nothing.
            return []
        return [p for p in plans if p.plan_id in approved_ids]

    @staticmethod
    def _parse_approved_ids(text: str) -> set[str] | None:
        cleaned = text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()
        try:
            verdict = json.loads(cleaned)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start == -1 or end <= start:
                return None
            try:
                verdict = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
        if not isinstance(verdict, dict) or "approved_plan_ids" not in verdict:
            return None
        return set(verdict["approved_plan_ids"])


def make_critic(
    model: str = DEFAULT_PLANNER_MODEL, client: AnthropicClientProtocol | None = None
) -> ClaudeCritic | MockCritic:
    """Return ClaudeCritic if LLM credentials are available, else MockCritic."""
    from physgate.planner.planner import llm_credentials_available

    if client is not None or llm_credentials_available():
        return ClaudeCritic(model=model, client=client)
    return MockCritic()
