"""Planner: natural-language task → N candidate Plans.

Two implementations behind the same callable interface
``planner(task, scene, n, feedback) -> list[Plan]``:

* :class:`ClaudePlanner` — calls the Claude API (Anthropic SDK). The real
  planner; requires ``ANTHROPIC_API_KEY``.
* :class:`MockPlanner` — deterministic, offline candidate generator for the
  fetch-and-place task. Used in tests and whenever no API key is set
  (see DECISIONS.md D-002).

:func:`make_planner` picks the right one from the environment.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from physgate.gate.schemas import Scene
from physgate.planner.schemas import Plan, PlanStep, RelationChange, ToolName
from physgate.world.scene_graph import objects_with_affordance, to_query_scene_payload

#: Planner LLM (architecture doc §1: Claude Opus 4.8 via cloud API).
DEFAULT_PLANNER_MODEL = "claude-opus-4-8"


def make_anthropic_client():
    """Create an Anthropic client from whichever credential is available.

    Supports both credential types:
    * ``ANTHROPIC_API_KEY``    — standard API key (x-api-key header)
    * ``ANTHROPIC_AUTH_TOKEN`` — Claude subscription OAuth token
      (Authorization: Bearer + oauth beta header)
    """
    import anthropic

    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if auth_token:
        return anthropic.Anthropic(
            auth_token=auth_token,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
        )
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def llm_credentials_available() -> bool:
    """True when either an API key or an OAuth token is configured."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

_SYSTEM_PROMPT = """\
You are a robot task planner for a Unitree Go2 quadruped with a top-mounted gripper.
Decompose the user's task into candidate plans. Each plan is a flat list of steps;
each step calls exactly one tool: query_scene, move_to_pose, or execute_skill.

Available skills for execute_skill: pick, place.

Output ONLY a JSON array of plan objects, no prose. Each plan object:
{
  "plan_id": "<unique id>",
  "task": "<the task>",
  "rationale": "<why this approach>",
  "steps": [
    {
      "step_id": <int>,
      "tool": "move_to_pose" | "execute_skill" | "query_scene",
      "args": {"target": "<object id>", ...},
      "preconditions": ["<object id> exists", "gripper_empty", "<subj> <pred> <obj>"],
      "effects": [{"op": "add"|"remove", "subject": "...", "predicate": "...", "object": "..."}]
    }
  ]
}

Generate plans that are meaningfully DIFFERENT (different routes, orderings,
intermediate checks) so physics validation can select the best one.
Every manipulation step must declare its preconditions.
"""


def _extract_json_array(text: str) -> list[Any]:
    """Pull the first JSON array out of an LLM response (handles ``` fences)."""
    cleaned = text.strip()
    # strip markdown fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    # direct parse, else find the outermost array
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start == -1 or end <= start:
            raise ValueError(f"could not parse a JSON array from LLM response: {text[:200]!r}")
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON array of plans, got {type(parsed).__name__}")
    return parsed


class ClaudePlanner:
    """Claude API planner. ``planner(task, scene, n, feedback) -> list[Plan]``."""

    def __init__(
        self,
        model: str = DEFAULT_PLANNER_MODEL,
        api_key: str | None = None,
        client: Any = None,
        max_tokens: int = 16384,
    ):
        if client is None:
            if api_key is not None:
                import anthropic

                client = anthropic.Anthropic(api_key=api_key)
            else:
                client = make_anthropic_client()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def __call__(
        self, task: str, scene: Scene, n: int, feedback: str | None = None
    ) -> list[Plan]:
        user_prompt = (
            f"Task: {task}\n\n"
            f"Current scene (query_scene output):\n"
            f"{json.dumps(to_query_scene_payload(scene), indent=2)}\n\n"
            f"Generate exactly {n} candidate plans as a JSON array."
        )
        if feedback:
            user_prompt += (
                f"\n\nPREVIOUS ATTEMPT FAILED. Failure report:\n{feedback}\n"
                "Generate new plans that avoid this failure mode."
            )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        raw_plans = _extract_json_array(text)

        plans: list[Plan] = []
        for raw in raw_plans:
            try:
                plans.append(Plan.model_validate(raw))
            except Exception:  # noqa: BLE001 — invalid candidates are dropped, not fatal
                continue
        if not plans:
            raise ValueError(f"could not parse any valid Plan from LLM response: {text[:200]!r}")
        return plans


class MockPlanner:
    """Deterministic offline planner for the fetch-and-place MVP task.

    Generates ``n`` structurally diverse candidates so the critic and gate have
    real selection work to do. Used when no ANTHROPIC_API_KEY is available.
    """

    _last_scene: Scene | None = None

    def __call__(
        self, task: str, scene: Scene, n: int, feedback: str | None = None
    ) -> list[Plan]:
        self._last_scene = scene  # used by route variants that need scene lookups
        fetch_target = self._fetch_target(scene)
        place_target = self._place_target(scene)
        rationale_suffix = (
            " (replan after gate feedback)" if feedback else ""
        )

        variants = [
            self._direct_plan,
            self._scan_first_plan,
            self._cautious_plan,
            self._no_precondition_plan,  # deliberately unsafe → critic should prune
        ]
        plans = []
        for i in range(n):
            variant = variants[i % len(variants)]
            plan = variant(i, task, fetch_target, place_target, rationale_suffix)
            plans.append(plan)
        return plans

    # ----- scene introspection -----

    @staticmethod
    def _fetch_target(scene: Scene) -> str:
        anomalies = [o for o in scene.objects if o.is_anomaly]
        if anomalies:
            return anomalies[0].id
        graspable = objects_with_affordance(scene, "graspable")
        return graspable[0].id if graspable else scene.objects[0].id

    @staticmethod
    def _place_target(scene: Scene) -> str:
        placeable = objects_with_affordance(scene, "placeable")
        if placeable:
            return placeable[0].id
        non_anomalies = [o for o in scene.objects if not o.is_anomaly]
        return non_anomalies[0].id if non_anomalies else scene.objects[-1].id

    @staticmethod
    def _waypoint(scene: Scene) -> str | None:
        """Find a navigation waypoint marker in the scene, if any."""
        waypoints = [o for o in scene.objects if o.label == "waypoint"]
        return waypoints[0].id if waypoints else None

    # ----- plan variants -----

    @staticmethod
    def _pick_step(step_id: int, target: str) -> PlanStep:
        return PlanStep(
            step_id=step_id,
            tool=ToolName.EXECUTE_SKILL,
            args={"skill": "pick", "target": target},
            preconditions=[f"{target} exists", "gripper_empty"],
            effects=[
                RelationChange(op="add", subject="gripper", predicate="holding", object=target)
            ],
        )

    @staticmethod
    def _place_step(step_id: int, target: str, place_on: str) -> PlanStep:
        return PlanStep(
            step_id=step_id,
            tool=ToolName.EXECUTE_SKILL,
            args={"skill": "place", "target": place_on},
            preconditions=[f"{place_on} exists", f"gripper holding {target}"],
            effects=[
                RelationChange(op="remove", subject="gripper", predicate="holding", object=target),
                RelationChange(op="add", subject=target, predicate="on", object=place_on),
            ],
        )

    @staticmethod
    def _move_step(step_id: int, target: str, standoff: float, speed: float = 0.5) -> PlanStep:
        return PlanStep(
            step_id=step_id,
            tool=ToolName.MOVE_TO_POSE,
            args={"target": target, "standoff_m": standoff, "speed": speed},
            preconditions=[f"{target} exists"],
            effects=[
                RelationChange(op="add", subject="robot", predicate="near", object=target)
            ],
        )

    def _direct_plan(self, i, task, fetch, place, suffix) -> Plan:
        return Plan(
            plan_id=f"mock_{i}_direct",
            task=task,
            rationale=f"direct route: approach, pick, carry, place{suffix}",
            steps=[
                self._move_step(1, fetch, standoff=0.3, speed=0.5 + 0.1 * (i % 3)),
                self._pick_step(2, fetch),
                self._move_step(3, place, standoff=0.4),
                self._place_step(4, fetch, place),
            ],
        )

    def _scan_first_plan(self, i, task, fetch, place, suffix) -> Plan:
        return Plan(
            plan_id=f"mock_{i}_scan_first",
            task=task,
            rationale=f"scan the scene before acting to confirm object pose{suffix}",
            steps=[
                PlanStep(step_id=1, tool=ToolName.QUERY_SCENE, args={}),
                self._move_step(2, fetch, standoff=0.25, speed=0.4),
                self._pick_step(3, fetch),
                self._move_step(4, place, standoff=0.35),
                self._place_step(5, fetch, place),
            ],
        )

    def _cautious_plan(self, i, task, fetch, place, suffix) -> Plan:
        """Slow detour route: goes via a waypoint (if the scene has one) instead
        of cutting straight across — trades time for clearance."""
        waypoint = self._waypoint(self._last_scene) if self._last_scene else None
        detour_steps = (
            [self._move_step(3, waypoint, standoff=0.0, speed=0.25)] if waypoint else []
        )
        steps = [
            self._move_step(1, fetch, standoff=0.5, speed=0.25),
            self._pick_step(2, fetch),
            *detour_steps,
            self._move_step(4, place, standoff=0.5, speed=0.25),
            self._place_step(5, fetch, place),
        ]
        return Plan(
            plan_id=f"mock_{i}_cautious",
            task=task,
            rationale=f"slow detour route via waypoint for clearance{suffix}",
            steps=steps,
        )

    def _no_precondition_plan(self, i, task, fetch, place, suffix) -> Plan:
        """Deliberately reckless candidate (no preconditions) — critic prunes it."""
        return Plan(
            plan_id=f"mock_{i}_reckless",
            task=task,
            rationale=f"fastest possible: act without checking{suffix}",
            steps=[
                PlanStep(
                    step_id=1,
                    tool=ToolName.EXECUTE_SKILL,
                    args={"skill": "pick", "target": fetch},
                    preconditions=[],
                ),
                PlanStep(
                    step_id=2,
                    tool=ToolName.EXECUTE_SKILL,
                    args={"skill": "place", "target": place},
                    preconditions=[],
                ),
            ],
        )


def make_planner(
    model: str = DEFAULT_PLANNER_MODEL, client: Any = None
) -> ClaudePlanner | MockPlanner:
    """Return ClaudePlanner if LLM credentials are available, else MockPlanner."""
    if client is not None or llm_credentials_available():
        return ClaudePlanner(model=model, client=client)
    return MockPlanner()
