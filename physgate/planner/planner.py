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
from typing import Any, Protocol, runtime_checkable

from physgate.gate.schemas import Scene
from physgate.planner.schemas import Plan, PlanStep, RelationChange, ToolName
from physgate.world.scene_graph import objects_with_affordance, to_query_scene_payload


@runtime_checkable
class AnthropicClientProtocol(Protocol):
    """Minimal interface matching the Anthropic SDK client (or headless wrapper)."""

    class _Messages(Protocol):
        def create(self, *, model: str, max_tokens: int, system: str, messages: list) -> Any: ...

    messages: _Messages

_MAX_FEEDBACK_LEN = 2000
_MAX_TASK_LEN = 1000


def _sanitize_prompt_input(text: str, max_len: int) -> str:
    """Truncate and strip prompt-injection patterns from user-controlled strings."""
    text = text[:max_len]
    text = re.sub(r"(?i)(system|assistant|human)\s*:", "", text)
    text = re.sub(r"<\|.*?\|>", "", text)
    return text.strip()

#: Planner LLM (architecture doc §1: Claude Opus 4.8 via cloud API).
DEFAULT_PLANNER_MODEL = "claude-opus-4-8"


#: Subscription OAuth tokens carry this prefix; they are rejected by the raw
#: Anthropic API and must be routed through claude -p (DECISIONS.md D-015).
_SUBSCRIPTION_TOKEN_PREFIX = "sk-ant-oat"


def make_anthropic_client():
    """Create an LLM client from whichever credential is available.

    Credential precedence (see DECISIONS.md D-015):

    1. ``ANTHROPIC_API_KEY``       — standard API key → raw Anthropic SDK
       (pay-per-token, unrestricted).
    2. ``CLAUDE_CODE_OAUTH_TOKEN`` — Claude subscription OAuth token
       (``sk-ant-oat01-...``, from ``claude setup-token``) → Claude Code
       headless mode (``claude -p``). The raw API rejects these tokens.
    3. ``ANTHROPIC_AUTH_TOKEN``    — if it holds a subscription token, route it
       through claude -p too; otherwise treat it as a gateway/proxy bearer
       token for the raw SDK.

    The anthropic SDK is imported only on the raw-SDK paths, so subscription-only
    installs (claude -p headless) do not need the ``anthropic`` package at all.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        import anthropic

        return anthropic.Anthropic(api_key=api_key)

    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if oauth_token or (auth_token and auth_token.startswith(_SUBSCRIPTION_TOKEN_PREFIX)):
        from physgate.planner.headless_client import ClaudeCodeHeadlessClient

        return ClaudeCodeHeadlessClient(oauth_token=oauth_token or auth_token)

    if auth_token:
        import anthropic

        return anthropic.Anthropic(
            auth_token=auth_token,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
        )
    raise ValueError("no LLM credentials: set ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN")


def llm_credentials_available() -> bool:
    """True when an API key or an OAuth token is configured."""
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )


_SYSTEM_PROMPT = """\
You are a robot task planner for a Unitree Go2 quadruped with a top-mounted gripper.

TOOL REGISTRY (complete — every tool the robot can use):

  query_scene()
    Returns: objects (id, label, floor), relations, gripper state, available_tools.
    No precondition.

  move_to_pose(target, standoff_m=0.3, speed=0.5)
    Moves robot base near the target object. A deterministic A* planner handles
    obstacle avoidance — you do NOT plan routes.
    Precondition: target exists in scene, robot and target on the SAME FLOOR.
    Effect: robot near <target>.

  execute_skill(skill="pick"|"place", target)
    pick: grasp target object. Precondition: robot near target, gripper empty,
          target is graspable. Effect: gripper holding <target>.
    place: release held object onto target. Precondition: robot near target,
           gripper holding something. Effect: <held> on <target>, gripper empty.

  open_door(door_id)
    Opens a closed door. Precondition: robot near door, door is NOT locked.
    Effect: door state → open.

  unlock_door(door_id, key_id)
    Unlocks a locked door using a key. Precondition: robot near door,
    gripper holding key_id. Effect: door locked → false.

  press_button(button_id)
    Presses a button; effect depends on what the button activates (shown in
    scene relations as "button activates <target>").
    Precondition: robot near button.

  call_elevator(elevator_id, target_floor)
    Rides the elevator to target_floor. Precondition: robot near elevator,
    robot on same floor as elevator. Effect: robot + elevator + held object
    all move to target_floor.

  push_object(object_id, direction="north"|"south"|"east"|"west")
    Pushes a movable obstacle in a cardinal direction. Precondition: robot
    near object, object is pushable. Effect: object pushed, blocking relations
    removed.

  inspect_object(object_id)
    Returns detailed properties: weight, graspability, lock state, pushability.
    These properties are NOT available from query_scene — you MUST inspect first.
    No nearness precondition.

  request_assistance(message)
    Signals the robot cannot complete the task alone. Use when inspection
    reveals the task is infeasible (object too heavy, no path, etc.).

YOUR JOB: Given a task and scene, select the right tools, put them in the right
order, and declare preconditions/effects for each step. You choose which tools
to use — there is no template. Different tasks need different tool combinations.

The scene's available_tools field tells you which tools are physically present
in this scene. Do NOT use a tool that is not listed in available_tools.

Output ONLY a JSON array of plan objects, no prose. Each plan object:
{
  "plan_id": "<unique id>",
  "task": "<the task>",
  "rationale": "<why this approach>",
  "steps": [
    {
      "step_id": <int>,
      "tool": "<tool name>",
      "args": {<tool-specific args>},
      "preconditions": ["<condition>", ...],
      "effects": [{"op": "add"|"remove", "subject": "...", "predicate": "...", "object": "..."}]
    }
  ]
}

HARD CONSTRAINTS (the gate rejects plans that violate these):
- Every id value MUST be copied EXACTLY from the scene — never invent ids.
- ONLY use tools listed in available_tools.
- move_to_pose BEFORE any tool requiring nearness.

RELATION VOCABULARY for preconditions and effects:
- "<id> exists"               object in scene
- "gripper_empty"             nothing held
- "robot near <id>"           effect of move_to_pose
- "gripper holding <id>"      effect of pick
- "<id> on <id>"              support relation

Use literal subjects "robot" and "gripper" — not the robot's scene object id.

If the task is impossible with available tools, return [] or plans ending with
request_assistance explaining why.
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
        client: AnthropicClientProtocol | None = None,
        max_tokens: int = 16384,
    ):
        """Initialize with an Anthropic client, model name, and token budget."""
        if client is None:
            if api_key is not None:
                import anthropic

                client = anthropic.Anthropic(api_key=api_key)
            else:
                client = make_anthropic_client()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def __call__(self, task: str, scene: Scene, n: int, feedback: str | None = None) -> list[Plan]:
        """Generate n candidate plans for the task via the Claude API."""
        safe_task = _sanitize_prompt_input(task, _MAX_TASK_LEN)
        user_prompt = (
            f"Task: {safe_task}\n\n"
            f"Current scene (query_scene output):\n"
            f"{json.dumps(to_query_scene_payload(scene), indent=2)}\n\n"
            f"Generate exactly {n} candidate plans as a JSON array."
        )
        if feedback:
            safe_feedback = _sanitize_prompt_input(feedback, _MAX_FEEDBACK_LEN)
            user_prompt += (
                f"\n\nPREVIOUS ATTEMPT FAILED. Failure report:\n{safe_feedback}\n"
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

        # an empty array is the LLM's explicit "this task is impossible" signal
        # (per the system prompt) — the orchestrator escalates on zero candidates
        if not raw_plans:
            return []

        plans: list[Plan] = []
        for raw in raw_plans:
            try:
                plans.append(Plan.model_validate(raw))
            except (ValueError, KeyError, TypeError):
                continue
        if not plans:
            raise ValueError(f"could not parse any valid Plan from LLM response: {text[:200]!r}")
        return plans


class MockPlanner:
    """Deterministic baseline planner — NOT an agent.

    Uses if-else task-type detection and template-based plan generation.
    This is a testing/evaluation baseline, not LLM reasoning. The real
    agent planner is :class:`ClaudePlanner`, which selects tools purely
    from the scene's available_tools via LLM inference.
    """

    def __call__(self, task: str, scene: Scene, n: int, feedback: str | None = None) -> list[Plan]:
        """Generate n candidate plans, detecting task type from scene."""
        task_type = self._detect_task_type(scene, task)
        generator = {
            "locked_door": self._locked_door_plans,
            "blocked_path": self._blocked_path_plans,
            "elevator": self._elevator_plans,
            "sequential": self._sequential_plans,
            "infeasible_heavy": self._infeasible_plans,
            "infeasible_sealed": self._infeasible_plans,
            "infeasible_blocked": self._infeasible_plans,
            "fetch_and_place": self._fetch_and_place_plans,
        }.get(task_type, self._fetch_and_place_plans)
        return generator(task, scene, n, feedback)

    # ----- task type detection -----

    @staticmethod
    def _detect_task_type(scene: Scene, task: str) -> str:
        obj_map = {o.id: o for o in scene.objects}
        has_locked_door = any(
            "door" in o.affordances and o.locked for o in scene.objects
        )
        has_elevator = any("elevator" in o.affordances for o in scene.objects)
        blocking_rels = [r for r in scene.relations if r[1] == "blocking"]
        has_pushable_blocking = any(
            r[0] in obj_map and obj_map[r[0]].pushable for r in blocking_rels
        )
        has_immovable_blocking = any(
            r[0] in obj_map and not obj_map[r[0]].pushable for r in blocking_rels
        ) and not has_pushable_blocking
        has_sealed = any(r[1] == "enclosed_by" for r in scene.relations)
        has_heavy = any(
            o.weight_kg > 5.0 and not o.pushable and "graspable" not in o.affordances
            for o in scene.objects
            if o.is_anomaly
        )
        anomalies = [o for o in scene.objects if o.is_anomaly]
        multiple_anomalies = len(anomalies) >= 2

        if has_locked_door:
            return "locked_door"
        if has_sealed:
            return "infeasible_sealed"
        if has_heavy:
            return "infeasible_heavy"
        if has_immovable_blocking:
            return "infeasible_blocked"
        if has_elevator:
            return "elevator"
        if has_pushable_blocking:
            return "blocked_path"
        if multiple_anomalies:
            return "sequential"
        return "fetch_and_place"

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

    # ----- step builders -----

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
            effects=[RelationChange(op="add", subject="robot", predicate="near", object=target)],
        )

    # ----- T1: fetch and place (original) -----

    def _fetch_and_place_plans(
        self, task: str, scene: Scene, n: int, feedback: str | None
    ) -> list[Plan]:
        fetch = self._fetch_target(scene)
        place = self._place_target(scene)
        suffix = " (replan)" if feedback else ""
        variants = [
            self._direct_plan,
            self._scan_first_plan,
            self._cautious_plan,
            self._no_precondition_plan,
        ]
        return [
            variants[i % len(variants)](i, task, fetch, place, suffix)
            for i in range(n)
        ]

    # ----- T2: locked door delivery -----

    def _locked_door_plans(
        self, task: str, scene: Scene, n: int, feedback: str | None
    ) -> list[Plan]:
        fetch = self._fetch_target(scene)
        place = self._place_target(scene)
        door = next((o.id for o in scene.objects if "door" in o.affordances), "door_01")
        key = next(
            (o.id for o in scene.objects if o.label == "key" and "graspable" in o.affordances),
            None,
        )
        if key is None:
            return []

        plans: list[Plan] = []
        for i in range(n):
            sid = 1
            steps: list[PlanStep] = []
            if i % 2 == 0:
                steps.append(PlanStep(step_id=sid, tool=ToolName.QUERY_SCENE, args={}))
                sid += 1
            steps.append(self._move_step(sid, key, standoff=0.3)); sid += 1
            steps.append(self._pick_step(sid, key)); sid += 1
            steps.append(self._move_step(sid, door, standoff=0.3)); sid += 1
            steps.append(PlanStep(
                step_id=sid, tool=ToolName.UNLOCK_DOOR,
                args={"door_id": door, "key_id": key},
                preconditions=[f"robot near {door}", f"gripper holding {key}"],
            )); sid += 1
            steps.append(PlanStep(
                step_id=sid, tool=ToolName.OPEN_DOOR,
                args={"door_id": door},
                preconditions=[f"robot near {door}", f"{door} unlocked"],
            )); sid += 1
            # put key down, pick box, deliver
            floor = next((o.id for o in scene.objects if o.label == "floor"), "floor_01")
            steps.append(self._move_step(sid, floor, standoff=0.3)); sid += 1
            steps.append(self._place_step(sid, key, floor)); sid += 1
            steps.append(self._move_step(sid, fetch, standoff=0.3)); sid += 1
            steps.append(self._pick_step(sid, fetch)); sid += 1
            steps.append(self._move_step(sid, place, standoff=0.4)); sid += 1
            steps.append(self._place_step(sid, fetch, place)); sid += 1
            plans.append(Plan(
                plan_id=f"mock_{i}_locked_door", task=task,
                rationale=f"key→unlock→open→deliver chain (variant {i})",
                steps=steps,
            ))
        return plans

    # ----- T3: blocked path -----

    def _blocked_path_plans(
        self, task: str, scene: Scene, n: int, feedback: str | None
    ) -> list[Plan]:
        fetch = self._fetch_target(scene)
        place = self._place_target(scene)
        blocker = next((o.id for o in scene.objects if o.pushable), None)

        plans: list[Plan] = []
        for i in range(n):
            sid = 1
            steps: list[PlanStep] = []
            if blocker:
                steps.append(PlanStep(
                    step_id=sid, tool=ToolName.INSPECT_OBJECT,
                    args={"object_id": blocker},
                )); sid += 1
                steps.append(self._move_step(sid, blocker, standoff=0.3)); sid += 1
                steps.append(PlanStep(
                    step_id=sid, tool=ToolName.PUSH_OBJECT,
                    args={"object_id": blocker, "direction": "east"},
                    preconditions=[f"robot near {blocker}", f"{blocker} pushable"],
                )); sid += 1
            steps.append(self._move_step(sid, fetch, standoff=0.3)); sid += 1
            steps.append(self._pick_step(sid, fetch)); sid += 1
            steps.append(self._move_step(sid, place, standoff=0.4)); sid += 1
            steps.append(self._place_step(sid, fetch, place)); sid += 1
            plans.append(Plan(
                plan_id=f"mock_{i}_blocked_path", task=task,
                rationale=f"push blocker aside then deliver (variant {i})",
                steps=steps,
            ))
        return plans

    # ----- T4: sequential multi-object -----

    def _sequential_plans(
        self, task: str, scene: Scene, n: int, feedback: str | None
    ) -> list[Plan]:
        anomalies = [o for o in scene.objects if o.is_anomaly]
        placeable = [o for o in scene.objects if "placeable" in o.affordances]

        plans: list[Plan] = []
        for i in range(n):
            sid = 1
            steps: list[PlanStep] = []
            # clear occupied shelves first
            for shelf in placeable:
                occupants = [
                    r[0] for r in scene.relations
                    if r[1] == "on" and r[2] == shelf.id
                    and r[0] not in [a.id for a in anomalies]
                ]
                for occ in occupants:
                    floor = next((o.id for o in scene.objects if o.label == "floor"), "floor_01")
                    steps.append(self._move_step(sid, occ, standoff=0.3)); sid += 1
                    steps.append(self._pick_step(sid, occ)); sid += 1
                    steps.append(self._move_step(sid, floor, standoff=0.3)); sid += 1
                    steps.append(self._place_step(sid, occ, floor)); sid += 1

            # deliver each anomaly to a placeable target
            for j, anomaly in enumerate(anomalies):
                target = placeable[j % len(placeable)].id if placeable else "floor_01"
                steps.append(self._move_step(sid, anomaly.id, standoff=0.3)); sid += 1
                steps.append(self._pick_step(sid, anomaly.id)); sid += 1
                steps.append(self._move_step(sid, target, standoff=0.4)); sid += 1
                steps.append(self._place_step(sid, anomaly.id, target)); sid += 1

            plans.append(Plan(
                plan_id=f"mock_{i}_sequential", task=task,
                rationale=f"clear then deliver in sequence (variant {i})",
                steps=steps,
            ))
        return plans

    # ----- T5: elevator -----

    def _elevator_plans(
        self, task: str, scene: Scene, n: int, feedback: str | None
    ) -> list[Plan]:
        fetch = self._fetch_target(scene)
        place = self._place_target(scene)
        elevator = next((o.id for o in scene.objects if "elevator" in o.affordances), "elevator_01")
        target_floor = next(
            (o.floor for o in scene.objects if o.id == place), 2
        )

        plans: list[Plan] = []
        for i in range(n):
            sid = 1
            steps: list[PlanStep] = [
                self._move_step(sid, fetch, standoff=0.3),
            ]; sid += 1
            steps.append(self._pick_step(sid, fetch)); sid += 1
            steps.append(self._move_step(sid, elevator, standoff=0.3)); sid += 1
            steps.append(PlanStep(
                step_id=sid, tool=ToolName.CALL_ELEVATOR,
                args={"elevator_id": elevator, "target_floor": target_floor},
                preconditions=[f"robot near {elevator}"],
            )); sid += 1
            steps.append(self._move_step(sid, place, standoff=0.4)); sid += 1
            steps.append(self._place_step(sid, fetch, place)); sid += 1
            plans.append(Plan(
                plan_id=f"mock_{i}_elevator", task=task,
                rationale=f"pick→elevator→deliver across floors (variant {i})",
                steps=steps,
            ))
        return plans

    # ----- T6: infeasible -----

    def _infeasible_plans(
        self, task: str, scene: Scene, n: int, feedback: str | None
    ) -> list[Plan]:
        anomaly = next((o for o in scene.objects if o.is_anomaly), None)
        if anomaly is None:
            return []
        plans: list[Plan] = []
        for i in range(n):
            steps = [
                PlanStep(
                    step_id=1, tool=ToolName.INSPECT_OBJECT,
                    args={"object_id": anomaly.id},
                ),
                PlanStep(
                    step_id=2, tool=ToolName.REQUEST_ASSISTANCE,
                    args={"message": f"cannot handle {anomaly.id}: task infeasible"},
                ),
            ]
            plans.append(Plan(
                plan_id=f"mock_{i}_infeasible", task=task,
                rationale=f"inspect then request assistance (variant {i})",
                steps=steps,
            ))
        return plans

    # ----- T1 plan variants (original) -----

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
        steps = [
            self._move_step(1, fetch, standoff=0.5, speed=0.25),
            self._pick_step(2, fetch),
            self._move_step(3, place, standoff=0.5, speed=0.25),
            self._place_step(4, fetch, place),
        ]
        return Plan(
            plan_id=f"mock_{i}_cautious",
            task=task,
            rationale=f"slow approach with wide standoff for clearance{suffix}",
            steps=steps,
        )

    def _no_precondition_plan(self, i, task, fetch, place, suffix) -> Plan:
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
    model: str = DEFAULT_PLANNER_MODEL, client: AnthropicClientProtocol | None = None
) -> ClaudePlanner | MockPlanner:
    """Return ClaudePlanner if LLM credentials are available, else MockPlanner."""
    if client is not None or llm_credentials_available():
        return ClaudePlanner(model=model, client=client)
    return MockPlanner()
