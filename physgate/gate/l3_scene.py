"""Sim-Gate Layer 3: scene-graph precondition check.

Cheapest deterministic layer alongside L1 (kinematic). Validates that a plan
step's symbolic preconditions hold in the current scene graph *before* spending
GPU time on L2 physics simulation. Pure Python, no GPU.

Supported precondition predicates (MVP, fetch-and-place):
    "<id> exists"            object present in scene
    "gripper_empty"          gripper currently holds nothing
    "<a> <pred> <b>"         relation triple holds, e.g. "shelf_A unoccupied"
                             (negative relations expressed as their own predicate)

Design reference: docs/design/architecture.md section 5 (validation gate).
"""

from __future__ import annotations

from physgate.gate.schemas import (
    FailureCode,
    FailureReport,
    GateLayer,
    Scene,
    Violation,
)


def _check_one(precondition: str, scene: Scene) -> Violation | None:
    """Return a Violation if `precondition` does NOT hold, else None."""
    tokens = precondition.strip().split()

    # "gripper_empty"
    if precondition.strip() == "gripper_empty":
        if scene.gripper_empty:
            return None
        return Violation(type="gripper_not_empty", detail="gripper is holding an object")

    # "<id> exists"
    if len(tokens) == 2 and tokens[1] == "exists":
        object_id = tokens[0]
        if scene.has_object(object_id):
            return None
        return Violation(type="missing_object", detail=f"object '{object_id}' not in scene")

    # "<subject> <predicate> <object>"  relation triple
    if len(tokens) == 3:
        subject, predicate, obj = tokens
        if scene.has_relation(subject, predicate, obj):
            return None
        return Violation(
            type="unmet_relation",
            detail=f"relation ({subject}, {predicate}, {obj}) does not hold",
        )

    # Unknown precondition form: fail closed (safer than silently passing).
    return Violation(type="unparseable_precondition", detail=f"cannot parse: '{precondition}'")


def check_preconditions(
    preconditions: list[str],
    scene: Scene,
    step_id: int | None = None,
) -> FailureReport | None:
    """Check all preconditions of a plan step against the scene graph.

    Returns ``None`` if every precondition holds (PASS), or a populated
    :class:`FailureReport` describing every unmet precondition (DENIED).
    """
    violations = [v for p in preconditions if (v := _check_one(p, scene)) is not None]
    if not violations:
        return None

    return FailureReport(
        failed_step_id=step_id,
        failure_code=FailureCode.PRECONDITION_VIOLATION,
        layer=GateLayer.SEMANTIC_PRECONDITION,
        violations=violations,
        remediation_hint="resolve unmet preconditions before this step (see violations)",
        retryable=True,
    )


def check_step_targets(plan, scene: Scene) -> FailureReport | None:
    """Validate that every step's ``target`` arg names an object in the scene.

    LLM planners can hallucinate object ids (invented waypoints, staging areas,
    misspelled ids). Such plans must be rejected here — cheaply and with a clear
    violation — instead of being silently degraded into do-nothing missions at
    L2 (the failure mode behind the 2026-06-03 real-LLM demo escalation, see
    DECISIONS.md D-016).

    Args:
        plan: the candidate :class:`~physgate.planner.schemas.Plan`.
        scene: the current scene graph.

    Returns:
        ``None`` when every referenced target exists, else a FailureReport
        listing one ``unknown_object`` violation per hallucinated id.
    """
    violations = [
        Violation(
            type="unknown_object",
            detail=(
                f"step {step.step_id} ({step.tool.value}) references object "
                f"'{step.args.get('target')}' which does not exist in the scene"
            ),
        )
        for step in plan.steps
        if step.args.get("target") is not None and not scene.has_object(step.args["target"])
    ]
    if not violations:
        return None

    return FailureReport(
        failure_code=FailureCode.PRECONDITION_VIOLATION,
        layer=GateLayer.SEMANTIC_PRECONDITION,
        violations=violations,
        remediation_hint=(
            "only reference object ids that exist in the current scene; "
            "do not invent waypoints or locations"
        ),
        retryable=True,
    )
