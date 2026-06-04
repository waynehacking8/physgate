"""Agent-orchestrator metrics: what the evaluation suite scores (REBUILD.md Phase 3).

These replace the deprecated best-of-N cost/quality framing. The orchestrator is
scored on what it actually exists to do:

    end_to_end_success_rate     — feasible tasks completed (relations verified)
    infeasible_recognition_rate — impossible tasks recognized and escalated
    recovery_rate               — transient failures recovered via retry/replan
    decomposition_validity_rate — selected plans are well-ordered
    invalid_plan_catch_rate     — known-invalid probe plans rejected by the gate
    orchestrator_score          — unweighted mean of the above
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from physgate.planner.schemas import Plan, ToolName


class ScenarioResult(BaseModel):
    """Outcome of running one scenario through the orchestrator."""

    scenario_id: str
    category: str
    expected_outcome: str
    actual_outcome: str
    #: expected==actual AND (for done) required relations hold
    outcome_correct: bool
    #: outcome done AND every required final relation holds
    task_completed: bool
    #: the selected plan is well-ordered (None when no plan was selected)
    decomposition_valid: bool | None = None
    #: for gate-probe scenarios: the known-invalid plan was rejected (None otherwise)
    invalid_probe_caught: bool | None = None
    #: handoff was correctly triggered (True) or correctly not triggered (True)
    handoff_correct: bool | None = None
    #: fraction of steps using the correct tool for the scenario category
    tool_selection_correct: bool | None = None
    #: for infeasible scenarios: True if agent recognized infeasibility
    failure_recognized: bool | None = None
    #: fraction of prefix steps preserved during replan (0-1, None if no replan)
    prefix_preservation: float | None = None
    replans_used: int = 0
    retries_used: int = 0
    candidates_generated: int = 0
    survivors_after_critic: int = 0
    wall_s: float = 0.0
    detail: str = ""


class OrchestratorReport(BaseModel):
    """Full evaluation report for one planner/orchestrator configuration."""

    planner_name: str
    timestamp: str = ""
    results: list[ScenarioResult] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


# -------------------------------------------------------- decomposition check


def check_decomposition(plan: Plan, initially_held: str | None = None) -> bool:
    """Is the plan well-ordered? (symbolic check, no physics)

    Rules:
        * pick(X) requires the robot to have moved to X first,
        * pick(X) requires an empty gripper (no double-pick),
        * place(Y) requires holding something and having moved to Y,
        * open_door / unlock_door / press_button / push_object require nearness,
        * unlock_door requires holding a key,
        * call_elevator requires nearness to the elevator.

    Args:
        plan: the plan to check.
        initially_held: object id the gripper holds BEFORE the plan starts
            (occupied-gripper scenarios). Without this, a correct
            "set it down first" plan is scored invalid — an eval-interface
            bug masquerading as a planner deficiency (D-019).
    """
    near: str | None = None
    held: str | None = initially_held
    for step in plan.steps:
        if step.tool == ToolName.MOVE_TO_POSE:
            near = step.args.get("target")
        elif step.tool == ToolName.EXECUTE_SKILL:
            skill = step.args.get("skill")
            target = step.args.get("target")
            if skill == "pick":
                if near != target or held is not None:
                    return False
                held = target
            elif skill == "place":
                if held is None or near != target:
                    return False
                held = None
        elif step.tool == ToolName.OPEN_DOOR:
            if near != step.args.get("door_id"):
                return False
        elif step.tool == ToolName.UNLOCK_DOOR:
            if near != step.args.get("door_id") or held is None:
                return False
        elif step.tool == ToolName.PRESS_BUTTON:
            if near != step.args.get("button_id"):
                return False
        elif step.tool == ToolName.CALL_ELEVATOR:
            if near != step.args.get("elevator_id"):
                return False
        elif step.tool == ToolName.PUSH_OBJECT:
            if near != step.args.get("object_id"):
                return False
    return True


# ------------------------------------------------------------------ aggregates


def _rate(items: list[bool]) -> float:
    return round(sum(items) / len(items), 3) if items else 0.0


def compute_metrics(results: list[ScenarioResult]) -> dict[str, float]:
    """Aggregate scenario results into the orchestrator metrics."""
    feasible = [r for r in results if r.expected_outcome == "done"]
    infeasible = [r for r in results if r.expected_outcome == "escalated"]
    recovery = [r for r in results if r.category == "recovery" and r.expected_outcome == "done"]
    completed = [r for r in results if r.task_completed and r.decomposition_valid is not None]
    probes = [r for r in results if r.invalid_probe_caught is not None]
    handoffs = [r for r in results if r.handoff_correct is not None]

    tool_sel = [r for r in results if r.tool_selection_correct is not None]
    fail_rec = [r for r in results if r.failure_recognized is not None]
    prefix = [r for r in results if r.prefix_preservation is not None]

    metrics = {
        "end_to_end_success_rate": _rate([r.task_completed for r in feasible]),
        "infeasible_recognition_rate": _rate(
            [r.actual_outcome in ("escalated", "partial_success") for r in infeasible]
        ),
        "recovery_rate": _rate([r.task_completed for r in recovery]),
        "decomposition_validity_rate": _rate([bool(r.decomposition_valid) for r in completed]),
        "invalid_plan_catch_rate": _rate([bool(r.invalid_probe_caught) for r in probes]),
    }
    if handoffs:
        metrics["handoff_accuracy"] = _rate([bool(r.handoff_correct) for r in handoffs])
    if tool_sel:
        metrics["tool_selection_accuracy"] = _rate([bool(r.tool_selection_correct) for r in tool_sel])
    if fail_rec:
        tp = sum(1 for r in fail_rec if r.failure_recognized and r.expected_outcome == "escalated")
        fp = sum(1 for r in fail_rec if r.failure_recognized and r.expected_outcome == "done")
        fn = sum(1 for r in fail_rec if not r.failure_recognized and r.expected_outcome == "escalated")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["failure_recognition_precision"] = round(precision, 3)
        metrics["failure_recognition_recall"] = round(recall, 3)
    if prefix:
        metrics["plan_prefix_preservation"] = round(
            sum(r.prefix_preservation for r in prefix) / len(prefix), 3
        )
    replan_scenarios = [r for r in results if r.replans_used > 0]
    if replan_scenarios:
        metrics["replan_efficiency"] = _rate([r.task_completed for r in replan_scenarios])

    core_keys = [
        "end_to_end_success_rate", "infeasible_recognition_rate",
        "recovery_rate", "decomposition_validity_rate", "invalid_plan_catch_rate",
    ]
    metrics["orchestrator_score"] = round(
        sum(metrics[k] for k in core_keys) / len(core_keys), 3
    )
    return metrics
