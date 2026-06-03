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


def check_decomposition(plan: Plan) -> bool:
    """Is the plan well-ordered? (symbolic check, no physics)

    Rules:
        * pick(X) requires the robot to have moved to X first,
        * pick(X) requires an empty gripper (no double-pick),
        * place(Y) requires holding something and having moved to Y.
    """
    near: str | None = None
    held: str | None = None
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

    metrics = {
        "end_to_end_success_rate": _rate([r.task_completed for r in feasible]),
        "infeasible_recognition_rate": _rate(
            [r.actual_outcome == "escalated" for r in infeasible]
        ),
        "recovery_rate": _rate([r.task_completed for r in recovery]),
        "decomposition_validity_rate": _rate([bool(r.decomposition_valid) for r in completed]),
        "invalid_plan_catch_rate": _rate([bool(r.invalid_probe_caught) for r in probes]),
    }
    metrics["orchestrator_score"] = round(sum(metrics.values()) / len(metrics), 3)
    return metrics
