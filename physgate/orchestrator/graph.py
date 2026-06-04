"""Orchestrator: LangGraph state machine wiring planner → critic → gate → executor.

Implements the failure-recovery state machine from architecture doc §2:

    planning → reviewing → validating → awaiting_approval → executing → done
        ↑__________|              |            |                |
        ↑ (all rejected)          | (all fail) | (denied → end) |
        ↑_________________________|            ↓                | (fail, retries
        ↑  replan budget: 2                   END               |  exhausted)
        ↑______________________________________________________|

All domain components are *injected callables* so the graph can run with mocks
(unit tests), with the real Claude planner + Isaac Lab gate (production), or
any mix. The orchestrator itself never calls an LLM and never touches the GPU.

Checkpointing: the graph accepts any LangGraph checkpointer. The MVP/demo uses
MemorySaver (see DECISIONS.md D-003); swap in PostgresSaver at the call site
for durable state.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, TypedDict, runtime_checkable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from physgate.gate.scoring import SelectionResult
from physgate.gate.schemas import Scene
from physgate.planner.schemas import DEFAULT_NUM_CANDIDATES, Plan

#: Sentinel: pause the graph at the approval gate and wait for a human verdict
#: (LangGraph interrupt). Requires a checkpointer. Resume with
#: :func:`resume_with_approval`.
HUMAN_APPROVAL = object()

# --------------------------------------------------------------------- config


class OrchestratorConfig(BaseModel):
    """Budgets and sizes for one orchestrated task."""

    num_candidates: int = DEFAULT_NUM_CANDIDATES
    max_replans: int = 2
    max_execution_retries: int = 3


# ---------------------------------------------------------------------- state


class OrchestratorState(TypedDict, total=False):
    """Mutable state carried through the LangGraph nodes."""

    task: str
    task_id: str
    scene: Scene
    candidates: list[Plan]
    survivors: list[Plan]
    selection: SelectionResult | None
    approved: bool
    execution_result: dict[str, Any] | None
    replan_count: int
    retry_count: int
    failure_feedback: str | None
    outcome: str
    trace: list[str]


# Component signatures (duck-typed; see tests and planner/critic modules).
PlannerFn = Callable[[str, Scene, int, str | None], list[Plan]]
CriticFn = Callable[[list[Plan], Scene], list[Plan]]
GateFn = Callable[[list[Plan], Scene], SelectionResult]
ExecutorFn = Callable[[Plan, Scene], dict[str, Any]]
ApprovalFn = Callable[[SelectionResult], bool]


@runtime_checkable
class Checkpointer(Protocol):
    """Minimal interface for LangGraph state persistence."""

    def get(self, config: dict) -> Any: ...
    def put(self, config: dict, data: Any) -> None: ...


@runtime_checkable
class AuditTrailProtocol(Protocol):
    """Minimal interface expected by the orchestrator for audit recording."""

    def record(self, stream: Any, event: str, correlation_id: str, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------- graph


def build_orchestrator(
    planner_fn: PlannerFn,
    critic_fn: CriticFn,
    gate_fn: GateFn,
    executor_fn: ExecutorFn,
    approval_fn: ApprovalFn,
    config: OrchestratorConfig | None = None,
    checkpointer: Checkpointer | None = None,
    audit_trail: AuditTrailProtocol | None = None,
):
    """Build and compile the orchestrator graph with injected components.

    Args:
        audit_trail: optional :class:`physgate.audit.trail.AuditTrail`; when
            given, every pipeline phase emits decision/human/physical records
            (architecture doc §4 three-stream audit).
    """
    cfg = config or OrchestratorConfig()

    def _audit(stream_name: str, event: str, state: OrchestratorState, payload: dict) -> None:
        if audit_trail is None:
            return
        from physgate.audit.records import AuditStream

        audit_trail.record(
            AuditStream(stream_name),
            event,
            correlation_id=f"{state.get('task_id', 'task')}/{event}",
            payload=payload,
        )

    def plan_node(state: OrchestratorState) -> dict:
        """Generate candidate plans from the planner."""
        candidates = planner_fn(
            state["task"],
            state["scene"],
            cfg.num_candidates,
            state.get("failure_feedback"),
        )
        _audit("decision", "candidates_generated", state, {"n": len(candidates)})
        return {
            "candidates": candidates,
            # reset downstream state for this fresh planning round
            "survivors": [],
            "selection": None,
            "approved": False,
            "execution_result": None,
            "retry_count": 0,
            "trace": state.get("trace", []) + ["planning"],
        }

    def review_node(state: OrchestratorState) -> dict:
        """Filter candidates through the safety critic."""
        survivors = critic_fn(state["candidates"], state["scene"])
        _audit(
            "decision",
            "critic_verdict",
            state,
            {"survivors": len(survivors), "rejected": len(state["candidates"]) - len(survivors)},
        )
        update: dict = {
            "survivors": survivors,
            "trace": state["trace"] + ["reviewing"],
        }
        if not survivors:
            update["failure_feedback"] = (
                "safety critic rejected all candidate plans; "
                "generate plans that respect the safety contracts"
            )
        return update

    def validate_node(state: OrchestratorState) -> dict:
        """Run survivors through the physics gate and select the best plan."""
        selection = gate_fn(state["survivors"], state["scene"])
        _audit(
            "decision",
            "plan_selected",
            state,
            {
                "best_plan_id": selection.best_plan_id,
                "any_feasible": selection.any_feasible,
                "scores": selection.scores,
            },
        )
        update: dict = {
            "selection": selection,
            "trace": state["trace"] + ["validating"],
        }
        if not selection.any_feasible:
            update["failure_feedback"] = selection.rationale
        return update

    def approve_node(state: OrchestratorState) -> dict:
        """Request human or automatic approval for the selected plan."""
        selection: SelectionResult = state["selection"]
        if approval_fn is HUMAN_APPROVAL:
            # pause the graph; a human inspects the selection and resumes with
            # True/False (architecture doc §5: approval via orchestrator interrupt)
            approved = interrupt(
                {
                    "question": "Approve execution of the selected plan?",
                    "task": state["task"],
                    "best_plan_id": selection.best_plan_id,
                    "rationale": selection.rationale,
                    "scores": selection.scores,
                    "num_candidates": len(state["candidates"]),
                    "num_survivors": len(state["survivors"]),
                }
            )
        else:
            approved = approval_fn(selection)
        _audit("human", "approval", state, {"approved": bool(approved)})
        return {
            "approved": bool(approved),
            "trace": state["trace"] + ["awaiting_approval"],
        }

    def execute_node(state: OrchestratorState) -> dict:
        """Execute the approved plan on the backend."""
        selection: SelectionResult = state["selection"]
        best_plan = next(
            p for p in state["survivors"] if p.plan_id == selection.best_plan_id
        )
        result = executor_fn(best_plan, state["scene"])
        _audit(
            "physical",
            "execution_result",
            state,
            {"plan_id": best_plan.plan_id, "success": bool(result.get("success"))},
        )
        update: dict = {
            "execution_result": result,
            "trace": state["trace"] + ["executing"],
        }
        if not result.get("success"):
            update["retry_count"] = state.get("retry_count", 0) + 1
            update["failure_feedback"] = (
                f"execution failed after validation: {result.get('error', 'unknown error')}"
            )
        return update

    def done_node(state: OrchestratorState) -> dict:
        """Mark the task as successfully completed."""
        return {"outcome": "done", "trace": state["trace"] + ["done"]}

    def escalate_node(state: OrchestratorState) -> dict:
        """Mark the task as escalated after exhausting retry budgets."""
        return {"outcome": "escalated", "trace": state["trace"] + ["escalated"]}

    def deny_node(state: OrchestratorState) -> dict:
        """Mark the task as denied by the approval gate."""
        return {"outcome": "denied", "trace": state["trace"] + ["denied"]}

    # ----- conditional routing -----

    def _replan_or_escalate(state: OrchestratorState) -> str:
        if state.get("replan_count", 0) < cfg.max_replans:
            return "replan"
        return "escalate"

    def route_after_review(state: OrchestratorState) -> str:
        """Route to validation if survivors exist, otherwise replan or escalate."""
        if state["survivors"]:
            return "validate"
        return _replan_or_escalate(state)

    def route_after_validate(state: OrchestratorState) -> str:
        """Route to approval if a feasible plan was found, otherwise replan or escalate."""
        if state["selection"].any_feasible:
            return "approve"
        return _replan_or_escalate(state)

    def route_after_approve(state: OrchestratorState) -> str:
        """Route to execution if approved, otherwise deny."""
        return "execute" if state["approved"] else "deny"

    def route_after_execute(state: OrchestratorState) -> str:
        """Route to done on success, retry or replan on failure."""
        if state["execution_result"].get("success"):
            return "done"
        if state.get("retry_count", 0) < cfg.max_execution_retries:
            return "retry"
        return _replan_or_escalate(state)

    # replan transitions pass through a counter bump
    def bump_replan_node(state: OrchestratorState) -> dict:
        """Increment the replan counter before re-entering the planning node."""
        return {"replan_count": state.get("replan_count", 0) + 1}

    graph = StateGraph(OrchestratorState)
    graph.add_node("plan", plan_node)
    graph.add_node("review", review_node)
    graph.add_node("validate", validate_node)
    graph.add_node("approve", approve_node)
    graph.add_node("execute", execute_node)
    graph.add_node("bump_replan", bump_replan_node)
    graph.add_node("finish_done", done_node)
    graph.add_node("finish_escalated", escalate_node)
    graph.add_node("finish_denied", deny_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "review")
    graph.add_conditional_edges(
        "review",
        route_after_review,
        {"validate": "validate", "replan": "bump_replan", "escalate": "finish_escalated"},
    )
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"approve": "approve", "replan": "bump_replan", "escalate": "finish_escalated"},
    )
    graph.add_conditional_edges(
        "approve",
        route_after_approve,
        {"execute": "execute", "deny": "finish_denied"},
    )
    graph.add_conditional_edges(
        "execute",
        route_after_execute,
        {
            "done": "finish_done",
            "retry": "execute",
            "replan": "bump_replan",
            "escalate": "finish_escalated",
        },
    )
    graph.add_edge("bump_replan", "plan")
    graph.add_edge("finish_done", END)
    graph.add_edge("finish_escalated", END)
    graph.add_edge("finish_denied", END)

    return graph.compile(checkpointer=checkpointer)


def run_task(
    compiled_graph,
    task: str,
    scene: Scene,
    thread_id: str = "default",
    recursion_limit: int = 100,
) -> OrchestratorState:
    """Run one task through a compiled orchestrator graph and return final state."""
    initial: OrchestratorState = {
        "task": task,
        "task_id": thread_id,
        "scene": scene,
        "candidates": [],
        "survivors": [],
        "selection": None,
        "approved": False,
        "execution_result": None,
        "replan_count": 0,
        "retry_count": 0,
        "failure_feedback": None,
        "outcome": "",
        "trace": [],
    }
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if compiled_graph.checkpointer is not None:
        config["configurable"] = {"thread_id": thread_id}
    return compiled_graph.invoke(initial, config=config)


def resume_with_approval(
    compiled_graph,
    approved: bool,
    thread_id: str = "default",
    recursion_limit: int = 100,
) -> OrchestratorState:
    """Resume a graph paused at the human-approval interrupt with a verdict.

    Args:
        compiled_graph: the graph previously run with ``approval_fn=HUMAN_APPROVAL``.
        approved: the human's verdict (True = execute the selected plan).
        thread_id: the same thread_id the paused run used.
    """
    if compiled_graph.checkpointer is None:
        raise ValueError("resume_with_approval requires the graph to have a checkpointer")
    config: dict[str, Any] = {
        "recursion_limit": recursion_limit,
        "configurable": {"thread_id": thread_id},
    }
    return compiled_graph.invoke(Command(resume=approved), config=config)
