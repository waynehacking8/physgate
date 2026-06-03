"""Fetch-and-place demo pipeline: the physgate MVP end to end.

    natural-language task
        -> planner (Claude API or mock)        N=8 candidate plans
        -> safety critic (Claude API or mock)  prune unsafe candidates
        -> Sim-Gate (L1 -> L3 -> L2 physics)   best-of-N selection
        -> approval gate                        (auto-approve in demo)
        -> executor                             run the winning plan
        -> report

The pipeline pieces are swappable:
  * planner/critic: real Claude when LLM credentials are set (API key or subscription OAuth), mock otherwise
  * L2 physics:     Isaac Lab when available, symbolic rollout otherwise
  * executor:       Isaac Sim backend when available, symbolic backend otherwise

examples/fetch_and_place.py is the runnable terminal entry point.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from physgate.executor.backend import MockWorldBackend, WorldBackend
from physgate.executor.plan_executor import execute_plan
from physgate.gate.parallel import L2Fn, run_gate, symbolic_l2
from physgate.gate.schemas import Scene, SceneObject
from physgate.gate.scoring import SelectionResult
from physgate.orchestrator.graph import OrchestratorConfig, build_orchestrator, run_task
from physgate.planner.critic import make_critic
from physgate.planner.planner import make_planner
from physgate.planner.schemas import Plan

DEFAULT_TASK = "put the fallen box back on shelf A"


def build_demo_scene() -> Scene:
    """The MVP scene: a box has fallen off shelf A onto the floor."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03",
                label="cardboard_box",
                affordances=["graspable"],
                is_anomaly=True,  # it should be on the shelf, not the floor
            ),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[("box_03", "on", "floor_01"), ("shelf_A", "unoccupied", "shelf_A")],
        gripper_empty=True,
    )


def _format_report(
    task: str,
    final_state: dict[str, Any],
    planner_name: str,
    critic_name: str,
    l2_name: str,
    backend_name: str,
) -> str:
    """Human-readable end-of-run report for the terminal."""
    candidates: list[Plan] = final_state.get("candidates", [])
    survivors: list[Plan] = final_state.get("survivors", [])
    selection: SelectionResult | None = final_state.get("selection")
    execution = final_state.get("execution_result") or {}
    outcome = final_state.get("outcome", "?")

    lines = [
        "=" * 72,
        "physgate — fetch-and-place demo",
        "=" * 72,
        f"task:            {task}",
        f"planner:         {planner_name}",
        f"critic:          {critic_name}",
        f"L2 physics:      {l2_name}",
        f"executor:        {backend_name}",
        "-" * 72,
        f"candidates:      {len(candidates)} plans generated",
        f"survivors:       {len(survivors)} plans passed the safety critic",
    ]

    if selection is not None:
        lines.append(f"gate verdict:    {'PASS' if selection.any_feasible else 'ALL REJECTED'}")
        lines.append(f"selected plan:   {selection.best_plan_id}")
        lines.append(f"selection logic: {selection.rationale}")
        lines.append("-" * 72)
        lines.append("per-plan physics scores:")
        for result in selection.ranked:
            marker = " <== selected" if result.plan_id == selection.best_plan_id else ""
            status = "ok" if result.success else "FAIL"
            lines.append(
                f"  {result.plan_id:<24} score={selection.scores[result.plan_id]:>12.1f}  "
                f"[{status}] {result.completion_time_s:>6.1f}s "
                f"{result.collision_count} collisions{marker}"
            )
            # failed plans must say WHY — silent failures are undiagnosable
            if result.failure is not None:
                for violation in result.failure.violations:
                    lines.append(f"      └─ {violation.type}: {violation.detail}")

    # full candidate step dump: makes any planner/critic/gate failure diagnosable
    if candidates:
        lines.append("-" * 72)
        lines.append("candidate plan steps:")
        for plan in candidates:
            lines.append(f"  {plan.plan_id}:")
            for step in plan.steps:
                args = json.dumps(step.args, ensure_ascii=False)
                lines.append(f"    {step.step_id}. {step.tool.value} {args}")

    lines.append("-" * 72)
    if execution:
        lines.append(
            f"execution:       {execution.get('steps_completed', 0)}/"
            f"{execution.get('steps_total', 0)} steps completed, "
            f"success={execution.get('success')}"
        )
    lines.append(f"trace:           {' -> '.join(final_state.get('trace', []))}")
    lines.append(f"OUTCOME:         {outcome.upper()}")
    lines.append("=" * 72)
    return "\n".join(lines)


def run_fetch_and_place(
    task: str = DEFAULT_TASK,
    scene: Scene | None = None,
    l2_fn: L2Fn = symbolic_l2,
    backend_factory: Callable[[Scene], WorldBackend] = MockWorldBackend,
    executor_fn_override: Callable[[Plan, Scene], dict[str, Any]] | None = None,
    auto_approve: bool = True,
    config: OrchestratorConfig | None = None,
) -> dict[str, Any]:
    """Run the full fetch-and-place pipeline. Returns all artifacts + report.

    Args:
        task: natural-language task.
        scene: initial scene (default: the MVP demo scene).
        l2_fn: L2 physics implementation (symbolic by default; pass the Isaac
            Lab gate from gate/l2_physics.py for real physics).
        backend_factory: builds the execution backend from the initial scene
            (MockWorldBackend by default; SimBackend for Isaac Sim execution).
        executor_fn_override: replaces the backend-based executor entirely
            (e.g. PolicySimExecutor — the robot walks the plan with the trained
            locomotion policy). When given, backend_factory is only used for
            final-scene readout.
        auto_approve: approve the gate's selection without human input (demo).
        config: orchestrator budgets (default: N=8, 2 replans, 3 retries).
    """
    initial_scene = scene or build_demo_scene()

    planner = make_planner()
    critic = make_critic()

    # The execution backend is created LAZILY, when execution actually starts.
    # Rationale: the L2 gate's parallel rollouts disturb the (shared) simulation
    # world; the executor must start from a freshly reset world, which the
    # backend's constructor performs. Creating it up front would execute against
    # post-rollout state. The last backend is kept for final-scene readout.
    backends: list[WorldBackend] = []

    def gate_fn(plans: list[Plan], gate_scene: Scene) -> SelectionResult:
        return run_gate(plans, gate_scene, l2_fn=l2_fn)

    def default_executor_fn(plan: Plan, _scene: Scene) -> dict[str, Any]:
        execution_backend = backend_factory(initial_scene)
        backends.append(execution_backend)
        return execute_plan(plan, execution_backend)

    executor_fn = executor_fn_override or default_executor_fn

    # audit trail: every run records its three-stream audit (architecture §4)
    from physgate.audit.trail import AuditTrail

    audit_trail = AuditTrail()

    if auto_approve:

        def approval_fn(selection: SelectionResult) -> bool:
            return True

        checkpointer = None
    else:
        # human-in-the-loop: pause at the approval gate (LangGraph interrupt)
        from langgraph.checkpoint.memory import MemorySaver

        from physgate.orchestrator.graph import HUMAN_APPROVAL

        approval_fn = HUMAN_APPROVAL
        checkpointer = MemorySaver()

    graph = build_orchestrator(
        planner_fn=planner,
        critic_fn=critic,
        gate_fn=gate_fn,
        executor_fn=executor_fn,
        approval_fn=approval_fn,
        config=config or OrchestratorConfig(),
        checkpointer=checkpointer,
        audit_trail=audit_trail,
    )
    final_state = run_task(graph, task=task, scene=initial_scene)

    # interactive approval: show the selection on the terminal, ask, resume
    if not auto_approve and final_state.get("__interrupt__"):
        from physgate.orchestrator.graph import resume_with_approval

        payload = final_state["__interrupt__"][0].value
        print("\n" + "=" * 60)
        print("HUMAN APPROVAL REQUIRED")
        print(f"  task:          {payload['task']}")
        print(f"  selected plan: {payload['best_plan_id']}")
        print(f"  rationale:     {payload['rationale']}")
        print("=" * 60)
        answer = input("Approve execution? [y/N] ").strip().lower()
        final_state = resume_with_approval(graph, approved=answer == "y")

    # final world state comes from the backend that actually executed
    if executor_fn_override is not None:
        executor_name = type(executor_fn_override).__name__
        final_scene = initial_scene  # physical truth lives in the sim / execution result
    else:
        final_backend = backends[-1] if backends else backend_factory(initial_scene)
        executor_name = type(final_backend).__name__
        final_scene = final_backend.get_scene()

    report = _format_report(
        task=task,
        final_state=final_state,
        planner_name=type(planner).__name__,
        critic_name=type(critic).__name__,
        l2_name=getattr(l2_fn, "__name__", type(l2_fn).__name__),
        backend_name=executor_name,
    )

    # seal the audit trail (final Merkle checkpoint)
    audit_root = audit_trail.close()

    return {
        **final_state,
        "final_scene": final_scene,
        "report": report,
        "audit_trail": audit_trail,
        "audit_merkle_root": audit_root,
    }
