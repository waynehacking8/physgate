"""Orchestration-evaluation runner: scenarios -> orchestrator -> scored results.

Builds the REAL orchestrator graph (LangGraph state machine) for every scenario
with injected components, runs it, and scores the outcome against the
scenario's expectations. No GPU: the gate uses symbolic L2 (plan-logic
validation); physical-outcome validation of winning plans is covered by the
Isaac integration tests and demo.
"""

from __future__ import annotations

import time

from physgate.eval.fault_backend import FaultInjector
from physgate.eval.metrics import (
    OrchestratorReport,
    ScenarioResult,
    check_decomposition,
    compute_metrics,
)
from physgate.eval.scenarios import OrchestrationScenario
from physgate.executor.backend import MockWorldBackend, WorldBackend
from physgate.executor.plan_executor import execute_plan
from physgate.gate.parallel import run_gate, symbolic_l2
from physgate.gate.schemas import Scene
from physgate.gate.scoring import SelectionResult
from physgate.orchestrator.graph import OrchestratorConfig, build_orchestrator, run_task
from physgate.planner.schemas import Plan


def run_scenario(
    scenario: OrchestrationScenario,
    planner_fn,
    critic_fn,
    config: OrchestratorConfig | None = None,
) -> ScenarioResult:
    """Run one scenario through the orchestrator and score it.

    Args:
        scenario: the scenario definition (scene, task, expectations).
        planner_fn: ``(task, scene, n, feedback) -> list[Plan]`` — the planner
            under evaluation (mock or real LLM). Ignored for probe scenarios.
        critic_fn: ``(plans, scene) -> list[Plan]`` — the safety critic.
        config: orchestrator budgets (defaults: N=8, 2 replans, 3 retries).
    """
    t0 = time.perf_counter()

    # ---- planner: probe plans replace the planner for gate-probing scenarios ----
    if scenario.probe_plans:

        def effective_planner(task: str, scene: Scene, n: int, feedback: str | None) -> list[Plan]:
            """Return the scenario's pre-defined probe plans instead of calling the planner."""
            return list(scenario.probe_plans)

    else:
        effective_planner = planner_fn

    # ---- execution backend (with fault injection when the scenario asks) ----
    injector = FaultInjector(scenario.fault) if scenario.fault else None
    backends: list[WorldBackend] = []

    def executor_fn(plan: Plan, scene: Scene) -> dict:
        """Execute a plan against a fresh backend, optionally with fault injection."""
        factory = injector.backend_factory if injector else MockWorldBackend
        backend = factory(scenario.scene)
        backends.append(backend)
        return execute_plan(plan, backend)

    def gate_fn(plans: list[Plan], scene: Scene) -> SelectionResult:
        """Run the symbolic L2 gate over candidate plans."""
        return run_gate(plans, scene, l2_fn=symbolic_l2)

    graph = build_orchestrator(
        planner_fn=effective_planner,
        critic_fn=critic_fn,
        gate_fn=gate_fn,
        executor_fn=executor_fn,
        approval_fn=lambda selection: True,
        config=config or OrchestratorConfig(),
    )
    final_state = run_task(graph, task=scenario.task, scene=scenario.scene)
    wall_s = time.perf_counter() - t0

    # ---- score the outcome ----
    actual_outcome = final_state.get("outcome", "?")
    final_scene = backends[-1].get_scene() if backends else scenario.scene
    relations_hold = all(
        final_scene.has_relation(*relation) for relation in scenario.required_final_relations
    )
    task_completed = actual_outcome == "done" and relations_hold

    if scenario.expected_outcome == "done":
        outcome_correct = task_completed
    else:  # escalated expected: partial_success (handoff) also counts as correct
        outcome_correct = actual_outcome in (scenario.expected_outcome, "partial_success")

    # decomposition check on the selected (executed) plan. The check must start
    # from the scenario's ACTUAL initial gripper state: assuming an empty gripper
    # scores correct "set the held object down first" plans as invalid — an
    # eval-interface bug masquerading as a planner deficiency (D-019).
    initially_held = next(
        (rel[2] for rel in scenario.scene.relations if rel[0] == "gripper" and rel[1] == "holding"),
        None,
    )
    selection: SelectionResult | None = final_state.get("selection")
    decomposition_valid: bool | None = None
    if selection is not None and selection.best_plan_id is not None:
        selected_plan = next(
            (p for p in final_state.get("survivors", []) if p.plan_id == selection.best_plan_id),
            None,
        )
        if selected_plan is not None:
            decomposition_valid = check_decomposition(selected_plan, initially_held=initially_held)

    # probe scenarios: was the known-invalid plan caught (not selected, marked infeasible)?
    invalid_probe_caught: bool | None = None
    if scenario.probe_plans and selection is not None:
        invalid_ids = {
            p.plan_id
            for p in scenario.probe_plans
            if not check_decomposition(p, initially_held=initially_held)
        }
        selected_ok = selection.best_plan_id not in invalid_ids
        invalid_marked_infeasible = all(
            not result.success for result in selection.ranked if result.plan_id in invalid_ids
        )
        invalid_probe_caught = selected_ok and invalid_marked_infeasible

    handoff_correct: bool | None = None
    if scenario.category in ("assistance",):
        handoff_correct = actual_outcome in ("escalated", "partial_success")
    elif scenario.expected_outcome == "done":
        handoff_correct = actual_outcome not in ("partial_success",)

    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        expected_outcome=scenario.expected_outcome,
        actual_outcome=actual_outcome,
        outcome_correct=outcome_correct,
        task_completed=task_completed,
        decomposition_valid=decomposition_valid,
        invalid_probe_caught=invalid_probe_caught,
        handoff_correct=handoff_correct,
        replans_used=final_state.get("replan_count", 0),
        retries_used=final_state.get("retry_count", 0),
        candidates_generated=len(final_state.get("candidates", [])),
        survivors_after_critic=len(final_state.get("survivors", [])),
        wall_s=round(wall_s, 2),
        detail=(
            "; ".join(
                f"{relation}: {'ok' if final_scene.has_relation(*relation) else 'MISSING'}"
                for relation in scenario.required_final_relations
            )
            if scenario.required_final_relations
            else ""
        ),
    )


def run_suite(
    scenarios: list[OrchestrationScenario],
    planner_fn,
    critic_fn,
    planner_name: str,
    config: OrchestratorConfig | None = None,
) -> OrchestratorReport:
    """Run the full scenario suite and aggregate orchestrator metrics."""
    results = [
        run_scenario(scenario, planner_fn, critic_fn, config=config) for scenario in scenarios
    ]
    return OrchestratorReport(
        planner_name=planner_name,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        results=results,
        metrics=compute_metrics(results),
    )
