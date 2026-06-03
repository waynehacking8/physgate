"""Tests for the agent-orchestrator evaluation suite (REBUILD.md Phase 3).

The evaluation measures ORCHESTRATION quality — decomposition, ordering,
preconditions, failure recovery, infeasibility recognition — not navigation
geometry (the deterministic nav layer owns that). These tests verify the suite
itself works correctly using the deterministic MockPlanner/MockCritic, so the
benchmark numbers it produces for real LLM planners are trustworthy.
"""

from __future__ import annotations

from physgate.eval.metrics import check_decomposition, compute_metrics
from physgate.eval.runner import run_scenario, run_suite
from physgate.eval.scenarios import build_scenario_suite
from physgate.planner.critic import MockCritic
from physgate.planner.planner import MockPlanner
from physgate.planner.schemas import Plan, PlanStep, ToolName

# ------------------------------------------------------------------ scenarios


def test_suite_covers_all_orchestration_categories():
    suite = build_scenario_suite()
    categories = {s.category for s in suite}
    assert categories == {"ordering", "preconditions", "recovery", "multi_step", "infeasible"}
    # at least one feasible and one infeasible scenario
    assert any(s.expected_outcome == "done" for s in suite)
    assert any(s.expected_outcome == "escalated" for s in suite)


def test_scenario_ids_unique():
    suite = build_scenario_suite()
    ids = [s.scenario_id for s in suite]
    assert len(ids) == len(set(ids))


# ------------------------------------------------------- decomposition checker


def _step(step_id, tool, **args):
    return PlanStep(step_id=step_id, tool=tool, args=args, preconditions=["x exists"])


def test_decomposition_valid_for_correct_order():
    plan = Plan(
        plan_id="ok",
        task="t",
        steps=[
            _step(1, ToolName.MOVE_TO_POSE, target="box_03"),
            _step(2, ToolName.EXECUTE_SKILL, skill="pick", target="box_03"),
            _step(3, ToolName.MOVE_TO_POSE, target="shelf_A"),
            _step(4, ToolName.EXECUTE_SKILL, skill="place", target="shelf_A"),
        ],
    )
    assert check_decomposition(plan) is True


def test_decomposition_invalid_place_before_pick():
    plan = Plan(
        plan_id="inverted",
        task="t",
        steps=[
            _step(1, ToolName.MOVE_TO_POSE, target="shelf_A"),
            _step(2, ToolName.EXECUTE_SKILL, skill="place", target="shelf_A"),
            _step(3, ToolName.MOVE_TO_POSE, target="box_03"),
            _step(4, ToolName.EXECUTE_SKILL, skill="pick", target="box_03"),
        ],
    )
    assert check_decomposition(plan) is False


def test_decomposition_invalid_pick_without_approach():
    plan = Plan(
        plan_id="teleport",
        task="t",
        steps=[
            _step(1, ToolName.EXECUTE_SKILL, skill="pick", target="box_03"),
            _step(2, ToolName.MOVE_TO_POSE, target="shelf_A"),
            _step(3, ToolName.EXECUTE_SKILL, skill="place", target="shelf_A"),
        ],
    )
    assert check_decomposition(plan) is False


# ------------------------------------------------------------ single scenarios


def _mock_components():
    return MockPlanner(), MockCritic()


def test_basic_scenario_completes_with_mock_planner():
    suite = {s.scenario_id: s for s in build_scenario_suite()}
    planner, critic = _mock_components()
    result = run_scenario(suite["fetch_and_place_basic"], planner, critic)
    assert result.actual_outcome == "done"
    assert result.task_completed is True
    assert result.outcome_correct is True


def test_gate_catches_inverted_probe_plan():
    """The gate (critic + L1/L3/L2) must reject a place-before-pick plan and
    select the correct one — fraction-of-invalid-plans-caught metric."""
    suite = {s.scenario_id: s for s in build_scenario_suite()}
    planner, critic = _mock_components()
    result = run_scenario(suite["gate_catches_place_before_pick"], planner, critic)
    assert result.invalid_probe_caught is True
    assert result.actual_outcome == "done"


def test_recovery_scenario_retries_through_transient_fault():
    """A transient pick failure must be recovered by the orchestrator's retry
    loop — outcome done, retries > 0."""
    suite = {s.scenario_id: s for s in build_scenario_suite()}
    planner, critic = _mock_components()
    result = run_scenario(suite["recovery_transient_pick_failure"], planner, critic)
    assert result.actual_outcome == "done"
    assert result.retries_used >= 1
    assert result.outcome_correct is True


def test_persistent_fault_escalates():
    """A persistent failure must end in escalation (recognized, not looped forever)."""
    suite = {s.scenario_id: s for s in build_scenario_suite()}
    planner, critic = _mock_components()
    result = run_scenario(suite["recovery_persistent_failure_escalates"], planner, critic)
    assert result.actual_outcome == "escalated"
    assert result.outcome_correct is True


def test_infeasible_task_escalates_with_mock_planner():
    """An ungraspable object cannot be fetched: the orchestrator must escalate
    (recognize infeasibility), never claim success."""
    suite = {s.scenario_id: s for s in build_scenario_suite()}
    planner, critic = _mock_components()
    result = run_scenario(suite["infeasible_ungraspable_object"], planner, critic)
    assert result.actual_outcome == "escalated"
    assert result.outcome_correct is True


def test_multi_step_scenario_detects_incomplete_decomposition():
    """MockPlanner only handles one box — the two-box scenario must be scored as
    NOT completed (this is exactly the decomposition-quality signal the eval
    exists to measure; a real LLM orchestrator should do better)."""
    suite = {s.scenario_id: s for s in build_scenario_suite()}
    planner, critic = _mock_components()
    result = run_scenario(suite["multi_step_two_boxes"], planner, critic)
    assert result.task_completed is False


# ------------------------------------------------------------------ full suite


def test_full_suite_with_mock_planner_produces_report():
    planner, critic = _mock_components()
    report = run_suite(build_scenario_suite(), planner, critic, planner_name="mock")

    assert report.planner_name == "mock"
    assert len(report.results) == len(build_scenario_suite())
    metrics = report.metrics
    # all metric keys present and in [0, 1]
    for key in (
        "end_to_end_success_rate",
        "infeasible_recognition_rate",
        "recovery_rate",
        "decomposition_validity_rate",
        "invalid_plan_catch_rate",
        "orchestrator_score",
    ):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    # the deterministic mock orchestrator's known profile:
    # - recognizes infeasible tasks (critic rejects ungraspable picks)
    assert metrics["infeasible_recognition_rate"] == 1.0
    # - recovers from transient faults (retry loop)
    assert metrics["recovery_rate"] == 1.0
    # - the gate catches invalid probe plans
    assert metrics["invalid_plan_catch_rate"] == 1.0
    # - but FAILS multi-step decomposition (single-box planner)
    assert metrics["end_to_end_success_rate"] < 1.0


def test_metrics_handle_empty_results():
    metrics = compute_metrics([])
    assert metrics["orchestrator_score"] == 0.0


def test_report_is_deterministic():
    planner, critic = _mock_components()
    a = run_suite(build_scenario_suite(), planner, critic, planner_name="mock")
    b = run_suite(build_scenario_suite(), planner, critic, planner_name="mock")
    assert a.metrics == b.metrics
    assert [r.actual_outcome for r in a.results] == [r.actual_outcome for r in b.results]
