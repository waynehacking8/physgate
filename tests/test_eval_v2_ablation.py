"""Tests for E1 pipeline ablation: the value-of-the-gate experiment.

Four pipeline conditions over the same plan pool and task instances:

    A0 no-validation   execute a randomly drawn plan
    A1 critic-only     critic filters, execute a surviving plan
    A2 symbolic-gate   critic + L1/L3 + symbolic L2 select, then execute
    A3 nav-aware gate  A2 + navigation compilation check (world-level feasibility)

Measured on feasible instances: task success rate.
Measured on infeasible instances: false-execution rate (attempting a task that
is impossible) vs pre-execution rejection.
"""

from __future__ import annotations

import math

import pytest

from physgate.eval_v2.ablation import instance_ci, wilson_ci
from physgate.eval_v2.scenario_gen import generate_instances

TASK = "put the fallen box back on shelf A"


# --------------------------------------------------------- wilson_ci tests


@pytest.mark.parametrize(
    "successes, n, expected_lo, expected_hi",
    [
        (0, 10, 0.0, 0.2775),
        (10, 10, 0.7225, 1.0),
        (5, 10, 0.2366, 0.7634),
        (50, 100, 0.4038, 0.5962),
    ],
    ids=["all_fail", "all_pass", "half", "large_n"],
)
def test_wilson_ci_known_values(successes, n, expected_lo, expected_hi):
    lo, hi = wilson_ci(successes, n)
    assert lo == pytest.approx(expected_lo, abs=0.001)
    assert hi == pytest.approx(expected_hi, abs=0.001)


def test_wilson_ci_empty_sample():
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(3, 20)
    assert 0.0 <= lo <= hi <= 1.0


# --------------------------------------------------------- instance_ci tests


@pytest.mark.parametrize(
    "rates, check",
    [
        ([1.0, 1.0, 1.0, 1.0, 1.0], lambda lo, hi: lo > 0.5 and hi == 1.0),
        ([0.0, 0.0, 0.0, 0.0, 0.0], lambda lo, hi: lo == 0.0 and hi < 0.5),
        ([1.0, 0.0, 1.0, 0.0], lambda lo, hi: 0.0 < lo < 0.5 < hi < 1.0),
        ([0.8, 0.9, 0.7, 0.85], lambda lo, hi: 0.5 < lo < hi < 1.0),
    ],
    ids=["all_pass", "all_fail", "mixed_binary", "fractional"],
)
def test_instance_ci_parametrized(rates, check):
    lo, hi = instance_ci(rates)
    assert 0.0 <= lo <= hi <= 1.0
    assert check(lo, hi)


def test_instance_ci_empty():
    assert instance_ci([]) == (0.0, 1.0)


def test_instance_ci_single():
    assert instance_ci([0.75]) == (0.0, 1.0)


def test_instance_ci_binary_delegates_to_wilson():
    rates = [1.0, 1.0, 0.0, 0.0, 1.0]
    assert instance_ci(rates) == wilson_ci(3, 5)


def test_instance_ci_normal_approx_for_fractional():
    rates = [0.5, 0.6, 0.7, 0.8]
    lo, hi = instance_ci(rates)
    mean = sum(rates) / len(rates)
    assert lo < mean < hi


@pytest.fixture(scope="module")
def instances():
    return generate_instances(n_feasible=5, n_infeasible=3, seed=123)


@pytest.fixture(scope="module")
def plan_pool():
    """The contaminated plan pool: clean plans + defect-injected variants,
    modeling the reality that LLM planners produce both."""
    from physgate.eval_v2.ablation import build_plan_pool

    return build_plan_pool(seed=123)


def test_plan_pool_contains_clean_and_defective_plans(plan_pool):
    clean = [p for p in plan_pool if p.defect_id == "D0_clean"]
    defective = [p for p in plan_pool if p.defect_id != "D0_clean"]
    assert clean and defective


def test_condition_a0_executes_without_any_validation(instances, plan_pool):
    """A0 executes whatever it draws — success rate equals the pool's clean-plan
    fraction on feasible instances (no validation to save it)."""
    from physgate.eval_v2.ablation import run_condition

    feasible = [i for i in instances if i.feasible]
    outcome = run_condition("A0_no_validation", feasible, plan_pool)

    assert outcome.condition == "A0_no_validation"
    assert outcome.n_instances == len(feasible)
    # with defective plans in the pool, A0 must NOT reach 100%
    assert outcome.success_rate < 1.0


def test_condition_a2_outperforms_a0_on_feasible_instances(instances, plan_pool):
    """The gate's value on plan-level defects: A2 selects a validated plan, so its
    success rate must exceed A0's."""
    from physgate.eval_v2.ablation import run_condition

    feasible = [i for i in instances if i.feasible]
    a0 = run_condition("A0_no_validation", feasible, plan_pool)
    a2 = run_condition("A2_symbolic_gate", feasible, plan_pool)

    assert a2.success_rate > a0.success_rate
    assert a2.success_rate == 1.0, "the symbolic gate must always select a working plan"


def test_condition_a3_rejects_infeasible_instances_before_execution(instances, plan_pool):
    """World-level infeasibility (unreachable box): ONLY the navigation-aware
    condition can reject before execution; symbolic conditions cannot see it."""
    from physgate.eval_v2.ablation import run_condition

    infeasible = [i for i in instances if not i.feasible]

    a2 = run_condition("A2_symbolic_gate", infeasible, plan_pool)
    a3 = run_condition("A3_nav_aware_gate", infeasible, plan_pool)

    # the symbolic gate executes impossible tasks (it has no geometry)
    assert a2.false_execution_rate == 1.0
    # the navigation-aware gate rejects every one of them before execution
    assert a3.false_execution_rate == 0.0
    assert a3.rejection_rate == 1.0


def test_ablation_report_structure(instances, plan_pool):
    from physgate.eval_v2.ablation import CONDITIONS, run_ablation

    report = run_ablation(instances, plan_pool)
    assert set(report.keys()) == set(CONDITIONS)
    for outcome in report.values():
        assert 0.0 <= outcome.success_rate <= 1.0
        assert 0.0 <= outcome.false_execution_rate <= 1.0
        # Wilson confidence intervals present
        assert outcome.success_ci[0] <= outcome.success_rate <= outcome.success_ci[1]
