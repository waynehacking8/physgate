"""E1 pipeline ablation: quantify what each validation layer adds end to end.

Conditions (docs/design/EVALUATION_METHODOLOGY.md §4 E1):

    A0_no_validation   execute a plan drawn from the pool, no checks
    A1_critic_only     critic filters the pool, execute a surviving plan
    A2_symbolic_gate   critic + L1/L3 + symbolic L2 select a plan, execute it
    A3_nav_aware_gate  A2 + navigation compilation + task-goal verification
                       (the symbolic surrogate of the Isaac physics gate, which
                       verifies both world-level reachability and the physical
                       task outcome)

Execution model: symbolic execution (MockWorldBackend) + navigation compilation
against the instance's layout. "Success" on a feasible instance means the task
goal relation holds after execution AND the navigation could actually be
compiled for that layout. On infeasible instances we measure whether the
condition attempts execution at all (false execution) vs rejecting beforehand.

A0/A1 average over every plan they could draw (the exact expectation of a
uniform random draw) instead of sampling, so results are deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from physgate.eval_v2.defects import build_defect_corpus
from physgate.eval_v2.scenario_gen import TaskInstance
from physgate.examples_lib.fetch_and_place import build_demo_scene
from physgate.executor.backend import MockWorldBackend
from physgate.executor.plan_executor import execute_plan
from physgate.gate.parallel import run_gate, symbolic_l2
from physgate.gate.trajectory import compile_mission
from physgate.nav.path_planner import PathPlannerError
from physgate.planner.critic import MockCritic
from physgate.planner.planner import MockPlanner
from physgate.planner.schemas import Plan

TASK = "put the fallen box back on shelf A"
GOAL_RELATION = ("box_03", "on", "shelf_A")

CONDITIONS = [
    "A0_no_validation",
    "A1_critic_only",
    "A2_symbolic_gate",
    "A3_nav_aware_gate",
]


@dataclass(frozen=True)
class PooledPlan:
    """A plan in the contaminated pool with its provenance label."""

    plan: Plan
    defect_id: str
    ground_truth_invalid: bool


@dataclass(frozen=True)
class ConditionOutcome:
    """Aggregate outcome of one pipeline condition over a set of instances."""

    condition: str
    n_instances: int
    #: feasible instances: fraction where the goal relation held after execution
    #: (instance-level: each instance's expected success is one observation)
    success_rate: float
    success_ci: tuple[float, float]
    #: plan-level breakdown (for transparency — NOT used for CI)
    n_plan_trials: int = 0
    plan_level_success_rate: float = 0.0
    #: infeasible instances: fraction where execution was attempted at all
    false_execution_rate: float = 0.0
    #: infeasible instances: fraction correctly rejected before execution
    rejection_rate: float = 0.0
    details: list[dict] = field(default_factory=list)


# ----------------------------------------------------------------- utilities


def wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a binomial rate."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def instance_ci(rates: list[float], z: float = 1.96) -> tuple[float, float]:
    """95% CI for the mean of instance-level success rates.

    Uses Wilson CI when all rates are binary (0 or 1), otherwise a
    normal-approximation CI on the mean.
    """
    n = len(rates)
    if n == 0:
        return (0.0, 1.0)
    binary = all(r in (0.0, 1.0) for r in rates)
    if binary:
        return wilson_ci(sum(rates), n, z)
    mean = sum(rates) / n
    if n == 1:
        return (0.0, 1.0)
    var = sum((r - mean) ** 2 for r in rates) / (n - 1)
    se = math.sqrt(var / n)
    lo = max(0.0, mean - z * se)
    hi = min(1.0, mean + z * se)
    return (round(lo, 4), round(hi, 4))


def build_plan_pool(seed: int = 0) -> list[PooledPlan]:
    """The contaminated plan pool: clean critic-surviving plans + one variant per
    PLAN-LEVEL defect class — modeling a planner that produces both good and bad
    plans. D5 (world-level defect) is excluded because its defect is in the
    layout, not the plan — it is evaluated in E2 (gate classifier), not E1."""
    scene = build_demo_scene()
    base_plans = MockCritic()(MockPlanner()(TASK, scene, 8, None), scene)
    corpus = build_defect_corpus(base_plans[:3])
    return [
        PooledPlan(
            plan=item.plan,
            defect_id=item.defect_id,
            ground_truth_invalid=item.ground_truth_invalid,
        )
        for item in corpus
        if item.layout_override is None  # exclude world-level defects (D5)
    ]


# ----------------------------------------------------------- execution model


def _navigation_compiles(plan: Plan, layout: dict) -> bool:
    """Can the deterministic navigation layer actually route this plan in this
    world? (the world-level feasibility check the physics gate performs)"""
    try:
        compile_mission(plan, layout)
        return True
    except PathPlannerError:
        return False
    except (ValueError, KeyError):
        return False


def _execute_symbolically(plan: Plan, instance: TaskInstance) -> bool:
    """Execute on the symbolic backend; success = task goal relation holds."""
    backend = MockWorldBackend(instance.scene)
    outcome = execute_plan(plan, backend)
    if not outcome.get("success"):
        return False
    return backend.get_scene().has_relation(*GOAL_RELATION)


def _instance_execution(plan: Plan, instance: TaskInstance) -> bool:
    """Full execution model: navigation must compile for THIS layout AND the
    symbolic execution must achieve the goal."""
    if not _navigation_compiles(plan, instance.layout):
        return False
    return _execute_symbolically(plan, instance)


# ----------------------------------------------------------------- conditions


def _select_plans_a0(pool: list[PooledPlan], instance: TaskInstance) -> list[Plan]:
    """A0: no validation — any plan in the pool could be drawn."""
    return [p.plan for p in pool]


def _select_plans_a1(pool: list[PooledPlan], instance: TaskInstance) -> list[Plan]:
    """A1: critic filters the pool; any survivor could be drawn."""
    survivors = MockCritic()([p.plan for p in pool], instance.scene)
    return survivors if survivors else []


def _select_plans_a2(pool: list[PooledPlan], instance: TaskInstance) -> list[Plan]:
    """A2: critic + symbolic gate select the single best plan."""
    survivors = MockCritic()([p.plan for p in pool], instance.scene)
    if not survivors:
        return []
    selection = run_gate(survivors, instance.scene, l2_fn=symbolic_l2)
    if not selection.any_feasible or selection.best_plan_id is None:
        return []
    return [next(p for p in survivors if p.plan_id == selection.best_plan_id)]


def _select_plans_a3(pool: list[PooledPlan], instance: TaskInstance) -> list[Plan]:
    """A3: A2 + navigation compilation + task-goal verification — the symbolic
    surrogate of the Isaac physics gate. Plans whose navigation cannot be
    compiled for THIS world or that do not achieve the goal are rejected."""
    survivors = MockCritic()([p.plan for p in pool], instance.scene)
    if not survivors:
        return []
    selection = run_gate(survivors, instance.scene, l2_fn=symbolic_l2)
    # re-rank: walk the gate's ranking, accept the first plan that passes the
    # physics-gate-equivalent checks (navigation + goal)
    ranked_ids = [r.plan_id for r in selection.ranked if r.success]
    for plan_id in ranked_ids:
        plan = next(p for p in survivors if p.plan_id == plan_id)
        if _navigation_compiles(plan, instance.layout) and _execute_symbolically(plan, instance):
            return [plan]
    return []  # nothing passes -> reject (escalate)


_SELECTORS = {
    "A0_no_validation": _select_plans_a0,
    "A1_critic_only": _select_plans_a1,
    "A2_symbolic_gate": _select_plans_a2,
    "A3_nav_aware_gate": _select_plans_a3,
}


# -------------------------------------------------------------------- runner


def run_condition(
    condition: str, instances: list[TaskInstance], pool: list[PooledPlan]
) -> ConditionOutcome:
    """Run one pipeline condition over the instances."""
    selector = _SELECTORS[condition]

    # instance-level rates (the independent observations for CI computation)
    instance_rates: list[float] = []
    # plan-level outcomes (for transparency, NOT for CI)
    all_plan_outcomes: list[bool] = []
    infeasible_attempted: list[bool] = []
    details: list[dict] = []

    for instance in instances:
        selected = selector(pool, instance)

        if instance.feasible:
            if not selected:
                instance_rates.append(0.0)
                details.append(
                    {"instance": instance.instance_id, "selected": None, "success": False}
                )
                continue
            successes = [_instance_execution(plan, instance) for plan in selected]
            rate = sum(successes) / len(successes)
            instance_rates.append(rate)
            all_plan_outcomes.extend(successes)
            details.append(
                {
                    "instance": instance.instance_id,
                    "selected": [p.plan_id for p in selected],
                    "success": round(rate, 4),
                    "n_plans_tested": len(successes),
                }
            )
        else:
            attempted = bool(selected)
            infeasible_attempted.append(attempted)
            details.append(
                {
                    "instance": instance.instance_id,
                    "selected": [p.plan_id for p in selected] if selected else None,
                    "false_execution": attempted,
                }
            )

    # instance-level aggregation (correct for CI computation: n = n_instances)
    n_inst = len(instance_rates)
    success_rate = sum(instance_rates) / n_inst if n_inst else 0.0

    # plan-level breakdown (strictly for transparency)
    n_plans = len(all_plan_outcomes)
    plan_success = sum(all_plan_outcomes) / n_plans if n_plans else 0.0

    n_infeasible = len(infeasible_attempted)
    false_exec = sum(infeasible_attempted) / n_infeasible if n_infeasible else 0.0

    return ConditionOutcome(
        condition=condition,
        n_instances=len(instances),
        success_rate=round(success_rate, 4),
        success_ci=instance_ci(instance_rates),
        n_plan_trials=n_plans,
        plan_level_success_rate=round(plan_success, 4),
        false_execution_rate=round(false_exec, 4),
        rejection_rate=round(1.0 - false_exec, 4) if n_infeasible else 0.0,
        details=details,
    )


def run_ablation(
    instances: list[TaskInstance], pool: list[PooledPlan]
) -> dict[str, ConditionOutcome]:
    """Run all four conditions over the instance set."""
    return {condition: run_condition(condition, instances, pool) for condition in CONDITIONS}
