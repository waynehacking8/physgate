#!/usr/bin/env python3
"""A5 ablation: targeted repair vs blind regeneration.

Runs the orchestration eval suite under two conditions:
- blind: no failure analyst (the original replan-from-scratch behavior)
- targeted: with MockFailureAnalyst (structured diagnosis → targeted repair)

Compares success rates, replan counts, and recovery efficiency.

Usage:
    .venv/bin/python benchmarks/eval_v2/run_ablation_a5.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from physgate.eval.fault_backend import FaultInjector
from physgate.eval.metrics import compute_metrics
from physgate.eval.runner import run_scenario
from physgate.eval.scenarios import build_scenario_suite
from physgate.executor.backend import MockWorldBackend
from physgate.executor.plan_executor import execute_plan
from physgate.gate.parallel import run_gate, symbolic_l2
from physgate.orchestrator.graph import OrchestratorConfig, build_orchestrator, run_task
from physgate.planner.critic import MockCritic
from physgate.planner.failure_analyst import MockFailureAnalyst
from physgate.planner.planner import MockPlanner


def run_condition(condition: str, suite, planner, critic, config):
    """Run all scenarios under one ablation condition."""
    analyst = MockFailureAnalyst() if condition == "targeted" else None
    results = []

    for scenario in suite:
        result = run_scenario(scenario, planner, critic, config=config)
        results.append(result)

    return results


def run_condition_with_analyst(suite, planner, critic, config, use_analyst: bool):
    """Run scenarios with or without failure analyst via full orchestrator."""
    analyst = MockFailureAnalyst() if use_analyst else None
    results = []

    for scenario in suite:
        injector = FaultInjector(scenario.fault) if scenario.fault else None

        def executor_fn(plan, scene, _inj=injector, _sc=scenario):
            factory = _inj.backend_factory if _inj else MockWorldBackend
            backend = factory(_sc.scene)
            return execute_plan(plan, backend)

        def gate_fn(plans, scene):
            return run_gate(plans, scene, l2_fn=symbolic_l2)

        if scenario.probe_plans:
            def effective_planner(task, scene, n, feedback, _probes=scenario.probe_plans):
                return list(_probes)
        else:
            effective_planner = planner

        graph = build_orchestrator(
            planner_fn=effective_planner,
            critic_fn=critic,
            gate_fn=gate_fn,
            executor_fn=executor_fn,
            approval_fn=lambda sel: True,
            config=config,
            failure_analyst_fn=analyst,
        )
        state = run_task(graph, task=scenario.task, scene=scenario.scene)

        from physgate.eval.metrics import ScenarioResult
        actual = state.get("outcome", "?")
        results.append(ScenarioResult(
            scenario_id=scenario.scenario_id,
            category=scenario.category,
            expected_outcome=scenario.expected_outcome,
            actual_outcome=actual,
            outcome_correct=(actual == scenario.expected_outcome)
                if scenario.expected_outcome != "done"
                else actual == "done",
            task_completed=actual == "done",
            replans_used=state.get("replan_count", 0),
            retries_used=state.get("retry_count", 0),
        ))

    return results


def main():
    suite = build_scenario_suite()
    planner = MockPlanner()
    critic = MockCritic()
    config = OrchestratorConfig(max_replans=3, max_execution_retries=1)

    print("=" * 60)
    print("A5 Ablation: Targeted Repair vs Blind Regeneration")
    print("=" * 60)

    t0 = time.perf_counter()
    blind_results = run_condition_with_analyst(suite, planner, critic, config, use_analyst=False)
    t_blind = time.perf_counter() - t0

    t0 = time.perf_counter()
    targeted_results = run_condition_with_analyst(suite, planner, critic, config, use_analyst=True)
    t_targeted = time.perf_counter() - t0

    blind_metrics = compute_metrics(blind_results)
    targeted_metrics = compute_metrics(targeted_results)

    print(f"\n{'Metric':<40} {'Blind':>10} {'Targeted':>10} {'Delta':>10}")
    print("-" * 70)
    all_keys = sorted(set(blind_metrics) | set(targeted_metrics))
    for key in all_keys:
        bv = blind_metrics.get(key, 0.0)
        tv = targeted_metrics.get(key, 0.0)
        delta = tv - bv
        sign = "+" if delta > 0 else ""
        print(f"  {key:<38} {bv:>10.3f} {tv:>10.3f} {sign}{delta:>9.3f}")

    print(f"\nWall time: blind={t_blind:.1f}s, targeted={t_targeted:.1f}s")

    # per-scenario comparison
    print(f"\n{'Scenario':<45} {'Blind':>8} {'Targeted':>10}")
    print("-" * 65)
    for b, t in zip(blind_results, targeted_results):
        print(f"  {b.scenario_id:<43} {b.actual_outcome:>8} {t.actual_outcome:>10}")

    # save results
    out_dir = REPO / "benchmarks" / "eval_v2" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "condition_blind": {
            "metrics": blind_metrics,
            "per_scenario": [
                {"id": r.scenario_id, "outcome": r.actual_outcome, "replans": r.replans_used}
                for r in blind_results
            ],
        },
        "condition_targeted": {
            "metrics": targeted_metrics,
            "per_scenario": [
                {"id": r.scenario_id, "outcome": r.actual_outcome, "replans": r.replans_used}
                for r in targeted_results
            ],
        },
    }
    result_file = out_dir / "a5_ablation.json"
    result_file.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved: {result_file}")


if __name__ == "__main__":
    main()
