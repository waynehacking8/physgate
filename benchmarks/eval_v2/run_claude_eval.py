#!/usr/bin/env python3
"""Run all 18 orchestration scenarios with real ClaudePlanner.

This is the critical evidence: real LLM tool selection across all 6 task types.
Also runs A5 ablation (targeted vs blind) with the real planner.

Usage:
    CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... .venv/bin/python benchmarks/eval_v2/run_claude_eval.py

Output: benchmarks/eval_v2/results/claude_18_scenarios.json
        benchmarks/eval_v2/results/a5_ablation_claude.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from physgate.eval.fault_backend import FaultInjector
from physgate.eval.metrics import ScenarioResult, compute_metrics
from physgate.eval.scenarios import OrchestrationScenario, build_scenario_suite
from physgate.executor.backend import MockWorldBackend
from physgate.executor.plan_executor import execute_plan
from physgate.gate.parallel import run_gate, symbolic_l2
from physgate.gate.scoring import SelectionResult
from physgate.orchestrator.graph import OrchestratorConfig, build_orchestrator, run_task
from physgate.planner.critic import make_critic
from physgate.planner.failure_analyst import MockFailureAnalyst
from physgate.planner.planner import make_planner
from physgate.planner.schemas import Plan


def run_one_scenario(
    scenario: OrchestrationScenario,
    planner,
    critic,
    config: OrchestratorConfig,
    failure_analyst=None,
) -> dict:
    """Run one scenario through the orchestrator, return state + timing."""
    injector = FaultInjector(scenario.fault) if scenario.fault else None

    def executor_fn(plan: Plan, scene):
        factory = injector.backend_factory if injector else MockWorldBackend
        backend = factory(scenario.scene)
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
        failure_analyst_fn=failure_analyst,
    )

    t0 = time.perf_counter()
    state = run_task(graph, task=scenario.task, scene=scenario.scene)
    wall_s = time.perf_counter() - t0

    return {
        "scenario_id": scenario.scenario_id,
        "category": scenario.category,
        "expected_outcome": scenario.expected_outcome,
        "actual_outcome": state.get("outcome", "?"),
        "replans_used": state.get("replan_count", 0),
        "retries_used": state.get("retry_count", 0),
        "candidates": len(state.get("candidates", [])),
        "survivors": len(state.get("survivors", [])),
        "wall_s": round(wall_s, 2),
        "trace": state.get("trace", []),
        "has_failure_report": state.get("failure_report") is not None,
        "replan_strategy": state.get("replan_strategy", ""),
    }


def main():
    suite = build_scenario_suite()
    planner = make_planner()
    critic = make_critic()
    config = OrchestratorConfig(max_replans=3, max_execution_retries=1)
    analyst = MockFailureAnalyst()

    planner_name = type(planner).__name__
    print(f"Planner: {planner_name}")
    print(f"Scenarios: {len(suite)}")
    print("=" * 70)

    # --- Phase 1: All 18 scenarios with real Claude (no analyst) ---
    print("\n=== Phase 1: 18 scenarios WITHOUT failure analyst ===\n")
    blind_results = []
    for i, scenario in enumerate(suite):
        print(f"  [{i+1}/{len(suite)}] {scenario.scenario_id}...", end=" ", flush=True)
        try:
            result = run_one_scenario(scenario, planner, critic, config, failure_analyst=None)
            status = "OK" if result["actual_outcome"] == result["expected_outcome"] or (
                result["expected_outcome"] == "escalated" and result["actual_outcome"] == "partial_success"
            ) else "FAIL"
            print(f"{result['actual_outcome']} ({result['wall_s']}s) [{status}]")
            blind_results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            blind_results.append({
                "scenario_id": scenario.scenario_id,
                "category": scenario.category,
                "expected_outcome": scenario.expected_outcome,
                "actual_outcome": "error",
                "error": str(e),
                "wall_s": 0,
            })

    # --- Phase 2: All 18 scenarios WITH failure analyst ---
    print("\n=== Phase 2: 18 scenarios WITH failure analyst (targeted repair) ===\n")
    targeted_results = []
    for i, scenario in enumerate(suite):
        print(f"  [{i+1}/{len(suite)}] {scenario.scenario_id}...", end=" ", flush=True)
        try:
            result = run_one_scenario(scenario, planner, critic, config, failure_analyst=analyst)
            status = "OK" if result["actual_outcome"] == result["expected_outcome"] or (
                result["expected_outcome"] == "escalated" and result["actual_outcome"] == "partial_success"
            ) else "FAIL"
            print(f"{result['actual_outcome']} ({result['wall_s']}s) [{status}]")
            targeted_results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
            targeted_results.append({
                "scenario_id": scenario.scenario_id,
                "category": scenario.category,
                "expected_outcome": scenario.expected_outcome,
                "actual_outcome": "error",
                "error": str(e),
                "wall_s": 0,
            })

    # --- Compute metrics ---
    def to_scenario_results(raw_list):
        out = []
        for r in raw_list:
            actual = r.get("actual_outcome", "error")
            expected = r.get("expected_outcome", "done")
            if expected == "done":
                oc = actual == "done"
            else:
                oc = actual in (expected, "partial_success")
            out.append(ScenarioResult(
                scenario_id=r["scenario_id"],
                category=r["category"],
                expected_outcome=expected,
                actual_outcome=actual,
                outcome_correct=oc,
                task_completed=actual == "done",
                replans_used=r.get("replans_used", 0),
                retries_used=r.get("retries_used", 0),
                candidates_generated=r.get("candidates", 0),
                survivors_after_critic=r.get("survivors", 0),
                wall_s=r.get("wall_s", 0),
            ))
        return out

    blind_sr = to_scenario_results(blind_results)
    targeted_sr = to_scenario_results(targeted_results)
    blind_metrics = compute_metrics(blind_sr)
    targeted_metrics = compute_metrics(targeted_sr)

    print("\n" + "=" * 70)
    print(f"\n{'Metric':<40} {'Blind':>10} {'Targeted':>10} {'Delta':>10}")
    print("-" * 70)
    all_keys = sorted(set(blind_metrics) | set(targeted_metrics))
    for key in all_keys:
        bv = blind_metrics.get(key, 0.0)
        tv = targeted_metrics.get(key, 0.0)
        delta = tv - bv
        sign = "+" if delta > 0 else ""
        print(f"  {key:<38} {bv:>10.3f} {tv:>10.3f} {sign}{delta:>9.3f}")

    total_blind = sum(r.get("wall_s", 0) for r in blind_results)
    total_targeted = sum(r.get("wall_s", 0) for r in targeted_results)
    print(f"\nTotal wall time: blind={total_blind:.1f}s, targeted={total_targeted:.1f}s")

    # --- Save results ---
    out_dir = REPO / "benchmarks" / "eval_v2" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "claude_18_scenarios.json").write_text(json.dumps({
        "planner": planner_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "blind": {"metrics": blind_metrics, "scenarios": blind_results},
        "targeted": {"metrics": targeted_metrics, "scenarios": targeted_results},
    }, indent=2, default=str))

    (out_dir / "a5_ablation_claude.json").write_text(json.dumps({
        "planner": planner_name,
        "condition_blind": blind_metrics,
        "condition_targeted": targeted_metrics,
    }, indent=2))

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
