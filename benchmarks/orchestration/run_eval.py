#!/usr/bin/env python3
"""Agent-orchestrator evaluation benchmark (REBUILD.md Phase 3).

Runs the orchestration scenario suite — ordering, preconditions, recovery,
multi-step decomposition, infeasibility recognition — and reports
agent-orchestrator metrics. This REPLACES the deprecated best-of-N cost/quality
benchmark (#8) as the project's headline evaluation.

No GPU needed (symbolic plan-logic gate); physical-outcome validation of winning
plans is covered separately by the Isaac integration tests and demo.

Usage (any venv with physgate installed):

    python benchmarks/orchestration/run_eval.py --planner mock
    python benchmarks/orchestration/run_eval.py --planner llm     # real Claude
    python benchmarks/orchestration/run_eval.py --planner both
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from physgate.eval.metrics import OrchestratorReport
from physgate.eval.runner import run_suite
from physgate.eval.scenarios import build_scenario_suite
from physgate.planner.critic import MockCritic, make_critic
from physgate.planner.planner import MockPlanner, llm_credentials_available, make_planner

RESULTS_DIR = Path(__file__).parent / "results"


def evaluate(planner_label: str) -> OrchestratorReport | None:
    """Run the suite for one planner configuration."""
    if planner_label == "mock":
        planner, critic = MockPlanner(), MockCritic()
    else:
        if not llm_credentials_available():
            print(f"WARNING: planner '{planner_label}' needs LLM credentials; skipping")
            return None
        planner, critic = make_planner(), make_critic()

    suite = build_scenario_suite()
    print(f"=== {planner_label}: {len(suite)} scenarios ===")
    report = run_suite(suite, planner, critic, planner_name=planner_label)
    for result in report.results:
        marker = "ok " if result.outcome_correct else "MISS"
        print(
            f"  [{marker}] {result.scenario_id:<40} expected={result.expected_outcome:<10}"
            f" actual={result.actual_outcome:<10} completed={result.task_completed}"
            f" ({result.wall_s:.1f}s)"
        )
    print(f"  metrics: {json.dumps(report.metrics, indent=4)}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", choices=["mock", "llm", "both"], default="mock")
    args = parser.parse_args()

    labels = ["mock", "llm"] if args.planner == "both" else [args.planner]
    reports = [r for label in labels if (r := evaluate(label)) is not None]
    if not reports:
        print("no evaluations ran")
        return 1

    output = {
        "benchmark": "orchestration evaluation - agent orchestrator quality",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "what_this_measures": (
            "Whether the agent orchestrator (planner + critic + gate + recovery loop) "
            "decomposes tasks correctly, respects preconditions, recovers from failures, "
            "and recognizes infeasible requests. Navigation/obstacle avoidance is NOT "
            "measured — the deterministic nav layer guarantees it for every plan "
            "(REBUILD.md Phase 1), so it cannot differentiate orchestrators."
        ),
        "metrics_legend": {
            "end_to_end_success_rate": "feasible tasks completed (final relations verified)",
            "infeasible_recognition_rate": "impossible tasks recognized and escalated",
            "recovery_rate": "transient failures recovered via retry/replan",
            "decomposition_validity_rate": "selected plans are well-ordered",
            "invalid_plan_catch_rate": "known-invalid probe plans rejected by the gate",
            "orchestrator_score": "unweighted mean of the above",
        },
        "reports": [report.model_dump(mode="json") for report in reports],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "orchestration_eval.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nresults -> {out_path}")

    for report in reports:
        print(f"{report.planner_name}: orchestrator_score = {report.metrics['orchestrator_score']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
