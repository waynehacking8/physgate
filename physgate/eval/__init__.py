"""Agent-orchestrator evaluation suite (REBUILD.md Phase 3).

Measures what the project actually claims to validate: whether the agent
orchestrator (planner + critic + gate + recovery loop) decomposes tasks
correctly, respects preconditions, recovers from failures, and recognizes
infeasible requests — NOT whether it can route around obstacles (the
deterministic navigation layer owns that).
"""

from physgate.eval.metrics import OrchestratorReport, ScenarioResult, compute_metrics
from physgate.eval.runner import run_scenario, run_suite
from physgate.eval.scenarios import FaultSpec, OrchestrationScenario, build_scenario_suite

__all__ = [
    "FaultSpec",
    "OrchestrationScenario",
    "OrchestratorReport",
    "ScenarioResult",
    "build_scenario_suite",
    "compute_metrics",
    "run_scenario",
    "run_suite",
]
