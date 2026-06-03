"""Sim-Gate pipeline: L1 -> L3 -> L2 -> scoring -> best-of-N selection (C12).

Layers run cheapest-first so candidates rejected by deterministic checks never
consume physics-simulation time:

    L1  kinematic limits   (numpy, <1 ms)        gate/l1_kinematic.py
    L3  symbolic rollout   (pure Python, <1 ms)  gate/l3_scene.py + scene_graph
    L2  physics simulation (GPU, pluggable)      symbolic_l2 here, or
                                                 gate/l2_physics.py on Isaac Lab

The L2 implementation is injected (``l2_fn``) so the same gate code runs in
unit tests (symbolic), the offline demo (symbolic), and production (Isaac Lab
parallel physics).

Design reference: docs/design/architecture.md section 5.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from physgate.executor.backend import MockWorldBackend
from physgate.executor.plan_executor import execute_plan
from physgate.gate.l1_kinematic import check_joint_trajectory
from physgate.gate.l3_scene import check_preconditions
from physgate.gate.schemas import FailureCode, FailureReport, GateLayer, Scene, Violation
from physgate.gate.scoring import PhysicsResult, ScoringWeights, SelectionResult, select_best
from physgate.planner.schemas import Plan
from physgate.world.scene_graph import apply_effects

#: L2 signature: (plans, scene) -> per-plan physics results.
L2Fn = Callable[[list[Plan], Scene], list[PhysicsResult]]


# ------------------------------------------------------------ deterministic layers


def _l1_check(plan: Plan) -> FailureReport | None:
    """L1: check any explicit joint trajectories carried in step args."""
    for step in plan.steps:
        trajectory = step.args.get("joint_trajectory")
        if trajectory is None:
            continue
        report = check_joint_trajectory(
            np.asarray(trajectory, dtype=np.float64),
            dt=step.args.get("trajectory_dt"),
            step_id=step.step_id,
        )
        if report is not None:
            return report
    return None


def _l3_check(plan: Plan, scene: Scene) -> FailureReport | None:
    """L3: symbolic rollout — check each step's preconditions against the
    scene state predicted by applying previous steps' effects."""
    current = scene
    for step in plan.steps:
        report = check_preconditions(step.preconditions, current, step.step_id)
        if report is not None:
            return report
        current = apply_effects(current, step.effects)
    return None


# --------------------------------------------------------------------- L2 (mock)


def symbolic_l2(plans: list[Plan], scene: Scene) -> list[PhysicsResult]:
    """Symbolic stand-in for physics simulation (no GPU).

    Rolls each plan out on a fresh MockWorldBackend and synthesizes physics
    metrics from plan structure:
        completion time ~ sum of (1 / speed) per motion step + skill costs
        energy          ~ proportional to motion at speed^2 (faster = costlier)
    Used by unit tests and the offline demo; replaced by Isaac Lab L2
    (gate/l2_physics.py) when a simulator is available.
    """
    results: list[PhysicsResult] = []
    for plan in plans:
        backend = MockWorldBackend(scene)
        execution = execute_plan(plan, backend)

        time_s = 0.0
        energy_j = 0.0
        for step in plan.steps:
            speed = float(step.args.get("speed", 0.5))
            if step.args.get("target") and "skill" not in step.args:
                # motion step: distance/speed time cost, kinetic energy cost
                time_s += 2.0 / max(speed, 0.05)
                energy_j += 50.0 * speed**2
            elif "skill" in step.args:
                time_s += 3.0  # fixed manipulation cost
                energy_j += 20.0
            else:
                time_s += 0.5  # perception step

        failure: FailureReport | None = None
        if not execution["success"]:
            if execution.get("failure_report"):
                failure = FailureReport.model_validate(execution["failure_report"])
            else:
                failure = FailureReport(
                    failed_step_id=execution.get("failed_step_id"),
                    failure_code=FailureCode.GRASP_FAILURE,
                    layer=GateLayer.PHYSICS,
                    violations=[
                        Violation(type="symbolic_rollout_failure", detail=execution.get("error", ""))
                    ],
                    remediation_hint="plan failed symbolic execution; fix step ordering or targets",
                )

        results.append(
            PhysicsResult(
                plan_id=plan.plan_id,
                success=execution["success"],
                collision_count=0,  # symbolic rollout cannot detect collisions
                completion_time_s=round(time_s, 2),
                energy_j=round(energy_j, 2),
                failure=failure,
            )
        )
    return results


# ---------------------------------------------------------------------- pipeline


def run_gate(
    plans: list[Plan],
    scene: Scene,
    l2_fn: L2Fn = symbolic_l2,
    weights: ScoringWeights | None = None,
) -> SelectionResult:
    """Run candidate plans through L1 -> L3 -> L2 -> scoring; select the best.

    Plans rejected by L1/L3 receive failed PhysicsResults (with the layer's
    FailureReport attached) and never reach L2.
    """
    if not plans:
        return SelectionResult(
            best_plan_id=None,
            any_feasible=False,
            rationale="gate received zero candidate plans (critic rejected everything)",
        )

    deterministic_failures: list[PhysicsResult] = []
    l2_candidates: list[Plan] = []

    for plan in plans:
        report = _l1_check(plan) or _l3_check(plan, scene)
        if report is not None:
            deterministic_failures.append(
                PhysicsResult(plan_id=plan.plan_id, success=False, failure=report)
            )
        else:
            l2_candidates.append(plan)

    l2_results = l2_fn(l2_candidates, scene) if l2_candidates else []

    return select_best(deterministic_failures + l2_results, weights)
