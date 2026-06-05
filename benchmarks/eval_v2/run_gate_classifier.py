#!/usr/bin/env python3
"""E2 — score every validation layer as a classifier over the defect corpus.

Layers scored (docs/design/EVALUATION_METHODOLOGY.md §4 E2):

    critic         LLM/mock safety critic
    l1_l3          deterministic kinematic + scene checks
    symbolic_gate  L1 -> L3 -> symbolic L2 + task-goal check
    physics_gate   L1 -> L3 -> Isaac Lab policy rollout (--physics, Isaac venv)

Output: benchmarks/eval_v2/results/gate_classifier.json with per-layer
precision / recall / F1 and per-defect-class catch rates.

Run:
    python benchmarks/eval_v2/run_gate_classifier.py             # symbolic layers
    python benchmarks/eval_v2/run_gate_classifier.py --physics   # + Isaac physics layer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
TASK = "put the fallen box back on shelf A"


def build_corpus():
    from physgate.eval_v2.defects import build_defect_corpus
    from physgate.examples_lib.fetch_and_place import build_demo_scene
    from physgate.planner.critic import MockCritic
    from physgate.planner.planner import MockPlanner

    scene = build_demo_scene()
    base_plans = MockCritic()(MockPlanner()(TASK, scene, 8, None), scene)
    return build_defect_corpus(base_plans), scene, base_plans


def evaluate_symbolic_layers(corpus, scene) -> dict:
    from physgate.eval_v2.classifier import (
        critic_rejects,
        evaluate_layer_on_corpus,
        l1_l3_rejects,
        nav_aware_gate_rejects,
        symbolic_gate_rejects,
    )
    from physgate.planner.critic import MockCritic

    layers = {
        "critic": critic_rejects(MockCritic(), scene),
        "l1_l3": l1_l3_rejects(scene),
        "symbolic_gate": symbolic_gate_rejects(scene),
        "nav_aware_gate": nav_aware_gate_rejects(scene),
    }
    evaluations = {}
    for name, rejects_fn in layers.items():
        t0 = time.perf_counter()
        evaluation = evaluate_layer_on_corpus(corpus, rejects_fn, layer_name=name)
        evaluations[name] = {
            "report": asdict(evaluation.report),
            "per_defect_rejection_rate": evaluation.per_defect_rejection_rate,
            "wall_s": round(time.perf_counter() - t0, 2),
            "n_items": len(corpus),
        }
        print(
            f"  {name:14s} P={evaluation.report.precision:.2f} "
            f"R={evaluation.report.recall:.2f} F1={evaluation.report.f1:.2f} "
            f"FPR={evaluation.report.false_positive_rate:.2f}"
        )
    return evaluations


def evaluate_physics_layer(corpus) -> dict:
    """Isaac physics gate over the corpus, batched 8 plans per rollout
    (requires Isaac venv + GPU)."""
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app  # noqa: F841

    from physgate.eval_v2.classifier import score_classifier
    from physgate.gate.l2_physics import rollout_plans_with_policy
    from physgate.world.fetch_scene import FetchSimWorld
    from physgate.world.locomotion import find_exported_policy

    policy = find_exported_policy()
    if policy is None:
        raise SystemExit("no exported Go2 policy found")

    batch_size = 8
    world = FetchSimWorld(num_envs=batch_size)
    t0 = time.perf_counter()

    # corpus plan_ids are unique; rollouts are batched for GPU parallelism
    verdicts: dict[str, bool] = {}
    for start in range(0, len(corpus), batch_size):
        batch = corpus[start : start + batch_size]
        results = rollout_plans_with_policy(world, [item.plan for item in batch], policy)
        for item, result in zip(batch, results):
            verdicts[item.plan.plan_id] = not result.success
        print(
            f"    physics batch {start // batch_size + 1}/"
            f"{(len(corpus) + batch_size - 1) // batch_size} done"
        )

    labels = [item.ground_truth_invalid for item in corpus]
    rejections = [verdicts[item.plan.plan_id] for item in corpus]
    report = score_classifier(labels, rejections)

    by_defect: dict[str, list[bool]] = {}
    for item in corpus:
        by_defect.setdefault(item.defect_id, []).append(verdicts[item.plan.plan_id])
    per_defect = {k: round(sum(v) / len(v), 4) for k, v in by_defect.items()}

    print(
        f"  physics_gate   P={report.precision:.2f} "
        f"R={report.recall:.2f} F1={report.f1:.2f} "
        f"FPR={report.false_positive_rate:.2f}"
    )
    return {
        "physics_gate": {
            "report": asdict(report),
            "per_defect_rejection_rate": per_defect,
            "wall_s": round(time.perf_counter() - t0, 2),
            "n_items": len(corpus),
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", action="store_true", help="also run the Isaac physics layer")
    args = parser.parse_args()

    corpus, scene, base_plans = build_corpus()
    by_class: dict[str, int] = {}
    for item in corpus:
        by_class[item.defect_id] = by_class.get(item.defect_id, 0) + 1
    print(f"defect corpus: {len(corpus)} items from {len(base_plans)} base plans")
    print(f"  classes: {by_class}")

    print("scoring layers:")
    evaluations = evaluate_symbolic_layers(corpus, scene)
    if args.physics:
        evaluations.update(evaluate_physics_layer(corpus))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "E2 gate-as-classifier (defect injection)",
        "methodology": "docs/design/EVALUATION_METHODOLOGY.md section 4 E2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus": {
            "n_items": len(corpus),
            "n_base_plans": len(base_plans),
            "items_per_class": by_class,
            "ground_truth": {
                item.plan.plan_id: item.ground_truth_invalid for item in corpus
            },
        },
        "layers": evaluations,
        "convention": (
            "positive class = 'plan rejected'; TP = invalid plan rejected; "
            "FP = valid plan rejected (a good plan killed); "
            "FPR = fraction of valid plans rejected"
        ),
    }
    out_path = RESULTS_DIR / "gate_classifier.json"
    out_path.write_text(json.dumps(output, indent=1))
    print(f"results -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
