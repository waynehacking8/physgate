"""E2 gate-as-classifier scoring: precision / recall / F1 over the defect corpus.

Convention (plan-verification literature): the positive class is "plan rejected".

    TP  invalid plan, rejected      (the gate caught a real defect)
    FP  valid plan, rejected        (the gate killed a good plan)
    FN  invalid plan, accepted      (a defect slipped through)
    TN  valid plan, accepted

Each validation layer (critic / L1+L3 / symbolic L2 / Isaac physics L2) is scored
by the same function over the same corpus, which is what substantiates the
architecture's layered-defense claim: WHICH layer catches WHAT.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from physgate.eval_v2.defects import CorpusItem
from physgate.gate.schemas import Scene
from physgate.planner.schemas import Plan


@dataclass(frozen=True)
class ClassifierReport:
    """Binary-classification metrics for one validation layer."""

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float


@dataclass(frozen=True)
class LayerEvaluation:
    """A layer's classification report + per-defect-class breakdown."""

    layer_name: str
    report: ClassifierReport
    #: defect_id -> fraction of that class's items the layer rejected
    per_defect_rejection_rate: dict[str, float] = field(default_factory=dict)
    #: corpus item plan_id -> rejected?
    verdicts: dict[str, bool] = field(default_factory=dict)


def score_classifier(ground_truth_invalid: list[bool], rejected: list[bool]) -> ClassifierReport:
    """Compute the confusion matrix and derived metrics."""
    if len(ground_truth_invalid) != len(rejected):
        raise ValueError("label/verdict length mismatch")

    tp = sum(1 for g, r in zip(ground_truth_invalid, rejected) if g and r)
    fp = sum(1 for g, r in zip(ground_truth_invalid, rejected) if not g and r)
    fn = sum(1 for g, r in zip(ground_truth_invalid, rejected) if g and not r)
    tn = sum(1 for g, r in zip(ground_truth_invalid, rejected) if not g and not r)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return ClassifierReport(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        false_positive_rate=round(fpr, 4),
    )


def evaluate_layer_on_corpus(
    corpus: list[CorpusItem],
    rejects_fn: Callable[[Plan], bool],
    layer_name: str,
) -> LayerEvaluation:
    """Run one validation layer over every corpus item and score it.

    If rejects_fn accepts an optional ``layout`` keyword argument and a corpus
    item carries a ``layout_override``, it is passed through — this supports
    world-level defects (e.g. D5 unreachable goal) where the defect is in the
    scene, not the plan.
    """
    import inspect

    fn_params = inspect.signature(rejects_fn).parameters
    accepts_layout = "layout" in fn_params

    verdicts: dict[str, bool] = {}
    labels: list[bool] = []
    rejections: list[bool] = []

    for item in corpus:
        if accepts_layout and item.layout_override is not None:
            rejected = bool(rejects_fn(item.plan, layout=item.layout_override))
        else:
            rejected = bool(rejects_fn(item.plan))
        verdicts[item.plan.plan_id] = rejected
        labels.append(item.ground_truth_invalid)
        rejections.append(rejected)

    per_defect: dict[str, float] = {}
    by_defect: dict[str, list[bool]] = {}
    for item, rejected in zip(corpus, rejections):
        by_defect.setdefault(item.defect_id, []).append(rejected)
    for defect_id, flags in by_defect.items():
        per_defect[defect_id] = round(sum(flags) / len(flags), 4)

    return LayerEvaluation(
        layer_name=layer_name,
        report=score_classifier(labels, rejections),
        per_defect_rejection_rate=per_defect,
        verdicts=verdicts,
    )


# ------------------------------------------------------------ layer adapters
#
# Each adapter turns one validation layer into a ``plan -> rejected?`` function.
# The full-gate adapters classify a plan as rejected when the gate marks it
# infeasible (PhysicsResult.success is False).


def critic_rejects(critic_fn, scene: Scene) -> Callable[..., bool]:
    """The safety critic as a classifier: rejected = pruned from the survivors."""

    def _rejects(plan: Plan, layout: dict | None = None) -> bool:
        survivors = critic_fn([plan], scene)
        return len(survivors) == 0

    return _rejects


def l1_l3_rejects(scene: Scene) -> Callable[..., bool]:
    """L1 kinematic + L3 scene checks as a classifier.

    Isolated by running the real gate with a no-op L2 (accepts everything), so
    any rejection is attributable to the deterministic L1/L3 layers alone.
    """
    from physgate.gate.parallel import run_gate
    from physgate.gate.scoring import PhysicsResult

    def _noop_l2(plans: list[Plan], _scene: Scene) -> list[PhysicsResult]:
        return [PhysicsResult(plan_id=p.plan_id, success=True) for p in plans]

    def _rejects(plan: Plan, layout: dict | None = None) -> bool:
        selection = run_gate([plan], scene, l2_fn=_noop_l2)
        return not selection.ranked[0].success

    return _rejects


def symbolic_gate_rejects(scene: Scene) -> Callable[..., bool]:
    """The full symbolic gate (L1 -> L3 -> symbolic L2 + task-goal check).

    Accepts optional ``layout`` kwarg for world-level defects (D5), but the
    symbolic gate does NOT use it — it cannot detect navigation unreachability.
    """
    from physgate.gate.parallel import run_gate, symbolic_l2

    def _rejects(plan: Plan, layout: dict | None = None) -> bool:
        selection = run_gate([plan], scene, l2_fn=symbolic_l2)
        result = selection.ranked[0]
        if not result.success:
            return True
        return not _achieves_goal(plan, scene)

    return _rejects


def _achieves_goal(plan: Plan, scene: Scene) -> bool:
    """Symbolically execute the plan and check the task goal relation holds."""
    from physgate.executor.backend import MockWorldBackend
    from physgate.executor.plan_executor import execute_plan

    backend = MockWorldBackend(scene)
    outcome = execute_plan(plan, backend)
    if not outcome.get("success"):
        return False
    return backend.get_scene().has_relation("box_03", "on", "shelf_A")


def nav_aware_gate_rejects(scene: Scene) -> Callable[..., bool]:
    """The symbolic gate + navigation compilation as a classifier.

    Unlike ``symbolic_gate_rejects``, this adapter compiles the plan through the
    A* path planner. Plans targeting unreachable positions (D5) are caught here
    but NOT by the pure symbolic gate.
    """
    from physgate.gate.parallel import run_gate, symbolic_l2
    from physgate.gate.trajectory import UnknownTargetError, synthesize_base_trajectory
    from physgate.nav.path_planner import PathPlannerError

    def _rejects(plan: Plan, layout: dict | None = None) -> bool:
        selection = run_gate([plan], scene, l2_fn=symbolic_l2)
        result = selection.ranked[0]
        if not result.success:
            return True
        if not _achieves_goal(plan, scene):
            return True
        try:
            synthesize_base_trajectory(plan, 0.02, layout=layout)
        except (PathPlannerError, UnknownTargetError):
            return True
        return False

    return _rejects


def physics_gate_rejects(world, policy_path) -> Callable[..., bool]:
    """The Isaac physics gate (L2 policy rollout) as a classifier.

    Requires a live FetchSimWorld — only callable inside the Isaac venv.
    """
    from physgate.gate.l2_physics import rollout_plans_with_policy

    def _rejects(plan: Plan, layout: dict | None = None) -> bool:
        results = rollout_plans_with_policy(world, [plan], policy_path)
        return not results[0].success

    return _rejects
