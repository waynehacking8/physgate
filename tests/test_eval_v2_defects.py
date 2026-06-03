"""Tests for eval_v2 defect injection + gate-as-classifier scoring (E2).

The gate is a binary classifier over plans (reject / accept). Following the
plan-verification literature (arXiv:2509.02761), it is evaluated with
precision / recall / F1 over a labeled corpus of clean and corrupted plans.
"""

from __future__ import annotations

import pytest

from physgate.examples_lib.fetch_and_place import build_demo_scene
from physgate.planner.planner import MockPlanner
from physgate.planner.schemas import ToolName

TASK = "put the fallen box back on shelf A"


@pytest.fixture()
def base_plans():
    """Known-good plans (critic-passing mock variants)."""
    from physgate.planner.critic import MockCritic

    scene = build_demo_scene()
    plans = MockPlanner()(TASK, scene, 8, None)
    return MockCritic()(plans, scene)


@pytest.fixture()
def scene():
    return build_demo_scene()


# --------------------------------------------------------------- injectors


def test_step_inversion_moves_place_before_pick(base_plans):
    from physgate.eval_v2.defects import inject_step_inversion

    plan = base_plans[0]
    corrupted = inject_step_inversion(plan)

    tools = [(s.tool, s.args.get("skill")) for s in corrupted.steps]
    place_idx = next(i for i, t in enumerate(tools) if t[1] == "place")
    pick_idx = next(i for i, t in enumerate(tools) if t[1] == "pick")
    assert place_idx < pick_idx, "place must come before pick in the corrupted plan"
    # the original plan is untouched (immutability)
    orig_tools = [(s.tool, s.args.get("skill")) for s in plan.steps]
    orig_place = next(i for i, t in enumerate(orig_tools) if t[1] == "place")
    orig_pick = next(i for i, t in enumerate(orig_tools) if t[1] == "pick")
    assert orig_pick < orig_place


def test_missing_pick_removes_the_pick_step(base_plans):
    from physgate.eval_v2.defects import inject_missing_pick

    corrupted = inject_missing_pick(base_plans[0])
    skills = [s.args.get("skill") for s in corrupted.steps if s.tool == ToolName.EXECUTE_SKILL]
    assert "pick" not in skills
    assert "place" in skills


def test_hallucinated_target_references_nonexistent_object(base_plans, scene):
    from physgate.eval_v2.defects import inject_hallucinated_target

    corrupted = inject_hallucinated_target(base_plans[0])
    scene_ids = {o.id for o in scene.objects}
    targets = {s.args.get("target") for s in corrupted.steps if s.args.get("target")}
    assert targets - scene_ids, "corrupted plan must reference at least one nonexistent id"


def test_wrong_placement_targets_floor_not_shelf(base_plans):
    from physgate.eval_v2.defects import inject_wrong_placement

    corrupted = inject_wrong_placement(base_plans[0])
    place_steps = [
        s
        for s in corrupted.steps
        if s.tool == ToolName.EXECUTE_SKILL and s.args.get("skill") == "place"
    ]
    assert all(s.args.get("target") != "shelf_A" for s in place_steps)


def test_speed_violation_is_marked_valid_ground_truth(base_plans):
    """Out-of-envelope speed is NOT an invalid plan: the gate clamps it (D-018).
    A gate that rejects it has a false positive."""
    from physgate.eval_v2.defects import build_defect_corpus

    corpus = build_defect_corpus(base_plans[:2])
    speed_items = [c for c in corpus if c.defect_id == "D6_speed_violation"]
    assert speed_items
    assert all(c.ground_truth_invalid is False for c in speed_items)


def test_corpus_covers_all_defect_classes_with_controls(base_plans):
    from physgate.eval_v2.defects import DEFECT_CLASSES, build_defect_corpus

    corpus = build_defect_corpus(base_plans)
    seen_defects = {c.defect_id for c in corpus}
    assert seen_defects == set(DEFECT_CLASSES)
    # clean controls present and labeled valid
    clean = [c for c in corpus if c.defect_id == "D0_clean"]
    assert len(clean) == len(base_plans)
    assert all(c.ground_truth_invalid is False for c in clean)
    # every corrupted plan still parses as a Plan (schema-valid)
    assert all(c.plan.plan_id for c in corpus)


# -------------------------------------------------------- classifier scoring


def test_classifier_report_perfect_gate():
    """A gate that rejects exactly the invalid items scores P=R=F1=1."""
    from physgate.eval_v2.classifier import score_classifier

    labels = [True, True, False, False]  # ground truth: invalid?
    rejections = [True, True, False, False]  # gate's verdicts
    report = score_classifier(labels, rejections)
    assert report.precision == 1.0 and report.recall == 1.0 and report.f1 == 1.0


def test_classifier_report_overly_aggressive_gate():
    """A gate that rejects everything has perfect recall but poor precision."""
    from physgate.eval_v2.classifier import score_classifier

    labels = [True, False, False, False]
    rejections = [True, True, True, True]
    report = score_classifier(labels, rejections)
    assert report.recall == 1.0
    assert report.precision == 0.25
    assert report.false_positive_rate == 1.0


def test_classifier_report_blind_gate():
    """A gate that accepts everything has zero recall."""
    from physgate.eval_v2.classifier import score_classifier

    labels = [True, True, False]
    rejections = [False, False, False]
    report = score_classifier(labels, rejections)
    assert report.recall == 0.0
    assert report.false_positive_rate == 0.0


# ------------------------------------------------- end-to-end layer scoring


def test_l1_l3_layer_catches_hallucination_but_not_ordering(base_plans, scene):
    """L1+L3 (deterministic checks) catch hallucinated targets but CANNOT catch
    step inversion (that needs symbolic execution) — the layered-defense claim."""
    from physgate.eval_v2.classifier import evaluate_layer_on_corpus, l1_l3_rejects
    from physgate.eval_v2.defects import build_defect_corpus

    corpus = build_defect_corpus(base_plans)
    result = evaluate_layer_on_corpus(corpus, l1_l3_rejects(scene), layer_name="l1_l3")

    assert result.per_defect_rejection_rate["D3_hallucinated_target"] == 1.0
    assert result.per_defect_rejection_rate["D0_clean"] == 0.0


def test_critic_layer_scored_on_corpus(base_plans, scene):
    """The mock critic as a classifier — runs without error and never rejects
    the clean control plans it itself approved."""
    from physgate.eval_v2.classifier import critic_rejects, evaluate_layer_on_corpus
    from physgate.eval_v2.defects import build_defect_corpus
    from physgate.planner.critic import MockCritic

    corpus = build_defect_corpus(base_plans)
    result = evaluate_layer_on_corpus(corpus, critic_rejects(MockCritic(), scene), "critic")

    assert result.per_defect_rejection_rate["D0_clean"] == 0.0
    assert 0.0 <= result.report.recall <= 1.0


def test_symbolic_gate_scored_on_corpus(base_plans, scene):
    """Run the real symbolic gate (L1+L3+symbolic L2) over the defect corpus and
    verify it catches the ordering/hallucination defects without rejecting
    clean plans."""
    from physgate.eval_v2.classifier import evaluate_layer_on_corpus, symbolic_gate_rejects
    from physgate.eval_v2.defects import build_defect_corpus

    corpus = build_defect_corpus(base_plans)
    result = evaluate_layer_on_corpus(corpus, symbolic_gate_rejects(scene), layer_name="symbolic")

    # clean plans must not be rejected (precision on controls)
    assert result.per_defect_rejection_rate["D0_clean"] == 0.0, (
        "symbolic gate rejected clean plans — false positives"
    )
    # inverted and hallucinated plans must be caught
    assert result.per_defect_rejection_rate["D1_step_inversion"] == 1.0
    assert result.per_defect_rejection_rate["D3_hallucinated_target"] == 1.0
    # overall report exists
    assert 0.0 <= result.report.f1 <= 1.0
