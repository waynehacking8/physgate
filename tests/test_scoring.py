"""Tests for the Sim-Gate physics scoring + best-of-N selection (A3).

Scoring converts per-plan L2 physics outcomes into a single comparable score.
Design intent (architecture doc §6): physics feasibility is the quality signal;
success dominates, then fewer collisions, then faster, then lower energy.
"""

import pytest

from physgate.gate.schemas import FailureCode, FailureReport, GateLayer, Violation
from physgate.gate.scoring import (
    PhysicsResult,
    ScoringWeights,
    SelectionResult,
    score_result,
    select_best,
)


def _result(plan_id="p", success=True, collisions=0, time_s=10.0, energy_j=100.0, failure=None):
    return PhysicsResult(
        plan_id=plan_id,
        success=success,
        collision_count=collisions,
        completion_time_s=time_s,
        energy_j=energy_j,
        failure=failure,
    )


def _collision_failure() -> FailureReport:
    return FailureReport(
        failure_code=FailureCode.COLLISION,
        layer=GateLayer.PHYSICS,
        violations=[Violation(type="collision", detail="box hit shelf edge")],
    )


# ---------------------------------------------------------------- score_result


def test_success_dominates_everything_else():
    """A successful plan always outscores a failed one, however ugly its metrics."""
    ugly_success = _result(success=True, collisions=5, time_s=300.0, energy_j=9000.0)
    clean_failure = _result(success=False, collisions=0, time_s=1.0, energy_j=1.0)
    assert score_result(ugly_success) > score_result(clean_failure)


def test_fewer_collisions_scores_higher():
    a = _result(collisions=0)
    b = _result(collisions=3)
    assert score_result(a) > score_result(b)


def test_faster_completion_scores_higher():
    a = _result(time_s=5.0)
    b = _result(time_s=50.0)
    assert score_result(a) > score_result(b)


def test_lower_energy_scores_higher():
    a = _result(energy_j=50.0)
    b = _result(energy_j=500.0)
    assert score_result(a) > score_result(b)


def test_collision_penalty_outweighs_time_and_energy():
    """Priority ordering: collisions are a stronger penalty than time/energy."""
    collided_but_fast = _result(collisions=1, time_s=1.0, energy_j=1.0)
    clean_but_slow = _result(collisions=0, time_s=60.0, energy_j=1000.0)
    assert score_result(clean_but_slow) > score_result(collided_but_fast)


def test_custom_weights_respected():
    weights = ScoringWeights(collision_penalty=0.0)
    a = _result(collisions=10, time_s=10.0)
    b = _result(collisions=0, time_s=10.0)
    assert score_result(a, weights) == pytest.approx(score_result(b, weights))


# ----------------------------------------------------------------- select_best


def test_select_best_picks_highest_score():
    results = [
        _result(plan_id="bad", success=False),
        _result(plan_id="good", success=True, collisions=1),
        _result(plan_id="best", success=True, collisions=0),
    ]
    selection = select_best(results)
    assert isinstance(selection, SelectionResult)
    assert selection.best_plan_id == "best"
    assert selection.any_feasible is True
    assert len(selection.ranked) == 3
    assert selection.ranked[0].plan_id == "best"


def test_select_best_all_infeasible():
    results = [
        _result(plan_id="f1", success=False, failure=_collision_failure()),
        _result(plan_id="f2", success=False, failure=_collision_failure()),
    ]
    selection = select_best(results)
    assert selection.any_feasible is False
    assert selection.best_plan_id is None


def test_select_best_empty_input_rejected():
    with pytest.raises(ValueError):
        select_best([])


def test_selection_rationale_mentions_winner_and_count():
    results = [
        _result(plan_id="alpha", success=True),
        _result(plan_id="beta", success=False),
    ]
    selection = select_best(results)
    assert "alpha" in selection.rationale
    assert "2" in selection.rationale  # candidate count


def test_scores_recorded_per_plan():
    results = [_result(plan_id="a"), _result(plan_id="b", collisions=2)]
    selection = select_best(results)
    assert set(selection.scores.keys()) == {"a", "b"}
    assert selection.scores["a"] > selection.scores["b"]
