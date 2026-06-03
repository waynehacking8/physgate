"""Sim-Gate physics scoring and best-of-N selection.

Converts per-plan L2 physics outcomes into a single comparable score and picks
the most feasible candidate. Priority ordering (architecture doc §6):

    success  >>  collision count  >>  completion time  >  energy

Success is a hard tier (a successful plan always beats a failed one); within a
tier the weighted penalties order the candidates.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from physgate.gate.schemas import FailureReport

#: Score bonus that puts every successful plan above every failed plan.
_SUCCESS_TIER = 1_000_000.0


class PhysicsResult(BaseModel):
    """Outcome of executing one candidate plan in L2 physics simulation."""

    plan_id: str
    success: bool
    collision_count: int = 0
    completion_time_s: float = 0.0
    energy_j: float = 0.0
    failure: FailureReport | None = None


class ScoringWeights(BaseModel):
    """Penalty weights. Defaults encode the design priority ordering."""

    collision_penalty: float = 1_000.0
    time_penalty_per_s: float = 1.0
    energy_penalty_per_j: float = 0.01


class SelectionResult(BaseModel):
    """Best-of-N selection outcome handed to the orchestrator."""

    best_plan_id: str | None
    any_feasible: bool
    scores: dict[str, float] = Field(default_factory=dict)
    ranked: list[PhysicsResult] = Field(default_factory=list)
    rationale: str = ""


def score_result(result: PhysicsResult, weights: ScoringWeights | None = None) -> float:
    """Score one physics result. Higher is better."""
    w = weights or ScoringWeights()
    score = _SUCCESS_TIER if result.success else 0.0
    score -= w.collision_penalty * result.collision_count
    score -= w.time_penalty_per_s * result.completion_time_s
    score -= w.energy_penalty_per_j * result.energy_j
    return score


def select_best(
    results: list[PhysicsResult],
    weights: ScoringWeights | None = None,
) -> SelectionResult:
    """Rank candidate plans by physics score and select the best feasible one.

    Raises:
        ValueError: if ``results`` is empty (the gate must never be called
            with zero candidates — that is an orchestrator bug).
    """
    if not results:
        raise ValueError("select_best called with no physics results")

    scores = {r.plan_id: score_result(r, weights) for r in results}
    ranked = sorted(results, key=lambda r: scores[r.plan_id], reverse=True)

    feasible = [r for r in ranked if r.success]
    if not feasible:
        return SelectionResult(
            best_plan_id=None,
            any_feasible=False,
            scores=scores,
            ranked=ranked,
            rationale=(
                f"all {len(results)} candidate plans failed physics validation; "
                "replanning required"
            ),
        )

    best = ranked[0]
    return SelectionResult(
        best_plan_id=best.plan_id,
        any_feasible=True,
        scores=scores,
        ranked=ranked,
        rationale=(
            f"selected '{best.plan_id}' out of {len(results)} candidates: "
            f"score {scores[best.plan_id]:.1f}, "
            f"{best.collision_count} collisions, "
            f"{best.completion_time_s:.1f}s, {best.energy_j:.0f}J "
            f"({len(feasible)}/{len(results)} candidates were feasible)"
        ),
    )
