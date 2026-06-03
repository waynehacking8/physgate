"""E2 defect injection: build a labeled corpus of clean and corrupted plans.

The defect taxonomy follows the plan-verification literature
(arXiv:2509.02761 — irrelevant actions, contradictions, missing steps), adapted
to this domain. Every injector is pure: it returns a NEW Plan and never mutates
the input (plans are pydantic models; copies are made with ``model_copy``).

Ground-truth labels are assigned by construction — no manual annotation
(PlanBench's approach): a defect class is "invalid" exactly when a correct
validation gate should reject plans of that class.
"""

from __future__ import annotations

from dataclasses import dataclass

from physgate.planner.schemas import Plan, PlanStep, ToolName

#: defect_id -> ground_truth_invalid (True = a correct gate rejects this)
DEFECT_CLASSES: dict[str, bool] = {
    "D0_clean": False,
    "D1_step_inversion": True,
    "D2_missing_pick": True,
    "D3_hallucinated_target": True,
    "D4_wrong_placement": True,
    "D6_speed_violation": False,  # the gate clamps speeds (D-018); rejection = false positive
    "D7_redundant_step": False,  # wasteful but executable; rejection = false positive
}

#: The object id that no scene contains (for hallucination injection).
NONEXISTENT_ID = "staging_area_99"


@dataclass(frozen=True)
class CorpusItem:
    """One labeled item of the defect corpus."""

    plan: Plan
    defect_id: str
    ground_truth_invalid: bool
    base_plan_id: str


# ------------------------------------------------------------------ injectors


def _renumber(steps: list[PlanStep]) -> list[PlanStep]:
    """Return steps with sequential step_ids (schema requires uniqueness)."""
    return [s.model_copy(update={"step_id": i + 1}) for i, s in enumerate(steps)]


def _skill_of(step: PlanStep) -> str | None:
    return step.args.get("skill") if step.tool == ToolName.EXECUTE_SKILL else None


def inject_step_inversion(plan: Plan) -> Plan:
    """D1: move the place step (and its approach) before the pick step."""
    pick_block: list[PlanStep] = []
    place_block: list[PlanStep] = []
    rest: list[PlanStep] = []

    pick_seen = False
    for step in plan.steps:
        skill = _skill_of(step)
        if skill == "pick":
            pick_block.append(step)
            pick_seen = True
        elif skill == "place" or (not pick_seen and not pick_block and skill is None and rest):
            place_block.append(step)
        elif pick_seen:
            place_block.append(step)
        else:
            rest.append(step)

    inverted = rest[:1] + place_block + pick_block + rest[1:]
    return plan.model_copy(
        update={"plan_id": f"{plan.plan_id}__D1", "steps": _renumber(inverted)}
    )


def inject_missing_pick(plan: Plan) -> Plan:
    """D2: delete the pick step — the plan tries to place a box it never picked."""
    steps = [s for s in plan.steps if _skill_of(s) != "pick"]
    return plan.model_copy(update={"plan_id": f"{plan.plan_id}__D2", "steps": _renumber(steps)})


def inject_hallucinated_target(plan: Plan) -> Plan:
    """D3: the first move step targets an object id that does not exist."""
    steps: list[PlanStep] = []
    replaced = False
    for step in plan.steps:
        if not replaced and step.tool == ToolName.MOVE_TO_POSE:
            new_args = {**step.args, "target": NONEXISTENT_ID}
            new_preconds = [f"{NONEXISTENT_ID} exists"]
            steps.append(
                step.model_copy(update={"args": new_args, "preconditions": new_preconds})
            )
            replaced = True
        else:
            steps.append(step)
    return plan.model_copy(update={"plan_id": f"{plan.plan_id}__D3", "steps": steps})


def inject_wrong_placement(plan: Plan) -> Plan:
    """D4: place onto the floor instead of the shelf — executable, but the task
    goal (box on shelf) is never achieved."""
    steps: list[PlanStep] = []
    for step in plan.steps:
        if _skill_of(step) == "place" or (
            step.tool == ToolName.MOVE_TO_POSE and step.args.get("target") == "shelf_A"
        ):
            new_args = {**step.args, "target": "floor_01"}
            new_preconds = ["floor_01 exists"]
            steps.append(
                step.model_copy(update={"args": new_args, "preconditions": new_preconds})
            )
        else:
            steps.append(step)
    return plan.model_copy(update={"plan_id": f"{plan.plan_id}__D4", "steps": steps})


def inject_speed_violation(plan: Plan) -> Plan:
    """D6: command a speed far outside the locomotion envelope. The gate clamps
    speeds (D-018), so this plan is VALID — rejecting it is a false positive."""
    steps = [
        s.model_copy(update={"args": {**s.args, "speed": 5.0}})
        if s.tool == ToolName.MOVE_TO_POSE
        else s
        for s in plan.steps
    ]
    return plan.model_copy(update={"plan_id": f"{plan.plan_id}__D6", "steps": steps})


def inject_redundant_step(plan: Plan) -> Plan:
    """D7: duplicate the first move step — wasteful but perfectly executable."""
    first_move = next(s for s in plan.steps if s.tool == ToolName.MOVE_TO_POSE)
    steps = [plan.steps[0]] + [first_move] + list(plan.steps[1:])
    if plan.steps[0] is not first_move:
        steps = [plan.steps[0], first_move, *plan.steps[1:]]
    else:
        steps = [first_move, first_move.model_copy(), *plan.steps[1:]]
    return plan.model_copy(update={"plan_id": f"{plan.plan_id}__D7", "steps": _renumber(steps)})


_INJECTORS = {
    "D1_step_inversion": inject_step_inversion,
    "D2_missing_pick": inject_missing_pick,
    "D3_hallucinated_target": inject_hallucinated_target,
    "D4_wrong_placement": inject_wrong_placement,
    "D6_speed_violation": inject_speed_violation,
    "D7_redundant_step": inject_redundant_step,
}


# -------------------------------------------------------------------- corpus


def build_defect_corpus(base_plans: list[Plan]) -> list[CorpusItem]:
    """Build the labeled corpus: every base plan in clean form + one variant per
    defect class. Labels come from DEFECT_CLASSES by construction."""
    corpus: list[CorpusItem] = []
    for plan in base_plans:
        corpus.append(
            CorpusItem(
                plan=plan,
                defect_id="D0_clean",
                ground_truth_invalid=DEFECT_CLASSES["D0_clean"],
                base_plan_id=plan.plan_id,
            )
        )
        for defect_id, injector in _INJECTORS.items():
            corpus.append(
                CorpusItem(
                    plan=injector(plan),
                    defect_id=defect_id,
                    ground_truth_invalid=DEFECT_CLASSES[defect_id],
                    base_plan_id=plan.plan_id,
                )
            )
    return corpus
