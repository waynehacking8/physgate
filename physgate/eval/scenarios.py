"""Orchestration-evaluation scenarios: failure modes that are about ORCHESTRATION.

Each scenario isolates one agent-orchestrator capability (REBUILD.md Phase 3):

    ordering        — does the system order steps correctly / catch inverted plans?
    preconditions   — does it handle unmet preconditions (occupied gripper)?
    recovery        — does it retry transient failures and give up on persistent ones?
    multi_step      — does decomposition cover ALL required objects?
    infeasible      — does it recognize impossible tasks instead of pretending?

Navigation geometry is deliberately ABSENT from every scenario — the
deterministic nav layer guarantees it, so it cannot differentiate orchestrators.

Two REBUILD.md Phase 3 examples are intentionally NOT scenarios here:

* "goal navigation cannot reach" — this suite runs the symbolic gate (no
  geometry), so it cannot test navigation reachability meaningfully. That case
  is covered at the GATE level instead: PathPlannerError surfaces as a clean
  ``infeasible_navigation`` verdict in both Isaac rollouts and the executor
  (tests/test_isaac_sim_gate.py::test_*_unreachable_*).
* "place on an occupied shelf" — the shelf in this world is multi-capacity
  (0.8 m holds several 0.2 m boxes); declaring single-occupancy would
  contradict the multi_step_two_boxes scenario, which places two boxes on the
  same shelf. There is no occupancy precondition to violate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from physgate.gate.schemas import Scene, SceneObject
from physgate.planner.schemas import Plan, PlanStep, RelationChange, ToolName


@dataclass(frozen=True)
class FaultSpec:
    """Inject transient skill failures into execution (recovery testing).

    The fault budget is held across execution attempts: a "grasp slip" is a
    property of the world, not of one backend instance.
    """

    skill: str  # which skill fails ("pick" | "place")
    fail_count: int  # how many attempts fail before succeeding (large = persistent)


@dataclass(frozen=True)
class OrchestrationScenario:
    """One evaluation scenario with its expected orchestrator behaviour."""

    scenario_id: str
    category: str  # ordering | preconditions | recovery | multi_step | infeasible
    description: str
    task: str
    scene: Scene
    #: "done" for feasible tasks, "escalated" for tasks the orchestrator must
    #: recognize as infeasible/unrecoverable and report instead of pretending.
    expected_outcome: str
    #: relations that must hold in the final scene for the task to count as done
    required_final_relations: tuple[tuple[str, str, str], ...] = ()
    fault: FaultSpec | None = None
    #: when set, these plans REPLACE the planner output (gate-probing scenarios)
    probe_plans: tuple[Plan, ...] = field(default_factory=tuple)


# ----------------------------------------------------------------------- scenes


def _basic_scene() -> Scene:
    """The standard fetch-and-place scene: one fallen box, one shelf."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[("box_03", "on", "floor_01"), ("shelf_A", "unoccupied", "shelf_A")],
        gripper_empty=True,
    )


def _occupied_gripper_scene() -> Scene:
    """The gripper already holds another box — plans must deal with it first."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(id="box_99", label="cardboard_box", affordances=["graspable"]),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[
            ("box_03", "on", "floor_01"),
            ("gripper", "holding", "box_99"),
            ("shelf_A", "unoccupied", "shelf_A"),
        ],
        gripper_empty=False,
    )


def _two_box_scene() -> Scene:
    """Two fallen boxes — decomposition must cover both."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(
                id="box_04", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[
            ("box_03", "on", "floor_01"),
            ("box_04", "on", "floor_01"),
            ("shelf_A", "unoccupied", "shelf_A"),
        ],
        gripper_empty=True,
    )


def _two_shelf_scene() -> Scene:
    """Two shelves — the task names a specific shelf the MockPlanner might not target."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="shelf_B", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[("box_03", "on", "floor_01")],
        gripper_empty=True,
    )


def _partial_infeasible_scene() -> Scene:
    """One feasible target + one infeasible (no graspable affordance)."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(id="safe_01", label="safe", affordances=[], is_anomaly=True),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[("box_03", "on", "floor_01"), ("safe_01", "on", "floor_01")],
        gripper_empty=True,
    )


def _ungraspable_scene() -> Scene:
    """The fallen object cannot be grasped — the task is genuinely infeasible."""
    return Scene(
        objects=[
            # a fallen safe: an anomaly, but NOT graspable (too heavy, no handle)
            SceneObject(id="safe_01", label="safe", affordances=[], is_anomaly=True),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[("safe_01", "on", "floor_01"), ("shelf_A", "unoccupied", "shelf_A")],
        gripper_empty=True,
    )


# ----------------------------------------------------------------- probe plans


def _correct_probe_plan() -> Plan:
    """A well-ordered fetch-and-place plan (the one the gate should select)."""
    return Plan(
        plan_id="probe_correct_order",
        task="put the fallen box back on shelf A",
        rationale="correct ordering: approach, pick, carry, place",
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "box_03", "standoff_m": 0.3, "speed": 0.5},
                preconditions=["box_03 exists"],
            ),
            PlanStep(
                step_id=2,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=["box_03 exists", "gripper_empty"],
                effects=[
                    RelationChange(op="add", subject="gripper", predicate="holding", object="box_03")
                ],
            ),
            PlanStep(
                step_id=3,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "shelf_A", "standoff_m": 0.4, "speed": 0.5},
                preconditions=["shelf_A exists"],
            ),
            PlanStep(
                step_id=4,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "place", "target": "shelf_A"},
                preconditions=["gripper holding box_03"],
                effects=[
                    RelationChange(
                        op="remove", subject="gripper", predicate="holding", object="box_03"
                    ),
                    RelationChange(op="add", subject="box_03", predicate="on", object="shelf_A"),
                ],
            ),
        ],
    )


def _inverted_probe_plan() -> Plan:
    """Place-before-pick: structurally invalid ordering the gate MUST catch.

    Note it declares only existence preconditions — it "lies by omission" about
    needing to hold the box, so catching it requires actual rollout/physics, not
    just reading the declared preconditions.
    """
    return Plan(
        plan_id="probe_place_before_pick",
        task="put the fallen box back on shelf A",
        rationale="INVALID: places before picking",
        steps=[
            PlanStep(
                step_id=1,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "shelf_A", "standoff_m": 0.4, "speed": 0.5},
                preconditions=["shelf_A exists"],
            ),
            PlanStep(
                step_id=2,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "place", "target": "shelf_A"},
                preconditions=["shelf_A exists"],
            ),
            PlanStep(
                step_id=3,
                tool=ToolName.MOVE_TO_POSE,
                args={"target": "box_03", "standoff_m": 0.3, "speed": 0.5},
                preconditions=["box_03 exists"],
            ),
            PlanStep(
                step_id=4,
                tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=["box_03 exists"],
            ),
        ],
    )


# ------------------------------------------------------------------- the suite


TASK_FETCH = "put the fallen box back on shelf A"


def build_scenario_suite() -> list[OrchestrationScenario]:
    """The standard orchestration-evaluation suite (deterministic, no GPU)."""
    return [
        OrchestrationScenario(
            scenario_id="fetch_and_place_basic",
            category="ordering",
            description="baseline: single fallen box, correct decomposition expected",
            task=TASK_FETCH,
            scene=_basic_scene(),
            expected_outcome="done",
            required_final_relations=(("box_03", "on", "shelf_A"),),
        ),
        OrchestrationScenario(
            scenario_id="gate_catches_place_before_pick",
            category="ordering",
            description=(
                "gate probe: an inverted (place-before-pick) plan is injected next to a "
                "correct one — the gate must reject the invalid one and select the valid one"
            ),
            task=TASK_FETCH,
            scene=_basic_scene(),
            expected_outcome="done",
            required_final_relations=(("box_03", "on", "shelf_A"),),
            probe_plans=(_inverted_probe_plan(), _correct_probe_plan()),
        ),
        OrchestrationScenario(
            scenario_id="precondition_occupied_gripper",
            category="preconditions",
            description=(
                "the gripper already holds another box — the orchestrator must deal with "
                "it (place it somewhere) before picking the target box"
            ),
            task=(
                "put the fallen box (box_03) back on shelf A; note the gripper is "
                "currently holding box_99"
            ),
            scene=_occupied_gripper_scene(),
            expected_outcome="done",
            required_final_relations=(("box_03", "on", "shelf_A"),),
        ),
        OrchestrationScenario(
            scenario_id="recovery_transient_pick_failure",
            category="recovery",
            description="the first pick attempt fails (grasp slip) — retry must recover",
            task=TASK_FETCH,
            scene=_basic_scene(),
            expected_outcome="done",
            required_final_relations=(("box_03", "on", "shelf_A"),),
            fault=FaultSpec(skill="pick", fail_count=1),
        ),
        OrchestrationScenario(
            scenario_id="recovery_persistent_failure_escalates",
            category="recovery",
            description=(
                "every pick attempt fails — the orchestrator must exhaust its budgets and "
                "escalate (report failure), not loop forever or claim success"
            ),
            task=TASK_FETCH,
            scene=_basic_scene(),
            expected_outcome="escalated",
            fault=FaultSpec(skill="pick", fail_count=10_000),
        ),
        OrchestrationScenario(
            scenario_id="multi_step_two_boxes",
            category="multi_step",
            description=(
                "two fallen boxes — decomposition must cover BOTH (single-object plans "
                "complete 'successfully' but leave the task half done)"
            ),
            task="put both fallen boxes (box_03 and box_04) back on shelf A",
            scene=_two_box_scene(),
            expected_outcome="done",
            required_final_relations=(
                ("box_03", "on", "shelf_A"),
                ("box_04", "on", "shelf_A"),
            ),
        ),
        OrchestrationScenario(
            scenario_id="infeasible_ungraspable_object",
            category="infeasible",
            description=(
                "the fallen object has no graspable affordance — the task is impossible; "
                "the orchestrator must recognize this and escalate"
            ),
            task="put the fallen safe back on shelf A",
            scene=_ungraspable_scene(),
            expected_outcome="escalated",
        ),
        # ---- harder scenarios (H2: break the 100% ceiling) ----
        OrchestrationScenario(
            scenario_id="ambiguous_multi_shelf",
            category="multi_step",
            description=(
                "two shelves — the task says 'the shelf' (ambiguous). The planner "
                "must pick ONE and place on it consistently. MockPlanner defaults to "
                "shelf_A even if the task says 'shelf B'."
            ),
            task="put the fallen box on shelf B",
            scene=_two_shelf_scene(),
            expected_outcome="done",
            required_final_relations=(("box_03", "on", "shelf_B"),),
        ),
        OrchestrationScenario(
            scenario_id="partial_infeasible_two_tasks",
            category="infeasible",
            description=(
                "two sub-tasks: one feasible (box_03 → shelf_A), one infeasible "
                "(safe_01 has no graspable affordance). The orchestrator must complete "
                "the feasible part and report the infeasible part."
            ),
            task=(
                "put the fallen box (box_03) on shelf A AND put the safe (safe_01) "
                "on shelf A"
            ),
            scene=_partial_infeasible_scene(),
            expected_outcome="done",
            required_final_relations=(("box_03", "on", "shelf_A"),),
        ),
        OrchestrationScenario(
            scenario_id="recovery_place_failure_replan",
            category="recovery",
            description=(
                "the place step fails twice — the orchestrator must replan and try a "
                "different approach (e.g. place on the floor first then retry)"
            ),
            task=TASK_FETCH,
            scene=_basic_scene(),
            expected_outcome="done",
            required_final_relations=(("box_03", "on", "shelf_A"),),
            fault=FaultSpec(skill="place", fail_count=2),
        ),
    ]
