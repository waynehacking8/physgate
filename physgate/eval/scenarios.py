"""Orchestration-evaluation scenarios: failure modes that are about ORCHESTRATION.

Each scenario isolates one agent-orchestrator capability:

    ordering        — does the system order steps correctly / catch inverted plans?
    preconditions   — does it handle unmet preconditions (occupied gripper)?
    recovery        — does it retry transient failures and give up on persistent ones?
    multi_step      — does decomposition cover ALL required objects?
    infeasible      — does it recognize impossible tasks instead of pretending?
    locked_door     — T2: key→unlock→open→deliver chain
    blocked_path    — T3: inspect→push or A* detour
    sequential      — T4: multi-object ordering dependency
    elevator        — T5: cross-floor delivery
    assistance      — T6: infeasible task → request_assistance

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
                "(safe_01 has no graspable affordance). The agent correctly "
                "recognizes the infeasible part and requests assistance."
            ),
            task=(
                "put the fallen box (box_03) on shelf A AND put the safe (safe_01) "
                "on shelf A"
            ),
            scene=_partial_infeasible_scene(),
            expected_outcome="escalated",
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
    ] + _t2_t6_scenarios()


# ============================================= T2-T6 scene constructors


def _locked_door_scene() -> Scene:
    """T2: a locked room with the delivery target behind the door."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(id="shelf_B", label="shelf", affordances=["placeable"]),
            SceneObject(id="door_01", label="door", affordances=["door"], locked=True),
            SceneObject(id="key_01", label="key", affordances=["graspable"], weight_kg=0.1),
            SceneObject(id="table_01", label="table", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[
            ("box_03", "on", "floor_01"),
            ("key_01", "on", "table_01"),
            ("door_01", "state", "closed"),
            ("door_01", "state", "locked"),
            ("shelf_B", "behind", "door_01"),
        ],
        gripper_empty=True,
    )


def _blocked_path_scene() -> Scene:
    """T3: a pushable crate blocks the direct path to the shelf."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="crate_01", label="crate", pushable=True, weight_kg=20.0),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[
            ("box_03", "on", "floor_01"),
            ("crate_01", "blocking", "path_to_shelf_A"),
        ],
        gripper_empty=True,
    )


def _sequential_delivery_scene() -> Scene:
    """T4: two boxes, shelf_B occupied by box_05 which must be cleared first."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(
                id="box_04", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(
                id="box_05", label="cardboard_box", affordances=["graspable"]
            ),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="shelf_B", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[
            ("box_03", "on", "floor_01"),
            ("box_04", "on", "floor_01"),
            ("box_05", "on", "shelf_B"),
        ],
        gripper_empty=True,
    )


def _elevator_scene() -> Scene:
    """T5: box on floor 1, shelf on floor 2, elevator available."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"],
                is_anomaly=True, floor=1,
            ),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"], floor=2),
            SceneObject(id="elevator_01", label="elevator", affordances=["elevator"], floor=1),
            SceneObject(id="floor_01", label="floor", floor=1),
            SceneObject(id="floor_02", label="floor", floor=2),
            SceneObject(id="go2", label="robot", floor=1),
        ],
        relations=[("box_03", "on", "floor_01")],
        gripper_empty=True,
    )


def _heavy_object_scene() -> Scene:
    """T6 (infeasible): anvil is 200kg, robot max carry is 5kg."""
    return Scene(
        objects=[
            SceneObject(
                id="anvil_01", label="anvil", affordances=[], weight_kg=200.0, is_anomaly=True
            ),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[("anvil_01", "on", "floor_01")],
        gripper_empty=True,
    )


def _sealed_room_scene() -> Scene:
    """T6 (infeasible): room_C has no door — completely walled off."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(id="shelf_C", label="shelf", affordances=["placeable"]),
            SceneObject(id="wall_N", label="wall"),
            SceneObject(id="wall_S", label="wall"),
            SceneObject(id="wall_E", label="wall"),
            SceneObject(id="wall_W", label="wall"),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[
            ("box_03", "on", "floor_01"),
            ("shelf_C", "enclosed_by", "wall_N"),
            ("shelf_C", "enclosed_by", "wall_S"),
            ("shelf_C", "enclosed_by", "wall_E"),
            ("shelf_C", "enclosed_by", "wall_W"),
        ],
        gripper_empty=True,
    )


def _blocked_path_no_push_scene() -> Scene:
    """T3 variant: path blocked by an immovable (non-pushable) object."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True
            ),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="pillar_01", label="pillar", pushable=False, weight_kg=500.0),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[
            ("box_03", "on", "floor_01"),
            ("pillar_01", "blocking", "path_to_shelf_A"),
        ],
        gripper_empty=True,
    )


# ============================================= T2-T6 scenario definitions


def _t2_t6_scenarios() -> list[OrchestrationScenario]:
    """Multi-task scenarios added by the agent architecture upgrade."""
    return [
        # ---- T2: Locked Room Delivery ----
        OrchestrationScenario(
            scenario_id="t2_locked_door_delivery",
            category="locked_door",
            description=(
                "T2: shelf_B is behind a locked door. Agent must find key on table, "
                "pick key, unlock door, open door, then deliver box."
            ),
            task="deliver box_03 to shelf_B; shelf_B is behind locked door_01; key_01 is on table_01",
            scene=_locked_door_scene(),
            expected_outcome="done",
            required_final_relations=(("box_03", "on", "shelf_B"),),
        ),
        OrchestrationScenario(
            scenario_id="t2_locked_door_no_key_escalate",
            category="locked_door",
            description=(
                "T2 infeasible: locked door but no key in scene — agent must escalate"
            ),
            task="deliver box_03 to shelf_B; shelf_B is behind locked door_01",
            scene=Scene(
                objects=[
                    SceneObject(
                        id="box_03", label="cardboard_box",
                        affordances=["graspable"], is_anomaly=True,
                    ),
                    SceneObject(id="shelf_B", label="shelf", affordances=["placeable"]),
                    SceneObject(id="door_01", label="door", affordances=["door"], locked=True),
                    SceneObject(id="floor_01", label="floor"),
                    SceneObject(id="go2", label="robot"),
                ],
                relations=[
                    ("box_03", "on", "floor_01"),
                    ("door_01", "state", "closed"),
                    ("door_01", "state", "locked"),
                    ("shelf_B", "behind", "door_01"),
                ],
                gripper_empty=True,
            ),
            expected_outcome="escalated",
        ),
        # ---- T3: Blocked Path Clearance ----
        OrchestrationScenario(
            scenario_id="t3_blocked_path_push",
            category="blocked_path",
            description=(
                "T3: direct path to shelf blocked by a pushable crate. Agent must "
                "inspect the crate, push it aside, then deliver."
            ),
            task="deliver box_03 to shelf_A; the path is blocked by crate_01",
            scene=_blocked_path_scene(),
            expected_outcome="done",
            required_final_relations=(("box_03", "on", "shelf_A"),),
        ),
        OrchestrationScenario(
            scenario_id="t3_blocked_path_immovable",
            category="blocked_path",
            description=(
                "T3 hard: path blocked by an immovable pillar. A real LLM agent "
                "may attempt delivery (A* nav handles detour) or escalate. "
                "Mock backend does not enforce blocking — A* detour succeeds."
            ),
            task="deliver box_03 to shelf_A; the path is blocked by pillar_01 (immovable)",
            scene=_blocked_path_no_push_scene(),
            expected_outcome="done",
            required_final_relations=(("box_03", "on", "shelf_A"),),
        ),
        # ---- T4: Multi-Object Sequential Delivery ----
        OrchestrationScenario(
            scenario_id="t4_sequential_delivery",
            category="sequential",
            description=(
                "T4: deliver box_03 to shelf_A, box_04 to shelf_B. But shelf_B has "
                "box_05 on it — must clear shelf_B first, respecting ordering."
            ),
            task=(
                "deliver box_03 to shelf_A and box_04 to shelf_B; "
                "shelf_B currently has box_05 on it — clear it first"
            ),
            scene=_sequential_delivery_scene(),
            expected_outcome="done",
            required_final_relations=(
                ("box_03", "on", "shelf_A"),
                ("box_04", "on", "shelf_B"),
            ),
        ),
        # ---- T5: Elevator Floor Transfer ----
        OrchestrationScenario(
            scenario_id="t5_elevator_delivery",
            category="elevator",
            description=(
                "T5: box on floor 1, shelf on floor 2. Agent must pick box, "
                "call elevator, ride to floor 2, then deliver."
            ),
            task="move box_03 from floor 1 to shelf_A on floor 2 using elevator_01",
            scene=_elevator_scene(),
            expected_outcome="done",
            required_final_relations=(("box_03", "on", "shelf_A"),),
        ),
        # ---- T6: Infeasible Task Recognition ----
        OrchestrationScenario(
            scenario_id="t6_too_heavy",
            category="assistance",
            description=(
                "T6: anvil is 200kg, robot can carry 5kg max. Agent must inspect, "
                "recognize infeasibility, and call request_assistance."
            ),
            task="deliver anvil_01 to shelf_A",
            scene=_heavy_object_scene(),
            expected_outcome="escalated",
        ),
        OrchestrationScenario(
            scenario_id="t6_sealed_room",
            category="assistance",
            description=(
                "T6: shelf_C is in a fully sealed room with no door. Agent must "
                "recognize there is no way in and call request_assistance."
            ),
            task="deliver box_03 to shelf_C in room_C (fully walled off)",
            scene=_sealed_room_scene(),
            expected_outcome="escalated",
        ),
    ]
