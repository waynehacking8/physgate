"""Cooperative delivery demo: agent_1 clears path → handoff → agent_2 delivers.

Demonstrates multi-agent cooperative handoff (AGENT_UPGRADE.md §4):
1. Agent 1 encounters a blocked path, pushes the obstacle, then signals
   request_assistance because it cannot carry the box (e.g., low battery).
2. The orchestrator detects the partial_success + handoff_request.
3. Agent 2 is spawned with the handoff context and completes the delivery.

Run: python examples/cooperative_delivery.py
"""

from __future__ import annotations

from physgate.eval.scenarios import build_scenario_suite
from physgate.executor.backend import MockWorldBackend
from physgate.executor.plan_executor import execute_plan
from physgate.gate.parallel import run_gate, symbolic_l2
from physgate.gate.schemas import Scene, SceneObject
from physgate.gate.scoring import SelectionResult
from physgate.orchestrator.graph import (
    OrchestratorConfig,
    build_orchestrator,
    run_task,
)
from physgate.planner.critic import MockCritic
from physgate.planner.schemas import Plan, PlanStep, RelationChange, ToolName


def _cooperative_scene() -> Scene:
    """A scene where two agents must cooperate: one clears, one delivers."""
    return Scene(
        objects=[
            SceneObject(
                id="box_03", label="cardboard_box",
                affordances=["graspable"], is_anomaly=True,
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


def _agent1_planner(task: str, scene: Scene, n: int, feedback: str | None = None) -> list[Plan]:
    """Agent 1: clears the path then requests assistance for delivery."""
    return [Plan(
        plan_id="agent1_clear_and_handoff",
        task=task,
        rationale="clear the path then hand off delivery to agent 2",
        steps=[
            PlanStep(
                step_id=1, tool=ToolName.INSPECT_OBJECT,
                args={"object_id": "crate_01"},
            ),
            PlanStep(
                step_id=2, tool=ToolName.MOVE_TO_POSE,
                args={"target": "crate_01", "standoff_m": 0.3, "speed": 0.5},
                preconditions=["crate_01 exists"],
                effects=[RelationChange(op="add", subject="robot", predicate="near", object="crate_01")],
            ),
            PlanStep(
                step_id=3, tool=ToolName.PUSH_OBJECT,
                args={"object_id": "crate_01", "direction": "east"},
                preconditions=["crate_01 exists", "robot near crate_01"],
            ),
            PlanStep(
                step_id=4, tool=ToolName.REQUEST_ASSISTANCE,
                args={"message": "path cleared; need another agent to deliver box_03 to shelf_A"},
            ),
        ],
    )]


def _agent2_planner(task: str, scene: Scene, n: int, feedback: str | None = None) -> list[Plan]:
    """Agent 2: delivers the box after agent 1 cleared the path."""
    return [Plan(
        plan_id="agent2_deliver",
        task=task,
        rationale="deliver box to shelf (path already cleared by agent 1)",
        steps=[
            PlanStep(
                step_id=1, tool=ToolName.MOVE_TO_POSE,
                args={"target": "box_03", "standoff_m": 0.3, "speed": 0.5},
                preconditions=["box_03 exists"],
                effects=[RelationChange(op="add", subject="robot", predicate="near", object="box_03")],
            ),
            PlanStep(
                step_id=2, tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=["box_03 exists", "gripper_empty"],
                effects=[RelationChange(op="add", subject="gripper", predicate="holding", object="box_03")],
            ),
            PlanStep(
                step_id=3, tool=ToolName.MOVE_TO_POSE,
                args={"target": "shelf_A", "standoff_m": 0.4, "speed": 0.5},
                preconditions=["shelf_A exists"],
                effects=[RelationChange(op="add", subject="robot", predicate="near", object="shelf_A")],
            ),
            PlanStep(
                step_id=4, tool=ToolName.EXECUTE_SKILL,
                args={"skill": "place", "target": "shelf_A"},
                preconditions=["shelf_A exists", "gripper holding box_03"],
                effects=[
                    RelationChange(op="remove", subject="gripper", predicate="holding", object="box_03"),
                    RelationChange(op="add", subject="box_03", predicate="on", object="shelf_A"),
                ],
            ),
        ],
    )]


def run_cooperative_delivery() -> dict:
    """Run the two-agent cooperative delivery demo."""
    scene = _cooperative_scene()
    critic = MockCritic()

    # --- Agent 1: clear the path ---
    print("=== Agent 1: clearing path ===")

    backend1 = MockWorldBackend(scene)

    def executor1(plan, _scene):
        return execute_plan(plan, backend1)

    def gate_fn(plans, _scene):
        return run_gate(plans, _scene, l2_fn=symbolic_l2)

    graph1 = build_orchestrator(
        planner_fn=_agent1_planner,
        critic_fn=critic,
        gate_fn=gate_fn,
        executor_fn=executor1,
        approval_fn=lambda sel: True,
        config=OrchestratorConfig(num_candidates=1, max_replans=0),
    )
    state1 = run_task(graph1, task="clear path for delivery", scene=scene)
    print(f"  outcome: {state1['outcome']}")
    print(f"  handoff: {state1.get('handoff_request')}")
    print(f"  trace: {state1['trace']}")

    assert state1["outcome"] == "partial_success", f"expected partial_success, got {state1['outcome']}"
    assert state1.get("handoff_request") is not None

    # --- Agent 2: deliver the box (using agent 1's modified scene) ---
    print("\n=== Agent 2: delivering box ===")

    scene2 = backend1.get_scene()

    backend2 = MockWorldBackend(scene2)

    def executor2(plan, _scene):
        return execute_plan(plan, backend2)

    graph2 = build_orchestrator(
        planner_fn=_agent2_planner,
        critic_fn=critic,
        gate_fn=gate_fn,
        executor_fn=executor2,
        approval_fn=lambda sel: True,
        config=OrchestratorConfig(num_candidates=1, max_replans=0),
    )
    state2 = run_task(graph2, task="deliver box_03 to shelf_A", scene=scene2)
    print(f"  outcome: {state2['outcome']}")
    print(f"  trace: {state2['trace']}")

    final_scene = backend2.get_scene()
    box_on_shelf = final_scene.has_relation("box_03", "on", "shelf_A")
    print(f"\n=== Result: box_03 on shelf_A = {box_on_shelf} ===")

    return {
        "agent1_outcome": state1["outcome"],
        "agent1_handoff": state1.get("handoff_request"),
        "agent2_outcome": state2["outcome"],
        "box_delivered": box_on_shelf,
    }


if __name__ == "__main__":
    result = run_cooperative_delivery()
    print(f"\nFinal: {result}")
