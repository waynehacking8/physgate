"""Tests for multi-agent cooperative handoff (Phase 4).

Tests cover:
- partial_success outcome when request_assistance is called
- handoff_request populated in orchestrator state
- cooperative delivery demo runs correctly
- handoff accuracy: agent triggers handoff at the right time
"""

from physgate.executor.backend import MockWorldBackend
from physgate.executor.plan_executor import execute_plan
from physgate.gate.parallel import run_gate, symbolic_l2
from physgate.gate.schemas import Scene, SceneObject
from physgate.orchestrator.graph import (
    OrchestratorConfig,
    build_orchestrator,
    run_task,
)
from physgate.planner.critic import MockCritic
from physgate.planner.schemas import Plan, PlanStep, RelationChange, ToolName


def _handoff_scene() -> Scene:
    return Scene(
        objects=[
            SceneObject(id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="go2", label="robot"),
        ],
        relations=[("box_03", "on", "floor_01")],
        gripper_empty=True,
    )


def _handoff_plan() -> Plan:
    return Plan(
        plan_id="handoff_plan",
        task="deliver then handoff",
        steps=[
            PlanStep(
                step_id=1, tool=ToolName.REQUEST_ASSISTANCE,
                args={"message": "task too complex for single agent"},
            ),
        ],
    )


def _mixed_plan() -> Plan:
    """A plan that does some work then requests assistance."""
    return Plan(
        plan_id="mixed_plan",
        task="partial work then handoff",
        steps=[
            PlanStep(
                step_id=1, tool=ToolName.MOVE_TO_POSE,
                args={"target": "box_03", "standoff_m": 0.3, "speed": 0.5},
                preconditions=["box_03 exists"],
                effects=[RelationChange(op="add", subject="robot", predicate="near", object="box_03")],
            ),
            PlanStep(
                step_id=2, tool=ToolName.INSPECT_OBJECT,
                args={"object_id": "box_03"},
            ),
            PlanStep(
                step_id=3, tool=ToolName.REQUEST_ASSISTANCE,
                args={"message": "inspected object, need help with delivery"},
            ),
        ],
    )


class TestPartialSuccessOutcome:
    def test_request_assistance_yields_partial_success(self):
        scene = _handoff_scene()

        def planner_fn(task, _scene, n, feedback):
            return [_handoff_plan()]

        backend = MockWorldBackend(scene)

        def executor_fn(plan, _scene):
            return execute_plan(plan, backend)

        def gate_fn(plans, _scene):
            return run_gate(plans, _scene, l2_fn=symbolic_l2)

        graph = build_orchestrator(
            planner_fn=planner_fn,
            critic_fn=MockCritic(),
            gate_fn=gate_fn,
            executor_fn=executor_fn,
            approval_fn=lambda sel: True,
            config=OrchestratorConfig(num_candidates=1),
        )
        state = run_task(graph, task="deliver box", scene=scene)
        assert state["outcome"] == "partial_success"
        assert "partial_success" in state["trace"]

    def test_handoff_request_populated(self):
        scene = _handoff_scene()

        def planner_fn(task, _scene, n, feedback):
            return [_handoff_plan()]

        backend = MockWorldBackend(scene)

        def executor_fn(plan, _scene):
            return execute_plan(plan, backend)

        def gate_fn(plans, _scene):
            return run_gate(plans, _scene, l2_fn=symbolic_l2)

        graph = build_orchestrator(
            planner_fn=planner_fn,
            critic_fn=MockCritic(),
            gate_fn=gate_fn,
            executor_fn=executor_fn,
            approval_fn=lambda sel: True,
            config=OrchestratorConfig(num_candidates=1),
        )
        state = run_task(graph, task="deliver box", scene=scene)
        assert state.get("handoff_request") is not None
        assert "complex" in state["handoff_request"]

    def test_mixed_plan_partial_success(self):
        scene = _handoff_scene()

        def planner_fn(task, _scene, n, feedback):
            return [_mixed_plan()]

        backend = MockWorldBackend(scene)

        def executor_fn(plan, _scene):
            return execute_plan(plan, backend)

        def gate_fn(plans, _scene):
            return run_gate(plans, _scene, l2_fn=symbolic_l2)

        graph = build_orchestrator(
            planner_fn=planner_fn,
            critic_fn=MockCritic(),
            gate_fn=gate_fn,
            executor_fn=executor_fn,
            approval_fn=lambda sel: True,
            config=OrchestratorConfig(num_candidates=1),
        )
        state = run_task(graph, task="deliver box", scene=scene)
        assert state["outcome"] == "partial_success"
        assert "inspected" in state["handoff_request"]

    def test_partial_success_distinguished_from_escalated(self):
        """partial_success (intentional handoff) vs escalated (budget exhaustion)."""
        scene = _handoff_scene()

        def handoff_planner(task, _scene, n, feedback):
            return [_handoff_plan()]

        def empty_planner(task, _scene, n, feedback):
            return []

        backend = MockWorldBackend(scene)

        def executor_fn(plan, _scene):
            return execute_plan(plan, backend)

        def gate_fn(plans, _scene):
            return run_gate(plans, _scene, l2_fn=symbolic_l2)

        critic = MockCritic()

        # handoff → partial_success
        g1 = build_orchestrator(
            planner_fn=handoff_planner, critic_fn=critic,
            gate_fn=gate_fn, executor_fn=executor_fn,
            approval_fn=lambda sel: True,
            config=OrchestratorConfig(num_candidates=1),
        )
        s1 = run_task(g1, task="a", scene=scene)

        # empty plans → escalated
        g2 = build_orchestrator(
            planner_fn=empty_planner, critic_fn=critic,
            gate_fn=gate_fn, executor_fn=executor_fn,
            approval_fn=lambda sel: True,
            config=OrchestratorConfig(num_candidates=1, max_replans=0),
        )
        s2 = run_task(g2, task="b", scene=scene)

        assert s1["outcome"] == "partial_success"
        assert s2["outcome"] == "escalated"


class TestCooperativeDeliveryDemo:
    def test_demo_runs_successfully(self):
        from examples.cooperative_delivery import run_cooperative_delivery
        result = run_cooperative_delivery()
        assert result["agent1_outcome"] == "partial_success"
        assert result["agent1_handoff"] is not None
        assert result["agent2_outcome"] == "done"
        assert result["box_delivered"] is True
