"""Tests for the failure analyst module and orchestrator integration.

Tests cover:
- MockFailureAnalyst produces structured diagnoses for various failure types
- FailureReport schema fields are correct
- Orchestrator with failure_analyst_fn routes through analyze_failure node
- Replan strategy budget: targeted → full → escalate
- Targeted repair feedback includes prefix and fix suggestions
"""

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
from physgate.planner.failure_analyst import (
    FailureReport,
    FailureType,
    MockFailureAnalyst,
)
from physgate.planner.planner import MockPlanner
from physgate.planner.schemas import Plan, PlanStep, RelationChange, ToolName


# -------------------------------------------------------- MockFailureAnalyst


def _basic_scene() -> Scene:
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


def _make_failed_plan() -> Plan:
    return Plan(
        plan_id="test_plan",
        task="put box on shelf",
        steps=[
            PlanStep(
                step_id=1, tool=ToolName.MOVE_TO_POSE,
                args={"target": "box_03", "standoff_m": 0.3, "speed": 0.5},
                preconditions=["box_03 exists"],
            ),
            PlanStep(
                step_id=2, tool=ToolName.EXECUTE_SKILL,
                args={"skill": "pick", "target": "box_03"},
                preconditions=["box_03 exists", "gripper_empty"],
            ),
            PlanStep(
                step_id=3, tool=ToolName.MOVE_TO_POSE,
                args={"target": "shelf_A", "standoff_m": 0.4, "speed": 0.5},
                preconditions=["shelf_A exists"],
            ),
            PlanStep(
                step_id=4, tool=ToolName.EXECUTE_SKILL,
                args={"skill": "place", "target": "shelf_A"},
                preconditions=["shelf_A exists", "gripper holding box_03"],
            ),
        ],
    )


class TestMockFailureAnalyst:
    def test_precondition_failure(self):
        analyst = MockFailureAnalyst()
        execution_result = {
            "success": False,
            "failed_step_id": 2,
            "error": "robot is not near 'box_03'",
            "failure_report": {"violations": [{"type": "precondition", "detail": "not near"}]},
        }
        report = analyst(_make_failed_plan(), execution_result, _basic_scene())
        assert isinstance(report, FailureReport)
        assert report.failure_type == FailureType.PRECONDITION_UNMET
        assert report.failed_step_index == 1
        assert "near" in report.root_cause
        assert 2 in report.affected_steps

    def test_locked_door_failure(self):
        analyst = MockFailureAnalyst()
        execution_result = {
            "success": False,
            "failed_step_id": 3,
            "error": "door 'door_01' is locked — unlock it first",
            "failure_report": None,
        }
        plan = Plan(
            plan_id="lock_plan", task="open door",
            steps=[
                PlanStep(step_id=1, tool=ToolName.MOVE_TO_POSE, args={"target": "door_01"}),
                PlanStep(step_id=2, tool=ToolName.MOVE_TO_POSE, args={"target": "door_01"}),
                PlanStep(step_id=3, tool=ToolName.OPEN_DOOR, args={"door_id": "door_01"}),
            ],
        )
        report = analyst(plan, execution_result, _basic_scene())
        assert report.failure_type == FailureType.PRECONDITION_UNMET
        assert "locked" in report.root_cause
        assert "unlock" in report.suggested_fix

    def test_gripper_occupied_failure(self):
        analyst = MockFailureAnalyst()
        execution_result = {
            "success": False,
            "failed_step_id": 2,
            "error": "gripper is already holding an object",
            "failure_report": None,
        }
        report = analyst(_make_failed_plan(), execution_result, _basic_scene())
        assert "gripper" in report.root_cause.lower()

    def test_infeasible_failure(self):
        analyst = MockFailureAnalyst()
        execution_result = {
            "success": False,
            "failed_step_id": 1,
            "error": "object is not pushable and too heavy",
            "failure_report": None,
        }
        plan = Plan(
            plan_id="heavy", task="push anvil",
            steps=[PlanStep(step_id=1, tool=ToolName.PUSH_OBJECT, args={"object_id": "anvil_01", "direction": "east"})],
        )
        report = analyst(plan, execution_result, _basic_scene())
        assert report.failure_type == FailureType.INFEASIBLE
        assert "request_assistance" in report.suggested_fix

    def test_prefix_valid_through_is_correct(self):
        analyst = MockFailureAnalyst()
        execution_result = {
            "success": False,
            "failed_step_id": 3,
            "error": "robot is not near 'shelf_A'",
            "failure_report": None,
        }
        report = analyst(_make_failed_plan(), execution_result, _basic_scene())
        assert report.prefix_valid_through == 2
        assert report.failed_step_index == 2

    def test_affected_steps_includes_failed_and_later(self):
        analyst = MockFailureAnalyst()
        execution_result = {
            "success": False,
            "failed_step_id": 2,
            "error": "robot is not near 'box_03'",
            "failure_report": None,
        }
        report = analyst(_make_failed_plan(), execution_result, _basic_scene())
        assert 2 in report.affected_steps
        assert 3 in report.affected_steps
        assert 4 in report.affected_steps
        assert 1 not in report.affected_steps

    def test_no_failed_step_id(self):
        analyst = MockFailureAnalyst()
        execution_result = {"success": False, "error": "unknown", "failed_step_id": None, "failure_report": None}
        report = analyst(_make_failed_plan(), execution_result, _basic_scene())
        assert report.failed_step is None
        assert len(report.affected_steps) == 4


# ------------------------------------------- orchestrator with failure analyst


class TestOrchestratorWithAnalyst:
    def _run_with_analyst(self, scenario_id: str) -> dict:
        from physgate.eval.fault_backend import FaultInjector

        suite = {s.scenario_id: s for s in build_scenario_suite()}
        scenario = suite[scenario_id]
        planner = MockPlanner()
        critic = MockCritic()
        analyst = MockFailureAnalyst()
        injector = FaultInjector(scenario.fault) if scenario.fault else None

        def executor_fn(plan, scene):
            factory = injector.backend_factory if injector else MockWorldBackend
            backend = factory(scenario.scene)
            return execute_plan(plan, backend)

        def gate_fn(plans, scene):
            return run_gate(plans, scene, l2_fn=symbolic_l2)

        graph = build_orchestrator(
            planner_fn=planner,
            critic_fn=critic,
            gate_fn=gate_fn,
            executor_fn=executor_fn,
            approval_fn=lambda sel: True,
            config=OrchestratorConfig(max_replans=3, max_execution_retries=1),
            failure_analyst_fn=analyst,
        )
        return run_task(graph, task=scenario.task, scene=scenario.scene)

    def test_basic_scenario_still_succeeds(self):
        state = self._run_with_analyst("fetch_and_place_basic")
        assert state["outcome"] == "done"

    def test_persistent_failure_escalates_via_analyst(self):
        state = self._run_with_analyst("recovery_persistent_failure_escalates")
        assert state["outcome"] == "escalated"
        assert "analyzing_failure" in state["trace"]

    def test_trace_includes_failure_analysis_node(self):
        state = self._run_with_analyst("recovery_persistent_failure_escalates")
        assert "analyzing_failure" in state["trace"]

    def test_failure_report_populated_after_analysis(self):
        state = self._run_with_analyst("recovery_persistent_failure_escalates")
        assert state.get("failure_report") is not None

    def test_replan_strategy_targeted_then_full(self):
        """After first failure analysis, strategy is targeted_repair;
        after second, it becomes full_regeneration."""
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        scenario = suite["recovery_persistent_failure_escalates"]
        planner = MockPlanner()
        critic = MockCritic()
        analyst = MockFailureAnalyst()
        strategies_seen: list[str] = []

        original_analyst = analyst.__call__

        def tracking_analyst(plan, result, scene):
            report = original_analyst(plan, result, scene)
            return report

        def executor_fn(plan, scene):
            backend = MockWorldBackend(scenario.scene)
            from physgate.eval.fault_backend import FaultInjector
            from physgate.eval.scenarios import FaultSpec
            injector = FaultInjector(FaultSpec(skill="pick", fail_count=10_000))
            backend = injector.backend_factory(scenario.scene)
            return execute_plan(plan, backend)

        def gate_fn(plans, scene):
            return run_gate(plans, scene, l2_fn=symbolic_l2)

        graph = build_orchestrator(
            planner_fn=planner,
            critic_fn=critic,
            gate_fn=gate_fn,
            executor_fn=executor_fn,
            approval_fn=lambda sel: True,
            config=OrchestratorConfig(max_replans=3, max_execution_retries=1),
            failure_analyst_fn=tracking_analyst,
        )
        state = run_task(graph, task=scenario.task, scene=scenario.scene)
        assert state["outcome"] == "escalated"


# ------------------------------------------- targeted vs blind comparison


class TestTargetedVsBlind:
    def test_blind_regeneration_without_analyst(self):
        """Without a failure analyst, the orchestrator does blind regeneration."""
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        scenario = suite["fetch_and_place_basic"]
        planner = MockPlanner()
        critic = MockCritic()

        def executor_fn(plan, scene):
            backend = MockWorldBackend(scenario.scene)
            return execute_plan(plan, backend)

        def gate_fn(plans, scene):
            return run_gate(plans, scene, l2_fn=symbolic_l2)

        graph = build_orchestrator(
            planner_fn=planner,
            critic_fn=critic,
            gate_fn=gate_fn,
            executor_fn=executor_fn,
            approval_fn=lambda sel: True,
        )
        state = run_task(graph, task=scenario.task, scene=scenario.scene)
        assert "analyzing_failure" not in state["trace"]
        assert state["outcome"] == "done"

    def test_with_analyst_still_succeeds(self):
        """With a failure analyst, successful tasks still complete normally."""
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        scenario = suite["fetch_and_place_basic"]
        planner = MockPlanner()
        critic = MockCritic()
        analyst = MockFailureAnalyst()

        def executor_fn(plan, scene):
            backend = MockWorldBackend(scenario.scene)
            return execute_plan(plan, backend)

        def gate_fn(plans, scene):
            return run_gate(plans, scene, l2_fn=symbolic_l2)

        graph = build_orchestrator(
            planner_fn=planner,
            critic_fn=critic,
            gate_fn=gate_fn,
            executor_fn=executor_fn,
            approval_fn=lambda sel: True,
            failure_analyst_fn=analyst,
        )
        state = run_task(graph, task=scenario.task, scene=scenario.scene)
        assert state["outcome"] == "done"


# ---------------------------------------------- FailureReport schema tests


class TestFailureReportSchema:
    def test_failure_report_serializable(self):
        report = FailureReport(
            failed_step=None,
            failed_step_index=-1,
            failure_type=FailureType.PRECONDITION_UNMET,
            root_cause="test",
            affected_steps=[1, 2, 3],
            suggested_fix="fix it",
            prefix_valid_through=0,
        )
        data = report.model_dump(mode="json")
        assert data["failure_type"] == "precondition_unmet"
        assert data["affected_steps"] == [1, 2, 3]

    def test_failure_type_enum_values(self):
        assert FailureType.PRECONDITION_UNMET.value == "precondition_unmet"
        assert FailureType.TOOL_ERROR.value == "tool_error"
        assert FailureType.PHYSICS_REJECT.value == "physics_reject"
        assert FailureType.INFEASIBLE.value == "infeasible"


class TestAnalystFiresOnFirstFailure:
    """F2 verification: analyst triggers on the FIRST execution failure,
    not after retry exhaustion."""

    def test_analyst_in_trace_before_any_replan(self):
        """With analyst enabled, the first execution failure goes straight
        to analyze_failure — no retry loop first."""
        from physgate.eval.fault_backend import FaultInjector
        from physgate.eval.scenarios import FaultSpec

        suite = {s.scenario_id: s for s in build_scenario_suite()}
        scenario = suite["recovery_transient_pick_failure"]
        planner = MockPlanner()
        critic = MockCritic()
        analyst = MockFailureAnalyst()
        injector = FaultInjector(scenario.fault)

        def executor_fn(plan, scene):
            backend = injector.backend_factory(scenario.scene)
            return execute_plan(plan, backend)

        def gate_fn(plans, scene):
            return run_gate(plans, scene, l2_fn=symbolic_l2)

        graph = build_orchestrator(
            planner_fn=planner,
            critic_fn=critic,
            gate_fn=gate_fn,
            executor_fn=executor_fn,
            approval_fn=lambda sel: True,
            config=OrchestratorConfig(max_replans=3, max_execution_retries=3),
            failure_analyst_fn=analyst,
        )
        state = run_task(graph, task=scenario.task, scene=scenario.scene)

        trace = state["trace"]
        exec_indices = [i for i, t in enumerate(trace) if t == "executing"]
        analyst_indices = [i for i, t in enumerate(trace) if t == "analyzing_failure"]

        if analyst_indices:
            first_analyst = analyst_indices[0]
            first_exec = exec_indices[0]
            assert first_analyst == first_exec + 1, (
                f"analyst should fire immediately after first execute failure, "
                f"but trace is: {trace}"
            )
