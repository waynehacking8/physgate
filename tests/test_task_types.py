"""Tests for T2-T6 task types and their scenario definitions.

Verifies that:
- Each new task type has at least one feasible and one infeasible variant
- Scenario scenes are well-formed (objects exist, relations valid)
- MockPlanner detects task types correctly
- The scenario suite has the expected count and coverage
"""

from physgate.eval.scenarios import build_scenario_suite
from physgate.planner.planner import MockPlanner
from physgate.planner.schemas import ToolName


# -------------------------------------------------------- scenario suite


def test_suite_has_18_scenarios():
    assert len(build_scenario_suite()) == 18


def test_suite_ids_unique():
    ids = [s.scenario_id for s in build_scenario_suite()]
    assert len(ids) == len(set(ids))


def test_suite_has_feasible_and_infeasible_for_each_new_category():
    suite = build_scenario_suite()
    categories_with_escalated = {
        s.category for s in suite if s.expected_outcome == "escalated"
    }
    assert "locked_door" in categories_with_escalated
    assert "assistance" in categories_with_escalated


def test_suite_t2_through_t6_present():
    ids = {s.scenario_id for s in build_scenario_suite()}
    expected = {
        "t2_locked_door_delivery", "t2_locked_door_no_key_escalate",
        "t3_blocked_path_push", "t3_blocked_path_immovable",
        "t4_sequential_delivery",
        "t5_elevator_delivery",
        "t6_too_heavy", "t6_sealed_room",
    }
    assert expected.issubset(ids)


# -------------------------------------------------------- MockPlanner detection


class TestMockPlannerTaskDetection:
    def test_detects_locked_door(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        planner = MockPlanner()
        task_type = planner._detect_task_type(
            suite["t2_locked_door_delivery"].scene, ""
        )
        assert task_type == "locked_door"

    def test_detects_blocked_path(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        task_type = MockPlanner._detect_task_type(
            suite["t3_blocked_path_push"].scene, ""
        )
        assert task_type == "blocked_path"

    def test_detects_immovable_block_as_infeasible(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        task_type = MockPlanner._detect_task_type(
            suite["t3_blocked_path_immovable"].scene, ""
        )
        assert task_type == "infeasible_blocked"

    def test_detects_elevator(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        task_type = MockPlanner._detect_task_type(
            suite["t5_elevator_delivery"].scene, ""
        )
        assert task_type == "elevator"

    def test_detects_heavy_infeasible(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        task_type = MockPlanner._detect_task_type(
            suite["t6_too_heavy"].scene, ""
        )
        assert task_type == "infeasible_heavy"

    def test_detects_sealed_infeasible(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        task_type = MockPlanner._detect_task_type(
            suite["t6_sealed_room"].scene, ""
        )
        assert task_type == "infeasible_sealed"

    def test_detects_sequential(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        task_type = MockPlanner._detect_task_type(
            suite["t4_sequential_delivery"].scene, ""
        )
        assert task_type == "sequential"


# --------------------------------------------------- MockPlanner plan generation


class TestMockPlannerGeneration:
    def test_locked_door_plans_use_unlock_and_open(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        s = suite["t2_locked_door_delivery"]
        planner = MockPlanner()
        plans = planner(s.task, s.scene, 2)
        assert len(plans) == 2
        tools_used = {step.tool for p in plans for step in p.steps}
        assert ToolName.UNLOCK_DOOR in tools_used
        assert ToolName.OPEN_DOOR in tools_used

    def test_blocked_path_plans_use_push(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        s = suite["t3_blocked_path_push"]
        planner = MockPlanner()
        plans = planner(s.task, s.scene, 2)
        tools_used = {step.tool for p in plans for step in p.steps}
        assert ToolName.PUSH_OBJECT in tools_used
        assert ToolName.INSPECT_OBJECT in tools_used

    def test_elevator_plans_use_call_elevator(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        s = suite["t5_elevator_delivery"]
        planner = MockPlanner()
        plans = planner(s.task, s.scene, 2)
        tools_used = {step.tool for p in plans for step in p.steps}
        assert ToolName.CALL_ELEVATOR in tools_used

    def test_infeasible_plans_use_request_assistance(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        s = suite["t6_too_heavy"]
        planner = MockPlanner()
        plans = planner(s.task, s.scene, 2)
        tools_used = {step.tool for p in plans for step in p.steps}
        assert ToolName.REQUEST_ASSISTANCE in tools_used
        assert ToolName.INSPECT_OBJECT in tools_used

    def test_locked_door_no_key_returns_empty(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        s = suite["t2_locked_door_no_key_escalate"]
        planner = MockPlanner()
        plans = planner(s.task, s.scene, 4)
        assert plans == []

    def test_sequential_plans_handle_multiple_objects(self):
        suite = {s.scenario_id: s for s in build_scenario_suite()}
        s = suite["t4_sequential_delivery"]
        planner = MockPlanner()
        plans = planner(s.task, s.scene, 2)
        assert len(plans) == 2
        for plan in plans:
            pick_targets = {
                step.args["target"]
                for step in plan.steps
                if step.tool == ToolName.EXECUTE_SKILL and step.args.get("skill") == "pick"
            }
            assert len(pick_targets) >= 2


# -------------------------------------------------------- scene validation


class TestSceneIntegrity:
    def test_all_scenarios_have_valid_scenes(self):
        for s in build_scenario_suite():
            assert len(s.scene.objects) >= 2, f"{s.scenario_id}: too few objects"
            obj_ids = {o.id for o in s.scene.objects}
            for subj, pred, obj in s.scene.relations:
                assert subj in obj_ids or subj == "gripper", (
                    f"{s.scenario_id}: relation subject '{subj}' not in scene"
                )

    def test_all_feasible_scenarios_have_required_relations(self):
        for s in build_scenario_suite():
            if s.expected_outcome == "done":
                assert s.required_final_relations, (
                    f"{s.scenario_id}: feasible scenario must have required_final_relations"
                )
