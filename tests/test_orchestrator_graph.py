"""Tests for the orchestrator LangGraph state machine (A5).

The orchestrator wires planner -> critic -> gate -> approval -> executor with
the failure-recovery loops from architecture doc §2 (replan budget 2, execution
retry budget 3). All components are injected callables, mocked here — no GPU,
no LLM, no simulator.
"""

from physgate.gate.scoring import PhysicsResult, SelectionResult, select_best
from physgate.gate.schemas import Scene, SceneObject
from physgate.orchestrator.graph import OrchestratorConfig, build_orchestrator, run_task
from physgate.planner.schemas import Plan, PlanStep, ToolName


# ------------------------------------------------------------------- fixtures


def _scene() -> Scene:
    return Scene(
        objects=[
            SceneObject(id="box_03", label="cardboard_box", is_anomaly=True),
            SceneObject(id="shelf_A", label="shelf"),
        ],
        relations=[("box_03", "on", "floor_01")],
    )


def _plan(plan_id: str) -> Plan:
    return Plan(
        plan_id=plan_id,
        task="put the box back",
        steps=[PlanStep(step_id=1, tool=ToolName.EXECUTE_SKILL, args={"skill": "pick"})],
    )


def _mock_planner(n_plans: int = 8):
    """Planner that returns n fresh candidate plans, counting invocations."""
    calls = []

    def planner_fn(task, scene, n, feedback=None):
        calls.append({"task": task, "feedback": feedback})
        round_no = len(calls)
        return [_plan(f"r{round_no}_p{i}") for i in range(n_plans)]

    planner_fn.calls = calls
    return planner_fn


def _passthrough_critic(plans, scene):
    return list(plans)


def _rejecting_critic(plans, scene):
    return []


def _all_pass_gate(plans, scene):
    results = [
        PhysicsResult(plan_id=p.plan_id, success=True, collision_count=i, completion_time_s=10.0)
        for i, p in enumerate(plans)
    ]
    return select_best(results)


def _all_fail_gate(plans, scene):
    results = [PhysicsResult(plan_id=p.plan_id, success=False) for p in plans]
    return select_best(results)


def _ok_executor(plan, scene):
    return {"success": True, "steps_completed": len(plan.steps)}


def _failing_executor(plan, scene):
    return {"success": False, "steps_completed": 0, "error": "actuator fault"}


def _auto_approve(selection):
    return True


def _auto_deny(selection):
    return False


def _run(planner, critic, gate, executor, approval=_auto_approve, config=None):
    graph = build_orchestrator(
        planner_fn=planner,
        critic_fn=critic,
        gate_fn=gate,
        executor_fn=executor,
        approval_fn=approval,
        config=config or OrchestratorConfig(),
    )
    return run_task(graph, task="put the fallen box back on shelf A", scene=_scene())


# ----------------------------------------------------------------- happy path


def test_happy_path_reaches_done():
    planner = _mock_planner()
    final = _run(planner, _passthrough_critic, _all_pass_gate, _ok_executor)
    assert final["outcome"] == "done"
    # best plan is the one with 0 collisions => first candidate of round 1
    assert final["selection"].best_plan_id == "r1_p0"
    assert final["execution_result"]["success"] is True
    # planner called exactly once, with no failure feedback
    assert len(planner.calls) == 1
    assert planner.calls[0]["feedback"] is None


def test_planner_receives_n_candidates_request():
    planner = _mock_planner()
    config = OrchestratorConfig(num_candidates=8)
    final = _run(planner, _passthrough_critic, _all_pass_gate, _ok_executor, config=config)
    assert len(final["candidates"]) == 8


def test_state_trace_records_phases():
    final = _run(_mock_planner(), _passthrough_critic, _all_pass_gate, _ok_executor)
    assert final["trace"] == ["planning", "reviewing", "validating", "awaiting_approval", "executing", "done"]


# ------------------------------------------------------------- recovery loops


def test_critic_rejects_all_triggers_replan_then_succeeds():
    """Round 1: critic rejects everything. Round 2: critic passes them through."""
    rounds = {"n": 0}

    def flaky_critic(plans, scene):
        rounds["n"] += 1
        return [] if rounds["n"] == 1 else list(plans)

    planner = _mock_planner()
    final = _run(planner, flaky_critic, _all_pass_gate, _ok_executor)
    assert final["outcome"] == "done"
    assert len(planner.calls) == 2  # replanned once
    # second planning round got failure feedback
    assert planner.calls[1]["feedback"] is not None


def test_gate_all_infeasible_replans_with_feedback():
    gates = {"n": 0}

    def flaky_gate(plans, scene):
        gates["n"] += 1
        return _all_fail_gate(plans, scene) if gates["n"] == 1 else _all_pass_gate(plans, scene)

    planner = _mock_planner()
    final = _run(planner, _passthrough_critic, flaky_gate, _ok_executor)
    assert final["outcome"] == "done"
    assert len(planner.calls) == 2
    assert "fail" in planner.calls[1]["feedback"].lower()


def test_replan_budget_exhausted_escalates():
    planner = _mock_planner()
    final = _run(planner, _passthrough_critic, _all_fail_gate, _ok_executor)
    assert final["outcome"] == "escalated"
    # initial plan + 2 replans = 3 planner calls, then give up
    assert len(planner.calls) == 3


def test_execution_transient_failure_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky_executor(plan, scene):
        attempts["n"] += 1
        return {"success": attempts["n"] >= 3, "steps_completed": 0}

    final = _run(_mock_planner(), _passthrough_critic, _all_pass_gate, flaky_executor)
    assert final["outcome"] == "done"
    assert attempts["n"] == 3  # failed twice, succeeded on third


def test_execution_retry_budget_exhausted_then_replan():
    """Persistent execution failure: 3 retries -> replan -> still fails -> escalate."""
    planner = _mock_planner()
    final = _run(planner, _passthrough_critic, _all_pass_gate, _failing_executor)
    assert final["outcome"] == "escalated"
    assert len(planner.calls) == 3  # initial + 2 replans


# ------------------------------------------------------------------- approval


def test_approval_denied_terminates():
    final = _run(
        _mock_planner(), _passthrough_critic, _all_pass_gate, _ok_executor, approval=_auto_deny
    )
    assert final["outcome"] == "denied"
    assert final["execution_result"] is None
