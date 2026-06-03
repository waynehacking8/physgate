"""Tests for the human-approval gate via LangGraph interrupt (#4 part A).

The architecture (doc §1, §5) specifies approval as an *orchestrator interrupt*:
the graph pauses after gate selection, a human inspects the selection, and the
graph resumes with the verdict. This replaces the demo's auto-approve callable.
"""

from langgraph.checkpoint.memory import MemorySaver

from physgate.gate.scoring import PhysicsResult, select_best
from physgate.gate.schemas import Scene, SceneObject
from physgate.orchestrator.graph import (
    HUMAN_APPROVAL,
    OrchestratorConfig,
    build_orchestrator,
    resume_with_approval,
    run_task,
)
from physgate.planner.schemas import Plan, PlanStep, ToolName


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


def _planner(task, scene, n, feedback=None):
    return [_plan(f"p{i}") for i in range(n)]


def _critic(plans, scene):
    return list(plans)


def _gate(plans, scene):
    results = [
        PhysicsResult(plan_id=p.plan_id, success=True, collision_count=i) for i, p in enumerate(plans)
    ]
    return select_best(results)


def _executor(plan, scene):
    return {"success": True, "steps_completed": len(plan.steps)}


def _build_interrupt_graph():
    return build_orchestrator(
        planner_fn=_planner,
        critic_fn=_critic,
        gate_fn=_gate,
        executor_fn=_executor,
        approval_fn=HUMAN_APPROVAL,
        config=OrchestratorConfig(),
        checkpointer=MemorySaver(),
    )


# ------------------------------------------------------------------ interrupt


def test_human_approval_pauses_before_execution():
    graph = _build_interrupt_graph()
    state = run_task(graph, task="put the box back", scene=_scene(), thread_id="t1")

    # the graph paused at the approval gate: selection made, nothing executed
    assert state.get("__interrupt__") is not None
    assert state["execution_result"] is None
    assert state["outcome"] == ""


def test_interrupt_payload_describes_selection():
    graph = _build_interrupt_graph()
    state = run_task(graph, task="put the box back", scene=_scene(), thread_id="t2")

    interrupts = state["__interrupt__"]
    payload = interrupts[0].value
    assert payload["best_plan_id"] == "p0"
    assert "rationale" in payload
    assert payload["num_candidates"] == 8
    assert "scores" in payload


def test_resume_approved_runs_execution():
    graph = _build_interrupt_graph()
    run_task(graph, task="put the box back", scene=_scene(), thread_id="t3")

    final = resume_with_approval(graph, approved=True, thread_id="t3")
    assert final["outcome"] == "done"
    assert final["execution_result"]["success"] is True
    assert final["approved"] is True


def test_resume_denied_terminates_without_execution():
    graph = _build_interrupt_graph()
    run_task(graph, task="put the box back", scene=_scene(), thread_id="t4")

    final = resume_with_approval(graph, approved=False, thread_id="t4")
    assert final["outcome"] == "denied"
    assert final["execution_result"] is None


def test_separate_threads_are_independent():
    """Two tasks on different thread_ids interrupt and resume independently."""
    graph = _build_interrupt_graph()
    run_task(graph, task="task A", scene=_scene(), thread_id="a")
    run_task(graph, task="task B", scene=_scene(), thread_id="b")

    final_a = resume_with_approval(graph, approved=True, thread_id="a")
    final_b = resume_with_approval(graph, approved=False, thread_id="b")
    assert final_a["outcome"] == "done"
    assert final_b["outcome"] == "denied"


# ------------------------------------------------------- backward compatibility


def test_callable_approval_still_works_without_checkpointer():
    """The original auto-approve callable path must keep working (no interrupt)."""
    graph = build_orchestrator(
        planner_fn=_planner,
        critic_fn=_critic,
        gate_fn=_gate,
        executor_fn=_executor,
        approval_fn=lambda selection: True,
        config=OrchestratorConfig(),
    )
    final = run_task(graph, task="put the box back", scene=_scene())
    assert final["outcome"] == "done"
