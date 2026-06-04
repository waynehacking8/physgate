"""Backend-agnostic plan executor.

Walks a validated Plan step by step against any WorldBackend:

1. re-check the step's L3 preconditions against the backend's *current* scene
   (defense in depth — the gate validated against the initial scene, but the
   world may have drifted),
2. dispatch the step to the matching MCP tool,
3. stop at the first failure and return a structured result the orchestrator
   can route on (retry / replan / escalate).

The same executor drives the MockWorldBackend (symbolic demo) and the Isaac
Sim backend (executor/sim_backend.py) — the dual-backend design from
architecture doc §5.
"""

from __future__ import annotations

from typing import Any

from physgate.executor.backend import WorldBackend
from physgate.gate.l3_scene import check_preconditions
from physgate.mcp_server.tools.core import (
    call_elevator_tool,
    execute_skill_tool,
    inspect_object_tool,
    move_to_pose_tool,
    open_door_tool,
    press_button_tool,
    push_object_tool,
    query_scene_tool,
    request_assistance_tool,
    unlock_door_tool,
)
from physgate.planner.schemas import Plan, PlanStep, ToolName


def _dispatch_step(step: PlanStep, backend: WorldBackend) -> dict[str, Any]:
    """Route one plan step to its MCP tool."""
    if step.tool == ToolName.QUERY_SCENE:
        return {"success": True, "payload": query_scene_tool(backend)}
    if step.tool == ToolName.MOVE_TO_POSE:
        return move_to_pose_tool(
            backend,
            target=step.args["target"],
            standoff_m=step.args.get("standoff_m", 0.3),
            speed=step.args.get("speed", 0.5),
        )
    if step.tool == ToolName.EXECUTE_SKILL:
        return execute_skill_tool(
            backend, skill=step.args["skill"], target=step.args["target"]
        )
    if step.tool == ToolName.OPEN_DOOR:
        return open_door_tool(backend, door_id=step.args["door_id"])
    if step.tool == ToolName.UNLOCK_DOOR:
        return unlock_door_tool(
            backend, door_id=step.args["door_id"], key_id=step.args["key_id"]
        )
    if step.tool == ToolName.PRESS_BUTTON:
        return press_button_tool(backend, button_id=step.args["button_id"])
    if step.tool == ToolName.CALL_ELEVATOR:
        return call_elevator_tool(
            backend,
            elevator_id=step.args["elevator_id"],
            target_floor=step.args["target_floor"],
        )
    if step.tool == ToolName.PUSH_OBJECT:
        return push_object_tool(
            backend,
            object_id=step.args["object_id"],
            direction=step.args["direction"],
        )
    if step.tool == ToolName.INSPECT_OBJECT:
        result = inspect_object_tool(backend, object_id=step.args["object_id"])
        return {**result, "success": result.get("success", True)}
    if step.tool == ToolName.REQUEST_ASSISTANCE:
        return request_assistance_tool(backend, message=step.args.get("message", ""))
    return {"success": False, "error": f"unknown tool {step.tool}"}


def execute_plan(plan: Plan, backend: WorldBackend) -> dict[str, Any]:
    """Execute a plan on a backend. Returns a structured execution result.

    Result keys:
        success: overall success
        steps_completed / steps_total
        step_results: per-step outcome dicts (step_id, success, ...)
        failed_step_id: id of the first failed step (None on success)
        error: human-readable error of the first failure (absent on success)
        failure_report: serialized L3 FailureReport if a precondition recheck
            failed (None otherwise)
    """
    step_results: list[dict[str, Any]] = []

    for step in plan.steps:
        # 1. precondition recheck against the live scene
        report = check_preconditions(step.preconditions, backend.get_scene(), step.step_id)
        if report is not None:
            return {
                "success": False,
                "steps_completed": len(step_results),
                "steps_total": len(plan.steps),
                "step_results": step_results,
                "failed_step_id": step.step_id,
                "error": f"precondition recheck failed at step {step.step_id}: "
                + "; ".join(v.detail for v in report.violations),
                "failure_report": report.model_dump(mode="json"),
            }

        # 2. dispatch to the tool
        outcome = _dispatch_step(step, backend)
        step_results.append({"step_id": step.step_id, **outcome})

        if outcome.get("assistance_requested"):
            return {
                "success": False,
                "assistance_requested": True,
                "message": outcome.get("message", ""),
                "steps_completed": len(step_results),
                "steps_total": len(plan.steps),
                "step_results": step_results,
                "failed_step_id": None,
                "error": f"task requires assistance: {outcome.get('message', '')}",
                "failure_report": None,
            }

        if not outcome.get("success"):
            return {
                "success": False,
                "steps_completed": len(step_results) - 1,
                "steps_total": len(plan.steps),
                "step_results": step_results,
                "failed_step_id": step.step_id,
                "error": outcome.get("error", f"step {step.step_id} failed"),
                "failure_report": None,
            }

    return {
        "success": True,
        "steps_completed": len(step_results),
        "steps_total": len(plan.steps),
        "step_results": step_results,
        "failed_step_id": None,
        "failure_report": None,
    }
