"""physgate MCP server: exposes the three tools over the MCP protocol.

The server wraps a WorldBackend (mock or Isaac Sim) so an external LLM agent
(e.g. Claude with MCP support) can perceive and act through the same gate-
validated interface the internal pipeline uses.

Run standalone (stdio transport, mock backend):

    python -m physgate.mcp_server.server
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from physgate.executor.backend import MockWorldBackend, WorldBackend
from physgate.gate.schemas import Scene, SceneObject
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


def create_server(backend: WorldBackend, name: str = "physgate") -> FastMCP:
    """Create a FastMCP server exposing all physgate tools."""
    server = FastMCP(name)

    @server.tool(name="query_scene", description="Get the current scene graph: objects, relations, gripper state, anomalies, available tools.")
    def query_scene() -> dict[str, Any]:
        return query_scene_tool(backend)

    @server.tool(name="move_to_pose", description="Move the robot base to a standoff pose near a target object.")
    def move_to_pose(target: str, standoff_m: float = 0.3, speed: float = 0.5) -> dict[str, Any]:
        return move_to_pose_tool(backend, target=target, standoff_m=standoff_m, speed=speed)

    @server.tool(name="execute_skill", description="Execute a manipulation skill ('pick' or 'place') on a target object.")
    def execute_skill(skill: str, target: str) -> dict[str, Any]:
        return execute_skill_tool(backend, skill=skill, target=target)

    @server.tool(name="open_door", description="Open a closed, unlocked door near the robot.")
    def open_door(door_id: str) -> dict[str, Any]:
        return open_door_tool(backend, door_id=door_id)

    @server.tool(name="unlock_door", description="Unlock a locked door using a key the robot is holding.")
    def unlock_door(door_id: str, key_id: str) -> dict[str, Any]:
        return unlock_door_tool(backend, door_id=door_id, key_id=key_id)

    @server.tool(name="press_button", description="Press a button near the robot; effect depends on what it activates.")
    def press_button(button_id: str) -> dict[str, Any]:
        return press_button_tool(backend, button_id=button_id)

    @server.tool(name="call_elevator", description="Call the elevator to the target floor (robot must be near the elevator panel).")
    def call_elevator(elevator_id: str, target_floor: int) -> dict[str, Any]:
        return call_elevator_tool(backend, elevator_id=elevator_id, target_floor=target_floor)

    @server.tool(name="push_object", description="Push a pushable object in a cardinal direction to clear a path.")
    def push_object(object_id: str, direction: str) -> dict[str, Any]:
        return push_object_tool(backend, object_id=object_id, direction=direction)

    @server.tool(name="inspect_object", description="Inspect an object to learn weight, graspability, lock state, floor, etc.")
    def inspect_object(object_id: str) -> dict[str, Any]:
        return inspect_object_tool(backend, object_id=object_id)

    @server.tool(name="request_assistance", description="Signal that the robot cannot complete the task alone and needs help.")
    def request_assistance(message: str) -> dict[str, Any]:
        return request_assistance_tool(backend, message=message)

    return server


def _default_demo_scene() -> Scene:
    """The fetch-and-place MVP scene, for standalone server runs."""
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


if __name__ == "__main__":
    create_server(MockWorldBackend(_default_demo_scene())).run()
