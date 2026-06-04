"""physgate MCP tools, as backend-agnostic functions.

Each tool is a thin adapter: validate arguments, delegate to a WorldBackend,
return a JSON-safe dict. The MCP server (server.py) wraps these for the wire;
the executor calls them directly in-process.

Design reference: architecture doc §5 + AGENT_UPGRADE.md §1.
"""

from __future__ import annotations

from typing import Any

from physgate.executor.backend import WorldBackend
from physgate.world.scene_graph import to_query_scene_payload


def query_scene_tool(backend: WorldBackend) -> dict[str, Any]:
    """Return the current scene graph (objects, relations, gripper, anomalies)."""
    return to_query_scene_payload(backend.get_scene())


def move_to_pose_tool(
    backend: WorldBackend, target: str, standoff_m: float = 0.3, speed: float = 0.5
) -> dict[str, Any]:
    """Move the robot base to a standoff pose near the target object."""
    return backend.move_to_pose(target=target, standoff_m=standoff_m, speed=speed)


def execute_skill_tool(backend: WorldBackend, skill: str, target: str) -> dict[str, Any]:
    """Execute a manipulation skill ('pick' or 'place') on the target object."""
    return backend.execute_skill(skill=skill, target=target)


def open_door_tool(backend: WorldBackend, door_id: str) -> dict[str, Any]:
    """Open a closed, unlocked door near the robot."""
    return backend.open_door(door_id=door_id)


def unlock_door_tool(backend: WorldBackend, door_id: str, key_id: str) -> dict[str, Any]:
    """Unlock a locked door using a key the robot is holding."""
    return backend.unlock_door(door_id=door_id, key_id=key_id)


def press_button_tool(backend: WorldBackend, button_id: str) -> dict[str, Any]:
    """Press a button near the robot; effect depends on what it activates."""
    return backend.press_button(button_id=button_id)


def call_elevator_tool(backend: WorldBackend, elevator_id: str, target_floor: int) -> dict[str, Any]:
    """Call the elevator to the target floor (robot must be near the elevator panel)."""
    return backend.call_elevator(elevator_id=elevator_id, target_floor=target_floor)


def push_object_tool(backend: WorldBackend, object_id: str, direction: str) -> dict[str, Any]:
    """Push a pushable object in a cardinal direction to clear a path."""
    return backend.push_object(object_id=object_id, direction=direction)


def inspect_object_tool(backend: WorldBackend, object_id: str) -> dict[str, Any]:
    """Inspect an object to learn weight, graspability, lock state, etc."""
    return backend.inspect_object(object_id=object_id)


def request_assistance_tool(backend: WorldBackend, message: str) -> dict[str, Any]:
    """Signal that the robot cannot complete the task alone and needs help."""
    return backend.request_assistance(message=message)
