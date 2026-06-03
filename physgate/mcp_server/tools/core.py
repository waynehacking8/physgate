"""The three physgate MCP tools, as backend-agnostic functions.

Each tool is a thin adapter: validate arguments, delegate to a WorldBackend,
return a JSON-safe dict. The MCP server (server.py) wraps these for the wire;
the executor calls them directly in-process.

Design reference: architecture doc §5 — exactly 3 MCP tools for the MVP.
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
