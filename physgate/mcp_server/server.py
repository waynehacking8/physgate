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
    execute_skill_tool,
    move_to_pose_tool,
    query_scene_tool,
)


def create_server(backend: WorldBackend, name: str = "physgate") -> FastMCP:
    """Create a FastMCP server exposing query_scene / move_to_pose / execute_skill."""
    server = FastMCP(name)

    @server.tool(name="query_scene", description="Get the current scene graph: objects, relations, gripper state, anomalies.")
    def query_scene() -> dict[str, Any]:
        return query_scene_tool(backend)

    @server.tool(name="move_to_pose", description="Move the robot base to a standoff pose near a target object.")
    def move_to_pose(target: str, standoff_m: float = 0.3, speed: float = 0.5) -> dict[str, Any]:
        return move_to_pose_tool(backend, target=target, standoff_m=standoff_m, speed=speed)

    @server.tool(name="execute_skill", description="Execute a manipulation skill ('pick' or 'place') on a target object.")
    def execute_skill(skill: str, target: str) -> dict[str, Any]:
        return execute_skill_tool(backend, skill=skill, target=target)

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
