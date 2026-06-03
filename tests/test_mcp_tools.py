"""Tests for the MCP tools + world backend (D16).

The three MCP tools (query_scene / move_to_pose / execute_skill) are thin
adapters over a WorldBackend. MockWorldBackend executes symbolically against a
Scene (no GPU); SimBackend (executor/) will implement the same protocol against
Isaac Sim.
"""

import asyncio
import json

from physgate.executor.backend import MockWorldBackend
from physgate.gate.schemas import Scene, SceneObject
from physgate.mcp_server.server import create_server
from physgate.mcp_server.tools.core import (
    execute_skill_tool,
    move_to_pose_tool,
    query_scene_tool,
)


def _scene() -> Scene:
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


# ----------------------------------------------------------- MockWorldBackend


def test_backend_get_scene():
    backend = MockWorldBackend(_scene())
    assert backend.get_scene().has_object("box_03")


def test_backend_move_to_pose():
    backend = MockWorldBackend(_scene())
    result = backend.move_to_pose(target="box_03", standoff_m=0.3)
    assert result["success"] is True
    assert backend.get_scene().has_relation("robot", "near", "box_03")


def test_backend_move_to_unknown_target_fails():
    backend = MockWorldBackend(_scene())
    result = backend.move_to_pose(target="box_99")
    assert result["success"] is False
    assert "box_99" in result["error"]


def test_backend_pick_and_place_flow():
    backend = MockWorldBackend(_scene())
    backend.move_to_pose(target="box_03")
    pick = backend.execute_skill(skill="pick", target="box_03")
    assert pick["success"] is True
    assert backend.get_scene().gripper_empty is False
    assert backend.get_scene().has_relation("gripper", "holding", "box_03")

    backend.move_to_pose(target="shelf_A")
    place = backend.execute_skill(skill="place", target="shelf_A")
    assert place["success"] is True
    scene = backend.get_scene()
    assert scene.gripper_empty is True
    assert scene.has_relation("box_03", "on", "shelf_A")
    assert not scene.has_relation("box_03", "on", "floor_01")


def test_backend_pick_without_being_near_fails():
    backend = MockWorldBackend(_scene())
    result = backend.execute_skill(skill="pick", target="box_03")
    assert result["success"] is False
    assert "near" in result["error"]


def test_backend_pick_with_full_gripper_fails():
    backend = MockWorldBackend(_scene())
    backend.move_to_pose(target="box_03")
    backend.execute_skill(skill="pick", target="box_03")
    backend.move_to_pose(target="shelf_A")
    result = backend.execute_skill(skill="pick", target="shelf_A")
    assert result["success"] is False


def test_backend_unknown_skill_fails():
    backend = MockWorldBackend(_scene())
    result = backend.execute_skill(skill="backflip", target="box_03")
    assert result["success"] is False
    assert "backflip" in result["error"]


def test_backend_does_not_mutate_initial_scene():
    initial = _scene()
    backend = MockWorldBackend(initial)
    backend.move_to_pose(target="box_03")
    backend.execute_skill(skill="pick", target="box_03")
    # the Scene object passed in is never mutated (immutability)
    assert initial.gripper_empty is True
    assert not initial.has_relation("robot", "near", "box_03")


# ------------------------------------------------------------------ MCP tools


def test_query_scene_tool_returns_json_payload():
    backend = MockWorldBackend(_scene())
    payload = query_scene_tool(backend)
    assert payload["gripper_empty"] is True
    assert payload["anomalies"] == ["box_03"]
    json.dumps(payload)  # must be JSON-serializable


def test_move_to_pose_tool_delegates():
    backend = MockWorldBackend(_scene())
    result = move_to_pose_tool(backend, target="box_03", standoff_m=0.5)
    assert result["success"] is True


def test_execute_skill_tool_delegates():
    backend = MockWorldBackend(_scene())
    move_to_pose_tool(backend, target="box_03")
    result = execute_skill_tool(backend, skill="pick", target="box_03")
    assert result["success"] is True


# ------------------------------------------------------------------ MCP server


def test_server_registers_three_tools():
    backend = MockWorldBackend(_scene())
    server = create_server(backend)
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"query_scene", "move_to_pose", "execute_skill"}


def test_server_tools_are_callable():
    backend = MockWorldBackend(_scene())
    server = create_server(backend)

    result = asyncio.run(server.call_tool("query_scene", {}))
    # FastMCP returns (content_blocks, raw_result) or content blocks
    text = json.dumps(str(result))
    assert "box_03" in text
