"""Tests for the 7 new MCP tools added in the agent architecture upgrade.

Each tool is tested for:
- precondition checks (robot nearness, holding key, etc.)
- successful state transitions
- immutability (initial scene never mutated)
- edge cases (already open, already unlocked, invalid direction, etc.)
"""

from physgate.executor.backend import MockWorldBackend
from physgate.gate.schemas import Scene, SceneObject
from physgate.mcp_server.tools.core import (
    call_elevator_tool,
    inspect_object_tool,
    open_door_tool,
    press_button_tool,
    push_object_tool,
    query_scene_tool,
    request_assistance_tool,
    unlock_door_tool,
)
from physgate.planner.schemas import RelationChange
from physgate.world.scene_graph import apply_effects


# ------------------------------------------------------------------ fixtures


def _door_scene(locked: bool = False) -> Scene:
    """A scene with a door (optionally locked) and a key."""
    return Scene(
        objects=[
            SceneObject(id="go2", label="robot"),
            SceneObject(id="door_01", label="door", affordances=["door"], locked=locked),
            SceneObject(id="key_01", label="key", affordances=["graspable"], weight_kg=0.1),
            SceneObject(id="floor_01", label="floor"),
            SceneObject(id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
        ],
        relations=[
            ("door_01", "state", "closed"),
            ("box_03", "on", "floor_01"),
            ("key_01", "on", "floor_01"),
        ],
        gripper_empty=True,
    )


def _elevator_scene() -> Scene:
    return Scene(
        objects=[
            SceneObject(id="go2", label="robot", floor=1),
            SceneObject(id="elevator_01", label="elevator", affordances=["elevator"], floor=1),
            SceneObject(id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True, floor=1),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"], floor=2),
            SceneObject(id="floor_01", label="floor", floor=1),
        ],
        relations=[("box_03", "on", "floor_01")],
        gripper_empty=True,
    )


def _pushable_scene() -> Scene:
    return Scene(
        objects=[
            SceneObject(id="go2", label="robot"),
            SceneObject(id="crate_01", label="crate", pushable=True, weight_kg=20.0),
            SceneObject(id="box_03", label="cardboard_box", affordances=["graspable"], is_anomaly=True),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
        ],
        relations=[
            ("crate_01", "blocking", "path_to_shelf"),
            ("box_03", "on", "floor_01"),
        ],
        gripper_empty=True,
    )


def _button_scene() -> Scene:
    return Scene(
        objects=[
            SceneObject(id="go2", label="robot"),
            SceneObject(id="button_01", label="button", affordances=["button"]),
            SceneObject(id="gate_01", label="gate", affordances=["door"]),
            SceneObject(id="floor_01", label="floor"),
        ],
        relations=[
            ("gate_01", "state", "closed"),
            ("button_01", "activates", "gate_01"),
        ],
        gripper_empty=True,
    )


def _heavy_object_scene() -> Scene:
    return Scene(
        objects=[
            SceneObject(id="go2", label="robot"),
            SceneObject(id="anvil_01", label="anvil", affordances=[], weight_kg=200.0),
            SceneObject(id="shelf_A", label="shelf", affordances=["placeable"]),
            SceneObject(id="floor_01", label="floor"),
        ],
        relations=[("anvil_01", "on", "floor_01")],
        gripper_empty=True,
    )


# ================================================================= open_door


class TestOpenDoor:
    def test_open_door_success(self):
        backend = MockWorldBackend(_door_scene())
        backend.move_to_pose(target="door_01")
        result = backend.open_door(door_id="door_01")
        assert result["success"] is True
        assert backend.get_scene().has_relation("door_01", "state", "open")
        assert not backend.get_scene().has_relation("door_01", "state", "closed")

    def test_open_door_not_near_fails(self):
        backend = MockWorldBackend(_door_scene())
        result = backend.open_door(door_id="door_01")
        assert result["success"] is False
        assert "near" in result["error"]

    def test_open_locked_door_fails(self):
        backend = MockWorldBackend(_door_scene(locked=True))
        backend.move_to_pose(target="door_01")
        result = backend.open_door(door_id="door_01")
        assert result["success"] is False
        assert "locked" in result["error"]

    def test_open_door_not_a_door_fails(self):
        backend = MockWorldBackend(_door_scene())
        backend.move_to_pose(target="box_03")
        result = backend.open_door(door_id="box_03")
        assert result["success"] is False
        assert "not a door" in result["error"]

    def test_open_door_already_open(self):
        backend = MockWorldBackend(_door_scene())
        backend.move_to_pose(target="door_01")
        backend.open_door(door_id="door_01")
        result = backend.open_door(door_id="door_01")
        assert result["success"] is True
        assert "already open" in result.get("note", "")

    def test_open_door_unknown_id_fails(self):
        backend = MockWorldBackend(_door_scene())
        result = backend.open_door(door_id="nonexistent")
        assert result["success"] is False

    def test_open_door_immutability(self):
        initial = _door_scene()
        backend = MockWorldBackend(initial)
        backend.move_to_pose(target="door_01")
        backend.open_door(door_id="door_01")
        assert initial.has_relation("door_01", "state", "closed")


# ================================================================ unlock_door


class TestUnlockDoor:
    def test_unlock_door_success(self):
        scene = _door_scene(locked=True)
        backend = MockWorldBackend(scene)
        backend.move_to_pose(target="key_01")
        backend.execute_skill(skill="pick", target="key_01")
        backend.move_to_pose(target="door_01")
        result = backend.unlock_door(door_id="door_01", key_id="key_01")
        assert result["success"] is True
        door = next(o for o in backend.get_scene().objects if o.id == "door_01")
        assert door.locked is False

    def test_unlock_door_not_holding_key_fails(self):
        backend = MockWorldBackend(_door_scene(locked=True))
        backend.move_to_pose(target="door_01")
        result = backend.unlock_door(door_id="door_01", key_id="key_01")
        assert result["success"] is False
        assert "holding" in result["error"]

    def test_unlock_door_not_near_fails(self):
        scene = _door_scene(locked=True)
        backend = MockWorldBackend(scene)
        backend.move_to_pose(target="key_01")
        backend.execute_skill(skill="pick", target="key_01")
        result = backend.unlock_door(door_id="door_01", key_id="key_01")
        assert result["success"] is False
        assert "near" in result["error"]

    def test_unlock_already_unlocked(self):
        backend = MockWorldBackend(_door_scene(locked=False))
        backend.move_to_pose(target="key_01")
        backend.execute_skill(skill="pick", target="key_01")
        backend.move_to_pose(target="door_01")
        result = backend.unlock_door(door_id="door_01", key_id="key_01")
        assert result["success"] is True
        assert "already unlocked" in result.get("note", "")

    def test_unlock_door_immutability(self):
        initial = _door_scene(locked=True)
        backend = MockWorldBackend(initial)
        backend.move_to_pose(target="key_01")
        backend.execute_skill(skill="pick", target="key_01")
        backend.move_to_pose(target="door_01")
        backend.unlock_door(door_id="door_01", key_id="key_01")
        initial_door = next(o for o in initial.objects if o.id == "door_01")
        assert initial_door.locked is True


# ============================================================== press_button


class TestPressButton:
    def test_press_button_opens_gate(self):
        backend = MockWorldBackend(_button_scene())
        backend.move_to_pose(target="button_01")
        result = backend.press_button(button_id="button_01")
        assert result["success"] is True
        assert backend.get_scene().has_relation("gate_01", "state", "open")
        assert not backend.get_scene().has_relation("gate_01", "state", "closed")

    def test_press_button_not_near_fails(self):
        backend = MockWorldBackend(_button_scene())
        result = backend.press_button(button_id="button_01")
        assert result["success"] is False
        assert "near" in result["error"]

    def test_press_button_not_a_button_fails(self):
        backend = MockWorldBackend(_button_scene())
        backend.move_to_pose(target="gate_01")
        result = backend.press_button(button_id="gate_01")
        assert result["success"] is False
        assert "not a button" in result["error"]

    def test_press_button_marks_pressed(self):
        backend = MockWorldBackend(_button_scene())
        backend.move_to_pose(target="button_01")
        backend.press_button(button_id="button_01")
        assert backend.get_scene().has_relation("button_01", "state", "pressed")


# ============================================================= call_elevator


class TestCallElevator:
    def test_call_elevator_success(self):
        backend = MockWorldBackend(_elevator_scene())
        backend.move_to_pose(target="elevator_01")
        result = backend.call_elevator(elevator_id="elevator_01", target_floor=2)
        assert result["success"] is True
        assert result["arrived_floor"] == 2
        robot = next(o for o in backend.get_scene().objects if o.id == "go2")
        assert robot.floor == 2
        elev = next(o for o in backend.get_scene().objects if o.id == "elevator_01")
        assert elev.floor == 2

    def test_call_elevator_carries_held_object(self):
        backend = MockWorldBackend(_elevator_scene())
        backend.move_to_pose(target="box_03")
        backend.execute_skill(skill="pick", target="box_03")
        backend.move_to_pose(target="elevator_01")
        backend.call_elevator(elevator_id="elevator_01", target_floor=2)
        box = next(o for o in backend.get_scene().objects if o.id == "box_03")
        assert box.floor == 2

    def test_call_elevator_not_near_fails(self):
        backend = MockWorldBackend(_elevator_scene())
        result = backend.call_elevator(elevator_id="elevator_01", target_floor=2)
        assert result["success"] is False
        assert "near" in result["error"]

    def test_call_elevator_not_an_elevator_fails(self):
        backend = MockWorldBackend(_elevator_scene())
        backend.move_to_pose(target="box_03")
        result = backend.call_elevator(elevator_id="box_03", target_floor=2)
        assert result["success"] is False
        assert "not an elevator" in result["error"]

    def test_call_elevator_wrong_floor_fails(self):
        """Robot on floor 1 cannot even reach elevator on floor 2 (F3 floor check)."""
        scene = Scene(
            objects=[
                SceneObject(id="go2", label="robot", floor=1),
                SceneObject(id="elevator_01", label="elevator", affordances=["elevator"], floor=2),
                SceneObject(id="floor_01", label="floor"),
            ],
            relations=[],
            gripper_empty=True,
        )
        backend = MockWorldBackend(scene)
        move_result = backend.move_to_pose(target="elevator_01")
        assert move_result["success"] is False
        assert "floor" in move_result["error"]


# ============================================================== push_object


class TestPushObject:
    def test_push_object_success(self):
        backend = MockWorldBackend(_pushable_scene())
        backend.move_to_pose(target="crate_01")
        result = backend.push_object(object_id="crate_01", direction="east")
        assert result["success"] is True
        assert backend.get_scene().has_relation("crate_01", "pushed", "east")
        assert not backend.get_scene().has_relation("crate_01", "blocking", "path_to_shelf")

    def test_push_object_not_pushable_fails(self):
        backend = MockWorldBackend(_pushable_scene())
        backend.move_to_pose(target="box_03")
        result = backend.push_object(object_id="box_03", direction="north")
        assert result["success"] is False
        assert "not pushable" in result["error"]

    def test_push_object_not_near_fails(self):
        backend = MockWorldBackend(_pushable_scene())
        result = backend.push_object(object_id="crate_01", direction="north")
        assert result["success"] is False
        assert "near" in result["error"]

    def test_push_object_invalid_direction_fails(self):
        backend = MockWorldBackend(_pushable_scene())
        backend.move_to_pose(target="crate_01")
        result = backend.push_object(object_id="crate_01", direction="up")
        assert result["success"] is False
        assert "invalid direction" in result["error"]

    def test_push_object_immutability(self):
        initial = _pushable_scene()
        backend = MockWorldBackend(initial)
        backend.move_to_pose(target="crate_01")
        backend.push_object(object_id="crate_01", direction="north")
        assert initial.has_relation("crate_01", "blocking", "path_to_shelf")


# ============================================================ inspect_object


class TestInspectObject:
    def test_inspect_graspable_object(self):
        backend = MockWorldBackend(_door_scene())
        result = backend.inspect_object(object_id="box_03")
        assert result["success"] is True
        assert result["graspable"] is True
        assert result["carriable"] is True
        assert result["weight_kg"] == 1.0

    def test_inspect_heavy_object(self):
        backend = MockWorldBackend(_heavy_object_scene())
        result = backend.inspect_object(object_id="anvil_01")
        assert result["success"] is True
        assert result["graspable"] is False
        assert result["carriable"] is False
        assert result["weight_kg"] == 200.0

    def test_inspect_locked_door(self):
        backend = MockWorldBackend(_door_scene(locked=True))
        result = backend.inspect_object(object_id="door_01")
        assert result["success"] is True
        assert result["locked"] is True

    def test_inspect_pushable(self):
        backend = MockWorldBackend(_pushable_scene())
        result = backend.inspect_object(object_id="crate_01")
        assert result["success"] is True
        assert result["pushable"] is True

    def test_inspect_unknown_object_fails(self):
        backend = MockWorldBackend(_door_scene())
        result = backend.inspect_object(object_id="nonexistent")
        assert result["success"] is False

    def test_inspect_returns_floor(self):
        backend = MockWorldBackend(_elevator_scene())
        result = backend.inspect_object(object_id="box_03")
        assert result["floor"] == 1


# ======================================================= request_assistance


class TestRequestAssistance:
    def test_request_assistance_success(self):
        backend = MockWorldBackend(_heavy_object_scene())
        result = backend.request_assistance(message="anvil too heavy to carry")
        assert result["success"] is True
        assert result["assistance_requested"] is True
        assert "anvil" in result["message"]


# ========================================================= available_tools


class TestAvailableTools:
    def test_basic_scene_tools(self):
        scene = Scene(
            objects=[
                SceneObject(id="go2", label="robot"),
                SceneObject(id="box_03", label="box", affordances=["graspable"]),
            ],
            relations=[],
            gripper_empty=True,
        )
        backend = MockWorldBackend(scene)
        payload = query_scene_tool(backend)
        tools = payload["available_tools"]
        assert "query_scene" in tools
        assert "move_to_pose" in tools
        assert "execute_skill" in tools
        assert "inspect_object" in tools
        assert "request_assistance" in tools
        assert "open_door" not in tools
        assert "call_elevator" not in tools

    def test_door_scene_includes_door_tools(self):
        backend = MockWorldBackend(_door_scene(locked=True))
        payload = query_scene_tool(backend)
        tools = payload["available_tools"]
        assert "open_door" in tools
        assert "unlock_door" in tools

    def test_elevator_scene_includes_elevator_tool(self):
        backend = MockWorldBackend(_elevator_scene())
        payload = query_scene_tool(backend)
        tools = payload["available_tools"]
        assert "call_elevator" in tools

    def test_pushable_scene_includes_push_tool(self):
        backend = MockWorldBackend(_pushable_scene())
        payload = query_scene_tool(backend)
        tools = payload["available_tools"]
        assert "push_object" in tools

    def test_button_scene_includes_button_tool(self):
        backend = MockWorldBackend(_button_scene())
        payload = query_scene_tool(backend)
        tools = payload["available_tools"]
        assert "press_button" in tools


# ========================================================= tool adapters


class TestToolAdapters:
    """Test the thin adapter functions in core.py delegate correctly."""

    def test_open_door_tool_delegates(self):
        backend = MockWorldBackend(_door_scene())
        backend.move_to_pose(target="door_01")
        from physgate.mcp_server.tools.core import open_door_tool
        result = open_door_tool(backend, door_id="door_01")
        assert result["success"] is True

    def test_inspect_object_tool_delegates(self):
        backend = MockWorldBackend(_door_scene())
        result = inspect_object_tool(backend, object_id="box_03")
        assert result["success"] is True
        assert "weight_kg" in result

    def test_request_assistance_tool_delegates(self):
        backend = MockWorldBackend(_heavy_object_scene())
        result = request_assistance_tool(backend, message="need help")
        assert result["success"] is True


# =========================================== full key→unlock→open→deliver chain


class TestLockedDoorDeliveryChain:
    """Integration test: the full T2 locked-room delivery tool chain."""

    def test_key_unlock_open_deliver(self):
        scene = _door_scene(locked=True)
        backend = MockWorldBackend(scene)

        # 1. pick up the key
        backend.move_to_pose(target="key_01")
        assert backend.execute_skill(skill="pick", target="key_01")["success"]

        # 2. go to door and unlock it
        backend.move_to_pose(target="door_01")
        assert backend.unlock_door(door_id="door_01", key_id="key_01")["success"]

        # 3. open the door
        assert backend.open_door(door_id="door_01")["success"]
        assert backend.get_scene().has_relation("door_01", "state", "open")

        # 4. put key down, pick box, deliver
        backend.move_to_pose(target="floor_01")
        assert backend.execute_skill(skill="place", target="floor_01")["success"]
        backend.move_to_pose(target="box_03")
        assert backend.execute_skill(skill="pick", target="box_03")["success"]
        backend.move_to_pose(target="shelf_A")
        assert backend.execute_skill(skill="place", target="shelf_A")["success"]
        assert backend.get_scene().has_relation("box_03", "on", "shelf_A")


# ============================================ elevator floor transfer chain


class TestElevatorFloorTransfer:
    """Integration test: T5 cross-floor delivery via elevator."""

    def test_pick_ride_elevator_place(self):
        backend = MockWorldBackend(_elevator_scene())

        backend.move_to_pose(target="box_03")
        assert backend.execute_skill(skill="pick", target="box_03")["success"]

        backend.move_to_pose(target="elevator_01")
        assert backend.call_elevator(elevator_id="elevator_01", target_floor=2)["success"]

        robot = next(o for o in backend.get_scene().objects if o.id == "go2")
        assert robot.floor == 2

        backend.move_to_pose(target="shelf_A")
        assert backend.execute_skill(skill="place", target="shelf_A")["success"]
        assert backend.get_scene().has_relation("box_03", "on", "shelf_A")


class TestFloorAwareness:
    """F3 verification: move_to_pose rejects cross-floor movement."""

    def test_cross_floor_move_rejected(self):
        backend = MockWorldBackend(_elevator_scene())
        result = backend.move_to_pose(target="shelf_A")
        assert result["success"] is False
        assert "floor" in result["error"]
        assert "elevator" in result["error"]

    def test_same_floor_move_allowed(self):
        backend = MockWorldBackend(_elevator_scene())
        result = backend.move_to_pose(target="box_03")
        assert result["success"] is True

    def test_after_elevator_cross_floor_allowed(self):
        backend = MockWorldBackend(_elevator_scene())
        backend.move_to_pose(target="elevator_01")
        backend.call_elevator(elevator_id="elevator_01", target_floor=2)
        result = backend.move_to_pose(target="shelf_A")
        assert result["success"] is True
