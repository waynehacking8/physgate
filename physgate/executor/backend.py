"""World backends: the execution surface behind the MCP tools.

A WorldBackend is anything that can report the scene and execute primitive
robot actions. Two implementations:

* :class:`MockWorldBackend` (here) — symbolic execution against a Scene graph.
  No GPU, no simulator. Used by tests and the offline demo.
* ``SimBackend`` (executor/sim_backend.py) — same protocol against Isaac Sim.

The dual-backend design is the architecture's "execution abstraction" decision
(architecture doc §5): the executor and MCP tools never know which backend
they are driving.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from physgate.gate.schemas import Scene, SceneObject
from physgate.planner.schemas import RelationChange
from physgate.world.scene_graph import apply_effects, objects_with_affordance


@runtime_checkable
class WorldBackend(Protocol):
    """Protocol every execution backend must satisfy."""

    def get_scene(self) -> Scene: ...

    def move_to_pose(self, target: str, standoff_m: float = 0.3, **kwargs: Any) -> dict[str, Any]: ...

    def execute_skill(self, skill: str, target: str, **kwargs: Any) -> dict[str, Any]: ...

    def open_door(self, door_id: str, **kwargs: Any) -> dict[str, Any]: ...

    def unlock_door(self, door_id: str, key_id: str, **kwargs: Any) -> dict[str, Any]: ...

    def press_button(self, button_id: str, **kwargs: Any) -> dict[str, Any]: ...

    def call_elevator(self, elevator_id: str, target_floor: int, **kwargs: Any) -> dict[str, Any]: ...

    def push_object(self, object_id: str, direction: str, **kwargs: Any) -> dict[str, Any]: ...

    def inspect_object(self, object_id: str, **kwargs: Any) -> dict[str, Any]: ...

    def request_assistance(self, message: str, **kwargs: Any) -> dict[str, Any]: ...


class MockWorldBackend:
    """Symbolic (scene-graph-only) execution backend.

    Action semantics:
        move_to_pose: robot becomes 'near' the target (single nearness at a time).
        pick:         requires nearness + empty gripper + graspable target.
        place:        requires nearness + holding something; held object goes 'on' target.

    The backend's scene evolves via immutable updates; the Scene passed to the
    constructor is never modified.
    """

    def __init__(self, scene: Scene):
        """Initialize with a deep copy of the given scene."""
        self._scene = scene.model_copy(deep=True)

    # ----- WorldBackend protocol -----

    def get_scene(self) -> Scene:
        """Return the current symbolic scene state."""
        return self._scene

    def _robot_floor(self) -> int:
        robot = self._get_object("go2")
        return robot.floor if robot else 1

    def move_to_pose(self, target: str, standoff_m: float = 0.3, **kwargs: Any) -> dict[str, Any]:
        """Move the robot near the target by updating the scene graph."""
        if not self._scene.has_object(target):
            return {"success": False, "error": f"move_to_pose target '{target}' not in scene"}

        target_obj = self._get_object(target)
        if target_obj is not None and target_obj.floor != self._robot_floor():
            return {
                "success": False,
                "error": (
                    f"cannot move to '{target}' on floor {target_obj.floor} — "
                    f"robot is on floor {self._robot_floor()}; use call_elevator first"
                ),
            }

        # drop any previous nearness, then add the new one
        effects = [
            RelationChange(op="remove", subject="robot", predicate="near", object=rel[2])
            for rel in self._scene.relations
            if rel[0] == "robot" and rel[1] == "near"
        ]
        effects.append(RelationChange(op="add", subject="robot", predicate="near", object=target))
        self._scene = apply_effects(self._scene, effects)
        return {"success": True, "target": target, "standoff_m": standoff_m}

    def execute_skill(self, skill: str, target: str, **kwargs: Any) -> dict[str, Any]:
        """Dispatch a pick or place skill against the symbolic scene."""
        if skill == "pick":
            return self._pick(target)
        if skill == "place":
            return self._place(target)
        return {"success": False, "error": f"unknown skill '{skill}' (available: pick, place)"}

    # ----- skills -----

    def _is_reachable(self, target: str) -> bool:
        """True if robot is near the target OR near the object the target sits on."""
        if self._scene.has_relation("robot", "near", target):
            return True
        for rel in self._scene.relations:
            if rel[0] == target and rel[1] == "on":
                if self._scene.has_relation("robot", "near", rel[2]):
                    return True
        return False

    def _pick(self, target: str) -> dict[str, Any]:
        scene = self._scene
        if not scene.has_object(target):
            return {"success": False, "error": f"pick target '{target}' not in scene"}
        if not self._is_reachable(target):
            return {"success": False, "error": f"robot is not near '{target}'"}
        if not scene.gripper_empty:
            return {"success": False, "error": "gripper is already holding an object"}
        if target not in {o.id for o in objects_with_affordance(scene, "graspable")}:
            return {"success": False, "error": f"'{target}' is not graspable"}

        # picking removes the object's support relation and fills the gripper
        effects = [
            RelationChange(op="remove", subject=target, predicate="on", object=rel[2])
            for rel in scene.relations
            if rel[0] == target and rel[1] == "on"
        ]
        effects.append(
            RelationChange(op="add", subject="gripper", predicate="holding", object=target)
        )
        self._scene = apply_effects(scene, effects)
        return {"success": True, "skill": "pick", "target": target}

    def _place(self, target: str) -> dict[str, Any]:
        scene = self._scene
        if not scene.has_object(target):
            return {"success": False, "error": f"place target '{target}' not in scene"}
        held = [rel[2] for rel in scene.relations if rel[0] == "gripper" and rel[1] == "holding"]
        if not held:
            return {"success": False, "error": "gripper is not holding anything to place"}
        if not scene.has_relation("robot", "near", target):
            return {"success": False, "error": f"robot is not near '{target}'"}

        held_object = held[0]
        effects = [
            RelationChange(op="remove", subject="gripper", predicate="holding", object=held_object),
            RelationChange(op="add", subject=held_object, predicate="on", object=target),
        ]
        self._scene = apply_effects(scene, effects)
        return {"success": True, "skill": "place", "target": target, "placed_object": held_object}

    # ----- new tools (agent architecture upgrade) -----

    def _get_object(self, object_id: str) -> SceneObject | None:
        return next((o for o in self._scene.objects if o.id == object_id), None)

    def open_door(self, door_id: str, **kwargs: Any) -> dict[str, Any]:
        obj = self._get_object(door_id)
        if obj is None:
            return {"success": False, "error": f"door '{door_id}' not in scene"}
        if "door" not in obj.affordances:
            return {"success": False, "error": f"'{door_id}' is not a door"}
        if not self._scene.has_relation("robot", "near", door_id):
            return {"success": False, "error": f"robot is not near '{door_id}'"}
        if obj.locked:
            return {"success": False, "error": f"door '{door_id}' is locked — unlock it first"}
        if self._scene.has_relation(door_id, "state", "open"):
            return {"success": True, "door_id": door_id, "note": "door was already open"}

        effects = [
            RelationChange(op="remove", subject=door_id, predicate="state", object="closed"),
            RelationChange(op="add", subject=door_id, predicate="state", object="open"),
        ]
        self._scene = apply_effects(self._scene, effects)
        return {"success": True, "door_id": door_id}

    def unlock_door(self, door_id: str, key_id: str, **kwargs: Any) -> dict[str, Any]:
        obj = self._get_object(door_id)
        if obj is None:
            return {"success": False, "error": f"door '{door_id}' not in scene"}
        if "door" not in obj.affordances:
            return {"success": False, "error": f"'{door_id}' is not a door"}
        if not self._scene.has_relation("robot", "near", door_id):
            return {"success": False, "error": f"robot is not near '{door_id}'"}
        if not self._scene.has_relation("gripper", "holding", key_id):
            return {"success": False, "error": f"robot is not holding key '{key_id}'"}
        if not obj.locked:
            return {"success": True, "door_id": door_id, "note": "door was already unlocked"}

        updated_objects = [
            o.model_copy(update={"locked": False}) if o.id == door_id else o
            for o in self._scene.objects
        ]
        updated_relations = [
            r for r in self._scene.relations
            if not (r[0] == door_id and r[1] == "state" and r[2] == "locked")
        ]
        self._scene = Scene(
            objects=updated_objects,
            relations=updated_relations,
            gripper_empty=self._scene.gripper_empty,
        )
        return {"success": True, "door_id": door_id, "key_id": key_id}

    def press_button(self, button_id: str, **kwargs: Any) -> dict[str, Any]:
        obj = self._get_object(button_id)
        if obj is None:
            return {"success": False, "error": f"button '{button_id}' not in scene"}
        if "button" not in obj.affordances:
            return {"success": False, "error": f"'{button_id}' is not a button"}
        if not self._scene.has_relation("robot", "near", button_id):
            return {"success": False, "error": f"robot is not near '{button_id}'"}

        activations: list[RelationChange] = []
        for rel in self._scene.relations:
            if rel[0] == button_id and rel[1] == "activates":
                target_id = rel[2]
                target_obj = self._get_object(target_id)
                if target_obj is not None and "door" in target_obj.affordances:
                    activations.append(
                        RelationChange(op="remove", subject=target_id, predicate="state", object="closed")
                    )
                    activations.append(
                        RelationChange(op="add", subject=target_id, predicate="state", object="open")
                    )
                elif target_obj is not None and "elevator" in target_obj.affordances:
                    activations.append(
                        RelationChange(op="add", subject=target_id, predicate="state", object="called")
                    )

        effects = [
            RelationChange(op="add", subject=button_id, predicate="state", object="pressed"),
        ] + activations
        self._scene = apply_effects(self._scene, effects)
        return {"success": True, "button_id": button_id, "activated": [a.object for a in activations]}

    def call_elevator(self, elevator_id: str, target_floor: int, **kwargs: Any) -> dict[str, Any]:
        obj = self._get_object(elevator_id)
        if obj is None:
            return {"success": False, "error": f"elevator '{elevator_id}' not in scene"}
        if "elevator" not in obj.affordances:
            return {"success": False, "error": f"'{elevator_id}' is not an elevator"}
        if not self._scene.has_relation("robot", "near", elevator_id):
            return {"success": False, "error": f"robot is not near '{elevator_id}'"}

        robot_obj = self._get_object("go2")
        robot_floor = robot_obj.floor if robot_obj else 1
        if robot_floor != obj.floor:
            return {
                "success": False,
                "error": f"elevator is on floor {obj.floor}, robot is on floor {robot_floor}",
            }

        updated_objects = []
        for o in self._scene.objects:
            if o.id == elevator_id:
                updated_objects.append(o.model_copy(update={"floor": target_floor}))
            elif o.id == "go2":
                updated_objects.append(o.model_copy(update={"floor": target_floor}))
            else:
                updated_objects.append(o)

        held = [rel[2] for rel in self._scene.relations if rel[0] == "gripper" and rel[1] == "holding"]
        for o_idx, o in enumerate(updated_objects):
            if o.id in held:
                updated_objects[o_idx] = o.model_copy(update={"floor": target_floor})

        self._scene = Scene(
            objects=updated_objects,
            relations=list(self._scene.relations),
            gripper_empty=self._scene.gripper_empty,
        )
        return {"success": True, "elevator_id": elevator_id, "arrived_floor": target_floor}

    def push_object(self, object_id: str, direction: str, **kwargs: Any) -> dict[str, Any]:
        obj = self._get_object(object_id)
        if obj is None:
            return {"success": False, "error": f"object '{object_id}' not in scene"}
        if not obj.pushable:
            return {"success": False, "error": f"'{object_id}' is not pushable"}
        if not self._scene.has_relation("robot", "near", object_id):
            return {"success": False, "error": f"robot is not near '{object_id}'"}

        valid_directions = {"north", "south", "east", "west"}
        if direction not in valid_directions:
            return {"success": False, "error": f"invalid direction '{direction}'; use {valid_directions}"}

        effects = [
            RelationChange(op="add", subject=object_id, predicate="pushed", object=direction),
        ]
        for rel in self._scene.relations:
            if rel[0] == object_id and rel[1] == "blocking":
                effects.append(
                    RelationChange(op="remove", subject=object_id, predicate="blocking", object=rel[2])
                )
        self._scene = apply_effects(self._scene, effects)
        return {"success": True, "object_id": object_id, "direction": direction}

    def inspect_object(self, object_id: str, **kwargs: Any) -> dict[str, Any]:
        obj = self._get_object(object_id)
        if obj is None:
            return {"success": False, "error": f"object '{object_id}' not in scene"}

        max_carry_kg = 5.0
        return {
            "success": True,
            "object_id": object_id,
            "label": obj.label,
            "weight_kg": obj.weight_kg,
            "graspable": "graspable" in obj.affordances,
            "carriable": obj.weight_kg <= max_carry_kg and "graspable" in obj.affordances,
            "locked": obj.locked,
            "pushable": obj.pushable,
            "floor": obj.floor,
            "affordances": obj.affordances,
        }

    def request_assistance(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return {"success": True, "assistance_requested": True, "message": message}
