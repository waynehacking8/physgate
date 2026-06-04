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

from physgate.gate.schemas import Scene
from physgate.planner.schemas import RelationChange
from physgate.world.scene_graph import apply_effects, objects_with_affordance


@runtime_checkable
class WorldBackend(Protocol):
    """Protocol every execution backend must satisfy."""

    def get_scene(self) -> Scene: ...

    def move_to_pose(self, target: str, standoff_m: float = 0.3, **kwargs: Any) -> dict[str, Any]: ...

    def execute_skill(self, skill: str, target: str, **kwargs: Any) -> dict[str, Any]: ...


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

    def move_to_pose(self, target: str, standoff_m: float = 0.3, **kwargs: Any) -> dict[str, Any]:
        """Move the robot near the target by updating the scene graph."""
        if not self._scene.has_object(target):
            return {"success": False, "error": f"move_to_pose target '{target}' not in scene"}

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

    def _pick(self, target: str) -> dict[str, Any]:
        scene = self._scene
        if not scene.has_object(target):
            return {"success": False, "error": f"pick target '{target}' not in scene"}
        if not scene.has_relation("robot", "near", target):
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
