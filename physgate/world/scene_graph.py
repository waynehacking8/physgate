"""Scene-graph construction, query, and symbolic effect application.

The scene graph (:class:`physgate.gate.schemas.Scene`) is the shared
world-state representation used by:

* the L3 precondition gate (``gate/l3_scene.py``),
* the planner prompt (``query_scene`` MCP tool payload),
* symbolic plan simulation (applying step effects to predict future state).

All updates are **immutable**: :func:`apply_effects` returns a new Scene and
never mutates its input — this lets the gate evaluate N candidate plans against
the same initial scene without cross-contamination.

Design reference: docs/design/architecture.md section 5 (world state).
"""

from __future__ import annotations

from typing import Any

from physgate.gate.schemas import Scene, SceneObject
from physgate.planner.schemas import RelationChange

#: The special relation that mirrors into Scene.gripper_empty.
_GRIPPER_HOLDING = ("gripper", "holding")


def build_scene(
    objects: list[dict[str, Any] | SceneObject],
    relations: list[tuple[str, str, str]],
    gripper_empty: bool = True,
) -> Scene:
    """Build a Scene from raw object dicts (or SceneObjects) and relation triples.

    Raises:
        ValueError: if two objects share an id.
    """
    parsed = [o if isinstance(o, SceneObject) else SceneObject(**o) for o in objects]
    ids = [o.id for o in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate object ids in scene: {ids}")
    return Scene(
        objects=parsed,
        relations=[tuple(r) for r in relations],
        gripper_empty=gripper_empty,
    )


def objects_with_label(scene: Scene, label: str) -> list[SceneObject]:
    """All objects whose label matches exactly."""
    return [o for o in scene.objects if o.label == label]


def objects_with_affordance(scene: Scene, affordance: str) -> list[SceneObject]:
    """All objects offering the given affordance (e.g. 'graspable')."""
    return [o for o in scene.objects if affordance in o.affordances]


def relations_of(scene: Scene, subject: str) -> list[tuple[str, str, str]]:
    """All relation triples whose subject is the given object id."""
    return [r for r in scene.relations if r[0] == subject]


def apply_effects(scene: Scene, effects: list[RelationChange]) -> Scene:
    """Apply symbolic plan-step effects, returning a NEW Scene (immutable update).

    Special case: the ``(gripper, holding, X)`` relation also drives the
    ``gripper_empty`` flag, so L3 ``gripper_empty`` preconditions stay
    consistent with symbolic execution.
    """
    relations = list(scene.relations)
    gripper_empty = scene.gripper_empty

    for effect in effects:
        triple = (effect.subject, effect.predicate, effect.object)
        if effect.op == "add":
            if triple not in relations:
                relations.append(triple)
        else:  # remove
            relations = [r for r in relations if r != triple]

        if (effect.subject, effect.predicate) == _GRIPPER_HOLDING:
            gripper_empty = effect.op == "remove"

    return Scene(
        objects=list(scene.objects),
        relations=relations,
        gripper_empty=gripper_empty,
    )


def _compute_available_tools(scene: Scene) -> list[str]:
    """Derive which tools are usable given the current scene objects."""
    tools = ["query_scene", "move_to_pose", "execute_skill", "inspect_object"]

    has_door = any("door" in o.affordances for o in scene.objects)
    has_locked_door = any("door" in o.affordances and o.locked for o in scene.objects)
    has_button = any("button" in o.affordances for o in scene.objects)
    has_elevator = any("elevator" in o.affordances for o in scene.objects)
    has_pushable = any(o.pushable for o in scene.objects)

    if has_door:
        tools.append("open_door")
    if has_locked_door:
        tools.append("unlock_door")
    if has_button:
        tools.append("press_button")
    if has_elevator:
        tools.append("call_elevator")
    if has_pushable:
        tools.append("push_object")

    tools.append("request_assistance")
    return tools


def to_query_scene_payload(scene: Scene) -> dict[str, Any]:
    """Serialize a Scene into the ``query_scene`` MCP tool response payload.

    JSON-safe dict with the object list, relation triples, gripper state,
    the ids of anomalous objects, and the available tools for this scene.
    """
    return {
        "objects": [o.model_dump() for o in scene.objects],
        "relations": [list(r) for r in scene.relations],
        "gripper_empty": scene.gripper_empty,
        "anomalies": [o.id for o in scene.objects if o.is_anomaly],
        "available_tools": _compute_available_tools(scene),
    }
