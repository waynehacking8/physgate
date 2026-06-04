"""Tests for scene-graph construction, query, and effect application (A4).

The scene graph is the world-state representation shared by L3 preconditions,
the planner prompt (query_scene output), and symbolic plan simulation. Updates
are immutable: applying effects returns a NEW Scene, never mutates in place.
"""

import pytest

from physgate.gate.schemas import Scene, SceneObject
from physgate.planner.schemas import RelationChange
from physgate.world.scene_graph import (
    apply_effects,
    build_scene,
    objects_with_affordance,
    objects_with_label,
    relations_of,
    to_query_scene_payload,
)


def _fetch_scene() -> Scene:
    return build_scene(
        objects=[
            {"id": "box_03", "label": "cardboard_box", "affordances": ["graspable"], "is_anomaly": True},
            {"id": "shelf_A", "label": "shelf", "affordances": ["placeable"]},
            {"id": "floor_01", "label": "floor"},
            {"id": "go2", "label": "robot"},
        ],
        relations=[("box_03", "on", "floor_01")],
        gripper_empty=True,
    )


# ----------------------------------------------------------------- build/query


def test_build_scene_from_dicts():
    scene = _fetch_scene()
    assert scene.has_object("box_03")
    assert scene.has_relation("box_03", "on", "floor_01")
    assert scene.gripper_empty is True


def test_objects_with_label():
    scene = _fetch_scene()
    shelves = objects_with_label(scene, "shelf")
    assert [o.id for o in shelves] == ["shelf_A"]


def test_objects_with_affordance():
    scene = _fetch_scene()
    graspable = objects_with_affordance(scene, "graspable")
    assert [o.id for o in graspable] == ["box_03"]


def test_relations_of_subject():
    scene = _fetch_scene()
    assert relations_of(scene, "box_03") == [("box_03", "on", "floor_01")]
    assert relations_of(scene, "shelf_A") == []


# -------------------------------------------------------------- apply_effects


def test_apply_effects_returns_new_scene_original_unchanged():
    scene = _fetch_scene()
    effects = [
        RelationChange(op="remove", subject="box_03", predicate="on", object="floor_01"),
        RelationChange(op="add", subject="box_03", predicate="on", object="shelf_A"),
    ]
    new_scene = apply_effects(scene, effects)

    # new scene reflects the change
    assert new_scene.has_relation("box_03", "on", "shelf_A")
    assert not new_scene.has_relation("box_03", "on", "floor_01")
    # original is untouched (immutability)
    assert scene.has_relation("box_03", "on", "floor_01")
    assert not scene.has_relation("box_03", "on", "shelf_A")


def test_gripper_holding_effect_updates_gripper_empty():
    scene = _fetch_scene()
    pick = [RelationChange(op="add", subject="gripper", predicate="holding", object="box_03")]
    holding_scene = apply_effects(scene, pick)
    assert holding_scene.gripper_empty is False
    assert holding_scene.has_relation("gripper", "holding", "box_03")

    place = [RelationChange(op="remove", subject="gripper", predicate="holding", object="box_03")]
    released_scene = apply_effects(holding_scene, place)
    assert released_scene.gripper_empty is True


def test_removing_nonexistent_relation_is_noop():
    scene = _fetch_scene()
    effects = [RelationChange(op="remove", subject="ghost", predicate="on", object="floor_01")]
    new_scene = apply_effects(scene, effects)
    assert new_scene.relations == scene.relations


def test_adding_duplicate_relation_is_idempotent():
    scene = _fetch_scene()
    effects = [RelationChange(op="add", subject="box_03", predicate="on", object="floor_01")]
    new_scene = apply_effects(scene, effects)
    assert new_scene.relations.count(("box_03", "on", "floor_01")) == 1


# --------------------------------------------------------- query_scene payload


def test_query_scene_payload_shape():
    """The MCP query_scene tool returns this JSON-able payload."""
    scene = _fetch_scene()
    payload = to_query_scene_payload(scene)
    assert set(payload.keys()) == {"objects", "relations", "gripper_empty", "anomalies", "available_tools"}
    assert payload["gripper_empty"] is True
    ids = [o["id"] for o in payload["objects"]]
    assert "box_03" in ids and "shelf_A" in ids
    # the fallen box is flagged as the anomaly to fix
    assert payload["anomalies"] == ["box_03"]
    # payload must be JSON-serializable
    import json

    json.dumps(payload)


def test_build_scene_rejects_duplicate_ids():
    with pytest.raises(ValueError):
        build_scene(
            objects=[{"id": "box_03"}, {"id": "box_03"}],
            relations=[],
        )


def test_build_scene_accepts_scene_objects():
    scene = build_scene(
        objects=[SceneObject(id="a", label="thing")],
        relations=[],
        gripper_empty=False,
    )
    assert scene.has_object("a")
    assert scene.gripper_empty is False
