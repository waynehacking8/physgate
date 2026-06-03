"""Tests for the Sim-Gate L3 scene-graph precondition checker.

Written test-first: this layer is pure Python and needs no GPU, so it is the
first thing built (Phase 1 in the build order). The fetch-and-place MVP task is
the source of these cases.
"""

from physgate.gate.l3_scene import check_preconditions
from physgate.gate.schemas import FailureCode, GateLayer, Scene, SceneObject


def _fetch_scene() -> Scene:
    """A box on the floor near a shelf; gripper empty; shelf unoccupied."""
    return Scene(
        objects=[
            SceneObject(id="box_03", label="cardboard_box", affordances=["graspable"]),
            SceneObject(id="shelf_A", label="shelf"),
            SceneObject(id="floor_01", label="floor"),
        ],
        relations=[("box_03", "on", "floor_01"), ("shelf_A", "unoccupied", "shelf_A")],
        gripper_empty=True,
    )


def test_all_preconditions_met_returns_none():
    scene = _fetch_scene()
    preconds = ["box_03 exists", "gripper_empty", "shelf_A unoccupied shelf_A"]
    assert check_preconditions(preconds, scene, step_id=4) is None


def test_missing_object_is_denied():
    scene = _fetch_scene()
    report = check_preconditions(["box_99 exists"], scene, step_id=2)
    assert report is not None
    assert report.verdict == "DENIED"
    assert report.failure_code == FailureCode.PRECONDITION_VIOLATION
    assert report.layer == GateLayer.SEMANTIC_PRECONDITION
    assert report.failed_step_id == 2
    assert report.violations[0].type == "missing_object"


def test_gripper_not_empty_is_denied():
    scene = _fetch_scene()
    scene.gripper_empty = False
    report = check_preconditions(["gripper_empty"], scene)
    assert report is not None
    assert report.violations[0].type == "gripper_not_empty"


def test_unmet_relation_is_denied():
    scene = _fetch_scene()
    # box is on the floor, not on the shelf yet
    report = check_preconditions(["box_03 on shelf_A"], scene)
    assert report is not None
    assert report.violations[0].type == "unmet_relation"


def test_multiple_violations_all_reported():
    scene = _fetch_scene()
    scene.gripper_empty = False
    report = check_preconditions(["box_99 exists", "gripper_empty"], scene)
    assert report is not None
    assert len(report.violations) == 2


def test_unparseable_precondition_fails_closed():
    scene = _fetch_scene()
    report = check_preconditions(["this is not a valid precondition form"], scene)
    assert report is not None
    assert report.violations[0].type == "unparseable_precondition"
