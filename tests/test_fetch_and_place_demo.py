"""End-to-end test of the fetch-and-place demo pipeline (E18/E19).

Runs the COMPLETE pipeline (planner -> critic -> gate -> approval -> executor)
exactly as examples/fetch_and_place.py does, using the mock planner/critic and
symbolic physics. This is the no-GPU, no-API-key proof that the pipeline works
end to end; the demo script adds Isaac Sim / Claude API when available.
"""

from physgate.examples_lib.fetch_and_place import build_demo_scene, run_fetch_and_place


def test_demo_pipeline_end_to_end():
    outcome = run_fetch_and_place(task="put the fallen box back on shelf A")

    # the pipeline ran to completion
    assert outcome["outcome"] == "done"

    # the planner produced 8 candidates, critic pruned at least the reckless ones
    assert len(outcome["candidates"]) == 8
    assert 1 <= len(outcome["survivors"]) < 8

    # the gate selected a best plan from the survivors
    selection = outcome["selection"]
    assert selection.any_feasible is True
    assert selection.best_plan_id in {p.plan_id for p in outcome["survivors"]}

    # execution succeeded and the world reflects the completed task
    assert outcome["execution_result"]["success"] is True
    final_scene = outcome["final_scene"]
    assert final_scene.has_relation("box_03", "on", "shelf_A")
    assert not final_scene.has_relation("box_03", "on", "floor_01")


def test_demo_scene_has_anomaly_to_fix():
    scene = build_demo_scene()
    anomalies = [o for o in scene.objects if o.is_anomaly]
    assert [o.id for o in anomalies] == ["box_03"]
    assert scene.has_relation("box_03", "on", "floor_01")


def test_demo_report_is_printable():
    """The demo produces a human-readable report (used for the terminal demo)."""
    outcome = run_fetch_and_place(task="put the fallen box back on shelf A")
    report = outcome["report"]
    assert "put the fallen box back on shelf A" in report
    assert "candidates" in report.lower()
    assert outcome["selection"].best_plan_id in report
    assert "PASS" in report or "done" in report.lower()
