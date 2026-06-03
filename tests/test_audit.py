"""Tests for the audit trail: three-stream records + Merkle checkpoints (#4 part B).

Architecture doc §4: every pipeline event lands in one of three streams
(decision / physical / human), all joined by a correlation_id, with periodic
Merkle roots for tamper evidence (certificate-transparency style, NOT an inline
hash chain).
"""

import json

import pytest

from physgate.audit.merkle import hash_record, merkle_root, verify_records
from physgate.audit.records import AuditRecord, AuditStream
from physgate.audit.trail import AuditTrail

# --------------------------------------------------------------------- merkle


def test_hash_record_is_deterministic():
    record = {"a": 1, "b": "two"}
    assert hash_record(record) == hash_record({"b": "two", "a": 1})  # key order irrelevant
    assert len(hash_record(record)) == 64  # sha256 hex


def test_merkle_root_changes_when_any_record_changes():
    hashes = [hash_record({"i": i}) for i in range(8)]
    root = merkle_root(hashes)

    tampered = list(hashes)
    tampered[3] = hash_record({"i": 999})
    assert merkle_root(tampered) != root


def test_merkle_root_single_and_odd_counts():
    assert merkle_root([hash_record({"x": 1})]) is not None
    assert merkle_root([hash_record({"i": i}) for i in range(3)]) is not None


def test_merkle_root_empty_raises():
    with pytest.raises(ValueError):
        merkle_root([])


def test_verify_records_detects_tampering():
    records = [{"event": "plan", "i": i} for i in range(5)]
    root = merkle_root([hash_record(r) for r in records])
    assert verify_records(records, root) is True

    records[2]["i"] = 42  # tamper
    assert verify_records(records, root) is False


# -------------------------------------------------------------------- records


def test_audit_record_schema():
    record = AuditRecord(
        correlation_id="task_001/step_2",
        stream=AuditStream.DECISION,
        event="plan_selected",
        payload={"best_plan_id": "p0"},
        model_id="claude-opus-4-8",
        operator_id="wayne",
        sim_flag=True,
    )
    assert record.timestamp > 0
    assert record.clock_source == "monotonic_ns"
    # JSON-serializable for export
    json.dumps(record.model_dump(mode="json"))


def test_correlation_id_requires_task_prefix():
    with pytest.raises(ValueError):
        AuditRecord(
            correlation_id="",  # empty is invalid
            stream=AuditStream.HUMAN,
            event="approval",
        )


# ---------------------------------------------------------------------- trail


def _populate(trail: AuditTrail, task_id: str = "task_001") -> None:
    trail.record(AuditStream.DECISION, "candidates_generated", f"{task_id}/planning", {"n": 8})
    trail.record(AuditStream.DECISION, "critic_verdict", f"{task_id}/reviewing", {"survivors": 6})
    trail.record(AuditStream.DECISION, "plan_selected", f"{task_id}/validating", {"best": "p0"})
    trail.record(AuditStream.HUMAN, "approval", f"{task_id}/approval", {"approved": True})
    trail.record(AuditStream.PHYSICAL, "execution_step", f"{task_id}/step_1", {"success": True})
    trail.record(AuditStream.PHYSICAL, "execution_step", f"{task_id}/step_2", {"success": True})


def test_trail_records_and_correlates():
    trail = AuditTrail()
    _populate(trail, "task_001")
    _populate(trail, "task_002")

    task1_records = trail.records_for("task_001")
    assert len(task1_records) == 6
    assert all(r.correlation_id.startswith("task_001") for r in task1_records)

    # all three streams are represented for the task
    streams = {r.stream for r in task1_records}
    assert streams == {AuditStream.DECISION, AuditStream.PHYSICAL, AuditStream.HUMAN}


def test_trail_periodic_merkle_checkpoints():
    trail = AuditTrail(checkpoint_every=4)
    _populate(trail)  # 6 records -> at least 1 automatic checkpoint at 4
    assert len(trail.checkpoints) >= 1

    # closing flushes the remaining records into a final checkpoint
    final_root = trail.close()
    assert final_root is not None
    assert trail.verify_integrity() is True


def test_trail_tamper_detection():
    trail = AuditTrail(checkpoint_every=2)
    _populate(trail)
    trail.close()
    assert trail.verify_integrity() is True

    # tamper with a recorded payload after checkpointing
    trail._records[1].payload["survivors"] = 999
    assert trail.verify_integrity() is False


def test_trail_jsonl_export(tmp_path):
    trail = AuditTrail()
    _populate(trail)
    trail.close()

    out = tmp_path / "audit.jsonl"
    trail.to_jsonl(out)
    lines = out.read_text().strip().split("\n")
    # one line per record + one line per checkpoint
    assert len(lines) == 6 + len(trail.checkpoints)
    parsed = [json.loads(line) for line in lines]
    assert any(p.get("type") == "merkle_checkpoint" for p in parsed)


# ------------------------------------------------------- orchestrator integration


def test_orchestrator_emits_audit_records():
    from physgate.gate.scoring import PhysicsResult, select_best
    from physgate.gate.schemas import Scene, SceneObject
    from physgate.orchestrator.graph import OrchestratorConfig, build_orchestrator, run_task
    from physgate.planner.schemas import Plan, PlanStep, ToolName

    scene = Scene(objects=[SceneObject(id="box_03"), SceneObject(id="shelf_A")])
    plan = Plan(
        plan_id="p0",
        task="t",
        steps=[PlanStep(step_id=1, tool=ToolName.EXECUTE_SKILL, args={"skill": "pick"})],
    )

    trail = AuditTrail()
    graph = build_orchestrator(
        planner_fn=lambda task, s, n, fb=None: [plan],
        critic_fn=lambda plans, s: list(plans),
        gate_fn=lambda plans, s: select_best(
            [PhysicsResult(plan_id=p.plan_id, success=True) for p in plans]
        ),
        executor_fn=lambda p, s: {"success": True, "steps_completed": 1},
        approval_fn=lambda sel: True,
        config=OrchestratorConfig(),
        audit_trail=trail,
    )
    final = run_task(graph, task="test task", scene=scene, thread_id="audit_test")
    assert final["outcome"] == "done"

    # every pipeline phase produced a decision record
    events = [r.event for r in trail._records]
    assert "candidates_generated" in events
    assert "critic_verdict" in events
    assert "plan_selected" in events
    assert "execution_result" in events
    # the human stream recorded the approval
    human = [r for r in trail._records if r.stream == AuditStream.HUMAN]
    assert len(human) == 1
    assert human[0].payload["approved"] is True
    # integrity holds
    trail.close()
    assert trail.verify_integrity() is True
