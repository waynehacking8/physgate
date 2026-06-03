"""AuditTrail: three-stream collection + periodic Merkle checkpoints.

MVP implementation of the architecture's audit design (doc §4):

* records accumulate in memory (and export to JSONL),
* every ``checkpoint_every`` records, a Merkle root is computed over the new
  records and appended to the checkpoint log,
* :meth:`verify_integrity` recomputes every checkpoint to detect tampering.

Production back-ends (Langfuse/OTEL for decisions, MCAP for physical) plug in
behind the same ``record()`` call — see DECISIONS.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from physgate.audit.merkle import hash_record, merkle_root
from physgate.audit.records import AuditRecord, AuditStream


class MerkleCheckpoint(BaseModel):
    """A periodic root over a contiguous range of records."""

    start_index: int
    end_index: int  # exclusive
    root: str


class AuditTrail:
    """Collects audit records and maintains periodic Merkle checkpoints."""

    def __init__(self, checkpoint_every: int = 16):
        self._records: list[AuditRecord] = []
        self.checkpoints: list[MerkleCheckpoint] = []
        self._checkpoint_every = checkpoint_every
        self._next_checkpoint_start = 0

    # ----- recording -----

    def record(
        self,
        stream: AuditStream,
        event: str,
        correlation_id: str,
        payload: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditRecord:
        """Append one audit record; checkpoint automatically when due."""
        entry = AuditRecord(
            correlation_id=correlation_id,
            stream=stream,
            event=event,
            payload=payload or {},
            **kwargs,
        )
        self._records.append(entry)
        if len(self._records) - self._next_checkpoint_start >= self._checkpoint_every:
            self._checkpoint()
        return entry

    def _checkpoint(self) -> MerkleCheckpoint | None:
        """Compute a Merkle root over records since the last checkpoint."""
        start, end = self._next_checkpoint_start, len(self._records)
        if end <= start:
            return None
        hashes = [hash_record(r.model_dump(mode="json")) for r in self._records[start:end]]
        checkpoint = MerkleCheckpoint(start_index=start, end_index=end, root=merkle_root(hashes))
        self.checkpoints.append(checkpoint)
        self._next_checkpoint_start = end
        return checkpoint

    def close(self) -> str | None:
        """Flush remaining records into a final checkpoint; return its root."""
        checkpoint = self._checkpoint()
        return checkpoint.root if checkpoint else (
            self.checkpoints[-1].root if self.checkpoints else None
        )

    # ----- queries -----

    def records_for(self, task_id: str) -> list[AuditRecord]:
        """Three-stream correlation: every record whose correlation_id starts with task_id."""
        return [r for r in self._records if r.correlation_id.startswith(task_id)]

    # ----- integrity -----

    def verify_integrity(self) -> bool:
        """Recompute every checkpoint root; False if any record was tampered with."""
        for checkpoint in self.checkpoints:
            window = self._records[checkpoint.start_index : checkpoint.end_index]
            hashes = [hash_record(r.model_dump(mode="json")) for r in window]
            try:
                if merkle_root(hashes) != checkpoint.root:
                    return False
            except ValueError:
                return False
        return True

    # ----- export -----

    def to_jsonl(self, path: str | Path) -> None:
        """Export records + checkpoints as JSON Lines (one object per line)."""
        path = Path(path)
        lines = [r.model_dump_json() for r in self._records]
        for checkpoint in self.checkpoints:
            lines.append(
                checkpoint.model_copy()
                .model_dump_json()
                .replace("{", '{"type": "merkle_checkpoint", ', 1)
            )
        path.write_text("\n".join(lines) + "\n")
