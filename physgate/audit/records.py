"""Audit record schema: the three-stream correlation format (architecture doc §4).

Every pipeline event is one AuditRecord in one of three streams, all joined by
``correlation_id`` (``task_id/step_id``). The fields mirror the architecture's
AuditRecord spec: model identity, operator, environment, sim/real flag, and a
primary timestamp with an explicit clock source.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class AuditStream(str, Enum):
    """The three audit streams that correlate into one record set."""

    DECISION = "decision"    # planner / critic / gate decisions (-> Langfuse/OTEL)
    PHYSICAL = "physical"    # robot / sim actions and outcomes (-> MCAP)
    HUMAN = "human"          # approvals and operator interventions


class AuditRecord(BaseModel):
    """One audit event. JSON-serializable; hashed into Merkle checkpoints."""

    correlation_id: str
    stream: AuditStream
    event: str
    payload: dict[str, Any] = Field(default_factory=dict)

    # provenance (architecture doc §4 AuditRecord fields)
    model_id: str | None = None
    operator_id: str | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    sim_flag: bool = True

    # timing: monotonic clock for ordering + wall clock for humans
    timestamp: float = Field(default_factory=time.monotonic)
    wall_time: float = Field(default_factory=time.time)
    clock_source: str = "monotonic_ns"

    @field_validator("correlation_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("correlation_id must be non-empty (use 'task_id/step_id')")
        return value
