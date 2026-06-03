"""Shared data structures for the Sim-Gate.

These are pure data contracts (no GPU, no Isaac dependency) so the gate's
deterministic layers (L1 kinematic, L3 scene-graph) and the failure-reporting
format can be built and tested test-first, before any simulator is wired up.

Design reference: docs/design/architecture.md sections 4-5.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FailureCode(str, Enum):
    """Predefined, specific failure codes.

    Generic free-text failure strings do not improve LLM recovery; a closed
    vocabulary of codes does (see design doc, failure-handling decision).
    """

    COLLISION = "collision"
    JOINT_LIMIT = "joint_limit"
    GRASP_FAILURE = "grasp_failure"
    UNREACHABLE_POSE = "unreachable_pose"
    PRECONDITION_VIOLATION = "precondition_violation"
    TIMEOUT = "timeout"
    DIVERGENCE = "divergence"


class GateLayer(str, Enum):
    KINEMATIC_LIMIT = "kinematic_limit"
    PHYSICS = "physics"
    SEMANTIC_PRECONDITION = "semantic_precondition"


class Violation(BaseModel):
    """One concrete reason a plan step was rejected."""

    type: str
    detail: str = ""
    severity: str = "critical"


class FailureReport(BaseModel):
    """Structured, LLM-consumable rejection report (REFLECT-style).

    Returned by the Sim-Gate when a step is DENIED so the orchestrator can route
    recovery and, on L2 replanning, hand the planner a precise reason.
    """

    failed_step_id: int | None = None
    verdict: str = "DENIED"
    failure_code: FailureCode
    layer: GateLayer
    violations: list[Violation] = Field(default_factory=list)
    remediation_hint: str = ""
    retryable: bool = True


class SceneObject(BaseModel):
    """One object in the world-state scene graph (subset of query_scene output)."""

    id: str
    label: str = ""
    affordances: list[str] = Field(default_factory=list)
    is_anomaly: bool = False


class Scene(BaseModel):
    """Flat object-list + relations scene graph.

    `relations` are (subject, predicate, object) triples, e.g.
    ``("box_03", "on", "floor_01")``. This doubles as the L3 precondition input.
    """

    objects: list[SceneObject] = Field(default_factory=list)
    relations: list[tuple[str, str, str]] = Field(default_factory=list)
    gripper_empty: bool = True

    def has_object(self, object_id: str) -> bool:
        return any(o.id == object_id for o in self.objects)

    def has_relation(self, subject: str, predicate: str, obj: str) -> bool:
        return (subject, predicate, obj) in self.relations
