"""Plan / PlanStep data contracts (planner → gate → executor boundary).

These models define the *plan artifact*: the only thing that crosses the
dual-envelope boundary toward the executor is this structured JSON — never code.
One PlanStep maps to exactly one MCP tool call (flat step list; behaviour trees
deferred until the skill library exceeds ~20 skills, per architecture doc §5).

Design reference: docs/design/architecture.md sections 1 and 5.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

#: Number of candidate plans the planner generates per task (fixed for MVP).
DEFAULT_NUM_CANDIDATES = 8


class ToolName(str, Enum):
    """MCP tools a plan step may call (architecture doc §5)."""

    QUERY_SCENE = "query_scene"
    MOVE_TO_POSE = "move_to_pose"
    EXECUTE_SKILL = "execute_skill"
    OPEN_DOOR = "open_door"
    UNLOCK_DOOR = "unlock_door"
    PRESS_BUTTON = "press_button"
    CALL_ELEVATOR = "call_elevator"
    PUSH_OBJECT = "push_object"
    INSPECT_OBJECT = "inspect_object"
    REQUEST_ASSISTANCE = "request_assistance"


class RelationChange(BaseModel):
    """One symbolic effect of a plan step on the scene graph.

    Applying an ``add`` inserts the (subject, predicate, object) relation;
    ``remove`` deletes it. The scene-graph module applies these immutably.
    """

    op: Literal["add", "remove"]
    subject: str
    predicate: str
    object: str


class PlanStep(BaseModel):
    """One step of a plan — exactly one tool call plus its symbolic contract."""

    step_id: int
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    #: L3-checkable precondition strings (see gate/l3_scene.py grammar).
    preconditions: list[str] = Field(default_factory=list)
    #: Symbolic scene-graph changes this step is expected to produce.
    effects: list[RelationChange] = Field(default_factory=list)


class Plan(BaseModel):
    """A complete candidate plan for one natural-language task."""

    plan_id: str
    task: str
    rationale: str = ""
    steps: list[PlanStep] = Field(min_length=1)

    @field_validator("steps")
    @classmethod
    def _step_ids_unique(cls, steps: list[PlanStep]) -> list[PlanStep]:
        ids = [s.step_id for s in steps]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate step_id values in plan: {ids}")
        return steps
