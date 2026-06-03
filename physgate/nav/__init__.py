"""Deterministic navigation: the LOW level of the dual-system architecture.

The agent (LLM) emits semantic skills (``move_to_pose(<object id>)``); this
package guarantees obstacle avoidance deterministically — the agent never does
geometry. See ``docs/design/REBUILD.md`` §2.
"""

from physgate.nav.path_planner import (
    Obstacle,
    PathPlannerError,
    plan_path,
    plan_standoff_route,
    segment_clear,
)

__all__ = [
    "Obstacle",
    "PathPlannerError",
    "plan_path",
    "plan_standoff_route",
    "segment_clear",
]
