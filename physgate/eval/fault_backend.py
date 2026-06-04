"""Fault-injecting execution backend (recovery-scenario support).

Wraps :class:`MockWorldBackend` and makes the first N attempts of a chosen
skill fail — simulating transient physical failures (grasp slips) so the
orchestrator's retry/replan/escalate behaviour can be measured.
"""

from __future__ import annotations

from typing import Any

from physgate.eval.scenarios import FaultSpec
from physgate.executor.backend import MockWorldBackend
from physgate.gate.schemas import Scene


class FaultInjector:
    """Holds a fault budget ACROSS execution attempts.

    The orchestrator creates a fresh backend per execution attempt, but a
    transient fault (a grasp slip) is a property of the world, not of one
    backend instance — so the injection count lives here.
    """

    def __init__(self, fault: FaultSpec):
        """Initialize with a fault spec and zero injection count."""
        self.fault = fault
        self.injected = 0

    def backend_factory(self, scene: Scene) -> "FaultInjectionBackend":
        """WorldBackend factory compatible with the executor's backend_factory."""
        return FaultInjectionBackend(scene, self)


class FaultInjectionBackend:
    """MockWorldBackend wrapper that injects skill failures from a FaultInjector."""

    def __init__(self, scene: Scene, injector: FaultInjector):
        """Initialize with a mock backend and shared fault injector."""
        self._inner = MockWorldBackend(scene)
        self._injector = injector

    # ----- WorldBackend protocol -----

    def get_scene(self) -> Scene:
        """Return the current scene from the inner backend."""
        return self._inner.get_scene()

    def move_to_pose(self, target: str, standoff_m: float = 0.3, **kwargs: Any) -> dict[str, Any]:
        """Delegate move_to_pose to the inner backend."""
        return self._inner.move_to_pose(target=target, standoff_m=standoff_m, **kwargs)

    def execute_skill(self, skill: str, target: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a skill, injecting failures when the fault budget allows."""
        injector = self._injector
        if skill == injector.fault.skill and injector.injected < injector.fault.fail_count:
            injector.injected += 1
            return {
                "success": False,
                "error": (
                    f"injected transient {skill} failure "
                    f"#{injector.injected} (grasp slipped)"
                ),
            }
        return self._inner.execute_skill(skill=skill, target=target, **kwargs)
