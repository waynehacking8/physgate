"""Tests for E3 procedural scenario generation with certified solvability.

Every generated instance carries its ground-truth label BY CONSTRUCTION
(PlanBench's approach): feasible instances are certified solvable by the A*
planner; infeasible instances are certified unreachable by the same planner.
No manual annotation.
"""

from __future__ import annotations

import pytest

from physgate.nav.path_planner import PathPlannerError, plan_standoff_route
from physgate.world.layout import ROBOT_COLLISION_RADIUS, navigation_obstacles, target_half_extents


def _route_exists(layout: dict, frm: str, to: str, standoff: float = 0.3) -> bool:
    try:
        plan_standoff_route(
            tuple(layout[frm][:2]),
            tuple(layout[to][:2]),
            standoff=standoff,
            obstacles=navigation_obstacles(layout),
            robot_radius=ROBOT_COLLISION_RADIUS,
            target_half_extents=target_half_extents(to),
        )
        return True
    except PathPlannerError:
        return False


def test_feasible_instances_are_certified_solvable():
    from physgate.eval_v2.scenario_gen import generate_instances

    instances = generate_instances(n_feasible=10, n_infeasible=0, seed=42)
    assert len(instances) == 10
    for inst in instances:
        assert inst.feasible is True
        # certification holds: robot->box and box->shelf routes exist
        assert _route_exists(inst.layout, "go2", "box_03")
        assert _route_exists(inst.layout, "box_03", "shelf_A", standoff=0.4)


def test_infeasible_instances_are_certified_unreachable():
    from physgate.eval_v2.scenario_gen import generate_instances

    instances = generate_instances(n_feasible=0, n_infeasible=5, seed=42)
    assert len(instances) == 5
    for inst in instances:
        assert inst.feasible is False
        # the box is unreachable: no standoff route exists
        assert not _route_exists(inst.layout, "go2", "box_03")


def test_generation_is_deterministic_per_seed():
    from physgate.eval_v2.scenario_gen import generate_instances

    a = generate_instances(n_feasible=5, n_infeasible=2, seed=7)
    b = generate_instances(n_feasible=5, n_infeasible=2, seed=7)
    assert [i.layout for i in a] == [i.layout for i in b]

    c = generate_instances(n_feasible=5, n_infeasible=2, seed=8)
    assert [i.layout for i in a] != [i.layout for i in c]


def test_layouts_vary_across_instances():
    """Generalization requires actual variation — not the same layout repeated."""
    from physgate.eval_v2.scenario_gen import generate_instances

    instances = generate_instances(n_feasible=10, n_infeasible=0, seed=1)
    box_positions = {tuple(round(v, 2) for v in i.layout["box_03"][:2]) for i in instances}
    assert len(box_positions) >= 8, "generated layouts barely vary"


def test_instances_have_matching_symbolic_scenes():
    """Every instance carries a Scene whose objects match the layout entities."""
    from physgate.eval_v2.scenario_gen import generate_instances

    instances = generate_instances(n_feasible=3, n_infeasible=1, seed=3)
    for inst in instances:
        scene_ids = {o.id for o in inst.scene.objects}
        assert {"box_03", "shelf_A", "obstacle_P", "go2", "floor_01"} <= scene_ids
        assert inst.scene.has_relation("box_03", "on", "floor_01")


def test_entities_do_not_overlap_in_feasible_layouts():
    from physgate.eval_v2.scenario_gen import MIN_SEPARATION_M, generate_instances

    instances = generate_instances(n_feasible=10, n_infeasible=0, seed=11)
    for inst in instances:
        entities = ["box_03", "shelf_A", "obstacle_P"]
        for i, a in enumerate(entities):
            for b in entities[i + 1 :]:
                ax, ay = inst.layout[a][:2]
                bx, by = inst.layout[b][:2]
                dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                assert dist >= MIN_SEPARATION_M, f"{a} and {b} overlap in {inst.instance_id}"


def test_generation_fails_loudly_on_impossible_constraints():
    from physgate.eval_v2.scenario_gen import generate_instances

    with pytest.raises(ValueError, match="workspace"):
        generate_instances(n_feasible=1, n_infeasible=0, seed=1, workspace=(0.1, 0.1))
