"""Same-state parallel reset workaround (Isaac Lab bug #2133) — C10.

Best-of-N validation requires every parallel env to start from an IDENTICAL
initial state, otherwise plan scores are not comparable. Isaac Lab's stock
``env.reset()`` randomizes each env independently (and bug #2133 makes
identical resets unreliable through the manager API).

Workaround: bypass the reset managers and write the default (deterministic)
root/joint state of every articulation and rigid object directly into the
simulation for ALL envs, offset only by each env's origin, then step a few
times so PhysX settles contacts.

IMPORTANT: import only after SimulationApp launch.
"""

from __future__ import annotations

import torch
from isaaclab.scene import InteractiveScene


def reset_scene_to_identical_state(scene: InteractiveScene, sim, settle_steps: int = 10) -> None:
    """Reset every env of an InteractiveScene to the same deterministic state.

    Args:
        scene: the interactive scene (N envs).
        sim: the SimulationContext (for stepping).
        settle_steps: physics steps after the state write so contacts settle.
    """
    # Articulations: default root state + env origin, default joints.
    for articulation in scene.articulations.values():
        root_state = articulation.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        articulation.write_root_pose_to_sim(root_state[:, :7])
        articulation.write_root_velocity_to_sim(root_state[:, 7:])
        articulation.write_joint_state_to_sim(
            articulation.data.default_joint_pos.clone(),
            articulation.data.default_joint_vel.clone(),
        )
        # Actuator TARGETS must reset too: stale position targets from a
        # previous rollout (e.g. a crouched/stalled robot) would drive the
        # joints back toward that posture during the settle steps below,
        # breaking reset identity and cascading failures across rollouts.
        articulation.set_joint_position_target(articulation.data.default_joint_pos.clone())
        articulation.set_joint_velocity_target(articulation.data.default_joint_vel.clone())
        articulation.set_joint_effort_target(
            torch.zeros_like(articulation.data.default_joint_pos)
        )

    # Rigid objects: default root state + env origin.
    for rigid_object in scene.rigid_objects.values():
        root_state = rigid_object.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        rigid_object.write_root_pose_to_sim(root_state[:, :7])
        rigid_object.write_root_velocity_to_sim(root_state[:, 7:])

    # Clear internal buffers (sensors, command managers, ...).
    scene.reset()

    # Let PhysX settle the contacts deterministically.
    dt = sim.get_physics_dt()
    for _ in range(settle_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)


def max_state_deviation(scene: InteractiveScene) -> float:
    """Maximum cross-env deviation of entity states, relative to env origins.

    Returns 0.0 when every env is in an identical state. Used to *verify* the
    workaround (the test asserts this stays below a tolerance after reset).
    """
    deviations: list[float] = []

    for articulation in scene.articulations.values():
        local_pos = articulation.data.root_pos_w - scene.env_origins
        deviations.append((local_pos - local_pos[0:1]).abs().max().item())
        joint_pos = articulation.data.joint_pos
        deviations.append((joint_pos - joint_pos[0:1]).abs().max().item())

    for rigid_object in scene.rigid_objects.values():
        local_pos = rigid_object.data.root_pos_w - scene.env_origins
        deviations.append((local_pos - local_pos[0:1]).abs().max().item())

    return max(deviations) if deviations else 0.0


def verify_identical_reset(
    scene: InteractiveScene, sim, tolerance: float = 1e-3, settle_steps: int = 10
) -> tuple[bool, float]:
    """Run the workaround and verify all envs ended up identical.

    Returns:
        (ok, max_deviation): ok is True when the deviation is within tolerance.
    """
    reset_scene_to_identical_state(scene, sim, settle_steps=settle_steps)
    deviation = max_state_deviation(scene)
    return deviation <= tolerance, deviation
