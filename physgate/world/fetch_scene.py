"""Isaac Lab scene for the fetch-and-place MVP task.

Defines the physical world that mirrors the symbolic demo scene
(``examples_lib.fetch_and_place.build_demo_scene``):

* a Unitree Go2 robot,
* a fallen cardboard box (dynamic rigid body) on the floor,
* a target shelf (static),
* an obstacle pillar between the box and the shelf (static) — the deterministic
  navigation layer (``physgate/nav``) routes around it for every plan.

Spatial layout (per env, relative to the env origin) lives in
:mod:`physgate.world.layout` (pure logic, no Isaac import) and is shared by
trajectory synthesis (gate/trajectory), collision checks, navigation, and the
executor (executor/sim_backend).

IMPORTANT: this module imports Isaac Lab and may only be imported AFTER
``isaacsim.SimulationApp`` has been launched (headless or not).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# layout constants are pure logic — re-exported here for backwards compatibility
from physgate.world.layout import (  # noqa: F401  (re-exports)
    BOX_SIZE,
    CARRY_OFFSET,
    OBSTACLE_SIZE,
    ROBOT_BASE_HEIGHT,
    ROBOT_COLLISION_RADIUS,
    SCENE_LAYOUT,
    SEMANTIC_LABELS,
    SHELF_SIZE,
    SHELF_TOP_Z,
    STATIC_FOOTPRINTS,
)

# ---------------------------------------------------------------------- scene


@configclass
class FetchSceneCfg(InteractiveSceneCfg):
    """N-env fetch-and-place scene: Go2 + box + shelf + obstacle."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.5, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=SCENE_LAYOUT["box_03"]),
    )

    shelf = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.CuboidCfg(
            size=SHELF_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.6)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=SCENE_LAYOUT["shelf_A"]),
    )

    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=OBSTACLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=SCENE_LAYOUT["obstacle_P"]),
    )


# ----------------------------------------------------------------------- world


class FetchSimWorld:
    """Owns the Isaac simulation context + N-env fetch scene.

    One instance per process (Isaac allows a single SimulationContext).
    The Sim-Gate (gate/l2_physics.py) uses all N envs for parallel plan
    validation; the executor (executor/sim_backend.py) uses env 0.
    """

    #: prim entity name -> symbolic scene object id
    ENTITY_TO_OBJECT_ID = {"box": "box_03", "shelf": "shelf_A", "obstacle": "obstacle_P"}

    def __init__(self, num_envs: int = 8, device: str = "cuda:0", physics_dt: float = 0.005):
        # default physics_dt matches the Go2 locomotion policy's training timestep
        # (velocity_env_cfg: sim.dt = 0.005, decimation 4 -> 50 Hz policy)
        self.sim = SimulationContext(sim_utils.SimulationCfg(dt=physics_dt, device=device))
        self.scene = InteractiveScene(FetchSceneCfg(num_envs=num_envs, env_spacing=10.0))
        self.sim.reset()
        self.num_envs = num_envs
        self.device = device
        self.dt = self.sim.get_physics_dt()
        self._apply_semantic_labels()

    # ----- entities -----

    @property
    def robot(self):
        return self.scene["robot"]

    @property
    def box(self):
        return self.scene["box"]

    @property
    def env_origins(self) -> torch.Tensor:
        return self.scene.env_origins

    # ----- stepping -----

    def step(self) -> None:
        """One physics step (write -> step -> update)."""
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)

    def settle(self, steps: int = 60) -> None:
        """Let dynamics settle (e.g. after releasing an object)."""
        for _ in range(steps):
            self.step()

    # ----- state I/O -----

    def write_robot_poses(self, positions: torch.Tensor, quaternions: torch.Tensor) -> None:
        """Kinematically drive every robot base (world-frame positions, (N,3)/(N,4))."""
        pose = torch.cat([positions, quaternions], dim=-1)
        self.robot.write_root_pose_to_sim(pose)
        # zero velocities so physics does not accumulate momentum on the kinematic base
        self.robot.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, device=self.device))
        # hold the default standing joint configuration
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos.clone(),
            self.robot.data.default_joint_vel.clone(),
        )

    def apply_joint_targets(self, targets: torch.Tensor) -> None:
        """Drive the robots via joint position targets (policy control mode).

        Unlike :meth:`write_robot_poses` (kinematic mode), the base is NOT
        written — the robot moves itself through its actuators and physics.
        """
        self.robot.set_joint_position_target(targets)

    def write_box_poses(self, positions: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Kinematically place boxes (carried), world frame."""
        ids = env_ids if env_ids is not None else torch.arange(self.num_envs, device=self.device)
        quat = torch.zeros(len(ids), 4, device=self.device)
        quat[:, 0] = 1.0
        pose = torch.cat([positions, quat], dim=-1)
        self.box.write_root_pose_to_sim(pose, env_ids=ids)
        self.box.write_root_velocity_to_sim(
            torch.zeros(len(ids), 6, device=self.device), env_ids=ids
        )

    def release_boxes(self, velocities: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Release carried boxes with an initial velocity — physics takes over."""
        vel6 = torch.zeros(len(env_ids), 6, device=self.device)
        vel6[:, :3] = velocities
        self.box.write_root_velocity_to_sim(vel6, env_ids=env_ids)

    def box_positions(self) -> torch.Tensor:
        """Box positions per env, relative to each env origin. Shape (N, 3)."""
        return self.box.data.root_pos_w - self.env_origins

    def robot_positions(self) -> torch.Tensor:
        """Robot base positions per env, relative to each env origin. Shape (N, 3)."""
        return self.robot.data.root_pos_w - self.env_origins

    # ----- semantics -----

    def _apply_semantic_labels(self) -> None:
        """Tag scene prims with semantic labels for world/usd_semantics.py."""
        from isaaclab.sim.utils.semantics import add_labels
        from isaaclab.sim.utils.stage import get_current_stage

        stage = get_current_stage()
        prim_to_label = {
            "Box": SEMANTIC_LABELS["box_03"],
            "Shelf": SEMANTIC_LABELS["shelf_A"],
            "Obstacle": SEMANTIC_LABELS["obstacle_P"],
        }
        for env_idx in range(self.num_envs):
            for prim_name, label in prim_to_label.items():
                prim = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/{prim_name}")
                if prim and prim.IsValid():
                    add_labels(prim, labels=[label], instance_name="class")


# Module-level singleton so the gate and the executor share one simulation.
_shared_world: FetchSimWorld | None = None


def get_shared_world(num_envs: int = 8, device: str = "cuda:0") -> FetchSimWorld:
    """Get or create the process-wide FetchSimWorld."""
    global _shared_world
    if _shared_world is None:
        _shared_world = FetchSimWorld(num_envs=num_envs, device=device)
    return _shared_world
