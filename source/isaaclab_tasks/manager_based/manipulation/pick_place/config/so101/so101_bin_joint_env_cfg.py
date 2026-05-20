# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SO101 pick-place environment with joint-space action control.

Derived from so101_bin_ik_env_cfg.py with IK action term replaced by JointPositionActionCfg.
Action space: [arm_joint_pos(5), gripper_cmd(1)] = 6 dimensions.

.. code-block:: bash

    # Use from state machine script
    from isaaclab_tasks.manager_based.manipulation.pick_place.config.so101.so101_bin_joint_env_cfg import SO101BinPickPlaceJointEnvCfg
    env = ManagerBasedRLEnv(cfg=SO101BinPickPlaceJointEnvCfg())

"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg, Articulation, RigidObject
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import events as mdp_events
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp
from isaaclab_tasks.manager_based.manipulation.pick_place.pick_place_env_cfg import PickPlaceEnvCfg
import numpy as np
import torch
from torch import Tensor

##
# Pre-defined configs
##
from isaaclab_assets.robots.so_arm101 import SO_ARM101_CFG  # isort: skip

# ── Scene Constants ────────────────────────────────────────────────────────
BIN_USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd"
BIN_SCALE = (0.35, 0.35, 0.35)

CUBE_POSE_RANGE = {"x": (0.14, 0.25), "y": (-0.08, 0.08), "z": (0.015, 0.015)}
BIN_POSE_RANGE = {"x": (0.10, 0.21), "y": (0.09, 0.21), "z": (0.0, 0.0)}

MIN_CUBE_BIN_DISTANCE = 0.10
EE_TARGET_XY_OFFSET = (0.0, 0.02)
EE_WORKSPACE_RADIUS_RANGE = (0.16, 0.34)
MAX_LAYOUT_SAMPLE_TRIES = 100

# ── Helper Functions (same as IK config) ───────────────────────────────────

def _get_root_pos_w(asset) -> torch.Tensor:
    if hasattr(asset, "data") and hasattr(asset.data, "root_pos_w"):
        return asset.data.root_pos_w[:, :3]
    positions, _ = asset.get_world_poses()
    return positions[:, :3]


def box_position_in_robot_root_frame(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    box = env.scene[box_cfg.name]
    return _get_root_pos_w(box) - robot.data.root_pos_w[:, :3]


def ee_to_box_distance(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="gripper"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    box = env.scene[box_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :].clone()
    box_pos_w = _get_root_pos_w(box)
    return torch.norm(ee_pos_w - box_pos_w, dim=1, keepdim=True)


def cube_near_box(
    env,
    threshold: float,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> Tensor:
    cube: RigidObject = env.scene[cube_cfg.name]
    box = env.scene[box_cfg.name]
    distance = torch.norm(cube.data.root_pos_w[:, :3] - _get_root_pos_w(box), dim=1)
    return torch.where(distance < threshold, 1.0, 0.0)


def _sample_pose_values(
    num_samples: int,
    pose_range: dict[str, tuple[float, float]] | None,
    device: torch.device | str,
) -> torch.Tensor:
    pose_range = pose_range or {}
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=device)
    return math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_samples, 6), device=device)


def _workspace_mask(
    pose_samples: torch.Tensor,
    robot_root_xy: torch.Tensor,
    ee_target_xy_offset: tuple[float, float],
    ee_workspace_radius_range: tuple[float, float],
) -> torch.Tensor:
    target_xy_offset = torch.tensor(ee_target_xy_offset, device=pose_samples.device)
    target_xy = pose_samples[:, :2] + target_xy_offset
    target_radius = torch.linalg.norm(target_xy - robot_root_xy, dim=1)
    return (target_radius > ee_workspace_radius_range[0]) & (target_radius < ee_workspace_radius_range[1])


def _set_pose_midpoint(
    pose_samples: torch.Tensor,
    mask: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
) -> None:
    for axis_id, key in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
        low, high = pose_range.get(key, (0.0, 0.0))
        pose_samples[mask, axis_id] = 0.5 * (low + high)


def _write_dynamic_root_pose(
    asset: RigidObject,
    positions: torch.Tensor,
    orientations: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=positions.device), env_ids=env_ids)


def _write_kinematic_root_pose(
    asset: RigidObject,
    positions: torch.Tensor,
    orientations: torch.Tensor,
    env_ids: torch.Tensor,
) -> None:
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)


def reset_scene_to_default_except_kinematic_objects(
    env,
    env_ids: torch.Tensor | None,
    reset_joint_targets: bool = False,
    kinematic_rigid_object_names: tuple[str, ...] = ("box",),
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.scene.env_origins.shape[0], device=env.scene.env_origins.device)

    for rigid_object_name, rigid_object in env.scene.rigid_objects.items():
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        if rigid_object_name not in kinematic_rigid_object_names:
            rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

    for articulation_asset in env.scene.articulations.values():
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)

        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        if reset_joint_targets:
            articulation_asset.set_joint_position_target(default_joint_pos, env_ids=env_ids)
            articulation_asset.set_joint_velocity_target(default_joint_vel, env_ids=env_ids)

    for deformable_object in env.scene.deformable_objects.values():
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


def randomize_cube_and_box_positions(
    env,
    env_ids: torch.Tensor | None,
    cube_pose_range: dict[str, tuple[float, float]],
    box_pose_range: dict[str, tuple[float, float]],
    min_cube_box_distance: float,
    max_sample_tries: int = MAX_LAYOUT_SAMPLE_TRIES,
    ee_target_xy_offset: tuple[float, float] = EE_TARGET_XY_OFFSET,
    ee_workspace_radius_range: tuple[float, float] = EE_WORKSPACE_RADIUS_RANGE,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.scene.env_origins.shape[0], device=env.scene.env_origins.device)

    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    num_envs = len(env_ids)
    robot_root_xy = robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]

    cube_samples = _sample_pose_values(num_envs, cube_pose_range, cube.device)
    box_samples = _sample_pose_values(num_envs, box_pose_range, cube.device)
    for _ in range(max_sample_tries):
        cube_valid = _workspace_mask(cube_samples, robot_root_xy, ee_target_xy_offset, ee_workspace_radius_range)
        box_valid = _workspace_mask(box_samples, robot_root_xy, ee_target_xy_offset, ee_workspace_radius_range)
        too_close = torch.linalg.norm(cube_samples[:, :2] - box_samples[:, :2], dim=1) < min_cube_box_distance
        invalid_cube = ~cube_valid
        invalid_box = ~box_valid | too_close
        if not bool(torch.any(invalid_cube | invalid_box).item()):
            break
        cube_samples[invalid_cube] = _sample_pose_values(
            int(invalid_cube.sum().item()), cube_pose_range, cube.device
        )
        box_samples[invalid_box] = _sample_pose_values(
            int(invalid_box.sum().item()), box_pose_range, cube.device
        )

    cube_valid = _workspace_mask(cube_samples, robot_root_xy, ee_target_xy_offset, ee_workspace_radius_range)
    box_valid = _workspace_mask(box_samples, robot_root_xy, ee_target_xy_offset, ee_workspace_radius_range)
    too_close = torch.linalg.norm(cube_samples[:, :2] - box_samples[:, :2], dim=1) < min_cube_box_distance
    invalid = ~cube_valid | ~box_valid | too_close
    _set_pose_midpoint(cube_samples, invalid, cube_pose_range)
    _set_pose_midpoint(box_samples, invalid, box_pose_range)

    cube_positions = cube_samples[:, :3] + env.scene.env_origins[env_ids]
    box_positions = box_samples[:, :3] + env.scene.env_origins[env_ids]
    cube_orientations = math_utils.quat_from_euler_xyz(
        cube_samples[:, 3], cube_samples[:, 4], cube_samples[:, 5]
    )
    box_orientations = math_utils.quat_from_euler_xyz(
        box_samples[:, 3], box_samples[:, 4], box_samples[:, 5]
    )

    _write_dynamic_root_pose(cube, cube_positions, cube_orientations, env_ids)
    _write_kinematic_root_pose(box, box_positions, box_orientations, env_ids)


# ── Env Config ─────────────────────────────────────────────────────────────

@configclass
class SO101BinPickPlaceJointEnvCfg(PickPlaceEnvCfg):
    """SO101 pick-place environment with joint-space action control.

    Action space: [arm_joint_pos(5), gripper_cmd(1)] = 6 dimensions.
    - arm_joint_pos: absolute joint positions for Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll
    - gripper_cmd: positive=open, negative=close (BinaryJointPositionAction)
    """

    def __post_init__(self):
        super().__post_init__()

        # Simulation settings (same as IK config)
        self.sim.dt = 1.0 / 120.0
        self.decimation = 1
        self.sim.render_interval = 4

        # Robot
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ── Actions: Joint-space (no IK) ────────────────────────────────
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
            scale=1.0,
            use_default_offset=False,  # Action = absolute joint position
        )

        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw"],
            open_command_expr={"Jaw": 1.0},
            close_command_expr={"Jaw": 0.0},
        )

        # ── Scene Objects (same as IK config) ───────────────────────────
        self.scene.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.20, 0.0, 0.015]),
            spawn=sim_utils.CuboidCfg(
                size=(0.03, 0.03, 0.03),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            ),
        )

        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.15, 0.15, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=BIN_USD_PATH,
                scale=BIN_SCALE,
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_depenetration_velocity=5.0,
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
        )

        # Gripper camera
        self.scene.camera_ego = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper/gripper_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                projection_type="pinhole",
                f_stop=100.0,
                focal_length=13.5,
                focus_distance=0.05,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.005, 0.06, -0.062),
                rot=euler_angles_to_quat(np.array([-45, 0, 0]), degrees=True),
                convention="opengl",
            ),
        )

        # ── Events (same as IK config) ──────────────────────────────────
        self.events.reset_all = EventTerm(
            func=reset_scene_to_default_except_kinematic_objects,
            mode="reset",
            params={"kinematic_rigid_object_names": {"box"}},
        )
        self.events.reset_cube_position = None
        self.events.randomize_cube_and_box = EventTerm(
            func=randomize_cube_and_box_positions,
            mode="reset",
            params={
                "cube_pose_range": CUBE_POSE_RANGE,
                "box_pose_range": BIN_POSE_RANGE,
                "min_cube_box_distance": MIN_CUBE_BIN_DISTANCE,
                "max_sample_tries": MAX_LAYOUT_SAMPLE_TRIES,
                "ee_target_xy_offset": EE_TARGET_XY_OFFSET,
                "ee_workspace_radius_range": EE_WORKSPACE_RADIUS_RANGE,
                "robot_cfg": SceneEntityCfg("robot"),
                "cube_cfg": SceneEntityCfg("cube"),
                "box_cfg": SceneEntityCfg("box"),
            },
        )

        # ── Observations (same as IK config) ────────────────────────────
        self.observations.policy.box_position = ObsTerm(func=box_position_in_robot_root_frame)
        self.observations.policy.ee_to_box_dist = ObsTerm(
            func=ee_to_box_distance,
            params={"robot_cfg": SceneEntityCfg("robot", body_names="gripper")},
        )

        # ── Rewards (same as IK config) ─────────────────────────────────
        self.rewards.cube_near_box = RewTerm(
            func=cube_near_box, params={"threshold": 0.1}, weight=20.0
        )


@configclass
class SO101BinPickPlaceJointEnvCfg_PLAY(SO101BinPickPlaceJointEnvCfg):
    """Play mode configuration for joint-space SO101 pick-place."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
