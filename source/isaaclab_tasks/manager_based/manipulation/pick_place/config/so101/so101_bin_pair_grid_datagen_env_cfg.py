#from __future__ import annotaions

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.rotations import euler_angles_to_quat
import isaaclab.utils.math as math_utils
import torch
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp
from isaaclab_tasks.manager_based.manipulation.pick_place.pick_place_env_cfg import (
    ObservationsCfg,
    PickPlaceSceneCfg
)

from isaaclab_assets.robots.so_arm101 import SO_ARM101_CFG

from .so101_bin_ik_env_cfg import (
    BIN_SCALE,
    BIN_USD_PATH,
    CUBE_POSE_RANGE,
    EE_TARGET_XY_OFFSET,
    EE_WORKSPACE_RADIUS_RANGE,
    MIN_CUBE_BIN_DISTANCE,
    _workspace_mask,
    _write_dynamic_root_pose,
    _write_kinematic_root_pose,
    box_position_in_robot_root_frame,
    ee_to_box_distance,
    reset_scene_to_default_except_kinematic_objects,
)

PAIR_GRID_X_RANGE = (0.13, 0.24)
PAIR_GRID_Y_RANGE = (-0.12, 0.20)
PAIR_GRID_CUBE_POSE_RANGE = {
    "x": PAIR_GRID_X_RANGE,
    "y": PAIR_GRID_Y_RANGE,
    "z": CUBE_POSE_RANGE["z"],
}
PAIR_GRID_BOX_POSE_RANGE = {
    "x": PAIR_GRID_X_RANGE,
    "y": PAIR_GRID_Y_RANGE,
    "z": (0.0, 0.0),
}
PAIR_GRID_SHAPE = (5, 5)
PAIR_GRID_CELL_JITTER_FRACTION = 0.60
PAIR_GRID_SHUFFLE_PAIRS_EACH_CYCLE = True
PAIR_GRID_MAX_JITTER_SAMPLE_TRIES = 20


def _sample_pose_midpoints(
    num_samples: int,
    pose_range: dict[str, tuple[float, float]],
    device: torch.device | str
) -> torch.Tensor:
    pose_samples = torch.zeros((num_samples, 6), device=device)
    for axis_id, key in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
        low, high = pose_range.get(key, (0.0, 0.0))
        pose_samples[:, axis_id] = 0.5 * (low + high)
    return pose_samples

def _sample_grid_pose_values(
    num_samples: int,
    cell_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    grid_shape: tuple[int, int],
    jitter_fraction: float,
    device: torch.device | str,
) -> torch.Tensor:
    num_x, num_y = grid_shape
    x_min, x_max = pose_range["x"]
    y_min, y_max = pose_range["y"]
    cell_width = (x_max - x_min) / float(num_x)
    cell_height = (y_max - y_min) / float(num_y)
    
    pose_samples = _sample_pose_midpoints(num_samples, pose_range, device)
    cell_ids = cell_ids.to(device=device, dtype=torch.long)
    cell_x = cell_ids % num_x
    cell_y = torch.div(cell_ids, num_x, rounding_mode="floor")
    
    pose_samples[:, 0] = x_min + (cell_x.float() + 0.5) * cell_width
    pose_samples[:, 1] = y_min + (cell_y.float() + 0.5) * cell_height
    
    jitter_fraction = max(0.0, min(float(jitter_fraction), 1.0))
    if jitter_fraction > 0.0:
        pose_samples[:, 0] += (torch.rand(num_samples, device=device) - 0.5) * cell_width * jitter_fraction
        pose_samples[:, 1] += (torch.rand(num_samples, device=device) - 0.5) * cell_height * jitter_fraction
        
    return pose_samples

def _validate_pose_pairs(
    cube_samples: torch.Tensor,
    box_samples: torch.Tensor,
    robot_root_xy: torch.Tensor,
    min_cube_box_distance: float,
    ee_target_xy_offset: tuple[float, float],
    ee_workspace_radius_range: tuple[float, float],
) -> torch.Tensor:
    cube_valid = _workspace_mask(cube_samples, robot_root_xy, ee_target_xy_offset, ee_workspace_radius_range)
    box_valid = _workspace_mask(box_samples, robot_root_xy, ee_target_xy_offset, ee_workspace_radius_range)
    far_enough = torch.linalg.norm(cube_samples[:, :2] - box_samples[:, :2], dim=-1) >= min_cube_box_distance
    return cube_valid & box_valid & far_enough

def _build_valid_grid_pair_ids(
    grid_shape: tuple[int, int],
    cube_pose_range: dict[str, tuple[float, float]],
    box_pose_range: dict[str, tuple[float, float]],
    robot_root_xy: torch.Tensor,
    min_cube_box_distance: float,
    ee_target_xy_offset: tuple[float, float],
    ee_workspace_radius_range: tuple[float, float],
    device: torch.device | str,
) -> torch.Tensor:
    num_cells = grid_shape[0] * grid_shape[1]
    if num_cells <= 0:
        raise ValueError(f"Invalid grid shape: {grid_shape}. Both dimension must be positive.")
    
    cube_cell_ids = torch.arange(num_cells, device=device, dtype=torch.long).repeat_interleave(num_cells)
    box_cell_ids = torch.arange(num_cells, device=device, dtype=torch.long).repeat(num_cells)
    cube_samples = _sample_grid_pose_values(
        cube_cell_ids.shape[0], cube_cell_ids, cube_pose_range, grid_shape, 0.0, device
    )
    box_samples = _sample_grid_pose_values(
        box_cell_ids.shape[0], box_cell_ids, box_pose_range, grid_shape, 0.0, device
    )    
    valid = _validate_pose_pairs(
        cube_samples,
        box_samples,
        robot_root_xy[:1],
        min_cube_box_distance,
        ee_target_xy_offset,
        ee_workspace_radius_range,
    )
    valid_pair_ids = torch.stack((cube_cell_ids[valid], box_cell_ids[valid]), dim=-1)
    if valid_pair_ids.shape[0] == 0:
        raise RuntimeError(
            "No valid cube/bin grid pairs were found. Reduce min_cube_box_distance or shrink the grid range."
        )
    return valid_pair_ids

def _get_valid_grid_pair_ids(
    env,
    grid_shape: tuple[int, int],
    cube_pose_range: dict[str, tuple[float, float]],
    box_pose_range: dict[str, tuple[float, float]],
    robot_root_xy: torch.Tensor,
    min_cube_box_distance: float,
    ee_target_xy_offset: tuple[float, float],
    ee_workspace_radius_range: tuple[float, float],
    device: torch.device | str,
) -> torch.Tensor:
    if not hasattr(env, "_so101_pair_grid_pair_ids"):
        env._so101_pair_grid_pair_ids = _build_valid_grid_pair_ids(
            grid_shape,
            cube_pose_range,
            box_pose_range,
            robot_root_xy,
            min_cube_box_distance,
            ee_target_xy_offset,
            ee_workspace_radius_range,
            device,
        )
    return env._so101_pair_grid_pair_ids

def _next_grid_pair_ids(
    env,
    num_samples: int,
    valid_pair_ids: torch.Tensor,
    device: torch.device | str,
    shuffle_each_cycle: bool,
) -> torch.Tensor:
    num_pairs = valid_pair_ids.shape[0]
    if num_pairs <= 0:
        raise ValueError("valid_pair_ids must contain at least one pair")
        
    counter = int(getattr(env, "_so101_pair_grid_reset_counter", 0))
    pair_indices = torch.arange(counter, counter + num_samples, device=device, dtype=torch.long) % num_pairs
    if shuffle_each_cycle:
        shuffled_indices = []
        for sequence_id in range(counter, counter + num_samples):
            cycle_position = sequence_id % num_pairs
            if cycle_position == 0 or not hasattr(env, "_so101_pair_grid_pair_order"):
                env._so101_pair_grid_pair_order = torch.randperm(num_pairs, device=device)
            shuffled_indices.append(env._so101_pair_grid_pair_order[cycle_position])
        pair_indices = torch.stack(shuffled_indices)
    env._so101_pair_grid_reset_counter = counter + num_samples
    return valid_pair_ids[pair_indices]

def pair_grid_randomize_cube_and_box_positions(
    env,
    env_ids: torch.Tensor | None,
    cube_pose_range: dict[str, tuple[float, float]],
    box_pose_range: dict[str, tuple[float, float]],
    min_cube_box_distance: float,
    grid_shape: tuple[int, int] = PAIR_GRID_SHAPE,
    cell_jitter_fraction: float = PAIR_GRID_CELL_JITTER_FRACTION,
    shuffle_pairs_each_cycle: bool = PAIR_GRID_SHUFFLE_PAIRS_EACH_CYCLE,
    max_jitter_sample_tries: int = PAIR_GRID_MAX_JITTER_SAMPLE_TRIES,
    ee_target_xy_offset: tuple[float, float]= EE_TARGET_XY_OFFSET,
    ee_workspace_radius_range: tuple[float, float]= EE_WORKSPACE_RADIUS_RANGE,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> None:
    if env_ids is None:
        env_ids = torch.arange(env.scene.env_origins.shape[0], device=env.scene.env_origins.device)
    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]
    box = env.scene[box_cfg.name]
    num_envs = len(env_ids)
    robot_root_xy = robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    
    valid_pair_ids = _get_valid_grid_pair_ids(
        env,
        grid_shape,
        cube_pose_range,
        box_pose_range,
        robot_root_xy,
        min_cube_box_distance,
        ee_target_xy_offset,
        ee_workspace_radius_range,
        cube.device,
    )
    pair_ids = _next_grid_pair_ids(env, num_envs, valid_pair_ids, cube.device, shuffle_pairs_each_cycle)
    cube_cell_ids = pair_ids[:, 0]
    box_cell_ids = pair_ids[:, 1]
    
    cube_samples = _sample_grid_pose_values(
        num_envs, cube_cell_ids, cube_pose_range, grid_shape, cell_jitter_fraction, cube.device
    )
    box_samples = _sample_grid_pose_values(
        num_envs, box_cell_ids, box_pose_range, grid_shape, cell_jitter_fraction, cube.device
    )
    
    valid = _validate_pose_pairs(
        cube_samples,
        box_samples,
        robot_root_xy,
        min_cube_box_distance,
        ee_target_xy_offset,
        ee_workspace_radius_range,
    )
    for _ in range(max_jitter_sample_tries):
        invalid = ~valid
        if not bool(torch.any(invalid).item()):
            break
        invalid_count = int(invalid.sum().item())
        
        cube_samples[invalid] = _sample_grid_pose_values(
            invalid_count,
            cube_cell_ids[invalid],
            cube_pose_range,
            grid_shape,
            cell_jitter_fraction,
            cube.device
        )
        box_samples[invalid] = _sample_grid_pose_values(
            invalid_count,
            box_cell_ids[invalid],
            box_pose_range,
            grid_shape,
            cell_jitter_fraction,
            cube.device
        )
        valid = _validate_pose_pairs(
            cube_samples,
            box_samples,
            robot_root_xy,
            min_cube_box_distance,
            ee_target_xy_offset,
            ee_workspace_radius_range,
        )
    
    invalid = ~valid
    if bool(torch.any(invalid).item()):
        invalid_count = int(invalid.sum().item())
        cube_samples[invalid] = _sample_grid_pose_values(
            invalid_count,
            cube_cell_ids[invalid],
            cube_pose_range,
            grid_shape,
            0.0,
            cube.device
        )
        box_samples[invalid] = _sample_grid_pose_values(
            invalid_count,
            box_cell_ids[invalid],
            box_pose_range,
            grid_shape,
            0.0,
            cube.device
        )
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
    
@configclass
class PairGridDatagenEventCfg:
    
    reset_all = EventTerm(
        func=reset_scene_to_default_except_kinematic_objects,
        mode="reset",
        params={"kinematic_rigid_object_names": ("box",)}        
    )
    randomize_cube_and_box = EventTerm(
        func=pair_grid_randomize_cube_and_box_positions,
        mode="reset",
        params={
            "cube_pose_range": PAIR_GRID_CUBE_POSE_RANGE,
            "box_pose_range": PAIR_GRID_BOX_POSE_RANGE,
            "min_cube_box_distance": MIN_CUBE_BIN_DISTANCE,
            "grid_shape": PAIR_GRID_SHAPE,
            "cell_jitter_fraction": PAIR_GRID_CELL_JITTER_FRACTION,
            "shuffle_pairs_each_cycle": PAIR_GRID_SHUFFLE_PAIRS_EACH_CYCLE,
            "max_jitter_sample_tries": PAIR_GRID_MAX_JITTER_SAMPLE_TRIES,
            "ee_target_xy_offset": EE_TARGET_XY_OFFSET,
            "ee_workspace_radius_range": EE_WORKSPACE_RADIUS_RANGE,
            "robot_cfg": SceneEntityCfg("robot"),
            "cube_cfg": SceneEntityCfg("cube"),
            "box_cfg": SceneEntityCfg("box"),
        }
    )
    
@configclass
class JointActionCfg:
    
    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"],
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
        preserve_order=True,
    )
    gripper_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Jaw"],
        scale=1.0,
        offset=0.0,
        use_default_offset=False,
        preserve_order=True,
    )

@configclass
class SO101BinPickPlacePairGridDatagenEnvCfg(ManagerBasedEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: JointActionCfg = JointActionCfg()
    events: PairGridDatagenEventCfg = PairGridDatagenEventCfg()
    
    def __post_init__(self):
        self.decimation = 1
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.sim.sync_real_time = True
        
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        self.scene.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.20, 0.0, 0.015], rot=[1.0, 0.0, 0.0, 0.0]),
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
            )
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
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
        )
        
        self.scene.camera_ego = TiledCameraCfg(
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
            offset=TiledCameraCfg.OffsetCfg(
                pos=(-0.005, 0.06, -0.062),
                rot=euler_angles_to_quat(np.array([-45, 0, 0]), degrees=True),
                convention="opengl"
            )
        )
        
        self.observations.policy.box_position = ObsTerm(func=box_position_in_robot_root_frame)
        self.observations.policy.ee_to_box_dist = ObsTerm(
            func=ee_to_box_distance,
            params={"robot_cfg": SceneEntityCfg("robot", body_names="gripper")},
        )
        
        
@configclass
class SO101BinPickPlacePairGridDatagenEnvCfg_PLAY(SO101BinPickPlacePairGridDatagenEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 20.5
        self.observations.policy.enable_corruption = False