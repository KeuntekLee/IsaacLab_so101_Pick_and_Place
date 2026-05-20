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

from isaaclab_tasks.manager_based.manipulation.pick_place import mdp
from isaaclab_tasks.manager_based.manipulation.pick_place.pick_place_env_cfg import (
    ObservationsCfg,
    PickPlaceSceneCfg
)

from isaaclab_assets.robots.so_arm101 import SO_ARM101_CFG

from .so101_bin_ik_env_cfg import (
    BIN_POSE_RANGE,
    BIN_SCALE,
    BIN_USD_PATH,
    CUBE_POSE_RANGE,
    EE_TARGET_XY_OFFSET,
    EE_WORKSPACE_RADIUS_RANGE,
    MAX_LAYOUT_SAMPLE_TRIES,
    MIN_CUBE_BIN_DISTANCE,
    box_position_in_robot_root_frame,
    ee_to_box_distance,
    randomize_cube_and_box_positions,
    reset_scene_to_default_except_kinematic_objects,
)

@configclass
class JointDatagenEventCfg:
    
    reset_all = EventTerm(
        func=reset_scene_to_default_except_kinematic_objects,
        mode="reset",
        params={"kinematic_rigid_object_names": ("box",)}        
    )
    randomize_cube_and_box = EventTerm(
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
class SO101BinPickPlaceJointDatagenEnvCfg(ManagerBasedEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: JointActionCfg = JointActionCfg()
    events: JointDatagenEventCfg = JointDatagenEventCfg()
    
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
class SO101BinPickPlaceJointDatagenEnvCfg_PLAY(SO101BinPickPlaceJointDatagenEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False