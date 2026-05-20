import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pick and place state machine for SO101 environments with LeRobot data collection.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--dataset_dir", type=str, default=None, help="Directory to save LeRobot dataset (e.g., ./datasets/so101_pick_and_place)."
)
parser.add_argument(
    "--num_episodes", type=int, default=10, help="Number of episodes to record (0 = infinite)."
)
parser.add_argument(
    "--vcodec", type=str, default="libsvtav1", help="Video codec: libsvtav1 (AV1), h264, hevc."
)
parser.add_argument(
    "--save_failed_episodes",
    action="store_true",
    default=False,
    help="Save episodes that did not meet the success criteria (cube not placed within threshold distance of target).",
)
parser.add_argument(
    "--streaming_encoding", action="store_true", default=True, help="Use streaming video encoding (faster save_episode)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import CameraCfg,TiledCameraCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat
import isaaclab_tasks

from isaaclab_tasks.manager_based.manipulation.pick_place.config.so101.so101_bin_pair_grid_datagen_env_cfg import (
    PAIR_GRID_BOX_POSE_RANGE,
    PAIR_GRID_CELL_JITTER_FRACTION,
    PAIR_GRID_CUBE_POSE_RANGE,
    PAIR_GRID_SHAPE,
    PAIR_GRID_SHUFFLE_PAIRS_EACH_CYCLE,
    SO101BinPickPlacePairGridDatagenEnvCfg,
)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("[WARNING] LeRobot not installed. Data Collection disabled.")
    
    
DATASET_FPS = 30
EXPECTED_CONTROL_HZ = 120
EXPECTED_RECORD_INTERVAL = EXPECTED_CONTROL_HZ // DATASET_FPS
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640
GRIPPER_VIDEO_KEY = "observation.images.gripper"
PHONE_VIDEO_KEY = "observation.images.phone"

ARM_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
ISAAC_SO101_JOINT_NAMES = [*ARM_JOINT_NAMES, "Jaw"]
LEROBOT_SO101_FEATURE_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos"
]

STATE_MACHINE_EPISODE_STEP = 350
RECORD_START_STEP = 0
RECORD_END_STEP = 300
EPISODE_TIMEOUT_MARGIN_INTERVALS = 2
MAX_TARGET_STEP_DIST = 0.02
SUCCESS_THRESHOLD = 0.05
BOX_PLACEMENT_OFFSET_W = (0.0, 0.0, 0.0)
JOINT_LIMIT_SATURATION_TOL = 1.0e-3

LINK_OFFSET = (0.0, 0.01, 0.105)
HOME_EE_POS_W = (0.050, 0.021, 0.126)
HOME_EE_QUAT_WXYZ = (-0.693, -0.140, 0.140, 0.693)
TARGET_EE_QUAT_WXYZ = (-0.7071, 0.0, 0.0, 0.7071)
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
JAW_OPEN_POS = 1.0
JAW_CLOSE_POS = 0.0
TOP_CAMERA_POS = (0.18, 0.025, 0.85)

def phase_from_step(step: int) -> int:
    if step < 50:
        return 0
    if step < 100:
        return 1
    if step < 150:
        return 2
    if step < 200:
        return 3
    if step < 250:
        return 4
    if step < 300:
        return 5
    return 6

class TorchPickPlaceStateMachine:
    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        self.step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.des_ee_pos = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        self.des_ee_quat = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
        self.des_gripper = torch.full((num_envs,), GRIPPER_CLOSE, dtype=torch.float32, device=device)
        self._home_pos = torch.tensor(HOME_EE_POS_W, dtype=torch.float32, device=device)
        self._home_quat = torch.tensor(HOME_EE_QUAT_WXYZ, dtype=torch.float32, device=device)
        self._target_quat = torch.tensor(TARGET_EE_QUAT_WXYZ, dtype=torch.float32, device=device)
        self._link_offset = torch.tensor(LINK_OFFSET, dtype=torch.float32, device=device)
        self.reset_idx()
        
    def reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        self.step_count[env_ids] = 0
        self.des_ee_pos[env_ids] = self._home_pos
        self.des_ee_quat[env_ids] = self._home_quat
        self.des_gripper[env_ids] = GRIPPER_CLOSE
        
    def compute(
        self,
        ee_pos_w: torch.Tensor,
        ee_quat_w: torch.Tensor,
        cube_pos_w: torch.Tensor,
        box_pos_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del ee_pos_w, ee_quat_w
        step = self.step_count % STATE_MACHINE_EPISODE_STEP
        #mask = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        mask = step == 0
        #print("COMPUTE1", step,mask)
        self.des_ee_pos[mask] = cube_pos_w[mask] + self._link_offset + torch.tensor(
            (0.0, 0.0, 0.05), dtype=torch.float32, device=self.device
        )
        
        self.des_ee_quat[mask] = self._target_quat
        self.des_gripper[mask] = GRIPPER_OPEN
        
        
        mask = step == 50
        self.des_ee_pos[mask] = cube_pos_w[mask] + self._link_offset
        self.des_ee_quat[mask] = self._target_quat
        self.des_gripper[mask] = GRIPPER_OPEN
        
        mask = step == 100
        self.des_ee_pos[mask] = cube_pos_w[mask] + self._link_offset
        self.des_ee_quat[mask] = self._target_quat
        self.des_gripper[mask] = GRIPPER_CLOSE
        
        mask = step == 150
        self.des_ee_pos[mask] = cube_pos_w[mask] + self._link_offset + torch.tensor(
            (0.0, 0.0, 0.08), dtype=torch.float32, device=self.device
        )
        self.des_ee_quat[mask] = self._target_quat
        self.des_gripper[mask] = GRIPPER_CLOSE

        mask = step == 200
        self.des_ee_pos[mask] = box_pos_w[mask] + torch.tensor(
            (0.0, 0.0, 0.18), dtype=torch.float32, device=self.device
        )
        self.des_ee_quat[mask] = self._target_quat
        self.des_gripper[mask] = GRIPPER_CLOSE

        mask = step == 250
        self.des_ee_pos[mask] = box_pos_w[mask] + torch.tensor(
            (0.0, 0.0, 0.18), dtype=torch.float32, device=self.device
        )
        self.des_ee_quat[mask] = self._target_quat
        self.des_gripper[mask] = GRIPPER_OPEN

        mask = step == 300
        self.des_ee_pos[mask] = self._home_pos
        self.des_ee_quat[mask] = self._target_quat
        self.des_gripper[mask] = GRIPPER_CLOSE

        output = torch.cat((self.des_ee_pos, self.des_ee_quat, self.des_gripper.unsqueeze(-1)), dim=-1)
        self.step_count += 1
        return output, step
    
class SO101DiffIKTeacher:
    
    def __init__(self, robot, num_envs: int, device: str):
        self.robot = robot
        self.device = device
        missing_joint_names = [joint_name for joint_name in ARM_JOINT_NAMES if joint_name not in robot.joint_names]
        if missing_joint_names:
            raise ValueError(f"Robot is missing arm joints required by the IK teacher: {missing_joint_names}")
        self.joint_names = ARM_JOINT_NAMES
        self.joint_ids = [robot.joint_names.index(joint_name) for joint_name in ARM_JOINT_NAMES]
        body_ids, body_names = robot.find_bodies("gripper")
        if len(body_ids) != 1:
            raise ValueError(f"Expected one bdy named 'gripper', found {len(body_ids)}: {body_names}")
        self.body_idx = body_ids[0]
        if robot.is_fixed_base:
            self.jacobi_body_idx = self.body_idx - 1
            self.jacobi_joint_ids = self.joint_ids
        else:
            self.jacobi_body_idx = self.body_idx
            self.jacobi_joint_ids = [joint_id + 6 for joint_id in self.joint_ids]
        self.controller = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=num_envs,
            device=device
        )
        
    def reset(self) -> None:
        self.controller.reset()
        
    def compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
        return math_utils.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )
    
    def compute_frame_jacobian(self) -> torch.Tensor:
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.jacobi_body_idx, :, self.jacobi_joint_ids
        ].clone()
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(self.robot.data.root_quat_w))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian
    
    def compute(self, target_pose_b: torch.Tensor) -> torch.Tensor:
        ee_pos_b, ee_quat_b = self.compute_frame_pose()
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        if ee_quat_b.norm() == 0:
            return joint_pos.clone()
        
        self.controller.set_command(target_pose_b, ee_pos_b, ee_quat_b)
        return self.controller.compute(ee_pos_b, ee_quat_b, self.compute_frame_jacobian(), joint_pos)
    
def get_box_placement_target(box_pos_w: torch.Tensor) -> torch.Tensor:
    offset = torch.tensor(BOX_PLACEMENT_OFFSET_W, dtype=box_pos_w.dtype, device=box_pos_w.device)
    return box_pos_w + offset

def check_success(cube_pos_w: torch.Tensor, box_pos_w: torch.Tensor, threshold: float = SUCCESS_THRESHOLD) -> bool:
    return bool(torch.any(check_success_mask(cube_pos_w, box_pos_w, threshold)).item())

def check_success_mask(
    cube_pos_w: torch.Tensor,
    box_pos_w: torch.Tensor,
    threshold: float = SUCCESS_THRESHOLD,
) -> torch.Tensor:
    placement_target_w = get_box_placement_target(box_pos_w)
    xy_distance = torch.linalg.norm(cube_pos_w[:, :2] - placement_target_w[:, :2], dim=-1)
    return xy_distance < threshold
        
def get_box_pos_w(box) -> torch.Tensor:
    if hasattr(box, "data") and hasattr(box.data, "root_pos_w"):
        return box.data.root_pos_w[:, :3].clone()
    box_pos_w, _ = box.get_world_poses()
    return box_pos_w[:, :3].clone()

def create_lerobot_dataset(dataset_dir: str, vcodec: str = "libsvtav1", streaming_encoding: bool = True):
    """Create a LeRobot v3.0 dataset for SO101 pick-place.

    Args:
        dataset_dir: Directory to save the dataset.
        vcodec: Video codec (libsvtav1, h264, hevc).
        streaming_encoding: Use streaming video encoding.

    Returns:
        LeRobotDataset instance or None if LeRobot not available.
    """
    if not LEROBOT_AVAILABLE:
        return None

    # Define features matching SO101 robot
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),  # 5 joint positions + 1 gripper
            "names": LEROBOT_SO101_FEATURE_NAMES,
            "fps": DATASET_FPS,
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),  # 5 joint positions + 1 gripper
            "names": LEROBOT_SO101_FEATURE_NAMES,
            "fps": DATASET_FPS,
        },
    }
    for video_key in (GRIPPER_VIDEO_KEY, PHONE_VIDEO_KEY):
        features[video_key] = {
            "dtype": "video",
            "shape": (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": CAMERA_HEIGHT,
                "video.width": CAMERA_WIDTH,
                "video.codec": vcodec,
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": DATASET_FPS,
                "video.channels": 3,
                "has_audio": False,
            },
        }

    # Create dataset (detect supported params for version compatibility)
    kwargs = {
        "repo_id": "so101_pick_place_bin_joint_next_state",
        "root": dataset_dir,
        "fps": DATASET_FPS,
        "features": features,
    }

    # Check if newer params are supported (lerobot >= 0.4.0)
    import inspect
    sig = inspect.signature(LeRobotDataset.create)
    if "vcodec" in sig.parameters:
        kwargs["vcodec"] = vcodec
    if "streaming_encoding" in sig.parameters:
        kwargs["streaming_encoding"] = streaming_encoding

    print(f"[LeRobot] Creating dataset with kwargs: {list(kwargs.keys())}")
    return LeRobotDataset.create(**kwargs)
    
def get_ordered_joint_ids(robot, joint_names: Sequence[str] = ISAAC_SO101_JOINT_NAMES) -> list[int]:
    missing_joint_names = [joint_name for joint_name in joint_names if joint_name not in robot.joint_names]
    if missing_joint_names:
        raise ValueError(f"Robot is missing joints required by the dataset: {missing_joint_names}")
    return [robot.joint_names.index(joint_name) for joint_name in joint_names]

def get_observation_state(robot, ordered_joint_ids: Sequence[int], env_idx: int = 0) -> torch.Tensor:
    return robot.data.joint_pos[env_idx, ordered_joint_ids]

def compute_gripper_joint_target(gripper_cmd: torch.Tensor) -> torch.Tensor:
    open_target = torch.full(
        (gripper_cmd.shape[0], 1), JAW_OPEN_POS, dtype=gripper_cmd.dtype, device=gripper_cmd.device
    )
    close_target = torch.full(
        (gripper_cmd.shape[0], 1), JAW_CLOSE_POS, dtype=gripper_cmd.dtype, device=gripper_cmd.device
    )
    return torch.where(gripper_cmd.unsqueeze(-1) < 0.0, close_target, open_target)

def clamp_joint_targets(robot, joint_ids: Sequence[int], joint_targets: torch.Tensor) -> torch.Tensor:
    limits = robot.data.soft_joint_pos_limits[:, joint_ids, :]
    return torch.maximum(torch.minimum(joint_targets, limits[..., 1]), limits[..., 0])

def add_top_camera_to_scene_cfg(scene_cfg) -> None:
    scene_cfg.camera_top = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraTop",
        update_period=0.0,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            projection_type="pinhole",
            f_stop=1000.0,
            focal_length=20.0,
            focus_distance=TOP_CAMERA_POS[2],
            #clipping_range=(0.05, 2.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.75, 0.025, 0.85),
            rot=euler_angles_to_quat(np.array([35, 0, 90]), degrees=True),
            #pos=TOP_CAMERA_POS, 
            #rot=(1.0, 0.0, 0.0, 0.0), 
            convention="opengl")
    )
    
def camera_rgb_to_uint8(camera_data: torch.Tensor, env_idx: int = 0) -> np.ndarray:
    rgb_frame = camera_data[env_idx].detach().cpu().numpy().copy()
    if rgb_frame.shape[-1] > 3:
        rgb_frame = rgb_frame[..., :3]
    if rgb_frame.dtype != np.uint8:
        if rgb_frame.max() > 1.0:
            rgb_frame = rgb_frame / 255.0
        rgb_frame = np.clip(rgb_frame * 255.0, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb_frame).copy()

def build_lerobot_observation_frame(
    robot,
    ordered_joint_ids: Sequence[int],
    gripper_camera_data: torch.Tensor,
    phone_camera_data: torch.Tensor | None = None,
    action: torch.Tensor | np.ndarray | None = None,
    task: str = "Pick the cube and place it in the bin",
    env_idx: int = 0,
) -> dict:
    observation_state = (
        get_observation_state(robot, ordered_joint_ids, env_idx).detach().cpu().numpy().astype(np.float32, copy=True)
    )
    frame = {
        "observation.state": observation_state,
        GRIPPER_VIDEO_KEY: camera_rgb_to_uint8(gripper_camera_data, env_idx),
        "task": task,
    }
    if phone_camera_data is not None:
        frame[PHONE_VIDEO_KEY] = camera_rgb_to_uint8(phone_camera_data, env_idx)
    if action is not None:
        if isinstance(action, torch.Tensor):
            action = action[env_idx].detach().cpu().numpy()
        frame["action"] = np.asarray(action, dtype=np.float32).copy()
    return frame

def validate_command_action_episode_frames(episode_frames: Sequence[dict]) -> list[dict]:
    completed_frames = []
    for frame_idx, frame in enumerate(episode_frames):
        if "action" not in frame:
            raise KeyError(f"Buffered frame {frame_idx} is missing an IK command action.")
        completed_frames.append(dict(frame))
        
    return completed_frames

def get_record_interval(control_dt: float) -> tuple[int, float]:
    control_hz = 1.0 / control_dt
    record_interval = int(round(control_hz / DATASET_FPS))
    if record_interval <= 0:
        raise ValueError(f"Invalid record interval {record_interval} from control_hz={control_hz: .3f}")
    if abs(control_hz - record_interval * DATASET_FPS) > 1e-3:
        raise ValueError(f"Control rate {control_hz: .3f} Hz is not an integer multiple of {DATASET_FPS} Hz.")
    if record_interval != EXPECTED_RECORD_INTERVAL:
        print(
            f"[Timing] WARNING: exppected {EXPECTED_CONTROL_HZ} Hz / {DATASET_FPS} Hz = "
            f"{EXPECTED_RECORD_INTERVAL}, got control_hz={control_hz:.3f}, interval={record_interval}."
        )
    return record_interval, control_hz

def get_required_episode_length_s(control_dt: float, record_interval: int) -> float:
    state_machine_control_steps = STATE_MACHINE_EPISODE_STEP * record_interval
    margin_control_steps = EPISODE_TIMEOUT_MARGIN_INTERVALS * record_interval
    return (state_machine_control_steps + margin_control_steps) * control_dt

def compute_joint_policy_action(
    robot,
    teacher: SO101DiffIKTeacher,
    home_joint_target: torch.Tensor,
    ordered_joint_ids: Sequence[int],
    raw_actions_w: torch.Tensor,
    ee_pos_w: torch.Tensor,
    step_at_command: torch.Tensor,
    return_limit_satureation: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del home_joint_target, step_at_command
    
    final_target_pos_w = raw_actions_w[:, :3]
    diff_pos = final_target_pos_w - ee_pos_w
    dist = torch.norm(diff_pos, dim=-1, keepdim=True)
    step_sizes = torch.clamp(dist, max=MAX_TARGET_STEP_DIST)
    smoothed_target_pos_w = ee_pos_w + (diff_pos / (dist + 1.0e-6)) * step_sizes
    
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w
    target_pos_b = math_utils.quat_rotate_inverse(base_quat_w, smoothed_target_pos_w - base_pos_w)
    _, target_quat_b = math_utils.subtract_frame_transforms(
        base_pos_w,
        base_quat_w,
        smoothed_target_pos_w,
        raw_actions_w[:, 3:7],
    )
    
    arm_joint_target = teacher.compute(torch.cat((target_pos_b, target_quat_b), dim=-1))
    jaw_target = compute_gripper_joint_target(raw_actions_w[:, -1])
    joint_action = torch.cat((arm_joint_target, jaw_target), dim=-1)
    if joint_action.shape[-1] != len(ISAAC_SO101_JOINT_NAMES):
        raise RuntimeError(f"Expected SO101 joint action shape (*,6) got {tuple(joint_action.shape)}")
    clamped_joint_action = clamp_joint_targets(robot, ordered_joint_ids, joint_action)
    if return_limit_satureation:
        limit_saturated = torch.any(
            torch.abs(joint_action - clamped_joint_action) > JOINT_LIMIT_SATURATION_TOL,
            dim=-1
        )
        return clamped_joint_action, limit_saturated
    return clamped_joint_action
    #return clamp_joint_targets(robot, ordered_joint_ids, joint_action)

def main():
    env_cfg = SO101BinPickPlacePairGridDatagenEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    record_interval, control_hz = get_record_interval(control_dt)
    _ = get_required_episode_length_s(control_dt, record_interval)
    add_top_camera_to_scene_cfg(env_cfg.scene)
    if args_cli.dataset_dir is not None:
        env_cfg.num_rerenders_on_reset = max(env_cfg.num_rerenders_on_reset, 20)
        
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    
    pick_sm = TorchPickPlaceStateMachine(num_envs=env.num_envs, device=env.device)
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    box = env.scene["box"]
    gripper_camera = env.scene["camera_ego"]
    top_camera = env.scene["camera_top"]
    ee_idx = robot.body_names.index("gripper")
    ordered_joint_ids = get_ordered_joint_ids(robot)
    teacher = SO101DiffIKTeacher(robot=robot, num_envs=env.num_envs, device=env.device)
    
    lerobot_dataset = None
    if args_cli.dataset_dir is not None and LEROBOT_AVAILABLE:
        print(f"[LeRobot] Creating dataset at: {args_cli.dataset_dir}")
        lerobot_dataset = create_lerobot_dataset(
            dataset_dir=args_cli.dataset_dir,
            vcodec=args_cli.vcodec,
            streaming_encoding=args_cli.streaming_encoding,
        )
        print("[LeRobot] Dataset created successfully")   
    elif args_cli.dataset_dir is not None and not LEROBOT_AVAILABLE:
        print("[LeRobot] WARNING: LeRobot not installed. Data collection disabled.")
    
    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), dtype=torch.float32, device=env.device)
    
    state_names = ["APPROACH_ABOVE", "APPROACH_CUBE", "GRASP", "LIFT", "MOVE_BOX", "RELEASE", "GOHOME"]
    main._last_state = -1
    
    saved_episode_count = 0
    attempted_episode_count = 0
    current_episode_step = 0
    episode_control_step = 0
    total_control_step = 0
    pending_episode_save = False
    force_policy_update = True
    is_recording = lerobot_dataset is not None
    current_joint_target = None
    joint_increment = None
    episode_record_frames = [[] for _ in range(env.num_envs)]
    bad_episode = [False for _ in range(env.num_envs)]
    home_joint_target = robot.data.default_joint_pos[:, ordered_joint_ids].clone()
    valid_pair_ids = getattr(env, "_so101_pair_grid_pair_ids", None)
    valid_pair_count = int(valid_pair_ids.shape[0]) if valid_pair_ids is not None else 0
    
    def target_episode_count_reached() -> bool:
        if args_cli.num_episodes <= 0:
            return False
        if is_recording:
            return saved_episode_count >= args_cli.num_episodes
        return attempted_episode_count >= args_cli.num_episodes

    def finish_episode_batch(reason: str, success_mask: torch.Tensor, control_steps: int) -> None:
        nonlocal attempted_episode_count, saved_episode_count, episode_record_frames, bad_episode
        
        success_values = success_mask.detach().cpu().tolist()
        save_this_batch = 0
        discarded_this_batch = 0
        for env_idx in range(env.num_envs):
            attempted_episode_count += 1
            success = bool(success_values[env_idx])
            frame_count = len(episode_record_frames[env_idx])
            should_save = frame_count > 0 and not bad_episode[env_idx] and (
                success or args_cli.save_failed_episodes
            )
            
            if lerobot_dataset is not None and should_save and not target_episode_count_reached():
                try:
                    for frame in validate_command_action_episode_frames(episode_record_frames[env_idx]):
                        lerobot_dataset.add_frame(frame)
                    lerobot_dataset.save_episode()
                    saved_episode_count += 1
                    save_this_batch += 1
                    print(
                        f"[Episode {saved_episode_count}] Saved "
                        f"(env={env_idx}, attempt={attempted_episode_count}, reason={reason}, "
                        f"success={success}, frames={frame_count}, control_steps={control_steps})"
                    )
                except Exception as e:
                    bad_episode[env_idx] = True
                    discarded_this_batch += 1
                    if hasattr(lerobot_dataset, "clear_episode_buffer"):
                        lerobot_dataset.clear_episode_buffer()
                    print(f"[WARN] env={env_idx} save_episode failed; discarding buffered episode: {e}")
            else:
                discarded_this_batch += 1
                if not is_recording:
                    print(
                        f"[Attempt {attempted_episode_count}] Finished "
                        f"(env={env_idx}, reason={reason}, success={success}, "
                        f"frames={frame_count}, bad={bad_episode[env_idx]}, control_steps={control_steps})"
                    )
                elif frame_count > 0:
                    print(
                        f"[Discard] env={env_idx} attempt={attempted_episode_count} "
                        f"(reason={reason}, success={success}, bad={bad_episode[env_idx]}, "
                        f"frames={frame_count})"
                    )
        print(
            f"[Batch] attempts={attempted_episode_count}, saved={saved_episode_count}, "
            f"saved_this_batch={save_this_batch}, discarded_this_batch={discarded_this_batch}"
        )
        episode_record_frames = [[] for _ in range(env.num_envs)]
        bad_episode = [False for _ in range(env.num_envs)]
        
    def reset_episode_state() -> None:
        nonlocal current_joint_target, joint_increment, episode_control_step, force_policy_update
        nonlocal pending_episode_save, home_joint_target, episode_record_frames, current_episode_step
        env.reset()
        pick_sm.reset_idx()
        teacher.reset()
        actions.zero_()
        current_joint_target = None
        joint_increment = None
        episode_record_frames = [[] for _ in range(env.num_envs)]
        bad_episode = [False for _ in range(env.num_envs)]
        episode_control_step = 0
        pending_episode_save = False
        force_policy_update = True
        home_joint_target = robot.data.default_joint_pos[:, ordered_joint_ids].clone()
        main._last_state = -1
        
    print(f"\n{'=' * 60}")
    print("SO101 Pick-Place Joint-Space State Machine")
    print(f"{'=' * 60}")
    print(f"Environments: {env.num_envs}")
    print(f"Device: {env.device}")
    print(f"Control rate: {control_hz:.1f} Hz")
    print(f"Dataset rate: {DATASET_FPS} Hz (record_interval={record_interval})")
    print("Action space: 6D next actual joint position")
    print(f"Joint order: {ISAAC_SO101_JOINT_NAMES}")
    print(
        f"Shared pair grid: {PAIR_GRID_SHAPE}, cube={PAIR_GRID_CUBE_POSE_RANGE} "
        f"box={PAIR_GRID_BOX_POSE_RANGE}"
    )
    print(
        f"Pair sampling: valid pairs={valid_pair_count}, jitter={PAIR_GRID_CELL_JITTER_FRACTION:.2f}, "
        f"shuffle={PAIR_GRID_SHUFFLE_PAIRS_EACH_CYCLE}"
    )
    print(f"Recording: {'YES' if is_recording else 'NO'}")
    if is_recording:
        print(f"Dataset: {args_cli.dataset_dir}")
        print(f"Max episodes: {args_cli.num_episodes}")
        print(f"Video codec: {args_cli.vcodec}")
        print(f"External camera key: {PHONE_VIDEO_KEY}")
        print(f"State-machine episode steps: {STATE_MACHINE_EPISODE_STEP}")
        print(f"Recorded state-machine steps: [{RECORD_START_STEP}, {RECORD_END_STEP}]")
        print("Action label: rolled next observation.state per episode")
    print(f"{'=' * 60}\n")
    
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if target_episode_count_reached():
                    if is_recording:
                        print(f"\n[Done] Saved {saved_episode_count} episodes. Exiting.")
                    else:
                        print(f"\n[Done] Completed {attempted_episode_count} episodes. Exiting.")
                    break
                
                if pending_episode_save and episode_control_step % record_interval == 0:
                    cube_pos_w = cube.data.root_pos_w[:, :3].clone()
                    box_pos_w = get_box_pos_w(box)
                    success_mask = check_success_mask(cube_pos_w, box_pos_w)
                    finish_episode_batch("state_machine_cycle", success_mask, episode_control_step)
                    reset_episode_state()
                    continue
                
                ee_pos_w = robot.data.body_pos_w[:, ee_idx, :].clone()
                ee_quat_w = robot.data.body_quat_w[:, ee_idx, :].clone()
                cube_pos_w = cube.data.root_pos_w[:, :3].clone()
                box_pos_w = get_box_pos_w(box)
                
                policy_update_due = force_policy_update or episode_control_step % record_interval == 0
                if policy_update_due:
                    raw_actions_w, step_at_command = pick_sm.compute(ee_pos_w, ee_quat_w, cube_pos_w, box_pos_w)
                    
                    command_step = int(step_at_command[0].item())
                    current_phase = phase_from_step(command_step)
                    is_record_step = RECORD_START_STEP <= command_step < RECORD_END_STEP
                    should_record_frame = (
                        is_recording
                        and lerobot_dataset is not None
                        and episode_control_step % record_interval == 0
                        and is_record_step
                    )
                    
                    gripper_camera_data = None
                    phone_camera_data = None
                    if should_record_frame and gripper_camera is not None:
                        try:
                            env.sim.render()
                            gripper_camera_data = gripper_camera.data.output["rgb"]
                            if top_camera is not None:
                                phone_camera_data = top_camera.data.output["rgb"]
                        except Exception as e:
                            print(f"[ERROR] Failed to capture camera frame: {e}")
                    
                    policy_joint_action, limit_saturated = compute_joint_policy_action(
                        robot=robot,
                        teacher=teacher,
                        home_joint_target=home_joint_target,
                        ordered_joint_ids=ordered_joint_ids,
                        raw_actions_w=raw_actions_w,
                        ee_pos_w=ee_pos_w,
                        step_at_command=step_at_command,
                        return_limit_satureation=True,
                    )
                    if is_record_step:
                        limit_saturated_values = limit_saturated.detach().cpu().tolist()
                        for env_idx, saturated in enumerate(limit_saturated_values):
                            if saturated and not bad_episode[env_idx]:
                                bad_episode[env_idx] = True
                                episode_record_frames[env_idx] = []
                                print(
                                    f"[Discard] env={env_idx} marked bad due to joint limit saturation "
                                    f"at state_machine_step={command_step}"
                                )
                                
                    has_camera_data = gripper_camera_data is not None and (
                        top_camera is None or phone_camera_data is not None
                    )
                    if should_record_frame and has_camera_data:
                        for env_idx in range(env.num_envs):
                            if bad_episode[env_idx]:
                                continue
                            try:
                                episode_record_frames[env_idx].append(
                                    build_lerobot_observation_frame(
                                        robot,
                                        ordered_joint_ids,
                                        gripper_camera_data,
                                        phone_camera_data,
                                        action=policy_joint_action,
                                        env_idx=env_idx,
                                    )
                                )
                                frame_count = len(episode_record_frames[env_idx])
                                if frame_count % 100 == 0:
                                    print(f"[LeRobot] env={env_idx} buffered {frame_count} frames")
                            except Exception as e:
                                bad_episode[env_idx] = True
                                episode_record_frames[env_idx] = []
                                print(f"[ERROR] env={env_idx} failed to buffer frame; discarding episode: {e}")
                                import traceback
                                
                                traceback.print_exc()
                    elif should_record_frame:
                        for env_idx in range(env.num_envs):
                            if not bad_episode[env_idx]:
                                bad_episode[env_idx] = True
                                episode_record_frames[env_idx] = []
                        print("[ERROR] Camera data missing for a recording step; discarding current vector batch")
                     
                    if current_joint_target is None:
                        current_joint_target = robot.data.joint_pos[:, ordered_joint_ids].clone()
                    joint_increment = (policy_joint_action - current_joint_target) / float(record_interval)
                    force_policy_update = False
                    
                    if int(pick_sm.step_count[0].item()) >= STATE_MACHINE_EPISODE_STEP:
                        pending_episode_save = True
                    
                    if current_phase != main._last_state:
                        tgt = pick_sm.des_ee_pos[0].cpu().numpy()
                        action_np = policy_joint_action[0].detach().cpu().numpy()
                        print(
                            f" STATE->{state_names[current_phase]} "
                            f"step={command_step} "
                            f"ee_tgt=({tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f}) "
                            f"teacher_joint_tgt=({', '.join(f'{v:.3f}' for v in action_np)})"
                        )
                        main._last_state = current_phase
                        
                if current_joint_target is not None and joint_increment is not None:
                    current_joint_target = current_joint_target + joint_increment
                    current_joint_target = clamp_joint_targets(robot, ordered_joint_ids, current_joint_target)
                    actions[:] = current_joint_target
                
                _, _ = env.step(actions)
                total_control_step += 1
                episode_control_step += 1
                
                if total_control_step % 100 == 0 or total_control_step <= 3:
                    ee_pos_np = ee_pos_w[0].cpu().numpy()
                    tgt_np = pick_sm.des_ee_pos[0].cpu().numpy()
                    cube_np = cube_pos_w[0].cpu().numpy()
                    jaw_pos = robot.data.joint_pos[:, ordered_joint_ids[-1]][0].item()
                    print(
                        f" step={total_control_step} "
                        f"ee_pos=({ee_pos_np[0]:.3f},{ee_pos_np[1]:.3f},{ee_pos_np[2]:.3f}) "
                        f"tgt=({tgt_np[0]:.3f},{tgt_np[1]:.3f},{tgt_np[2]:.3f}) "
                        f"cube=({cube_np[0]:.3f},{cube_np[1]:.3f},{cube_np[2]:.3f}) "
                        f"jaw_pos-{jaw_pos:.3f}"
                    )
    except KeyboardInterrupt:
        print("\n[Interrupted] Saving current episode...")
    finally:
        if lerobot_dataset is not None:
            print("[LeRobot] Finalizing dataset...")
            try:
                if current_episode_step > 0:
                    cube_pos_w = cube.data.root_pos_w[:, :3].clone()
                    box_pos_w = get_box_pos_w(box)
                    success_mask = check_success_mask(cube_pos_w, box_pos_w)
                    finish_episode_batch("interrupted_or_shutdown", success_mask, episode_control_step)
            except Exception as e:
                print(f"[WARN] Failed to save final episode: {e}")
            try:
                lerobot_dataset.finalize()
                print("[LeRobot] Dataset finalized successfully")
            except Exception as e:
                print(f"[WARN] Failed to finalize dataset: {e}")
        env.close()
        simulation_app.close()
        
if __name__ == "__main__":
    main()