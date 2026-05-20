import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="LeRobot Inferece for Simulation.")
parser.add_argument("--policy_path", type=str, required=True, help="Path to the policy checkpoint.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes to run.")
parser.add_argument("--episode_policy_steps", type=int, default=400, help="Number of steps to run the policy for in each episode.")
parser.add_argument("--action_hz", type=float, default=30.0, help="Hz at which to run the policy.")
parser.add_argument("--policy_device", type=str, default="cuda:0", help="Device to run the policy on.")
parser.add_argument("--n_action_steps", type=int, default=1, help="Number of simulation steps to apply each action for.")
parser.add_argument("--task", type=str, default="Pick the cube and place it in the bin", help="Description of the task to be performed, used for prompting the policy.")
parser.add_argument("--use_amp", action="store_true", default=False, help="Whether to use automatic mixed precision for policy inference.")
parser.add_argument(
    "--binary_gripper",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to use binary open/close commands for the gripper. If False, the policy will output continuous position commands for the gripper.",
)
parser.add_argument(
    "--gripper_threshold",
    type=float,
    default=0.5,
    help="Threshold for determining the gripper state when using binary gripper actions."
)
parser.add_argument("--gripper_open_target", type=float, default=1.0, help="Target gripper joint position for the open command when using binary gripper actions.")
parser.add_argument("--gripper_close_target", type=float, default=0.0, help="Target gripper joint position for the close command when using binary gripper actions.")
parser.add_argument(
    "--phone_camera_source",
    type=str,
    default="top",
    choices=("top", "gripper"),
    help="Which camera to use as input to the policy. Can be either 'top' for the overhead camera or 'gripper' for the gripper-mounted camera.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Whether to disable the fabric simulation for the bin. Disabling fabric can improve simulation performance at the cost of realism."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import CameraCfg, TiledCameraCfg

import isaaclab_tasks

from isaaclab_tasks.manager_based.manipulation.pick_place.config.so101.so101_bin_joint_datagen_env_cfg import (
    SO101BinPickPlaceJointDatagenEnvCfg,
)
from isaacsim.core.utils.rotations import euler_angles_to_quat
try:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.control_utils import predict_action
    
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("LeRobot is not available. Please install LeRobot to run this script.")
    
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640
TOP_CAMERA_POS = (0.18, 0.025, 0.85)
SUCCESS_THRESHOLD = 0.05
BOX_PLACEMENT_OFFSET_W = (0.0, 0.0, 0.0)
POLICY_CONFIG_FILE = "config.json"
POLICY_WEIGHTS_FILE = "model.safetensors"
PRETRAINED_MODEL_DIR = "pretrained_model"

OBS_STATE_KEY = "observation.state"
GRIPPER_VIDEO_KEY = "observation.images.gripper"
TOP_VIDEO_KEY = "observation.images.top"
PHONE_VIDEO_KEY = "observation.images.phone"
CAMERA1_VIDEO_KEY = "observation.images.camera1"
CAMERA2_VIDEO_KEY = "observation.images.camera2"
CAMERA3_VIDEO_KEY = "observation.images.camera3"

ARM_JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
ISAAC_SO101_JOINT_NAMES = [*ARM_JOINT_NAMES, "Jaw"]

def add_top_camera_to_scene_cfg(scene_cfg) -> None:
    scene_cfg.camera_top = CameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraTop",
        update_period=0.0,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
        spawn=sim_utils.PinholeCameraCfg(
            projection_type="pinhole",
            f_stop=100.0,
            focal_length=18.0,
            focus_distance=TOP_CAMERA_POS[2],
            clipping_range=(0.05, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            #pos=(0.75, 0.025, 0.85),
            #rot=euler_angles_to_quat(np.array([35, 0, 90]), degrees=True),
            pos=TOP_CAMERA_POS, 
            rot=(1.0, 0.0, 0.0, 0.0), 
            convention="opengl")
    )
    
    scene_cfg.recording_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RecordingCamera",  # 대소문자 명확히 구분
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        # 💡 매우 중요: CameraCfg 대신 PinholeCameraCfg를 지정해야 실제로 카메라 객체가 스폰됩니다.
        spawn=sim_utils.PinholeCameraCfg(
            projection_type="pinhole",
            f_stop=1000.0,                # 팬포커스 (전체 선명하게)
            focal_length=24.0,         # 화각 조절 (숫자가 작을수록 넓게 보임)
            clipping_range=(0.1, 10.0)
        ),
        # 💡 아까 잡아두신 마음에 드는 eye / target 좌표를 기반으로 오프셋을 설정합니다.
        offset=CameraCfg.OffsetCfg(
            pos=(1.5, 1.5, 1.8),       # 카메라 위치 (X, Y, Z)
            #target=(0.0, 0.0, 0.1),    # 카메라가 바라볼 로봇 중심점 (X, Y, Z)
            rot=(0.280, 0.123, 0.392, 0.867),
            convention="opengl"
        ),
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

def get_ordered_joint_ids(robot, joint_names: Sequence[str] = ISAAC_SO101_JOINT_NAMES) -> list[int]:
    missing_joint_names = [joint_name for joint_name in joint_names if joint_name not in robot.joint_names]
    if missing_joint_names:
        raise ValueError(f"Robot is missing joints required by the dataset: {missing_joint_names}")
    return [robot.joint_names.index(joint_name) for joint_name in joint_names]

def get_observation_state(robot, ordered_joint_ids: Sequence[int], env_idx: int = 0) -> torch.Tensor:
    return robot.data.joint_pos[env_idx, ordered_joint_ids].detach().cpu().numpy().astype(np.float32, copy=True)

def clamp_joint_targets(robot, joint_ids: Sequence[int], joint_targets: torch.Tensor) -> torch.Tensor:
    limits = robot.data.soft_joint_pos_limits[:, joint_ids, :]
    return torch.maximum(torch.minimum(joint_targets, limits[..., 1]), limits[..., 0])

def apply_binary_gripper_target(
    joint_targets: torch.Tensor,
    threshold: float,
    open_target: float,
    close_target: float,
) -> torch.Tensor:
    processed_targets = joint_targets.clone()
    jaw = processed_targets[:, -1]
    processed_targets[:, -1] = torch.where(
        jaw < threshold,
        torch.full_like(jaw, close_target),
        torch.full_like(jaw, open_target),
    )
    return processed_targets

def get_box_pos_w(box) -> torch.Tensor:
    if hasattr(box, "data") and hasattr(box.data, "root_pos_w"):
        return box.data.root_pos_w[:, :3].clone()
    box_pos_w, _ = box.get_world_poses()
    return box_pos_w[:, :3].clone()

def check_success(cube_pos_w: torch.Tensor, box_pos_w: torch.Tensor, threshold: float = SUCCESS_THRESHOLD) -> bool:
    offset = torch.tensor(BOX_PLACEMENT_OFFSET_W, dtype=box_pos_w.dtype, device=box_pos_w.device)
    placement_target_w = box_pos_w + offset
    xy_distance = torch.linalg.norm(cube_pos_w[:, :2] - placement_target_w[:, :2], dim=-1)
    return bool((xy_distance < threshold).any())

def get_policy_interval(control_dt: float, action_hz: float) -> tuple[int, float]:
    control_hz = 1.0 / control_dt
    interval = int(round(control_hz / action_hz))
    if interval <= 0:
        raise ValueError(f"Invalid policy interval {interval} from control_hz={control_hz:.3f}, action_hz={action_hz}.")
    if abs(control_hz - interval * action_hz) > 1.0e-3:
        raise ValueError(f"Control rate {control_hz:.3f} Hz is not an integer multiple of action_hz={action_hz}.")
    return interval, control_hz

def reset_processor(processor) -> None:
    if hasattr(processor, "reset"):
        processor.reset()
        
def resolve_policy_path(policy_path: str) -> str:
    path = Path(policy_path).expanduser()
    if not path.exists():
        return policy_path
    
    if path.is_dir() and (path / POLICY_CONFIG_FILE).is_file() and (path / POLICY_WEIGHTS_FILE).is_file():
        return str(path)
    
    nested_pretrained_path = path / PRETRAINED_MODEL_DIR
    if (
        nested_pretrained_path.is_dir()
        and (nested_pretrained_path / POLICY_CONFIG_FILE).is_file()
        and (nested_pretrained_path / POLICY_WEIGHTS_FILE).is_file()
    ):
        print(f"[Policy] Using nested pretrained model: {nested_pretrained_path}")
        return str(nested_pretrained_path)
    
    if path.is_dir() and (path / POLICY_CONFIG_FILE).is_file():
        raise FileNotFoundError(
            f"Found {path / POLICY_CONFIG_FILE}, but {path / POLICY_WEIGHTS_FILE} is missing. "
        )
    
    raise FileNotFoundError(
        f"Could not find a LeRobot pretrained_model at {path}. Expected either:\n"
        f"   {path / POLICY_CONFIG_FILE}\n"
        f"   {path / POLICY_WEIGHTS_FILE}\n"
        f"or a nested directory:\n"
        f"   {path / PRETRAINED_MODEL_DIR / POLICY_CONFIG_FILE}\n"
        f"   {path / PRETRAINED_MODEL_DIR / POLICY_WEIGHTS_FILE}"
    )
    
def load_lerobot_policy(policy_path: str, device: torch.device):
    if not LEROBOT_AVAILABLE:
        raise ImportError("LeRobot is not installed in this env.")
    
    policy_path = resolve_policy_path(policy_path)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    print(f"[Policy] Loading LeRobot policy type: {policy_cfg.type}")
    policy_cfg.device = str(device)
    if args_cli.n_action_steps is not None:
        if args_cli.n_action_steps <= 0:
            raise ValueError("--n_action_steps must be positive.")
        if hasattr(policy_cfg, "chunk_size") and args_cli.n_action_steps > policy_cfg.chunk_size:
            raise ValueError(
                f"--n_action_steps={args_cli.n_action_steps} exceeds ACT chunk_size={policy_cfg.chunk_size}."
            )
        policy_cfg.n_action_steps = args_cli.n_action_steps
        
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=policy_cfg)
    policy.to(device)
    policy.eval()
    
    device_override = {"device": str(device)}
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": device_override},
    )
    return policy, preprocessor, postprocessor

def build_policy_observation(
    policy,
    robot,
    ordered_joint_ids: Sequence[int],
    gripper_camera,
    top_camera,
    phone_camera_source: str,
    env_idx: int = 0,
) -> dict[str, np.ndarray]:
    expected_keys = set(policy.config.input_features.keys())
    observation = {}
    image_cache = {}
    
    def gripper_image() -> np.ndarray:
        if "gripper" not in image_cache:
            image_cache["gripper"] = camera_rgb_to_uint8(gripper_camera.data.output["rgb"], env_idx)
        return image_cache["gripper"]
    def top_image() -> np.ndarray:
        if top_camera is None:
            raise RuntimeError("Need Top Camera in this Policy")
        if "top" not in image_cache:
            image_cache["top"] = camera_rgb_to_uint8(top_camera.data.output["rgb"], env_idx)
        return image_cache["top"]
    
    if OBS_STATE_KEY in expected_keys:
        observation[OBS_STATE_KEY] = get_observation_state(robot, ordered_joint_ids, env_idx)
    
    if GRIPPER_VIDEO_KEY in expected_keys:
        observation[GRIPPER_VIDEO_KEY] = gripper_image()
    
    if TOP_VIDEO_KEY in expected_keys:
        observation[TOP_VIDEO_KEY] = top_image()
        
    if PHONE_VIDEO_KEY in expected_keys:
        if phone_camera_source == "top":
            observation[PHONE_VIDEO_KEY] = top_image()
        elif phone_camera_source == "gripper":
            observation[PHONE_VIDEO_KEY] = gripper_image()
        else:
            raise ValueError(f"Unsopported phone camera source: {phone_camera_source}")
        
        
    if CAMERA1_VIDEO_KEY in expected_keys:
        observation[CAMERA1_VIDEO_KEY] = top_image()
    if CAMERA2_VIDEO_KEY in expected_keys:
        observation[CAMERA2_VIDEO_KEY] = gripper_image()
        
    missing_keys = expected_keys - set(observation.keys())
    optional_missing_keys = set()
    if getattr(policy.config, "type", None) == "smolvla":
        optional_missing_keys = {key for key in missing_keys if key.startswith("observation.images.")}
    required_missing_keys = missing_keys - optional_missing_keys
    if required_missing_keys:
        raise RuntimeError(f"Cannot build policy observation. Missing keyss: {sorted(required_missing_keys)}")
    return observation

def main():
    env_cfg = SO101BinPickPlaceJointDatagenEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.num_rerenders_on_reset = max(env_cfg.num_rerenders_on_reset, 1)
    env_cfg.viewer.eye = (1.5, 1.5, 1.8)      # 초기 카메라 위치
    env_cfg.viewer.lookat = (0.0, 0.0, 0.1)   # 바라보는 목표 지점
    add_top_camera_to_scene_cfg(env_cfg.scene)
    
    
    
    video_frames = []
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    policy_interval, control_hz = get_policy_interval(control_dt, args_cli.action_hz)
    
    if args_cli.num_envs != 1:
        raise ValueError("Inference ACT need to be set num_envs=1")

    policy_device = torch.device(args_cli.policy_device)
    if policy_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested --policy_device {args_cli.policy_device}, but CUDA is not available.")
    policy, preprocessor, postprocessor = load_lerobot_policy(args_cli.policy_path, policy_device)
    
    env = ManagerBasedEnv(cfg=env_cfg)
    env.reset()
    
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    box = env.scene["box"]
    gripper_camera = env.scene["camera_ego"]
    top_camera = env.scene["camera_top"]
    ordered_joint_ids = get_ordered_joint_ids(robot)
    
    expected_action_dim = len(ISAAC_SO101_JOINT_NAMES)
    if env.action_manager.total_action_dim != expected_action_dim:
        raise RuntimeError(
            f"Expected env action dim {expected_action_dim}, got {env.action_manager.total_action_dim}."
        )
    if policy.config.output_features["action"].shape[0] != expected_action_dim:
        raise RuntimeError(
            f"Expected policy action dim {expected_action_dim}, got {policy.config.output_features['action'].shape[0]}."
        )
        
    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), dtype=torch.float32, device=env.device)
    current_joint_target = robot.data.joint_pos[:, ordered_joint_ids].clone()
    joint_increment = torch.zeros_like(current_joint_target)
    actions[:] = current_joint_target
    
    episode_count = 0
    episode_control_step = 0
    total_control_step = 0
    
    def reset_episode() -> None:
        nonlocal current_joint_target, joint_increment, episode_control_step
        env.reset()
        policy.reset()
        reset_processor(preprocessor)
        reset_processor(postprocessor)
        current_joint_target = robot.data.joint_pos[:, ordered_joint_ids].clone()
        joint_increment = torch.zeros_like(current_joint_target)
        actions[:] = current_joint_target
        episode_control_step = 0
        
    '''LOGS'''
    reset_episode()
    print("Starting Loop", simulation_app.is_running())
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if args_cli.num_episodes > 0 and episode_count >= args_cli.num_episodes:
                    print(f"\n[Done] Completed {episode_count} episodes. Exiting.")
                    break
                
                policy_step = episode_control_step // policy_interval
                if policy_step >= args_cli.episode_policy_steps:
                    cube_pos_w = cube.data.root_pos_w[:, :3].clone()
                    box_pos_w = get_box_pos_w(box)
                    success = check_success(cube_pos_w, box_pos_w)
                    episode_count += 1
                    print(f"[Episode {episode_count}] Finished success={success}")
                    reset_episode()
                    continue
                
                if episode_control_step % policy_interval == 0:
                    observation = build_policy_observation(
                        policy=policy,
                        robot=robot,
                        ordered_joint_ids=ordered_joint_ids,
                        gripper_camera=gripper_camera,
                        top_camera=top_camera,
                        phone_camera_source=args_cli.phone_camera_source,
                    )
                    policy_action = predict_action(
                        observation=observation,
                        policy=policy,
                        device=policy_device,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=args_cli.use_amp,
                        task=args_cli.task,
                        robot_type="so101"
                    )
                    policy_joint_target = policy_action.reshape(1, -1).to(device=env.device, dtype=torch.float32)
                    raw_policy_joint_target = policy_joint_target.clone()
                    if args_cli.binary_gripper:
                        policy_joint_target = apply_binary_gripper_target(
                            policy_joint_target,
                            threshold=args_cli.gripper_threshold,
                            open_target=args_cli.gripper_open_target,
                            close_target=args_cli.gripper_close_target,
                        )
                    policy_joint_target = clamp_joint_targets(robot, ordered_joint_ids, policy_joint_target)
                    joint_increment = (policy_joint_target - current_joint_target) / float(policy_interval)
                    
                    if policy_step % 30 == 0:
                        action_np = policy_joint_target[0].detach().cpu().numpy()
                        raw_jaw = raw_policy_joint_target[0, -1].item()
                        cmd_jaw = policy_joint_target[0, -1].item()
                        print(
                            f"  ep={episode_count + 1} policy_step={policy_step} "
                            f"joint_tgt=({', '.join(f'{v:.3f}' for v in action_np)}) "
                            f"raw_jaw={raw_jaw:.3f} cmd_jaw={cmd_jaw:.3f}"
                        )
                        
                current_joint_target = current_joint_target + joint_increment
                current_joint_target = clamp_joint_targets(robot, ordered_joint_ids, current_joint_target)
                actions[:] = current_joint_target
                
                _, _ = env.step(actions)
                
                
                viewport_cam = env.scene["recording_camera"]
        
                # RGB 데이터 가져오기 (이때 텐서 크기는 [1, 720, 1280, 3] 입니다)
                rgb_tensor = viewport_cam.data.output["rgb"]
                
                # 카메라는 1개이므로 배치 차원 [0]을 선택하고, GPU -> CPU 넘파이 배열로 변환
                frame = rgb_tensor[0].cpu().numpy().astype(np.uint8)
                
                # 프레임 리스트에 추가
                video_frames.append(frame)
                
                
                
                total_control_step += 1
                episode_control_step += 1
    except KeyboardInterrupt:
        print("\n[Interrupted] Closing simulation.")
    finally:
        if len(video_frames) > 0:
            import imageio
            video_path = "github_demo.mp4"
            imageio.mimsave(video_path, video_frames, fps=60)
            print(f"🎉 데모 비디오 저장 완료: {video_path}")
        env.close()
        simulation_app.close()
        
if __name__ == "__main__":
    main()