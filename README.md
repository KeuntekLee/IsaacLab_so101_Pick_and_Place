# SO101 Pick-Place LeRobot Pipeline

End-to-end data collection and VLA policy inference for the **SO101** robot using Isaac Lab and LeRobot.

| Script | Purpose |
|--------|---------|
| [`pick_and_place_so101_lerobot_bin_ik_action_pair_grid_multi.py`](#data-collection) | Automated state-machine data collection producing LeRobot v3.0 datasets |
| [`pick_and_place_so101_lerobot_bin_act_infer.py`](#vla-policy-inference) | Load a trained VLA policy (e.g., ACT) and evaluate it in simulation |

---

## Table of Contents

- [Data Collection](#data-collection)
  - [Overview](#overview)
  - [State Machine Flow](#state-machine-flow)
  - [Environment Configuration](#environment-configuration)
  - [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Dataset Schema](#dataset-schema)
  - [Architecture](#architecture)
- [VLA Policy Inference](#vla-policy-inference)
  - [Overview](#overview-1)
  - [Usage](#usage-1)
  - [Command-Line Arguments](#command-line-arguments-1)
  - [Policy Path Resolution](#policy-path-resolution)
  - [Camera Input Mapping](#camera-input-mapping)
  - [Architecture](#architecture-1)
- [End-to-End Workflow](#end-to-end-workflow)
- [Success Criteria](#success-criteria)
- [Windows Notes](#windows-notes)
- [Troubleshooting](#troubleshooting)

---

## Data Collection

## Overview

This script performs the following:

1. **Automated Pick-and-Place** — A state machine drives the SO101 robot through a repeatable 7-phase cycle: approach, grasp, lift, move to bin, release, and return home.
2. **LeRobot Dataset Recording** — Each episode's joint states, actions, and dual-camera RGB frames (Gripper Ego + Top Phone) are saved in LeRobotDataset v3.0 format.
3. **Pair Grid Sampling** — Systematically explores cube/bin position combinations across a predefined grid for diverse training data.
4. **Quality Filtering** — Episodes with joint limit saturation are automatically discarded.
5. **Success Tracking** — Successful and failed episodes are tracked separately, with optional saving of failed episodes.

### Key Features

| Feature | Description |
|---------|-------------|
| **Differential IK Teacher** | Isaac Lab's `DifferentialIKController` (DLS method) maps end-effector poses to joint positions |
| **Smoothed Trajectory** | `MAX_TARGET_STEP_DIST=0.02m` constraint ensures smooth discrete motion per step |
| **Multi-Environment** | Parallel environments via `--num_envs` for faster data collection |
| **Pair Grid Sampling** | Systematic cube/bin position sampling across a `(PAIR_GRID_SHAPE)` grid |
| **Real-Time Status Output** | Console prints EE target and joint targets on each phase transition |

---

## State Machine Flow

Each episode runs a **350-step** cycle through 7 phases:

```
Step  0  ── APPROACH_ABOVE   ── Move above cube (Z+0.05m)
Step 50  ── APPROACH_CUBE    ── Descend to cube position
Step 100 ── GRASP            ── Close gripper (GRIPPER_CLOSE)
Step 150 ── LIFT             ── Lift cube (Z+0.08m)
Step 200 ── MOVE_BOX         ── Move above bin (Z+0.18m)
Step 250 ── RELEASE          ── Open gripper (GRIPPER_OPEN)
Step 300 ── GOHOME           ── Return to home pose
Step 350 ── (episode ends → reset)
```

> **Recording window**: Only frames from Step 0 to 299 (`RECORD_START_STEP` to `RECORD_END_STEP`) are saved to the dataset.

---

## Environment Configuration

| Component | Value |
|-----------|-------|
| **Environment Config** | `SO101BinPickPlacePairGridDatagenEnvCfg` |
| **Robot** | SO101 (5-axis arm + Jaw gripper) |
| **Objects** | Cube (target), Box (bin) |
| **Cameras** | `camera_ego` (Gripper RGB), `camera_top` (Top RGB, 480×640) |
| **Control Rate** | 120 Hz (sim dt × decimation) |
| **Dataset FPS** | 30 FPS (record_interval = 4) |
| **Joint Action Space** | 6D — `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]` |

### Home Pose

```
EE Position (world):  (0.050, 0.021, 0.126)
EE Quat (wxyz):       (-0.693, -0.140, 0.140, 0.693)
Target Quat (wxyz):   (-0.7071, 0.0, 0.0, 0.7071)  # -90° about X-axis
```

---

## Usage

### Prerequisites

- **Isaac Lab** installed and configured
- **LeRobot** installed (`pip install lerobot`) — required for data collection
- GPU environment recommended

### Basic Execution

```bash
# Run with Isaac Lab Python
isaaclab -p scripts/environments/state_machine/pick_and_place_so101_lerobot_bin_ik_action_pair_grid_multi.py \
    --dataset_dir ./datasets/so101_pick_place \
    --num_episodes 50
```

### Tuning Parameters

```bash
# 32 parallel environments, save failed episodes, H.264 codec
isaaclab -p scripts/environments/state_machine/pick_and_place_so101_lerobot_bin_ik_action_pair_grid_multi.py \
    --num_envs 32 \
    --dataset_dir ./datasets/so101_pick_place \
    --num_episodes 100 \
    --save_failed_episodes \
    --vcodec h264
```

### Demo Without LeRobot

```bash
# Run state machine demo without data collection
isaaclab -p scripts/environments/state_machine/pick_and_place_so101_lerobot_bin_ik_action_pair_grid_multi.py \
    --num_episodes 5
```

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_envs` | (env_cfg default) | Number of parallel simulation environments |
| `--dataset_dir` | `None` | Directory for LeRobot dataset (enables data collection when set) |
| `--num_episodes` | `10` | Number of episodes to collect (`0` = infinite loop) |
| `--vcodec` | `libsvtav1` | Video codec (`libsvtav1`, `h264`, `hevc`) |
| `--save_failed_episodes` | `False` | Also save episodes that did not meet success criteria |
| `--streaming_encoding` | `True` | Use streaming video encoding (faster `save_episode`) |
| `--disable_fabric` | `False` | Disable Fabric and use USD I/O operations |

---

## Dataset Schema

The generated LeRobot dataset contains the following features:

| Feature Key | dtype | shape | fps | Description |
|-------------|-------|-------|-----|-------------|
| `observation.state` | float32 | (6,) | 30 | Current joint positions (5 arm + 1 gripper) |
| `action` | float32 | (6,) | 30 | Next target joint positions (Joint-Space policy) |
| `observation.images.gripper` | video | (480, 640, 3) | 30 | Gripper Ego camera RGB |
| `observation.images.phone` | video | (480, 640, 3) | 30 | Top Phone camera RGB |

### Joint Name Mapping

```
Isaac Lab SO101          →  LeRobot Feature Names
───────────────────────────────────────────────────
Rotation (shoulder)      →  shoulder_pan.pos
Pitch (shoulder lift)    →  shoulder_lift.pos
Elbow                    →  elbow_flex.pos
Wrist_Pitch              →  wrist_flex.pos
Wrist_Roll               →  wrist_roll.pos
Jaw (gripper)            →  gripper.pos
```

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                       main() Loop                       │
│                                                         │
│  ┌────────────────┐      ┌──────────────────────────┐  │
│  │  State Machine  │─────→│  Differential IK Teacher │  │
│  │  (7 phases)     │      │  (DLS method)            │  │
│  └──────┬─────────┘      └───────────┬──────────────┘  │
│         │                            │                  │
│         ▼                            ▼                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │     compute_joint_policy_action()               │   │
│  │  • Smoothed target (MAX_TARGET_STEP_DIST)       │   │
│  │  • World → Base frame transform                 │   │
│  │  • Joint limit clamping                         │   │
│  └──────────────────┬──────────────────────────────┘   │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │     env.step(actions)                           │   │
│  │  • Joint-space command interpolation            │   │
│  │  • Policy update every record_interval          │   │
│  └──────────────────┬──────────────────────────────┘   │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │     LeRobot Frame Buffering                     │   │
│  │  • observation.state + action + 2 cameras       │   │
│  │  • Joint limit saturation → episode discard     │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────┘
```

### Key Classes

| Class | Role |
|-------|------|
| `TorchPickPlaceStateMachine` | Generates EE target positions and gripper commands based on step count |
| `SO101DiffIKTeacher` | Converts end-effector poses to joint positions via Differential IK |
| `create_lerobot_dataset()` | Initializes LeRobot v3.0 dataset with version-aware parameter detection |
| `build_lerobot_observation_frame()` | Packages a single frame's observation, action, and camera data |

---

## Success Criteria

```python
SUCCESS_THRESHOLD = 0.05  # 5 cm
```

At episode end, the **XY distance** between the cube position and the bin center must be within `5 cm`. The Z axis is not considered.

```python
xy_distance = ||cube_pos_w[:, :2] - box_pos_w[:, :2]||
success = xy_distance < 0.05
```

---

## Windows Notes

| Item | Recommendation |
|------|---------------|
| **pandas** | `2.1.4` (latest versions cause `save_episode()` access violations) |
| **pyarrow** | `15.0.0` |
| **Video codec** | `h264` or `hevc` recommended (AV1 is slow on Windows) |
| **pyright** | Commented out in `.pre-commit-config.yaml` (hangs behind VPN). Run manually: `isaaclab -p -m pyright` |

---

## Troubleshooting

### LeRobot import error

```
[WARNING] LeRobot not installed. Data Collection disabled.
```

→ Install with `pip install lerobot` and re-run. The script will still run as a demo without dataset saving.

### Episodes continuously discarded due to joint limit saturation

→ The state machine targets may be outside the robot's workspace. Adjust `LINK_OFFSET`, `HOME_EE_POS_W`, or target position offsets to match your robot configuration.

### Camera data missing error

→ Try adding `--disable_fabric` or verify camera spawn settings in the scene configuration.

### Video codec compatibility issues

→ Switch to `--vcodec h264`. On Windows, `libsvtav1` (AV1) can be slow or unsupported.

### Episode count not reaching target

→ Add `--save_failed_episodes` to include failed episodes, or increase `--num_envs` for more parallel throughput.

---

## VLA Policy Inference

### Overview

`pick_and_place_so101_lerobot_bin_act_infer.py` loads a **trained VLA policy** (e.g., ACT, SmolVLA) from a LeRobot checkpoint and evaluates it in an Isaac Lab simulation environment. This closes the sim-to-sim loop: data collected by the data collection script is used to train a policy, which is then tested back in simulation.

#### Key Features

| Feature | Description |
|---------|-------------|
| **LeRobot Policy Loading** | Loads any LeRobot-compatible policy via `PreTrainedConfig.from_pretrained()` with automatic pre/post-processor setup |
| **Flexible Camera Input** | Supports `observation.images.gripper`, `observation.images.top`, `observation.images.phone`, `camera1`, `camera2` keys — mapped dynamically based on policy's `input_features` |
| **Binary Gripper Mode** | Optional threshold-based open/close conversion (`--binary_gripper`) for policies that output continuous gripper values |
| **Joint-Space Interpolation** | Policy targets are smoothly interpolated over `policy_interval` steps to avoid jerky motion |
| **AMP Support** | Optional automatic mixed precision inference (`--use_amp`) for faster evaluation |
| **Nested Policy Path** | Automatically resolves `pretrained_model/` subdirectory structure in LeRobot checkpoints |

### Usage

#### Basic Inference

```bash
# Evaluate a trained ACT policy in simulation
isaaclab -p scripts/environments/state_machine/pick_and_place_so101_lerobot_bin_act_infer.py \
    --policy_path ./outputs/act_so101_pick_place \
    --num_episodes 100
```

#### With Binary Gripper

```bash
# Use binary open/close gripper commands with custom threshold
isaaclab -p scripts/environments/state_machine/pick_and_place_so101_lerobot_bin_act_infer.py \
    --policy_path ./outputs/act_so101_pick_place \
    --num_episodes 200 \
    --binary_gripper \
    --gripper_threshold 0.5 \
    --gripper_open_target 1.0 \
    --gripper_close_target 0.0
```

#### Custom Camera and Task Prompt

```bash
# Use top camera as phone input, custom task description, AMP enabled
isaaclab -p scripts/environments/state_machine/pick_and_place_so101_lerobot_bin_act_infer.py \
    --policy_path ./outputs/smolvla_so101 \
    --num_episodes 50 \
    --phone_camera_source top \
    --task "Pick the red cube and place it in the bin" \
    --use_amp
```

#### Performance Tuning

```bash
# Faster inference: disable fabric, use AMP, higher action Hz
isaaclab -p scripts/environments/state_machine/pick_and_place_so101_lerobot_bin_act_infer.py \
    --policy_path ./outputs/act_so101_pick_place \
    --num_episodes 500 \
    --disable_fabric \
    --use_amp \
    --action_hz 30.0 \
    --policy_device cuda:0
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--policy_path` | *(required)* | Path to LeRobot policy checkpoint directory |
| `--num_envs` | `128` | Number of parallel environments (must be `1` for ACT policies) |
| `--num_episodes` | `1000` | Number of episodes to evaluate |
| `--episode_policy_steps` | `400` | Policy steps per episode before forced reset |
| `--action_hz` | `30.0` | Frequency at which to query the policy |
| `--policy_device` | `cuda:0` | Device for policy inference |
| `--n_action_steps` | `1` | Number of simulation steps per policy action (for ACT chunking) |
| `--task` | `"Pick the cube..."` | Task description prompt (used by VLA policies like SmolVLA) |
| `--use_amp` | `False` | Enable automatic mixed precision for faster inference |
| `--binary_gripper` | `False` | Convert continuous gripper output to binary open/close |
| `--gripper_threshold` | `0.5` | Threshold for binary gripper (below = close, above = open) |
| `--gripper_open_target` | `1.0` | Joint position for gripper open command |
| `--gripper_close_target` | `0.0` | Joint position for gripper close command |
| `--phone_camera_source` | `top` | Camera source for `observation.images.phone` (`top` or `gripper`) |
| `--disable_fabric` | `False` | Disable Fabric simulation for better performance |

### Policy Path Resolution

The script automatically resolves the policy checkpoint directory:

```
./my_policy/                          ← passed as --policy_path
├── config.json                       ← required
└── model.safetensors                 ← required
```

or nested structure:

```
./my_policy/                          ← passed as --policy_path
└── pretrained_model/
    ├── config.json                   ← resolved automatically
    └── model.safetensors
```

### Camera Input Mapping

The `build_policy_observation()` function dynamically maps available cameras to the keys expected by the policy:

| Policy Key | Source |
|------------|--------|
| `observation.images.gripper` | `camera_ego` (wrist-mounted) |
| `observation.images.top` | `camera_top` (overhead) |
| `observation.images.phone` | `camera_top` or `camera_ego` (controlled by `--phone_camera_source`) |
| `observation.images.camera1` | `camera_top` |
| `observation.images.camera2` | `camera_ego` |

For **SmolVLA** policies, missing image keys are treated as optional (only `observation.state` and `task` are required).

### Architecture

```
┌────────────────────────────────────────────────────────┐
│                    Inference Loop                       │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │     env.reset() → randomize cube/bin positions  │   │
│  └──────────────────┬──────────────────────────────┘   │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │     build_policy_observation()                  │   │
│  │  • robot joint states (observation.state)       │   │
│  │  • gripper camera RGB                           │   │
│  │  • top camera RGB                               │   │
│  └──────────────────┬──────────────────────────────┘   │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │     predict_action() (LeRobot)                  │   │
│  │  • preprocessor → policy → postprocessor        │   │
│  │  • supports ACT, Diffusion, SmolVLA, etc.       │   │
│  └──────────────────┬──────────────────────────────┘   │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │     Joint-Space Execution                       │   │
│  │  • Optional binary gripper conversion           │   │
│  │  • Joint limit clamping                         │   │
│  │  • Smooth interpolation over policy_interval    │   │
│  └──────────────────┬──────────────────────────────┘   │
│                     │                                   │
│                     ▼                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │     env.step(actions)                           │   │
│  │  • Apply joint targets each control step        │   │
│  │  • Policy queried every policy_interval steps   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  After episode_policy_steps:                            │
│  → Check success (cube within 5cm of bin in XY)        │
│  → Reset episode and repeat                             │
└────────────────────────────────────────────────────────┘
```

## End-to-End Workflow

The two scripts form a complete sim-to-sim evaluation pipeline:

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Data Collection    │     │   VLA Training    │     │   Policy Inference   │
│                      │     │   (LeRobot)       │     │                      │
│  State Machine       │────→│  ACT / Diffusion  │────→│  Load checkpoint    │
│  → LeRobot dataset   │     │  SmolVLA / etc.   │     │  → Sim evaluation   │
│  (pair grid sampling)│     │                   │     │  → Success rate     │
└─────────────────────┘     └──────────────────┘     └─────────────────────┘
       Step 1                        Step 2                     Step 3
```

1. **Collect data**: Run the data collection script to generate a LeRobot dataset with diverse cube/bin configurations.
2. **Train a policy**: Use LeRobot's training tools (`lerobot train`) to train an ACT, Diffusion, or SmolVLA policy on the collected dataset.
3. **Evaluate in simulation**: Run the inference script to load the trained policy and measure its success rate in simulation.

## Related Scripts

| Script | Environment | Purpose |
|--------|-------------|---------|
| `pick_place_sm.py` | `Isaac-PickPlace-Cube-SO101-IK-v0` | Original SM — infinite loop, no recording |
| `pick_place_codex.py` | `Isaac-PickPlace-Cube-SO101-IK-v0` | LeRobot recording (Parquet + MP4), episode limit, cleanup |
| `pick_place_so101_lerobot_bin_fix.py` | `Isaac-PickPlace-Cube-SO101-Bin-IK-v0` | Work-in-progress Bin variant |
| `pick_and_place_so101_lerobot_bin_ik_action_pair_grid_multi.py` | `SO101BinPickPlacePairGridDatagenEnvCfg` | **Data collection** — Pair grid, joint-space actions, dual camera |
| `pick_and_place_so101_lerobot_bin_act_infer.py` | `SO101BinPickPlaceJointDatagenEnvCfg` | **Policy inference** — Load trained VLA policy, evaluate in sim |

---

## References

- **Isaac Lab Documentation**: https://isaac-lab.github.io/
- **LeRobot Documentation**: https://github.com/huggingface/lerobot
- **SO101 Robot**: https://www.so-robotics.com/



## VLA Inference Simulation Results

https://github.com/user-attachments/assets/6bbcafca-2e1b-4591-b699-c40062a3daa1


https://github.com/user-attachments/assets/19316a77-cbcc-4b3a-b163-59037d19fd63



https://github.com/user-attachments/assets/2a0e3cfc-d0be-472a-a1ff-f3a55bf3da61



https://github.com/user-attachments/assets/12b8c1a0-0cd9-473b-858c-de8a236ee7a4


