"""
Convert SimEval DROID trajectories to LeRobot format for π₀-FAST-DROID training.

This script converts HDF5 trajectory files recorded from the SimEval environment
to LeRobot dataset format compatible with OpenPI training pipeline.

Usage:
    python convert_simeval_to_lerobot.py \
        --input_dir recorded_trajectories/scene1 \
        --repo_id your_username/simeval_droid_scene1 \
        --push_to_hub
"""

import argparse
import h5py
import numpy as np
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# LeRobot 默认数据目录
HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"

# PlayingCardsKitchen 固定 goal pose (LOCAL 坐标, wxyz 四元数)
FIXED_GOAL_POSE = np.array([0.5, 0.0, 0.03, 0.02, 0.00, -0.7047, 0.7095], dtype=np.float32)


def assemble_42d_state(obs: dict, t: int) -> np.ndarray:
    """将单帧 obs 组装为 42D 特权 state 向量。

    布局: [joint_pos(7), joint_vel(7), gripper(1), ee_pose(7),
           object_pose(7), goal_pose(7), tcp_to_obj(3), obj_to_goal(3)]
    """
    joint_pos = obs["joint_position"][t].astype(np.float32)           # (7,)
    joint_vel = obs["joint_velocity"][t].astype(np.float32)           # (7,)
    gripper = np.atleast_1d(obs["gripper_position"][t]).astype(np.float32)[:1]  # (1,)
    ee_pose = obs["ee_pose"][t].astype(np.float32)                    # (7,)

    if "object_pose" in obs:
        obj_pose = obs["object_pose"][t].astype(np.float32)           # (7,)
    else:
        obj_pose = np.zeros(7, dtype=np.float32)

    goal_pose = FIXED_GOAL_POSE                                       # (7,)
    tcp_to_obj = ee_pose[:3] - obj_pose[:3]                           # (3,)
    obj_to_goal = obj_pose[:3] - goal_pose[:3]                        # (3,)

    return np.concatenate([
        joint_pos, joint_vel, gripper, ee_pose,
        obj_pose, goal_pose, tcp_to_obj, obj_to_goal,
    ])  # (42,)


def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    """Resize image using PIL with BICUBIC interpolation.

    Args:
        image: Image array in (H, W, C) format
        size: Target size as (width, height)

    Returns:
        Resized image array
    """
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def downsample_to_15hz(data: dict, fps_source: float = 30.0, fps_target: float = 15.0) -> dict:
    """Downsample trajectory data from source fps to target fps.

    Args:
        data: Dictionary with all trajectory data arrays
        fps_source: Source frame rate (default 30Hz)
        fps_target: Target frame rate (default 15Hz)

    Returns:
        Downsampled data dictionary
    """
    # Calculate downsampling factor
    downsample_factor = int(fps_source / fps_target)

    if downsample_factor <= 1:
        return data  # No downsampling needed

    # Downsample all arrays by selecting every Nth frame
    downsampled = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            downsampled[key] = value[::downsample_factor]
        else:
            downsampled[key] = value

    return downsampled


def load_episode_from_hdf5(hdf5_file: h5py.File, episode_key: str) -> dict | None:
    """Load a single episode from HDF5 file.

    Args:
        hdf5_file: Open HDF5 file handle
        episode_key: Episode group name (e.g., 'episode_0')

    Returns:
        Dictionary with episode data, or None if episode is incomplete
    """
    ep_group = hdf5_file[episode_key]

    # 检查必要字段是否存在（不完整 episode 可能只有图像）
    required_obs = ["joint_position", "joint_velocity", "gripper_position",
                    "gripper_velocity", "ee_pose", "ee_velocity", "timestamp"]
    # object_pose 是可选的（旧数据可能没有）
    optional_obs = ["object_pose"]
    obs_group = ep_group.get("observations")
    if obs_group is None:
        return None
    for key in required_obs:
        if key not in obs_group:
            return None
    if "actions" not in ep_group:
        return None
    acts_group = ep_group["actions"]
    if "joint_position_command" not in acts_group or "gripper_command" not in acts_group:
        return None

    # Load observations
    obs = {
        "external_cam": ep_group["observations/external_cam"][:],
        "wrist_cam": ep_group["observations/wrist_cam"][:],
        "joint_position": ep_group["observations/joint_position"][:],
        "joint_velocity": ep_group["observations/joint_velocity"][:],
        "gripper_position": ep_group["observations/gripper_position"][:],
        "gripper_velocity": ep_group["observations/gripper_velocity"][:],
        "ee_pose": ep_group["observations/ee_pose"][:],
        "ee_velocity": ep_group["observations/ee_velocity"][:],
        "timestamp": ep_group["observations/timestamp"][:],
    }

    # 可选字段
    for opt_key in optional_obs:
        if opt_key in obs_group:
            obs[opt_key] = obs_group[opt_key][:]

    # Load actions
    actions = {
        "joint_position_command": ep_group["actions/joint_position_command"][:],
        "gripper_command": ep_group["actions/gripper_command"][:],
    }

    # 组合动作向量: 7D 绝对关节位置 + 1D 绝对夹爪命令
    # 使用录制的 joint_position_command（绝对位置），而非 joint_velocity
    # OpenPI 训练管道的 DeltaActions(make_bool_mask(7,-1)) 会自动将
    # 前 7 维从绝对位置转为 delta，第 8 维（夹爪）保持绝对
    obs["actions"] = np.concatenate([
        actions["joint_position_command"],  # (T, 7) 绝对关节位置命令
        actions["gripper_command"]          # (T, 1) 绝对夹爪命令
    ], axis=1)

    # Load metadata
    metadata = dict(ep_group["metadata"].attrs)

    return {"observations": obs, "actions": actions, "metadata": metadata}


def convert_trajectory_to_lerobot(
    input_dir: str,
    repo_id: str,
    push_to_hub: bool = False,
    local_dir: str = None,
    state_42d: bool = False,
    success_only: bool = False,
):
    """Convert SimEval trajectory to LeRoBot dataset.

    Args:
        input_dir: Directory containing trajectory HDF5 files
        repo_id: LeRoBot dataset repository ID
        push_to_hub: Whether to push to Hugging Face Hub
        local_dir: Optional custom directory for storing dataset (saves home dir space)
        state_42d: 输出预组装 42D state (无图像)，用于 DP/ACT state-only 训练
    """
    input_path = Path(input_dir)

    # Find all trajectory files
    traj_files = list(input_path.glob("trajectory_*.h5"))
    if not traj_files:
        raise ValueError(f"No trajectory files found in {input_dir}")

    print(f"Found {len(traj_files)} trajectory file(s)")

    # Determine output path
    if local_dir:
        output_path = Path(local_dir) / repo_id
        print(f"Using custom dataset directory: {output_path}")
        # Clean up existing dataset in custom location
        if output_path.exists():
            print(f"Removing existing dataset at {output_path}")
            shutil.rmtree(output_path)
        # Ensure parent directory exists, but let LeRobotDataset create the final directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = HF_LEROBOT_HOME / repo_id
        print(f"Using default dataset directory: {output_path}")
        # Clean up existing dataset
        if output_path.exists():
            print(f"Removing existing dataset at {output_path}")
            shutil.rmtree(output_path)

    # Create LeRoBot dataset
    print(f"Creating LeRoBot dataset: {repo_id} (state_42d={state_42d})")

    if state_42d:
        features = {
            "state": {
                "dtype": "float32",
                "shape": (42,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        }
    else:
        features = {
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "joint_velocity": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_velocity"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "gripper_velocity": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_velocity"],
            },
            "ee_pose": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["ee_pose"],
            },
            "ee_velocity": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["ee_velocity"],
            },
            "object_pose": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["object_pose"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        }

    create_kwargs = {
        "repo_id": repo_id,
        "robot_type": "panda",
        "fps": 15,  # Target FPS after downsampling
        "features": features,
        "image_writer_threads": 10,
        "image_writer_processes": 5,
    }

    # Add root directory if specified (custom dataset location)
    if local_dir:
        create_kwargs["root"] = str(output_path)

    dataset = LeRobotDataset.create(**create_kwargs)

    # Process all trajectory files
    total_episodes = 0
    skipped_episodes = 0
    for traj_file in traj_files:
        print(f"\nProcessing {traj_file.name}...")

        try:
            f = h5py.File(traj_file, "r")
        except OSError as e:
            print(f"  Skipping locked/corrupted file: {e}")
            continue

        with f:
            # Get all episode keys
            episode_keys = [k for k in f.keys() if k.startswith("episode_")]
            episode_keys.sort(key=lambda x: int(x.split("_")[1]))

            print(f"Found {len(episode_keys)} episodes in {traj_file.name}")

            for ep_key in tqdm(episode_keys, desc="Converting episodes"):
                # Load episode (skip incomplete ones)
                episode_data = load_episode_from_hdf5(f, ep_key)
                if episode_data is None:
                    skipped_episodes += 1
                    continue
                obs = episode_data["observations"]
                metadata = episode_data["metadata"]

                # success-only 过滤: metadata 中的 success 是字符串 "True" / "False"
                if success_only:
                    succ_val = metadata.get("success", "")
                    if str(succ_val).lower() != "true":
                        skipped_episodes += 1
                        continue

                # 根据 HDF5 metadata 中的 fps 决定是否降采样
                source_fps = metadata.get("fps", 15.0)
                if source_fps > 15.0:
                    obs_downsampled = downsample_to_15hz(obs, fps_source=source_fps, fps_target=15.0)
                else:
                    obs_downsampled = obs  # 已经是 15Hz，不降采样

                # Get episode length after downsampling
                episode_length = len(obs_downsampled["timestamp"])

                # Convert each timestep
                for t in range(episode_length):
                    if state_42d:
                        # 42D state-only 模式: 预组装 42D state，无图像
                        frame = {
                            "state": assemble_42d_state(obs_downsampled, t),
                            "actions": obs_downsampled["actions"][t].astype(np.float32),
                        }
                    else:
                        # 完整模式: 个别 obs key + 图像
                        raw_ext = obs_downsampled["external_cam"][t]
                        raw_wrist = obs_downsampled["wrist_cam"][t]
                        target_size = (320, 180)  # PIL uses (width, height)
                        if raw_ext.shape[:2] == (180, 320):
                            exterior_image = raw_ext
                        else:
                            exterior_image = resize_image(raw_ext, size=target_size)
                        if raw_wrist.shape[:2] == (180, 320):
                            wrist_image = raw_wrist
                        else:
                            wrist_image = resize_image(raw_wrist, size=target_size)

                        frame = {
                            "exterior_image_1_left": exterior_image,
                            "wrist_image_left": wrist_image,
                            "joint_position": obs_downsampled["joint_position"][t].astype(np.float32),
                            "joint_velocity": obs_downsampled["joint_velocity"][t].astype(np.float32),
                            "gripper_position": np.array([obs_downsampled["gripper_position"][t]], dtype=np.float32),
                            "gripper_velocity": np.array([obs_downsampled["gripper_velocity"][t]], dtype=np.float32),
                            "ee_pose": obs_downsampled["ee_pose"][t].astype(np.float32),
                            "ee_velocity": obs_downsampled["ee_velocity"][t].astype(np.float32),
                            "actions": obs_downsampled["actions"][t].astype(np.float32),
                        }
                        # object_pose (可选)
                        if "object_pose" in obs_downsampled:
                            frame["object_pose"] = obs_downsampled["object_pose"][t].astype(np.float32)
                        else:
                            frame["object_pose"] = np.zeros(7, dtype=np.float32)
                    # task 不作为 frame 字段（新版 LeRobot 在 save_episode 中传入）
                    dataset.add_frame(frame)

                # Save episode (task/instruction 在此传入)
                task_str = str(metadata.get("instruction", "Flip the playing card"))
                dataset.save_episode(task=task_str)
                total_episodes += 1

    print(f"\n{'='*60}")
    print(f"Successfully converted {total_episodes} episodes (skipped {skipped_episodes} incomplete)")
    print(f"Dataset saved to: {output_path}")
    print(f"{'='*60}")

    # Optionally push to Hugging Face Hub
    if push_to_hub:
        print("\nPushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["droid", "panda", "simeval", "isaaclab"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SimEval DROID trajectories to LeRobot format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing trajectory HDF5 files",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="your_username/simeval_droid",
        help="LeRobot dataset repository ID (e.g., 'username/dataset_name')",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Custom directory for storing dataset (saves home dir space). "
             "Dataset will be stored in <local_dir>/<repo_id>",
    )
    parser.add_argument(
        "--state_42d",
        action="store_true",
        help="输出预组装 42D state (无图像)，用于 DP/ACT state-only 训练",
    )
    parser.add_argument(
        "--success_only",
        action="store_true",
        help="仅转换 metadata.success == True 的 episode",
    )

    args = parser.parse_args()

    # Validate input directory
    if not Path(args.input_dir).exists():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    # Convert trajectories
    convert_trajectory_to_lerobot(
        input_dir=args.input_dir,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
        local_dir=args.local_dir,
        state_42d=args.state_42d,
        success_only=args.success_only,
    )

    print("\nConversion complete!")
    print(f"\nNext steps:")
    print(f"1. Compute normalization stats:")
    print(f"   cd libs/openpi")
    print(f"   uv run scripts/compute_norm_stats.py --config-name pi05_droid_simeval_lora")
    print(f"\n2. Fine-tune π₀.5-DROID:")
    print(f"   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_droid_simeval_lora --exp-name=simeval_finetuned")


if __name__ == "__main__":
    main()
