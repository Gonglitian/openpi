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

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


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


def downsample_trajectory(data: dict, fps_source: float, fps_target: float) -> dict:
    """Downsample trajectory data if needed. Returns data as-is when source fps <= target fps.

    Args:
        data: Dictionary with all trajectory data arrays
        fps_source: Source frame rate
        fps_target: Target frame rate

    Returns:
        Downsampled data dictionary (or original data if no downsampling needed)
    """
    downsample_factor = int(fps_source / fps_target)

    if downsample_factor <= 1:
        return data  # No downsampling needed

    downsampled = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            downsampled[key] = value[::downsample_factor]
        else:
            downsampled[key] = value

    return downsampled


def load_episode_from_hdf5(hdf5_file: h5py.File, episode_key: str) -> dict:
    """Load a single episode from HDF5 file.

    Args:
        hdf5_file: Open HDF5 file handle
        episode_key: Episode group name (e.g., 'episode_0')

    Returns:
        Dictionary with episode data
    """
    ep_group = hdf5_file[episode_key]

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

    # Load actions
    actions = {
        "joint_position_command": ep_group["actions/joint_position_command"][:],
        "gripper_command": ep_group["actions/gripper_command"][:],
    }

    # Combine into action array (7D joint vel + 1D gripper)
    # Note: For training, we use joint_velocity as actions, not position commands
    # This matches the DROID training format
    obs["actions"] = np.concatenate([
        obs["joint_velocity"],  # 7D joint velocities
        obs["gripper_position"][:, None]  # 1D gripper position -> (T, 1)
    ], axis=1)

    # Load metadata
    metadata = dict(ep_group["metadata"].attrs)

    return {"observations": obs, "actions": actions, "metadata": metadata}


def convert_trajectory_to_lerobot(
    input_dir: str,
    repo_id: str,
    push_to_hub: bool = False,
    local_dir: str = None,
):
    """Convert SimEval trajectory to LeRobot dataset.

    Args:
        input_dir: Directory containing trajectory HDF5 files
        repo_id: LeRobot dataset repository ID
        push_to_hub: Whether to push to Hugging Face Hub
        local_dir: Optional custom directory for storing dataset (saves home dir space)
    """
    input_path = Path(input_dir)

    # Find all trajectory files
    traj_files = list(input_path.glob("trajectory_*.h5"))
    if not traj_files:
        raise ValueError(f"No trajectory files found in {input_dir}")

    print(f"Found {len(traj_files)} trajectory file(s)")

    # Determine output path
    if local_dir:
        # --local_dir is used directly as dataset root, without appending repo_id
        output_path = Path(local_dir)
        print(f"Using custom dataset directory: {output_path}")
        if output_path.exists():
            print(f"Removing existing dataset at {output_path}")
            shutil.rmtree(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = HF_LEROBOT_HOME / repo_id
        print(f"Using default dataset directory: {output_path}")
        if output_path.exists():
            print(f"Removing existing dataset at {output_path}")
            shutil.rmtree(output_path)

    # Create LeRobot dataset with DROID features
    print(f"Creating LeRobot dataset: {repo_id}")
    create_kwargs = {
        "repo_id": repo_id,
        "robot_type": "panda",
        "fps": 15,  # Target FPS after downsampling
        "features": {
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
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        "image_writer_threads": 10,
        "image_writer_processes": 5,
    }

    # Add root directory if specified (custom dataset location)
    if local_dir:
        create_kwargs["root"] = str(output_path)

    dataset = LeRobotDataset.create(**create_kwargs)

    # Process all trajectory files
    total_episodes = 0
    for traj_file in traj_files:
        print(f"\nProcessing {traj_file.name}...")

        with h5py.File(traj_file, "r") as f:
            # Get all episode keys
            episode_keys = [k for k in f.keys() if k.startswith("episode_")]
            episode_keys.sort(key=lambda x: int(x.split("_")[1]))

            print(f"Found {len(episode_keys)} episodes in {traj_file.name}")

            for ep_key in tqdm(episode_keys, desc="Converting episodes"):
                # Load episode
                episode_data = load_episode_from_hdf5(f, ep_key)
                obs = episode_data["observations"]
                metadata = episode_data["metadata"]

                # Isaac Lab records at 15Hz (decimation=8, sim.dt=1/120), no downsampling needed.
                # Automatically adapts if source data has a different fps via HDF5 metadata.
                source_fps = float(metadata.get("fps", 15.0))
                obs_downsampled = downsample_trajectory(obs, fps_source=source_fps, fps_target=15.0)

                # Get episode length after downsampling
                episode_length = len(obs_downsampled["timestamp"])

                # Convert each timestep
                for t in range(episode_length):
                    # Resize images from (360, 640) to (180, 320)
                    exterior_image = resize_image(
                        obs_downsampled["external_cam"][t],
                        size=(320, 180)  # PIL uses (width, height)
                    )
                    wrist_image = resize_image(
                        obs_downsampled["wrist_cam"][t],
                        size=(320, 180)
                    )

                    # Add frame to LeRobot dataset
                    dataset.add_frame({
                        "exterior_image_1_left": exterior_image,
                        "wrist_image_left": wrist_image,
                        "joint_position": obs_downsampled["joint_position"][t].astype(np.float32),
                        "joint_velocity": obs_downsampled["joint_velocity"][t].astype(np.float32),
                        "gripper_position": np.array([obs_downsampled["gripper_position"][t]], dtype=np.float32),
                        "gripper_velocity": np.array([obs_downsampled["gripper_velocity"][t]], dtype=np.float32),
                        "ee_pose": obs_downsampled["ee_pose"][t].astype(np.float32),
                        "ee_velocity": obs_downsampled["ee_velocity"][t].astype(np.float32),
                        "actions": obs_downsampled["actions"][t].astype(np.float32),
                        "task": metadata["instruction"],
                    })

                # Save episode
                dataset.save_episode()
                total_episodes += 1

    print(f"\n{'='*60}")
    print(f"Successfully converted {total_episodes} episodes")
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
        help="Custom directory for storing dataset (direct output path, no repo_id appended).",
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
    )

    print("\nConversion complete!")
    print(f"\nNext steps:")
    print(f"1. Compute normalization stats:")
    print(f"   cd /home/mingxuanyan/workspace/RegraspGen/openpi")
    print(f"   uv run scripts/compute_norm_stats.py --config-name pi0_fast_droid")
    print(f"\n2. Fine-tune π₀-FAST-DROID:")
    print(f"   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_droid --exp-name=simeval_finetuned")


if __name__ == "__main__":
    main()
