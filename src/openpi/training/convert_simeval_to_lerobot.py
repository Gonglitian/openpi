"""
Convert SimEval DROID trajectories to LeRobot format for π₀-FAST-DROID training.

This script converts HDF5 trajectory files recorded from the SimEval environment
to LeRobot dataset format compatible with OpenPI training pipeline.

Optimized with:
  - Pipeline parallelism: background thread prefetches next episode while current writes
  - OpenCV batch resize: ~9x faster than PIL per-image resize
  - Both h5py reads and cv2 resize release GIL, enabling true thread-level overlap

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
import threading
import queue
import time
from pathlib import Path
from tqdm import tqdm

try:
    import cv2
    USE_CV2 = True
except ImportError:
    from PIL import Image
    USE_CV2 = False

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# Sentinel value to signal end of episode stream
_SENTINEL = None


def resize_images_batch(images: np.ndarray, target_wh: tuple[int, int] = (320, 180)) -> np.ndarray:
    """Resize a batch of images. Uses OpenCV if available (9x faster), falls back to PIL.

    Args:
        images: (N, H, W, C) uint8 array
        target_wh: Target (width, height)

    Returns:
        (N, target_h, target_w, C) uint8 array
    """
    if USE_CV2:
        out = np.empty((len(images), target_wh[1], target_wh[0], images.shape[3]), dtype=np.uint8)
        for i in range(len(images)):
            cv2.resize(images[i], target_wh, dst=out[i], interpolation=cv2.INTER_LINEAR)
        return out
    else:
        resized = []
        for i in range(len(images)):
            img = Image.fromarray(images[i])
            resized.append(np.array(img.resize(target_wh, resample=Image.BICUBIC)))
        return np.stack(resized)


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


def load_and_preprocess_episode(hdf5_file: h5py.File, episode_key: str) -> dict | None:
    """Load a single episode from HDF5, build actions, resize images.

    Both h5py reads and cv2.resize release the GIL, so this function
    can run in a background thread without blocking the main thread.

    Returns:
        Dict with preprocessed frames ready for add_frame(), or None on error.
    """
    ep_group = hdf5_file[episode_key]

    # Read all arrays (GIL released during HDF5 I/O + LZF decompression)
    t_read_start = time.perf_counter()
    ext_cam = ep_group["observations/external_cam"][:]
    wrist_cam = ep_group["observations/wrist_cam"][:]
    joint_pos = ep_group["observations/joint_position"][:]
    joint_vel = ep_group["observations/joint_velocity"][:]
    gripper_pos = ep_group["observations/gripper_position"][:]
    gripper_vel = ep_group["observations/gripper_velocity"][:]
    ee_pose = ep_group["observations/ee_pose"][:]
    ee_vel = ep_group["observations/ee_velocity"][:]
    timestamp = ep_group["observations/timestamp"][:]
    metadata = dict(ep_group["metadata"].attrs)
    t_read = time.perf_counter() - t_read_start

    # Auto-downsample based on HDF5 metadata fps
    source_fps = float(metadata.get("fps", 15.0))
    downsample_factor = int(source_fps / 15.0)
    if downsample_factor > 1:
        ext_cam = ext_cam[::downsample_factor]
        wrist_cam = wrist_cam[::downsample_factor]
        joint_pos = joint_pos[::downsample_factor]
        joint_vel = joint_vel[::downsample_factor]
        gripper_pos = gripper_pos[::downsample_factor]
        gripper_vel = gripper_vel[::downsample_factor]
        ee_pose = ee_pose[::downsample_factor]
        ee_vel = ee_vel[::downsample_factor]
        timestamp = timestamp[::downsample_factor]

    # Batch resize images (GIL released during cv2 calls)
    t_resize_start = time.perf_counter()
    ext_resized = resize_images_batch(ext_cam, target_wh=(320, 180))
    wrist_resized = resize_images_batch(wrist_cam, target_wh=(320, 180))
    t_resize = time.perf_counter() - t_resize_start

    # Build actions: joint_velocity(7D) + gripper_position(1D)
    actions = np.concatenate([joint_vel, gripper_pos[:, None]], axis=1).astype(np.float32)

    return {
        "ext_images": ext_resized,
        "wrist_images": wrist_resized,
        "joint_position": joint_pos.astype(np.float32),
        "joint_velocity": joint_vel.astype(np.float32),
        "gripper_position": gripper_pos,
        "gripper_velocity": gripper_vel,
        "ee_pose": ee_pose.astype(np.float32),
        "ee_velocity": ee_vel.astype(np.float32),
        "actions": actions,
        "instruction": metadata.get("instruction", ""),
        "num_frames": len(timestamp),
        "_t_read": t_read,
        "_t_resize": t_resize,
    }


def _prefetch_worker(
    traj_files: list[Path],
    out_queue: queue.Queue,
    progress_desc: str = "Reading",
):
    """Background thread: reads HDF5 files and puts preprocessed episodes into queue.

    Runs one file at a time to avoid HDD seek thrashing. Each episode is fully
    read + resized before being enqueued, so the main thread only does lightweight
    add_frame() + save_episode() calls.
    """
    for traj_file in traj_files:
        with h5py.File(traj_file, "r") as f:
            episode_keys = [k for k in f.keys() if k.startswith("episode_")]
            episode_keys.sort(key=lambda x: int(x.split("_")[1]))

            for ep_key in episode_keys:
                ep_data = load_and_preprocess_episode(f, ep_key)
                if ep_data is not None:
                    ep_data["source_file"] = traj_file.name
                    ep_data["episode_key"] = ep_key
                    out_queue.put(ep_data)

    # Signal that all episodes have been produced
    out_queue.put(_SENTINEL)


def convert_trajectory_to_lerobot(
    input_dir: str,
    repo_id: str,
    push_to_hub: bool = False,
    local_dir: str = None,
    prefetch_depth: int = 2,
):
    """Convert SimEval trajectory to LeRobot dataset with pipeline parallelism.

    Architecture:
        [Background Thread]  read HDF5 + LZF decompress + cv2 resize
                |  (queue, depth=prefetch_depth)
        [Main Thread]        add_frame() + save_episode() to LeRobot

    Both h5py I/O and cv2 resize release the GIL, so the background thread
    runs truly in parallel with the main thread's Python-level work.

    Args:
        input_dir: Directory containing trajectory HDF5 files
        repo_id: LeRobot dataset repository ID
        push_to_hub: Whether to push to Hugging Face Hub
        local_dir: Optional custom directory for storing dataset
        prefetch_depth: Number of episodes to buffer ahead (default 2)
    """
    input_path = Path(input_dir)

    # Find all trajectory files
    traj_files = sorted(input_path.glob("trajectory_*.h5"))
    if not traj_files:
        raise ValueError(f"No trajectory files found in {input_dir}")

    # Count total episodes for progress bar
    total_episodes = 0
    for tf in traj_files:
        with h5py.File(tf, "r") as f:
            total_episodes += len([k for k in f.keys() if k.startswith("episode_")])

    print(f"Found {len(traj_files)} file(s), {total_episodes} episodes total")
    print(f"Pipeline: prefetch_depth={prefetch_depth}, resize={'OpenCV' if USE_CV2 else 'PIL'}")

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
        "fps": 15,
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

    if local_dir:
        create_kwargs["root"] = str(output_path)

    dataset = LeRobotDataset.create(**create_kwargs)

    # Launch prefetch thread
    ep_queue: queue.Queue = queue.Queue(maxsize=prefetch_depth)
    reader_thread = threading.Thread(
        target=_prefetch_worker,
        args=(traj_files, ep_queue),
        daemon=True,
    )

    t_start = time.perf_counter()
    reader_thread.start()

    # Main thread: consume preprocessed episodes and write to LeRobot
    episodes_done = 0
    current_file = ""
    total_t_read = 0.0
    total_t_resize = 0.0
    total_t_write = 0.0
    total_t_wait = 0.0
    pbar = tqdm(total=total_episodes, desc="Converting", unit="ep")

    while True:
        t_wait_start = time.perf_counter()
        ep_data = ep_queue.get()
        t_wait = time.perf_counter() - t_wait_start

        if ep_data is _SENTINEL:
            break

        # Log file transitions
        src = ep_data["source_file"]
        if src != current_file:
            current_file = src

        n = ep_data["num_frames"]
        ep_t_read = ep_data.pop("_t_read", 0.0)
        ep_t_resize = ep_data.pop("_t_resize", 0.0)

        t_write_start = time.perf_counter()
        for t in range(n):
            dataset.add_frame({
                "exterior_image_1_left": ep_data["ext_images"][t],
                "wrist_image_left": ep_data["wrist_images"][t],
                "joint_position": ep_data["joint_position"][t],
                "joint_velocity": ep_data["joint_velocity"][t],
                "gripper_position": np.array([ep_data["gripper_position"][t]], dtype=np.float32),
                "gripper_velocity": np.array([ep_data["gripper_velocity"][t]], dtype=np.float32),
                "ee_pose": ep_data["ee_pose"][t],
                "ee_velocity": ep_data["ee_velocity"][t],
                "actions": ep_data["actions"][t],
                "task": ep_data["instruction"],
            })
        dataset.save_episode()
        t_write = time.perf_counter() - t_write_start

        episodes_done += 1
        total_t_read += ep_t_read
        total_t_resize += ep_t_resize
        total_t_write += t_write
        total_t_wait += t_wait

        # Per-episode timing in progress bar
        pbar.set_postfix_str(
            f"{src}  read={ep_t_read:.1f}s resize={ep_t_resize:.1f}s "
            f"write={t_write:.1f}s wait={t_wait:.1f}s  [{n}fr]",
            refresh=False,
        )
        pbar.update(1)

    reader_thread.join()
    pbar.close()

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"Converted {episodes_done} episodes in {elapsed:.1f}s ({elapsed/episodes_done:.2f}s/ep)")
    print(f"  HDF5 read (bg):  {total_t_read:7.1f}s  ({total_t_read/episodes_done:.2f}s/ep)")
    print(f"  Image resize:    {total_t_resize:7.1f}s  ({total_t_resize/episodes_done:.2f}s/ep)")
    print(f"  LeRobot write:   {total_t_write:7.1f}s  ({total_t_write/episodes_done:.2f}s/ep)")
    print(f"  Queue wait:      {total_t_wait:7.1f}s  ({total_t_wait/episodes_done:.2f}s/ep)")
    overlap = (total_t_read + total_t_resize + total_t_write) - elapsed
    if overlap > 0:
        print(f"  I/O overlap:     {overlap:7.1f}s  (saved by pipeline)")
    print(f"Dataset saved to: {output_path}")
    print(f"{'='*60}")

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
    parser.add_argument(
        "--prefetch_depth",
        type=int,
        default=2,
        help="Number of episodes to prefetch in background (default: 2)",
    )

    args = parser.parse_args()

    if not Path(args.input_dir).exists():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    convert_trajectory_to_lerobot(
        input_dir=args.input_dir,
        repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
        local_dir=args.local_dir,
        prefetch_depth=args.prefetch_depth,
    )

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
