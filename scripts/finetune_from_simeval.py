#!/usr/bin/env python3
"""
End-to-end fine-tuning script for SimEval DROID recordings.

This script automates the complete pipeline:
1. Convert HDF5 trajectories → LeRobot format
2. (Optional) Visualize dataset
3. Compute normalization statistics
4. Fine-tune the model

Usage:
    python scripts/finetune_from_simeval.py \
        --input_dir /path/to/recorded_trajectories \
        --repo_id username/dataset_name \
        --config pi05_droid_simeval_lora \
        --exp_name my_experiment

Example:
    python scripts/finetune_from_simeval.py \
        --input_dir ../RegraspGen/my_regrasp/recorded_trajectories/scene1 \
        --repo_id mingxuanyan/simeval_scene1_droid \
        --config pi05_droid_simeval_lora \
        --exp_name scene1_lora_v1 \
        --dataset_dir /data/datasets \
        --visualize \
        --push_to_hub \
        --overwrite
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def run_command(cmd: list, env: dict = None, description: str = ""):
    """Run a subprocess command and handle errors."""
    print(f"\n{'▶ ' if description else ''}{description}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        print(f"\n❌ {description or 'Command'} failed with exit code {result.returncode}")
        sys.exit(1)

    return result


def visualize_dataset(repo_id: str, output_dir: str = "dataset_visualizations", local_dir: str = None):
    """Visualize the converted LeRobot dataset."""
    print("\n" + "="*70)
    print("OPTIONAL: Visualizing Dataset")
    print("="*70)

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        # Load dataset
        print(f"Loading dataset: {repo_id}")
        load_kwargs = {"repo_id": repo_id}
        if local_dir:
            # Use the full path to the dataset
            dataset_path = Path(local_dir) / repo_id
            load_kwargs["root"] = str(dataset_path)
            print(f"Using custom dataset directory: {dataset_path}")

        dataset = LeRobotDataset(**load_kwargs)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Print dataset statistics
        print("\n" + "-"*60)
        print("Dataset Statistics")
        print("-"*60)
        print(f"Total frames: {len(dataset)}")
        print(f"Number of episodes: {dataset.num_episodes}")
        print(f"FPS: {dataset.fps}")

        # Episode lengths
        episode_lengths = []
        for ep_idx in range(dataset.num_episodes):
            start = dataset.episode_data_index["from"][ep_idx].item()
            end = dataset.episode_data_index["to"][ep_idx].item()
            episode_lengths.append(end - start)

        print(f"\nEpisode lengths:")
        print(f"  Min: {min(episode_lengths)} frames")
        print(f"  Max: {max(episode_lengths)} frames")
        print(f"  Mean: {np.mean(episode_lengths):.1f} frames")
        print("-"*60)

        # Visualize each episode
        for ep_idx in range(min(dataset.num_episodes, 5)):  # Limit to first 5 episodes
            print(f"\nVisualizing episode {ep_idx}...")

            episode_start = dataset.episode_data_index["from"][ep_idx].item()
            episode_end = dataset.episode_data_index["to"][ep_idx].item()
            episode_length = episode_end - episode_start

            # Sample frames evenly across episode
            num_frames = min(8, episode_length)
            frame_indices = np.linspace(episode_start, episode_end - 1, num_frames, dtype=int).tolist()

            # Create visualization
            fig = plt.figure(figsize=(20, 12))
            gs = GridSpec(4, num_frames, figure=fig, hspace=0.4, wspace=0.3)

            # Plot frames
            for col_idx, frame_idx in enumerate(frame_indices):
                frame = dataset[frame_idx]

                # External camera
                ax_ext = fig.add_subplot(gs[0, col_idx])
                img_ext = frame["exterior_image_1_left"].permute(1, 2, 0).numpy()
                ax_ext.imshow(img_ext)
                ax_ext.set_title(f"Frame {frame_idx - episode_start}\nExternal", fontsize=8)
                ax_ext.axis('off')

                # Wrist camera
                ax_wrist = fig.add_subplot(gs[1, col_idx])
                img_wrist = frame["wrist_image_left"].permute(1, 2, 0).numpy()
                ax_wrist.imshow(img_wrist)
                ax_wrist.set_title("Wrist", fontsize=8)
                ax_wrist.axis('off')

            # Plot joint positions over time
            ax_joints = fig.add_subplot(gs[2, :])
            joint_positions = []
            for idx in range(episode_start, episode_end):
                joint_positions.append(dataset[idx]["joint_position"].numpy())
            joint_positions = np.array(joint_positions)

            for i in range(7):
                ax_joints.plot(joint_positions[:, i], label=f"Joint {i+1}", alpha=0.7)
            ax_joints.set_xlabel("Frame")
            ax_joints.set_ylabel("Position (rad)")
            ax_joints.set_title("Joint Positions Over Episode")
            ax_joints.legend(ncol=7, fontsize=8)
            ax_joints.grid(True, alpha=0.3)

            # Plot actions over time
            ax_actions = fig.add_subplot(gs[3, :])
            actions = []
            for idx in range(episode_start, episode_end):
                actions.append(dataset[idx]["actions"].numpy())
            actions = np.array(actions)

            for i in range(min(8, actions.shape[1])):
                label = f"Joint {i+1}" if i < 7 else "Gripper"
                ax_actions.plot(actions[:, i], label=label, alpha=0.7)
            ax_actions.set_xlabel("Frame")
            ax_actions.set_ylabel("Action")
            ax_actions.set_title("Actions Over Episode")
            ax_actions.legend(ncol=8, fontsize=8)
            ax_actions.grid(True, alpha=0.3)

            task_name = dataset[episode_start]['task']
            plt.suptitle(f"Episode {ep_idx}: {task_name}",
                        fontsize=14, fontweight='bold')
            plt.tight_layout()

            output_file = output_path / f"episode_{ep_idx}_visualization.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_file}")
            plt.close()

        if dataset.num_episodes > 5:
            print(f"\n(Visualized first 5 of {dataset.num_episodes} episodes)")

        print(f"\n✅ Visualizations saved to: {output_path}")

    except Exception as e:
        print(f"\n⚠️  Warning: Dataset visualization failed: {e}")
        print("Continuing with pipeline...")


def convert_to_lerobot(input_dir: str, repo_id: str, push_to_hub: bool, local_dir: str = None):
    """Convert HDF5 trajectories to LeRobot format."""
    print("\n" + "="*70)
    print("STEP 1: Converting HDF5 → LeRobot format")
    print("="*70)

    cmd = [
        "uv", "run",
        "src/openpi/training/convert_simeval_to_lerobot.py",
        "--input_dir", input_dir,
        "--repo_id", repo_id,
    ]

    if push_to_hub:
        cmd.append("--push_to_hub")

    if local_dir:
        cmd.extend(["--local_dir", local_dir])

    run_command(cmd, description="Converting trajectories to LeRobot format")
    print("✅ Conversion complete!")


def compute_norm_stats(config_name: str, repo_id: str):
    """Compute normalization statistics."""
    print("\n" + "="*70)
    print("STEP 2: Computing normalization statistics")
    print("="*70)

    cmd = [
        "uv", "run",
        "scripts/compute_norm_stats.py",
        "--config-name", config_name,
        f"data.repo_id={repo_id}"
    ]

    run_command(cmd, description="Computing normalization statistics")
    print("✅ Norm stats computed!")


def train_model(
    config_name: str,
    repo_id: str,
    exp_name: str,
    overwrite: bool,
    batch_size: int = None,
    num_train_steps: int = None,
    learning_rate: float = None,
    mem_fraction: float = 0.9
):
    """Launch training."""
    print("\n" + "="*70)
    print("STEP 3: Fine-tuning model")
    print("="*70)

    cmd = [
        "uv", "run",
        "scripts/train.py",
        config_name,
        f"data.repo_id={repo_id}",
        f"--exp-name={exp_name}"
    ]

    if overwrite:
        cmd.append("--overwrite")

    if batch_size is not None:
        cmd.append(f"batch_size={batch_size}")

    if num_train_steps is not None:
        cmd.append(f"num_train_steps={num_train_steps}")

    if learning_rate is not None:
        cmd.append(f"learning_rate={learning_rate}")

    # Set memory fraction
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)

    run_command(cmd, env=env, description="Fine-tuning model with LoRA")
    print("✅ Training complete!")


def print_next_steps(config: str, exp_name: str, repo_id: str):
    """Print next steps for evaluation."""
    print("\n" + "="*70)
    print("🎉 Fine-tuning pipeline complete!")
    print("="*70)
    print(f"\n📁 Checkpoints saved to: checkpoints/{config}/{exp_name}/")
    print(f"📊 Dataset: {repo_id}")

    # Determine model type for serving
    model_type = "pi05_droid" if "pi05" in config else "pi0_fast_droid"

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)

    print("\n1️⃣  Export checkpoint (optional - for inference):")
    print("   " + "-"*66)
    print(f"   uv run scripts/export_params.py \\")
    print(f"       --checkpoint-dir checkpoints/{config}/{exp_name} \\")
    print(f"       --output-dir exported_models/{exp_name} \\")
    print(f"       --step <step_number>")

    print("\n2️⃣  Serve fine-tuned policy:")
    print("   " + "-"*66)
    print(f"   uv run scripts/serve_policy.py {model_type} \\")
    print(f"       --checkpoint exported_models/{exp_name} \\")
    print(f"       --host 0.0.0.0 \\")
    print(f"       --port 9002")

    print("\n3️⃣  Evaluate in SimEval environment:")
    print("   " + "-"*66)
    print(f"   cd ../RegraspGen/my_regrasp")
    print(f"   python scripts/run_simeval_droid_env.py \\")
    print(f"       --episodes 20 \\")
    print(f"       --scene 1 \\")
    print(f"       --policy_host <your_host> \\")
    print(f"       --policy_port 9002")

    print("\n💡 TIP: To visualize the dataset, re-run with --visualize flag")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end fine-tuning from SimEval recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with LoRA (recommended)
  python scripts/finetune_from_simeval.py \\
      --input_dir ../RegraspGen/my_regrasp/recorded_trajectories/scene1 \\
      --repo_id mingxuanyan/simeval_scene1_droid \\
      --config pi05_droid_simeval_lora \\
      --exp_name scene1_lora_v1 \\
      --overwrite

  # With custom dataset directory and visualization
  python scripts/finetune_from_simeval.py \\
      --input_dir ../RegraspGen/my_regrasp/recorded_trajectories/scene1 \\
      --repo_id mingxuanyan/simeval_scene1_droid \\
      --config pi05_droid_simeval_lora \\
      --exp_name scene1_lora_v1 \\
      --dataset_dir /data/datasets \\
      --visualize \\
      --overwrite

  # With custom training parameters
  python scripts/finetune_from_simeval.py \\
      --input_dir ../RegraspGen/my_regrasp/recorded_trajectories/scene1 \\
      --repo_id mingxuanyan/simeval_scene1_droid \\
      --config pi05_droid_simeval_lora \\
      --exp_name scene1_custom \\
      --batch_size 16 \\
      --num_train_steps 10000 \\
      --learning_rate 3e-4 \\
      --overwrite

  # Skip conversion if dataset already exists
  python scripts/finetune_from_simeval.py \\
      --input_dir ../RegraspGen/my_regrasp/recorded_trajectories/scene1 \\
      --repo_id mingxuanyan/simeval_scene1_droid \\
      --config pi05_droid_simeval_lora \\
      --exp_name scene1_retry \\
      --skip_conversion \\
      --overwrite
        """
    )

    # Required arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing HDF5 trajectory files from SimEval recording"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="LeRobot dataset repo ID (e.g., username/dataset_name)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=[
            "pi0_fast_droid_joinpos_simeval_test",
            "pi05_droid_joinpos_simeval_test",
            "pi05_droid_simeval_lora"
        ],
        help="Training configuration to use (LoRA recommended for memory efficiency)"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name for checkpoint directory"
    )

    # Dataset options
    dataset_group = parser.add_argument_group("Dataset options")
    dataset_group.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push converted dataset to HuggingFace Hub"
    )
    dataset_group.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Custom directory for storing intermediate dataset (saves home dir disk space). "
             "Dataset will be stored in <dataset_dir>/<repo_id>"
    )
    dataset_group.add_argument(
        "--skip_conversion",
        action="store_true",
        help="Skip conversion step (use if dataset already exists)"
    )
    dataset_group.add_argument(
        "--skip_norm_stats",
        action="store_true",
        help="Skip norm stats computation (use if stats already exist)"
    )
    dataset_group.add_argument(
        "--visualize",
        action="store_true",
        help="Generate dataset visualizations after conversion"
    )

    # Training options
    training_group = parser.add_argument_group("Training options")
    training_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing checkpoint directory"
    )
    training_group.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size (default: from config, typically 8 for LoRA)"
    )
    training_group.add_argument(
        "--num_train_steps",
        type=int,
        default=None,
        help="Number of training steps (default: from config, typically 5000)"
    )
    training_group.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: from config)"
    )
    training_group.add_argument(
        "--mem_fraction",
        type=float,
        default=0.9,
        help="XLA memory fraction (default: 0.9)"
    )

    args = parser.parse_args()

    # Verify input directory exists
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Check for HDF5 files
    hdf5_files = list(input_path.glob("*.h5"))
    if not hdf5_files and not args.skip_conversion:
        print(f"❌ Error: No HDF5 files found in {args.input_dir}")
        sys.exit(1)

    # Print configuration summary
    print("\n" + "="*70)
    print("SimEval DROID Fine-Tuning Pipeline")
    print("="*70)
    print(f"\n📂 Input directory:     {args.input_dir}")
    print(f"📦 Dataset repo ID:     {args.repo_id}")
    if args.dataset_dir:
        print(f"💾 Dataset directory:   {args.dataset_dir}/{args.repo_id}")
    print(f"⚙️  Training config:     {args.config}")
    print(f"🏷️  Experiment name:     {args.exp_name}")
    print(f"☁️  Push to hub:         {args.push_to_hub}")
    print(f"♻️  Overwrite existing:  {args.overwrite}")
    print(f"📊 Visualize dataset:   {args.visualize}")

    if not args.skip_conversion:
        print(f"\n📊 Found {len(hdf5_files)} HDF5 file(s) to convert")

    training_params = []
    if args.batch_size:
        training_params.append(f"batch_size={args.batch_size}")
    if args.num_train_steps:
        training_params.append(f"num_train_steps={args.num_train_steps}")
    if args.learning_rate:
        training_params.append(f"learning_rate={args.learning_rate}")

    if training_params:
        print(f"\n🎛️  Custom parameters:    {', '.join(training_params)}")

    print("="*70)

    # Confirm before proceeding
    try:
        response = input("\nProceed with fine-tuning pipeline? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Aborted by user")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n❌ Aborted by user")
        sys.exit(0)

    # Execute pipeline
    try:
        # Step 1: Convert to LeRobot
        if not args.skip_conversion:
            convert_to_lerobot(
                args.input_dir,
                args.repo_id,
                args.push_to_hub,
                args.dataset_dir
            )
        else:
            print("\n⏭️  Skipping conversion (--skip_conversion specified)")

        # Optional: Visualize dataset
        if args.visualize:
            visualize_dataset(
                args.repo_id,
                output_dir="dataset_visualizations",
                local_dir=args.dataset_dir
            )

        # Step 2: Compute norm stats
        if not args.skip_norm_stats:
            compute_norm_stats(args.config, args.repo_id)
        else:
            print("\n⏭️  Skipping norm stats computation (--skip_norm_stats specified)")

        # Step 3: Train model
        train_model(
            args.config,
            args.repo_id,
            args.exp_name,
            args.overwrite,
            args.batch_size,
            args.num_train_steps,
            args.learning_rate,
            args.mem_fraction
        )

        # Print next steps
        print_next_steps(args.config, args.exp_name, args.repo_id)

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
