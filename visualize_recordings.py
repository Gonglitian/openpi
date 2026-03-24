#!/usr/bin/env python3
"""
Visualize policy recordings created with --record flag.

This script loads and visualizes the recorded inference data from the policy server.
"""

import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json


def load_recording(record_path: pathlib.Path) -> dict:
    """Load a single recording file."""
    data = np.load(record_path, allow_pickle=True).item()
    return data


def unflatten_dict(flat_dict: dict, sep: str = "/") -> dict:
    """Unflatten a dictionary with separator-based keys."""
    result = {}
    for flat_key, value in flat_dict.items():
        parts = flat_key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def print_recording_info(data: dict):
    """Print information about a recording."""
    print("\n" + "=" * 60)
    print("Recording Information")
    print("=" * 60)

    # Unflatten for easier reading
    unflat = unflatten_dict(data)

    # Show inputs
    if "inputs" in unflat:
        print("\nInputs:")
        for key, value in unflat["inputs"].items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value}")

    # Show outputs
    if "outputs" in unflat:
        print("\nOutputs:")
        for key, value in unflat["outputs"].items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")


def visualize_recording(data: dict, save_path: str = None):
    """Visualize a single recording with images and actions."""
    unflat = unflatten_dict(data)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Plot images
    images_plotted = 0
    image_keys = []

    # Look for images in inputs
    if "inputs" in unflat:
        for key, value in unflat["inputs"].items():
            if isinstance(value, np.ndarray) and "image" in key.lower() and value.ndim >= 2:
                image_keys.append((key, value))

    # Plot up to 3 images
    for idx, (key, img) in enumerate(image_keys[:3]):
        ax = fig.add_subplot(gs[0, idx])

        # Handle different image formats
        if img.ndim == 4:  # (batch, height, width, channels)
            img = img[0]
        elif img.ndim == 2:  # (height, width)
            pass

        # Normalize if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.min() < 0 or img.max() > 1:
                img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.set_title(key)
        ax.axis('off')
        images_plotted = idx + 1

    # Plot actions
    if "outputs" in unflat and "actions" in unflat["outputs"]:
        actions = unflat["outputs"]["actions"]

        # Plot action trajectory
        ax = fig.add_subplot(gs[1, :])

        if actions.ndim == 2:  # (horizon, action_dim)
            for i in range(actions.shape[1]):
                ax.plot(actions[:, i], label=f"Action dim {i}", marker='o')
            ax.set_xlabel("Time step")
            ax.set_ylabel("Action value")
            ax.set_title("Predicted Action Trajectory")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)

        # Plot action statistics per dimension
        ax = fig.add_subplot(gs[2, :])

        if actions.ndim == 2:
            action_means = actions.mean(axis=0)
            action_stds = actions.std(axis=0)
            x = np.arange(len(action_means))

            ax.bar(x, action_means, yerr=action_stds, capsize=5, alpha=0.7)
            ax.set_xlabel("Action dimension")
            ax.set_ylabel("Mean value")
            ax.set_title("Action Statistics (Mean ± Std across time)")
            ax.grid(True, alpha=0.3, axis='y')

    # Add timing info if available
    if "outputs" in unflat and "policy_timing" in unflat["outputs"]:
        timing = unflat["outputs"]["policy_timing"]
        title_text = f"Inference time: {timing.get('infer_ms', 'N/A'):.2f} ms"
        fig.suptitle(title_text, fontsize=14, y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()


def visualize_all_recordings(record_dir: pathlib.Path, output_dir: pathlib.Path = None):
    """Visualize all recordings in a directory."""
    # Sort recording files numerically by step number
    recording_files = sorted(
        record_dir.glob("step_*.npy"),
        key=lambda p: int(p.stem.split('_')[1])
    )

    if not recording_files:
        print(f"No recordings found in {record_dir}")
        return

    print(f"Found {len(recording_files)} recordings")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, record_file in enumerate(recording_files):
        print(f"\nProcessing {record_file.name} ({i+1}/{len(recording_files)})")

        try:
            data = load_recording(record_file)

            if output_dir:
                save_path = output_dir / f"{record_file.stem}.png"
                visualize_recording(data, save_path=str(save_path))
            else:
                print_recording_info(data)

        except Exception as e:
            print(f"Error processing {record_file}: {e}")


def create_video(record_dir: pathlib.Path, output_path: str, fps: int = 10):
    """Create a video from recorded images."""
    try:
        from moviepy import ImageSequenceClip
    except ImportError:
        try:
            # Try old API for older moviepy versions
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            print("moviepy not installed. Install with: uv pip install moviepy")
            return

    # Sort recording files numerically by step number
    recording_files = sorted(
        record_dir.glob("step_*.npy"),
        key=lambda p: int(p.stem.split('_')[1])
    )

    if not recording_files:
        print(f"No recordings found in {record_dir}")
        return

    print(f"Creating video from {len(recording_files)} recordings...")

    # Helper function to find all images in a recording
    def find_all_images(d, path=""):
        images = []
        if isinstance(d, dict):
            for key, value in d.items():
                images.extend(find_all_images(value, f"{path}/{key}" if path else key))
        elif isinstance(d, np.ndarray) and "image" in path.lower() and d.ndim >= 2:
            img = d
            if img.ndim == 4:
                img = img[0]

            # Normalize
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            elif img.dtype == np.uint8:
                # Already in correct format
                pass

            images.append((path, img))
        return images

    # First pass: determine max number of images and target dimensions
    max_num_images = 0
    target_height = None
    target_width = None

    for record_file in recording_files:
        data = load_recording(record_file)
        unflat = unflatten_dict(data)
        all_images = find_all_images(unflat.get("inputs", {}))

        if all_images:
            all_images.sort(key=lambda x: x[0])
            max_num_images = max(max_num_images, len(all_images))

            # Set target dimensions from first image found
            if target_height is None or target_width is None:
                _, first_img = all_images[0]
                target_height = first_img.shape[0]
                target_width = first_img.shape[1]

    if max_num_images == 0:
        print("No images found in recordings")
        return

    print(f"Max images per frame: {max_num_images}")
    print(f"Target image size: {target_width}x{target_height}")
    print(f"Output video size: {target_width * max_num_images}x{target_height}")

    # Second pass: create frames with padding for recordings with fewer images
    frames = []

    for record_file in recording_files:
        data = load_recording(record_file)
        unflat = unflatten_dict(data)
        all_images = find_all_images(unflat.get("inputs", {}))

        if all_images:
            # Sort images by name for consistent ordering
            all_images.sort(key=lambda x: x[0])
            imgs = [img for _, img in all_images]

            # Resize all images to target size
            resized_imgs = []
            try:
                import cv2
                cv2_available = True
            except ImportError:
                cv2_available = False

            for img in imgs:
                if img.shape[0] != target_height or img.shape[1] != target_width:
                    if cv2_available:
                        img = cv2.resize(img, (target_width, target_height))
                    else:
                        print(f"Warning: cv2 not available, skipping {record_file.name}")
                        resized_imgs = []
                        break
                resized_imgs.append(img)

            if len(resized_imgs) > 0:
                # Pad with black images if this recording has fewer images than max
                while len(resized_imgs) < max_num_images:
                    # Create black padding image
                    black_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    resized_imgs.append(black_img)

                # Concatenate images horizontally (side-by-side)
                combined_img = np.concatenate(resized_imgs, axis=1)
                frames.append(combined_img)

    if frames:
        print(f"Creating video with {len(frames)} frames...")
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(output_path, codec='libx264')
        print(f"Video saved to: {output_path}")
    else:
        print("No valid frames created")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize policy recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List information about recordings
  python visualize_recordings.py policy_records --info

  # Visualize a single recording
  python visualize_recordings.py policy_records --step 0

  # Generate visualization images for all recordings
  python visualize_recordings.py policy_records --all --output viz_output

  # Create a video from recordings
  python visualize_recordings.py policy_records --video output.mp4
        """
    )

    parser.add_argument(
        "record_dir",
        type=str,
        help="Directory containing policy recordings (default: policy_records)"
    )

    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Visualize a specific step number"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all recordings"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Print information about recordings without visualizing"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for saved visualizations"
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Create a video from recordings (specify output filename)"
    )

    args = parser.parse_args()

    record_dir = pathlib.Path(args.record_dir)

    if not record_dir.exists():
        print(f"Error: Directory {record_dir} does not exist")
        return

    # Create video
    if args.video:
        create_video(record_dir, args.video)
        return

    # Visualize specific step
    if args.step is not None:
        record_file = record_dir / f"step_{args.step}.npy"
        if not record_file.exists():
            print(f"Error: {record_file} does not exist")
            return

        data = load_recording(record_file)
        print_recording_info(data)

        if not args.info:
            visualize_recording(data)
        return

    # Process all recordings
    if args.all:
        output_dir = pathlib.Path(args.output) if args.output else None
        visualize_all_recordings(record_dir, output_dir)
        return

    # Default: show info about all recordings
    recording_files = sorted(
        record_dir.glob("step_*.npy"),
        key=lambda p: int(p.stem.split('_')[1])
    )
    print(f"\nFound {len(recording_files)} recordings in {record_dir}")
    print("\nUse --step N to visualize a specific recording")
    print("Use --all to process all recordings")
    print("Use --video output.mp4 to create a video")


if __name__ == "__main__":
    main()
