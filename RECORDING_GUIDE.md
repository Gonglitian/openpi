# Policy Recording and Visualization Guide

This guide explains how to record and visualize policy inference data for debugging and analysis.

## What Gets Recorded

When you enable recording with `--record`, the policy server saves each inference step to disk, including:
- **Input observations**: Camera images, robot state, language prompts
- **Output actions**: Predicted action trajectories
- **Timing information**: Inference latency
- **Metadata**: All data passed to and from the policy

## Recording Data

### Enable Recording When Launching Server

```bash
# Python launcher
python launch_pi05_droid.py --host 100.79.185.61 --port 9001 --record

# Bash launcher
./launch_pi05_droid.sh --host 100.79.185.61 -p 9001 --record
```

### Where Data is Saved

By default, recordings are saved to `policy_records/` directory:
```
policy_records/
├── step_0.npy
├── step_1.npy
├── step_2.npy
└── ...
```

Each file contains:
- `inputs/`: All input observations (images, state, prompt, etc.)
- `outputs/`: Predicted actions and timing info

## Visualizing Recordings

We provide a `visualize_recordings.py` script for viewing recorded data.

### Quick Start

```bash
# Show information about recordings
python visualize_recordings.py policy_records --info

# Visualize a specific step
python visualize_recordings.py policy_records --step 0

# Generate images for all recordings
python visualize_recordings.py policy_records --all --output viz_output

# Create a video from recordings
python visualize_recordings.py policy_records --video output.mp4
```

### Detailed Usage

#### 1. List Recording Information

See what was recorded without visualizing:

```bash
python visualize_recordings.py policy_records --step 0 --info
```

**Output:**
```
Recording Information
============================================================

Inputs:
  observation/exterior_image_1_left: shape=(224, 224, 3), dtype=float32
  observation/wrist_image_left: shape=(224, 224, 3), dtype=float32
  observation/joint_position: shape=(7,), dtype=float32
  observation/gripper_position: shape=(1,), dtype=float32
  prompt: pick up the fork

Outputs:
  actions: shape=(10, 8), dtype=float32
  state: shape=(8,), dtype=float32
  policy_timing:
    infer_ms: 145.23
```

#### 2. Visualize a Single Step

View images and action predictions:

```bash
python visualize_recordings.py policy_records --step 0
```

This opens an interactive plot showing:
- Camera images (external + wrist views)
- Predicted action trajectory over time
- Action statistics per dimension
- Inference timing

#### 3. Generate Visualizations for All Steps

Save visualization images for all recorded steps:

```bash
# Create output directory
mkdir viz_output

# Generate images
python visualize_recordings.py policy_records --all --output viz_output
```

This creates:
```
viz_output/
├── step_0.png
├── step_1.png
├── step_2.png
└── ...
```

#### 4. Create a Video

Generate a video from the recorded camera images:

```bash
python visualize_recordings.py policy_records --video rollout.mp4
```

You can adjust the frame rate:

```python
# Edit visualize_recordings.py, line with create_video:
create_video(record_dir, args.video, fps=15)  # Change fps here
```

## Advanced Usage

### Load and Analyze Recordings Programmatically

```python
import numpy as np
from visualize_recordings import load_recording, unflatten_dict

# Load a recording
data = load_recording("policy_records/step_0.npy")
data = unflatten_dict(data)

# Access inputs
images = data["inputs"]["observation/exterior_image_1_left"]
robot_state = data["inputs"]["observation/joint_position"]
prompt = data["inputs"]["prompt"]

# Access outputs
actions = data["outputs"]["actions"]
timing = data["outputs"]["policy_timing"]

print(f"Actions shape: {actions.shape}")
print(f"Inference time: {timing['infer_ms']:.2f} ms")

# Analyze action distribution
import matplotlib.pyplot as plt
plt.plot(actions)
plt.xlabel("Time step")
plt.ylabel("Action value")
plt.title("Predicted Actions")
plt.legend([f"Dim {i}" for i in range(actions.shape[1])])
plt.show()
```

### Batch Analysis

Analyze all recordings to compute statistics:

```python
import pathlib
import numpy as np
from visualize_recordings import load_recording, unflatten_dict

record_dir = pathlib.Path("policy_records")
recording_files = sorted(record_dir.glob("step_*.npy"))

inference_times = []
action_magnitudes = []

for record_file in recording_files:
    data = unflatten_dict(load_recording(record_file))

    # Collect inference times
    timing = data["outputs"]["policy_timing"]
    inference_times.append(timing["infer_ms"])

    # Collect action magnitudes
    actions = data["outputs"]["actions"]
    action_magnitudes.append(np.linalg.norm(actions, axis=-1).mean())

print(f"Average inference time: {np.mean(inference_times):.2f} ms")
print(f"Std inference time: {np.std(inference_times):.2f} ms")
print(f"Average action magnitude: {np.mean(action_magnitudes):.4f}")
```

## Use Cases

### 1. Debugging Policy Behavior

Record a failing rollout and inspect what the policy was seeing and predicting:

```bash
# Run server with recording
python launch_pi05_droid.py --record --host 100.79.185.61 -p 9001

# On DROID robot, run the task
# (it fails at step 50)

# Visualize the failing step
python visualize_recordings.py policy_records --step 50
```

### 2. Analyzing Inference Latency

Check if inference is fast enough for real-time control:

```python
import pathlib
import numpy as np
from visualize_recordings import load_recording, unflatten_dict

record_dir = pathlib.Path("policy_records")
times = []

for record_file in sorted(record_dir.glob("step_*.npy")):
    data = unflatten_dict(load_recording(record_file))
    times.append(data["outputs"]["policy_timing"]["infer_ms"])

print(f"Mean: {np.mean(times):.2f} ms")
print(f"Median: {np.median(times):.2f} ms")
print(f"95th percentile: {np.percentile(times, 95):.2f} ms")
print(f"Max: {np.max(times):.2f} ms")
```

### 3. Comparing Different Policies

Record multiple rollouts with different policies and compare:

```bash
# Policy 1: pi0.5-DROID
python launch_pi05_droid.py --record --port 9001
# Run task, recordings in policy_records/

# Move recordings
mv policy_records policy_records_pi05

# Policy 2: pi0-FAST-DROID
python launch_pi05_droid.py --record --port 9001 \
    --checkpoint gs://openpi-assets/checkpoints/pi0_fast_droid \
    --config pi0_fast_droid
# Run task again

# Move recordings
mv policy_records policy_records_pi0_fast

# Compare
python visualize_recordings.py policy_records_pi05 --all --output viz_pi05
python visualize_recordings.py policy_records_pi0_fast --all --output viz_pi0_fast
```

### 4. Creating Demonstration Videos

Record a successful rollout and create a video:

```bash
# Record
python launch_pi05_droid.py --record --host 100.79.185.61 -p 9001

# After rollout completes, create video
python visualize_recordings.py policy_records --video success_rollout.mp4

# Share the video
```

## Performance Considerations

### Storage

Each recording typically uses:
- **~1-2 MB per step** (depends on image resolution and action horizon)
- A 100-step rollout = ~100-200 MB

### Latency

Recording adds **minimal overhead** (~0.1-1 ms per step):
- Just a numpy save operation
- Doesn't affect real-time performance

### Cleanup

Remove old recordings to free space:

```bash
# Remove all recordings
rm -rf policy_records/

# Remove specific rollout (steps 0-100)
rm policy_records/step_{0..100}.npy

# Keep only recent recordings
ls -t policy_records/step_*.npy | tail -n +51 | xargs rm
```

## Troubleshooting

### Issue: "No recordings found"

**Solution:** Make sure the server was launched with `--record` flag and you ran at least one inference.

### Issue: "Cannot load recording file"

**Solution:** The recording might be corrupted. Try deleting it:
```bash
rm policy_records/step_<N>.npy
```

### Issue: Visualizations show blank images

**Solution:** Check image normalization. Images should be in [0, 1] range for float32 or [0, 255] for uint8.

### Issue: moviepy not installed for video creation

**Solution:** Install moviepy:
```bash
pip install moviepy
# or
uv pip install moviepy
```

## File Format

Recordings are saved as numpy `.npy` files containing a flattened dictionary:

```python
{
    'inputs/observation/exterior_image_1_left': array([...]),
    'inputs/observation/wrist_image_left': array([...]),
    'inputs/observation/joint_position': array([...]),
    'inputs/prompt': 'pick up the fork',
    'outputs/actions': array([...]),
    'outputs/policy_timing/infer_ms': 145.23,
    ...
}
```

Keys use `/` as separator. Use `unflatten_dict()` to convert back to nested structure.

## Related Documentation

- [Launch Scripts README](LAUNCH_SCRIPTS_README.md) - How to launch server with recording
- [Quick Start Guide](QUICK_START.md) - Getting started
- [DROID Inference Guide](examples/droid/README.md) - Running on DROID robot

## Tips

1. **Always record during development** - It's invaluable for debugging
2. **Use meaningful names** - Copy recordings to descriptive directories after each experiment
3. **Clean up regularly** - Recordings can use significant disk space
4. **Analyze timing** - Check if inference is fast enough for your control loop
5. **Compare policies** - Record with different checkpoints to compare behavior
