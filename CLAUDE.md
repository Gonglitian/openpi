# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenPI is Physical Intelligence's open-source VLA (Vision-Language-Action) training and serving framework. It supports **Pi0**, **Pi0.5**, and **Pi0-FAST** model families with JAX+Flax (primary) and PyTorch backends. Used in RegraspGen for LoRA fine-tuning on simulated regrasp trajectories.

## Build & Run

Package manager: `uv` (workspace-based). Python >= 3.11. Use **conda env `env_isaaclab`**.

```bash
conda activate env_isaaclab
cd libs/openpi

# Install dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Compute normalization stats (required before training)
GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/compute_norm_stats.py --config-name pi05_droid_simeval_lora

# Train with LoRA
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 GIT_LFS_SKIP_SMUDGE=1 uv run python scripts/train.py \
    pi05_droid_simeval_lora --exp-name=my_experiment

# Serve a trained policy (WebSocket on port 8000)
uv run python scripts/serve_policy.py --config pi05_droid_simeval_lora --checkpoint checkpoints/...
```

### Key Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | JAX training with FSDP, LoRA, WandB |
| `scripts/train_pytorch.py` | PyTorch training alternative |
| `scripts/compute_norm_stats.py` | Compute mean/std for normalization |
| `scripts/serve_policy.py` | WebSocket inference server |

## Architecture

### Config System (`src/openpi/training/config.py`)

All configs are frozen `@dataclass` objects composed via `TrainConfig`:

```python
TrainConfig(
    name="...",
    model=Pi0Config(...) | Pi0FASTConfig(...),    # Model architecture
    data=SimpleDataConfig(...) | RLDSDroidDataConfig(...),  # Dataset + transforms
    weight_loader=CheckpointWeightLoader("gs://..."),       # Pre-trained weights
    # + training hyperparams: batch_size, num_train_steps, fsdp_devices, etc.
)
```

**Data Config Variants**:
- `SimpleDataConfig` — Custom LeRobot datasets (RegraspGen uses this). **Does NOT auto-add DeltaActions**.
- `RLDSDroidDataConfig` — DROID RLDS format. Auto-adds DeltaActions based on `action_space`.
- `LeRobotAlohaDataConfig`, `LeRobotLiberoDataConfig` — Robot-specific presets.

Access pre-defined configs: `_config.get_config("pi05_droid_simeval_lora")`.

### Transform Pipeline (`src/openpi/transforms.py`)

Transforms are organized in a `Group(inputs=[], outputs=[])`:
- **inputs**: Applied during training (data → model)
- **outputs**: Applied during inference (model → robot), reversed order

```python
Group(
    inputs=[
        RepackTransform({...}),           # Rename/restructure dict keys
        droid_policy.DroidInputs(...),    # DROID-specific state assembly
        DeltaActions(make_bool_mask(7,-1)), # Absolute → delta (7 joints, 1 gripper absolute)
    ],
    outputs=[
        AbsoluteActions(make_bool_mask(7,-1)), # Delta → absolute (inverse)
        droid_policy.DroidOutputs(),
    ],
)
```

**Critical transforms**:
- `DeltaActions(mask)`: `actions[:7] -= state[:7]` — converts absolute joint positions to deltas
- `AbsoluteActions(mask)`: Inverse of DeltaActions during inference
- `make_bool_mask(7, -1)`: Creates `[T,T,T,T,T,T,T,F]` — 7 joints delta, 1 gripper absolute
- `Normalize` / `Unnormalize`: Z-score or quantile normalization
- `ResizeImages`: Resize images to 224x224 for model input

Full pipeline order: repack → data_transforms → normalize → model_transforms.

### Model Types

```python
class ModelType(enum.Enum):
    PI0 = "pi0"        # 2B PaliGemma + 300M action expert
    PI0_FAST = "pi0_fast"  # Autoregressive with FASTTokenizer
    PI05 = "pi05"       # Pi0 with discrete_state_input (multi-task)
```

### Weight Loaders (`src/openpi/training/weight_loaders.py`)

- `CheckpointWeightLoader(path)`: Load from local or GCS (`gs://openpi-assets/...`)
- `PaliGemmaWeightLoader`: Load PaliGemma base weights
- LoRA: Missing LoRA weights auto-merged via `_merge_params(loaded, params, missing_regex=".*lora.*")`

## RegraspGen Integration

### Config: `pi05_droid_simeval_lora`

Located in `src/openpi/training/config.py` (search for `pi05_droid_simeval_lora`). Key settings:

- **Base model**: PolaRiS cotrained Pi0.5 (`gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/params`)
- **repo_id**: Must match `HF_LEROBOT_HOME` + LeRobot dataset name (e.g., `regraspgen/PlayingCardsKitchen`)
- **DeltaActions**: Must be explicitly added in `data_transforms` (SimpleDataConfig doesn't auto-add)
- **Image features**: `exterior_image_1_left` (180x320) and `wrist_image_left` (180x320)
- **Action dim**: 8 (7 joint positions + 1 gripper)

### Data Conversion (`src/openpi/training/convert_simeval_to_lerobot.py`)

Converts HDF5 trajectories to LeRobot format:
- Actions: `joint_position_command` (7D) + `gripper_command` (1D) = 8D absolute
- Images: 180x320 (skip resize if already correct size)
- FPS: Check HDF5 metadata, only downsample if source > 15Hz
- `--local_dir`: Custom dataset location (avoids filling home dir)

### Environment Variables

```bash
export HF_LEROBOT_HOME="/path/to/lerobot_dataset"  # LeRobot dataset root
export OPENPI_DATA_HOME="/path/to/openpi_cache"     # Norm stats & assets cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9           # JAX GPU memory
```

## Key Checkpoints

| Name | Path | Description |
|------|------|-------------|
| Pi0.5 base | `gs://openpi-assets/checkpoints/pi05_base/params` | Pre-trained base |
| Pi0.5 DROID | `gs://openpi-assets/checkpoints/pi05_droid/params` | DROID fine-tuned |
| PolaRiS Pi0.5 | `gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris/params` | 90% DROID + 10% sim cotrained |

## Conventions

- Config as frozen dataclass: Always pass `TrainConfig` or sub-configs, not loose params
- Image key names: `exterior_image_1_left`, `wrist_image_left` (DROID standard)
- Normalization: Quantile norm for Pi0.5+ (multi-task), Z-score for Pi0
- Checkpoint structure: `params` (JAX) or `model.safetensors` (PyTorch)
- `GIT_LFS_SKIP_SMUDGE=1`: Always prefix `uv run` commands to avoid downloading LFS files
