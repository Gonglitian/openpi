# Pi0.5 DROID Environment Installation Guide

This guide provides step-by-step instructions for installing the pi0.5 DROID environment.

## Prerequisites

- Ubuntu 22.04 (tested, other OS not currently supported)
- NVIDIA GPU with at least 8GB memory (for inference) or 70GB+ (for full fine-tuning)
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) package manager installed

## Installation Steps

### 1. Clone the Repository with Submodules

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git
cd openpi

# If you already cloned the repo without submodules:
git submodule update --init --recursive
```

### 2. Install Basic OpenPI Dependencies

```bash
# Clean uv cache if you encounter any cache corruption issues
uv cache clean

# Install base dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Install openpi package in editable mode
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**Note:** `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

### 3. Install RLDS Dependencies (Required for Full DROID Dataset Training)

If you plan to train on the full DROID dataset (1.8TB), you need additional RLDS dependencies:

```bash
uv sync --group rlds
```

**Note:** This step is optional if you only plan to:
- Run inference with pre-trained models
- Fine-tune on smaller custom datasets (using LeRobot format)

### 4. Verify Installation

```bash
uv run python -c "import openpi; import jax; import torch; print('Installation successful!')"
```

You may see some warnings about cuDNN/cuFFT/cuBLAS factory registration - these are normal and don't indicate problems.

## What's Installed

After completing these steps, you have:

1. **Basic OpenPI environment** - Ready for:
   - Running inference with pi0.5-DROID model
   - Fine-tuning on custom datasets using LeRobot format
   - Training on ALOHA, LIBERO, and other platforms

2. **RLDS dependencies** (if installed) - Ready for:
   - Training on the full DROID dataset (1.8TB)
   - Large-scale robot learning experiments

## Quick Start: Launch Pi0.5 DROID Policy Server

We provide convenient launch scripts to start the pi0.5 DROID policy server:

### Option 1: Using the Python launcher (recommended)

```bash
# Launch with default settings
python launch_pi05_droid.py

# Launch on a custom port
python launch_pi05_droid.py --port 9000

# List available GPUs (useful if you have multiple GPUs)
python launch_pi05_droid.py --list-gpus

# Launch with specific GPU
python launch_pi05_droid.py --gpu 2

# List available IP addresses (useful if you have multiple network interfaces)
python launch_pi05_droid.py --list-ips

# Launch with specific IP address
python launch_pi05_droid.py --host 192.168.1.100

# Complete example: GPU 2, custom IP and port
python launch_pi05_droid.py --gpu 2 --host 100.79.185.61 --port 9001

# Launch with a custom fine-tuned checkpoint
python launch_pi05_droid.py --checkpoint checkpoints/pi05_droid/my_experiment/20000

# View all options
python launch_pi05_droid.py --help
```

### Option 2: Using the bash launcher

```bash
# Launch with default settings
./launch_pi05_droid.sh

# Launch on a custom port
./launch_pi05_droid.sh --port 9000

# View all options
./launch_pi05_droid.sh --help
```

### Option 3: Using the underlying script directly

```bash
# Launch the default pi0.5-DROID checkpoint
uv run scripts/serve_policy.py --env=DROID

# Or specify checkpoint explicitly
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

The policy server will automatically:
- Download the checkpoint if needed (cached in `~/.cache/openpi`)
- Display the server's IP address and port
- Start listening for inference requests

## Next Steps

### For Running Inference

See the [DROID README](examples/droid/README.md) for instructions on:
- Connecting the DROID robot to the policy server
- Running the pi0.5-DROID model on your robot

### For Training on Full DROID Dataset

See the [DROID Training README](examples/droid/README_train.md) for instructions on:
- Downloading the DROID dataset (requires gsutil and 1.8TB storage)
- Computing normalization statistics
- Launching training

### For Fine-Tuning on Custom Datasets

See the main [README](README.md) for examples on:
- Converting your data to LeRobot format
- Defining training configs
- Running fine-tuning experiments

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `uv sync` fails with dependency conflicts | Try `rm -rf .venv` and run `uv sync` again. Update uv: `uv self update` |
| Cache corruption errors | Run `uv cache clean` and retry installation |
| Import errors | Make sure you've run both `uv sync` and `uv pip install -e .` |
| CUDA errors | Verify NVIDIA drivers. You don't need system CUDA libraries - they're installed via uv |

## Hardware Requirements

| Use Case | GPU Memory | Example GPU |
|----------|------------|-------------|
| Inference | > 8 GB | RTX 4090 |
| Fine-Tuning (LoRA) | > 22.5 GB | RTX 4090 |
| Fine-Tuning (Full) | > 70 GB | A100 (80GB) / H100 |
| Full DROID Training | 8x H100 | ~2 days for convergence |

For more details, see the main [README](README.md).
