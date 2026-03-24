# CUDA Device Selection for Pi0.5 DROID

This guide explains how to control which GPU(s) the pi0.5 DROID policy server uses.

## Default Behavior

By default, JAX (which powers the pi0 models) will:
- **Use GPU 0** if available
- Automatically detect and use all available GPUs for model parallelism if configured

## Selecting a Specific GPU

Use the `CUDA_VISIBLE_DEVICES` environment variable to control which GPU(s) are visible to the application:

### Using Python Launcher

```bash
# Use GPU 0 (default)
CUDA_VISIBLE_DEVICES=0 python launch_pi05_droid.py

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 python launch_pi05_droid.py --host 100.79.185.61 -p 9001

# Use GPU 2
CUDA_VISIBLE_DEVICES=2 python launch_pi05_droid.py

# Use multiple GPUs (GPUs 0 and 1)
CUDA_VISIBLE_DEVICES=0,1 python launch_pi05_droid.py

# Use GPUs 3, 4, 5 (for multi-GPU inference)
CUDA_VISIBLE_DEVICES=3,4,5 python launch_pi05_droid.py
```

### Using Bash Launcher

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 ./launch_pi05_droid.sh --host 192.168.1.100

# Use GPU 2
CUDA_VISIBLE_DEVICES=2 ./launch_pi05_droid.sh --port 9001

# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 ./launch_pi05_droid.sh
```

### Direct Command

```bash
# Use specific GPU with direct command
CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

## Checking GPU Usage

### Before Starting Server

Check available GPUs:
```bash
nvidia-smi
```

### While Server is Running

Monitor GPU usage in real-time:
```bash
# Watch GPU usage (updates every 2 seconds)
watch -n 2 nvidia-smi

# Or check specific process
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### Expected Memory Usage

| Model | GPU Memory Required |
|-------|-------------------|
| pi0.5-DROID | ~8-10 GB |
| pi0-DROID | ~6-8 GB |
| pi0-FAST-DROID | ~8-10 GB |

## Common Scenarios

### Scenario 1: Server with Multiple GPUs, Use Specific One

```bash
# Check which GPUs are available
nvidia-smi

# Use GPU 3 (least loaded)
CUDA_VISIBLE_DEVICES=3 python launch_pi05_droid.py --host 192.168.1.100
```

### Scenario 2: Avoid Busy GPUs

```bash
# Check GPU usage
nvidia-smi

# GPUs 0,1 are busy, use GPU 2
CUDA_VISIBLE_DEVICES=2 python launch_pi05_droid.py
```

### Scenario 3: Run Multiple Servers on Different GPUs

```bash
# Terminal 1: Pi0.5-DROID on GPU 0, port 8000
CUDA_VISIBLE_DEVICES=0 python launch_pi05_droid.py --port 8000

# Terminal 2: Custom model on GPU 1, port 8001
CUDA_VISIBLE_DEVICES=1 python launch_pi05_droid.py \
    --checkpoint checkpoints/my_model/20000 \
    --port 8001

# Terminal 3: Another model on GPU 2, port 8002
CUDA_VISIBLE_DEVICES=2 python launch_pi05_droid.py \
    --checkpoint gs://openpi-assets/checkpoints/pi0_fast_droid \
    --config pi0_fast_droid \
    --port 8002
```

### Scenario 4: Multi-GPU Inference (Advanced)

For models configured with FSDP (Fully Sharded Data Parallelism):

```bash
# Use 4 GPUs for model parallelism
CUDA_VISIBLE_DEVICES=0,1,2,3 python launch_pi05_droid.py
```

**Note:** Multi-GPU inference requires the model to be configured for FSDP in the training config.

## Troubleshooting

### Problem: "Out of memory" error

**Solution:**
1. Check GPU memory availability:
   ```bash
   nvidia-smi
   ```
2. Use a GPU with more free memory:
   ```bash
   CUDA_VISIBLE_DEVICES=<gpu-with-free-memory> python launch_pi05_droid.py
   ```
3. Try a smaller model (e.g., pi0-DROID instead of pi0.5-DROID)

### Problem: Server not using the GPU I specified

**Solution:**
1. Verify CUDA_VISIBLE_DEVICES is set correctly:
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   ```
2. Check that JAX can see the GPU:
   ```bash
   CUDA_VISIBLE_DEVICES=1 python -c "import jax; print(jax.devices())"
   ```

### Problem: Multiple processes fighting for same GPU

**Solution:**
- Explicitly assign each process to a different GPU using CUDA_VISIBLE_DEVICES
- Monitor GPU usage with `nvidia-smi` before starting new servers

### Problem: Want to use CPU instead of GPU (not recommended for inference)

**Solution:**
```bash
# Force CPU-only mode
CUDA_VISIBLE_DEVICES="" python launch_pi05_droid.py
```
**Warning:** CPU inference will be extremely slow and not suitable for real-time robot control.

## GPU Selection Best Practices

1. **Check availability first:**
   ```bash
   nvidia-smi
   ```

2. **Use least-loaded GPU:**
   - Look at GPU memory usage in nvidia-smi
   - Choose GPU with most free memory

3. **For production:**
   - Reserve specific GPUs for robot control
   - Avoid sharing GPUs with training jobs

4. **Monitor during inference:**
   ```bash
   watch -n 1 nvidia-smi
   ```

5. **Log GPU selection:**
   The server will log which devices it's using when it starts

## Example: Complete Workflow

```bash
# Step 1: Check available GPUs
nvidia-smi

# Output shows:
# GPU 0: 30GB used (training job)
# GPU 1: 30GB used (training job)
# GPU 2: 2GB used (idle)
# GPU 3: 2GB used (idle)

# Step 2: Select GPU 2 for policy server
CUDA_VISIBLE_DEVICES=2 python launch_pi05_droid.py \
    --host 192.168.1.100 \
    --port 9001

# Step 3: Verify server is using GPU 2
nvidia-smi  # Should show policy server on GPU 2 with ~8-10GB usage

# Step 4: Connect DROID robot
# On DROID laptop:
python3 scripts/main.py --remote_host=192.168.1.100 --remote_port=9001 --external_camera=left
```

## JAX-Specific GPU Configuration

For advanced users, you can also use JAX-specific environment variables:

```bash
# Limit JAX to use only 90% of GPU memory
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0 python launch_pi05_droid.py

# Pre-allocate memory (can improve performance)
XLA_PYTHON_CLIENT_PREALLOCATE=true CUDA_VISIBLE_DEVICES=1 python launch_pi05_droid.py

# Combine multiple settings
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
XLA_PYTHON_CLIENT_PREALLOCATE=true \
CUDA_VISIBLE_DEVICES=2 \
python launch_pi05_droid.py --host 192.168.1.100
```

## Related Documentation

- [Launch Scripts README](LAUNCH_SCRIPTS_README.md) - Full launcher options
- [Quick Start Guide](QUICK_START.md) - Getting started
- [Installation Guide](INSTALL_PI05_DROID.md) - Setup instructions
