# Pi0.5 DROID Launch Scripts Guide

This directory contains convenient launcher scripts for the pi0.5 DROID policy server.

## Available Scripts

### 1. `launch_pi05_droid.py` (Recommended)

A Python-based launcher with helpful output and configuration display.

**Basic Usage:**
```bash
# Launch with default settings (pi0.5-DROID checkpoint, port 8000)
python launch_pi05_droid.py

# Launch on custom port
python launch_pi05_droid.py --port 9000

# Launch with custom checkpoint
python launch_pi05_droid.py --checkpoint checkpoints/pi05_droid/my_experiment/20000

# Launch with recording enabled (saves inference data for debugging)
python launch_pi05_droid.py --record

# View all options
python launch_pi05_droid.py --help
```

**Output:** The script displays:
- Configuration summary
- Network information (hostname, IP address, server URL)
- Command to connect from DROID robot

### 2. `launch_pi05_droid.sh`

A bash-based launcher with similar functionality.

**Basic Usage:**
```bash
# Make executable (first time only)
chmod +x launch_pi05_droid.sh

# Launch with default settings
./launch_pi05_droid.sh

# Launch on custom port
./launch_pi05_droid.sh --port 9000

# View all options
./launch_pi05_droid.sh --help
```

## Selecting GPU

The launch scripts now include built-in GPU selection, making it easy to choose which GPU(s) to use without manually setting environment variables.

### List Available GPUs

```bash
# Python script
python launch_pi05_droid.py --list-gpus

# Bash script
./launch_pi05_droid.sh --list-gpus
```

**Example output:**
```
Available GPUs:
  GPU 0: NVIDIA RTX 6000 Ada Generation (42113 MiB free / 49140 MiB total)
  GPU 1: NVIDIA RTX 6000 Ada Generation (42115 MiB free / 49140 MiB total)
  GPU 2: NVIDIA RTX 6000 Ada Generation (48424 MiB free / 49140 MiB total)
  GPU 3: NVIDIA RTX 6000 Ada Generation (48424 MiB free / 49140 MiB total)
```

### Launch with Specific GPU

```bash
# Use GPU 2 (has most free memory from above)
python launch_pi05_droid.py --gpu 2

# Use GPU 0 explicitly
./launch_pi05_droid.sh --gpu 0

# Use multiple GPUs (for models configured with FSDP)
python launch_pi05_droid.py --gpu 0,1,2,3
```

### Complete Example with GPU, IP, and Port

```bash
# List GPUs to find the least loaded one
python launch_pi05_droid.py --list-gpus

# Launch on GPU 2 with custom IP and port
python launch_pi05_droid.py --gpu 2 --host 100.79.185.61 --port 9001
```

**Why use `--gpu`?**
- Easily select which GPU to use without environment variables
- Avoid busy GPUs (check with `--list-gpus` first)
- Run multiple policy servers on different GPUs simultaneously
- Better control in multi-GPU systems

## Selecting IP Address for Multiple Network Interfaces

If your server has multiple network interfaces (e.g., ethernet, wifi, VPN), you can specify which IP to use:

### List Available IPs

```bash
# Python script
python launch_pi05_droid.py --list-ips

# Bash script
./launch_pi05_droid.sh --list-ips
```

This will show all available network interfaces and their IP addresses:
```
Available network interfaces and IP addresses:
  eth0: 192.168.1.100
  wlan0: 10.0.0.5
  docker0: 172.17.0.1
```

### Launch with Specific IP

```bash
# Use the ethernet interface
python launch_pi05_droid.py --host 192.168.1.100

# Or use WiFi interface
python launch_pi05_droid.py --host 10.0.0.5
```

**Why use `--host`?**
- You have multiple network interfaces and want to specify which one DROID robot should connect to
- You want to use a specific network (e.g., ethernet over wifi for better latency)
- You're on a VPN and need to specify the correct accessible IP

**Note:** The server always listens on all interfaces (0.0.0.0), so clients can connect via any IP. The `--host` flag only affects which IP is displayed in the connection instructions.

## Common Scenarios

### Scenario 1: Running Pre-trained Pi0.5-DROID Model

```bash
# Start the server
python launch_pi05_droid.py

# On DROID robot (after server starts):
python3 scripts/main.py --remote_host=<SERVER_IP> --remote_port=8000 --external_camera=left
```

Replace `<SERVER_IP>` with the IP address displayed by the launcher.

### Scenario 2: Running Your Fine-tuned Model

```bash
# Start server with your checkpoint
python launch_pi05_droid.py \
    --checkpoint checkpoints/pi05_droid_finetune/my_experiment/20000 \
    --config pi05_droid_finetune

# On DROID robot:
python3 scripts/main.py --remote_host=<SERVER_IP> --remote_port=8000 --external_camera=left
```

### Scenario 3: Running on a Custom Port

```bash
# Start server on port 9000
python launch_pi05_droid.py --port 9000

# On DROID robot:
python3 scripts/main.py --remote_host=<SERVER_IP> --remote_port=9000 --external_camera=left
```

### Scenario 4: Recording Inference Data for Debugging

```bash
# Start server with recording enabled
python launch_pi05_droid.py --record

# Inference data will be saved to policy_records/ directory
```

### Scenario 5: Select Least-Loaded GPU

```bash
# Step 1: Check GPU availability
python launch_pi05_droid.py --list-gpus

# Output shows:
# GPU 0: NVIDIA RTX 6000 (10GB free / 49GB total)  - busy
# GPU 1: NVIDIA RTX 6000 (15GB free / 49GB total)  - busy
# GPU 2: NVIDIA RTX 6000 (48GB free / 49GB total)  - free!
# GPU 3: NVIDIA RTX 6000 (47GB free / 49GB total)  - free!

# Step 2: Launch on least-loaded GPU (GPU 2)
python launch_pi05_droid.py --gpu 2 --host 100.79.185.61 --port 9001

# On DROID robot:
python3 scripts/main.py --remote_host=100.79.185.61 --remote_port=9001 --external_camera=left
```

### Scenario 6: Multiple Network Interfaces (Select Specific IP)

```bash
# First, list available IPs
python launch_pi05_droid.py --list-ips

# Output:
# Available network interfaces and IP addresses:
#   eth0: 192.168.1.100
#   wlan0: 10.0.0.50
#   docker0: 172.17.0.1

# Launch with ethernet IP (more stable for robot control)
python launch_pi05_droid.py --host 192.168.1.100

# On DROID robot (connects via ethernet network):
python3 scripts/main.py --remote_host=192.168.1.100 --remote_port=8000 --external_camera=left
```

## Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-p, --port` | Port to serve the policy on | 8000 |
| `-c, --checkpoint` | Path to checkpoint directory | `gs://openpi-assets/checkpoints/pi05_droid` |
| `--config` | Config name | `pi05_droid` |
| `--host` | IP address to display for connections (useful with multiple network interfaces) | auto-detect |
| `--gpu` | GPU device(s) to use (e.g., '0', '1', '0,1,2') | default (GPU 0) |
| `--list-ips` | List all available IP addresses and exit | - |
| `--list-gpus` | List all available GPUs and exit | - |
| `--record` | Enable recording of inference data | false |
| `--prompt` | Default prompt when none provided | "" |
| `-h, --help` | Show help message | - |

**Notes:**
- The server always binds to all network interfaces (0.0.0.0), but the `--host` option controls which IP address is displayed in the connection instructions.
- GPU selection sets the `CUDA_VISIBLE_DEVICES` environment variable. The script internally uses this to control which GPU(s) the model runs on.

## Checkpoint Paths

### Pre-trained Models (from Google Cloud Storage)

```bash
# Pi0.5-DROID (recommended - best generalist policy)
--checkpoint gs://openpi-assets/checkpoints/pi05_droid
--config pi05_droid

# Pi0-FAST-DROID (autoregressive, good language following)
--checkpoint gs://openpi-assets/checkpoints/pi0_fast_droid
--config pi0_fast_droid

# Pi0-DROID (flow matching, faster inference)
--checkpoint gs://openpi-assets/checkpoints/pi0_droid
--config pi0_droid
```

### Local Fine-tuned Models

```bash
# Your fine-tuned checkpoint (adjust path as needed)
--checkpoint checkpoints/pi05_droid_finetune/my_experiment/20000
--config pi05_droid_finetune
```

## Network Configuration

### Finding Your Server IP

The launcher automatically displays your server's IP address. You can also find it manually:

```bash
# On Linux/Mac
hostname -I | awk '{print $1}'

# Or
ip addr show | grep "inet " | grep -v 127.0.0.1
```

### Testing Server Connectivity

From the DROID robot, verify you can reach the server:

```bash
# Ping the server
ping <SERVER_IP>

# Test websocket connection (if nc/netcat is available)
nc -zv <SERVER_IP> 8000
```

## Troubleshooting

### Issue: "Cannot connect to policy server"

**Solution:**
1. Verify the server is running: Check the launcher output
2. Verify network connectivity: `ping <SERVER_IP>` from DROID robot
3. Check firewall settings: Ensure port 8000 (or your custom port) is open
4. Verify IP address: Make sure you're using the correct server IP

### Issue: "Checkpoint download is slow"

**Solution:**
- First download will cache the checkpoint in `~/.cache/openpi`
- Subsequent launches will be much faster
- For local deployments, consider downloading once and using local path

### Issue: "Out of memory during inference"

**Solution:**
- Ensure your GPU has at least 8GB memory
- Close other GPU-intensive applications
- Try the pi0-DROID model (smaller, faster) instead of pi0.5

## Advanced Usage

### Running Multiple Servers on Different Ports

```bash
# Terminal 1: Pi0.5-DROID on port 8000
python launch_pi05_droid.py --port 8000

# Terminal 2: Your fine-tuned model on port 8001
python launch_pi05_droid.py --port 8001 \
    --checkpoint checkpoints/pi05_droid/my_experiment/20000
```

### Setting Up as a System Service

For production deployments, consider running as a systemd service:

```bash
# Create service file: /etc/systemd/system/pi05-droid.service
[Unit]
Description=Pi0.5 DROID Policy Server
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/openpi
ExecStart=/path/to/openpi/launch_pi05_droid.py --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable pi05-droid
sudo systemctl start pi05-droid
```

## Related Documentation

- [Installation Guide](INSTALL_PI05_DROID.md) - Setup instructions
- [DROID Inference README](examples/droid/README.md) - Running on DROID robot
- [DROID Training README](examples/droid/README_train.md) - Training on DROID dataset
- [Main README](README.md) - Full OpenPI documentation
