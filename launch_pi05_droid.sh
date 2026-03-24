#!/bin/bash
# Launch script for pi0.5 DROID policy server
# This script provides a convenient wrapper around the serve_policy.py script

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
PORT=8000
CHECKPOINT="gs://openpi-assets/checkpoints/pi05_droid"
CONFIG="pi05_droid"
RECORD=false
DEFAULT_PROMPT=""
HOST=""
LIST_IPS=false
GPU=""
LIST_GPUS=false

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Launch the pi0.5 DROID policy server for inference."
    echo ""
    echo "Options:"
    echo "  -p, --port PORT              Port to serve the policy on (default: 8000)"
    echo "  -c, --checkpoint PATH        Custom checkpoint path (default: gs://openpi-assets/checkpoints/pi05_droid)"
    echo "  --config CONFIG              Config name (default: pi05_droid)"
    echo "  --host IP                    IP address to display for connections (default: auto-detect)"
    echo "  --gpu GPU                    GPU device(s) to use (e.g., '0', '1', '0,1,2')"
    echo "  --list-ips                   List all available IP addresses and exit"
    echo "  --list-gpus                  List all available GPUs and exit"
    echo "  --record                     Record the policy's behavior for debugging"
    echo "  --prompt PROMPT              Default prompt to use when none is provided"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Launch with default settings (pi0.5-DROID checkpoint on port 8000)"
    echo "  $0"
    echo ""
    echo "  # Launch on custom port"
    echo "  $0 --port 9000"
    echo ""
    echo "  # Launch with specific GPU"
    echo "  $0 --gpu 0"
    echo ""
    echo "  # Launch with multiple GPUs"
    echo "  $0 --gpu 0,1,2"
    echo ""
    echo "  # List all available GPUs"
    echo "  $0 --list-gpus"
    echo ""
    echo "  # Launch with specific IP address"
    echo "  $0 --host 192.168.1.100"
    echo ""
    echo "  # List all available IP addresses"
    echo "  $0 --list-ips"
    echo ""
    echo "  # Launch with custom checkpoint"
    echo "  $0 --checkpoint checkpoints/pi05_droid/my_experiment/20000"
    echo ""
    echo "  # Complete example with GPU, IP, and port"
    echo "  $0 --gpu 2 --host 192.168.1.100 --port 9001"
    echo ""
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --list-ips)
            LIST_IPS=true
            shift
            ;;
        --list-gpus)
            LIST_GPUS=true
            shift
            ;;
        --record)
            RECORD=true
            shift
            ;;
        --prompt)
            DEFAULT_PROMPT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Handle --list-ips
if [ "$LIST_IPS" = true ]; then
    echo "Available network interfaces and IP addresses:"
    if command -v ip &> /dev/null; then
        ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | while read -r ip; do
            iface=$(ip -4 addr show | grep -B2 "$ip" | head -n1 | awk '{print $2}' | sed 's/:$//')
            echo "  $iface: $ip"
        done
    else
        hostname -I | tr ' ' '\n' | grep -v '^$' | while read -r ip; do
            echo "  $ip"
        done
    fi
    exit 0
fi

# Handle --list-gpus
if [ "$LIST_GPUS" = true ]; then
    echo "Available GPUs:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null | while IFS=',' read -r gpu_id name total free; do
            # Trim whitespace
            gpu_id=$(echo "$gpu_id" | xargs)
            name=$(echo "$name" | xargs)
            total=$(echo "$total" | xargs)
            free=$(echo "$free" | xargs)
            echo "  GPU $gpu_id: $name ($free free / $total total)"
        done
    else
        echo "  nvidia-smi not found. Make sure NVIDIA drivers are installed."
    fi
    exit 0
fi

# Print banner
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   Pi0.5 DROID Policy Server Launcher${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Display configuration
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Config: ${YELLOW}${CONFIG}${NC}"
echo -e "  Checkpoint: ${YELLOW}${CHECKPOINT}${NC}"
echo -e "  Port: ${YELLOW}${PORT}${NC}"
if [ -n "$HOST" ]; then
    echo -e "  Host: ${YELLOW}${HOST}${NC}"
fi
if [ -n "$GPU" ]; then
    echo -e "  GPU(s): ${YELLOW}${GPU}${NC}"
fi
echo -e "  Record: ${YELLOW}${RECORD}${NC}"
if [ -n "$DEFAULT_PROMPT" ]; then
    echo -e "  Default Prompt: ${YELLOW}${DEFAULT_PROMPT}${NC}"
fi
echo ""

# Get hostname and IP
HOSTNAME=$(hostname)

# Determine which IP to display
if [ -n "$HOST" ]; then
    DISPLAY_IP="$HOST"
else
    DISPLAY_IP=$(hostname -I | awk '{print $1}')
fi

echo -e "${GREEN}Network Information:${NC}"
echo -e "  Hostname: ${YELLOW}${HOSTNAME}${NC}"
echo -e "  Server IP: ${YELLOW}${DISPLAY_IP}${NC}"
echo -e "  Server URL: ${YELLOW}ws://${DISPLAY_IP}:${PORT}${NC}"

# Show all available IPs if there are multiple
ALL_IPS=$(hostname -I)
IP_COUNT=$(echo "$ALL_IPS" | wc -w)
if [ "$IP_COUNT" -gt 1 ]; then
    echo -e "\n  Available network interfaces:"
    for ip in $ALL_IPS; do
        if [ "$ip" = "$DISPLAY_IP" ]; then
            echo -e "    ${YELLOW}${ip}${NC} (selected)"
        else
            echo -e "    ${ip}"
        fi
    done
fi

echo ""
echo -e "${GREEN}To connect from DROID robot, use:${NC}"
echo -e "  ${YELLOW}python3 scripts/main.py --remote_host=${DISPLAY_IP} --remote_port=${PORT} --external_camera=left${NC}"
echo ""

echo -e "${GREEN}Starting policy server...${NC}"
echo ""

# Build the command - note: --port and --record must come BEFORE policy:checkpoint subcommand
CMD="uv run scripts/serve_policy.py --port ${PORT}"

if [ "$RECORD" = true ]; then
    CMD="${CMD} --record"
fi

if [ -n "$DEFAULT_PROMPT" ]; then
    CMD="${CMD} --default-prompt \"${DEFAULT_PROMPT}\""
fi

CMD="${CMD} policy:checkpoint --policy.config ${CONFIG} --policy.dir ${CHECKPOINT}"

# Set CUDA_VISIBLE_DEVICES if GPU is specified
if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo -e "${GREEN}Setting CUDA_VISIBLE_DEVICES=${GPU}${NC}"
    echo ""
fi

# Execute the command
echo -e "${BLUE}Executing: ${CMD}${NC}"
echo ""
eval $CMD
