#!/bin/bash
# Helper script to clear OpenPI checkpoint cache

set -e

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get cache directory
CACHE_DIR="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}   OpenPI Cache Cleanup Tool${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "Cache directory: ${YELLOW}$CACHE_DIR${NC}"
echo ""

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS] [CHECKPOINT_NAME]"
    echo ""
    echo "Options:"
    echo "  --all                Clear all cached checkpoints"
    echo "  --list               List all cached checkpoints"
    echo "  --help               Show this help message"
    echo ""
    echo "Checkpoint Names:"
    echo "  pi05_droid          Clear pi0.5 DROID checkpoint"
    echo "  pi0_fast_droid      Clear pi0-FAST DROID checkpoint"
    echo "  pi0_droid           Clear pi0 DROID checkpoint"
    echo "  pi05_libero         Clear pi0.5 LIBERO checkpoint"
    echo "  (or any other checkpoint name)"
    echo ""
    echo "Examples:"
    echo "  $0 --list                    # List all cached checkpoints"
    echo "  $0 pi05_droid                # Clear pi05_droid cache"
    echo "  $0 --all                     # Clear all caches"
    echo ""
    exit 0
}

# Function to list cached checkpoints
list_checkpoints() {
    echo -e "${GREEN}Cached checkpoints:${NC}"
    if [ -d "$CACHE_DIR/openpi-assets/checkpoints" ]; then
        cd "$CACHE_DIR/openpi-assets/checkpoints"
        for dir in */; do
            if [ -d "$dir" ]; then
                name=$(basename "$dir")
                size=$(du -sh "$dir" 2>/dev/null | cut -f1)
                echo -e "  ${YELLOW}$name${NC} - $size"
            fi
        done
    else
        echo "  No cached checkpoints found"
    fi
}

# Function to clear specific checkpoint
clear_checkpoint() {
    local checkpoint_name=$1
    local checkpoint_path="$CACHE_DIR/openpi-assets/checkpoints/$checkpoint_name"
    local lock_file="$checkpoint_path.lock"

    if [ -d "$checkpoint_path" ] || [ -f "$lock_file" ]; then
        echo -e "${YELLOW}Clearing cache for: $checkpoint_name${NC}"

        # Show size before deletion
        if [ -d "$checkpoint_path" ]; then
            size=$(du -sh "$checkpoint_path" 2>/dev/null | cut -f1)
            echo "  Current size: $size"
        fi

        # Delete checkpoint directory
        if [ -d "$checkpoint_path" ]; then
            rm -rf "$checkpoint_path"
            echo -e "  ${GREEN}✓${NC} Removed checkpoint directory"
        fi

        # Delete lock file
        if [ -f "$lock_file" ]; then
            rm -f "$lock_file"
            echo -e "  ${GREEN}✓${NC} Removed lock file"
        fi

        # Delete partial download if exists
        if [ -d "${checkpoint_path}.partial" ]; then
            rm -rf "${checkpoint_path}.partial"
            echo -e "  ${GREEN}✓${NC} Removed partial download"
        fi

        echo -e "${GREEN}Cache cleared successfully!${NC}"
    else
        echo -e "${RED}Checkpoint '$checkpoint_name' not found in cache${NC}"
        exit 1
    fi
}

# Function to clear all checkpoints
clear_all() {
    echo -e "${YELLOW}Clearing ALL cached checkpoints...${NC}"
    echo ""

    if [ ! -d "$CACHE_DIR/openpi-assets/checkpoints" ]; then
        echo "No checkpoints to clear"
        return
    fi

    cd "$CACHE_DIR/openpi-assets/checkpoints"
    for dir in */; do
        if [ -d "$dir" ]; then
            name=$(basename "$dir")
            clear_checkpoint "$name"
            echo ""
        fi
    done

    # Clear any remaining lock files
    rm -f *.lock 2>/dev/null || true

    echo -e "${GREEN}All caches cleared!${NC}"
}

# Parse arguments
if [ $# -eq 0 ]; then
    usage
fi

case "$1" in
    --help|-h)
        usage
        ;;
    --list|-l)
        list_checkpoints
        ;;
    --all|-a)
        read -p "Are you sure you want to clear ALL cached checkpoints? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            clear_all
        else
            echo "Cancelled"
        fi
        ;;
    *)
        clear_checkpoint "$1"
        ;;
esac

echo ""
echo -e "${BLUE}Tip:${NC} Next launch will re-download the checkpoint(s)"
