#!/usr/bin/env python3
"""
Launch script for pi0.5 DROID policy server.

This script provides a convenient wrapper around the serve_policy.py script
with sensible defaults and helpful output.
"""

import argparse
import os
import socket
import subprocess
import sys
from typing import List


def print_banner():
    """Print a nice banner."""
    print("=" * 50)
    print("   Pi0.5 DROID Policy Server Launcher")
    print("=" * 50)
    print()


def get_all_ip_addresses():
    """Get all available IP addresses on the system."""
    import netifaces

    ip_addresses = []
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs:
            for addr_info in addrs[netifaces.AF_INET]:
                ip = addr_info.get('addr')
                if ip and ip != '127.0.0.1':
                    ip_addresses.append((interface, ip))
    return ip_addresses


def get_default_ip():
    """Get the default IP address."""
    hostname = socket.gethostname()
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return "localhost"


def print_config(args: argparse.Namespace):
    """Print the configuration."""
    print("Configuration:")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Port: {args.port}")
    if args.host:
        print(f"  Host: {args.host}")
    if args.gpu is not None:
        print(f"  GPU(s): {args.gpu}")
    print(f"  Record: {args.record}")
    if args.default_prompt:
        print(f"  Default Prompt: {args.default_prompt}")
    print()


def print_network_info(port: int, host: str = None):
    """Print network information."""
    hostname = socket.gethostname()

    # Determine which IP to display
    if host:
        display_ip = host
    else:
        display_ip = get_default_ip()

    print("Network Information:")
    print(f"  Hostname: {hostname}")
    print(f"  Server IP: {display_ip}")
    print(f"  Server URL: ws://{display_ip}:{port}")

    # Try to show all available IPs
    try:
        all_ips = get_all_ip_addresses()
        if len(all_ips) > 1:
            print(f"\n  Available network interfaces:")
            for iface, ip in all_ips:
                marker = " (selected)" if ip == display_ip else ""
                print(f"    {iface}: {ip}{marker}")
    except ImportError:
        # netifaces not available, skip showing all interfaces
        pass
    except Exception:
        # If there's any error getting interfaces, just skip it
        pass

    print()
    print("To connect from DROID robot, use:")
    print(f"  python3 scripts/main.py --remote_host={display_ip} --remote_port={port} --external_camera=left")
    print()


def build_command(args: argparse.Namespace) -> List[str]:
    """Build the command to execute."""
    # Note: --port, --record, and --default-prompt must come BEFORE the policy:checkpoint subcommand
    cmd = [
        "uv", "run", "scripts/serve_policy.py",
        "--port", str(args.port),
    ]

    if args.record:
        cmd.append("--record")

    if args.default_prompt:
        cmd.extend(["--default-prompt", args.default_prompt])

    # Now add the subcommand and its arguments
    cmd.extend([
        "policy:checkpoint",
        f"--policy.config={args.config}",
        f"--policy.dir={args.checkpoint}",
    ])

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Launch the pi0.5 DROID policy server for inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with default settings (pi0.5-DROID checkpoint on port 8000)
  python launch_pi05_droid.py

  # Launch on custom port
  python launch_pi05_droid.py --port 9000

  # Launch with specific GPU
  python launch_pi05_droid.py --gpu 0

  # Launch with multiple GPUs
  python launch_pi05_droid.py --gpu 0,1,2

  # List all available GPUs
  python launch_pi05_droid.py --list-gpus

  # Launch with specific IP address (useful when you have multiple network interfaces)
  python launch_pi05_droid.py --host 192.168.1.100

  # List all available IP addresses
  python launch_pi05_droid.py --list-ips

  # Launch with custom checkpoint
  python launch_pi05_droid.py --checkpoint checkpoints/pi05_droid/my_experiment/20000

  # Launch with recording enabled
  python launch_pi05_droid.py --record

  # Complete example with GPU, IP, and port
  python launch_pi05_droid.py --gpu 2 --host 192.168.1.100 --port 9001
        """
    )

    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Port to serve the policy on (default: 8000)"
    )

    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default="gs://openpi-assets/checkpoints/pi05_droid",
        help="Checkpoint path (default: gs://openpi-assets/checkpoints/pi05_droid)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="pi05_droid",
        help="Config name (default: pi05_droid)"
    )

    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the policy's behavior for debugging"
    )

    parser.add_argument(
        "--prompt",
        dest="default_prompt",
        type=str,
        default="",
        help="Default prompt to use when none is provided"
    )

    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="IP address to display for connections (default: auto-detect). Server always binds to all interfaces (0.0.0.0)"
    )

    parser.add_argument(
        "--list-ips",
        action="store_true",
        help="List all available IP addresses and exit"
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU device(s) to use (e.g., '0', '1', '0,1,2'). If not specified, uses default GPU selection."
    )

    parser.add_argument(
        "--list-gpus",
        action="store_true",
        help="List all available GPUs and exit"
    )

    args = parser.parse_args()

    # Handle --list-ips
    if args.list_ips:
        print("Available network interfaces and IP addresses:")
        try:
            all_ips = get_all_ip_addresses()
            for iface, ip in all_ips:
                print(f"  {iface}: {ip}")
        except ImportError:
            print("  (netifaces package not installed - showing default IP only)")
            print(f"  default: {get_default_ip()}")
        except Exception as e:
            print(f"  Error getting interfaces: {e}")
            print(f"  default: {get_default_ip()}")
        sys.exit(0)

    # Handle --list-gpus
    if args.list_gpus:
        print("Available GPUs:")
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id, name, total, free = parts[0], parts[1], parts[2], parts[3]
                    print(f"  GPU {gpu_id}: {name} ({free} free / {total} total)")
        except FileNotFoundError:
            print("  nvidia-smi not found. Make sure NVIDIA drivers are installed.")
        except subprocess.CalledProcessError as e:
            print(f"  Error running nvidia-smi: {e}")
        except Exception as e:
            print(f"  Error: {e}")
        sys.exit(0)

    # Print information
    print_banner()
    print_config(args)
    print_network_info(args.port, args.host)

    # Build and execute command
    cmd = build_command(args)
    print("Starting policy server...")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Set CUDA_VISIBLE_DEVICES if GPU is specified
    env = os.environ.copy()
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f"Setting CUDA_VISIBLE_DEVICES={args.gpu}")
        print()

    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\nError running server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
