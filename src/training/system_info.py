"""System information collection for training data metadata."""

import platform
import socket
from datetime import datetime
from typing import Dict, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_system_info() -> Dict[str, any]:
    """
    Collect system information for training data metadata.

    Returns
    -------
    Dict[str, any]
        Dictionary containing system information including:
        - hostname: Machine hostname
        - cpu_info: CPU information
        - gpu_info: GPU information (if available)
        - memory_info: Memory information
        - platform_info: Platform information
        - timestamp: Current timestamp
    """
    info: Dict[str, any] = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    }

    # CPU information
    if PSUTIL_AVAILABLE:
        info["cpu"] = {
            "count_physical": psutil.cpu_count(logical=False),
            "count_logical": psutil.cpu_count(logical=True),
            "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
        }
        # Try to get CPU model name from platform
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            info["cpu"]["model"] = line.split(":")[1].strip()
                            break
        except (FileNotFoundError, PermissionError):
            pass
    else:
        import os
        info["cpu"] = {
            "count_physical": os.cpu_count() or 0,
            "count_logical": os.cpu_count() or 0,
        }

    # Memory information
    if PSUTIL_AVAILABLE:
        memory = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent,
        }
    else:
        info["memory"] = {"total_gb": None, "available_gb": None}

    # GPU information
    info["gpu"] = None
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_info_list = []
        for i in range(gpu_count):
            gpu_info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                "compute_capability": f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}",
            }
            gpu_info_list.append(gpu_info)
        info["gpu"] = gpu_info_list
    else:
        # Try to detect GPU via nvidia-smi if available
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                gpu_info_list = []
                for i, line in enumerate(result.stdout.strip().split("\n")):
                    if line:
                        parts = line.split(", ")
                        if len(parts) >= 2:
                            gpu_info_list.append({
                                "index": i,
                                "name": parts[0].strip(),
                                "memory_total_gb": parts[1].strip() if len(parts) > 1 else "unknown",
                            })
                if gpu_info_list:
                    info["gpu"] = gpu_info_list
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass

    return info


def format_system_info(info: Dict[str, any]) -> str:
    """
    Format system information as a human-readable string.

    Parameters
    ----------
    info : Dict[str, any]
        System information dictionary from get_system_info().

    Returns
    -------
    str
        Formatted string with system information.
    """
    lines = []
    lines.append(f"Training Data Collection Metadata")
    lines.append(f"{'=' * 50}")
    lines.append(f"Timestamp: {info['timestamp']}")
    lines.append(f"Hostname: {info['hostname']}")
    lines.append("")
    lines.append("Platform:")
    lines.append(f"  System: {info['platform']['system']}")
    lines.append(f"  Release: {info['platform']['release']}")
    lines.append(f"  Machine: {info['platform']['machine']}")
    if info['platform'].get('processor'):
        lines.append(f"  Processor: {info['platform']['processor']}")
    lines.append("")
    lines.append("CPU:")
    lines.append(f"  Physical cores: {info['cpu'].get('count_physical', 'unknown')}")
    lines.append(f"  Logical cores: {info['cpu'].get('count_logical', 'unknown')}")
    if info['cpu'].get('model'):
        lines.append(f"  Model: {info['cpu']['model']}")
    if info['cpu'].get('freq'):
        lines.append(f"  Frequency: {info['cpu']['freq'].get('current', 'unknown')} MHz")
    lines.append("")
    lines.append("Memory:")
    if info['memory'].get('total_gb'):
        lines.append(f"  Total: {info['memory']['total_gb']} GB")
        lines.append(f"  Available: {info['memory']['available_gb']} GB")
        lines.append(f"  Used: {info['memory']['used_gb']} GB ({info['memory']['percent']}%)")
    else:
        lines.append("  Information not available")
    lines.append("")
    if info['gpu']:
        lines.append("GPU:")
        for gpu in info['gpu']:
            lines.append(f"  [{gpu['index']}] {gpu['name']}")
            if gpu.get('memory_total_gb'):
                lines.append(f"      Memory: {gpu['memory_total_gb']} GB")
            if gpu.get('compute_capability'):
                lines.append(f"      Compute Capability: {gpu['compute_capability']}")
    else:
        lines.append("GPU: None detected")
    lines.append("")
    return "\n".join(lines)

