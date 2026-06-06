"""
MecanTensor HAL - Hardware Abstraction Layer (Python Wrapper)

Wraps the C++ HAL discovery engine and the pure-Python OpenCL discovery
module. Provides a unified interface to probe all hardware on the system.

Usage:
    from mecantensor import hal
    devices = hal.discover()
    hal.print_devices()
"""

import os
import sys

# ─── Locate discovery.py robustly ───
# It lives at: <project_root>/src/hal/discovery.py
# We search multiple candidate paths to handle both editable and installed modes.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_search_paths = [
    os.path.abspath(os.path.join(_this_dir, "..", "src", "hal")),       # editable: mecantensor/src/hal/
    os.path.abspath(os.path.join(_this_dir, "..", "..", "src", "hal")), # if nested deeper
]

for _p in _search_paths:
    if os.path.isfile(os.path.join(_p, "discovery.py")) and _p not in sys.path:
        sys.path.insert(0, _p)
        break

try:
    from discovery import (
        ComputeDevice,
        UniversalDiscovery,
    )
    _HAS_DISCOVERY = True
except ImportError:
    _HAS_DISCOVERY = False
    ComputeDevice = None
    UniversalDiscovery = None


def discover():
    """
    Discover all compute devices on this machine.
    Returns a list of ComputeDevice objects.
    """
    if not _HAS_DISCOVERY:
        print("[HAL] Warning: discovery.py not found. Returning empty device list.")
        return []

    engine = UniversalDiscovery()
    return engine.discover_all()


def print_devices():
    """Print all discovered hardware devices."""
    devices = discover()
    if not devices:
        print("[HAL] No devices discovered.")
        return

    for i, dev in enumerate(devices):
        print(f"[{i}] {dev}")


def get_best_gpu():
    """Return the highest-VRAM GPU device, or None."""
    devices = discover()
    gpus = [d for d in devices if getattr(d, 'type', '') == 'GPU']
    if not gpus:
        return None
    return max(gpus, key=lambda d: getattr(d, 'vram_bytes', 0))


def get_cpu():
    """Return the CPU device."""
    devices = discover()
    cpus = [d for d in devices if getattr(d, 'type', '') == 'CPU']
    return cpus[0] if cpus else None
