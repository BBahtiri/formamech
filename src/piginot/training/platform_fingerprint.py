"""The platform part of an execution profile (run binding v2, S20, ADR 0101).

Trained weights depend on the operating system, the machine and the CPU
instruction set torch dispatches to (AVX2 and AVX512 round differently), so
runs started from S20 on bind all three into their ``run_hash``.
"""

from __future__ import annotations

import platform

import torch


def platform_fingerprint() -> dict[str, str]:
    """``os``, ``machine`` and ``cpu_capability`` of the running process."""

    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "cpu_capability": str(torch.backends.cpu.get_cpu_capability()),
    }
