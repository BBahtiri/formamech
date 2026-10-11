"""What the running platform supports, for tests that depend on it (S20, ADR 0100).

Each helper names one platform capability. A test that needs a capability the
platform lacks is skipped with the reason given here, never silently passed;
the Linux CI and reference profiles have every capability, so nothing is
skipped there.
"""

from __future__ import annotations

import functools
import os
import sys
import tempfile
from pathlib import Path

import pytest

SYMLINK_REASON = (
    "the OS refuses to create symbolic links (Windows without Developer Mode "
    "or an administrator shell, WinError 1314)"
)
POSIX_MODES_REASON = (
    "the OS has no POSIX permission bits (Windows reports 0o666); the "
    "owner-only file mode is not enforceable here"
)
PINNED_WEIGHTS_REASON = (
    "the pinned S13c initial weights come from Linux torch on AVX2/AVX512; "
    "Windows torch+cpu initialisation draws different bits (ADR 0100 findings)"
)


@functools.cache
def symlinks_available() -> bool:
    """Whether this process may create a symbolic link."""

    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "target"
        target.write_bytes(b"x")
        try:
            os.symlink(target, Path(scratch) / "link")
        except OSError:
            return False
    return True


def posix_modes_supported() -> bool:
    """Whether ``chmod`` permission bits are meaningful on this platform."""

    return os.name == "posix"


def pinned_initial_weights_supported() -> bool:
    """Whether the S13c initial-weights pin can reproduce on this platform."""

    return sys.platform != "win32"


requires_symlinks = pytest.mark.skipif(not symlinks_available(), reason=SYMLINK_REASON)
requires_posix_modes = pytest.mark.skipif(
    not posix_modes_supported(), reason=POSIX_MODES_REASON
)
requires_pinned_initial_weights = pytest.mark.skipif(
    not pinned_initial_weights_supported(), reason=PINNED_WEIGHTS_REASON
)


def windows_safe_environment(environment: dict[str, str]) -> dict[str, str]:
    """A minimal subprocess environment plus what Windows needs to start Python.

    Without ``SYSTEMROOT`` Windows cannot initialise its socket provider, so
    ``import torch`` fails in the child with WinError 10106.
    """

    if os.name != "nt":
        return environment
    keep = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP")
    return {**{k: os.environ[k] for k in keep if k in os.environ}, **environment}
