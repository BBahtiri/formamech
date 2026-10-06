"""Smoke-test an installed piginot release (G18.install).

Run with ``python -I scripts/release_smoke.py --expect-version 0.1.0`` from a
directory outside the source tree, after installing the release from an
index. It checks that the package is imported from the installation, not
from a checkout; that its version, console script and claim ceiling are the
released ones; and that a bundle that is not there is refused with a reason
code. It prints one JSON line and, on GitHub Actions, adds it to the job
summary. It exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path


def run(expected_version: str) -> dict[str, object]:
    import piginot
    from piginot.bundle import BundleIntegrityError, load_bundle
    from piginot.verification import ClaimLevel

    location = Path(piginot.__file__).resolve()
    scripts = Path(sys.executable).parent
    script = shutil.which("piginot", path=str(scripts)) or shutil.which("piginot")
    cli = subprocess.run(
        [script or "piginot", "--version"], capture_output=True, text=True
    )
    with tempfile.TemporaryDirectory() as directory:
        try:
            load_bundle(Path(directory) / "missing-bundle")
            refusal: list[str] = []
        except BundleIntegrityError as error:
            refusal = [code.value for code in error.reason_codes]
    checks = {
        "installed_not_source": "site-packages" in location.parts
        or "dist-packages" in location.parts,
        "version": piginot.__version__ == expected_version,
        "console_script": cli.returncode == 0
        and cli.stdout.strip() == f"piginot {expected_version}",
        "claim_ceiling": [level.value for level in ClaimLevel]
        == ["NOT_VERIFIED", "DEVELOPMENT_VERIFIED", "HOLDOUT_VERIFIED"],
        "missing_bundle_refused": bool(refusal),
    }
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "installed_version": piginot.__version__,
        "location": str(location.parent),
        "refusal_codes": refusal,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release_smoke")
    parser.add_argument("--expect-version", required=True)
    arguments = parser.parse_args(argv)
    result = run(arguments.expect_version)
    line = json.dumps(result, sort_keys=True)
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"```json\n{line}\n```\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
