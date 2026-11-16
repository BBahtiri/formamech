"""Smoke-test an installed FormaMech release without importing a source checkout.

Run with ``python -I scripts/release_smoke.py --expect-version 0.1.0`` from a
directory outside the source tree, after installing the release from an
index. It checks that the package is imported from the installation, not
from a checkout; that its version, console script and claim ceiling are the
released ones; and that a bundle that is not there is refused with a reason
code. ``--base-only`` instead checks package provenance, compiler metadata and
help without importing the numerical stack. Runtime claim/bundle checks are
explicitly not requested in that mode. Neither mode predicts, trains or solves
FEM. One JSON line is printed and added to the GitHub Actions job summary.
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
from dataclasses import asdict
from importlib.metadata import distribution
from pathlib import Path


def _loaded_modules(prefixes: tuple[str, ...]) -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    )


def _runtime_checks() -> tuple[dict[str, bool], list[str]]:
    from formamech.bundle import BundleIntegrityError, load_bundle
    from formamech.verification import ClaimLevel

    with tempfile.TemporaryDirectory() as directory:
        try:
            load_bundle(Path(directory) / "missing-bundle")
            refusal: list[str] = []
        except BundleIntegrityError as error:
            refusal = [code.value for code in error.reason_codes]
    return {
        "claim_ceiling": [level.value for level in ClaimLevel]
        == ["NOT_VERIFIED", "DEVELOPMENT_VERIFIED", "HOLDOUT_VERIFIED"],
        "missing_bundle_refused": bool(refusal),
    }, refusal


def _command(
    script: str | None, *arguments: str
) -> subprocess.CompletedProcess[str] | None:
    if script is None:
        return None
    return subprocess.run(
        [script, *arguments], capture_output=True, text=True, timeout=30, check=False
    )


def run(expected_version: str, *, base_only: bool = False) -> dict[str, object]:
    from formamech.compiler.adapters import resolve_adapter
    from formamech.compiler.provenance import require_clean_implementation_identity

    import formamech
    import piginot
    import piginot_campaigns

    location = Path(formamech.__file__).resolve()
    implementation_location = Path(piginot.__file__).resolve()
    public_location = Path(piginot_campaigns.__file__).resolve()
    installed = distribution("formamech")
    identity = require_clean_implementation_identity()
    expected_entries = {
        "dogbone-v1": "piginot_campaigns.adapters:DOGBONE",
        "notched-v1": "piginot_campaigns.adapters:NOTCHED",
        "cad-fixed-v1": "piginot.cad.adapter:FIXED_CAD",
    }
    entries = sorted(
        (entry.name, entry.value)
        for entry in installed.entry_points
        if entry.group == "piginot.compiler.adapters"
    )
    adapters = {
        name: resolve_adapter(name).descriptor.document()
        for name in ("dogbone-v1", "notched-v1")
    }
    script = shutil.which("formamech", path=str(Path(sys.executable).parent))
    version = _command(script, "--version")
    compiler = _command(script, "compile", "--help")
    checks = {
        "installed_not_source": location
        == Path(str(installed.locate_file("formamech/__init__.py"))).resolve()
        and implementation_location
        == Path(str(installed.locate_file("piginot/__init__.py"))).resolve()
        and public_location
        == Path(str(installed.locate_file("piginot_campaigns/__init__.py"))).resolve(),
        "version": formamech.__version__ == expected_version,
        "metadata_version": installed.version == formamech.__version__,
        "packaged_identity": identity.origin == "packaged",
        "adapter_entry_points": entries == sorted(expected_entries.items()),
        "console_script": version is not None
        and version.returncode == 0
        and version.stdout.strip() == f"formamech {expected_version}",
        "compiler_help": compiler is not None
        and compiler.returncode == 0
        and all(
            flag in compiler.stdout
            for flag in ("--spec", "--config", "--output", "--plan-only", "--resume")
        ),
    }
    refusal: list[str] = []
    not_requested = ["claim_ceiling", "missing_bundle_refused"]
    if not base_only:
        runtime, refusal = _runtime_checks()
        checks.update(runtime)
        not_requested = []
    forbidden = (
        ("torch", "numpy", "scipy", "sympy", "skfem", "piginot.fem", "campaigns")
        if base_only
        else ("skfem", "piginot.fem", "campaigns")
    )
    loaded = _loaded_modules(forbidden)
    checks["no_numerical_imports" if base_only else "no_fem_imports"] = not loaded
    return {
        "scope": "base-metadata" if base_only else "runtime-bundle-refusal",
        "os": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "installed_version": formamech.__version__,
        "location": str(location.parent),
        "implementation_location": str(implementation_location.parent),
        "public_adapter_location": str(public_location.parent),
        "console_script": script,
        "implementation": asdict(identity),
        "adapters": adapters,
        "resource_required_adapters": ["cad-fixed-v1"],
        "not_requested": not_requested,
        "forbidden_modules_loaded": loaded,
        "refusal_codes": refusal,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="release_smoke")
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--base-only", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        result = run(arguments.expect_version, base_only=arguments.base_only)
    except (ImportError, OSError, ValueError, subprocess.SubprocessError) as error:
        result = {
            "passed": False,
            "checks": {"preflight_completed": False},
            "scope": "base-metadata"
            if arguments.base_only
            else "runtime-bundle-refusal",
            "error": f"{type(error).__name__}: {error}",
        }
    line = json.dumps(result, sort_keys=True)
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"```json\n{line}\n```\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
