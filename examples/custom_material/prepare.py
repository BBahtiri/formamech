"""Prepare a fixed public material example; never train, solve or predict."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from piginot.compiler.adapters import resolve_adapter
from piginot.compiler.config import CampaignConfig
from piginot.compiler.public_source import select_public_cases
from piginot.contracts import ProblemSpec, canonical_json_bytes, strict_json_loads
from piginot.doe import PublishedCaseManifest

ASSETS = Path(__file__).resolve().parent
INPUTS = ("problem.json", "config-template.json", "source.json")
LIBRARIES = (
    "numpy",
    "rfc8785",
    "safetensors",
    "scikit-fem",
    "scipy",
    "sympy",
    "threadpoolctl",
    "torch",
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _cpu_capability() -> str:
    import torch

    return str(torch.backends.cpu.get_cpu_capability())


def _profile(*, execution_profile: bool) -> dict[str, Any]:
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    missing = [name for name, version in versions.items() if version == "NOT_INSTALLED"]
    if execution_profile and missing:
        raise ValueError(
            "Install piginot[verify] before capturing a profile: " + ", ".join(missing)
        )
    return {
        "python_version": platform.python_version(),
        "library_versions": versions,
        "blas_threads": 1,
        "device": "cpu",
        "dtype": "float32",
        "rng_algorithm": "torch-manual-seed",
        "seed": 2901,
        "container_digest": None,
        "platform_fingerprint": {
            "os": platform.system(),
            "machine": platform.machine(),
            "cpu_capability": _cpu_capability() if execution_profile else "NOT_PROBED",
        },
    }


def _read_inputs(assets: Path) -> dict[str, bytes]:
    raw: dict[str, bytes] = {}
    for name in INPUTS:
        path = assets / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"tutorial requires a regular input file: {name}")
        raw[name] = path.read_bytes()
    return raw


def _configuration(raw: dict[str, bytes], *, execution_profile: bool) -> dict[str, Any]:
    template = CampaignConfig.from_document(
        strict_json_loads(raw["config-template.json"])
    ).document()
    training = _profile(execution_profile=execution_profile)
    config = {
        **template,
        "adapter": {
            "id": "dogbone-v1",
            "implementation_sha256": resolve_adapter(
                "dogbone-v1"
            ).descriptor.implementation_sha256,
        },
        "source": {
            **template["source"],
            "path": "source.json",
            "sha256": _sha(raw["source.json"]),
        },
        "execution": {
            "training": training,
            "reference": {**training, "dtype": "float64"},
            "inference_dtype": "float64",
        },
    }
    campaign = CampaignConfig.from_document(config)
    policy = (
        campaign.training.optimizer_steps,
        campaign.training.seed,
        campaign.training.checkpoint_interval,
        campaign.reference.initialization,
        campaign.reference.refinement_levels,
        campaign.reference.workers,
        campaign.selection.development,
        campaign.source.development_case_ids,
    )
    if policy != (1, 2901, 1, "cold", (1, 2), 2, False, ()):
        raise ValueError(
            "fixed tutorial requires one step, seed 2901, two reference levels and no DEVELOPMENT"
        )
    select_public_cases(
        ProblemSpec.from_bytes(raw["problem.json"]),
        campaign,
        PublishedCaseManifest.from_bytes(raw["source.json"]),
    )
    return config


def prepare(output: Path, *, execution_profile: bool = False) -> dict[str, Any]:
    """Write new, hash-bound inputs; explicit execution remains a separate command."""

    if output.is_symlink():
        raise ValueError("output cannot be a symlink")
    output, assets = output.resolve(), ASSETS.resolve()
    if output.is_relative_to(assets) or assets.is_relative_to(output):
        raise ValueError("output cannot overlap tutorial assets")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    raw = _read_inputs(assets)
    config = _configuration(raw, execution_profile=execution_profile)
    files = {
        "problem.json": raw["problem.json"],
        "source.json": raw["source.json"],
        "config.json": canonical_json_bytes(config) + b"\n",
    }
    record = {
        "schema": "piginot.custom-material-preparation-v1",
        "status": "PREPARED",
        "profile_status": "ENVIRONMENT_CAPTURED_NOT_EXECUTED"
        if execution_profile
        else "METADATA_ONLY",
        "evidence_class": "software-engineering-example",
        "claim_level": "NOT_VERIFIED",
        "template_sha256": {name: _sha(payload) for name, payload in raw.items()},
        "preparer_sha256": _sha(Path(__file__).read_bytes()),
        "input_sha256": {name: _sha(payload) for name, payload in files.items()},
        "profile_sha256": {
            name: _sha(canonical_json_bytes(config["execution"][name]))
            for name in ("training", "reference")
        },
    }
    output.mkdir(parents=True)
    for name, payload in {
        **files,
        "preparation.json": canonical_json_bytes(record) + b"\n",
    }.items():
        with (output / name).open("xb") as stream:
            stream.write(payload)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--execution-profile",
        action="store_true",
        help="Capture actual dependencies/CPU for a later explicit execution; requires piginot[verify].",
    )
    arguments = parser.parse_args(argv)
    try:
        record = prepare(
            arguments.output, execution_profile=arguments.execution_profile
        )
    except (OSError, ValueError, ImportError) as error:
        print(json.dumps({"status": "REFUSED", "reason": str(error)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "PREPARED",
                "output": str(arguments.output.resolve()),
                "profile_status": record["profile_status"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
