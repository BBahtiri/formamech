"""Prepare bound CAD physics and campaign inputs; never start numerical work."""

from __future__ import annotations

import hashlib
import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

import piginot
import piginot_campaigns
from piginot.cad.adapter import ADAPTER_ID, implementation_sha256
from piginot.cad.cases import generate_cases
from piginot.cad.resource import GeometryResource, load_geometry_resource
from piginot.cad.settings import build_problem, load_settings
from piginot.compiler._validation import object_fields
from piginot.compiler.adapters import ENERGY_ARCHITECTURE
from piginot.compiler.config import CampaignConfig
from piginot.compiler.public_source import select_public_cases
from piginot.contracts import ProblemSpec, canonical_json_bytes

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
OBJECTIVE_DEFAULTS = {
    "points_per_cell": 3,
    "barrier_weight": 1000.0,
    "barrier_threshold": 0.05,
    "admissibility_floor": 1e-6,
}
MODEL_KEYS = (
    "architecture_id",
    "parameter_names",
    "hidden_dim",
    "latent_token_count",
    "parameter_hidden_dim",
    "num_attention_heads",
    "decoder_layers",
    "positional_frequencies",
    "output_init_std",
)
REFERENCE_DEFAULTS = {
    "initialization": "cold",
    "refinement_levels": [1, 2, 4],
    "workers": 2,
}


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _has_symlink(path: Path) -> bool:
    return any(
        part.is_symlink() for part in (path.absolute(), *path.absolute().parents)
    )


def _output_path(project: Path, settings_path: Path, output: Path) -> Path:
    if _has_symlink(output):
        raise ValueError("output cannot contain a symlink")
    output = output.resolve()
    package = Path(piginot.__file__).resolve().parent
    for protected, label in (
        (project, "project"),
        (settings_path, "settings"),
        (package, "package"),
        (Path(piginot_campaigns.__file__).resolve().parent, "package"),
    ):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(f"output cannot overlap its protected {label}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    return output


def _cpu_capability() -> str:
    import torch

    return str(torch.backends.cpu.get_cpu_capability())


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _profile(seed: int, *, execution_profile: bool) -> dict[str, Any]:
    versions = {name: _version(name) for name in LIBRARIES}
    if execution_profile and "NOT_INSTALLED" in versions.values():
        raise ValueError(
            "Install piginot[verify] before capturing an execution profile"
        )
    return {
        "python_version": platform.python_version(),
        "library_versions": versions,
        "blas_threads": 1,
        "device": "cpu",
        "dtype": "float32",
        "rng_algorithm": "torch-manual-seed",
        "seed": seed,
        "container_digest": None,
        "platform_fingerprint": {
            "os": platform.system(),
            "machine": platform.machine(),
            "cpu_capability": _cpu_capability() if execution_profile else "NOT_PROBED",
        },
    }


def _resource_path(resource: GeometryResource, output: Path) -> str:
    try:
        return os.path.relpath(resource.path, output)
    except ValueError:
        return str(resource.path)


def _config(
    spec: ProblemSpec,
    settings: dict[str, Any],
    resource: GeometryResource,
    output: Path,
    source: bytes,
    fit: tuple[str, ...],
    screen: tuple[str, ...],
    *,
    execution_profile: bool,
) -> CampaignConfig:
    model = object_fields(settings.get("model", {}), "model", (), MODEL_KEYS)
    objective = object_fields(
        settings.get("objective", {}), "objective", (), tuple(OBJECTIVE_DEFAULTS)
    )
    reference = object_fields(
        settings.get("reference", {}), "reference", (), tuple(REFERENCE_DEFAULTS)
    )
    profile = _profile(
        settings["training"]["seed"], execution_profile=execution_profile
    )
    config = CampaignConfig.from_document(
        {
            "schema": "piginot.campaign-config-v1",
            "name": settings["name"],
            "adapter": {
                "id": ADAPTER_ID,
                "implementation_sha256": implementation_sha256(),
                "geometry_resource": {
                    "path": _resource_path(resource, output),
                    "sha256": resource.sha256,
                },
            },
            "source": {
                "kind": "public-manifest",
                "path": "source.json",
                "sha256": _sha(source),
                "fit_case_ids": list(fit),
                "screen_case_ids": list(screen),
                "development_case_ids": [],
            },
            "model": {
                "architecture_id": ENERGY_ARCHITECTURE,
                "parameter_names": [p.name for p in spec.parameters],
                **model,
            },
            "normalization": "train-fit",
            "training": settings["training"],
            "objective": {**OBJECTIVE_DEFAULTS, **objective},
            "reference": {**REFERENCE_DEFAULTS, **reference},
            "execution": {
                "training": profile,
                "reference": {**profile, "dtype": "float64"},
                "inference_dtype": "float64",
            },
            "selection": {"checkpoint_rule": "final", "development": False},
        }
    )
    names = set(config.model.parameter_names)
    varying = {p.name for p in spec.parameters if p.lo < p.hi}
    if config.model.architecture_id != ENERGY_ARCHITECTURE or not varying <= names <= {
        p.name for p in spec.parameters
    }:
        raise ValueError(
            "model must use the supported energy architecture and condition every varying parameter"
        )
    if config.training.case_batch_size > len(fit):
        raise ValueError("training case batch exceeds the TRAIN-fit population")
    if config.reference.initialization != "cold" and spec.material.model != "hgo":
        raise ValueError("the full-load initializer is supported for HGO only")
    return config


def _receipt(
    resource: GeometryResource,
    settings_raw: bytes,
    config: CampaignConfig,
    files: dict[str, bytes],
    *,
    execution_profile: bool,
) -> dict[str, Any]:
    return {
        "schema": "piginot.cad-campaign-preparation-v1",
        "status": "PREPARED",
        "profile_status": "ENVIRONMENT_CAPTURED_NOT_EXECUTED"
        if execution_profile
        else "METADATA_ONLY",
        "claim_level": "NOT_VERIFIED",
        "settings_sha256": _sha(settings_raw),
        "geometry_resource_sha256": resource.sha256,
        "geometry_input_sha256": {
            name: _sha(path.read_bytes()) for name, path in resource.geometry_inputs
        },
        "input_sha256": {name: _sha(payload) for name, payload in files.items()},
        "profile_sha256": {
            name: _sha(canonical_json_bytes(config.document()["execution"][name]))
            for name in ("training", "reference")
        },
        "preparation_implementation_sha256": _sha(
            canonical_json_bytes(
                {
                    name: _sha((Path(__file__).parent / name).read_bytes())
                    for name in ("configuration.py", "settings.py", "cases.py")
                }
            )
        ),
        "development": "NOT_GENERATED_NOT_REQUESTED",
        "holdout": "NOT_GENERATED_NOT_REQUESTED",
        "training_started": False,
        "fem_started": False,
    }


def prepare_campaign(
    project: Path, settings_path: Path, output: Path, *, execution_profile: bool = False
) -> dict[str, Any]:
    """Write one new data-only campaign; execution remains an explicit command."""
    if (
        _has_symlink(project)
        or _has_symlink(settings_path)
        or not settings_path.is_file()
    ):
        raise ValueError("project and settings must be regular inputs without symlinks")
    project, settings_path = project.resolve(), settings_path.resolve()
    output = _output_path(project, settings_path, output)
    raw = settings_path.read_bytes()
    settings = load_settings(raw)
    geometry = project / "geometry.json"
    resource = load_geometry_resource(geometry, _sha(geometry.read_bytes()))
    spec = build_problem(resource, settings)
    sampling = settings["sampling"]
    manifest, fit, screen = generate_cases(
        spec,
        sampling["fit_cases"],
        sampling["screen_cases"],
        settings["training"]["seed"],
    )
    source = manifest.to_bytes()
    config = _config(
        spec,
        settings,
        resource,
        output,
        source,
        fit,
        screen,
        execution_profile=execution_profile,
    )
    select_public_cases(spec, config, manifest)
    files = {
        "problem.json": canonical_json_bytes(spec.document()) + b"\n",
        "config.json": canonical_json_bytes(config.document()) + b"\n",
        "source.json": source,
    }
    record = _receipt(resource, raw, config, files, execution_profile=execution_profile)
    load_geometry_resource(geometry, resource.sha256)
    if settings_path.read_bytes() != raw:
        raise ValueError("settings changed during preparation")
    _output_path(project, settings_path, output)
    output.mkdir(parents=True)
    for name, payload in {
        **files,
        "preparation.json": canonical_json_bytes(record) + b"\n",
    }.items():
        with (output / name).open("xb") as stream:
            stream.write(payload)
    return record
