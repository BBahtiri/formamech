"""Frozen, evaluation-only confirmation protocol for the S1 incumbent."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from experiments.spike.geometry import SpikeMesh, sample_geometry_cases
from experiments.spike.model import MixedPhysicsOperator, ModelConfig

CONFIRMATION_CASES_PER_FAMILY: Final = 25
CONFIRMATION_GEOMETRY_SEED: Final = 2026081301
CONFIRMATION_STATISTICS_SEED: Final = 2026081302
MIN_CONVERGED_PER_FAMILY: Final = 24
MIN_CONVERGED_TOTAL: Final = 48

_FAMILIES: Final = ("dogbone", "centre_notched")
_STRESS_THRESHOLD: Final = 0.15
_PASS_FRACTION_MINIMUM: Final = 0.80
_PASS_WILSON_LOWER_MINIMUM: Final = 0.60
_DISPLACEMENT_THRESHOLD: Final = 0.075
_STRESS_P90_THRESHOLD: Final = 0.20
_PEAK_STRESS_THRESHOLD: Final = 0.60
_MEAN_INFERENCE_SECONDS_THRESHOLD: Final = 0.05
_Z_95: Final = 1.959963984540054
_SHA256_LENGTH: Final = 64

# The compact layout is intentional: this quarantined protocol is kept in one
# module while the project enforces an 800-physical-line ceiling.
# fmt: off

@dataclass(frozen=True)
class FreshCampaign:
    """A fixed fresh campaign and the hash of its canonical manifest."""
    cases: tuple[SpikeMesh, ...]
    manifest: Mapping[str, Any]
    sha256: str
def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _plain_value(value: object, *, context: str) -> Any:
    """Copy a value into the primitive containers accepted by weights_only."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{context} contains a non-finite float")
        return result
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{context} mapping keys must be strings")
        return {
            key: _plain_value(item, context=f"{context}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _plain_value(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"{context} contains unsupported value {type(value).__name__}")
@contextmanager
def _atomic_destination(target: Path) -> Iterator[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
def _atomic_torch_save(target: Path, payload: Mapping[str, object]) -> None:
    with _atomic_destination(target) as temporary:
        torch.save(dict(payload), temporary)
def _atomic_write_json(target: Path, document: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    with _atomic_destination(target) as temporary:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
def _require_sha256(value: str, *, context: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")

def _require_conditioned_provenance(provenance: Mapping[str, object]) -> dict[str, Any]:
    plain = _plain_value(provenance, context="provenance")
    if plain.get("architecture") != "conditioned_mlp":
        raise ValueError("checkpoint provenance must select conditioned_mlp")
    return plain

def save_final_checkpoint(
    target: Path,
    model: MixedPhysicsOperator,
    *,
    provenance: Mapping[str, object],
) -> str:
    """Atomically save the complete incumbent in a weights-only-safe envelope."""
    config = model.spike_config
    if not isinstance(config, ModelConfig):
        raise ValueError("model must expose a ModelConfig as spike_config")
    if config.stress_head_variant != "conditioned_mlp":
        raise ValueError("final checkpoint must use conditioned_mlp")
    plain_provenance = _require_conditioned_provenance(provenance)
    plain_config = _plain_value(asdict(config), context="model_config")
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    if not state or not any(name.startswith("p_head.") for name in state):
        raise ValueError("final checkpoint must contain the direct stress head")
    payload: dict[str, object] = {
        "schema_version": "iteration3-final-checkpoint-v1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "model_config": plain_config,
        "provenance": plain_provenance,
        "model_state_dict": state,
    }
    _atomic_torch_save(Path(target), payload)
    return _sha256_file(Path(target))

def load_final_checkpoint(
    source: Path,
    *,
    expected_sha256: str,
    expected_provenance: Mapping[str, object],
) -> MixedPhysicsOperator:
    """Load a complete incumbent after exact digest and provenance checks."""
    source = Path(source)
    _require_sha256(expected_sha256, context="expected SHA-256")
    actual_sha256 = _sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, received {actual_sha256}"
        )
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    required_keys = {
        "schema_version",
        "role",
        "no_release_evidence",
        "model_config",
        "provenance",
        "model_state_dict",
    }
    if set(payload) != required_keys:
        raise ValueError("checkpoint payload fields do not match the frozen schema")
    if (
        payload["schema_version"] != "iteration3-final-checkpoint-v1"
        or payload["role"] != "LEGACY"
        or payload["no_release_evidence"] is not True
    ):
        raise ValueError("checkpoint metadata does not match the frozen LEGACY schema")
    expected_plain = _require_conditioned_provenance(expected_provenance)
    stored_provenance = _plain_value(payload["provenance"], context="provenance")
    if stored_provenance != expected_plain:
        raise ValueError("checkpoint provenance does not exactly match expectation")
    config_payload = _plain_value(payload["model_config"], context="model_config")
    expected_config_fields = {field.name for field in fields(ModelConfig)}
    if (
        not isinstance(config_payload, dict)
        or set(config_payload) != expected_config_fields
    ):
        raise ValueError("checkpoint ModelConfig fields do not match the current model")
    config = ModelConfig(**config_payload)
    if config.stress_head_variant != "conditioned_mlp":
        raise ValueError("checkpoint must use conditioned_mlp")
    state = payload["model_state_dict"]
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in state.items()
    ):
        raise ValueError("checkpoint model_state_dict must map names to tensors")
    if not any(name.startswith("p_head.") for name in state):
        raise ValueError("checkpoint omits the direct stress head")
    model = MixedPhysicsOperator(config)
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint state is incompatible with ModelConfig") from error
    return model

def _mesh_has_positive_areas(mesh: SpikeMesh) -> bool:
    points = np.asarray(mesh.points, dtype=np.float64)
    cells = np.asarray(mesh.cells)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 3
        or not np.all(np.isfinite(points))
        or cells.ndim != 2
        or cells.shape[1] != 3
        or len(cells) < 1
        or not np.issubdtype(cells.dtype, np.integer)
        or np.any(cells < 0)
        or np.any(cells >= len(points))
    ):
        return False
    triangles = points[cells]
    twice_area = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 1, 1] - triangles[:, 0, 1]) * (
        triangles[:, 2, 0] - triangles[:, 0, 0]
    )
    return bool(np.all(np.isfinite(twice_area)) and np.all(twice_area > 0.0))

def _identity_sets(cases: Sequence[SpikeMesh]) -> dict[str, set[object]]:
    return {
        "geometry": {case.geometry_id for case in cases},
        "mesh": {case.mesh_id for case in cases},
        "seed": {case.seed for case in cases},
        "node": {node for case in cases for node in case.node_ids},
    }

def validate_fresh_campaign(
    cases: Sequence[SpikeMesh],
    *,
    historical_cases: Sequence[SpikeMesh],
) -> dict[str, bool]:
    """Prove the fixed campaign is complete, valid, unique, and fresh."""
    fresh = tuple(cases)
    historical = tuple(historical_cases)
    expected_total = CONFIRMATION_CASES_PER_FAMILY * len(_FAMILIES)
    if len(fresh) != expected_total:
        raise ValueError(f"fresh campaign must contain exactly {expected_total} cases")
    family_counts = {
        family: sum(case.family == family for case in fresh) for family in _FAMILIES
    }
    if family_counts != {
        family: CONFIRMATION_CASES_PER_FAMILY for family in _FAMILIES
    } or any(case.family not in _FAMILIES for case in fresh):
        raise ValueError("fresh campaign must contain exactly 25 cases per family")
    if any(not _mesh_has_positive_areas(case) for case in fresh):
        raise ValueError("every fresh mesh must have finite positive triangle areas")
    fresh_ids = _identity_sets(fresh)
    historical_ids = _identity_sets(historical)
    expected_nodes = sum(len(case.node_ids) for case in fresh)
    uniqueness = {
        "geometry": len(fresh_ids["geometry"]) == len(fresh),
        "mesh": len(fresh_ids["mesh"]) == len(fresh),
        "seed": len(fresh_ids["seed"]) == len(fresh),
        "node": len(fresh_ids["node"]) == expected_nodes,
    }
    for identity, is_unique in uniqueness.items():
        if not is_unique:
            raise ValueError(
                f"fresh {identity} identities must be unique; duplicate found"
            )
    disjoint = {
        identity: fresh_ids[identity].isdisjoint(historical_ids[identity])
        for identity in ("geometry", "mesh", "seed", "node")
    }
    for identity, is_disjoint in disjoint.items():
        if not is_disjoint:
            raise ValueError(
                f"fresh {identity} identities collide with historical data"
            )
    return dict.fromkeys(
        (
            "fresh_geometry_ids_unique",
            "fresh_mesh_ids_unique",
            "fresh_rng_seeds_unique",
            "fresh_node_ids_unique",
            "geometry_ids_disjoint",
            "mesh_ids_disjoint",
            "rng_seeds_disjoint",
            "mesh_qualified_node_ids_disjoint",
        ),
        True,
    )

def _campaign_manifest(
    cases: Sequence[SpikeMesh],
    *,
    master_seed: int,
) -> dict[str, Any]:
    ordered = tuple(cases)
    return {
        "schema_version": "confirm-v1", "role": "LEGACY",
        "no_release_evidence": True, "master_seed": int(master_seed),
        "case_count": len(ordered),
        "family_counts": {
            family: sum(case.family == family for case in ordered)
            for family in _FAMILIES
        },
        "cases": [
            {
                "geometry_id": case.geometry_id, "mesh_id": case.mesh_id,
                "family": case.family, "seed": int(case.seed),
                "parameters": {
                    name: float(value)
                    for name, value in sorted(case.parameters.items())
                },
                "topology_signature": case.topology_signature,
                "point_count": int(len(case.points)), "cell_count": len(case.cells),
            }
            for case in ordered
        ],
    }

def build_fresh_campaign_manifest(
    *,
    historical_cases: Sequence[SpikeMesh],
    master_seed: int = CONFIRMATION_GEOMETRY_SEED,
    nx: int = 32,
    ny: int = 10,
) -> FreshCampaign:
    """Sample the preregistered 50-case campaign without collision resampling."""
    cases = sample_geometry_cases(
        master_seed=master_seed,
        count_per_family=CONFIRMATION_CASES_PER_FAMILY,
        nx=nx,
        ny=ny,
    )
    validate_fresh_campaign(cases, historical_cases=historical_cases)
    manifest = _campaign_manifest(cases, master_seed=master_seed)
    return FreshCampaign(
        cases=cases,
        manifest=manifest,
        sha256=hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest(),
    )

def _finite_metric(record: Mapping[str, object], name: str, *, positive: bool) -> float:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"metric {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"metric {name} must be finite and {qualifier}")
    return number

def _bootstrap_interval(values: np.ndarray, estimate: float) -> list[float]:
    lower, upper = np.quantile(values, (0.025, 0.975))
    return [min(float(lower), estimate), max(float(upper), estimate)]

def _wilson_interval(passed: int, count: int) -> list[float]:
    fraction = passed / count
    denominator = 1.0 + _Z_95**2 / count
    center = (fraction + _Z_95**2 / (2.0 * count)) / denominator
    margin = (
        _Z_95
        * math.sqrt(fraction * (1.0 - fraction) / count + _Z_95**2 / (4.0 * count**2))
        / denominator
    )
    return [
        min(fraction, max(0.0, center - margin)),
        max(fraction, min(1.0, center + margin)),
    ]

def summarize_confirmation_statistics(
    records: Sequence[Mapping[str, object]],
    *,
    seed: int,
    bootstrap_resamples: int,
    pass_threshold: float,
) -> dict[str, Any]:
    """Summarize families with deterministic whole-geometry resampling."""
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if not math.isfinite(pass_threshold) or pass_threshold <= 0.0:
        raise ValueError("pass_threshold must be finite and positive")
    ordered = sorted(records, key=lambda item: str(item.get("geometry_id", "")))
    geometry_ids = [record.get("geometry_id") for record in ordered]
    if any(not isinstance(item, str) or not item for item in geometry_ids):
        raise ValueError("each metric record needs a geometry_id")
    if len(set(geometry_ids)) != len(geometry_ids):
        raise ValueError("metric geometry_ids must be unique")
    if any(record.get("family") not in _FAMILIES for record in ordered):
        raise ValueError("metric records contain an unknown family")
    rng = np.random.default_rng(seed)
    summary: dict[str, Any] = {}
    for family in _FAMILIES:
        family_records = tuple(
            record for record in ordered if record.get("family") == family
        )
        if not family_records:
            raise ValueError(f"no metric records supplied for {family}")
        count = len(family_records)
        arrays = {
            name: np.asarray(
                [
                    _finite_metric(record, name, positive=name.endswith("reference"))
                    for record in family_records
                ],
                dtype=np.float64,
            )
            for name in (
                "P_squared_error",
                "P_squared_reference",
                "u_squared_error",
                "u_squared_reference",
                "relative_l2_P",
                "relative_peak_stress_error",
                "inference_seconds",
            )
        }
        indices = rng.integers(0, count, size=(bootstrap_resamples, count))
        p_estimate = math.sqrt(
            arrays["P_squared_error"].sum() / arrays["P_squared_reference"].sum()
        )
        u_estimate = math.sqrt(
            arrays["u_squared_error"].sum() / arrays["u_squared_reference"].sum()
        )
        p_bootstrap = np.sqrt(
            arrays["P_squared_error"][indices].sum(axis=1)
            / arrays["P_squared_reference"][indices].sum(axis=1)
        )
        u_bootstrap = np.sqrt(
            arrays["u_squared_error"][indices].sum(axis=1)
            / arrays["u_squared_reference"][indices].sum(axis=1)
        )
        peak_values = arrays["relative_peak_stress_error"]
        peak_bootstrap = np.median(peak_values[indices], axis=1)
        per_case_p = arrays["relative_l2_P"]
        passed = sum(value <= pass_threshold for value in per_case_p)
        median_peak = float(np.median(peak_values))
        inference_values = arrays["inference_seconds"]
        summary[family] = {
            "count": count,
            "relative_l2_P": {
                "estimate": p_estimate, "bootstrap_95_ci": _bootstrap_interval(
                    p_bootstrap, p_estimate),
                "per_case_90th_percentile": float(np.quantile(per_case_p, .9)),
            },
            "relative_l2_u": {
                "estimate": u_estimate, "bootstrap_95_ci": _bootstrap_interval(
                    u_bootstrap, u_estimate),
            },
            "P_within_15_percent": {
                "threshold": float(pass_threshold), "passed": passed,
                "count": count, "fraction": passed / count,
                "wilson_95_ci": _wilson_interval(passed, count),
            },
            "relative_peak_stress_error": {
                "median": median_peak, "median_bootstrap_95_ci":
                    _bootstrap_interval(peak_bootstrap, median_peak),
                "worst": float(np.max(peak_values)),
            },
            "inference_seconds": {"total": float(np.sum(inference_values)),
                                  "mean": float(np.mean(inference_values))},
        }
    return summary

def _snapshot_model(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }

def _assert_model_unchanged(
    model: torch.nn.Module,
    before: Mapping[str, torch.Tensor],
) -> None:
    after = model.state_dict()
    if set(after) != set(before) or any(
        not torch.equal(after[name].detach().cpu(), before[name]) for name in before
    ):
        raise ValueError("evaluation changed frozen model parameters or buffers")

def _validate_fem_result(mesh: SpikeMesh, result: object) -> None:
    if getattr(result, "converged", None) is not True:
        return
    displacement = np.asarray(getattr(result, "displacement", None))
    stress = np.asarray(getattr(result, "first_piola", None))
    if displacement.shape != (len(mesh.points), 2) or not np.all(
        np.isfinite(displacement)
    ):
        raise ValueError(
            f"converged FEM displacement is invalid for {mesh.geometry_id}"
        )
    if stress.shape != (len(mesh.points), 2, 2) or not np.all(np.isfinite(stress)):
        raise ValueError(f"converged FEM stress is invalid for {mesh.geometry_id}")

def _normalise_evaluation_records(
    records: Sequence[Mapping[str, object]],
    pairs: Sequence[tuple[SpikeMesh, object]],
) -> tuple[dict[str, Any], ...]:
    expected = {mesh.geometry_id: mesh.family for mesh, _result in pairs}
    normalised = tuple(
        _plain_value(record, context="evaluation record") for record in records
    )
    if not all(isinstance(record, dict) for record in normalised):
        raise ValueError("evaluator must return mapping records")
    ids = [record.get("geometry_id") for record in normalised]
    if (
        len(ids) != len(expected)
        or len(set(ids)) != len(ids)
        or set(ids) != set(expected)
    ):
        raise ValueError(
            "evaluator must return exactly one record per converged geometry"
        )
    for record in normalised:
        geometry_id = record["geometry_id"]
        if record.get("family") != expected[geometry_id]:
            raise ValueError("evaluation record family does not match its geometry")
        for name in (
            "P_squared_error",
            "P_squared_reference",
            "u_squared_error",
            "u_squared_reference",
            "relative_l2_P",
            "relative_l2_u",
            "relative_peak_stress_error",
            "inference_seconds",
        ):
            _finite_metric(
                record,
                name,
                positive=name in {"P_squared_reference", "u_squared_reference"},
            )
    return tuple(sorted(normalised, key=lambda record: record["geometry_id"]))

def _validate_artifacts(
    metadata: Mapping[str, object],
    *,
    artifact_dir: Path,
    expected_geometry_ids: set[str],
) -> dict[str, Any]:
    plain = _plain_value(metadata, context="artifacts")
    if not isinstance(plain, dict):
        raise ValueError("artifact writer must return a mapping")
    expected_sorted = sorted(expected_geometry_ids)
    predicted = plain.get("prediction_geometry_ids")
    plotted = plain.get("plot_geometry_ids")
    if predicted != expected_sorted or plotted != expected_sorted:
        raise ValueError(
            f"artifact ID sets must be exactly complete for {len(expected_sorted)} cases"
        )
    if plain.get("prediction_npz_count") != len(expected_sorted) or plain.get(
        "comparison_plot_svg_count"
    ) != len(expected_sorted):
        raise ValueError(
            f"artifact counts must be exactly complete for {len(expected_sorted)} cases"
        )
    declared_directory = plain.get("directory")
    if not isinstance(declared_directory, str) or Path(
        declared_directory
    ).resolve() != (artifact_dir.resolve()):
        raise ValueError("artifact directory does not match the requested destination")
    prediction_dir = artifact_dir / "predictions"
    plot_dir = artifact_dir / "plots"
    actual_predictions = {path.stem for path in prediction_dir.glob("*.npz")}
    actual_plot_names = {path.name for path in plot_dir.glob("*.svg")}
    expected_plot_names = {
        f"{geometry_id}_fields.svg" for geometry_id in expected_geometry_ids
    }
    if (
        actual_predictions != expected_geometry_ids
        or actual_plot_names != expected_plot_names
    ):
        raise ValueError(
            f"artifact files must be exactly complete for {len(expected_sorted)} cases"
        )
    return plain

def _confirmation_status(statistics: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    family_gates: dict[str, Any] = {}
    all_point_pass = True
    all_upper_pass = True
    all_other_pass = True
    for family in _FAMILIES:
        family_statistics = statistics[family]
        metric = family_statistics["relative_l2_P"]
        passing = statistics[family]["P_within_15_percent"]
        values = {
            "relative_l2_P": float(metric["estimate"]),
            "bootstrap_95_ci_upper": float(metric["bootstrap_95_ci"][1]),
            "P_within_15_fraction": float(passing["fraction"]),
            "P_within_15_wilson_lower": float(passing["wilson_95_ci"][0]),
            "relative_l2_u": float(family_statistics["relative_l2_u"]["estimate"]),
            "relative_l2_P_p90": float(metric["per_case_90th_percentile"]),
            "worst_relative_peak_stress_error": float(
                family_statistics["relative_peak_stress_error"]["worst"]),
            "mean_inference_seconds": float(
                family_statistics["inference_seconds"]["mean"]),
        }
        passes = {
            "point_estimate_pass": values["relative_l2_P"] <= _STRESS_THRESHOLD,
            "bootstrap_upper_pass":
                values["bootstrap_95_ci_upper"] <= _STRESS_THRESHOLD,
            "pass_fraction_gate":
                values["P_within_15_fraction"] >= _PASS_FRACTION_MINIMUM,
            "wilson_lower_gate":
                values["P_within_15_wilson_lower"] >= _PASS_WILSON_LOWER_MINIMUM,
            "displacement_gate": values["relative_l2_u"] <= _DISPLACEMENT_THRESHOLD,
            "stress_p90_gate": values["relative_l2_P_p90"] <= _STRESS_P90_THRESHOLD,
            "peak_stress_gate":
                values["worst_relative_peak_stress_error"] <= _PEAK_STRESS_THRESHOLD,
            "inference_time_gate":
                values["mean_inference_seconds"] <= _MEAN_INFERENCE_SECONDS_THRESHOLD,
        }
        point_pass = passes["point_estimate_pass"]
        upper_pass = passes["bootstrap_upper_pass"]
        other_pass = all(
            value for key, value in passes.items() if "bootstrap" not in key
        )
        all_point_pass = all_point_pass and point_pass
        all_upper_pass = all_upper_pass and upper_pass
        all_other_pass = all_other_pass and other_pass
        family_gates[family] = {
            **values,
            **passes,
            "all_metrics_finite": True,
            "all_non_bootstrap_gates_pass": other_pass,
        }
    if all_other_pass and all_point_pass and all_upper_pass:
        status = "CONFIRMED"
    elif all_other_pass and all_point_pass and not all_upper_pass:
        status = "INCONCLUSIVE"
    else:
        status = "NOT_CONFIRMED"
    return status, {
        "family_relative_l2_P_max": _STRESS_THRESHOLD,
        "family_relative_l2_P_bootstrap_upper_max": _STRESS_THRESHOLD,
        "family_fraction_P_within_15_percent_min": _PASS_FRACTION_MINIMUM,
        "family_wilson_lower_fraction_min": _PASS_WILSON_LOWER_MINIMUM,
        "family_relative_l2_u_max": _DISPLACEMENT_THRESHOLD,
        "family_per_case_P_90th_percentile_max": _STRESS_P90_THRESHOLD,
        "family_worst_peak_error_max": _PEAK_STRESS_THRESHOLD,
        "family_mean_inference_seconds_max": _MEAN_INFERENCE_SECONDS_THRESHOLD,
        "families": family_gates, "all_point_estimates_pass": all_point_pass,
        "all_bootstrap_upper_bounds_pass": all_upper_pass,
        "all_other_gates_pass": all_other_pass, "all_metrics_finite": True,
    }

def run_confirmation(
    *,
    model: MixedPhysicsOperator,
    checkpoint_sha256: str,
    checkpoint_provenance: Mapping[str, object],
    cases: Sequence[SpikeMesh],
    historical_cases: Sequence[SpikeMesh],
    campaign_manifest: Mapping[str, object],
    fem_solver: Callable[[SpikeMesh], object],
    evaluator: Callable[
        [object, Sequence[tuple[SpikeMesh, object]]],
        Sequence[Mapping[str, object]],
    ],
    artifact_writer: Callable[
        [object, Sequence[tuple[SpikeMesh, object]], Path], Mapping[str, object]
    ],
    artifact_dir: Path,
    report_path: Path,
    bootstrap_seed: int = CONFIRMATION_STATISTICS_SEED,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    """Execute a frozen FEM-and-inference campaign with no training path."""
    _require_sha256(checkpoint_sha256, context="checkpoint SHA-256")
    provenance = _require_conditioned_provenance(checkpoint_provenance)
    if model.spike_config.stress_head_variant != "conditioned_mlp":
        raise ValueError("confirmation model must use conditioned_mlp")
    cases = tuple(cases)
    freshness = validate_fresh_campaign(cases, historical_cases=historical_cases)
    manifest = _plain_value(campaign_manifest, context="campaign manifest")
    if not isinstance(manifest, dict):
        raise ValueError("campaign manifest must be a mapping")
    if manifest.get("master_seed") != CONFIRMATION_GEOMETRY_SEED:
        raise ValueError("campaign manifest must use the preregistered geometry seed")
    expected_manifest = _campaign_manifest(
        cases,
        master_seed=CONFIRMATION_GEOMETRY_SEED,
    )
    if manifest != expected_manifest:
        raise ValueError("campaign manifest does not exactly match the fresh cases")
    manifest_sha256 = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    solved = tuple((mesh, fem_solver(mesh)) for mesh in cases)
    for mesh, result in solved:
        _validate_fem_result(mesh, result)
    converged_pairs = tuple(
        (mesh, result)
        for mesh, result in solved
        if getattr(result, "converged", None) is True
    )
    family_campaign: dict[str, dict[str, int]] = {}
    for family in _FAMILIES:
        attempted = sum(mesh.family == family for mesh, _result in solved)
        converged = sum(mesh.family == family for mesh, _result in converged_pairs)
        family_campaign[family] = {
            "attempted": attempted,
            "converged": converged,
            "failed": attempted - converged,
        }
    converged_total = len(converged_pairs)
    if converged_total < MIN_CONVERGED_TOTAL or any(
        family_campaign[family]["converged"] < MIN_CONVERGED_PER_FAMILY
        for family in _FAMILIES
    ):
        raise ValueError(
            "confirmation convergence gate failed: requires at least "
            f"{MIN_CONVERGED_PER_FAMILY} per family and {MIN_CONVERGED_TOTAL} total"
        )
    before = _snapshot_model(model)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            evaluated = evaluator(model, converged_pairs)
            records = _normalise_evaluation_records(evaluated, converged_pairs)
            artifact_metadata = artifact_writer(
                model,
                converged_pairs,
                Path(artifact_dir),
            )
    finally:
        model.train(was_training)
    _assert_model_unchanged(model, before)
    artifacts = _validate_artifacts(
        artifact_metadata,
        artifact_dir=Path(artifact_dir),
        expected_geometry_ids={mesh.geometry_id for mesh, _result in converged_pairs},
    )
    statistics = summarize_confirmation_statistics(
        records,
        seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        pass_threshold=_STRESS_THRESHOLD,
    )
    status, gates = _confirmation_status(statistics)
    campaign_summary = {
        "attempted": len(solved), "converged": converged_total,
        "failed": len(solved) - converged_total, "families": family_campaign,
        "manifest_sha256": manifest_sha256, "manifest": manifest,
        "freshness_proof": freshness,
        "failed_cases": [
            {
                "geometry_id": mesh.geometry_id,
                "family": mesh.family,
                "failure_reason": str(getattr(result, "failure_reason", "unknown")),
            }
            for mesh, result in solved
            if getattr(result, "converged", None) is not True
        ],
    }
    report: dict[str, Any] = {
        "schema_version": "iteration8-confirmation-v1", "role": "LEGACY",
        "no_release_evidence": True, "no_tuning": True,
        "execution_mode": "evaluation_only", "status": status,
        "decision": {
            "status": status,
            "rule": (
                "CONFIRMED only when every preregistered family gate passes; "
                "INCONCLUSIVE only when non-bootstrap gates pass and a stress "
                "bootstrap upper bound exceeds 0.15; otherwise NOT_CONFIRMED"
            ),
        },
        "checkpoint": {"sha256": checkpoint_sha256, "provenance": provenance},
        "protocol": {
            "training_performed": False, "optimizer_steps": 0,
            "backward_calls": 0, "model_state_unchanged": True,
            "geometry_seed": CONFIRMATION_GEOMETRY_SEED,
            "statistics_seed": int(bootstrap_seed),
            "bootstrap_resamples": int(bootstrap_resamples),
            "minimum_converged_per_family": MIN_CONVERGED_PER_FAMILY,
            "minimum_converged_total": MIN_CONVERGED_TOTAL,
        },
        "campaign": campaign_summary, "gates": gates, "statistics": statistics,
        "per_case_metrics": list(records), "artifacts": artifacts,
    }
    report = _plain_value(report, context="confirmation report")
    _atomic_write_json(Path(report_path), report)
    return report
# fmt: on
