"""Amended G11 evidence for the S13c energy-form operator (ADR 0018).

The checks are measured in float64 on the registered S13c checkpoint:

- query order, query batching, parameter reachability and the amended heads
  rule on the first registered TRAIN-screen geometry (all mesh vertices);
- the safetensors refusal on a copy of the checkpoint whose weight files are
  replaced by a pickle payload;
- the exact Dirichlet lifting over the full primitive parameter box (its 32
  corners, its centre and 31 seeded interior draws) at 320, 480 and 4,800
  points per constrained edge, and the ``IGNORE_DIRICHLET_LIFTING`` mutant.

No FEM label, DEVELOPMENT case or HOLDOUT case is read.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import pickle
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import rfc8785
import torch

from campaigns.dogbone_v1.s11_closeout import (
    _clean_source_revision,
    _current_branch,
    _require_revision,
    _require_text,
)
from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
from campaigns.dogbone_v1.s13c_data import (
    DogboneS13cGeometryCase,
    conditioning_vector,
    geometry_case,
)
from campaigns.dogbone_v1.s13c_evaluate import load_registered_checkpoint
from campaigns.dogbone_v1.s13c_protocol import (
    DOGBONE_S13C_CONDITIONING_PARAMETERS,
    load_s13c_normalization,
)
from campaigns.dogbone_v1.s13c_trial import (
    RUN_BINDING_NAME,
    load_committed_s13c_protocol,
)
from campaigns.dogbone_v1.strain_pilot import prescribed_displacement_mm
from piginot.operators import (
    ENERGY_G11_LIFTING_POINTS_PER_EDGE,
    ENERGY_G11_MINIMUM_PARAMETER_VECTORS,
    AffineDirichletLifting,
    AxisAlignedTensionLifting,
    EnergyFormOperator,
    EnergyG11Measurements,
    QueryFrame,
    S11GateEvaluation,
    constitutive_in_plane_stress,
    embed_plane_stress_deformation,
    energy_module_digest,
    evaluate_energy_g11_gate,
)
from piginot.physics import energy_density, plane_stress_closure

_ROOT = Path(__file__).resolve().parents[2]
_ENVELOPE_PATH = (
    _ROOT / "artifacts" / "campaigns" / "dogbone-v1" / ("executable-envelope.json")
)
_INTERIOR_DRAW_SEED = 13_105
_INTERIOR_DRAWS = 31
_MUTANT_POINTS_PER_EDGE = 320
_REACHABILITY_FLOOR = 1.0e-12
_COVERAGE_COMMAND = (
    "pytest -q tests/operators --cov=src/piginot/operators "
    "--cov-report=term-missing --cov-fail-under=80"
)


@dataclass(frozen=True, slots=True)
class S13cG11Evidence:
    """Scalar evidence for the amended G11 gate of the S13c operator."""

    architecture_id: str
    module_digest: str
    run_hash: str
    checkpoint_manifest_hash: str
    model_state_hash: str
    weights_hash: str
    probe_case_id: str
    probe_query_count: int
    parameter_box: dict[str, tuple[float, float]]
    lifting_vector_count: int
    measurements: EnergyG11Measurements
    evaluation: S11GateEvaluation


class _DeserializationMarker:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return self.marker.write_text, ("DESERIALIZED",)


def parameter_box(path: Path = _ENVELOPE_PATH) -> dict[str, tuple[float, float]]:
    """Read the primitive and load bounds of the five conditioning inputs."""

    document = json.loads(path.read_text("utf-8"))
    bounds = [
        *document["geometry_membership"]["primitive_bounds"],
        *document["load_bounds"],
    ]
    by_name = {
        item["name"]: (
            float(item["lower_bound"]["value"]),
            float(item["upper_bound"]["value"]),
        )
        for item in bounds
    }
    box: dict[str, tuple[float, float]] = {}
    for name in DOGBONE_S13C_CONDITIONING_PARAMETERS:
        lower, upper = by_name[name]
        if not lower < upper:
            raise ValueError(f"{name} bound must be a non-empty interval")
        box[name] = (lower, upper)
    return box


def lifting_parameter_vectors(
    box: dict[str, tuple[float, float]],
) -> tuple[dict[str, float], ...]:
    """Return the box corners, its centre and seeded interior draws."""

    names = DOGBONE_S13C_CONDITIONING_PARAMETERS
    corners = [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(box[name] for name in names))
    ]
    centre = {name: 0.5 * (box[name][0] + box[name][1]) for name in names}
    generator = torch.Generator().manual_seed(_INTERIOR_DRAW_SEED)
    draws = torch.rand(
        (_INTERIOR_DRAWS, len(names)), generator=generator, dtype=torch.float64
    )
    interior = [
        {
            name: box[name][0]
            + float(draws[row, column]) * (box[name][1] - box[name][0])
            for column, name in enumerate(names)
        }
        for row in range(_INTERIOR_DRAWS)
    ]
    return (*corners, centre, *interior)


def _relative_change(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(left - right)
    denominator = torch.clamp(torch.linalg.vector_norm(left), min=1.0e-300)
    return float((numerator / denominator).item())


def _served(
    model: EnergyFormOperator,
    case: DogboneS13cGeometryCase,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(
        query_xy=query,
        parameters=case.parameters(),
        frame=case.frame(),
        lifting=case.lifting(),
        materials=(case.material,),
    )
    return output.u, output.P


def _node_order(
    model: EnergyFormOperator, case: DogboneS13cGeometryCase, query: torch.Tensor
) -> float:
    u, stress = _served(model, case, query)
    order = torch.arange(query.shape[1] - 1, -1, -1)
    u_reversed, stress_reversed = _served(model, case, query[:, order])
    return max(
        _relative_change(u, u_reversed[:, order]),
        _relative_change(stress, stress_reversed[:, order]),
    )


def _query_batch(
    model: EnergyFormOperator, case: DogboneS13cGeometryCase, query: torch.Tensor
) -> float:
    u, stress = _served(model, case, query)
    count = query.shape[1]
    sizes = [count // 3, count // 3, count - 2 * (count // 3)]
    pieces = [_served(model, case, part) for part in torch.split(query, sizes, dim=1)]
    return max(
        _relative_change(u, torch.cat([piece[0] for piece in pieces], dim=1)),
        _relative_change(stress, torch.cat([piece[1] for piece in pieces], dim=1)),
    )


def _reachable_parameters(
    model: EnergyFormOperator, case: DogboneS13cGeometryCase, query: torch.Tensor
) -> int:
    parameters = case.parameters().clone().requires_grad_(True)
    u, gradient = model.displacement_and_gradient(
        query_xy=query,
        parameters=parameters,
        frame=case.frame(),
        lifting=case.lifting(),
        create_graph=True,
    )
    stress = constitutive_in_plane_stress(gradient[0], case.material)
    (u_gradient,) = torch.autograd.grad(u.square().sum(), parameters, retain_graph=True)
    (p_gradient,) = torch.autograd.grad(stress.square().sum(), parameters)
    reachable = (torch.abs(u_gradient) > _REACHABILITY_FLOOR) & (
        torch.abs(p_gradient) > _REACHABILITY_FLOOR
    )
    return int(torch.count_nonzero(reachable).item())


def _heads(
    model: EnergyFormOperator, case: DogboneS13cGeometryCase, query: torch.Tensor
) -> tuple[bool, bool, bool, float]:
    u, served = _served(model, case, query)
    _, gradient = model.displacement_and_gradient(
        query_xy=query,
        parameters=case.parameters(),
        frame=case.frame(),
        lifting=case.lifting(),
        create_graph=False,
    )
    leaf = gradient[0].detach().clone().requires_grad_(True)
    deformation = torch.eye(2, dtype=leaf.dtype) + leaf
    closure = plane_stress_closure(deformation, case.material)
    density = energy_density(
        embed_plane_stress_deformation(deformation, closure), case.material
    )
    (energy_stress,) = torch.autograd.grad(density.sum(), leaf)
    count = query.shape[1]
    output_features = model.output.out_features
    return (
        tuple(u.shape) == (1, count, 2),
        tuple(served.shape) == (1, count, 2, 2),
        output_features == 2,
        _relative_change(energy_stress, served[0]),
    )


def _pickle_refused(protocol: Any, checkpoint_path: Path) -> bool:
    for name in ("model.safetensors", "state.safetensors"):
        with tempfile.TemporaryDirectory(prefix="piginot-s13c-g11-") as directory:
            root = Path(directory)
            copied = root / checkpoint_path.name
            shutil.copytree(checkpoint_path, copied)
            shutil.copy2(
                checkpoint_path.parent / RUN_BINDING_NAME, root / RUN_BINDING_NAME
            )
            marker = root / "executed.txt"
            (copied / name).write_bytes(pickle.dumps(_DeserializationMarker(marker)))
            try:
                load_registered_checkpoint(protocol, copied)
            except ValueError:
                if marker.exists():
                    return False
                continue
            return False
    return True


def _edge_points(
    length: float, grip_width: float, gauge_width: float, count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = torch.stack(
        (
            torch.zeros(count, dtype=torch.float64),
            torch.linspace(0.0, 0.5 * gauge_width, count, dtype=torch.float64),
        ),
        dim=-1,
    )
    bottom = torch.stack(
        (
            torch.linspace(0.0, 0.5 * length, count, dtype=torch.float64),
            torch.zeros(count, dtype=torch.float64),
        ),
        dim=-1,
    )
    loaded = torch.stack(
        (
            torch.full((count,), 0.5 * length, dtype=torch.float64),
            torch.linspace(0.0, 0.5 * grip_width, count, dtype=torch.float64),
        ),
        dim=-1,
    )
    return left, bottom, loaded


def _lifting_error(
    model: EnergyFormOperator,
    primitives: dict[str, float],
    normalization: Any,
    count: int,
    *,
    mutant: bool,
) -> float:
    length = primitives["L_total"]
    grip_width = primitives["W_grip"]
    displacement = prescribed_displacement_mm(
        length, primitives["nominal_axial_strain"]
    )
    extent = torch.tensor([[0.5 * length, 0.5 * grip_width]], dtype=torch.float64)
    frame = QueryFrame(torch.zeros((1, 2), dtype=torch.float64), extent)
    lifting = AxisAlignedTensionLifting(
        extent, torch.tensor([displacement], dtype=torch.float64)
    )

    def identity_lifting(query_xy: torch.Tensor) -> AffineDirichletLifting:
        return AffineDirichletLifting(
            torch.zeros_like(query_xy), torch.ones_like(query_xy)
        )

    left, bottom, loaded = _edge_points(
        length, grip_width, primitives["W_gauge"], count
    )
    query = torch.cat((left, bottom, loaded)).unsqueeze(0)
    parameters = conditioning_vector(primitives, normalization).unsqueeze(0)
    with torch.no_grad():
        u = model.displacement(
            query_xy=query,
            parameters=parameters,
            frame=frame,
            lifting=identity_lifting if mutant else lifting,
        )[0]
    errors = torch.cat(
        (
            torch.abs(u[:count, 0]),
            torch.abs(u[count : 2 * count, 1]),
            torch.abs(u[2 * count :, 0] - displacement),
        )
    )
    return float(errors.max().item()) / displacement


def run_s13c_g11_evidence(
    *,
    s10_run_root: Path,
    checkpoint_path: Path,
    code_coverage_percent: float | None,
) -> S13cG11Evidence:
    """Measure every amended G11 check on the registered S13c checkpoint."""

    protocol = load_committed_s13c_protocol()
    model, binding, manifest = load_registered_checkpoint(protocol, checkpoint_path)
    evaluation_model = copy.deepcopy(model).to(torch.float64).eval()
    normalization = load_s13c_normalization()
    with tempfile.TemporaryDirectory(prefix="piginot-s13c-g11-") as directory:
        access_record = Path(directory) / "development-access.json"
        source = open_selective_s10_source(
            s10_run_root, development_access_record=access_record
        )
        probe_id = protocol.screen_case_ids[0]
        (verified,) = tuple(source.iter_train_geometry((probe_id,)))
        if source.development_opened or access_record.exists():
            raise RuntimeError("DEVELOPMENT labels must remain closed during G11")
    case = geometry_case(verified, normalization, dtype=torch.float64)
    query = case.domain.vertices.unsqueeze(0)

    node_order = _node_order(evaluation_model, case, query)
    query_batch = _query_batch(evaluation_model, case, query)
    reachable = _reachable_parameters(evaluation_model, case, query)
    u_shape, p_shape, displacement_only, stress_identity = _heads(
        evaluation_model, case, query
    )
    pickle_refused = _pickle_refused(protocol, checkpoint_path)

    box = parameter_box()
    vectors = lifting_parameter_vectors(box)
    lifting_error = max(
        _lifting_error(evaluation_model, vector, normalization, count, mutant=False)
        for vector in vectors
        for count in ENERGY_G11_LIFTING_POINTS_PER_EDGE
    )
    control = max(
        _lifting_error(
            evaluation_model,
            vector,
            normalization,
            _MUTANT_POINTS_PER_EDGE,
            mutant=False,
        )
        for vector in vectors
    )
    mutant = max(
        _lifting_error(
            evaluation_model,
            vector,
            normalization,
            _MUTANT_POINTS_PER_EDGE,
            mutant=True,
        )
        for vector in vectors
    )
    measurements = EnergyG11Measurements(
        node_order_relative_change=node_order,
        query_batch_relative_change=query_batch,
        reachable_parameter_count=reachable,
        required_parameter_count=len(DOGBONE_S13C_CONDITIONING_PARAMETERS),
        pickle_refused_before_deserialization=pickle_refused,
        u_shape_verified=u_shape,
        p_shape_verified=p_shape,
        displacement_only_output=displacement_only,
        served_stress_energy_gradient_relative_difference=stress_identity,
        lifting_parameter_vector_count=len(vectors),
        lifting_points_per_edge=ENERGY_G11_LIFTING_POINTS_PER_EDGE,
        lifting_max_relative_error=lifting_error,
        lifting_mutant_control_relative_error=control,
        lifting_mutant_relative_error=mutant,
        code_coverage_percent=code_coverage_percent,
    )
    if len(vectors) < ENERGY_G11_MINIMUM_PARAMETER_VECTORS:
        raise RuntimeError("the lifting population is smaller than ADR 0018 fixes")
    return S13cG11Evidence(
        architecture_id=evaluation_model.architecture_id,
        module_digest=energy_module_digest(model),
        run_hash=binding.run_hash,
        checkpoint_manifest_hash=manifest.manifest_hash,
        model_state_hash=manifest.model_state_hash,
        weights_hash=manifest.model_export_hash,
        probe_case_id=probe_id,
        probe_query_count=int(query.shape[1]),
        parameter_box=box,
        lifting_vector_count=len(vectors),
        measurements=measurements,
        evaluation=evaluate_energy_g11_gate(measurements),
    )


def s13c_g11_gate_payload(
    evidence: S13cG11Evidence,
    *,
    implementation_source_revision: str,
    branch: str,
    timestamp_utc: str,
    focused_test_result: str,
) -> dict[str, object]:
    """Build the public amended-G11 gate payload."""

    if type(evidence) is not S13cG11Evidence:
        raise TypeError("evidence must be exact S13cG11Evidence")
    _require_revision(implementation_source_revision, "implementation_source_revision")
    _require_text(branch, "branch")
    _require_text(timestamp_utc, "timestamp_utc")
    _require_text(focused_test_result, "focused_test_result")
    measurements = asdict(evidence.measurements)
    measurements["lifting_points_per_edge"] = list(
        evidence.measurements.lifting_points_per_edge
    )
    checks = [
        {
            "id": check.check_id,
            "status": check.status.value,
            "measured": check.measured,
            "threshold": check.threshold,
            "comparison": check.comparison,
            "reason": check.reason,
        }
        for check in evidence.evaluation.checks
    ]
    passed = evidence.evaluation.verdict.value == "PASS"
    return {
        "schema_version": "1.0",
        "step": "S11",
        "gate_id": "G11",
        "amends": "artifacts/gates/S11.json",
        "decision_record": "docs/adr/0018-s13c-energy-form-operator.md",
        "timestamp_utc": timestamp_utc,
        "branch": branch,
        "implementation_source_revision": implementation_source_revision,
        "scope": {
            "claim": (
                "The S13c energy-form operator satisfies the eight required "
                "G11 checks fixed by ADR 0018. This is structural evidence, not "
                "an accuracy claim."
            ),
            "served_stress_source": (
                "P_served = P_NH(I + grad u_pred) through the S4 plane-stress "
                "kernel; there is no independent stress head"
            ),
            "dirichlet_policy": (
                "exact affine lifting, verified over the full primitive "
                "parameter box at three boundary refinements"
            ),
            "boundary_cloud_checks": (
                "G11.permutation, G11.resampling and G11.mask do not apply: the "
                "operator receives no boundary point cloud. ADR 0018 replaces "
                "them with G11.lifting and G11.lifting-mutant."
            ),
            "fem_labels_read": False,
            "development_or_holdout_read": False,
        },
        "architecture": {
            "architecture_id": evidence.architecture_id,
            "module_digest": evidence.module_digest,
            "conditioning_parameters": list(DOGBONE_S13C_CONDITIONING_PARAMETERS),
            "output_heads": {
                "u": {"shape": "(batch, query_points, 2)"},
                "P": {
                    "shape": "(batch, query_points, 2, 2)",
                    "source": "constitutive",
                },
            },
        },
        "measured_checkpoint": {
            "run_hash": evidence.run_hash,
            "checkpoint_manifest_hash": evidence.checkpoint_manifest_hash,
            "model_state_hash": evidence.model_state_hash,
            "weights_hash": evidence.weights_hash,
            "evaluation_dtype": "float64",
        },
        "probe": {
            "train_screen_case_id": evidence.probe_case_id,
            "query_points": evidence.probe_query_count,
            "parameter_box": {
                name: list(bounds) for name, bounds in evidence.parameter_box.items()
            },
            "lifting_parameter_vectors": evidence.lifting_vector_count,
            "interior_draw_seed": _INTERIOR_DRAW_SEED,
        },
        "measurements": measurements,
        "implementation_evidence": {
            "focused_coverage_command": _COVERAGE_COMMAND,
            "focused_test_result": focused_test_result,
            "scoped_coverage_threshold_percent": 80.0,
        },
        "checks": checks,
        "verdict": evidence.evaluation.verdict.value,
        "verdict_reason": (
            "all eight required amended G11 checks passed"
            if passed
            else "at least one required amended G11 check did not pass"
        ),
        "limitations": [
            "Structural evidence on one registered checkpoint; no accuracy claim.",
            "Box corners may violate coupled geometric bounds; the lifting is "
            "exact on the constrained lines regardless of the geometry.",
            "The operator is conditioned on the family parameters; it does not "
            "encode an arbitrary boundary point cloud.",
        ],
    }


def write_gate_artifact(payload: dict[str, object], target: Path) -> None:
    """Create a canonical gate artifact once; refuse divergent replacement."""

    if type(payload) is not dict:
        raise TypeError("payload must be a JSON object")
    if not isinstance(target, Path):
        raise TypeError("target must be a pathlib.Path")
    encoded = rfc8785.dumps(cast(Any, payload))
    if target.exists():
        if target.is_symlink() or target.read_bytes() != encoded:
            raise ValueError(f"refusing to replace different gate evidence: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(encoded)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--coverage-percent", required=True, type=float)
    parser.add_argument("--focused-test-result", required=True)
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence = run_s13c_g11_evidence(
        s10_run_root=args.s10_run_root,
        checkpoint_path=args.checkpoint_path,
        code_coverage_percent=args.coverage_percent,
    )
    payload = s13c_g11_gate_payload(
        evidence,
        implementation_source_revision=_clean_source_revision(),
        branch=_current_branch(),
        timestamp_utc=_timestamp(),
        focused_test_result=args.focused_test_result,
    )
    write_gate_artifact(payload, args.output)
    print(json.dumps(payload["checks"], indent=1), flush=True)
    print(f"verdict: {payload['verdict']}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "S13cG11Evidence",
    "build_parser",
    "lifting_parameter_vectors",
    "main",
    "parameter_box",
    "run_s13c_g11_evidence",
    "s13c_g11_gate_payload",
    "write_gate_artifact",
)
