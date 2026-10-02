from __future__ import annotations

import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from campaigns.dogbone_v1 import s13c_g11
from campaigns.dogbone_v1.s13c_g11 import (
    S13cG11Evidence,
    build_parser,
    lifting_parameter_vectors,
    parameter_box,
    s13c_g11_gate_payload,
    write_gate_artifact,
)
from campaigns.dogbone_v1.s13c_protocol import (
    build_initial_operator,
    load_s13c_normalization,
)
from piginot.contracts import CanonicalQuantity, MaterialSpec, Unit
from piginot.operators import (
    ENERGY_G11_LIFTING_POINTS_PER_EDGE,
    AxisAlignedTensionLifting,
    EnergyFormOperator,
    EnergyG11Measurements,
    QueryFrame,
    evaluate_energy_g11_gate,
)

_DTYPE = torch.float64
_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _material() -> MaterialSpec:
    return MaterialSpec(
        "g11-test",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(760.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.23, Unit.ONE),
        },
    )


@dataclass(frozen=True)
class _Case:
    """Duck-typed stand-in for a geometry case on a 30 x 8 mm rectangle."""

    material: MaterialSpec

    def frame(self) -> QueryFrame:
        return QueryFrame(
            torch.zeros((1, 2), dtype=_DTYPE),
            torch.tensor([[30.0, 8.0]], dtype=_DTYPE),
        )

    def lifting(self) -> AxisAlignedTensionLifting:
        return AxisAlignedTensionLifting(
            torch.tensor([[30.0, 8.0]], dtype=_DTYPE),
            torch.tensor([4.5], dtype=_DTYPE),
        )

    def parameters(self) -> torch.Tensor:
        return torch.tensor([[0.3, -0.2, 0.1, 0.5, -0.4]], dtype=_DTYPE)


def _model(scale: float = 0.05) -> EnergyFormOperator:
    model = build_initial_operator().to(_DTYPE).eval()
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        model.output.weight.copy_(
            scale
            * torch.randn(model.output.weight.shape, generator=generator, dtype=_DTYPE)
        )
    return model


def _query(count: int = 24) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    points = torch.rand((count, 2), generator=generator, dtype=_DTYPE)
    return (points * torch.tensor([30.0, 8.0], dtype=_DTYPE)).unsqueeze(0)


def test_parameter_box_reads_the_committed_envelope() -> None:
    assert parameter_box() == {
        "L_total": (40.0, 70.0),
        "W_grip": (16.0, 26.0),
        "W_gauge": (6.0, 14.0),
        "R_fillet": (8.0, 20.0),
        "nominal_axial_strain": (0.1, 0.2),
    }


def test_lifting_population_has_corners_centre_and_seeded_draws() -> None:
    box = parameter_box()
    vectors = lifting_parameter_vectors(box)

    assert len(vectors) == 64
    corners = {tuple(vector.values()) for vector in vectors[:32]}
    assert len(corners) == 32
    assert vectors[32] == {name: 0.5 * sum(bounds) for name, bounds in box.items()}
    for vector in vectors[33:]:
        for name, value in vector.items():
            assert box[name][0] < value < box[name][1]
    assert lifting_parameter_vectors(box) == vectors


def test_invariance_reachability_and_heads_on_a_probe_case() -> None:
    model = _model(scale=0.002)
    case = _Case(_material())
    query = _query()

    assert s13c_g11._node_order(model, case, query) < 1.0e-12  # type: ignore[arg-type]
    assert s13c_g11._query_batch(model, case, query) < 1.0e-12  # type: ignore[arg-type]
    assert s13c_g11._reachable_parameters(model, case, query) == 5  # type: ignore[arg-type]
    u_shape, p_shape, displacement_only, identity = s13c_g11._heads(
        model,
        case,  # type: ignore[arg-type]
        query,
    )
    assert (u_shape, p_shape, displacement_only) == (True, True, True)
    assert identity < 1.0e-10


def test_lifting_is_exact_and_the_identity_mutant_fires() -> None:
    model = _model()
    normalization = load_s13c_normalization()
    vector = lifting_parameter_vectors(parameter_box())[0]

    for count in ENERGY_G11_LIFTING_POINTS_PER_EDGE[:2]:
        assert (
            s13c_g11._lifting_error(model, vector, normalization, count, mutant=False)
            < 1.0e-12
        )
    assert (
        s13c_g11._lifting_error(model, vector, normalization, 320, mutant=True) > 1.0e-3
    )


def test_pickle_refusal_requires_rejection_before_deserialisation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-step-2400"
    checkpoint.mkdir(parents=True)
    for name in ("model.safetensors", "state.safetensors", "checkpoint.json"):
        (checkpoint / name).write_bytes(b"{}")
    (tmp_path / "run" / "run-binding.json").write_bytes(b"{}")

    def refusing(protocol: object, path: Path) -> None:
        raise ValueError("hash mismatch")

    def unpickling(protocol: object, path: Path) -> None:
        pickle.loads((path / "model.safetensors").read_bytes())  # noqa: S301
        raise ValueError("too late")

    def accepting(protocol: object, path: Path) -> None:
        return None

    monkeypatch.setattr(s13c_g11, "load_registered_checkpoint", refusing)
    assert s13c_g11._pickle_refused(object(), checkpoint) is True
    monkeypatch.setattr(s13c_g11, "load_registered_checkpoint", unpickling)
    assert s13c_g11._pickle_refused(object(), checkpoint) is False
    monkeypatch.setattr(s13c_g11, "load_registered_checkpoint", accepting)
    assert s13c_g11._pickle_refused(object(), checkpoint) is False


def _evidence(**changes: object) -> S13cG11Evidence:
    measurements = EnergyG11Measurements(
        node_order_relative_change=0.0,
        query_batch_relative_change=0.0,
        reachable_parameter_count=5,
        required_parameter_count=5,
        pickle_refused_before_deserialization=True,
        u_shape_verified=True,
        p_shape_verified=True,
        displacement_only_output=True,
        served_stress_energy_gradient_relative_difference=0.0,
        lifting_parameter_vector_count=64,
        lifting_points_per_edge=ENERGY_G11_LIFTING_POINTS_PER_EDGE,
        lifting_max_relative_error=0.0,
        lifting_mutant_control_relative_error=0.0,
        lifting_mutant_relative_error=0.7,
        code_coverage_percent=89.0,
    )
    measurements = replace(measurements, **changes)  # type: ignore[arg-type]
    return S13cG11Evidence(
        architecture_id="pi-ginot-energy-parametric-film-v1",
        module_digest="blake2b256:" + "a" * 64,
        run_hash="1" * 64,
        checkpoint_manifest_hash="2" * 64,
        model_state_hash="3" * 64,
        weights_hash="4" * 64,
        probe_case_id="candidate-000001",
        probe_query_count=969,
        parameter_box=parameter_box(),
        lifting_vector_count=64,
        measurements=measurements,
        evaluation=evaluate_energy_g11_gate(measurements),
    )


def test_gate_payload_records_every_check_and_the_amendment() -> None:
    payload = s13c_g11_gate_payload(
        _evidence(),
        implementation_source_revision=_REVISION,
        branch="codex/s13-closeout",
        timestamp_utc="2026-10-02T06:00:00Z",
        focused_test_result="120 passed; total coverage 89.00%",
    )

    assert payload["verdict"] == "PASS"
    assert payload["amends"] == "artifacts/gates/S11.json"
    checks = payload["checks"]
    assert isinstance(checks, list)
    assert [check["id"] for check in checks][-1] == "G11.coverage"
    assert len(checks) == 8
    failing = s13c_g11_gate_payload(
        _evidence(lifting_max_relative_error=1.0e-9),
        implementation_source_revision=_REVISION,
        branch="codex/s13-closeout",
        timestamp_utc="2026-10-02T06:00:00Z",
        focused_test_result="120 passed",
    )
    assert failing["verdict"] == "FAIL"
    with pytest.raises(ValueError, match="revision"):
        s13c_g11_gate_payload(
            _evidence(),
            implementation_source_revision="HEAD",
            branch="b",
            timestamp_utc="t",
            focused_test_result="r",
        )
    with pytest.raises(TypeError):
        s13c_g11_gate_payload(
            object(),  # type: ignore[arg-type]
            implementation_source_revision=_REVISION,
            branch="b",
            timestamp_utc="t",
            focused_test_result="r",
        )


def test_gate_artifact_is_written_once(tmp_path: Path) -> None:
    target = tmp_path / "gates" / "S11-ENERGY.json"
    write_gate_artifact({"verdict": "PASS"}, target)
    write_gate_artifact({"verdict": "PASS"}, target)

    assert target.read_bytes() == b'{"verdict":"PASS"}'
    with pytest.raises(ValueError, match="refusing"):
        write_gate_artifact({"verdict": "FAIL"}, target)
    with pytest.raises(TypeError):
        write_gate_artifact([], target)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        write_gate_artifact({}, "target")  # type: ignore[arg-type]


def test_parser_requires_every_input() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--s10-run-root", "x"])
