from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from campaigns.dogbone_v1 import s13_fem_recovery_reference as reference
from campaigns.dogbone_v1.s11_contingency_source import S10SourceBindings
from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.doe import EvidenceRole
from tests.fem.test_verification_norms import _request, _synthetic_exact_result


def test_p2_reference_reconstruction_preserves_exact_gradient_and_stress() -> None:
    request = _request(element_order=2)
    result = _synthetic_exact_result(request)

    summary = reference.summarize_fem_recovery_reference(request, result, (1,))

    assert summary.cell_count == 2
    assert summary.quadrature_point_count == 14
    assert summary.gradient_max_absolute_difference < 1.0e-14
    assert tuple(metric.population for metric in summary.populations) == (
        "all_cells",
        "root_incident_cells",
        "nonroot_cells",
    )
    assert tuple(metric.cell_count for metric in summary.populations) == (2, 1, 1)
    assert tuple(metric.quadrature_point_count for metric in summary.populations) == (
        14,
        7,
        7,
    )
    assert tuple(metric.integrated_area_mm2 for metric in summary.populations) == (
        pytest.approx(1.0),
        pytest.approx(0.5),
        pytest.approx(0.5),
    )
    assert all(metric.p_relative_l2 < 1.0e-14 for metric in summary.populations)
    assert all(
        metric.p_normalized_component_rms < 1.0e-14 for metric in summary.populations
    )


def test_quadrature_reference_uses_dx_weights_and_four_inplane_components() -> None:
    request = _request(element_order=2)
    request = replace(
        request,
        mesh=replace(
            request.mesh,
            nodes=((0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        ),
    )
    exact = _synthetic_exact_result(request)
    raw = np.asarray(exact.first_piola_stress).copy()
    raw[0, :, 0, 1] = 2.0
    raw[1, :, 0, 1] = 6.0
    # Out-of-plane components do not participate in this operator-target audit.
    raw[..., 2, 2] = 1000.0
    result = replace(exact, first_piola_stress=raw)

    summary = reference.summarize_fem_recovery_reference(request, result, (1,))

    all_cells, root, nonroot = summary.populations
    axial = float(raw[0, 0, 0, 0])
    assert root.p_relative_l2 == pytest.approx(2.0 / math.sqrt(axial**2 + 4.0))
    assert nonroot.p_relative_l2 == pytest.approx(6.0 / math.sqrt(axial**2 + 36.0))
    mean_squared_difference = 44.0 / 3.0
    assert all_cells.p_relative_l2 == pytest.approx(
        math.sqrt(mean_squared_difference / (axial**2 + mean_squared_difference))
    )
    assert root.integrated_area_mm2 == pytest.approx(1.0)
    assert nonroot.integrated_area_mm2 == pytest.approx(0.5)
    assert root.p_normalized_component_rms == pytest.approx(1.0 / 760.0)
    assert nonroot.p_normalized_component_rms == pytest.approx(3.0 / 760.0)
    assert all_cells.p_normalized_component_rms == pytest.approx(
        math.sqrt(11.0 / 3.0) / 760.0
    )


def test_reconstruction_uses_p2_edge_dofs_for_quadratic_displacement() -> None:
    request = _request(element_order=2)
    result = _synthetic_exact_result(request)
    coordinates = np.asarray(result.displacement_dof_coordinates)
    displacement = np.asarray(result.displacement).copy()
    displacement[:, 0] += 0.1 * coordinates[:, 0] * coordinates[:, 1]
    points = np.asarray(result.quadrature_points)
    deformation = np.asarray(result.deformation_gradient).copy()
    deformation[..., 0, 0] += 0.1 * points[..., 1]
    deformation[..., 0, 1] += 0.1 * points[..., 0]
    quadratic_result = replace(
        result, displacement=displacement, deformation_gradient=deformation
    )

    summary = reference.summarize_fem_recovery_reference(
        request, quadratic_result, (1,)
    )

    assert summary.gradient_max_absolute_difference < 1.0e-14


@pytest.mark.parametrize("qoi", ((), (1, 1), (3, 1), (-1,), (4,), (0,)))
def test_qoi_requires_sorted_unique_vertices_and_a_cell_complement(
    qoi: tuple[int, ...],
) -> None:
    request = _request(element_order=2)
    with pytest.raises(ValueError, match="QoI"):
        reference.summarize_fem_recovery_reference(
            request, _synthetic_exact_result(request), qoi
        )


@pytest.mark.parametrize(
    "invalid",
    (
        "request_id",
        "element_order",
        "coordinates",
        "gradient",
        "nonfinite",
        "zero_stress",
    ),
)
def test_reference_rejects_identity_layout_and_field_defects(invalid: str) -> None:
    request = _request(element_order=2)
    result = _synthetic_exact_result(request)
    if invalid == "request_id":
        result = replace(result, request_id="wrong")
    elif invalid == "element_order":
        result = replace(result, element_order=1)
    elif invalid == "coordinates":
        result = replace(
            result,
            displacement_dof_coordinates=tuple(
                reversed(result.displacement_dof_coordinates)
            ),
        )
    elif invalid == "gradient":
        result = replace(result, deformation_gradient=())
    elif invalid == "nonfinite":
        values = np.asarray(result.deformation_gradient).copy()
        values[0, 0, 0, 0] = np.nan
        result = replace(result, deformation_gradient=values)
    else:
        result = replace(
            result, first_piola_stress=np.zeros_like(result.first_piola_stress)
        )
    with pytest.raises(ValueError):
        reference.summarize_fem_recovery_reference(request, result, (1,))


@pytest.mark.parametrize(
    "invalid",
    ("relative", "source_output", "same_paths", "already_exists", "missing_parent"),
)
def test_output_guards_run_before_source_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    s10 = tmp_path / "s10"
    s10.mkdir()
    access, output = tmp_path / "access.json", tmp_path / "reference.json"
    if invalid == "relative":
        output = Path("reference.json")
    elif invalid == "source_output":
        output = s10 / "reference.json"
    elif invalid == "same_paths":
        output = access
    elif invalid == "already_exists":
        access.write_bytes(b"opened")
    else:
        output = tmp_path / "missing" / "reference.json"
    monkeypatch.setattr(
        reference,
        "open_selective_s10_source",
        lambda *args, **kwargs: pytest.fail("source opened before path guards"),
    )

    with pytest.raises((ValueError, FileExistsError)):
        reference.run_s13_fem_recovery_reference(
            s10_run_root=s10, development_access_record=access, output_path=output
        )


def _fake_source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    opened: bool = False,
    role: EvidenceRole = EvidenceRole.TRAIN,
) -> SimpleNamespace:
    request = replace(_request(element_order=2), request_id="candidate-002485")
    source = SimpleNamespace(
        development_opened=opened,
        bindings=S10SourceBindings(
            "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 64, 344, 344
        ),
    )
    verified = SimpleNamespace(
        case=SimpleNamespace(case_id="candidate-002485", role=role),
        request=request,
        result=_synthetic_exact_result(request),
    )
    requested: list[str] = []

    def load(case_id: str) -> tuple[SimpleNamespace, None]:
        requested.append(case_id)
        return verified, None

    source.load_fit_with_collocation = load
    source.requested = requested
    source.verified = verified
    monkeypatch.setattr(
        reference, "open_selective_s10_source", lambda *args, **kwargs: source
    )
    monkeypatch.setattr(reference, "qoi_vertex_indices", lambda *args, **kwargs: (1,))
    monkeypatch.setattr(reference, "_EXPECTED_COUNTS", (4, 2, 1))
    return source


def test_train_runner_binds_exact_existing_fem_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10 = tmp_path / "s10"
    s10.mkdir()
    source = _fake_source(monkeypatch)
    report = reference.run_s13_fem_recovery_reference(
        s10_run_root=s10,
        development_access_record=tmp_path / "access.json",
        output_path=tmp_path / "report.json",
    )

    assert source.requested == ["candidate-002485"]
    assert report.role is EvidenceRole.TRAIN
    assert report.non_gate is True
    assert report.source_bindings == source.bindings
    assert report.source_bindings_hash == source.bindings.content_hash
    assert report.request_content_hash == hash_spec(source.verified.request, "content")
    assert report.result_content_hash == hash_spec(source.verified.result, "content")
    assert report.stress_normalization_mpa == 760.0
    assert report.root_qoi_vertex_indices == (1,)
    assert "not operator vertex error" in report.population_definition
    assert not (tmp_path / "access.json").exists()
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize(
    "invalid", ("opened", "role", "during_load", "counts", "case_id", "during_summary")
)
def test_runner_failclosed_on_role_access_and_population_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    s10 = tmp_path / "s10"
    s10.mkdir()
    source = _fake_source(
        monkeypatch,
        opened=invalid == "opened",
        role=EvidenceRole.DEVELOPMENT if invalid == "role" else EvidenceRole.TRAIN,
    )
    if invalid == "during_load":
        loader = source.load_fit_with_collocation

        def opened_loader(case_id: str) -> tuple[SimpleNamespace, None]:
            source.development_opened = True
            return loader(case_id)

        source.load_fit_with_collocation = opened_loader
    elif invalid == "counts":
        monkeypatch.setattr(reference, "_EXPECTED_COUNTS", (969, 1792, 20))
    elif invalid == "case_id":
        source.verified.case.case_id = "other"
    elif invalid == "during_summary":
        summarize = reference.summarize_fem_recovery_reference

        def opened_summary(*args: object) -> object:
            result = summarize(*args)
            source.development_opened = True
            return result

        monkeypatch.setattr(
            reference, "summarize_fem_recovery_reference", opened_summary
        )
    with pytest.raises((RuntimeError, ValueError)):
        reference.run_s13_fem_recovery_reference(
            s10_run_root=s10,
            development_access_record=tmp_path / "access.json",
            output_path=tmp_path / "report.json",
        )
    assert not (tmp_path / "report.json").exists()


def test_cli_writes_canonical_reference_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s10 = tmp_path / "s10"
    s10.mkdir()
    _fake_source(monkeypatch)
    output = tmp_path / "report.json"
    arguments = (
        "--s10-run-root",
        str(s10),
        "--development-access-record",
        str(tmp_path / "access.json"),
        "--output",
        str(output),
    )

    assert reference.main(arguments) == 0
    report = reference.run_s13_fem_recovery_reference(
        s10_run_root=s10,
        development_access_record=tmp_path / "access.json",
        output_path=tmp_path / "unused.json",
    )
    assert output.read_bytes() == canonical_json_bytes(report) + b"\n"
    with pytest.raises(FileExistsError):
        reference.main(arguments)
