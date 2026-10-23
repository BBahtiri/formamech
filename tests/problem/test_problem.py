"""What a ProblemSpec builds: FEM requests, frames, liftings, evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest
import torch

from piginot.contracts import (
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    ProblemSpec,
    ProblemSpecError,
    canonical_json_bytes,
)
from piginot.contracts.units import Dimension, Unit
from piginot.lifting import AdfLifting, AxisAlignedTensionLifting
from piginot.problem import (
    adf_pieces,
    box_points,
    build_lifting,
    fem_request,
    query_frame,
    verify_lifting,
)
from piginot.problem.verification import LiftingEvidence
from tests.problem_helpers import plate_document, plate_mesh

PROFILE = ExecutionProfile(
    python_version="3.11.15",
    library_versions={"numpy": "2.2.6", "torch": "2.13.0"},
    device="cpu",
    dtype="float64",
    blas_threads=1,
    seed=0,
    rng_algorithm="none",
)


def spec(lifting: str = "quarter", **changes: Any) -> ProblemSpec:
    document = plate_document(lifting)
    document.update(changes)
    return ProblemSpec.from_document(document)


def inputs(
    width: float = 16.0, height: float = 6.0, strain: float = 0.08
) -> dict[str, Any]:
    return {
        "width": CanonicalQuantity(width, Unit.MM, Dimension.LENGTH),
        "height": CanonicalQuantity(height, Unit.MM, Dimension.LENGTH),
        "strain": CanonicalQuantity(strain, Unit.ONE, Dimension.DIMENSIONLESS),
        "modulus": CanonicalQuantity(760.0, Unit.MPA, Dimension.STRESS),
        "poisson": CanonicalQuantity(0.3, Unit.ONE, Dimension.DIMENSIONLESS),
    }


# ----------------------------------------------------------------------- FEM


def test_the_fem_request_follows_the_spec() -> None:
    request = fem_request(spec(), inputs(), plate_mesh(), "case-1", PROFILE)
    assert request.request_id == "case-1"
    assert request.material.name == "plate-material"
    assert request.material.model == "finite_strain_plane_stress_neo_hookean"
    assert request.material.properties["elastic_modulus"].value == 760.0
    names = [c.name for c in request.boundary_conditions]
    assert names == ["left_x", "bottom_y", "pull"]
    pull = request.boundary_conditions[2]
    assert pull.boundary_tag == "right" and pull.kind == "dirichlet"
    assert pull.values[0].value == 0.08 * 16.0
    assert request.config.element_order == 2 and request.config.load_steps == 8
    assert request.loads == () and request.qois == ()
    again = fem_request(spec(), inputs(), plate_mesh(), "case-1", PROFILE)
    assert canonical_json_bytes(again) == canonical_json_bytes(request)


def test_an_expression_value_is_evaluated() -> None:
    document = plate_document()
    document["boundary_conditions"][2]["values"] = [
        {"expression": "0.5 * strain * width + unit_length", "unit": "mm"}
    ]
    request = fem_request(
        ProblemSpec.from_document(document), inputs(), plate_mesh(), "c", PROFILE
    )
    assert request.boundary_conditions[2].values[0].value == 0.5 * 0.08 * 16.0 + 1.0


def test_traction_loads_and_missing_regions_are_refused() -> None:
    document = plate_document()
    document["boundary_conditions"].append(
        {
            "name": "push",
            "region": "top",
            "kind": "neumann",
            "components": ["y"],
            "values": [{"expression": "1", "unit": "MPa"}],
        }
    )
    with pytest.raises(ProblemSpecError) as error:
        fem_request(
            ProblemSpec.from_document(document), inputs(), plate_mesh(), "c", PROFILE
        )
    assert error.value.rule == "unsupported"
    mesh = plate_mesh()
    tags = {k: v for k, v in mesh.semantic_tags.items() if k != "right"}
    other = CanonicalMesh(**{**_mesh_fields(mesh), "semantic_tags": tags})
    with pytest.raises(ProblemSpecError) as error:
        fem_request(spec(), inputs(), other, "c", PROFILE)
    assert error.value.rule == "region"


def _mesh_fields(mesh: CanonicalMesh) -> dict[str, Any]:
    return {name: getattr(mesh, name) for name in mesh.__dataclass_fields__}


def test_inputs_must_be_canonical_quantities() -> None:
    values = inputs()
    values["modulus"] = 760.0
    with pytest.raises(TypeError, match="CanonicalQuantity"):
        fem_request(spec(), values, plate_mesh(), "c", PROFILE)


# ------------------------------------------------------------------- lifting


def test_the_quarter_lifting_is_the_bundle_lifting() -> None:
    problem = spec()
    values = [problem.values(inputs()), problem.values(inputs(12.5, 5.0, 0.06))]
    frame = query_frame(problem, values)
    assert frame.origin.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert frame.extent.tolist() == [[16.0, 6.0], [12.5, 5.0]]
    lifting = build_lifting(problem, values)
    assert isinstance(lifting, AxisAlignedTensionLifting)
    assert lifting.displacement.tolist() == [0.08 * 16.0, 0.06 * 12.5]
    with pytest.raises(ValueError, match="not an adf"):
        adf_pieces(problem, values)


def test_the_adf_lifting_takes_its_pieces_from_the_spec() -> None:
    problem = spec("adf")
    values = [problem.values(inputs())]
    lifting = build_lifting(problem, values)
    assert isinstance(lifting, AdfLifting)
    (left, right), (bottom,) = lifting.pieces
    assert left.start.tolist() == [[0.0, 6.0]] and right.value.tolist() == [0.08 * 16.0]
    assert bottom.end.tolist() == [[16.0, 0.0]]
    x_pieces, _ = adf_pieces(problem, values, drop=("pull", 0))
    assert len(x_pieces) == 1
    points = torch.tensor([[[16.0, 3.0], [0.0, 2.0], [8.0, 0.0]]], dtype=torch.float64)
    lifted = lifting(points)
    assert torch.allclose(
        lifted.offset[0, :, 0], torch.tensor([1.28, 0.0, 0.64], dtype=torch.float64)
    )


def test_constants_and_expression_values_reach_the_pieces() -> None:
    document = plate_document("adf")
    document["boundary_conditions"][2]["values"] = [
        {"expression": "2 * unit_length", "unit": "mm"}
    ]
    document["lifting"]["pieces"]["pull"] = [
        {
            "arc": {
                "start": ["width", "0"],
                "end": ["width", "height"],
                "center": ["width - height", "0.5 * height"],
            }
        }
    ]
    problem = ProblemSpec.from_document(document)
    (_, arc), _ = adf_pieces(problem, [{"width": 16.0, "height": 6.0, "strain": 0.1}])
    assert arc.value.tolist() == [2.0]
    assert arc.center is not None and arc.center.tolist() == [[10.0, 3.0]]


# -------------------------------------------------------------- verification


def _provider(width_key: str = "width") -> Any:
    def mesh_for(values: Mapping[str, float]) -> CanonicalMesh | None:
        if values[width_key] > 19.5:
            return None
        return plate_mesh(values["width"], values["height"], columns=6, rows=3)

    return mesh_for


def test_box_points_are_feasible_and_seeded() -> None:
    problem = spec()
    points = box_points(problem, interior=5, seed=3)
    assert points == box_points(problem, interior=5, seed=3)
    assert points != box_points(problem, interior=5, seed=4)
    for point in points:
        assert 1.5 <= point["width"] / point["height"] <= 4.5
        assert point["modulus"] == 760.0
    corners = len(points) - 5
    assert 1 <= corners <= 2**3 + 1


@pytest.mark.parametrize("kind", ["quarter", "adf"])
def test_the_plate_liftings_are_verified(kind: str) -> None:
    problem = spec(kind)
    evidence = verify_lifting(
        problem, _provider(), box_points(problem, interior=6), refinements=(8, 32)
    )
    assert evidence.passed, evidence.document()
    assert evidence.refused_geometries >= 1
    assert evidence.node_residual <= 1e-12 and evidence.mutant_miss > 1e-3
    document = evidence.document()
    assert document["kind"] == kind and document["passed"]
    if kind == "quarter":
        assert evidence.conformity is None and evidence.boundary_residual == {}
    else:
        assert evidence.conformity is not None and evidence.conformity <= 1e-12
        assert set(evidence.boundary_residual) == {8, 32}
    record = evidence.verification(problem)
    assert record.boundary_refinement_levels == (8, 32)
    assert record.full_parameter_range_verified


def _shifted(document: dict[str, Any], start: list[str], end: list[str]) -> ProblemSpec:
    document["lifting"]["pieces"]["pull"] = [{"line": {"start": start, "end": end}}]
    return ProblemSpec.from_document(document)


def test_a_misplaced_piece_fails_conformity() -> None:
    problem = _shifted(
        plate_document("adf"), ["0.9 * width", "0"], ["0.9 * width", "height"]
    )
    evidence = verify_lifting(problem, _provider(), box_points(problem, interior=2))
    assert evidence.conformity is not None and evidence.conformity > 1e-9
    assert not evidence.passed
    with pytest.raises(ValueError, match="did not pass"):
        evidence.verification(problem)


def test_a_reversed_piece_fails_orientation() -> None:
    problem = _shifted(plate_document("adf"), ["width", "height"], ["width", "0"])
    evidence = verify_lifting(problem, _provider(), box_points(problem, interior=2))
    assert evidence.orientation is not None and evidence.orientation > 1e-9
    assert not evidence.passed


def test_untrimmed_pieces_and_arcs_are_measured() -> None:
    document = plate_document("adf")
    for name in ("left_x", "bottom_y"):
        document["lifting"]["pieces"][name][0]["trimmed"] = False
    document["lifting"]["pieces"]["pull"] = [
        {
            "arc": {
                "start": ["width", "0"],
                "end": ["width", "height"],
                "center": ["0.5 * width", "0.5 * height"],
            }
        }
    ]
    problem = ProblemSpec.from_document(document)
    evidence = verify_lifting(problem, _provider(), box_points(problem, interior=2))
    assert evidence.conformity is not None and evidence.conformity > 1e-3
    assert evidence.orientation == 0.0


def test_no_measured_geometry_does_not_pass() -> None:
    problem = spec()
    evidence = verify_lifting(
        problem, lambda values: None, box_points(problem, interior=1)
    )
    assert evidence.parameter_vectors == 0 and not evidence.passed
    assert evidence.mutant_miss == 0.0


def test_evidence_document_is_complete() -> None:
    evidence = LiftingEvidence(
        problem_hash="0" * 64,
        kind="adf",
        parameter_vectors=1,
        refused_geometries=0,
        refinements=(2, 4),
        node_residual=0.0,
        conformity=0.0,
        orientation=0.0,
        boundary_residual={2: 0.0, 4: math.inf},
        mutant_miss=1.0,
    )
    assert not evidence.passed
    assert evidence.document()["boundary_residual"] == {"2": 0.0, "4": math.inf}
