from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from piginot.contracts import CanonicalMesh, Unit
from piginot.datasets.projection import (
    CrossMeshMethod,
    ProjectedField,
    ProjectionSupport,
    QuadratureField,
    TransferErrorArtifact,
    compare_cross_mesh_stress,
    project_quadrature_to_cells,
    project_quadrature_to_queries,
    verify_cross_mesh_comparison,
    verify_projected_field,
)
from piginot.datasets.units import AreaUnit


def _mesh() -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id="two-cell-square",
        nodes=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        cells=((0, 1, 2), (0, 2, 3)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 3), (3, 0)),
        semantic_tags={"boundary": (0, 1, 2, 3)},
        region_ids=(1, 1),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature="two-cell-square-v1",
        quality_metrics={},
        source_provenance={"fixture": "projection"},
    )


def _tensor(value: float) -> tuple[float, ...]:
    return (value, 0.0, 0.0, 0.0, value + 1.0, 0.0, 0.0, 0.0, 0.0)


def _stress_field() -> QuadratureField:
    return QuadratureField(
        field_name="first_piola_stress",
        mesh_id="two-cell-square",
        fem_result_hash="a" * 64,
        unit=Unit.MPA,
        component_shape=(3, 3),
        points=(
            ((0.6, 0.2), (0.8, 0.4)),
            ((0.2, 0.6), (0.4, 0.8)),
        ),
        weights=((0.2, 0.3), (0.25, 0.25)),
        values=((_tensor(10.0), _tensor(14.0)), (_tensor(20.0), _tensor(22.0))),
    )


def _transfer_error() -> TransferErrorArtifact:
    return TransferErrorArtifact(
        method="stored-quadrature-componentwise-linf-v1",
        scope=(
            "componentwise L-infinity envelope over the stored quadrature samples "
            "in the recorded source cells; not a continuum error bound"
        ),
        source_field_hash="b" * 64,
        unit=Unit.MPA,
        absolute_bounds=((0.0,),),
        source_sample_counts=(1,),
    )


def _projected_field() -> ProjectedField:
    return ProjectedField(
        field_name="stress",
        mesh_id="mesh",
        unit=Unit.MPA,
        component_shape=(1,),
        support=ProjectionSupport.CELL,
        query_points=(),
        values=((1.0,),),
        source_cell_ids=((0,),),
        transfer_error=_transfer_error(),
    )


def _single_triangle_mesh(mesh_id: str) -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id=mesh_id,
        nodes=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        cells=((0, 1, 2),),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 0)),
        semantic_tags={"boundary": (0, 1, 2)},
        region_ids=(1,),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature=f"{mesh_id}-v1",
        quality_metrics={},
        source_provenance={"fixture": "projection"},
    )


def _split_triangle_mesh(mesh_id: str) -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id=mesh_id,
        nodes=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0)),
        cells=((0, 3, 2), (3, 1, 2)),
        cell_type="triangle",
        boundary_facets=((0, 3), (3, 1), (1, 2), (2, 0)),
        semantic_tags={"boundary": (0, 1, 2, 3)},
        region_ids=(1, 1),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature=f"{mesh_id}-v1",
        quality_metrics={},
        source_provenance={"fixture": "projection"},
    )


def test_cell_stress_projection_is_weighted_and_always_carries_error() -> None:
    projected = project_quadrature_to_cells(_mesh(), _stress_field())

    assert projected.support is ProjectionSupport.CELL
    assert projected.values[0][0] == pytest.approx(12.4)
    assert projected.values[1][0] == pytest.approx(21.0)
    assert projected.source_cell_ids == ((0,), (1,))
    assert _stress_field().coordinate_unit is Unit.MM
    assert _stress_field().quadrature_weight_unit is AreaUnit.SQUARE_MILLIMETRE
    assert projected.transfer_error.absolute_bounds[0][0] == pytest.approx(2.4)
    assert projected.transfer_error.absolute_bounds[1][0] == pytest.approx(1.0)
    assert projected.transfer_error.source_sample_counts == (2, 2)
    assert projected.transfer_error.scope == (
        "componentwise L-infinity envelope over the stored quadrature samples "
        "in the recorded source cells; not a continuum error bound"
    )
    assert len(projected.content_hash) == 64

    with pytest.raises(FrozenInstanceError):
        projected.values = ()  # type: ignore[misc]


def test_projected_field_verifier_rejects_forged_zero_transfer_bound() -> None:
    mesh = _mesh()
    source = _stress_field()
    projected = project_quadrature_to_cells(mesh, source)
    forged_error = replace(
        projected.transfer_error,
        absolute_bounds=tuple(
            tuple(0.0 for _ in row) for row in projected.transfer_error.absolute_bounds
        ),
    )
    forged = replace(projected, transfer_error=forged_error)

    assert forged.transfer_error.source_field_hash == source.content_hash
    assert forged.transfer_error.absolute_bounds != (
        projected.transfer_error.absolute_bounds
    )
    with pytest.raises(ValueError, match="derived projection"):
        verify_projected_field(mesh, source, forged)
    verify_projected_field(mesh, source, projected)


def test_common_queries_use_containing_cells_and_bound_every_stored_sample() -> None:
    mesh = _mesh()
    source = _stress_field()
    projected = project_quadrature_to_queries(
        mesh,
        source,
        ((0.75, 0.25), (0.5, 0.5)),
    )

    assert projected.support is ProjectionSupport.QUERY
    assert projected.values[0][0] == pytest.approx(12.4)
    assert projected.source_cell_ids == ((0,), (0, 1))
    # The shared-edge value follows canonical cell 0, while the error envelope
    # includes samples from both incident cells (max |22 - 12.4|).
    assert projected.values[1][0] == pytest.approx(12.4)
    assert projected.transfer_error.absolute_bounds[1][0] == pytest.approx(9.6)
    assert projected.transfer_error.source_sample_counts == (2, 4)
    assert projected.transfer_error.source_field_hash == source.content_hash
    verify_projected_field(mesh, source, projected)


def test_query_projection_rejects_exterior_points() -> None:
    with pytest.raises(ValueError, match="outside the mesh"):
        project_quadrature_to_queries(
            _mesh(),
            _stress_field(),
            ((2.0, 2.0),),
        )

    with pytest.raises(ValueError, match="non-empty"):
        project_quadrature_to_queries(_mesh(), _stress_field(), ())


def test_projected_field_cannot_exist_without_a_real_transfer_error() -> None:
    with pytest.raises(ValueError, match="transfer_error"):
        replace(_projected_field(), transfer_error=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("change", "match"),
    (
        ({"method": "none"}, "method"),
        ({"scope": "continuum bound"}, "scope"),
        ({"absolute_bounds": ()}, "common layout"),
        (
            {
                "absolute_bounds": ((0.0,), (0.0, 1.0)),
                "source_sample_counts": (1, 1),
            },
            "component layout",
        ),
        ({"source_sample_counts": (0,)}, "positive integers"),
        ({"absolute_bounds": ((-1.0,),)}, "non-negative"),
    ),
)
def test_transfer_error_contract_rejects_invalid_evidence(
    change: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_transfer_error(), **cast(Any, change))


@pytest.mark.parametrize(
    ("change", "match"),
    (
        ({"support": "CELL"}, "ProjectionSupport"),
        ({"unit": Unit.ONE}, "unit must match"),
        ({"coordinate_unit": Unit.CM}, "coordinate unit"),
        ({"values": ()}, "share one layout"),
        ({"source_cell_ids": ((0, 0),)}, "sorted unique"),
        ({"query_points": ((0.0, 0.0),)}, "cannot carry query"),
        (
            {"support": ProjectionSupport.QUERY, "query_points": ()},
            "one query point",
        ),
        (
            {
                "transfer_error": replace(
                    _transfer_error(), absolute_bounds=((0.0, 0.0),)
                )
            },
            "component layout",
        ),
    ),
)
def test_projected_field_contract_rejects_inconsistent_evidence(
    change: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_projected_field(), **cast(Any, change))


@pytest.mark.parametrize(
    ("change", "match"),
    (
        ({"weights": ((0.5,), (0.25, 0.25))}, "layout"),
        ({"weights": ((-0.2, 0.7), (0.25, 0.25))}, "positive"),
        (
            {
                "values": (
                    ((_tensor(10.0)), (_tensor(float("nan")))),
                    (_tensor(20.0), _tensor(22.0)),
                )
            },
            "finite",
        ),
        ({"component_shape": (2, 2)}, "component_shape"),
    ),
)
def test_quadrature_field_rejects_invalid_layouts(
    change: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_stress_field(), **cast(Any, change))


def test_quadrature_field_declares_physical_coordinate_and_area_units() -> None:
    with pytest.raises(ValueError, match="coordinate unit"):
        replace(_stress_field(), coordinate_unit=Unit.CM)
    with pytest.raises(ValueError, match="quadrature weight unit"):
        replace(_stress_field(), quadrature_weight_unit=cast(Any, "mm2"))


def test_projection_requires_physical_quadrature_weights_for_each_cell() -> None:
    reference_cell_weights = replace(
        _stress_field(),
        weights=((0.2, 0.2), (0.25, 0.25)),
    )

    with pytest.raises(ValueError, match="physical cell area"):
        project_quadrature_to_cells(_mesh(), reference_cell_weights)


def test_projection_rejects_mesh_identity_and_nonstress_cell_conversion() -> None:
    with pytest.raises(ValueError, match="mesh_id"):
        project_quadrature_to_cells(
            replace(_mesh(), mesh_id="different"), _stress_field()
        )

    dimensionless = replace(_stress_field(), unit=Unit.ONE)
    with pytest.raises(ValueError, match="stress unit"):
        project_quadrature_to_cells(_mesh(), dimensionless)


def test_cross_mesh_stress_comparison_uses_physical_integrals() -> None:
    coarse_mesh = _single_triangle_mesh("coarse")
    fine_mesh = _split_triangle_mesh("fine")
    constant = _tensor(7.0)
    coarse = QuadratureField(
        field_name="first_piola_stress",
        mesh_id="coarse",
        fem_result_hash="c" * 64,
        unit=Unit.MPA,
        component_shape=(3, 3),
        points=(((1.0 / 3.0, 1.0 / 3.0),),),
        weights=((0.5,),),
        values=((constant,),),
    )
    fine = QuadratureField(
        field_name="first_piola_stress",
        mesh_id="fine",
        fem_result_hash="d" * 64,
        unit=Unit.MPA,
        component_shape=(3, 3),
        points=(((1.0 / 6.0, 1.0 / 3.0),), ((2.0 / 3.0, 1.0 / 6.0),)),
        weights=((0.25,), (0.25,)),
        values=((constant,), (constant,)),
    )

    comparison = compare_cross_mesh_stress(
        coarse_mesh,
        coarse,
        fine_mesh,
        fine,
    )

    assert comparison.method is CrossMeshMethod.PHYSICAL_DOMAIN_INTEGRAL
    assert comparison.integral_unit is Unit.N
    assert comparison.left_source_field_hash == coarse.content_hash
    assert comparison.right_source_field_hash == fine.content_hash
    assert comparison.left_integral[0] == pytest.approx(3.5)
    assert comparison.right_integral == pytest.approx(comparison.left_integral)
    assert comparison.absolute_difference == pytest.approx((0.0,) * 9)
    assert comparison.relative_l2_difference == pytest.approx(0.0)
    assert comparison.left_quadrature_sample_count == 1
    assert comparison.right_quadrature_sample_count == 2


def test_cross_mesh_verifier_rejects_forged_derived_integrals() -> None:
    mesh = _single_triangle_mesh("verification")
    field = QuadratureField(
        field_name="first_piola_stress",
        mesh_id=mesh.mesh_id,
        fem_result_hash="9" * 64,
        unit=Unit.MPA,
        component_shape=(1,),
        points=(((0.25, 0.25),),),
        weights=((0.5,),),
        values=(((2.0,),),),
    )
    comparison = compare_cross_mesh_stress(mesh, field, mesh, field)
    forged = replace(
        comparison,
        left_integral=(0.0,),
        right_integral=(0.0,),
        absolute_difference=(0.0,),
        relative_l2_difference=0.0,
    )

    assert forged.left_source_field_hash == field.content_hash
    with pytest.raises(ValueError, match="derived cross-mesh comparison"):
        verify_cross_mesh_comparison(mesh, field, mesh, field, forged)
    verify_cross_mesh_comparison(
        mesh,
        field,
        mesh,
        field,
        comparison,
    )


def test_cross_mesh_artifact_cannot_be_forged_with_node_pairing_metadata() -> None:
    mesh = _single_triangle_mesh("direct-contract")
    field = QuadratureField(
        field_name="first_piola_stress",
        mesh_id=mesh.mesh_id,
        fem_result_hash="2" * 64,
        unit=Unit.MPA,
        component_shape=(1,),
        points=(((0.25, 0.25),),),
        weights=((0.5,),),
        values=(((1.0,),),),
    )
    comparison = compare_cross_mesh_stress(mesh, field, mesh, field)

    with pytest.raises(ValueError, match="physical-domain integral"):
        replace(comparison, method=cast(Any, "raw_node_pairing"))
    with pytest.raises(ValueError, match="source field hash"):
        replace(comparison, left_source_field_hash="not-a-digest")
    with pytest.raises(ValueError, match="absolute difference"):
        replace(comparison, absolute_difference=(1.0,))
    with pytest.raises(ValueError, match="relative difference"):
        replace(comparison, relative_l2_difference=float("nan"))


@pytest.mark.parametrize("forbidden", ("raw_node_pairing", "nearest_node"))
def test_cross_mesh_comparison_explicitly_rejects_node_pairing(forbidden: str) -> None:
    mesh = _single_triangle_mesh("same")
    field = QuadratureField(
        field_name="first_piola_stress",
        mesh_id="same",
        fem_result_hash="e" * 64,
        unit=Unit.MPA,
        component_shape=(1,),
        points=(((0.25, 0.25),),),
        weights=((0.5,),),
        values=(((1.0,),),),
    )

    with pytest.raises(ValueError, match="node pairing is forbidden"):
        compare_cross_mesh_stress(mesh, field, mesh, field, method=forbidden)


def test_cross_mesh_comparison_rejects_incompatible_fields() -> None:
    mesh = _single_triangle_mesh("same")
    field = QuadratureField(
        field_name="first_piola_stress",
        mesh_id="same",
        fem_result_hash="f" * 64,
        unit=Unit.MPA,
        component_shape=(1,),
        points=(((0.25, 0.25),),),
        weights=((0.5,),),
        values=(((1.0,),),),
    )

    with pytest.raises(ValueError, match="same field"):
        compare_cross_mesh_stress(mesh, field, mesh, replace(field, field_name="other"))
    with pytest.raises(ValueError, match="stress unit"):
        compare_cross_mesh_stress(mesh, replace(field, unit=Unit.ONE), mesh, field)
    with pytest.raises(ValueError, match="identical units and components"):
        compare_cross_mesh_stress(
            mesh,
            field,
            mesh,
            replace(field, component_shape=(1, 1)),
        )


def test_zero_cross_mesh_integrals_have_zero_relative_difference() -> None:
    mesh = _single_triangle_mesh("zero")
    zero = QuadratureField(
        field_name="stress",
        mesh_id="zero",
        fem_result_hash="0" * 64,
        unit=Unit.MPA,
        component_shape=(1,),
        points=(((0.25, 0.25),),),
        weights=((0.5,),),
        values=(((0.0,),),),
    )

    comparison = compare_cross_mesh_stress(mesh, zero, mesh, zero)

    assert comparison.relative_l2_difference == 0.0


def test_projection_rejects_nonphysical_field_mesh_layouts() -> None:
    field = _stress_field()
    with pytest.raises(ValueError, match="cell layout"):
        project_quadrature_to_cells(
            _mesh(),
            replace(
                field,
                points=field.points[:1],
                weights=field.weights[:1],
                values=field.values[:1],
            ),
        )
    outside = replace(
        field,
        points=(((2.0, 2.0), field.points[0][1]), field.points[1]),
    )
    with pytest.raises(ValueError, match="outside its declared"):
        project_quadrature_to_cells(_mesh(), outside)


def test_clockwise_mesh_cells_are_located_without_assuming_orientation() -> None:
    mesh = replace(
        _single_triangle_mesh("clockwise"),
        cells=((0, 2, 1),),
        orientation="clockwise",
    )
    field = QuadratureField(
        field_name="stress",
        mesh_id="clockwise",
        fem_result_hash="1" * 64,
        unit=Unit.MPA,
        component_shape=(1,),
        points=(((0.25, 0.25),),),
        weights=((0.5,),),
        values=(((2.0,),),),
    )

    assert project_quadrature_to_cells(mesh, field).values == ((2.0,),)
