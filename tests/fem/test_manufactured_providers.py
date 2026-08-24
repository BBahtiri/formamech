"""Provider adapters from exact MMS fields into the primary FEM backend."""

from __future__ import annotations

import numpy as np

from piginot.contracts import CanonicalQuantity, MaterialSpec, Unit
from piginot.fem.body_forces import BodyForceProvider
from piginot.fem.field_providers import (
    DirichletFieldProvider,
    TractionFieldProvider,
)
from piginot.fem.manufactured import build_manufactured_solution
from piginot.fem.manufactured_providers import (
    ManufacturedBodyForceProvider,
    ManufacturedDisplacementProvider,
    ManufacturedTractionProvider,
)


def _material(young: float = 100.0) -> MaterialSpec:
    return MaterialSpec(
        "mms_provider_material",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(young, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
        },
    )


def test_adapters_evaluate_the_exact_manufactured_fields() -> None:
    material = _material()
    exact = build_manufactured_solution("mixed_dirichlet_traction", material)
    points = np.asarray(
        (((0.0, 0.2), (0.8, 1.0)), ((0.1, 0.4), (0.7, 0.9))),
        dtype=np.float64,
    )
    normals = np.broadcast_to(
        np.asarray(((1.0,), (0.0,)), dtype=np.float64)[:, :, None],
        points.shape,
    ).copy()

    displacement = ManufacturedDisplacementProvider(exact.name, material)
    body = ManufacturedBodyForceProvider(exact.name, material)
    traction = ManufacturedTractionProvider(exact.name, material)

    np.testing.assert_array_equal(
        displacement.evaluate(points), exact.displacement(points)
    )
    np.testing.assert_array_equal(body.evaluate(points), exact.body_force(points))
    np.testing.assert_array_equal(
        traction.evaluate(points, normals),
        exact.traction(points, normals),
    )
    assert isinstance(displacement, DirichletFieldProvider)
    assert isinstance(body, BodyForceProvider)
    assert isinstance(traction, TractionFieldProvider)


def test_provider_identity_is_stable_and_content_sensitive() -> None:
    material = _material()
    same = (
        ManufacturedBodyForceProvider("pure_dirichlet", material),
        ManufacturedBodyForceProvider("pure_dirichlet", material),
    )
    changed_case = ManufacturedBodyForceProvider("strongly_varying_f33", material)
    changed_material = ManufacturedBodyForceProvider(
        "pure_dirichlet", _material(young=101.0)
    )
    displacement = ManufacturedDisplacementProvider("pure_dirichlet", material)
    traction = ManufacturedTractionProvider("pure_dirichlet", material)

    assert same[0].provider_id == same[1].provider_id
    assert same[0].provider_id != changed_case.provider_id
    assert same[0].provider_id != changed_material.provider_id
    assert (
        len({same[0].provider_id, displacement.provider_id, traction.provider_id}) == 3
    )
