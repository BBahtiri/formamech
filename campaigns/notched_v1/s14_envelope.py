"""Notched-plate envelope checker, refusal fixtures and in-domain set.

The fixture set holds 216 labelled out-of-distribution requests:

- 72 primitive-bound violations: four inputs x two sides x three distances
  beyond the bound (0.5%, 5% and 25% of the range) x three base points;
- 24 material and load contract violations;
- 36 topology or tag violations (a punched hole, a renamed tag, a detached
  extra component);
- 48 query-containment violations (beyond the loaded end, inside the notch
  cut-out, above the top edge, negative coordinates, non-finite or malformed
  points);
- 36 guard-band near-misses one ulp, one part per million or one part per
  billion outside a primitive, material or load bound.

The family's derived bounds (net-section factor and notch sharpness) are
reached only at corners of the primitive box, so no request inside the
primitive bounds violates them: this family has no derived-bound category.
The in-domain set is the 200 active public cases plus the 500 accepted shapes
of the public family-validation sweep, each with 32 query points.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from campaigns.dogbone_v1.s14_envelope import (
    InDomainCase,
    RefusalFixture,
    _nan_quantity,
    _rebuild,
    in_domain_queries,
)
from campaigns.dogbone_v1.s14_envelope import mutated_mesh as _generic_mutation
from campaigns.notched_v1.family import (
    BOUNDS,
    GEOMETRY,
    INPUTS,
    LOADED_END_TAG,
    compiled_family,
    parameter_point,
)
from campaigns.notched_v1.pilot import public_samples
from campaigns.notched_v1.proposal import executable_envelope
from campaigns.notched_v1.s9 import EXECUTABLE_ENVELOPE_PATH, PUBLIC_MANIFEST_PATH
from piginot.contracts import CanonicalMesh, CanonicalQuantity, Unit
from piginot.doe import ExecutableEnvelope
from piginot.morphing import BlendedNotchParameters, apply_morph
from piginot.verification.envelope import (
    EnvelopeChecker,
    GeometryOutcome,
    RefusalDecision,
)

_ROOT = Path(__file__).resolve().parents[2]
DISTANCES = (0.005, 0.05, 0.25)
FIXTURE_COUNTS = {
    "primitive-bound": 72,
    "material-load": 24,
    "topology-tag": 36,
    "query-containment": 48,
    "guard-band": 36,
}
REQUEST_FORM_CATEGORIES = tuple(c for c in FIXTURE_COUNTS if c != "topology-tag")
BASE_POINTS = (
    {
        "width": 23.0,
        "notch_radius": 5.0,
        "notch_depth": 3.0,
        "nominal_axial_strain": 0.15,
    },
    {
        "width": 18.0,
        "notch_radius": 3.0,
        "notch_depth": 4.0,
        "nominal_axial_strain": 0.12,
    },
    {
        "width": 27.0,
        "notch_radius": 7.0,
        "notch_depth": 2.0,
        "nominal_axial_strain": 0.18,
    },
)


@lru_cache(maxsize=1)
def notched_envelope() -> ExecutableEnvelope:
    """The sealed executable envelope, checked against its committed bytes."""

    envelope = executable_envelope()
    if envelope.to_bytes() != (_ROOT / EXECUTABLE_ENVELOPE_PATH).read_bytes():
        raise RuntimeError("executable envelope differs from the committed artifact")
    return envelope


def morph(theta: Mapping[str, object]) -> GeometryOutcome:
    family = compiled_family()
    result = apply_morph(
        family.reference_mesh,
        family.morph_map,
        family.membership,
        {name: theta[name] for name in GEOMETRY},
    )
    return GeometryOutcome(
        result.mesh if result.accepted else None,
        tuple(reason.value for reason in result.reason_codes),
    )


@lru_cache(maxsize=1)
def notched_checker() -> EnvelopeChecker:
    return EnvelopeChecker(
        notched_envelope(),
        reference_mesh=compiled_family().reference_mesh,
        geometry=morph,
    )


def theta(
    values: Mapping[str, float],
    *,
    elastic_modulus: float = 760.0,
    poisson_ratio: float = 0.23,
) -> dict[str, object]:
    point: dict[str, object] = dict(
        parameter_point(*(values[name] for name in GEOMETRY))
    )
    point["elastic_modulus"] = CanonicalQuantity(elastic_modulus, Unit.MPA)
    point["poisson_ratio"] = CanonicalQuantity(poisson_ratio, Unit.ONE)
    point["nominal_axial_strain"] = CanonicalQuantity(
        values["nominal_axial_strain"], Unit.ONE
    )
    return point


def _bounds() -> dict[str, tuple[float, float]]:
    return {
        bound.name: (bound.lower_bound.value, bound.upper_bound.value)
        for bound in notched_envelope().primitive_bounds
    }


def _primitive_fixtures() -> list[RefusalFixture]:
    bounds = _bounds()
    fixtures = []
    for base_index, base in enumerate(BASE_POINTS):
        for name in INPUTS:
            lower, upper = bounds[name]
            span = upper - lower
            for side, distance in itertools.product(("below", "above"), DISTANCES):
                value = (
                    lower - distance * span
                    if side == "below"
                    else upper + distance * span
                )
                fixtures.append(
                    RefusalFixture(
                        f"primitive-{name}-{side}-{distance:g}-b{base_index}",
                        "primitive-bound",
                        {**base, name: value},
                        "PARAMETER_OUT_OF_BOUNDS",
                        description=f"{name} {side} its bound by {distance:.1%} of range",
                    )
                )
    return fixtures


def _material_fixtures() -> list[RefusalFixture]:
    base = BASE_POINTS[0]
    good = theta(base)

    def override(**changes: object) -> dict[str, object]:
        point = dict(good)
        for key, value in changes.items():
            if value is None:
                point.pop(key, None)
            else:
                point[key] = value
        return point

    cases: list[tuple[str, str, dict[str, object]]] = [
        ("E-700", "PARAMETER_OUT_OF_BOUNDS", theta(base, elastic_modulus=700.0)),
        ("E-800", "PARAMETER_OUT_OF_BOUNDS", theta(base, elastic_modulus=800.0)),
        ("E-761", "PARAMETER_OUT_OF_BOUNDS", theta(base, elastic_modulus=761.0)),
        ("E-0", "PARAMETER_OUT_OF_BOUNDS", theta(base, elastic_modulus=0.0)),
        ("nu-0.2", "PARAMETER_OUT_OF_BOUNDS", theta(base, poisson_ratio=0.2)),
        ("nu-0.3", "PARAMETER_OUT_OF_BOUNDS", theta(base, poisson_ratio=0.3)),
        ("nu-0.49", "PARAMETER_OUT_OF_BOUNDS", theta(base, poisson_ratio=0.49)),
        ("nu-negative", "PARAMETER_OUT_OF_BOUNDS", theta(base, poisson_ratio=-0.1)),
        (
            "E-length-unit",
            "PARAMETER_UNIT_MISMATCH",
            override(elastic_modulus=CanonicalQuantity(760.0, Unit.MM)),
        ),
        (
            "E-force-unit",
            "PARAMETER_UNIT_MISMATCH",
            override(elastic_modulus=CanonicalQuantity(760.0, Unit.N)),
        ),
        (
            "strain-mm",
            "PARAMETER_UNIT_MISMATCH",
            override(nominal_axial_strain=CanonicalQuantity(0.15, Unit.MM)),
        ),
        (
            "width-stress-unit",
            "PARAMETER_UNIT_MISMATCH",
            override(width=CanonicalQuantity(23.0, Unit.MPA)),
        ),
        ("E-float", "INVALID_PARAMETER_VALUE", override(elastic_modulus=760.0)),
        (
            "strain-nan",
            "INVALID_PARAMETER_VALUE",
            override(nominal_axial_strain=_nan_quantity()),
        ),
        (
            "strain-text",
            "INVALID_PARAMETER_VALUE",
            override(nominal_axial_strain="0.15"),
        ),
        ("nu-none", "MISSING_PARAMETER", override(poisson_ratio=None)),
        ("E-missing", "MISSING_PARAMETER", override(elastic_modulus=None)),
        ("strain-missing", "MISSING_PARAMETER", override(nominal_axial_strain=None)),
        ("depth-missing", "MISSING_PARAMETER", override(notch_depth=None)),
        (
            "temperature",
            "UNKNOWN_PARAMETER",
            override(temperature=CanonicalQuantity(20.0, Unit.ONE)),
        ),
        (
            "thickness",
            "UNKNOWN_PARAMETER",
            override(thickness=CanonicalQuantity(1.0, Unit.MM)),
        ),
        (
            "strain-0.05",
            "PARAMETER_OUT_OF_BOUNDS",
            theta({**base, "nominal_axial_strain": 0.05}),
        ),
        (
            "strain-0.3",
            "PARAMETER_OUT_OF_BOUNDS",
            theta({**base, "nominal_axial_strain": 0.3}),
        ),
        (
            "strain-compression",
            "PARAMETER_OUT_OF_BOUNDS",
            theta({**base, "nominal_axial_strain": -0.1}),
        ),
    ]
    return [
        RefusalFixture(
            f"material-load-{name}",
            "material-load",
            base,
            code,
            theta_override=point,
            description=name,
        )
        for name, code, point in cases
    ]


def _domain_values(index: int) -> dict[str, float]:
    strain = 0.10 + 0.10 * ((index * 7) % 11) / 10.0
    return {**BASE_POINTS[index % 3], "nominal_axial_strain": strain}


def _topology_fixtures() -> list[RefusalFixture]:
    fixtures = []
    for kind, code in (
        ("punched-hole", "TOPOLOGY_CHANGED"),
        ("renamed-tag", "TAG_IDENTITY_CHANGED"),
        ("extra-component", "TOPOLOGY_CHANGED"),
    ):
        for index in range(12):
            fixtures.append(
                RefusalFixture(
                    f"topology-{kind}-{index}",
                    "topology-tag",
                    _domain_values(index),
                    code,
                    mesh_mutation=f"{kind}:{index}",
                    description=kind,
                )
            )
    return fixtures


def _query_fixtures() -> list[RefusalFixture]:
    fixtures = []
    for index in range(48):
        values = _domain_values(index)
        shape = BlendedNotchParameters(
            25.0, values["width"], values["notch_radius"], values["notch_depth"]
        )
        half_width = 0.5 * values["width"]
        group = index // 6
        bad: object
        if group == 0:
            bad, kind = (25.0 + 0.5 + index, 0.25 * half_width), "beyond-loaded-end"
        elif group == 1:
            x = 0.15 * shape.blend_length_mm * (index - 5)
            bad = (x, 0.5 * (shape.profile_height_mm(x) + half_width))
            kind = "notch-cut-out"
        elif group == 2:
            bad, kind = (-1.0 - index, 1.0), "negative-x"
        elif group == 3:
            bad, kind = (1.0, -0.5 - 0.1 * index), "negative-y"
        elif group == 4:
            bad, kind = (20.0, half_width + 0.01 * (index + 1)), "above-top-edge"
        elif group == 5:
            bad, kind = (0.0, half_width - 0.01 * (index - 29)), "above-notch-root"
        else:
            bad = ((math.nan, 1.0), (1.0, math.inf), (1.0,), "1,1", (True, 1.0), None)[
                index % 6
            ]
            kind = "malformed"
        code = "INVALID_QUERY" if kind == "malformed" else "QUERY_OUTSIDE_DOMAIN"
        queries = ((5.0, 1.0), bad)
        fixtures.append(
            RefusalFixture(
                f"query-{kind}-{index}",
                "query-containment",
                values,
                code,
                queries=queries,
                description=kind,
            )
        )
    return fixtures


def _guard_fixtures() -> list[RefusalFixture]:
    bounds = _bounds()
    base = BASE_POINTS[0]
    fixtures = []
    for name in (*INPUTS, "elastic_modulus", "poisson_ratio"):
        lower, upper = bounds[name]
        for side, edge, direction in (
            ("below", lower, -math.inf),
            ("above", upper, math.inf),
        ):
            for label, value in (
                ("ulp", math.nextafter(edge, direction)),
                (
                    "ppm",
                    edge * (1.0 - 1e-6) if side == "below" else edge * (1.0 + 1e-6),
                ),
                (
                    "ppb",
                    edge * (1.0 - 1e-9) if side == "below" else edge * (1.0 + 1e-9),
                ),
            ):
                if name in ("elastic_modulus", "poisson_ratio"):
                    fixture = RefusalFixture(
                        f"guard-{name}-{side}-{label}",
                        "guard-band",
                        base,
                        "PARAMETER_OUT_OF_BOUNDS",
                        theta_override=theta(base, **{name: value}),
                        description=f"{name} {side} by one {label}",
                    )
                else:
                    fixture = RefusalFixture(
                        f"guard-{name}-{side}-{label}",
                        "guard-band",
                        {**base, name: value},
                        "PARAMETER_OUT_OF_BOUNDS",
                        description=f"{name} {side} by one {label}",
                    )
                fixtures.append(fixture)
    return fixtures


@lru_cache(maxsize=1)
def refusal_fixtures() -> tuple[RefusalFixture, ...]:
    built = (
        *_primitive_fixtures(),
        *_material_fixtures(),
        *_topology_fixtures(),
        *_query_fixtures(),
        *_guard_fixtures(),
    )
    # The shared fixture record builds DogBone requests from ``values``; every
    # notched fixture therefore carries its notched request explicitly.
    fixtures = tuple(
        f
        if f.theta_override is not None
        else replace(f, theta_override=theta(f.values))
        for f in built
    )
    counts = {c: sum(f.category == c for f in fixtures) for c in FIXTURE_COUNTS}
    if counts != FIXTURE_COUNTS:
        raise RuntimeError(f"fixture counts {counts} differ from the registered counts")
    if len({fixture.fixture_id for fixture in fixtures}) != len(fixtures):
        raise RuntimeError("fixture ids must be unique")
    return fixtures


def request_form_fixtures() -> tuple[RefusalFixture, ...]:
    """Fixtures expressible as a request (no mesh mutation), for the bundle."""

    return tuple(f for f in refusal_fixtures() if f.mesh_mutation is None)


def fixture_set_hash(fixtures: tuple[RefusalFixture, ...]) -> str:
    payload = json.dumps([f.summary() for f in fixtures], sort_keys=True, default=repr)
    return hashlib.sha256(payload.encode()).hexdigest()


def mutated_mesh(mesh: CanonicalMesh, mutation: str) -> CanonicalMesh:
    """A punched hole, a renamed tag or an extra component on a morphed mesh."""

    kind, index_text = mutation.split(":")
    if kind != "renamed-tag":
        return _generic_mutation(mesh, mutation)
    renames = (
        {LOADED_END_TAG: "loaded_end"}
        if int(index_text) % 2 == 0
        else {"top": "bottom"}
    )
    tag_of_edge = {
        frozenset(mesh.boundary_facets[facet]): renames.get(name, name)
        for name, facets in mesh.semantic_tags.items()
        for facet in facets
    }
    return _rebuild(
        mesh,
        [tuple(cell) for cell in mesh.cells],  # type: ignore[misc]
        list(mesh.region_ids),
        [tuple(node) for node in mesh.nodes],  # type: ignore[misc]
        tag_of_edge,
        "untagged",
    )


def check_fixture(fixture: RefusalFixture) -> RefusalDecision:
    request = fixture.request_theta()
    mesh = None
    if fixture.mesh_mutation is not None:
        outcome = morph(request)
        if outcome.mesh is None:
            raise RuntimeError(f"{fixture.fixture_id} base geometry is not valid")
        mesh = mutated_mesh(outcome.mesh, fixture.mesh_mutation)
    return notched_checker().check(request, mesh=mesh, queries=fixture.queries)


def in_domain_cases() -> tuple[InDomainCase, ...]:
    """200 active public cases plus 500 accepted public sweep shapes."""

    manifest = json.loads((_ROOT / PUBLIC_MANIFEST_PATH).read_text("utf-8"))
    cases = [
        InDomainCase(
            case["case_id"],
            {name: float(case["primitive_coordinates"][name]) for name in INPUTS},
        )
        for case in manifest["cases"]
        if case["status"] == "ACTIVE"
    ]
    family = compiled_family()
    sweep: list[InDomainCase] = []
    for values in public_samples(900):
        if not family.membership.evaluate(parameter_point(*values)).accepted:
            continue
        strain = 0.10 + 0.10 * (len(sweep) % 21) / 20.0
        sweep.append(
            InDomainCase(
                f"sweep-{len(sweep):03d}",
                {
                    **dict(zip(GEOMETRY, values, strict=True)),
                    "nominal_axial_strain": strain,
                },
                "sweep",
            )
        )
        if len(sweep) == 500:
            break
    if len(cases) != 200 or len(sweep) != 500:
        raise RuntimeError("in-domain population differs from the registered counts")
    return (*cases, *sweep)


def check_in_domain(case: InDomainCase) -> RefusalDecision:
    request = theta(case.values)
    outcome = morph(request)
    if outcome.mesh is None:
        return RefusalDecision(False, outcome.reason_codes or ("INVALID_MORPHED_MESH",))
    return notched_checker().check(
        request, mesh=outcome.mesh, queries=in_domain_queries(outcome.mesh)
    )


__all__ = (
    "BASE_POINTS",
    "BOUNDS",
    "FIXTURE_COUNTS",
    "REQUEST_FORM_CATEGORIES",
    "check_fixture",
    "check_in_domain",
    "fixture_set_hash",
    "in_domain_cases",
    "morph",
    "mutated_mesh",
    "notched_checker",
    "notched_envelope",
    "refusal_fixtures",
    "request_form_fixtures",
    "theta",
)
