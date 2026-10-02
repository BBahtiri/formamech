"""DogBone S14 envelope checker, refusal fixtures and in-domain set (ADR 0019).

The fixture set holds exactly 240 labelled out-of-distribution requests:

- 60 primitive-bound violations: five inputs x two sides x three distances
  beyond the bound (0.5%, 5% and 25% of the range) x two base points;
- 48 coupled derived-bound violations, every primitive inside its bound;
- 24 material and load contract violations;
- 36 topology or tag violations (a punched hole, a renamed or moved tag, a
  detached extra component);
- 36 query-containment violations (beyond the loaded end, in the fillet
  cut-out, negative coordinates, non-finite or malformed points);
- 36 guard-band near-misses one ulp or one part per million outside a bound.

The in-domain set is the 344 active public S10 cases plus the 500 accepted
shapes of the public family-validation sweep, each with 32 query points.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch

from campaigns.dogbone_v1.campaign_proposal import proposed_executable_envelope
from campaigns.dogbone_v1.family import CompiledFamily, compiled_family, parameter_point
from campaigns.dogbone_v1.pilot import _samples
from piginot.contracts import CanonicalMesh, CanonicalQuantity, Unit
from piginot.doe import ExecutableEnvelope
from piginot.geometry.canonicalize import canonicalize_mesh
from piginot.geometry.models import CellType, MeshPayload
from piginot.morphing import apply_morph
from piginot.verification.envelope import (
    EnvelopeChecker,
    GeometryOutcome,
    RefusalDecision,
)

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS = _ROOT / "artifacts" / "campaigns" / "dogbone-v1"
_ENVELOPE_PATH = _ARTIFACTS / "executable-envelope.json"
_MANIFEST_PATH = _ARTIFACTS / "public-case-manifest.json"
GEOMETRY = ("L_total", "W_grip", "W_gauge", "R_fillet")
INPUTS = (*GEOMETRY, "nominal_axial_strain")
DISTANCES = (0.005, 0.05, 0.25)
QUERIES_PER_CASE = 32
FIXTURE_COUNTS = {
    "primitive-bound": 60,
    "derived-bound": 48,
    "material-load": 24,
    "topology-tag": 36,
    "query-containment": 36,
    "guard-band": 36,
}
_SEARCH_SEED = 14_001


@lru_cache(maxsize=1)
def _family() -> CompiledFamily:
    return compiled_family()


@lru_cache(maxsize=1)
def dogbone_envelope() -> ExecutableEnvelope:
    """The ratified executable envelope, checked against its committed bytes."""

    envelope = proposed_executable_envelope()
    if envelope.to_bytes() != _ENVELOPE_PATH.read_bytes():
        raise RuntimeError("executable envelope differs from the committed artifact")
    return envelope


def _morph(theta: Mapping[str, object]) -> GeometryOutcome:
    family = _family()
    geometry = {name: theta[name] for name in GEOMETRY}
    result = apply_morph(
        family.reference_mesh, family.morph_map, family.membership, geometry
    )
    return GeometryOutcome(
        result.mesh if result.accepted else None,
        tuple(reason.value for reason in result.reason_codes),
    )


@lru_cache(maxsize=1)
def dogbone_checker() -> EnvelopeChecker:
    return EnvelopeChecker(
        dogbone_envelope(),
        reference_mesh=_family().reference_mesh,
        geometry=_morph,
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


@dataclass(frozen=True, slots=True)
class RefusalFixture:
    """One labelled request that must be refused with ``expected_code``."""

    fixture_id: str
    category: str
    values: Mapping[str, float]
    expected_code: str
    theta_override: Mapping[str, object] | None = None
    mesh_mutation: str | None = None
    queries: tuple[object, ...] | None = None
    description: str = ""

    def request_theta(self) -> dict[str, object]:
        if self.theta_override is not None:
            return dict(self.theta_override)
        return theta(self.values)

    def summary(self) -> dict[str, object]:
        return {
            "id": self.fixture_id,
            "category": self.category,
            "values": dict(sorted(self.values.items())),
            "expected_code": self.expected_code,
            "mesh_mutation": self.mesh_mutation,
            "queries": None
            if self.queries is None
            else [repr(point) for point in self.queries],
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class InDomainCase:
    case_id: str
    values: Mapping[str, float]
    source: str = field(default="public")


@lru_cache(maxsize=1)
def _bounds() -> dict[str, tuple[float, float]]:
    envelope = dogbone_envelope()
    return {
        bound.name: (bound.lower_bound.value, bound.upper_bound.value)
        for bound in envelope.primitive_bounds
    }


def _base_points() -> tuple[dict[str, float], dict[str, float]]:
    return (
        {
            "L_total": 55.0,
            "W_grip": 21.0,
            "W_gauge": 10.0,
            "R_fillet": 14.0,
            "nominal_axial_strain": 0.15,
        },
        {
            "L_total": 62.0,
            "W_grip": 19.0,
            "W_gauge": 8.0,
            "R_fillet": 11.0,
            "nominal_axial_strain": 0.12,
        },
    )


def _primitive_fixtures() -> list[RefusalFixture]:
    bounds = _bounds()
    fixtures: list[RefusalFixture] = []
    for base_index, base in enumerate(_base_points()):
        for name in INPUTS:
            lower, upper = bounds[name]
            span = upper - lower
            for side, distance in itertools.product(("below", "above"), DISTANCES):
                value = (
                    lower - distance * span
                    if side == "below"
                    else upper + distance * span
                )
                values = {**base, name: value}
                fixtures.append(
                    RefusalFixture(
                        f"primitive-{name}-{side}-{distance:g}-b{base_index}",
                        "primitive-bound",
                        values,
                        "PARAMETER_OUT_OF_BOUNDS",
                        description=f"{name} {side} its bound by {distance:.1%} of range",
                    )
                )
    return fixtures


def _derived_violation(values: Mapping[str, float]) -> tuple[str, str] | None:
    """Return (derived bound, side) when exactly the derived bounds reject."""

    family = _family()
    decision = family.membership.evaluate(
        parameter_point(*(values[name] for name in GEOMETRY))
    )
    codes = {getattr(code, "value", code) for code in decision.reason_codes}
    if codes != {"DERIVED_QUANTITY_OUT_OF_BOUNDS"}:
        return None
    for bound in family.membership.derived_bounds:
        value = decision.derived_values[bound.name].value
        if value < bound.lower_bound.value:
            return bound.name, "below"
        if value > bound.upper_bound.value:
            return bound.name, "above"
    return None


def _derived_fixtures() -> list[RefusalFixture]:
    bounds = _bounds()
    generator = torch.Generator().manual_seed(_SEARCH_SEED)
    groups: dict[tuple[str, str], list[dict[str, float]]] = {}
    for _ in range(20_000):
        draw = torch.rand(len(GEOMETRY), generator=generator, dtype=torch.float64)
        values = {
            name: bounds[name][0] + float(draw[i]) * (bounds[name][1] - bounds[name][0])
            for i, name in enumerate(GEOMETRY)
        }
        values["nominal_axial_strain"] = 0.15
        violation = _derived_violation(values)
        if violation is not None and len(groups.setdefault(violation, [])) < 48:
            groups[violation].append(values)
    fixtures: list[RefusalFixture] = []
    keys = sorted(groups)
    round_index = 0
    while len(fixtures) < FIXTURE_COUNTS["derived-bound"]:
        added = False
        for key in keys:
            if round_index < len(groups[key]) and (
                len(fixtures) < FIXTURE_COUNTS["derived-bound"]
            ):
                name, side = key
                fixtures.append(
                    RefusalFixture(
                        f"derived-{name}-{side}-{round_index}",
                        "derived-bound",
                        groups[key][round_index],
                        "DERIVED_QUANTITY_OUT_OF_BOUNDS",
                        description=f"{name} {side} its bound, primitives inside",
                    )
                )
                added = True
        if not added:
            raise RuntimeError("derived-bound search found too few violations")
        round_index += 1
    return fixtures


def _material_fixtures() -> list[RefusalFixture]:
    base = _base_points()[0]
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
            "L-stress-unit",
            "PARAMETER_UNIT_MISMATCH",
            override(L_total=CanonicalQuantity(55.0, Unit.MPA)),
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
        ("W-gauge-missing", "MISSING_PARAMETER", override(W_gauge=None)),
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


def _nan_quantity() -> object:
    quantity = CanonicalQuantity(0.15, Unit.ONE)
    object.__setattr__(quantity, "value", math.nan)
    return quantity


def _domain_values(index: int) -> dict[str, float]:
    strain = 0.10 + 0.10 * ((index * 7) % 11) / 10.0
    base = _base_points()[index % 2]
    return {**base, "nominal_axial_strain": strain}


def _topology_fixtures() -> list[RefusalFixture]:
    fixtures: list[RefusalFixture] = []
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
    fixtures: list[RefusalFixture] = []
    for index in range(36):
        values = _domain_values(index)
        half_length = 0.5 * values["L_total"]
        half_grip = 0.5 * values["W_grip"]
        half_gauge = 0.5 * values["W_gauge"]
        group = index // 6
        if group == 0:
            bad: object = (half_length + 0.5 + index, 0.25 * half_grip)
            code, kind = "QUERY_OUTSIDE_DOMAIN", "beyond-loaded-end"
        elif group == 1:
            bad = (0.1 * (index - 5), 0.5 * (half_gauge + half_grip))
            code, kind = "QUERY_OUTSIDE_DOMAIN", "fillet-cut-out"
        elif group == 2:
            bad = (-1.0 - index, 1.0)
            code, kind = "QUERY_OUTSIDE_DOMAIN", "negative-x"
        elif group == 3:
            bad = (1.0, -0.5 - 0.1 * index)
            code, kind = "QUERY_OUTSIDE_DOMAIN", "negative-y"
        elif group == 4:
            bad = (half_length, half_grip + 0.01 * (index + 1))
            code, kind = "QUERY_OUTSIDE_DOMAIN", "above-grip-corner"
        else:
            bad = ((math.nan, 1.0), (1.0, math.inf), (1.0,), "1,1", (True, 1.0), None)[
                index % 6
            ]
            code, kind = "INVALID_QUERY", "malformed"
        queries = ((0.25 * half_length, 0.25 * half_gauge), bad)
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
    base = _base_points()[0]
    candidates: list[RefusalFixture] = []
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
                    point = theta(base, **{name: value})
                    candidates.append(
                        RefusalFixture(
                            f"guard-{name}-{side}-{label}",
                            "guard-band",
                            base,
                            "PARAMETER_OUT_OF_BOUNDS",
                            theta_override=point,
                            description=f"{name} {side} by one {label}",
                        )
                    )
                else:
                    candidates.append(
                        RefusalFixture(
                            f"guard-{name}-{side}-{label}",
                            "guard-band",
                            {**base, name: value},
                            "PARAMETER_OUT_OF_BOUNDS",
                            description=f"{name} {side} by one {label}",
                        )
                    )
    candidates.extend(_derived_near_misses())
    count = FIXTURE_COUNTS["guard-band"]
    if len(candidates) < count:
        raise RuntimeError("too few guard-band candidates")
    order = ("ulp", "derived", "ppm", "ppb")
    ranked = sorted(
        candidates,
        key=lambda fixture: next(
            rank for rank, tag in enumerate(order) if fixture.fixture_id.endswith(tag)
        ),
    )
    return ranked[:count]


_SOLVED_INPUT = {
    "arc_feasibility_ratio": "R_fillet",
    "fillet_to_gauge_ratio": "R_fillet",
    "grip_to_gauge_ratio": "W_grip",
    "length_to_gauge_ratio": "L_total",
}


def _solve(name: str, target: float, values: Mapping[str, float]) -> float:
    if name == "arc_feasibility_ratio":
        return 0.5 * target * (values["W_grip"] - values["W_gauge"])
    if name == "fillet_to_gauge_ratio":
        return target * values["W_gauge"]
    return target * values["W_gauge"]


def _derived_near_misses() -> list[RefusalFixture]:
    """One point per feasible derived bound side, one part per million outside."""

    bounds = _bounds()
    family = _family()
    grid = [
        {
            "L_total": length,
            "W_grip": grip,
            "W_gauge": gauge,
            "R_fillet": 14.0,
            "nominal_axial_strain": 0.15,
        }
        for length in (40.0, 55.0, 70.0)
        for grip in (16.0, 21.0, 26.0)
        for gauge in (6.0, 10.0, 14.0)
    ]
    fixtures: list[RefusalFixture] = []
    for bound in family.membership.derived_bounds:
        for side, edge in (
            ("below", bound.lower_bound.value * (1.0 - 1e-6)),
            ("above", bound.upper_bound.value * (1.0 + 1e-6)),
        ):
            solved = _SOLVED_INPUT[bound.name]
            for base in grid:
                value = _solve(bound.name, edge, base)
                lower, upper = bounds[solved]
                if not lower <= value <= upper:
                    continue
                values = {**base, solved: value}
                if _derived_violation(values) == (bound.name, side):
                    fixtures.append(
                        RefusalFixture(
                            f"guard-{bound.name}-{side}-derived",
                            "guard-band",
                            values,
                            "DERIVED_QUANTITY_OUT_OF_BOUNDS",
                            description=f"{bound.name} {side} by one ppm",
                        )
                    )
                    break
    return fixtures


@lru_cache(maxsize=1)
def refusal_fixtures() -> tuple[RefusalFixture, ...]:
    fixtures = (
        *_primitive_fixtures(),
        *_derived_fixtures(),
        *_material_fixtures(),
        *_topology_fixtures(),
        *_query_fixtures(),
        *_guard_fixtures(),
    )
    counts = {
        category: sum(fixture.category == category for fixture in fixtures)
        for category in FIXTURE_COUNTS
    }
    if counts != FIXTURE_COUNTS:
        raise RuntimeError(f"fixture counts {counts} differ from ADR 0019")
    if len({fixture.fixture_id for fixture in fixtures}) != len(fixtures):
        raise RuntimeError("fixture ids must be unique")
    return fixtures


def fixture_set_hash(fixtures: tuple[RefusalFixture, ...]) -> str:
    payload = json.dumps(
        [fixture.summary() for fixture in fixtures], sort_keys=True, default=repr
    ).encode()
    return hashlib.sha256(payload).hexdigest()


# ------------------------------------------------------------------ meshes


def _edges(cell: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    a, b, c = cell
    return ((a, b), (b, c), (c, a))


def _rebuild(
    mesh: CanonicalMesh,
    cells: list[tuple[int, int, int]],
    regions: list[int],
    nodes: list[tuple[float, float]],
    tag_of_edge: dict[frozenset[int], str],
    new_tag: str,
) -> CanonicalMesh:
    used = sorted({index for cell in cells for index in cell})
    remap = {old: new for new, old in enumerate(used)}
    counts: dict[frozenset[int], int] = {}
    oriented: dict[frozenset[int], tuple[int, int]] = {}
    for cell in cells:
        for edge in _edges(cell):
            key = frozenset(edge)
            counts[key] = counts.get(key, 0) + 1
            oriented[key] = edge
    facets: list[tuple[int, int]] = []
    tags: dict[str, list[int]] = {}
    for key in sorted(oriented, key=lambda item: sorted(item)):
        if counts[key] != 1:
            continue
        left, right = oriented[key]
        tags.setdefault(tag_of_edge.get(key, new_tag), []).append(len(facets))
        facets.append((remap[left], remap[right]))
    payload = MeshPayload(
        nodes=tuple(nodes[index] for index in used),
        cells=tuple(tuple(remap[index] for index in cell) for cell in cells),
        cell_type=CellType.TRIANGLE,
        boundary_facets=tuple(facets),
        boundary_tags={name: tuple(ids) for name, ids in tags.items()},
        region_ids=tuple(regions),
        unit=mesh.unit,
        source_provenance={"adapter": "s14-refusal-fixture"},
    )
    return canonicalize_mesh(payload)


def mutated_mesh(mesh: CanonicalMesh, mutation: str) -> CanonicalMesh:
    """Apply one named topology or tag mutation to a valid morphed mesh."""

    kind, index_text = mutation.split(":")
    index = int(index_text)
    nodes = [tuple(node) for node in mesh.nodes]
    cells = [tuple(cell) for cell in mesh.cells]
    regions = list(mesh.region_ids)
    tag_of_edge = {
        frozenset(mesh.boundary_facets[facet]): name
        for name, facets in mesh.semantic_tags.items()
        for facet in facets
    }
    if kind == "renamed-tag":
        renamed = {
            key: ("loaded_end" if name == "right_grip" else name)
            for key, name in tag_of_edge.items()
        }
        if index % 2:
            renamed = {
                key: ("bottom" if name == "gauge_top" else name)
                for key, name in tag_of_edge.items()
            }
        return _rebuild(mesh, cells, regions, nodes, renamed, "untagged")  # type: ignore[arg-type]
    if kind == "punched-hole":
        boundary_nodes = {node for facet in mesh.boundary_facets for node in facet}
        near_boundary = {
            node for cell in cells if boundary_nodes.intersection(cell) for node in cell
        }
        interior = [i for i in range(len(nodes)) if i not in near_boundary]
        centre = interior[(index * 37) % len(interior)]
        keep = [cell for cell in cells if centre not in cell]
        keep_regions = [
            region
            for cell, region in zip(cells, regions, strict=True)
            if centre not in cell
        ]
        return _rebuild(mesh, keep, keep_regions, nodes, tag_of_edge, "void")  # type: ignore[arg-type]
    if kind == "extra-component":
        x = max(node[0] for node in nodes) + 2.0 + index
        start = len(nodes)
        nodes += [(x, 0.0), (x + 1.0, 0.0), (x, 1.0)]
        cells.append((start, start + 1, start + 2))
        regions.append(regions[0])
        extra = {
            frozenset((start, start + 1)): "bottom",
            frozenset((start + 1, start + 2)): "bottom",
            frozenset((start + 2, start)): "bottom",
        }
        return _rebuild(
            mesh,
            cells,
            regions,
            nodes,
            {**tag_of_edge, **extra},
            "untagged",  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown mesh mutation {mutation!r}")


def check_fixture(fixture: RefusalFixture) -> RefusalDecision:
    """Run one fixture through the DogBone checker."""

    checker = dogbone_checker()
    request = fixture.request_theta()
    mesh = None
    if fixture.mesh_mutation is not None:
        outcome = _morph(request)
        if outcome.mesh is None:
            raise RuntimeError(f"{fixture.fixture_id} base geometry is not valid")
        mesh = mutated_mesh(outcome.mesh, fixture.mesh_mutation)
    return checker.check(request, mesh=mesh, queries=fixture.queries)


# ------------------------------------------------------------------ in-domain


def in_domain_cases() -> tuple[InDomainCase, ...]:
    """344 active public cases plus the 500 accepted public sweep shapes."""

    manifest = json.loads(_MANIFEST_PATH.read_text("utf-8"))
    cases = [
        InDomainCase(
            case["case_id"],
            {name: float(case["primitive_coordinates"][name]) for name in INPUTS},
        )
        for case in manifest["cases"]
        if case["status"] == "ACTIVE"
    ]
    family = _family()
    sweep: list[InDomainCase] = []
    for values in _samples(900):
        point = parameter_point(*values)
        if not family.membership.evaluate(point).accepted:
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
    if len(cases) != 344 or len(sweep) != 500:
        raise RuntimeError("in-domain population differs from ADR 0019")
    return (*cases, *sweep)


def in_domain_queries(mesh: CanonicalMesh) -> tuple[tuple[float, float], ...]:
    """16 mesh vertices and 16 cell centroids spread over the mesh."""

    nodes = mesh.nodes
    stride = max(1, len(nodes) // 16)
    vertices = [tuple(nodes[i]) for i in range(0, len(nodes), stride)][:16]
    cell_stride = max(1, len(mesh.cells) // 16)
    centroids = [
        (
            sum(nodes[i][0] for i in mesh.cells[c]) / 3.0,
            sum(nodes[i][1] for i in mesh.cells[c]) / 3.0,
        )
        for c in range(0, len(mesh.cells), cell_stride)
    ][:16]
    return tuple((*vertices, *centroids))


def check_in_domain(case: InDomainCase) -> RefusalDecision:
    request = theta(case.values)
    outcome = _morph(request)
    if outcome.mesh is None:
        return RefusalDecision(False, outcome.reason_codes or ("INVALID_MORPHED_MESH",))
    return dogbone_checker().check(
        request, mesh=outcome.mesh, queries=in_domain_queries(outcome.mesh)
    )


__all__ = (
    "FIXTURE_COUNTS",
    "InDomainCase",
    "RefusalFixture",
    "check_fixture",
    "check_in_domain",
    "dogbone_checker",
    "dogbone_envelope",
    "fixture_set_hash",
    "in_domain_cases",
    "in_domain_queries",
    "mutated_mesh",
    "refusal_fixtures",
    "theta",
)
