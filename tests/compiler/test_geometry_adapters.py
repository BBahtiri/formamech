"""Installed family builders preserve the pinned meshes and refusal policy."""

from __future__ import annotations

import dataclasses
import hashlib
import math

import pytest

from piginot.contracts import ProblemSpec, Unit, canonical_json_bytes
from piginot.doe import EvidenceRole, PublishedCase
from piginot.doe.publication import PublishedCaseStatus
from piginot_campaigns.adapters import (
    DOGBONE,
    NOTCHED,
    PublicAdapter,
    load_geometry_templates,
    load_problem,
)

REFERENCE = {
    "dogbone-v1": {"L_total": 54.0, "W_grip": 20.0, "W_gauge": 10.0, "R_fillet": 12.0},
    "notched-v1": {"width": 23.0, "notch_radius": 5.0, "notch_depth": 3.0},
}


def values(spec):
    return {
        **{p.name: (p.lo + p.hi) / 2 for p in spec.parameters},
        **REFERENCE[spec.name],
    }


@pytest.mark.parametrize("adapter", (DOGBONE, NOTCHED))
def test_reference_template_and_morph_match_packaged_identities(adapter):
    spec = load_problem(adapter.adapter_id)
    mesh, morph = adapter.export_geometry(spec)
    assert mesh.mesh_id == spec.geometry.reference_mesh_id
    assert (
        hashlib.sha256(canonical_json_bytes(morph)).hexdigest()
        == spec.geometry.morph_map_sha256
    )
    spec.check_mesh(mesh)
    assert set(mesh.semantic_tags) == set(spec.geometry.regions)
    expected_columns = 56 if adapter is DOGBONE else 48
    assert len(mesh.nodes) == (expected_columns + 1) * 17
    assert len(mesh.cells) == expected_columns * 16 * 2


@pytest.mark.parametrize("adapter", (DOGBONE, NOTCHED))
@pytest.mark.parametrize("level", (1, 2, 4))
def test_geometry_refinement_is_deterministic_nested_and_preserves_tags(adapter, level):
    spec = load_problem(adapter.adapter_id)
    point = values(spec)
    mesh = adapter.geometry_values(spec, point, level=level)
    again = adapter.geometry_values(spec, point, level=level)
    assert mesh == again
    assert set(mesh.semantic_tags) == set(spec.geometry.regions)
    columns = 56 if adapter is DOGBONE else 48
    rows = 16 * level
    assert len(mesh.nodes) == (columns * level + 1) * (rows + 1)
    assert len(mesh.cells) == columns * level * rows * 2
    assert mesh.quality_metrics["minimum_angle_degrees"] >= 5.0
    assert mesh.quality_metrics["maximum_aspect_ratio"] <= 20.0
    coarse = adapter.geometry_values(spec, point)
    mapped = [
        mesh.nodes[level * column * (rows + 1) + level * row]
        for column in range(columns + 1)
        for row in range(17)
    ]
    assert mapped == list(coarse.nodes)
    for tag, facets in coarse.semantic_tags.items():
        assert len(mesh.semantic_tags[tag]) == level * len(facets)


@pytest.mark.parametrize("adapter", (DOGBONE, NOTCHED))
def test_published_case_wrapper_uses_primitive_coordinates(adapter):
    spec = load_problem(adapter.adapter_id)
    point = values(spec)
    case = PublishedCase(
        case_id="candidate-000001",
        leakage_group_id="group",
        role=EvidenceRole.TRAIN,
        primitive_coordinates=point,
        derived_coordinates={"test_ratio": 1.0},
        strata={"nominal_axial_strain": 0},
        status=PublishedCaseStatus.ACTIVE,
    )
    assert adapter.geometry(spec, case) == adapter.geometry_values(spec, point)


@pytest.mark.parametrize("adapter", (DOGBONE, NOTCHED))
@pytest.mark.parametrize("bad", (math.nan, math.inf, True, "20", -1.0))
def test_invalid_geometry_parameter_refuses(adapter, bad):
    spec = load_problem(adapter.adapter_id)
    point = values(spec)
    name = next(iter(REFERENCE[adapter.adapter_id]))
    with pytest.raises(ValueError, match="parameter"):
        adapter.geometry_values(spec, {**point, name: bad})


@pytest.mark.parametrize("level", (0, 3, -1, 1.0, True))
def test_undeclared_refinement_refuses(level):
    spec = load_problem(DOGBONE.adapter_id)
    with pytest.raises(ValueError, match="level"):
        DOGBONE.geometry_values(spec, values(spec), level=level)


@pytest.mark.parametrize("field", ("reference_mesh_id", "morph_map_sha256"))
def test_unregistered_geometry_identity_refuses(field):
    spec = load_problem(DOGBONE.adapter_id)
    changed = dataclasses.replace(
        spec, geometry=dataclasses.replace(spec.geometry, **{field: "a" * 64})
    )
    with pytest.raises(ValueError, match="registered"):
        DOGBONE.geometry_values(changed, values(spec))


def test_spec_boundary_check_runs_before_morphing():
    spec = load_problem(DOGBONE.adapter_id)
    condition = dataclasses.replace(spec.boundary_conditions[0], region="absent")
    changed = dataclasses.replace(
        spec,
        geometry=dataclasses.replace(
            spec.geometry, regions=(*spec.geometry.regions, "absent")
        ),
        boundary_conditions=(condition, *spec.boundary_conditions[1:]),
    )
    with pytest.raises(ValueError, match="absent"):
        DOGBONE.geometry_values(changed, values(spec))


def test_changed_notched_fixed_length_refuses():
    spec = load_problem(NOTCHED.adapter_id)
    changed = dataclasses.replace(
        spec, constants=(dataclasses.replace(spec.constants[0], value=26.0),)
    )
    with pytest.raises(ValueError, match="half_length"):
        NOTCHED.geometry_values(changed, values(spec))


def test_morph_quality_rejection_is_preserved_without_remeshing():
    spec = load_problem(DOGBONE.adapter_id)
    point = {
        **values(spec),
        "L_total": 40.91710672379104,
        "R_fillet": 9.88670386568316,
        "W_gauge": 6.4030515483433605,
        "W_grip": 20.880870557542075,
    }
    assert (
        DOGBONE.geometry_values(spec, point).quality_metrics["minimum_angle_degrees"]
        >= 5
    )
    with pytest.raises(ValueError, match="MESH_QUALITY_OUT_OF_BOUNDS"):
        DOGBONE.geometry_values(spec, point, level=2)


def test_geometry_does_not_require_neo_hookean_material_parameter_names():
    document = load_problem(DOGBONE.adapter_id).document()
    document["parameters"] = [
        p for p in document["parameters"] if p["role"] != "material"
    ] + [
        {
            "name": n,
            "role": "material",
            "lo": v,
            "hi": v,
            "unit": "MPa",
            "distribution": "fixed",
        }
        for n, v in (("c10", 120), ("c01", 20), ("bulk", 500))
    ]
    document["material"] = {
        "model": "mooney_rivlin",
        "parameters": {
            "c10": "c10",
            "c01": "c01",
            "bulk_modulus": "bulk",
        },
    }
    spec = ProblemSpec.from_document(document)
    mesh = DOGBONE.geometry_values(spec, values(spec))
    assert len(mesh.nodes) == 969


def test_geometry_code_bytes_are_bound_by_adapter_implementation_hash(monkeypatch):
    from piginot_campaigns import adapters

    before = adapters.implementation_sha256()
    original = adapters._read
    monkeypatch.setattr(
        adapters,
        "_read",
        lambda name: original(name) + (b" " if name == "geometry.py" else b""),
    )
    assert adapters.implementation_sha256() != before


@pytest.mark.parametrize("adapter", (DOGBONE, NOTCHED))
@pytest.mark.parametrize("level", (1, 2, 4))
def test_tiny_templates_are_registered_fixed_and_refinable(adapter, level):
    original = load_problem(adapter.adapter_id)
    standard, tiny = load_geometry_templates(adapter.adapter_id)
    assert standard.identity.reference_mesh_id == original.geometry.reference_mesh_id
    assert tiny.name == "qualification-tiny"
    spec = dataclasses.replace(
        original,
        geometry=dataclasses.replace(
            original.geometry, reference_mesh_id=tiny.identity.reference_mesh_id
        ),
    )
    reference, _ = adapter.export_geometry(spec)
    assert reference.mesh_id == tiny.identity.reference_mesh_id
    mesh = adapter.geometry_values(spec, values(spec), level)
    columns = sum(tiny.resolution[:2]) * level
    rows = tiny.resolution[2] * level
    assert len(mesh.nodes) == (columns + 1) * (rows + 1)
    assert len(mesh.cells) == columns * rows * 2
    assert set(mesh.semantic_tags) == set(original.geometry.regions)
    assert mesh.quality_metrics["minimum_angle_degrees"] >= 5.0
    assert mesh.quality_metrics["maximum_aspect_ratio"] <= 20.0
    for tag, facets in reference.semantic_tags.items():
        assert len(mesh.semantic_tags[tag]) == level * len(facets)


def test_registered_validity_policy_is_exact_and_immutable():
    criteria = DOGBONE.validity_criteria()
    assert criteria == NOTCHED.validity_criteria()
    assert criteria.jacobian_margin == 0.01
    assert criteria.minimum_feature_size_mm == 0.05
    assert criteria.minimum_angle_degrees == 5.0
    assert criteria.maximum_aspect_ratio == 20.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        criteria.minimum_angle_degrees = 0.0


def test_unknown_adapter_refuses():
    with pytest.raises(ValueError, match="unsupported"):
        load_geometry_templates("unknown")
    with pytest.raises(ValueError, match="unsupported"):
        PublicAdapter("unknown").export_geometry(load_problem(DOGBONE.adapter_id))


def test_missing_parameters_and_outside_constraint_refuse():
    spec = load_problem(DOGBONE.adapter_id)
    with pytest.raises(ValueError, match="parameter"):
        DOGBONE.geometry_values(spec, {})
    with pytest.raises(ValueError, match="constraint"):
        DOGBONE.geometry_values(
            spec, {**values(spec), "R_fillet": 8, "W_grip": 26, "W_gauge": 6}
        )


def test_execute_delegates_lazily_without_starting_numerical_work(monkeypatch):
    from piginot.compiler import execution

    plan = object()
    observed = []

    def fake_execute(actual_plan, adapter, *, resume=False):
        observed.append((actual_plan, adapter, resume))
        return 7

    monkeypatch.setattr(execution, "execute_campaign", fake_execute)
    assert DOGBONE.execute(plan, resume=True) == 7
    assert observed == [(plan, DOGBONE, True)]


def test_registered_template_code_identity_is_verified(monkeypatch):
    from piginot_campaigns import geometry

    spec = load_problem(DOGBONE.adapter_id)
    other = NOTCHED.export_geometry(load_problem(NOTCHED.adapter_id))
    monkeypatch.setattr(geometry, "_template", lambda *args: other)
    with pytest.raises(ValueError, match="installed geometry code differs"):
        DOGBONE.export_geometry(spec)


def test_wrong_geometry_vocabulary_refuses():
    spec = load_problem(DOGBONE.adapter_id)
    changed = dataclasses.replace(
        spec,
        parameters=(
            dataclasses.replace(spec.parameters[0], role="load"),
            *spec.parameters[1:],
        ),
    )
    with pytest.raises(ValueError, match="geometry parameter vocabulary"):
        DOGBONE.export_geometry(changed)


def test_undefined_constraint_refuses_explicitly():
    document = load_problem(DOGBONE.adapter_id).document()
    document["constraints"][0]["expression"] = "W_grip / (W_grip - W_grip)"
    spec = ProblemSpec.from_document(document)
    with pytest.raises(ValueError, match="constraint.*cannot be evaluated"):
        DOGBONE.geometry_values(spec, values(spec))


def test_registered_geometry_parameter_units_are_enforced():
    original = load_problem(DOGBONE.adapter_id)
    parameters = tuple(
        dataclasses.replace(p, unit=Unit.ONE) if p.name == "R_fillet" else p
        for p in original.parameters
    )
    spec = dataclasses.replace(original, parameters=parameters, constraints=())
    with pytest.raises(ValueError, match="millimetres"):
        DOGBONE.export_geometry(spec)
