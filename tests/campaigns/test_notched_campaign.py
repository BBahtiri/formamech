"""Notched campaign building blocks that need no campaign secret or FEM label."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from campaigns.notched_v1 import family, pilot, s10, s13
from piginot.contracts import hash_spec
from piginot.doe import EvidenceRole, PublishedCaseStatus
from piginot.morphing import BlendedNotchParameters

_ROOT = Path(__file__).resolve().parents[2]


def test_the_pilot_population_is_the_box_corners_and_reference_at_both_strains() -> (
    None
):
    cases = pilot.pilot_cases()
    assert len(cases) == 18 and len({case.case_id for case in cases}) == 18
    assert sum(case.sentinel == "difficult" for case in cases) == 2
    for case in cases:
        width, radius, depth, strain = case.parameters
        for name, value in zip(family.INPUTS, case.parameters, strict=True):
            lower, upper = family.BOUNDS[name]
            assert lower <= value <= upper
        assert strain in pilot.STRAIN_LEVELS
    samples = pilot.public_samples(64)
    assert samples == pilot.public_samples(64)
    for row in samples:
        for name, value in zip(family.GEOMETRY, row, strict=True):
            assert family.BOUNDS[name][0] < value < family.BOUNDS[name][1]


def test_the_committed_family_validation_binds_the_runtime_family() -> None:
    document = json.loads((_ROOT / pilot.VALIDATION_PATH).read_text())
    compiled = family.compiled_family()
    assert document["compiled_family_hash"] == hash_spec(compiled, "content")
    assert document["reference_mesh_id"] == compiled.reference_mesh.mesh_id
    assert document["accepted"] == 500 and not document["morph_rejections"]


def test_s10_queries_lie_inside_every_family_member_and_the_quota_tag_exists() -> None:
    for width, radius, depth in itertools.product(
        *(family.BOUNDS[name] for name in family.GEOMETRY)
    ):
        shape = BlendedNotchParameters(25.0, width, radius, depth)
        for x, y in s10.QUERY_POINTS:
            assert 0.0 < x < 25.0 and 0.0 < y < shape.profile_height_mm(x)
    tags = family.compiled_family().reference_mesh.semantic_tags
    (quota,) = s10.DATASET_POLICY.qoi_quotas
    assert quota.semantic_tag in tags
    assert sum(s10.ROLE_COUNTS.values()) == 200


def _case(index: int, role: EvidenceRole = EvidenceRole.TRAIN) -> SimpleNamespace:
    return SimpleNamespace(
        case_id=f"case-{index:03d}",
        role=role,
        status=PublishedCaseStatus.ACTIVE,
        strata={"net_section_factor": index % 4, "notch_sharpness": (index // 4) % 3},
        primitive_coordinates={
            "width": 16.0 + index % 7,
            "notch_radius": 2.0 + index % 5,
            "notch_depth": 1.0 + index % 4,
            "nominal_axial_strain": 0.1 + 0.001 * index,
        },
    )


def test_the_train_split_is_96_fit_and_32_screen_stratified_and_keyed() -> None:
    cases = tuple(_case(index) for index in range(128))
    fit, screen = s13.split_train_cases(cases)  # type: ignore[arg-type]
    assert len(fit) == 96 and len(screen) == 32 and not set(fit) & set(screen)
    assert (fit, screen) == s13.split_train_cases(tuple(reversed(cases)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="96/32"):
        s13.split_train_cases(cases[:100])  # type: ignore[arg-type]


def test_normalization_is_fitted_on_train_cases_only() -> None:
    cases = tuple(_case(index) for index in range(8))
    statistics = s13.fit_normalization(cases)  # type: ignore[arg-type]
    assert statistics.parameter_names == s13.CONDITIONING
    vector = statistics.vector(cases[0].primitive_coordinates)
    assert vector.shape == (4,) and torch.isfinite(vector).all()
    with pytest.raises(PermissionError):
        s13.fit_normalization((*cases, _case(9, EvidenceRole.DEVELOPMENT)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        replace(statistics, standard_deviation=(0.0, 1.0, 1.0, 1.0))


def test_the_sampler_visits_every_fit_case_once_per_epoch() -> None:
    protocol = SimpleNamespace(
        fit_case_ids=tuple(f"case-{index:03d}" for index in range(96)),
        search_space=s13.search_space(),
        sampler_seed=s13.SAMPLER_SEED,
        sampler_policy_id=s13.SAMPLER_POLICY_ID,
    )
    plan = s13.sampler_plan(protocol)  # type: ignore[arg-type]
    assert len(plan.batches) == 2400
    first_epoch = sorted(index for batch in plan.batches[:24] for index in batch)
    assert first_epoch == list(range(96))


def test_the_geometry_case_lifting_meets_the_quarter_model_conditions() -> None:
    reference = next(
        case for case in pilot.pilot_cases() if case.sentinel == "difficult"
    )
    request = pilot.pilot_request(reference, pilot.execution_profile())
    width, _, _, strain = reference.parameters
    normalization = s13.ConditioningNormalization(
        s13.CONDITIONING, (23.0, 5.0, 3.0, 0.15), (4.0, 1.7, 1.2, 0.03), ("a",)
    )
    coordinates = dict(zip(family.INPUTS, reference.parameters, strict=True))
    case = s13.geometry_case(
        request.request_id,
        "group",
        coordinates,
        request,
        normalization,
        dtype=torch.float64,
    )
    assert (
        case.half_width_mm == width / 2
        and case.prescribed_displacement_mm == strain * 25
    )
    model = s13.build_initial_operator().to(torch.float64)
    edges = torch.tensor(
        [[0.0, 1.0], [0.0, 2.5], [10.0, 0.0], [25.0, 0.0], [25.0, 7.9]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        u = model.displacement(
            query_xy=edges.unsqueeze(0),
            parameters=case.parameters(),
            frame=case.frame(),
            lifting=case.lifting(),
        )[0]
    assert abs(float(u[0, 0])) < 1e-12 and abs(float(u[1, 0])) < 1e-12
    assert abs(float(u[2, 1])) < 1e-12 and abs(float(u[3, 1])) < 1e-12
    assert float(u[3, 0]) == pytest.approx(strain * 25, abs=1e-12)
    assert float(u[4, 0]) == pytest.approx(strain * 25, abs=1e-12)
    terms = s13.case_energy(
        model.to(torch.float32),
        s13.geometry_case(
            request.request_id,
            "g",
            coordinates,
            request,
            normalization,
            dtype=torch.float32,
        ),
        s13.energy_config(),
        generator=torch.Generator().manual_seed(0),
        create_graph=False,
    )
    assert torch.isfinite(terms.total)
    with pytest.raises(ValueError, match="strain"):
        s13.geometry_case(
            request.request_id,
            "g",
            coordinates | {"nominal_axial_strain": 0.11},
            request,
            normalization,
            dtype=torch.float64,
        )
    with pytest.raises(ValueError, match="extent"):
        s13.geometry_case(
            request.request_id,
            "g",
            coordinates | {"width": width + 1.0},
            request,
            normalization,
            dtype=torch.float64,
        )
    broken = replace(request, boundary_conditions=request.boundary_conditions[:2])
    with pytest.raises(ValueError, match="boundary conditions"):
        s13.loaded_end_displacement(broken)
