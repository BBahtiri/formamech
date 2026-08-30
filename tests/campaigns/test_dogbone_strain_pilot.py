from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from campaigns.dogbone_v1.fem_pilot import pilot_geometry_cases
from campaigns.dogbone_v1.strain_pilot import (
    LOAD_POLICY_ID,
    NOMINAL_STRAIN_LEVELS,
    STRAIN_PILOT_ID,
    StrainPilotReport,
    build_strain_pilot_requests,
    prescribed_displacement_mm,
    run_strain_pilot,
    write_strain_pilot_report,
)
from piginot.contracts import (
    ExecutionProfile,
    Unit,
    canonical_json_bytes,
    hash_spec,
    strict_json_loads,
)
from piginot.contracts.errors import NonFiniteNumberError
from piginot.fem import FemAttempt, FemConvergenceEntry, FemRequest, FemResult

_SCREENING_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "campaigns"
    / "dogbone-v1"
    / "fem-strain-envelope-pilot.json"
)
_QUALIFICATION_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "campaigns"
    / "dogbone-v1"
    / "fem-strain-envelope-qualification.json"
)


@pytest.fixture(scope="module")
def execution_profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.12.0",
        library_versions={"numpy": "fixture", "scikit-fem": "fixture"},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def test_strain_requests_normalize_displacement_by_model_length(
    execution_profile: ExecutionProfile,
) -> None:
    requests = build_strain_pilot_requests(execution_profile=execution_profile)
    cases = pilot_geometry_cases()

    assert len(requests) == len(cases) * len(NOMINAL_STRAIN_LEVELS) == 24
    assert len({request.recipe_hash for request in requests}) == len(requests)

    for case_index, case in enumerate(cases):
        total_length_mm = case.parameters[0]
        for strain_index, strain in enumerate(NOMINAL_STRAIN_LEVELS):
            request = requests[case_index * len(NOMINAL_STRAIN_LEVELS) + strain_index]
            right = next(
                condition
                for condition in request.boundary_conditions
                if condition.name == "right_grip_displacement"
            )
            assert right.values[0].unit is Unit.MM
            assert right.values[0].value == pytest.approx(
                strain * total_length_mm / 2.0
            )
            assert request.config.element_order == 2
            assert request.config.load_steps == 32
            assert request.execution_profile == execution_profile

    short_low = requests[4 * len(NOMINAL_STRAIN_LEVELS)]
    long_low = requests[5 * len(NOMINAL_STRAIN_LEVELS)]
    short_u = short_low.boundary_conditions[-1].values[0].value
    long_u = long_low.boundary_conditions[-1].values[0].value
    assert short_u == pytest.approx(2.0)
    assert long_u == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("total_length_mm", "nominal_strain", "message"),
    (
        (0.0, 0.1, "total_length_mm"),
        (54.0, 0.0, "nominal_strain"),
        (float("inf"), 0.1, "finite"),
        (54.0, float("nan"), "finite"),
    ),
)
def test_strain_to_displacement_rejects_invalid_inputs(
    total_length_mm: float,
    nominal_strain: float,
    message: str,
) -> None:
    error = NonFiniteNumberError if "finite" in message else ValueError
    with pytest.raises(error, match=message):
        prescribed_displacement_mm(total_length_mm, nominal_strain)


class _NonlinearFakeBackend:
    backend_name = "nonlinear-fixture-backend"

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        request = requests[0]
        right = next(
            condition
            for condition in request.boundary_conditions
            if condition.name == "right_grip_displacement"
        )
        model_length = max(node[0] for node in request.mesh.nodes)
        strain = right.values[0].value / model_length
        reaction = 100.0 * strain * (1.0 - 0.25 * strain)
        gradient = (
            (
                (
                    (1.0 + strain, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            ),
        )
        history = tuple(
            FemConvergenceEntry(step, 1, 1.0e-12, 1.0e-13)
            for step in range(1, request.config.load_steps + 1)
        )
        result = FemResult(
            request_id=request.request_id,
            mesh_id=request.mesh.mesh_id,
            element_order=request.config.element_order,
            displacement_dof_coordinates=((0.0, 0.0), (model_length, 0.0)),
            displacement=((0.0, 0.0), (right.values[0].value, 0.0)),
            quadrature_points=(((0.0, 0.0),),),
            quadrature_weights=((1.0,),),
            first_piola_stress=gradient,
            deformation_gradient=gradient,
            recovered_first_piola_stress=(gradient[0][0],),
            reactions={"right_grip_displacement": (reaction, 0.0)},
            rigid_mode_multipliers=(),
            total_strain_energy=1.0,
            external_work=2.0,
            total_potential_energy=-1.0,
            convergence_history=history,
            qois={},
        )
        return (
            FemAttempt(
                request=request,
                recipe_hash=request.recipe_hash,
                success=True,
                result=result,
                result_hash=hash_spec(result, "content"),
                store_content_hash=None,
                failure_code=None,
                message="succeeded",
            ),
        )


def _clock_for(attempt_count: int) -> Callable[[], float]:
    values = iter(float(index) for index in range(attempt_count * 2))
    return lambda: next(values)


def test_runner_records_admissibility_and_reaction_nonlinearity(
    execution_profile: ExecutionProfile,
) -> None:
    report = run_strain_pilot(
        backend=_NonlinearFakeBackend(),
        execution_profile=execution_profile,
        clock=_clock_for(24),
        runtime_environment={"operating_system": "fixture-os"},
    )

    assert isinstance(report, StrainPilotReport)
    assert report.pilot_id == STRAIN_PILOT_ID
    assert report.load_policy_id == LOAD_POLICY_ID
    assert report.nominal_strain_levels == NOMINAL_STRAIN_LEVELS
    assert report.outcome_summary.attempted_count == 24
    assert report.outcome_summary.success_count == 24
    assert len(report.response_comparisons) == 12
    assert (
        min(record.minimum_deformation_jacobian or 0.0 for record in report.records)
        > 1.0
    )
    assert all(
        comparison.relative_departure_from_linear > 0.0
        for comparison in report.response_comparisons
    )
    assert canonical_json_bytes(report) == canonical_json_bytes(report)


def test_writer_emits_one_canonical_newline_terminated_document(
    tmp_path: Path,
    execution_profile: ExecutionProfile,
) -> None:
    report = run_strain_pilot(
        backend=_NonlinearFakeBackend(),
        execution_profile=execution_profile,
        clock=_clock_for(24),
    )
    target = tmp_path / "pilot.json"

    write_strain_pilot_report(report, target)

    assert target.read_bytes() == canonical_json_bytes(report) + b"\n"


def test_recorded_strain_screening_artifact_is_exact_public_evidence() -> None:
    payload = _SCREENING_ARTIFACT.read_bytes()
    assert payload.endswith(b"\n")
    canonical_payload = payload[:-1]
    document = strict_json_loads(canonical_payload)
    assert isinstance(document, dict)
    assert document["pilot_id"] == "dogbone-strain-envelope-20260830-v1"
    assert document["nominal_strain_levels"] == [0.1, 0.2]
    assert document["solver_config"]["load_steps"] == 16
    assert document["outcome_summary"]["attempted_count"] == 24
    assert document["outcome_summary"]["success_count"] == 21
    assert document["outcome_summary"]["failure_counts"] == {
        "CLOSURE_FAILURE": 2,
        "NEGATIVE_JACOBIAN": 1,
    }
    framed = b'{"domain":"content","payload":' + canonical_payload + b"}"
    digest = hashlib.blake2b(framed, digest_size=32).hexdigest()

    assert digest == (
        "4ff6dba3f311ff949d5cf662b2d68dd358e5b6f2592157677797f6173d0f14d2"
    )


def test_recorded_strain_qualification_is_exact_public_evidence() -> None:
    payload = _QUALIFICATION_ARTIFACT.read_bytes()
    assert payload.endswith(b"\n")
    canonical_payload = payload[:-1]
    document = strict_json_loads(canonical_payload)
    assert isinstance(document, dict)
    assert document["pilot_id"] == "dogbone-strain-envelope-20260830-v2"
    assert document["nominal_strain_levels"] == [0.1, 0.2]
    assert document["solver_config"]["load_steps"] == 32
    assert document["outcome_summary"]["attempted_count"] == 24
    assert document["outcome_summary"]["success_count"] == 23
    assert document["outcome_summary"]["failure_counts"] == {
        "CLOSURE_FAILURE": 1,
    }
    framed = b'{"domain":"content","payload":' + canonical_payload + b"}"
    digest = hashlib.blake2b(framed, digest_size=32).hexdigest()

    assert digest == (
        "fa5f031b761101987d8af642b86e2b893babf4d0e6e9289b27846bd12479f325"
    )
