from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from campaigns.dogbone_v1.family import compiled_family, parameter_point
from campaigns.dogbone_v1.fem_pilot import (
    FemPilotRecord,
    FemPilotReport,
    FemPilotSummary,
    PilotGeometryCase,
    build_pilot_requests,
    pilot_geometry_cases,
    run_fem_pilot,
)
from piginot.contracts import (
    ExecutionProfile,
    Unit,
    canonical_json_bytes,
    hash_spec,
    strict_json_loads,
)
from piginot.contracts.errors import NonFiniteNumberError
from piginot.fem import (
    FemAttempt,
    FemConvergenceEntry,
    FemFailureCode,
    FemRequest,
    FemResult,
)
from piginot.morphing import apply_morph

_EXPECTED_CASES = {
    "reference": (54.0, 20.0, 10.0, 12.0),
    "validation_worst_angle": (
        68.15490487612537,
        22.834462189194603,
        6.10723695220005,
        11.549413934481429,
    ),
    "validation_worst_jacobian": (
        42.45915219995253,
        24.61720261327944,
        6.381824787250642,
        19.47941478408407,
    ),
    "validation_worst_feature": (
        40.62180736726733,
        25.363441376354437,
        7.337443874975097,
        18.042085735135643,
    ),
    "wide_gauge_short": (40.0, 16.0, 14.0, 20.0),
    "narrow_gauge_long": (70.0, 26.0, 6.0, 20.0),
    "narrow_gauge_short": (40.0, 26.0, 6.0, 20.0),
    "wide_gauge_long": (70.0, 16.0, 14.0, 8.0),
    "low_dimensions": (40.0, 16.0, 6.0, 8.0),
    "arc_margin": (70.0, 26.0, 14.0, 8.11),
    "high_concentration_long": (70.0, 26.0, 6.0, 14.0),
    "high_concentration_short": (40.0, 24.0, 6.0, 14.0),
}
_PILOT_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "campaigns"
    / "dogbone-v1"
    / "fem-solvability-pilot.json"
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


@pytest.fixture(scope="module")
def requests(execution_profile: ExecutionProfile) -> tuple[FemRequest, ...]:
    return build_pilot_requests(execution_profile=execution_profile)


def test_public_geometry_selection_is_fixed_deterministic_and_valid() -> None:
    first = pilot_geometry_cases()
    second = pilot_geometry_cases()

    assert len(first) == 12
    assert first == second
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert {case.case_id: case.parameters for case in first} == _EXPECTED_CASES
    assert first[0].case_id == "reference"
    assert first[0].sentinel == "reference"
    assert {case.case_id for case in first if case.sentinel == "difficult"}.issuperset(
        {
            "validation_worst_angle",
            "validation_worst_jacobian",
            "validation_worst_feature",
            "arc_margin",
            "high_concentration_long",
            "high_concentration_short",
        }
    )

    family = compiled_family()
    for case in first:
        theta = parameter_point(*case.parameters)
        assert family.membership.evaluate(theta).accepted, case.case_id
        morph = apply_morph(
            family.reference_mesh,
            family.morph_map,
            family.membership,
            theta,
        )
        assert morph.accepted, case.case_id
        assert morph.mesh is not None


def test_requests_pin_source_material_loading_and_solver_recipe(
    requests: tuple[FemRequest, ...],
    execution_profile: ExecutionProfile,
) -> None:
    assert len(requests) == 12

    for request in requests:
        material = request.material
        assert material.model == "finite_strain_plane_stress_neo_hookean"
        assert material.properties["elastic_modulus"].value == 760.0
        assert material.properties["elastic_modulus"].unit is Unit.MPA
        assert material.properties["poisson_ratio"].value == 0.23
        assert material.properties["poisson_ratio"].unit is Unit.ONE

        conditions = {
            condition.name: condition for condition in request.boundary_conditions
        }
        left = conditions["left_x_symmetry"]
        assert (left.boundary_tag, left.kind, left.components, left.values) == (
            "left_symmetry",
            "symmetry",
            ("x",),
            (),
        )
        bottom = conditions["bottom_y_symmetry"]
        assert (bottom.boundary_tag, bottom.kind, bottom.components, bottom.values) == (
            "bottom",
            "symmetry",
            ("y",),
            (),
        )
        right = conditions["right_grip_displacement"]
        assert (right.boundary_tag, right.kind, right.components) == (
            "right_grip",
            "dirichlet",
            ("x",),
        )
        assert len(right.values) == 1
        assert right.values[0].value == 1.0
        assert right.values[0].unit is Unit.MM

        assert request.loads == ()
        assert request.config.element_order == 2
        assert request.config.load_steps == 8
        assert request.execution_profile == execution_profile


def test_recipe_hashes_are_unique_and_bind_each_morphed_mesh(
    requests: tuple[FemRequest, ...],
) -> None:
    cases = pilot_geometry_cases()
    family = compiled_family()
    recipe_hashes = tuple(request.recipe_hash for request in requests)

    assert len(set(recipe_hashes)) == len(requests)
    assert len({request.mesh.mesh_id for request in requests}) == len(requests)

    for index, (case, request) in enumerate(zip(cases, requests, strict=True)):
        morph = apply_morph(
            family.reference_mesh,
            family.morph_map,
            family.membership,
            parameter_point(*case.parameters),
        )
        assert morph.mesh is not None
        assert request.mesh.mesh_id == morph.mesh.mesh_id
        assert request.recipe_hash == hash_spec(request, "recipe")

        other_mesh = requests[(index + 1) % len(requests)].mesh
        assert replace(request, mesh=other_mesh).recipe_hash != request.recipe_hash


def _result(
    request: FemRequest,
    *,
    scalar_dof_count: int,
    iterations_per_step: int,
) -> FemResult:
    history = tuple(
        FemConvergenceEntry(
            load_step=load_step,
            iteration=iteration,
            residual_norm=1.0e-6 / (load_step * iteration),
            increment_norm=1.0e-5 / (load_step * iteration),
        )
        for load_step in range(1, request.config.load_steps + 1)
        for iteration in range(1, iterations_per_step + 1)
    )
    coordinates = tuple((float(index), 0.0) for index in range(scalar_dof_count))
    displacement = tuple((0.01 * index, 0.0) for index in range(scalar_dof_count))
    return FemResult(
        request_id=request.request_id,
        mesh_id=request.mesh.mesh_id,
        element_order=request.config.element_order,
        displacement_dof_coordinates=coordinates,
        displacement=displacement,
        quadrature_points=(),
        quadrature_weights=(),
        first_piola_stress=(),
        deformation_gradient=(),
        recovered_first_piola_stress=(),
        reactions={},
        rigid_mode_multipliers=(),
        total_strain_energy=1.0,
        external_work=2.0,
        total_potential_energy=-1.0,
        convergence_history=history,
        qois={},
    )


def _success_attempt(
    request: FemRequest,
    *,
    scalar_dof_count: int,
    iterations_per_step: int,
    result_hash: str | None = None,
) -> FemAttempt:
    result = _result(
        request,
        scalar_dof_count=scalar_dof_count,
        iterations_per_step=iterations_per_step,
    )
    return FemAttempt(
        request=request,
        recipe_hash=request.recipe_hash,
        success=True,
        result=result,
        result_hash=hash_spec(result, "content")
        if result_hash is None
        else result_hash,
        store_content_hash=None,
        failure_code=None,
        message="succeeded",
    )


def _failure_attempt(request: FemRequest, code: FemFailureCode) -> FemAttempt:
    return FemAttempt(
        request=request,
        recipe_hash=request.recipe_hash,
        success=False,
        result=None,
        result_hash=None,
        store_content_hash=None,
        failure_code=code,
        message=f"fixture failure: {code.value}",
    )


class _FakeBackend:
    backend_name = "fixture-fem-backend"

    def __init__(self) -> None:
        self.calls: list[tuple[FemRequest, ...]] = []
        self._attempt_index = 0

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        self.calls.append(requests)
        attempts: list[FemAttempt] = []
        for request in requests:
            index = self._attempt_index
            self._attempt_index += 1
            if index == 2:
                attempts.append(
                    _failure_attempt(request, FemFailureCode.NEWTON_NON_CONVERGENCE)
                )
            elif index == 8:
                attempts.append(
                    _failure_attempt(request, FemFailureCode.NEGATIVE_JACOBIAN)
                )
            else:
                attempts.append(
                    _success_attempt(
                        request,
                        scalar_dof_count=5 + index,
                        iterations_per_step=1 + index % 3,
                    )
                )
        return tuple(attempts)


def _clock_for(durations: tuple[float, ...]) -> Callable[[], float]:
    values: list[float] = []
    current = 100.0
    for duration in durations:
        values.extend((current, current + duration))
        current += duration + 1.0
    iterator = iter(values)
    return lambda: next(iterator)


def test_runner_preserves_order_failures_timing_hashes_and_diagnostics(
    execution_profile: ExecutionProfile,
) -> None:
    backend = _FakeBackend()
    durations = tuple(float(index) for index in range(1, 13))

    report = run_fem_pilot(
        backend=backend,
        execution_profile=execution_profile,
        clock=_clock_for(durations),
        runtime_environment={
            "operating_system": "fixture-os",
            "processor": "fixture-cpu",
        },
    )

    cases = pilot_geometry_cases()
    assert isinstance(report, FemPilotReport)
    assert len(backend.calls) == 12
    assert all(len(call) == 1 for call in backend.calls)
    assert tuple(record.case_id for record in report.records) == tuple(
        case.case_id for case in cases
    )
    assert tuple(record.elapsed_seconds for record in report.records) == durations
    assert dict(report.runtime_environment) == {
        "operating_system": "fixture-os",
        "processor": "fixture-cpu",
    }

    for index, record in enumerate(report.records):
        request = backend.calls[index][0]
        assert record.request_id == request.request_id
        assert record.recipe_hash == request.recipe_hash
        if index in (2, 8):
            assert not record.success
            assert record.result_hash is None
            assert record.scalar_dof_count is None
            assert record.canonical_result_bytes is None
            assert record.newton_iterations_by_load_step == ()
            assert record.final_residual_norm is None
        else:
            expected_iterations = 1 + index % 3
            assert record.success
            assert record.result_hash is not None
            assert record.scalar_dof_count == 2 * (5 + index)
            assert record.canonical_result_bytes is not None
            assert record.canonical_result_bytes > 0
            assert record.newton_iterations_by_load_step == (expected_iterations,) * 8
            assert record.final_residual_norm == pytest.approx(
                1.0e-6 / (8 * expected_iterations)
            )

    assert report.records[2].failure_code is FemFailureCode.NEWTON_NON_CONVERGENCE
    assert report.records[8].failure_code is FemFailureCode.NEGATIVE_JACOBIAN
    assert report.summary.attempted_count == 12
    assert report.summary.success_count == 10
    assert dict(report.summary.failure_counts) == {
        "NEGATIVE_JACOBIAN": 1,
        "NEWTON_NON_CONVERGENCE": 1,
    }
    assert report.summary.total_runtime_seconds == 78.0
    assert report.summary.median_runtime_seconds == 6.5
    assert report.summary.p95_runtime_seconds == 12.0
    assert report.summary.max_runtime_seconds == 12.0
    assert report.summary.total_result_bytes == sum(
        record.canonical_result_bytes or 0 for record in report.records
    )
    assert report.summary.max_result_bytes == max(
        record.canonical_result_bytes or 0 for record in report.records
    )
    assert report.summary.max_newton_iterations_observed == 3
    assert report.summary.max_final_residual_norm == pytest.approx(1.25e-7)
    assert canonical_json_bytes(report) == canonical_json_bytes(report)


class _TamperedBackend:
    backend_name = "tampered-fem-backend"

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        request = requests[0]
        return (
            _success_attempt(
                request,
                scalar_dof_count=3,
                iterations_per_step=1,
                result_hash="0" * 64,
            ),
        )


def test_runner_rejects_a_tampered_backend_result_hash(
    execution_profile: ExecutionProfile,
) -> None:
    with pytest.raises(ValueError, match="result hash"):
        run_fem_pilot(
            backend=_TamperedBackend(),
            execution_profile=execution_profile,
            clock=_clock_for((1.0,) * 12),
        )


def test_canonical_serialization_refuses_nonfinite_pilot_values(
    execution_profile: ExecutionProfile,
) -> None:
    cases = pilot_geometry_cases()
    with pytest.raises(NonFiniteNumberError):
        tampered_case = replace(cases[0], parameters=(float("nan"), 20.0, 10.0, 12.0))
        canonical_json_bytes(tampered_case)

    report = run_fem_pilot(
        backend=_FakeBackend(),
        execution_profile=execution_profile,
        clock=_clock_for((1.0,) * 12),
    )
    with pytest.raises(NonFiniteNumberError):
        tampered_record = replace(report.records[0], elapsed_seconds=float("inf"))
        canonical_json_bytes(tampered_record)


def test_case_contract_rejects_invalid_sentinel_text() -> None:
    with pytest.raises(ValueError, match="sentinel"):
        PilotGeometryCase(
            case_id="invalid",
            parameters=(54.0, 20.0, 10.0, 12.0),
            sentinel="unknown",
        )


def test_case_record_summary_and_report_contracts_fail_closed(
    execution_profile: ExecutionProfile,
) -> None:
    with pytest.raises(ValueError, match="case_id"):
        PilotGeometryCase("", (54.0, 20.0, 10.0, 12.0), "reference")
    with pytest.raises(ValueError, match="four values"):
        PilotGeometryCase("short", (54.0, 20.0, 10.0), "reference")  # type: ignore[arg-type]

    valid_record = FemPilotRecord(
        case_id="reference",
        parameters=(54.0, 20.0, 10.0, 12.0),
        sentinel="reference",
        request_id="request",
        mesh_id="mesh",
        recipe_hash="a" * 64,
        success=True,
        result_hash="b" * 64,
        failure_code=None,
        message="succeeded",
        elapsed_seconds=1.0,
        scalar_dof_count=1,
        canonical_result_bytes=1,
        newton_iterations_by_load_step=(1,),
        final_residual_norm=0.0,
    )
    with pytest.raises(ValueError, match="scalar DOFs"):
        replace(valid_record, scalar_dof_count=0)
    with pytest.raises(ValueError, match="result bytes"):
        replace(valid_record, canonical_result_bytes=0)
    with pytest.raises(ValueError, match="Newton diagnostics"):
        replace(valid_record, newton_iterations_by_load_step=())
    with pytest.raises(ValueError, match="final residual"):
        replace(valid_record, final_residual_norm=None)
    with pytest.raises(ValueError, match="failed pilot record"):
        replace(
            valid_record, success=False, failure_code=FemFailureCode.CLOSURE_FAILURE
        )

    with pytest.raises(ValueError, match="at least one attempt"):
        FemPilotSummary(
            attempted_count=0,
            success_count=0,
            failure_counts={},
            total_runtime_seconds=0.0,
            median_runtime_seconds=0.0,
            p95_runtime_seconds=0.0,
            max_runtime_seconds=0.0,
            total_result_bytes=0,
            max_result_bytes=0,
            max_newton_iterations_observed=0,
            max_final_residual_norm=None,
        )
    with pytest.raises(ValueError, match="failure counts"):
        FemPilotSummary(
            attempted_count=2,
            success_count=1,
            failure_counts={},
            total_runtime_seconds=1.0,
            median_runtime_seconds=1.0,
            p95_runtime_seconds=1.0,
            max_runtime_seconds=1.0,
            total_result_bytes=1,
            max_result_bytes=1,
            max_newton_iterations_observed=1,
            max_final_residual_norm=0.0,
        )

    report = run_fem_pilot(
        backend=_FakeBackend(),
        execution_profile=execution_profile,
        clock=_clock_for((1.0,) * 12),
        runtime_environment={"operating_system": "fixture-os"},
    )
    with pytest.raises(ValueError, match="runtime environment"):
        replace(report, runtime_environment={"": "fixture-os"})
    with pytest.raises(ValueError, match="population"):
        replace(report, records=report.records[:-1])
    with pytest.raises(ValueError, match="record order"):
        replace(report, records=tuple(reversed(report.records)))
    with pytest.raises(ValueError, match="unsupported FEM pilot schema"):
        replace(report, schema_version="2.0")


class _NoAttemptBackend:
    backend_name = "no-attempt-backend"

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        return ()


class _WrongRecipeBackend:
    backend_name = "wrong-recipe-backend"

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        request = requests[0]
        return (
            replace(
                _failure_attempt(request, FemFailureCode.UNSUPPORTED_REQUEST),
                recipe_hash="0" * 64,
            ),
        )


class _WrongRequestBackend:
    backend_name = "wrong-request-backend"

    def __init__(self, wrong_request: FemRequest) -> None:
        self._wrong_request = wrong_request

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        request = requests[0]
        return (
            FemAttempt(
                request=self._wrong_request,
                recipe_hash=request.recipe_hash,
                success=False,
                result=None,
                result_hash=None,
                store_content_hash=None,
                failure_code=FemFailureCode.UNSUPPORTED_REQUEST,
                message="wrong request",
            ),
        )


class _MalformedSuccessBackend:
    backend_name = "malformed-success-backend"

    def __init__(self, mode: str) -> None:
        self._mode = mode

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        request = requests[0]
        if self._mode == "missing-result":
            return (
                FemAttempt(
                    request=request,
                    recipe_hash=request.recipe_hash,
                    success=True,
                    result=None,
                    result_hash=None,
                    store_content_hash=None,
                    failure_code=None,
                    message="missing result",
                ),
            )
        result = _result(request, scalar_dof_count=3, iterations_per_step=1)
        if self._mode == "wrong-result-request":
            result = replace(result, request_id="different")
        elif self._mode == "wrong-result-mesh":
            result = replace(result, mesh_id="different")
        elif self._mode == "empty-history":
            result = replace(result, convergence_history=())
        return (
            FemAttempt(
                request=request,
                recipe_hash=request.recipe_hash,
                success=True,
                result=result,
                result_hash=hash_spec(result, "content"),
                store_content_hash=None,
                failure_code=None,
                message=self._mode,
            ),
        )


class _MalformedFailureBackend:
    backend_name = "malformed-failure-backend"

    def __init__(self, retain_result: bool) -> None:
        self._retain_result = retain_result

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        request = requests[0]
        result = _result(request, scalar_dof_count=3, iterations_per_step=1)
        return (
            FemAttempt(
                request=request,
                recipe_hash=request.recipe_hash,
                success=False,
                result=result if self._retain_result else None,
                result_hash=None,
                store_content_hash=None,
                failure_code=None
                if not self._retain_result
                else FemFailureCode.UNSUPPORTED_REQUEST,
                message="malformed failure",
            ),
        )


@pytest.mark.parametrize(
    ("backend", "message"),
    (
        (_NoAttemptBackend(), "exactly one attempt"),
        (_WrongRecipeBackend(), "different recipe"),
        (_MalformedSuccessBackend("missing-result"), "omitted its result"),
        (_MalformedSuccessBackend("wrong-result-request"), "request identity"),
        (_MalformedSuccessBackend("wrong-result-mesh"), "mesh identity"),
        (_MalformedSuccessBackend("empty-history"), "convergence history"),
        (_MalformedFailureBackend(retain_result=True), "retained a result"),
        (_MalformedFailureBackend(retain_result=False), "omitted a reason code"),
    ),
)
def test_runner_rejects_backend_protocol_violations(
    backend: object,
    message: str,
    execution_profile: ExecutionProfile,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_fem_pilot(
            backend=backend,  # type: ignore[arg-type]
            execution_profile=execution_profile,
            clock=_clock_for((1.0,) * 12),
        )


def test_runner_rejects_attempts_for_a_different_request(
    requests: tuple[FemRequest, ...],
    execution_profile: ExecutionProfile,
) -> None:
    with pytest.raises(ValueError, match="different request"):
        run_fem_pilot(
            backend=_WrongRequestBackend(requests[1]),
            execution_profile=execution_profile,
            clock=_clock_for((1.0,) * 12),
        )


def test_recorded_fem_pilot_artifact_is_exact_public_evidence() -> None:
    payload = _PILOT_ARTIFACT.read_bytes()
    assert payload.endswith(b"\n")
    canonical_payload = payload[:-1]
    document = strict_json_loads(canonical_payload)
    assert isinstance(document, dict)
    assert document["pilot_id"] == "dogbone-fem-solvability-20260829-v1"
    assert document["compiled_family_hash"] == (
        "a8143a75e62c614d40f52049cfc88c1cda74d38c9bd488cc3bed4c5678d973a4"
    )
    records = document["records"]
    assert isinstance(records, list)
    assert len(records) == 12
    assert all(isinstance(record, dict) and record["success"] for record in records)
    assert {record["scalar_dof_count"] for record in records} == {7_458}
    summary = document["summary"]
    assert isinstance(summary, dict)
    assert summary["attempted_count"] == summary["success_count"] == 12
    assert summary["failure_counts"] == {}
    assert summary["total_result_bytes"] == 52_117_983
    framed = b'{"domain":"content","payload":' + canonical_payload + b"}"
    assert hashlib.blake2b(framed, digest_size=32).hexdigest() == (
        "e2aa1e0e2cd9d20b31043d9c6cb066c138c9b3cb00c7432111ddd38e6f156928"
    )
