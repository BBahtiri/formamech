from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import campaigns.dogbone_v1.s10_production as production
import piginot
from campaigns.dogbone_v1.s10_production import (
    AttemptOrigin,
    ProductionFemRunError,
    ProductionFemRunLedger,
    ProductionFemRunRecord,
    load_production_attempts,
    persist_request_plan,
    run_dogbone_s10_production,
    run_public_fem_requests,
)
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    MaterialSpec,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.datasets import PublicFemRequestPlan, PublicFemRequestPlanEntry
from piginot.doe import EvidenceRole
from piginot.fem import (
    FemAttempt,
    FemConvergenceEntry,
    FemFailureCode,
    FemRequest,
    FemResult,
    FemSolverConfig,
)
from piginot.store import ObjectStore


def _mesh() -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id="triangle-v1",
        nodes=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        cells=((0, 1, 2),),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 0)),
        semantic_tags={"bottom": (0,), "side": (1, 2)},
        region_ids=(1,),
        unit=Unit.MM,
        orientation="ccw",
        topology_signature="triangle-v1",
        quality_metrics={"min_scaled_jacobian": 0.5},
        source_provenance={"fixture": "s10-production-run"},
    )


def _profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.13.0",
        library_versions={"numpy": "2.2.6", "piginot": "0.1.0"},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def _request(case_id: str) -> FemRequest:
    return FemRequest(
        request_id=case_id,
        mesh=_mesh(),
        material=MaterialSpec(
            name="rubber",
            model="compressible_neo_hookean_plane_stress",
            properties={
                "elastic_modulus": CanonicalQuantity(100.0, Unit.MPA),
                "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
            },
        ),
        boundary_conditions=(
            BoundaryConditionSpec(
                "fixed",
                "bottom",
                "dirichlet",
                ("x", "y"),
                (
                    CanonicalQuantity(0.0, Unit.MM),
                    CanonicalQuantity(0.0, Unit.MM),
                ),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=1),
        execution_profile=_profile(),
    )


def _result(request: FemRequest) -> FemResult:
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    stress = ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0))
    return FemResult(
        request_id=request.request_id,
        mesh_id=request.mesh.mesh_id,
        element_order=1,
        displacement_dof_coordinates=request.mesh.nodes,
        displacement=((0.0, 0.0), (0.1, 0.0), (0.0, 0.1)),
        quadrature_points=(((1.0 / 3.0, 1.0 / 3.0),),),
        quadrature_weights=((0.5,),),
        first_piola_stress=((stress,),),
        deformation_gradient=((identity,),),
        recovered_first_piola_stress=(stress, stress, stress),
        reactions={"bottom": (-1.0, 0.0)},
        rigid_mode_multipliers=(),
        total_strain_energy=0.5,
        external_work=1.0,
        total_potential_energy=-0.5,
        convergence_history=(
            FemConvergenceEntry(
                load_step=1,
                iteration=1,
                residual_norm=0.0,
                increment_norm=0.0,
            ),
        ),
        qois={},
    )


def _plan(requests: tuple[FemRequest, ...]) -> PublicFemRequestPlan:
    return PublicFemRequestPlan(
        campaign_hash="a" * 64,
        public_case_manifest_hash="b" * 64,
        execution_profile_hash=hash_spec(_profile(), "content"),
        entries=tuple(
            PublicFemRequestPlanEntry(
                case_id=request.request_id,
                leakage_group_id=f"group-{request.request_id}",
                role=EvidenceRole.TRAIN,
                request_recipe_hash=request.recipe_hash,
                request_content_hash=hash_spec(request, "content"),
                mesh_id=request.mesh.mesh_id,
                mesh_content_hash=hash_spec(request.mesh, "content"),
            )
            for request in requests
        ),
    )


class _StoredBackend:
    backend_name = "fixture-stored-backend"

    def __init__(
        self,
        store: ObjectStore,
        *,
        failure_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._store = store
        self._failure_ids = failure_ids
        self.calls: list[str] = []

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        assert len(requests) == 1
        request = requests[0]
        self.calls.append(request.request_id)
        if request.request_id in self._failure_ids:
            return (
                FemAttempt(
                    request=request,
                    recipe_hash=request.recipe_hash,
                    success=False,
                    result=None,
                    result_hash=None,
                    store_content_hash=None,
                    failure_code=FemFailureCode.NEWTON_NON_CONVERGENCE,
                    message="fixture detail that must not enter the ledger",
                ),
            )
        result = _result(request)
        store_hash = self._store.put(
            canonical_json_bytes({"domain": "content", "payload": result}),
            recipe_hash=request.recipe_hash,
        )
        return (
            FemAttempt(
                request=request,
                recipe_hash=request.recipe_hash,
                success=True,
                result=result,
                result_hash=hash_spec(result, "content"),
                store_content_hash=store_hash,
                failure_code=None,
                message="succeeded",
            ),
        )


class _MissingStoreBackend:
    backend_name = "fixture-stored-backend"

    def solve_many(self, requests: tuple[FemRequest, ...]) -> tuple[FemAttempt, ...]:
        request = requests[0]
        result = _result(request)
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


def _clock() -> Callable[[], float]:
    values = iter(float(value) for value in range(100))
    return lambda: next(values)


def test_runner_checkpoints_each_case_and_resumes_without_repeating_work(
    tmp_path: Path,
) -> None:
    requests = (_request("case-1"), _request("case-2"))
    request_map = {request.request_id: request for request in requests}
    plan = _plan(requests)
    store = ObjectStore.create(tmp_path / "store")
    ledger_path = tmp_path / "run-ledger.json"
    first_backend = _StoredBackend(store)

    partial = run_public_fem_requests(
        request_plan=plan,
        requests=request_map,
        backend=first_backend,
        store=store,
        ledger_path=ledger_path,
        source_revision="c" * 40,
        clock=_clock(),
        max_new_attempts=1,
    )

    assert first_backend.calls == ["case-1"]
    assert tuple(record.case_id for record in partial.records) == ("case-1",)
    assert not partial.is_complete
    assert ProductionFemRunLedger.from_bytes(ledger_path.read_bytes()) == partial

    second_backend = _StoredBackend(store)
    completed = run_public_fem_requests(
        request_plan=plan,
        requests=request_map,
        backend=second_backend,
        store=store,
        ledger_path=ledger_path,
        source_revision="c" * 40,
        clock=_clock(),
    )

    assert second_backend.calls == ["case-2"]
    assert completed.is_complete
    attempts = load_production_attempts(
        request_plan=plan,
        requests=request_map,
        ledger=completed,
        store=store,
    )
    assert set(attempts) == {"case-1", "case-2"}
    assert attempts["case-1"].result == _result(requests[0])
    assert not tuple(ledger_path.parent.glob(".tmp-*"))


def test_runner_recovers_a_stored_result_orphaned_before_ledger_commit(
    tmp_path: Path,
) -> None:
    requests = (_request("case-1"), _request("case-2"))
    request_map = {request.request_id: request for request in requests}
    plan = _plan(requests)
    store = ObjectStore.create(tmp_path / "store")
    orphan = _result(requests[0])
    store.put(
        canonical_json_bytes({"domain": "content", "payload": orphan}),
        recipe_hash=requests[0].recipe_hash,
    )
    backend = _StoredBackend(store)

    ledger = run_public_fem_requests(
        request_plan=plan,
        requests=request_map,
        backend=backend,
        store=store,
        ledger_path=tmp_path / "run-ledger.json",
        source_revision="c" * 40,
        clock=_clock(),
    )

    assert backend.calls == ["case-2"]
    assert ledger.records[0].origin is AttemptOrigin.RECOVERED_FROM_STORE
    assert ledger.records[0].elapsed_seconds == 0.0
    assert ledger.records[1].origin is AttemptOrigin.SOLVED


def test_failed_attempt_is_visible_sanitized_and_not_retried(tmp_path: Path) -> None:
    request = _request("case-1")
    requests = {request.request_id: request}
    plan = _plan((request,))
    store = ObjectStore.create(tmp_path / "store")
    backend = _StoredBackend(store, failure_ids=frozenset({"case-1"}))
    ledger_path = tmp_path / "run-ledger.json"

    failed = run_public_fem_requests(
        request_plan=plan,
        requests=requests,
        backend=backend,
        store=store,
        ledger_path=ledger_path,
        source_revision="c" * 40,
        clock=_clock(),
    )

    assert backend.calls == ["case-1"]
    assert failed.records[0].failure_code is FemFailureCode.NEWTON_NON_CONVERGENCE
    assert b"fixture detail" not in ledger_path.read_bytes()
    resume_backend = _StoredBackend(store)
    resumed = run_public_fem_requests(
        request_plan=plan,
        requests=requests,
        backend=resume_backend,
        store=store,
        ledger_path=ledger_path,
        source_revision="c" * 40,
        clock=_clock(),
    )
    assert resumed == failed
    assert resume_backend.calls == []
    attempt = load_production_attempts(
        request_plan=plan,
        requests=requests,
        ledger=resumed,
        store=store,
    )["case-1"]
    assert not attempt.success
    assert attempt.result is None


def test_resume_refuses_changed_source_plan_or_backend(tmp_path: Path) -> None:
    request = _request("case-1")
    requests = {request.request_id: request}
    plan = _plan((request,))
    store = ObjectStore.create(tmp_path / "store")
    ledger_path = tmp_path / "run-ledger.json"
    run_public_fem_requests(
        request_plan=plan,
        requests=requests,
        backend=_StoredBackend(store),
        store=store,
        ledger_path=ledger_path,
        source_revision="c" * 40,
        clock=_clock(),
    )

    with pytest.raises(ProductionFemRunError, match="source revision"):
        run_public_fem_requests(
            request_plan=plan,
            requests=requests,
            backend=_StoredBackend(store),
            store=store,
            ledger_path=ledger_path,
            source_revision="d" * 40,
            clock=_clock(),
        )

    class _OtherBackend(_StoredBackend):
        backend_name = "other-backend"

    with pytest.raises(ProductionFemRunError, match="backend"):
        run_public_fem_requests(
            request_plan=plan,
            requests=requests,
            backend=_OtherBackend(store),
            store=store,
            ledger_path=ledger_path,
            source_revision="c" * 40,
            clock=_clock(),
        )

    changed_request = _request("case-2")
    with pytest.raises(ProductionFemRunError, match="request IDs"):
        run_public_fem_requests(
            request_plan=plan,
            requests={changed_request.request_id: changed_request},
            backend=_StoredBackend(store),
            store=store,
            ledger_path=ledger_path,
            source_revision="c" * 40,
            clock=_clock(),
        )


def test_success_without_object_store_material_is_rejected(tmp_path: Path) -> None:
    request = _request("case-1")
    plan = _plan((request,))
    with pytest.raises(ProductionFemRunError, match="ObjectStore"):
        run_public_fem_requests(
            request_plan=plan,
            requests={request.request_id: request},
            backend=_MissingStoreBackend(),
            store=ObjectStore.create(tmp_path / "store"),
            ledger_path=tmp_path / "run-ledger.json",
            source_revision="c" * 40,
            clock=_clock(),
        )


def test_orphaned_store_payload_must_be_an_exact_canonical_fem_result(
    tmp_path: Path,
) -> None:
    request = _request("case-1")
    plan = _plan((request,))
    store = ObjectStore.create(tmp_path / "store")
    store.put(
        b'{"domain":"content","payload":{"$type":"wrong.FemResult"}}',
        recipe_hash=request.recipe_hash,
    )

    with pytest.raises(ProductionFemRunError, match="wrong type"):
        run_public_fem_requests(
            request_plan=plan,
            requests={request.request_id: request},
            backend=_StoredBackend(store),
            store=store,
            ledger_path=tmp_path / "run-ledger.json",
            source_revision="c" * 40,
            clock=_clock(),
        )


def test_resume_rejects_noncanonical_or_unknown_ledger_fields(tmp_path: Path) -> None:
    request = _request("case-1")
    plan = _plan((request,))
    store = ObjectStore.create(tmp_path / "store")
    path = tmp_path / "run-ledger.json"
    ledger = run_public_fem_requests(
        request_plan=plan,
        requests={request.request_id: request},
        backend=_StoredBackend(store),
        store=store,
        ledger_path=path,
        source_revision="c" * 40,
        clock=_clock(),
    )

    path.write_bytes(ledger.to_bytes() + b"\n")
    with pytest.raises(ProductionFemRunError, match="canonical JSON"):
        run_public_fem_requests(
            request_plan=plan,
            requests={request.request_id: request},
            backend=_StoredBackend(store),
            store=store,
            ledger_path=path,
            source_revision="c" * 40,
            clock=_clock(),
        )

    document = ledger.to_bytes().replace(b'"backend_name"', b'"unexpected_field"', 1)
    path.write_bytes(document)
    with pytest.raises(ProductionFemRunError, match="fields changed"):
        run_public_fem_requests(
            request_plan=plan,
            requests={request.request_id: request},
            backend=_StoredBackend(store),
            store=store,
            ledger_path=path,
            source_revision="c" * 40,
            clock=_clock(),
        )


def test_resume_rejects_stored_success_conflicting_with_recorded_failure(
    tmp_path: Path,
) -> None:
    request = _request("case-1")
    requests = {request.request_id: request}
    plan = _plan((request,))
    store = ObjectStore.create(tmp_path / "store")
    path = tmp_path / "run-ledger.json"
    run_public_fem_requests(
        request_plan=plan,
        requests=requests,
        backend=_StoredBackend(store, failure_ids=frozenset({"case-1"})),
        store=store,
        ledger_path=path,
        source_revision="c" * 40,
        clock=_clock(),
    )
    result = _result(request)
    store.put(
        canonical_json_bytes({"domain": "content", "payload": result}),
        recipe_hash=request.recipe_hash,
    )

    with pytest.raises(ProductionFemRunError, match="conflicts"):
        run_public_fem_requests(
            request_plan=plan,
            requests=requests,
            backend=_StoredBackend(store),
            store=store,
            ledger_path=path,
            source_revision="c" * 40,
            clock=_clock(),
        )


def test_persist_request_plan_is_create_once_and_exact(tmp_path: Path) -> None:
    request = _request("case-1")
    plan = _plan((request,))
    target = tmp_path / "request-plan.json"

    persist_request_plan(plan, target)
    persist_request_plan(plan, target)

    assert target.read_bytes() == plan.to_bytes()
    altered = target.read_bytes() + b"\n"
    target.write_bytes(altered)
    with pytest.raises(ProductionFemRunError, match="different request plan"):
        persist_request_plan(plan, target)


def test_atomic_production_write_fsyncs_parent_directory_when_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Path] = []
    fsynced: list[int] = []
    closed: list[int] = []

    def fake_open(path: Path, flags: int) -> int:
        assert flags == os.O_RDONLY
        opened.append(Path(path))
        return 12345

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "fsync", lambda descriptor: fsynced.append(descriptor))
    monkeypatch.setattr(os, "close", lambda descriptor: closed.append(descriptor))

    production._atomic_write(tmp_path / "ledger.json", b"{}")

    assert opened == [tmp_path]
    assert fsynced[-1] == 12345
    assert closed == [12345]


def test_incomplete_ledger_cannot_be_presented_as_attempt_population(
    tmp_path: Path,
) -> None:
    requests = (_request("case-1"), _request("case-2"))
    request_map = {request.request_id: request for request in requests}
    plan = _plan(requests)
    store = ObjectStore.create(tmp_path / "store")
    ledger = run_public_fem_requests(
        request_plan=plan,
        requests=request_map,
        backend=_StoredBackend(store),
        store=store,
        ledger_path=tmp_path / "run-ledger.json",
        source_revision="c" * 40,
        clock=_clock(),
        max_new_attempts=1,
    )

    with pytest.raises(ProductionFemRunError, match="incomplete"):
        load_production_attempts(
            request_plan=plan,
            requests=request_map,
            ledger=ledger,
            store=store,
        )


@pytest.mark.parametrize("limit", [0, -1, True])
def test_new_attempt_limit_must_be_positive(limit: object, tmp_path: Path) -> None:
    request = _request("case-1")
    with pytest.raises(ProductionFemRunError, match="positive integer"):
        run_public_fem_requests(
            request_plan=_plan((request,)),
            requests={request.request_id: request},
            backend=_StoredBackend(ObjectStore.create(tmp_path / "store")),
            store=ObjectStore.open(tmp_path / "store"),
            ledger_path=tmp_path / "run-ledger.json",
            source_revision="c" * 40,
            clock=_clock(),
            max_new_attempts=limit,  # type: ignore[arg-type]
        )


def test_campaign_wrapper_materializes_the_exact_plan_before_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request("case-1")
    plan = _plan((request,))
    store = ObjectStore.create(tmp_path / "store")
    backend = _StoredBackend(store)
    monkeypatch.setattr(
        production,
        "_load_public_sources",
        lambda: (
            SimpleNamespace(campaign_hash=plan.campaign_hash),
            SimpleNamespace(content_hash=plan.public_case_manifest_hash),
        ),
    )
    monkeypatch.setattr(
        production,
        "build_dogbone_s10_public_requests",
        lambda *, execution_profile: (request,),
    )
    monkeypatch.setattr(
        production,
        "build_dogbone_s10_request_plan",
        lambda *, execution_profile: plan,
    )

    ledger = run_dogbone_s10_production(
        run_root=tmp_path / "run",
        execution_profile=_profile(),
        backend=backend,
        store=store,
        source_revision="c" * 40,
        dry_run=True,
    )

    assert ledger.completed_case_count == 0
    assert (tmp_path / "run" / "request-plan.json").read_bytes() == plan.to_bytes()
    assert backend.calls == []


def test_default_execution_profile_records_every_numerical_dependency() -> None:
    profile = production.default_s10_execution_profile()

    assert profile.blas_threads == 1
    assert profile.device == "cpu"
    assert profile.dtype == "float64"
    assert set(profile.library_versions) == {
        "numpy",
        "piginot",
        "rfc8785",
        "scikit-fem",
        "scipy",
        "torch",
    }


def test_campaign_wrapper_submits_and_checkpoints_a_bounded_smoke_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = (_request("case-1"), _request("case-2"))
    plan = _plan(requests)
    store = ObjectStore.create(tmp_path / "store")
    backend = _StoredBackend(store)
    monkeypatch.setattr(
        production,
        "_load_public_sources",
        lambda: (
            SimpleNamespace(campaign_hash=plan.campaign_hash),
            SimpleNamespace(content_hash=plan.public_case_manifest_hash),
        ),
    )
    monkeypatch.setattr(
        production,
        "build_dogbone_s10_public_requests",
        lambda *, execution_profile: requests,
    )
    monkeypatch.setattr(
        production,
        "build_dogbone_s10_request_plan",
        lambda *, execution_profile: plan,
    )

    ledger = run_dogbone_s10_production(
        run_root=tmp_path / "run",
        execution_profile=_profile(),
        backend=backend,
        store=store,
        source_revision="c" * 40,
        max_new_attempts=1,
    )

    assert backend.calls == ["case-1"]
    assert ledger.completed_case_count == 1


def test_runtime_guards_reject_wrong_checkout_threads_and_dirty_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(piginot, "__file__", __file__)
    with pytest.raises(ProductionFemRunError, match="different checkout"):
        production._assert_runtime_source()

    for name in production._THREAD_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ProductionFemRunError, match="pinned to one"):
        production._assert_single_thread_environment()
    for name in production._THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    production._assert_single_thread_environment()

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=" M tracked.py\n"),
    )
    with pytest.raises(ProductionFemRunError, match="clean committed"):
        production._assert_clean_source_revision()


def test_clean_source_guard_returns_the_exact_git_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        output = "" if command[1] == "status" else f"{'d' * 40}\n"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert production._assert_clean_source_revision() == "d" * 40
    assert calls == [
        ("git", "status", "--porcelain", "--untracked-files=all"),
        ("git", "rev-parse", "HEAD"),
    ]


def test_cli_reports_dry_run_without_submitting_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = ProductionFemRunLedger(
        run_id=production.RUN_ID,
        campaign_hash="a" * 64,
        public_case_manifest_hash="b" * 64,
        request_plan_hash="c" * 64,
        execution_profile_hash="d" * 64,
        source_revision="e" * 40,
        backend_name="fixture",
        planned_case_count=1,
        records=(),
    )
    received: dict[str, object] = {}
    monkeypatch.setattr(production, "_assert_runtime_source", lambda: None)
    monkeypatch.setattr(production, "_assert_single_thread_environment", lambda: None)
    monkeypatch.setattr(
        production, "_assert_clean_source_revision", lambda: ledger.source_revision
    )

    def fake_run(**kwargs: object) -> ProductionFemRunLedger:
        received.update(kwargs)
        return ledger

    monkeypatch.setattr(production, "run_dogbone_s10_production", fake_run)

    assert production.main(("--dry-run", "--run-root", str(tmp_path))) == 0
    output = capsys.readouterr().out
    assert "dry run: no FEM request was submitted" in output
    assert "completed: 0/1" in output
    assert received["dry_run"] is True
    assert received["source_revision"] == ledger.source_revision


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda ledger: replace(ledger, run_id="wrong"), "run_id"),
        (lambda ledger: replace(ledger, campaign_hash="wrong"), "campaign_hash"),
        (lambda ledger: replace(ledger, source_revision="wrong"), "source_revision"),
        (lambda ledger: replace(ledger, backend_name=""), "backend_name"),
        (lambda ledger: replace(ledger, planned_case_count=0), "planned_case_count"),
    ],
)
def test_ledger_rejects_invalid_identity_fields(
    mutate: Callable[[ProductionFemRunLedger], ProductionFemRunLedger], message: str
) -> None:
    ledger = ProductionFemRunLedger(
        run_id=production.RUN_ID,
        campaign_hash="a" * 64,
        public_case_manifest_hash="b" * 64,
        request_plan_hash="c" * 64,
        execution_profile_hash="d" * 64,
        source_revision="e" * 40,
        backend_name="fixture",
        planned_case_count=1,
        records=(),
    )
    with pytest.raises(ProductionFemRunError, match=message):
        mutate(ledger)


def _valid_run_record() -> ProductionFemRunRecord:
    return ProductionFemRunRecord(
        case_id="case-1",
        request_recipe_hash="a" * 64,
        request_content_hash="b" * 64,
        success=True,
        result_hash="c" * 64,
        store_content_hash="d" * 64,
        failure_code=None,
        message="succeeded",
        elapsed_seconds=1.0,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda record: replace(record, case_id=""), "case_id"),
        (lambda record: replace(record, success=cast(bool, 1)), "success"),
        (
            lambda record: replace(record, elapsed_seconds=float("nan")),
            "elapsed_seconds",
        ),
        (
            lambda record: replace(record, origin=cast(AttemptOrigin, "UNKNOWN")),
            "origin",
        ),
        (lambda record: replace(record, result_hash=None), "result_hash"),
        (
            lambda record: replace(
                record,
                failure_code=FemFailureCode.NEWTON_NON_CONVERGENCE,
            ),
            "successful record",
        ),
    ],
)
def test_success_record_rejects_inconsistent_fields(
    mutate: Callable[[ProductionFemRunRecord], ProductionFemRunRecord], message: str
) -> None:
    with pytest.raises(ProductionFemRunError, match=message):
        mutate(_valid_run_record())


@pytest.mark.parametrize(
    "change",
    [
        {"result_hash": "c" * 64},
        {"store_content_hash": "d" * 64},
        {"failure_code": None},
        {"message": "internal solver detail"},
        {"origin": AttemptOrigin.RECOVERED_FROM_STORE},
    ],
)
def test_failure_record_rejects_result_material_or_unsanitized_state(
    change: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "case_id": "case-1",
        "request_recipe_hash": "a" * 64,
        "request_content_hash": "b" * 64,
        "success": False,
        "result_hash": None,
        "store_content_hash": None,
        "failure_code": FemFailureCode.NEWTON_NON_CONVERGENCE,
        "message": "failed:NEWTON_NON_CONVERGENCE",
        "elapsed_seconds": 1.0,
        "origin": AttemptOrigin.SOLVED,
    }
    values.update(change)
    with pytest.raises(ProductionFemRunError, match="failed record"):
        ProductionFemRunRecord(**values)  # type: ignore[arg-type]


def test_ledger_rejects_duplicate_excess_and_wrong_record_types() -> None:
    record = _valid_run_record()
    values: dict[str, object] = {
        "run_id": production.RUN_ID,
        "campaign_hash": "a" * 64,
        "public_case_manifest_hash": "b" * 64,
        "request_plan_hash": "c" * 64,
        "execution_profile_hash": "d" * 64,
        "source_revision": "e" * 40,
        "backend_name": "fixture",
        "planned_case_count": 2,
        "records": (record,),
    }

    with pytest.raises(ProductionFemRunError, match="exact.*values"):
        ProductionFemRunLedger(**{**values, "records": (object(),)})  # type: ignore[arg-type]
    with pytest.raises(ProductionFemRunError, match="unique case IDs"):
        ProductionFemRunLedger(**{**values, "records": (record, record)})  # type: ignore[arg-type]
    with pytest.raises(ProductionFemRunError, match="exceed"):
        ProductionFemRunLedger(
            **{
                **values,
                "planned_case_count": 1,
                "records": (record, replace(record, case_id="case-2")),
            }  # type: ignore[arg-type]
        )


def test_ledger_parser_rejects_non_document_inputs_and_schema_changes() -> None:
    with pytest.raises(ProductionFemRunError, match="text or bytes"):
        ProductionFemRunLedger.from_bytes(123)  # type: ignore[arg-type]

    ledger = ProductionFemRunLedger(
        run_id=production.RUN_ID,
        campaign_hash="a" * 64,
        public_case_manifest_hash="b" * 64,
        request_plan_hash="c" * 64,
        execution_profile_hash="d" * 64,
        source_revision="e" * 40,
        backend_name="fixture",
        planned_case_count=1,
        records=(),
    )
    with pytest.raises(ProductionFemRunError, match="unsupported.*schema"):
        replace(ledger, schema_version="2.0")
