"""Resumable public S10 FEM production runner for the DogBone campaign."""

from __future__ import annotations

import argparse
import math
import os
import platform
import re
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import ClassVar, cast

import piginot
from campaigns.dogbone_v1.s10_requests import (
    _load_public_sources,
    build_dogbone_s10_public_requests,
    build_dogbone_s10_request_plan,
)
from piginot.contracts import ExecutionProfile, hash_spec
from piginot.contracts.canonical import (
    JsonValue,
    canonical_json_bytes,
    strict_json_loads,
)
from piginot.datasets.request_plan import PublicFemRequestPlan
from piginot.datasets.snapshots import validate_successful_fem_attempt
from piginot.fem import (
    FemAttempt,
    FemBackend,
    FemConvergenceEntry,
    FemFailureCode,
    FemRequest,
    FemResult,
    ScikitFemBackend,
)
from piginot.store import ObjectStore

RUN_ID = "dogbone-s10-public-production-v1"
SCHEMA_VERSION = "1.0"
_LEDGER_NAME = "ledger.json"
_PLAN_NAME = "request-plan.json"
_PROFILE_NAME = "execution-profile.json"
_STORE_NAME = "object-store"
_LOCK_NAME = "run.lock"
_LEDGER_TYPE = "piginot.campaigns.ProductionFemRunLedger"
_RECORD_TYPE = "piginot.campaigns.ProductionFemRunRecord"
_RESULT_TYPE = "piginot.contracts.FemResult"
_CONVERGENCE_TYPE = "piginot.contracts.FemConvergenceEntry"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class ProductionFemRunError(RuntimeError):
    """An existing run root cannot be safely resumed for the requested plan."""


class AttemptOrigin(StrEnum):
    """How a durable outcome entered the production ledger."""

    SOLVED = "SOLVED"
    RECOVERED_FROM_STORE = "RECOVERED_FROM_STORE"


@dataclass(frozen=True, slots=True)
class ProductionFemRunRecord:
    """Compact durable outcome for one public active FEM case."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    case_id: str
    request_recipe_hash: str
    request_content_hash: str
    success: bool
    result_hash: str | None
    store_content_hash: str | None
    failure_code: FemFailureCode | None
    message: str
    elapsed_seconds: float
    origin: AttemptOrigin = AttemptOrigin.SOLVED

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ProductionFemRunError("case_id must be non-empty text")
        for label, value in (
            ("request_recipe_hash", self.request_recipe_hash),
            ("request_content_hash", self.request_content_hash),
        ):
            _require_hash(value, label)
        if type(self.success) is not bool:
            raise ProductionFemRunError("success must be boolean")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ProductionFemRunError(
                "elapsed_seconds must be finite and non-negative"
            )
        try:
            origin = AttemptOrigin(self.origin)
        except (TypeError, ValueError) as exc:
            raise ProductionFemRunError("origin must be an AttemptOrigin") from exc
        object.__setattr__(self, "origin", origin)
        if self.success:
            _require_hash(self.result_hash, "result_hash")
            _require_hash(self.store_content_hash, "store_content_hash")
            if self.failure_code is not None or self.message != "succeeded":
                raise ProductionFemRunError("successful record has inconsistent fields")
        else:
            if (
                self.result_hash is not None
                or self.store_content_hash is not None
                or not isinstance(self.failure_code, FemFailureCode)
                or self.message != f"failed:{self.failure_code.value}"
                or origin is not AttemptOrigin.SOLVED
            ):
                raise ProductionFemRunError("failed record has inconsistent fields")


@dataclass(frozen=True, slots=True)
class ProductionFemRunLedger:
    """Restart-safe public execution ledger for the finite S10 solve campaign."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    run_id: str
    campaign_hash: str
    public_case_manifest_hash: str
    request_plan_hash: str
    execution_profile_hash: str
    source_revision: str
    backend_name: str
    planned_case_count: int
    records: tuple[ProductionFemRunRecord, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.run_id != RUN_ID:
            raise ProductionFemRunError("unexpected production run_id")
        for label, value in (
            ("campaign_hash", self.campaign_hash),
            ("public_case_manifest_hash", self.public_case_manifest_hash),
            ("request_plan_hash", self.request_plan_hash),
            ("execution_profile_hash", self.execution_profile_hash),
        ):
            _require_hash(value, label)
        if (
            not isinstance(self.source_revision, str)
            or _REVISION.fullmatch(self.source_revision) is None
        ):
            raise ProductionFemRunError(
                "source_revision must be a lowercase 40-character Git revision"
            )
        if not isinstance(self.backend_name, str) or not self.backend_name.strip():
            raise ProductionFemRunError("backend_name must be non-empty text")
        records = tuple(self.records)
        if any(type(record) is not ProductionFemRunRecord for record in records):
            raise ProductionFemRunError(
                "records must contain exact ProductionFemRunRecord values"
            )
        records = tuple(sorted(records, key=lambda record: record.case_id))
        case_ids = tuple(record.case_id for record in records)
        if len(case_ids) != len(set(case_ids)):
            raise ProductionFemRunError(
                "S10 production records must have unique case IDs"
            )
        if type(self.planned_case_count) is not int or self.planned_case_count <= 0:
            raise ProductionFemRunError("planned_case_count must be positive")
        if len(records) > self.planned_case_count:
            raise ProductionFemRunError("completed records exceed planned population")
        if self.schema_version != SCHEMA_VERSION:
            raise ProductionFemRunError(
                f"unsupported S10 production schema {self.schema_version}"
            )
        object.__setattr__(self, "records", records)

    @property
    def completed_case_count(self) -> int:
        """Return the number of durably recorded outcomes."""

        return len(self.records)

    @property
    def success_count(self) -> int:
        """Return the number of successful public FEM solves."""

        return sum(record.success for record in self.records)

    @property
    def failure_count(self) -> int:
        """Return the number of visible public FEM failures."""

        return self.completed_case_count - self.success_count

    @property
    def is_complete(self) -> bool:
        """Return whether every planned public case has a durable outcome."""

        return self.completed_case_count == self.planned_case_count

    def to_bytes(self) -> bytes:
        """Return the exact canonical working-ledger representation."""

        return canonical_json_bytes(self)

    @classmethod
    def from_bytes(cls, document: str | bytes) -> ProductionFemRunLedger:
        """Strictly parse and canonical-verify a production ledger."""

        raw = document.encode("utf-8") if isinstance(document, str) else document
        if not isinstance(raw, bytes):
            raise ProductionFemRunError("ledger document must be text or bytes")
        ledger = _ledger_from_document(raw)
        if ledger.to_bytes() != raw:
            raise ProductionFemRunError("production ledger must use canonical JSON")
        return ledger


def default_s10_execution_profile() -> ExecutionProfile:
    """Return the local deterministic execution profile bound into S10 requests."""

    return ExecutionProfile(
        python_version=platform.python_version(),
        library_versions={
            package: version(package)
            for package in (
                "numpy",
                "piginot",
                "rfc8785",
                "scikit-fem",
                "scipy",
                "torch",
            )
        },
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )


def default_s10_run_root() -> Path:
    """Return the ignored local run directory for large public FEM bytes."""

    return Path(__file__).resolve().parents[2] / ".piginot-runs" / RUN_ID


def run_dogbone_s10_production(
    *,
    run_root: Path | None = None,
    execution_profile: ExecutionProfile | None = None,
    backend: FemBackend | None = None,
    store: ObjectStore | None = None,
    clock: Callable[[], float] = perf_counter,
    max_new_attempts: int | None = None,
    dry_run: bool = False,
    source_revision: str | None = None,
    on_record: Callable[[ProductionFemRunLedger, ProductionFemRunRecord], None]
    | None = None,
) -> ProductionFemRunLedger:
    """Execute or resume the active public DogBone S10 FEM solve campaign."""

    profile = (
        default_s10_execution_profile()
        if execution_profile is None
        else execution_profile
    )
    root = default_s10_run_root() if run_root is None else Path(run_root)
    with _exclusive_run_lock(root / _LOCK_NAME):
        object_store = (
            ObjectStore.create(root / _STORE_NAME) if store is None else store
        )
        solver = ScikitFemBackend(store=object_store) if backend is None else backend
        checkpoint, manifest = _load_public_sources()
        requests = build_dogbone_s10_public_requests(execution_profile=profile)
        request_plan = build_dogbone_s10_request_plan(execution_profile=profile)
        if (
            checkpoint.campaign_hash != request_plan.campaign_hash
            or manifest.content_hash != request_plan.public_case_manifest_hash
        ):
            raise ProductionFemRunError(
                "DogBone public sources do not match the production request plan"
            )
        persist_request_plan(request_plan, root / _PLAN_NAME)
        persist_execution_profile(profile, root / _PROFILE_NAME)
        revision = (
            _git_source_revision() if source_revision is None else source_revision
        )
        if dry_run:
            return _new_ledger(request_plan, solver.backend_name, revision)
        return run_public_fem_requests(
            request_plan=request_plan,
            requests={request.request_id: request for request in requests},
            backend=solver,
            store=object_store,
            ledger_path=root / _LEDGER_NAME,
            source_revision=revision,
            clock=clock,
            max_new_attempts=max_new_attempts,
            on_record=on_record,
        )


def run_public_fem_requests(
    *,
    request_plan: PublicFemRequestPlan,
    requests: Mapping[str, FemRequest],
    backend: FemBackend,
    store: ObjectStore,
    ledger_path: Path,
    source_revision: str,
    clock: Callable[[], float],
    max_new_attempts: int | None = None,
    on_record: Callable[[ProductionFemRunLedger, ProductionFemRunRecord], None]
    | None = None,
) -> ProductionFemRunLedger:
    """Run a plan-bound public FEM campaign with per-case atomic resume."""

    if max_new_attempts is not None and (
        type(max_new_attempts) is not int or max_new_attempts <= 0
    ):
        raise ProductionFemRunError("max_new_attempts must be a positive integer")
    if type(request_plan) is not PublicFemRequestPlan:
        raise ProductionFemRunError("request_plan must be exact")
    if type(store) is not ObjectStore:
        raise ProductionFemRunError("store must be an exact ObjectStore")
    if not isinstance(requests, Mapping):
        raise ProductionFemRunError("requests must be a case-ID mapping")
    requests_by_id = _requests_by_id(tuple(requests.values()))
    if set(requests) != set(requests_by_id) or any(
        requests[key] is not requests_by_id[key] for key in requests
    ):
        raise ProductionFemRunError("request IDs must equal mapping keys")
    _validate_population(request_plan, requests_by_id)
    path = Path(ledger_path)
    ledger = (
        _load_ledger(path)
        if path.exists()
        else _new_ledger(request_plan, backend.backend_name, source_revision)
    )
    _assert_ledger_matches(
        ledger,
        request_plan=request_plan,
        backend_name=backend.backend_name,
        source_revision=source_revision,
    )
    _validate_ledger_records(ledger, request_plan, requests_by_id, store)
    if not path.exists():
        _write_ledger(path, ledger)

    completed = {record.case_id for record in ledger.records}
    solved_count = 0
    for entry in request_plan.entries:
        case_id = entry.case_id
        if case_id in completed:
            continue
        request = requests_by_id[case_id]
        stored = store.recipe_results(request.recipe_hash)
        if stored:
            store_hash = store.resolve_recipe(request.recipe_hash)
            result = _decode_stored_result(request, store.get(store_hash))
            recovered = FemAttempt(
                request=request,
                recipe_hash=request.recipe_hash,
                success=True,
                result=result,
                result_hash=hash_spec(result, "content"),
                store_content_hash=store_hash,
                failure_code=None,
                message="succeeded",
            )
            record = _record_attempt(
                request_plan=request_plan,
                request=request,
                attempt=recovered,
                store=store,
                elapsed_seconds=0.0,
                origin=AttemptOrigin.RECOVERED_FROM_STORE,
            )
            ledger = replace(ledger, records=(*ledger.records, record))
            _write_ledger(path, ledger)
            if on_record is not None:
                on_record(ledger, record)
            completed.add(case_id)
            continue
        if max_new_attempts is not None and solved_count >= max_new_attempts:
            break
        started = clock()
        attempts = backend.solve_many((request,))
        elapsed = clock() - started
        if len(attempts) != 1:
            raise ProductionFemRunError(
                "FEM backend must return exactly one attempt per request"
            )
        record = _record_attempt(
            request_plan=request_plan,
            request=request,
            attempt=attempts[0],
            store=store,
            elapsed_seconds=elapsed,
            origin=AttemptOrigin.SOLVED,
        )
        ledger = replace(ledger, records=(*ledger.records, record))
        _write_ledger(path, ledger)
        solved_count += 1
        completed.add(case_id)
        if on_record is not None:
            on_record(ledger, record)
    return ledger


def load_s10_production_attempts(
    *,
    run_root: Path,
    execution_profile: ExecutionProfile | None = None,
    store: ObjectStore | None = None,
) -> Mapping[str, FemAttempt]:
    """Rehydrate completed public attempts from the ledger and object store."""

    profile = (
        default_s10_execution_profile()
        if execution_profile is None
        else execution_profile
    )
    root = Path(run_root)
    object_store = ObjectStore.open(root / _STORE_NAME) if store is None else store
    checkpoint, manifest = _load_public_sources()
    requests = build_dogbone_s10_public_requests(execution_profile=profile)
    request_plan = build_dogbone_s10_request_plan(execution_profile=profile)
    ledger = _load_ledger(root / _LEDGER_NAME)
    if (
        checkpoint.campaign_hash != request_plan.campaign_hash
        or manifest.content_hash != request_plan.public_case_manifest_hash
    ):
        raise ProductionFemRunError("public sources do not match request plan")
    requests_by_id = _requests_by_id(requests)
    return load_production_attempts(
        request_plan=request_plan,
        requests=requests_by_id,
        ledger=ledger,
        store=object_store,
    )


class _ProductionAttemptMapping(Mapping[str, FemAttempt]):
    """Lazy result rehydration keeps the 344-case population memory-bounded."""

    def __init__(
        self,
        records: Mapping[str, ProductionFemRunRecord],
        requests: Mapping[str, FemRequest],
        store: ObjectStore,
    ) -> None:
        self._records = dict(records)
        self._requests = dict(requests)
        self._store = store

    def __getitem__(self, case_id: str) -> FemAttempt:
        try:
            record = self._records[case_id]
            request = self._requests[case_id]
        except KeyError as exc:
            raise KeyError(case_id) from exc
        return _attempt_from_record(record, request, self._store)

    def __iter__(self) -> Iterator[str]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)


def load_production_attempts(
    *,
    request_plan: PublicFemRequestPlan,
    requests: Mapping[str, FemRequest],
    ledger: ProductionFemRunLedger,
    store: ObjectStore,
) -> Mapping[str, FemAttempt]:
    """Expose a complete ledger as a lazy, store-verified attempt mapping."""

    requests_by_id = _requests_by_id(tuple(requests.values()))
    _validate_population(request_plan, requests_by_id)
    if not ledger.is_complete:
        raise ProductionFemRunError(
            "incomplete production ledger cannot form an attempt population"
        )
    _assert_ledger_matches(
        ledger,
        request_plan=request_plan,
        backend_name=ledger.backend_name,
        source_revision=ledger.source_revision,
    )
    _validate_ledger_records(ledger, request_plan, requests_by_id, store)
    records = {record.case_id: record for record in ledger.records}
    return _ProductionAttemptMapping(records, requests_by_id, store)


def _requests_by_id(requests: tuple[FemRequest, ...]) -> dict[str, FemRequest]:
    if any(type(request) is not FemRequest for request in requests):
        raise ProductionFemRunError("requests must contain exact FemRequest values")
    mapping = {request.request_id: request for request in requests}
    if len(mapping) != len(requests):
        raise ProductionFemRunError(
            "S10 production requests must have unique request IDs"
        )
    return mapping


def _validate_population(
    request_plan: PublicFemRequestPlan,
    requests_by_id: Mapping[str, FemRequest],
) -> None:
    planned = {entry.case_id for entry in request_plan.entries}
    if set(requests_by_id) != planned:
        raise ProductionFemRunError(
            "S10 production request IDs do not match the request plan"
        )


def _new_ledger(
    request_plan: PublicFemRequestPlan,
    backend_name: str,
    source_revision: str,
) -> ProductionFemRunLedger:
    return ProductionFemRunLedger(
        run_id=RUN_ID,
        campaign_hash=request_plan.campaign_hash,
        public_case_manifest_hash=request_plan.public_case_manifest_hash,
        request_plan_hash=request_plan.content_hash,
        execution_profile_hash=request_plan.execution_profile_hash,
        source_revision=source_revision,
        backend_name=backend_name,
        planned_case_count=len(request_plan.entries),
        records=(),
    )


def _assert_ledger_matches(
    ledger: ProductionFemRunLedger,
    *,
    request_plan: PublicFemRequestPlan,
    backend_name: str,
    source_revision: str,
) -> None:
    expected = {
        "campaign": request_plan.campaign_hash,
        "public manifest": request_plan.public_case_manifest_hash,
        "request plan": request_plan.content_hash,
        "execution profile": request_plan.execution_profile_hash,
        "source revision": source_revision,
        "backend": backend_name,
    }
    observed = {
        "campaign": ledger.campaign_hash,
        "public manifest": ledger.public_case_manifest_hash,
        "request plan": ledger.request_plan_hash,
        "execution profile": ledger.execution_profile_hash,
        "source revision": ledger.source_revision,
        "backend": ledger.backend_name,
    }
    for label, expected_value in expected.items():
        if observed[label] != expected_value:
            raise ProductionFemRunError(
                f"cannot resume S10 production run with different {label}"
            )
    if ledger.planned_case_count != len(request_plan.entries):
        raise ProductionFemRunError(
            "cannot resume S10 production run with a different planned population"
        )


def _record_attempt(
    *,
    request_plan: PublicFemRequestPlan,
    request: FemRequest,
    attempt: FemAttempt,
    store: ObjectStore,
    elapsed_seconds: float,
    origin: AttemptOrigin,
) -> ProductionFemRunRecord:
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise ProductionFemRunError("elapsed_seconds must be finite and non-negative")
    if type(attempt) is not FemAttempt or attempt.request != request:
        raise ProductionFemRunError(
            "backend returned an attempt for a different request"
        )
    entry = request_plan.entry_by_case_id(request.request_id)
    if (
        attempt.recipe_hash != entry.request_recipe_hash
        or hash_spec(attempt.request, "content") != entry.request_content_hash
    ):
        raise ProductionFemRunError(
            "backend attempt does not match the public request plan"
        )
    if attempt.success:
        try:
            _, result_hash = validate_successful_fem_attempt(attempt, store)
        except ValueError as exc:
            raise ProductionFemRunError(
                "successful FEM attempt lacks valid ObjectStore material"
            ) from exc
        return ProductionFemRunRecord(
            case_id=request.request_id,
            request_recipe_hash=attempt.recipe_hash,
            request_content_hash=entry.request_content_hash,
            success=True,
            result_hash=result_hash,
            store_content_hash=attempt.store_content_hash,
            failure_code=None,
            message="succeeded",
            elapsed_seconds=float(elapsed_seconds),
            origin=origin,
        )
    if (
        attempt.result is not None
        or attempt.result_hash is not None
        or attempt.store_content_hash is not None
        or not isinstance(attempt.failure_code, FemFailureCode)
    ):
        raise ProductionFemRunError("failed FEM attempt has inconsistent result fields")
    return ProductionFemRunRecord(
        case_id=request.request_id,
        request_recipe_hash=attempt.recipe_hash,
        request_content_hash=entry.request_content_hash,
        success=False,
        result_hash=None,
        store_content_hash=None,
        failure_code=attempt.failure_code,
        message=f"failed:{attempt.failure_code.value}",
        elapsed_seconds=float(elapsed_seconds),
        origin=origin,
    )


def _attempt_from_record(
    record: ProductionFemRunRecord,
    request: FemRequest,
    store: ObjectStore,
) -> FemAttempt:
    if record.request_recipe_hash != request.recipe_hash:
        raise ProductionFemRunError("ledger record recipe does not match request")
    if record.request_content_hash != hash_spec(request, "content"):
        raise ProductionFemRunError("ledger record content does not match request")
    if record.success:
        if record.store_content_hash is None or record.result_hash is None:
            raise ProductionFemRunError("successful ledger record lacks result hash")
        payload = store.get(record.store_content_hash)
        resolved = store.resolve_recipe(request.recipe_hash)
        if resolved != record.store_content_hash:
            raise ProductionFemRunError(
                "stored recipe result does not match the ledger"
            )
        result = _decode_stored_result(request, payload)
        result_hash = hash_spec(result, "content")
        if result_hash != record.result_hash:
            raise ProductionFemRunError("stored result does not match the ledger")
        attempt = FemAttempt(
            request=request,
            recipe_hash=request.recipe_hash,
            success=True,
            result=result,
            result_hash=result_hash,
            store_content_hash=record.store_content_hash,
            failure_code=None,
            message="succeeded",
        )
        try:
            validate_successful_fem_attempt(attempt, store)
        except ValueError as exc:
            raise ProductionFemRunError(
                "stored result does not satisfy the S10 FEM trust boundary"
            ) from exc
        return attempt
    if record.failure_code is None:
        raise ProductionFemRunError("failed ledger record lacks a failure code")
    return FemAttempt(
        request=request,
        recipe_hash=request.recipe_hash,
        success=False,
        result=None,
        result_hash=None,
        store_content_hash=None,
        failure_code=record.failure_code,
        message=record.message,
    )


def _validate_ledger_records(
    ledger: ProductionFemRunLedger,
    request_plan: PublicFemRequestPlan,
    requests: Mapping[str, FemRequest],
    store: ObjectStore,
) -> None:
    plan_ids = {entry.case_id for entry in request_plan.entries}
    seen: set[str] = set()
    for record in ledger.records:
        if record.case_id not in plan_ids or record.case_id not in requests:
            raise ProductionFemRunError(
                "ledger contains a case outside the exact request plan"
            )
        if record.case_id in seen:
            raise ProductionFemRunError("ledger contains duplicate case IDs")
        seen.add(record.case_id)
        entry = request_plan.entry_by_case_id(record.case_id)
        if (
            record.request_recipe_hash != entry.request_recipe_hash
            or record.request_content_hash != entry.request_content_hash
        ):
            raise ProductionFemRunError("ledger record does not match request plan")
        if record.success:
            _attempt_from_record(record, requests[record.case_id], store)
        elif store.recipe_results(record.request_recipe_hash):
            raise ProductionFemRunError(
                "failed ledger record conflicts with stored recipe material"
            )


def _decode_stored_result(request: FemRequest, payload: bytes) -> FemResult:
    try:
        document = strict_json_loads(payload)
    except ValueError as exc:
        raise ProductionFemRunError("stored FEM result is not strict JSON") from exc
    if not isinstance(document, dict):
        raise ProductionFemRunError("stored FEM result payload is not an object")
    if frozenset(document) != {"domain", "payload"}:
        raise ProductionFemRunError("stored FEM result envelope fields changed")
    if document.get("domain") != "content":
        raise ProductionFemRunError("stored FEM result has the wrong domain")
    raw = document.get("payload")
    if not isinstance(raw, dict) or raw.get("$type") != _RESULT_TYPE:
        raise ProductionFemRunError("stored FEM result has the wrong type")
    expected_fields = {
        "$type",
        "request_id",
        "mesh_id",
        "element_order",
        "displacement_dof_coordinates",
        "displacement",
        "quadrature_points",
        "quadrature_weights",
        "first_piola_stress",
        "deformation_gradient",
        "recovered_first_piola_stress",
        "reactions",
        "rigid_mode_multipliers",
        "total_strain_energy",
        "external_work",
        "total_potential_energy",
        "convergence_history",
        "qois",
    }
    if frozenset(raw) != expected_fields:
        raise ProductionFemRunError("stored FEM result fields changed")
    try:
        result = FemResult(
            request_id=_string(raw["request_id"]),
            mesh_id=_string(raw["mesh_id"]),
            element_order=_integer(raw["element_order"]),
            displacement_dof_coordinates=_tuple2(raw["displacement_dof_coordinates"]),
            displacement=_tuple2(raw["displacement"]),
            quadrature_points=_tuple3(raw["quadrature_points"]),
            quadrature_weights=_tuple_float_nested(raw["quadrature_weights"]),
            first_piola_stress=_tuple5(raw["first_piola_stress"]),
            deformation_gradient=_tuple5(raw["deformation_gradient"]),
            recovered_first_piola_stress=_tuple4(raw["recovered_first_piola_stress"]),
            reactions={
                key: _pair(value) for key, value in _mapping(raw["reactions"]).items()
            },
            rigid_mode_multipliers=tuple(
                _number(value) for value in _sequence(raw["rigid_mode_multipliers"])
            ),
            total_strain_energy=_number(raw["total_strain_energy"]),
            external_work=_number(raw["external_work"]),
            total_potential_energy=_number(raw["total_potential_energy"]),
            convergence_history=tuple(
                _convergence_entry(entry)
                for entry in _sequence(raw["convergence_history"])
            ),
            qois={key: _number(value) for key, value in _mapping(raw["qois"]).items()},
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ProductionFemRunError):
            raise
        raise ProductionFemRunError("stored FEM result fields are invalid") from exc
    if (
        result.request_id != request.request_id
        or result.mesh_id != request.mesh.mesh_id
    ):
        raise ProductionFemRunError("stored FEM result does not match request")
    if canonical_json_bytes({"domain": "content", "payload": result}) != payload:
        raise ProductionFemRunError("stored FEM result is not canonical and exact")
    return result


def _convergence_entry(value: object) -> FemConvergenceEntry:
    entry = _mapping(value)
    if (
        frozenset(entry)
        != {
            "$type",
            "load_step",
            "iteration",
            "residual_norm",
            "increment_norm",
        }
        or entry["$type"] != _CONVERGENCE_TYPE
    ):
        raise ProductionFemRunError("stored convergence entry fields changed")
    return FemConvergenceEntry(
        load_step=_integer(entry["load_step"]),
        iteration=_integer(entry["iteration"]),
        residual_norm=_number(entry["residual_norm"]),
        increment_norm=_number(entry["increment_norm"]),
    )


def _load_ledger(path: Path) -> ProductionFemRunLedger:
    try:
        return ProductionFemRunLedger.from_bytes(path.read_bytes())
    except OSError as exc:
        raise ProductionFemRunError("cannot read production ledger") from exc


def _ledger_from_document(payload: bytes) -> ProductionFemRunLedger:
    try:
        document = strict_json_loads(payload)
    except ValueError as exc:
        raise ProductionFemRunError("production ledger is not strict JSON") from exc
    root = _mapping(document)
    if root.get("$type") != _LEDGER_TYPE:
        raise ProductionFemRunError("S10 production ledger has the wrong type")
    expected_fields = {
        "$type",
        "run_id",
        "campaign_hash",
        "public_case_manifest_hash",
        "request_plan_hash",
        "execution_profile_hash",
        "source_revision",
        "backend_name",
        "planned_case_count",
        "records",
        "schema_version",
    }
    if frozenset(root) != expected_fields:
        raise ProductionFemRunError("production ledger fields changed")
    return ProductionFemRunLedger(
        run_id=_string(root["run_id"]),
        campaign_hash=_string(root["campaign_hash"]),
        public_case_manifest_hash=_string(root["public_case_manifest_hash"]),
        request_plan_hash=_string(root["request_plan_hash"]),
        execution_profile_hash=_string(root["execution_profile_hash"]),
        source_revision=_string(root["source_revision"]),
        backend_name=_string(root["backend_name"]),
        planned_case_count=_integer(root["planned_case_count"]),
        records=tuple(_record_from_json(value) for value in _sequence(root["records"])),
        schema_version=_string(root["schema_version"]),
    )


def _record_from_json(value: object) -> ProductionFemRunRecord:
    record = _mapping(value)
    if record.get("$type") != _RECORD_TYPE:
        raise ProductionFemRunError("S10 production record has the wrong type")
    expected_fields = {
        "$type",
        "case_id",
        "request_recipe_hash",
        "request_content_hash",
        "success",
        "result_hash",
        "store_content_hash",
        "failure_code",
        "message",
        "elapsed_seconds",
        "origin",
    }
    if frozenset(record) != expected_fields:
        raise ProductionFemRunError("production record fields changed")
    raw_failure_code = record["failure_code"]
    try:
        failure_code = (
            None
            if raw_failure_code is None
            else FemFailureCode(_string(raw_failure_code))
        )
        origin = AttemptOrigin(_string(record["origin"]))
    except (TypeError, ValueError) as exc:
        raise ProductionFemRunError("production record enum is invalid") from exc
    return ProductionFemRunRecord(
        case_id=_string(record["case_id"]),
        request_recipe_hash=_string(record["request_recipe_hash"]),
        request_content_hash=_string(record["request_content_hash"]),
        success=_boolean(record["success"]),
        result_hash=_optional_string(record["result_hash"]),
        store_content_hash=_optional_string(record["store_content_hash"]),
        failure_code=failure_code,
        message=_string(record["message"]),
        elapsed_seconds=_number(record["elapsed_seconds"]),
        origin=origin,
    )


def _write_ledger(path: Path, ledger: ProductionFemRunLedger) -> None:
    _atomic_write(path, ledger.to_bytes())


def persist_request_plan(plan: PublicFemRequestPlan, path: Path) -> None:
    """Create the exact run plan once and refuse a different resume plan."""

    if type(plan) is not PublicFemRequestPlan:
        raise ProductionFemRunError("plan must be an exact PublicFemRequestPlan")
    target = Path(path)
    if target.exists():
        if target.is_symlink() or target.read_bytes() != plan.to_bytes():
            raise ProductionFemRunError("refusing to replace a different request plan")
        return
    _atomic_write(target, plan.to_bytes())


def persist_execution_profile(profile: ExecutionProfile, path: Path) -> None:
    """Create the exact numerical execution profile once and refuse drift."""

    if type(profile) is not ExecutionProfile:
        raise ProductionFemRunError("profile must be an exact ExecutionProfile")
    payload = canonical_json_bytes(profile)
    target = Path(path)
    if target.exists():
        if target.is_symlink() or target.read_bytes() != payload:
            raise ProductionFemRunError(
                "refusing to replace a different execution profile"
            )
        return
    _atomic_write(target, payload)


@contextmanager
def _exclusive_run_lock(path: Path) -> Iterator[None]:
    """Hold one non-blocking process lock for a campaign run root."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(  # type: ignore[attr-defined]
                    handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
                )
        except OSError as exc:
            raise ProductionFemRunError(
                "another process already owns this production run root"
            ) from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(  # type: ignore[attr-defined]
                    handle.fileno(),
                    fcntl.LOCK_UN,  # type: ignore[attr-defined]
                )


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() and path.is_symlink():
        raise ProductionFemRunError("production state path must not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            pass
    finally:
        os.close(descriptor)


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ProductionFemRunError(f"{label} must be a lowercase BLAKE2b-256 digest")
    return value


def _git_source_revision() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProductionFemRunError(
            "cannot resolve the production source revision"
        ) from exc
    revision = completed.stdout.strip()
    if _REVISION.fullmatch(revision) is None:
        raise ProductionFemRunError("Git returned an invalid source revision")
    return revision


def _mapping(value: JsonValue | object) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise ProductionFemRunError("expected a JSON object")
    return cast(Mapping[str, JsonValue], value)


def _sequence(value: JsonValue | object) -> tuple[JsonValue, ...]:
    if not isinstance(value, list):
        raise ProductionFemRunError("expected a JSON array")
    return tuple(value)


def _string(value: JsonValue | object) -> str:
    if not isinstance(value, str):
        raise ProductionFemRunError("expected a JSON string")
    return value


def _optional_string(value: JsonValue | object) -> str | None:
    if value is None:
        return None
    return _string(value)


def _integer(value: JsonValue | object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProductionFemRunError("expected a JSON integer")
    return value


def _boolean(value: JsonValue | object) -> bool:
    if not isinstance(value, bool):
        raise ProductionFemRunError("expected a JSON boolean")
    return value


def _number(value: JsonValue | object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionFemRunError("expected a JSON number")
    return float(value)


def _pair(value: JsonValue | object) -> tuple[float, float]:
    values = _sequence(value)
    if len(values) != 2:
        raise ProductionFemRunError("expected a two-vector")
    return (_number(values[0]), _number(values[1]))


def _tuple2(value: JsonValue | object) -> tuple[tuple[float, float], ...]:
    return tuple(_pair(item) for item in _sequence(value))


def _tuple3(
    value: JsonValue | object,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(_tuple2(item) for item in _sequence(value))


def _tuple_float_nested(value: JsonValue | object) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(_number(item) for item in _sequence(row)) for row in _sequence(value)
    )


def _row3(value: JsonValue | object) -> tuple[float, float, float]:
    values = _sequence(value)
    if len(values) != 3:
        raise ProductionFemRunError("expected a three-vector")
    return (_number(values[0]), _number(values[1]), _number(values[2]))


def _matrix3(
    value: JsonValue | object,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    rows = tuple(_row3(row) for row in _sequence(value))
    if len(rows) != 3:
        raise ProductionFemRunError("expected a 3x3 matrix")
    return rows


def _tuple4(
    value: JsonValue | object,
) -> tuple[
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    ...,
]:
    return tuple(_matrix3(item) for item in _sequence(value))


def _tuple5(
    value: JsonValue | object,
) -> tuple[
    tuple[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        ...,
    ],
    ...,
]:
    return tuple(_tuple4(item) for item in _sequence(value))


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assert_runtime_source() -> None:
    expected = (_repository_root() / "src" / "piginot" / "__init__.py").resolve()
    module_path = getattr(piginot, "__file__", None)
    if module_path is None or Path(module_path).resolve() != expected:
        raise ProductionFemRunError(
            "piginot was imported from a different checkout; set PYTHONPATH to this "
            "worktree's src directory"
        )


def _assert_single_thread_environment() -> None:
    invalid = {
        name: os.environ.get(name)
        for name in _THREAD_ENVIRONMENT
        if os.environ.get(name) != "1"
    }
    if invalid:
        details = ", ".join(f"{name}={value!r}" for name, value in invalid.items())
        raise ProductionFemRunError(
            f"BLAS/OpenMP thread environment must be pinned to one: {details}"
        )


def _assert_clean_source_revision() -> str:
    repository_root = _repository_root()
    try:
        completed = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProductionFemRunError(
            "cannot inspect the production source tree"
        ) from exc
    if completed.stdout.strip():
        raise ProductionFemRunError(
            "production FEM execution requires a clean committed source tree"
        )
    return _git_source_revision()


def _progress(ledger: ProductionFemRunLedger, record: ProductionFemRunRecord) -> None:
    if record.success:
        status = "SUCCEEDED"
    elif record.failure_code is not None:
        status = f"FAILED:{record.failure_code.value}"
    else:  # guarded by ProductionFemRunRecord.__post_init__
        raise ProductionFemRunError("failed progress record has no failure code")
    print(
        f"[{ledger.completed_case_count:03d}/{ledger.planned_case_count:03d}] "
        f"{record.case_id} {status} {record.elapsed_seconds:.3f}s "
        f"origin={record.origin.value}",
        flush=True,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the public DogBone S10 FEM campaign."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=default_s10_run_root(),
        help="local ignored directory for the ledger and ObjectStore",
    )
    parser.add_argument(
        "--max-new-attempts",
        type=int,
        default=None,
        help="stop safely after this many new backend solves",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="materialize and validate the exact plan without solving",
    )
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    """Run or resume the local public S10 production campaign."""

    arguments = _argument_parser().parse_args(argv)
    _assert_runtime_source()
    _assert_single_thread_environment()
    source_revision = _assert_clean_source_revision()
    root = arguments.run_root.resolve()
    ledger = run_dogbone_s10_production(
        run_root=root,
        max_new_attempts=arguments.max_new_attempts,
        dry_run=arguments.dry_run,
        source_revision=source_revision,
        on_record=_progress,
    )
    print(f"run root: {root}")
    print(f"source revision: {ledger.source_revision}")
    print(f"request plan: {ledger.request_plan_hash}")
    print(
        "completed: "
        f"{ledger.completed_case_count}/{ledger.planned_case_count} "
        f"({ledger.success_count} successes, {ledger.failure_count} failures)"
    )
    if arguments.dry_run:
        print("dry run: no FEM request was submitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))


__all__ = (
    "AttemptOrigin",
    "ProductionFemRunError",
    "ProductionFemRunLedger",
    "ProductionFemRunRecord",
    "RUN_ID",
    "default_s10_execution_profile",
    "default_s10_run_root",
    "load_s10_production_attempts",
    "load_production_attempts",
    "persist_execution_profile",
    "persist_request_plan",
    "run_dogbone_s10_production",
    "run_public_fem_requests",
)
