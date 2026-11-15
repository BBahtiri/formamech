"""Bound, pending-only reference solves with explicit initializer outcomes."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from typing import Any, cast

import numpy as np
import torch

from piginot.compiler.execution_records import CaseJob, Document, OutcomeStore, digest
from piginot.compiler.planning import ExecutionPlan
from piginot.compiler.reference_metrics import (
    compare_prediction,
    refinement_diagnostics,
)
from piginot.contracts import (
    CanonicalQuantity,
    MaterialSpec,
    Unit,
    canonical_json_bytes,
    hash_spec,
    strict_json_loads,
)
from piginot.fem.models import FemConvergenceEntry, FemRequest, FemResult
from piginot.operators.architecture import OperatorOutput
from piginot.problem.requests import fem_request


def _backend() -> Any:
    from piginot.fem import ScikitFemBackend

    return ScikitFemBackend()


def _request_sha256(request: FemRequest) -> str:
    return hashlib.sha256(canonical_json_bytes(request)).hexdigest()


def initializer_request(request: FemRequest) -> FemRequest:
    """The prospectively selected S26 NH initializer with matching moduli."""
    if request.material.model != "hgo":
        raise ValueError("the declared initializer requires HGO")
    mu = request.material.properties["shear_modulus"].value
    bulk = request.material.properties["bulk_modulus"].value
    material = MaterialSpec(
        name="campaign-hgo-reference-initializer",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": CanonicalQuantity(
                9 * bulk * mu / (3 * bulk + mu), Unit.MPA
            ),
            "poisson_ratio": CanonicalQuantity(
                (3 * bulk - 2 * mu) / (2 * (3 * bulk + mu)), Unit.ONE
            ),
        },
    )
    return replace(request, material=material)


def _decode_result(outcome: Mapping[str, Any], request: FemRequest) -> FemResult:
    payload = outcome["result_utf8"].encode("utf-8")
    if hashlib.sha256(payload).hexdigest() != outcome["result_sha256"]:
        raise ValueError("reference result bytes digest differs")
    document = strict_json_loads(payload)
    if not isinstance(document, dict):
        raise ValueError("reference result must be an object")
    expected = {field.name for field in fields(FemResult)} | {"$type"}
    if set(document) != expected or document["$type"] != "piginot.contracts.FemResult":
        raise ValueError("reference result schema differs")
    values = cast(Document, document)
    history = values["convergence_history"]
    entries = tuple(
        FemConvergenceEntry(**{k: v for k, v in row.items() if k != "$type"})
        for row in history
    )
    result = FemResult(
        **{
            **{k: v for k, v in values.items() if k != "$type"},
            "convergence_history": entries,
        }
    )
    if (
        hash_spec(result, "content") != outcome["result_hash"]
        or result.request_id != request.request_id
        or result.mesh_id != request.mesh.mesh_id
        or result.element_order != request.config.element_order
    ):
        raise ValueError("reference result identity differs")
    from piginot.datasets.snapshots import _validate_result

    _validate_result(request=request, result=result)
    return result


def _solve(
    request: FemRequest,
    *,
    initializer: Document | None = None,
    initial_request: FemRequest | None = None,
) -> Document:
    request_record: Document = {
        "recipe_hash": request.recipe_hash,
        "request_utf8": canonical_json_bytes(request).decode("utf-8"),
    }
    if initial_request is not None:
        request_record = {
            **request_record,
            "initializer_request_sha256": _request_sha256(initial_request),
            "load_factors": [1.0],
        }
        if initializer is None or initializer["status"] != "PASS":
            return {
                **request_record,
                "status": "INCOMPLETE",
                "reason": "declared initializer did not pass",
            }
        request_record = {
            **request_record,
            "initializer_result_hash": initializer["result_hash"],
        }
    backend = _backend()
    if initial_request is None:
        attempt = backend.solve_many((request,))[0]
    else:
        from piginot.fem.linearization import FemLinearization

        initial = _decode_result(cast(Document, initializer), initial_request)
        vector = FemLinearization(request).vector(
            np.asarray(initial.displacement, dtype=np.float64)
        )
        attempt = backend.solve_from(request, vector, (1.0,))
    record: Document = {
        **request_record,
        "status": "PASS" if attempt.success else "FAIL",
        "failure_code": None
        if attempt.failure_code is None
        else str(attempt.failure_code),
        "message": attempt.message,
    }
    if attempt.recipe_hash != request.recipe_hash or _request_sha256(
        attempt.request
    ) != _request_sha256(request):
        raise ValueError("solver returned a different recipe")
    if attempt.success:
        if attempt.result is None or attempt.result_hash != hash_spec(
            attempt.result, "content"
        ):
            raise ValueError("successful solver outcome has no matching result")
        payload = canonical_json_bytes(attempt.result)
        record = {
            **record,
            "result_utf8": payload.decode("utf-8"),
            "result_sha256": hashlib.sha256(payload).hexdigest(),
            "result_hash": attempt.result_hash,
        }
        _decode_result(record, request)
    return record


def _level_jobs(
    plan: ExecutionPlan, adapter: Any, case: Any, level: int, store: OutcomeStore
) -> tuple[tuple[CaseJob, ...], FemRequest | None]:
    request_id = f"{case.case_id}:level-{level}"
    try:
        mesh = adapter.geometry(plan.problem, case, level=level)
        values = {
            p.name: CanonicalQuantity(case.primitive_coordinates[p.name], p.unit)
            for p in plan.problem.parameters
        }
        request = fem_request(
            plan.problem, values, mesh, request_id, plan.config.execution.reference
        )
    except Exception as error:
        failure = {
            "status": "INCOMPLETE" if isinstance(error, ValueError) else "ERROR",
            "reason": "reference geometry/request unavailable",
            "error": {"category": type(error).__name__, "message": str(error)},
        }
        binding = digest(
            {"plan": plan.plan_sha256, "case": case.case_id, "level": level}
        )
        return (CaseJob(request_id, binding, lambda: failure),), None
    if plan.config.reference.initialization == "cold":
        return (
            CaseJob(request_id, _request_sha256(request), lambda: _solve(request)),
        ), request
    initial = initializer_request(request)
    initial_job = CaseJob(
        request_id + ":initializer",
        _request_sha256(initial),
        lambda: _solve(initial),
    )
    final_job = CaseJob(
        request_id,
        _request_sha256(request),
        lambda: _solve(
            request, initializer=store.cached(initial_job), initial_request=initial
        ),
    )
    return (initial_job, final_job), request


def _case_references(
    plan: ExecutionPlan,
    adapter: Any,
    case_id: str,
    partition: str,
    training: Document,
) -> tuple[OutcomeStore, tuple[tuple[tuple[CaseJob, ...], FemRequest | None], ...]]:
    allowed = {
        "train-screen": plan.config.source.screen_case_ids,
        "public-development": plan.config.source.development_case_ids,
    }
    if partition not in allowed or case_id not in allowed[partition]:
        raise ValueError("reference case is not authorized for this partition")
    case = next(case for case in plan.manifest.cases if case.case_id == case_id)
    binding = digest(
        {
            "plan": plan.plan_sha256,
            "partition": partition,
            "case": case_id,
            "training": training,
        }
    )
    key = hashlib.sha256(case_id.encode()).hexdigest()
    store = OutcomeStore(plan.output_path / "references" / partition / key, binding)
    prepared = tuple(
        _level_jobs(plan, adapter, case, level, store)
        for level in plan.config.reference.refinement_levels
    )
    return store, prepared


def validate_case_cache(
    plan: ExecutionPlan, adapter: Any, case_id: str, partition: str, training: Document
) -> list[Document]:
    """Verify a completed case's nested cache without predictions or solves."""
    store, prepared = _case_references(plan, adapter, case_id, partition, training)
    jobs = tuple(job for group, _ in prepared for job in group)
    outcomes = store.validate(jobs)
    requests = {
        job.request_id: (
            initializer_request(request) if len(group) == 2 and index == 0 else request
        )
        for group, request in prepared
        if request is not None
        for index, job in enumerate(group)
    }
    for job, outcome in zip(jobs, outcomes, strict=True):
        if (
            outcome is None
            or outcome.get("status") not in {"PASS", "FAIL", "INCOMPLETE", "ERROR"}
            or outcome.get("request_id") != job.request_id
            or outcome.get("request_sha256") != job.request_sha256
        ):
            raise ValueError(
                "reference cache is incomplete or has a different identity"
            )
        request = requests.get(job.request_id)
        if request is not None and outcome["status"] != "ERROR":
            if outcome.get("request_utf8") != canonical_json_bytes(request).decode(
                "utf-8"
            ):
                raise ValueError("reference cache request bytes differ")
        if outcome["status"] == "PASS":
            if request is None:
                raise ValueError(
                    "reference cache has an unexpected successful geometry"
                )
            _decode_result(outcome, request)
    return _reference_records(
        plan, store, jobs, tuple(cast(Document, outcome) for outcome in outcomes)
    )


def _reference_records(
    plan: ExecutionPlan,
    store: OutcomeStore,
    jobs: tuple[CaseJob, ...],
    outcomes: tuple[Document, ...],
) -> list[Document]:
    """Keep full raw outcomes in their bound files, outside aggregate reports."""
    references = []
    for job, outcome in zip(jobs, outcomes, strict=True):
        path = store._path(job)
        with path.open("rb") as stream:
            file_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        references.append(
            {
                **{
                    name: value
                    for name, value in outcome.items()
                    if name not in {"request_utf8", "result_utf8"}
                },
                "evidence": {
                    "path": path.relative_to(plan.output_path).as_posix(),
                    "sha256": file_sha256,
                    "binding_sha256": store._binding(job),
                },
            }
        )
    return references


def evaluate_case(
    plan: ExecutionPlan,
    adapter: Any,
    case_id: str,
    partition: str,
    training: Document,
    *,
    predict: Callable[[str, torch.Tensor], OperatorOutput],
) -> Document:
    """Compare one authorized row, retaining every terminal reference attempt."""
    store, prepared = _case_references(plan, adapter, case_id, partition, training)
    jobs = tuple(job for group, _ in prepared for job in group)
    outcomes = store.run(jobs)
    base = {
        "case_id": case_id,
        "references": _reference_records(plan, store, jobs, outcomes),
        "scientific_status": "NOT_VERIFIED",
        "claim_level": "NOT_VERIFIED",
    }
    if any(outcome["status"] != "PASS" for outcome in outcomes):
        status = (
            "ERROR"
            if any(outcome["status"] == "ERROR" for outcome in outcomes)
            else "INCOMPLETE"
        )
        return {
            **base,
            "status": status,
            "scientific_status": "INCOMPLETE",
            "discrete_fem_errors": None,
            "reference_diagnostics": {"status": "INCOMPLETE"},
        }
    by_id = {outcome["request_id"]: outcome for outcome in outcomes}
    requests = tuple(request for _, request in prepared if request is not None)
    results = tuple(
        _decode_result(by_id[request.request_id], request) for request in requests
    )
    errors = compare_prediction(
        plan.problem, requests[-1], results[-1], lambda points: predict(case_id, points)
    )
    diagnostics = refinement_diagnostics(results)
    if plan.adapter.adapter_id == "cad-fixed-v1":
        from piginot.compiler.reference_convergence import compare_fem_pair

        diagnostics = {
            **diagnostics,
            "same_polygon_field_changes": [
                compare_fem_pair(
                    plan.problem,
                    coarse,
                    coarse_result,
                    fine,
                    fine_result,
                    tolerances=None,
                )
                for coarse, coarse_result, fine, fine_result in zip(
                    requests[:-1], results[:-1], requests[1:], results[1:], strict=True
                )
            ],
        }
    return {
        **base,
        "status": "PASS",
        "fine_level": plan.config.reference.refinement_levels[-1],
        "discrete_fem_errors": errors,
        "reference_diagnostics": diagnostics,
    }


__all__ = ("evaluate_case", "initializer_request", "validate_case_cache")
