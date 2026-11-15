"""Public dataset boundaries refuse incomplete or cross-bound evidence."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.datasets.snapshots import (
    DatasetSnapshotValidationError,
    SampleTrace,
    StoredSnapshotReference,
    build_role_snapshots,
    persist_role_snapshot,
    validate_snapshot_attempt,
    validate_successful_fem_attempt,
)
from piginot.doe import EvidenceRole
from piginot.fem import FemConvergenceEntry, FemFailureCode
from piginot.store import ObjectStore
from piginot.store.store import content_hash
from tests.datasets.collocation_fixtures import snapshot
from tests.datasets.test_snapshots import (
    _build,
    _checkpoint,
    _failure,
    _manifest,
    _put_success,
    _request_plan,
    _success,
)


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"fem_result_hash": "a" * 64, "sampler_stream_hash": "b" * 64},
        {"fem_result_hash": "../result"},
        {"sampler_stream_hash": "A" * 64},
    ],
)
def test_sample_trace_requires_one_valid_content_identity(values):
    with pytest.raises(DatasetSnapshotValidationError):
        SampleTrace(**values)


def test_sampler_trace_keeps_its_own_identity_without_a_fem_label():
    trace = SampleTrace(sampler_stream_hash="a" * 64)
    assert trace.fem_result_hash is None
    assert trace.sampler_stream_hash == "a" * 64


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"case_id": ""}, "case_id"),
        ({"leakage_group_id": " "}, "leakage_group_id"),
        ({"role": EvidenceRole.HOLDOUT}, "HOLDOUT"),
        ({"request_recipe_hash": "bad"}, "request_recipe_hash"),
        ({"request_content_hash": "bad"}, "request_content_hash"),
        ({"mesh_id": ""}, "mesh_id"),
        ({"mesh_content_hash": "bad"}, "mesh_content_hash"),
        ({"message": ""}, "message"),
        ({"success": 1}, "boolean"),
        ({"sample_trace": None}, "sample trace"),
        (
            {"sample_trace": SampleTrace(sampler_stream_hash="a" * 64)},
            "fem_result_hash",
        ),
        ({"failure_code": FemFailureCode.NEWTON_NON_CONVERGENCE}, "failure code"),
        ({"store_content_hash": None}, "ObjectStore"),
        ({"store_content_hash": "bad"}, "store_content_hash"),
        ({"message": "private backend details"}, "sanitized"),
    ],
)
def test_successful_dataset_record_requires_complete_public_lineage(change, message):
    with pytest.raises(DatasetSnapshotValidationError, match=message):
        replace(snapshot().cases[0], **change)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sample_trace": SampleTrace(fem_result_hash="a" * 64)}, "result material"),
        ({"store_content_hash": "a" * 64}, "result material"),
        ({"failure_code": None}, "failure code"),
        ({"message": "arbitrary backend details"}, "sanitized"),
    ],
)
def test_failed_dataset_record_cannot_smuggle_result_material(change, message):
    with pytest.raises(DatasetSnapshotValidationError, match=message):
        replace(snapshot(success=False).cases[0], **change)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"role": EvidenceRole.HOLDOUT}, "HOLDOUT"),
        ({"cases": ()}, "empty"),
        ({"cases": (SimpleNamespace(case_id="untyped"),)}, "DatasetCaseRecord"),
        ({"cases": snapshot(role=EvidenceRole.DEVELOPMENT).cases}, "evidence role"),
        ({"cases": snapshot().cases * 2}, "unique"),
        ({"schema_version": "future"}, "schema"),
    ],
)
def test_role_snapshots_refuse_untyped_duplicate_or_mixed_role_rows(change, message):
    with pytest.raises(DatasetSnapshotValidationError, match=message):
        replace(snapshot(), **change)


@pytest.mark.parametrize("field", ["snapshot_content_hash", "store_content_hash"])
def test_stored_snapshot_reference_rejects_filesystem_names(field):
    with pytest.raises(DatasetSnapshotValidationError, match=field):
        StoredSnapshotReference(
            **{
                **dict.fromkeys(
                    ("snapshot_content_hash", "store_content_hash"), "a" * 64
                ),
                field: "/tmp/snapshot",
            }
        )


@pytest.mark.parametrize("which", ["snapshot", "store"])
def test_snapshot_persistence_requires_exact_contracts(tmp_path, which):
    arguments = {"snapshot": snapshot(), "store": ObjectStore.create(tmp_path)}
    arguments[which] = object()
    with pytest.raises(DatasetSnapshotValidationError, match=which):
        persist_role_snapshot(**arguments)


@pytest.fixture
def ingestion(tmp_path):
    manifest = _manifest()
    return {
        "checkpoint": _checkpoint(manifest),
        "manifest": manifest,
        "request_plan": _request_plan(manifest),
        "attempts": {"train-1": _failure("train-1"), "dev-1": _failure("dev-1")},
        "store": ObjectStore.create(tmp_path),
    }


@pytest.mark.parametrize(
    "field", ["checkpoint", "manifest", "request_plan", "store", "attempts"]
)
def test_snapshot_builder_refuses_untyped_inputs_before_ingestion(ingestion, field):
    with pytest.raises(DatasetSnapshotValidationError, match=field):
        build_role_snapshots(**{**ingestion, field: object()})


@pytest.mark.parametrize("field", ["campaign_hash", "public_case_manifest_hash"])
def test_request_plan_cannot_transfer_between_public_campaigns(ingestion, field):
    arguments = {
        **ingestion,
        "request_plan": replace(ingestion["request_plan"], **{field: "0" * 64}),
    }
    with pytest.raises(
        DatasetSnapshotValidationError, match="request plan does not match"
    ):
        build_role_snapshots(**arguments)


def test_snapshot_builder_refuses_missing_planned_rows_and_wrong_case_provenance(
    ingestion,
):
    plan = ingestion["request_plan"]
    with pytest.raises(DatasetSnapshotValidationError, match="request plan case IDs"):
        build_role_snapshots(
            **{**ingestion, "request_plan": replace(plan, entries=plan.entries[:1])}
        )
    changed = tuple(
        replace(entry, leakage_group_id=f"another-group-{entry.case_id}")
        for entry in plan.entries
    )
    with pytest.raises(DatasetSnapshotValidationError, match="provenance"):
        build_role_snapshots(
            **{**ingestion, "request_plan": replace(plan, entries=changed)}
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"request": _failure("other").request}, "different request identity"),
        ({"message": ""}, "attempt message"),
        ({"failure_code": None}, "failure code"),
    ],
)
def test_invalid_failed_attempts_are_not_ingested_as_visible_failures(
    ingestion, change, message
):
    attempts = {
        **ingestion["attempts"],
        "train-1": replace(_failure("train-1"), **change),
    }
    with pytest.raises(DatasetSnapshotValidationError, match=message):
        build_role_snapshots(**{**ingestion, "attempts": attempts})


def test_untyped_attempt_is_not_treated_as_a_numerical_failure(ingestion):
    with pytest.raises(DatasetSnapshotValidationError, match="FemAttempt"):
        build_role_snapshots(
            **{**ingestion, "attempts": {**ingestion["attempts"], "train-1": object()}}
        )


@pytest.mark.parametrize("change", ["store", "result", "failure", "recipe"])
def test_successful_attempt_requires_a_typed_result_and_matching_recipe(
    tmp_path, change
):
    attempt, store = _success("train-1"), ObjectStore.create(tmp_path)
    if change == "store":
        store = object()
    elif change == "result":
        attempt = replace(attempt, result=None)
    elif change == "failure":
        attempt = replace(attempt, failure_code=FemFailureCode.NEWTON_NON_CONVERGENCE)
    else:
        attempt = replace(attempt, recipe_hash="0" * 64)
    with pytest.raises(DatasetSnapshotValidationError):
        validate_successful_fem_attempt(attempt, store)


def test_store_address_is_not_enough_when_materialized_bytes_are_corrupt(
    tmp_path, monkeypatch
):
    attempt, store = _success("train-1"), ObjectStore.create(tmp_path)
    _put_success(store, attempt)
    monkeypatch.setattr(ObjectStore, "get", lambda self, digest: b"unrelated result")
    with pytest.raises(DatasetSnapshotValidationError, match="bytes do not match"):
        validate_successful_fem_attempt(attempt, store)


@pytest.mark.parametrize(
    "change",
    [
        {"recovered_first_piola_stress": (((1.0, 2.0),) * 3,) * 3},
        {"recovered_first_piola_stress": (((float("nan"), 0.0, 0.0),) * 3,) * 3},
        {"displacement": ((float("inf"), 0.0),) * 3},
        {
            "quadrature_points": ((),),
            "quadrature_weights": ((),),
            "first_piola_stress": ((),),
            "deformation_gradient": ((),),
        },
        {"total_strain_energy": float("nan")},
        {"qois": {"unexpected": 1.0}},
        {"reactions": {"bottom": (1.0,)}},
        {
            "convergence_history": (
                FemConvergenceEntry(
                    load_step=0, iteration=1, residual_norm=0.0, increment_norm=0.0
                ),
            )
        },
    ],
)
def test_result_validation_refuses_nonfinite_or_incomplete_field_payloads(
    tmp_path, change
):
    attempt = _success("train-1")
    attempt = replace(attempt, result=replace(attempt.result, **change))
    with pytest.raises(DatasetSnapshotValidationError):
        validate_successful_fem_attempt(attempt, ObjectStore.create(tmp_path))


@pytest.fixture
def trusted(tmp_path):
    snapshots = _build(tmp_path)
    role = next(item for item in snapshots if item.role is EvidenceRole.TRAIN)
    return {
        "snapshot": role,
        "request_plan": _request_plan(_manifest()),
        "attempt": _success("train-1"),
        "store": ObjectStore(tmp_path),
    }


@pytest.mark.parametrize("field", ["snapshot", "request_plan", "attempt", "store"])
def test_resolution_requires_the_exact_bound_contract_types(trusted, field):
    with pytest.raises(TypeError, match=field):
        validate_snapshot_attempt(**{**trusted, field: object()})


@pytest.mark.parametrize(
    "change",
    [
        "no-record",
        "missing-entry",
        "record-provenance",
        "failed",
        "result-trace",
        "store-trace",
    ],
)
def test_resolution_refuses_missing_or_transferred_snapshot_evidence(trusted, change):
    record = trusted["snapshot"].cases[0]
    if change == "no-record":
        record = replace(record, case_id="other")
    elif change == "missing-entry":
        plan = trusted["request_plan"]
        plan = replace(
            plan,
            entries=tuple(
                entry for entry in plan.entries if entry.case_id != record.case_id
            ),
        )
        trusted = {
            **trusted,
            "request_plan": plan,
            "snapshot": replace(
                trusted["snapshot"], request_plan_hash=plan.content_hash
            ),
        }
    elif change == "record-provenance":
        record = replace(record, request_content_hash="0" * 64)
    elif change == "failed":
        record = replace(
            record,
            success=False,
            sample_trace=None,
            store_content_hash=None,
            failure_code=FemFailureCode.NEWTON_NON_CONVERGENCE,
            message="failed:NEWTON_NON_CONVERGENCE",
        )
    elif change == "result-trace":
        record = replace(record, sample_trace=SampleTrace(fem_result_hash="0" * 64))
    else:
        record = replace(record, store_content_hash="0" * 64)
    trusted = {**trusted, "snapshot": replace(trusted["snapshot"], cases=(record,))}
    with pytest.raises(DatasetSnapshotValidationError):
        validate_snapshot_attempt(**trusted)


def test_result_address_binds_the_complete_payload_not_only_result_id(tmp_path):
    attempt, store = _success("train-1"), ObjectStore.create(tmp_path)
    changed = replace(attempt.result, total_strain_energy=123.0)
    payload = canonical_json_bytes({"domain": "content", "payload": changed})
    store.put(payload)
    mixed = replace(attempt, result=changed, result_hash=hash_spec(changed, "content"))
    assert mixed.store_content_hash != content_hash(payload)
    with pytest.raises(DatasetSnapshotValidationError, match="store_content_hash"):
        validate_successful_fem_attempt(mixed, store)
