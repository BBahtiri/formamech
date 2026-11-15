"""Collocation replay refuses malformed populations and semantic provenance."""

from dataclasses import replace

import pytest

from piginot.contracts import BoundaryConditionSpec, LoadSpec, Unit, hash_spec
from piginot.datasets.collocation import (
    classify_boundary_facet,
    persist_collocation_batch,
    replay_collocation,
    sample_collocation,
    sample_request_collocation,
    verify_collocation_batch,
    verify_request_collocation_batch,
)
from piginot.datasets.collocation_models import (
    CollocationKind,
    SemanticBCClass,
    SemanticCollocationQuota,
    _decode_seed,
)
from piginot.doe import EvidenceRole
from tests.datasets.collocation_fixtures import (
    CASE_ID,
    boundary_conditions,
    loads,
    mesh,
    request_with_qoi,
    snapshot,
)


@pytest.fixture
def batch():
    return sample_collocation(
        snapshot(),
        CASE_ID,
        mesh(),
        boundary_conditions(),
        loads(),
        stream_index=0,
        interior_count=2,
        boundary_count=1,
        interface_count=0,
    )


@pytest.mark.parametrize("value", ["", "f" * 63, "F" * 64, "x" * 64, b"0" * 64])
def test_replay_seed_parser_rejects_ambiguous_or_noncanonical_material(value):
    with pytest.raises(ValueError, match="64 lowercase"):
        _decode_seed(value)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"quota_id": ""}, "quota_id"),
        ({"count": 0}, "positive integer"),
        ({"count": True}, "positive integer"),
        ({"semantic_tag": None}, "semantic_tag"),
        ({"region_id": 0}, "must not declare region_id"),
        ({"kind": CollocationKind.INTERFACE}, "BOUNDARY or INTERIOR"),
    ],
)
def test_boundary_quota_requires_a_positive_count_and_one_support(change, message):
    arguments = {
        "quota_id": "peak",
        "kind": CollocationKind.BOUNDARY,
        "count": 1,
        "semantic_tag": "left",
    }
    with pytest.raises(ValueError, match=message):
        SemanticCollocationQuota(**{**arguments, **change})


def test_interior_quota_cannot_use_boundary_tags_or_boolean_region_ids():
    quota = SemanticCollocationQuota("region", CollocationKind.INTERIOR, 1, region_id=0)
    assert quota.region_id == 0 and quota.semantic_tag is None
    for change in ({"region_id": True}, {"semantic_tag": "left"}):
        with pytest.raises(ValueError):
            replace(quota, **change)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"algorithm_id": "future"}, "sampler algorithm"),
        ({"seed_commitment_algorithm_id": "future"}, "commitment algorithm"),
        ({"schema_version": "future"}, "stream schema"),
        ({"case_id": ""}, "case_id"),
        ({"campaign_hash": "path/to/data"}, "campaign_hash"),
        ({"role": 1}, "EvidenceRole"),
        ({"role": EvidenceRole.HOLDOUT}, "public role"),
        ({"qoi_quotas": (object(),)}, "SemanticCollocationQuota"),
        ({"quota_policy_hash": "0" * 64}, "does not match qoi_quotas"),
        ({"interior_count": -1}, "non-negative integers"),
        ({"interior_count": 0, "boundary_count": 0}, "at least one"),
        ({"interior_count": 100001}, "exceeds maximum"),
        ({"counter_start": -1}, "counter_start"),
        ({"counter_stop": True}, "counter_stop"),
        ({"counter_start": 2**53, "counter_stop": 2**53 + 3}, "safe-integer"),
        ({"seed_commitment": "0" * 64}, "seed commitment does not match"),
    ],
)
def test_stream_contract_refuses_changed_replay_inputs(batch, change, message):
    with pytest.raises((TypeError, ValueError), match=message):
        replace(batch.stream, **change)


def test_stream_requires_canonical_unique_quota_order(batch):
    first = SemanticCollocationQuota("a", CollocationKind.BOUNDARY, 1, "left")
    last = SemanticCollocationQuota("z", CollocationKind.BOUNDARY, 1, "right")
    for quotas, reason in (
        ((last, first), "canonical order"),
        ((first, first), "unique"),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(batch.stream, qoi_quotas=quotas)


def test_public_role_string_is_normalized_without_altering_replay(batch):
    assert replace(batch.stream, role="TRAIN").stream_hash == batch.stream.stream_hash


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sample_id": ""}, "sample_id"),
        ({"semantic_class": SemanticBCClass.FREE}, "INTERIOR semantics"),
        ({"kind": CollocationKind.INTERFACE}, "INTERFACE semantics"),
        ({"kind": CollocationKind.BOUNDARY}, "boundary-condition semantics"),
        ({"coordinate_unit": Unit.M}, "canonical mm"),
        ({"coordinates": (0.0,)}, "finite 2D"),
        ({"coordinates": (float("nan"), 0.0)}, "finite 2D"),
        ({"source_entity_index": -1}, "source_entity_index"),
        ({"source_node_ids": (0, 0, 1)}, "source_node_ids"),
        ({"source_node_ids": (0, True, 2)}, "source_node_ids"),
        ({"region_ids": ()}, "region_ids"),
        ({"region_ids": (True,)}, "region_ids"),
        ({"semantic_tags": ("",)}, "semantic_tags"),
        ({"counter": 2**53}, "counter"),
        ({"qoi_quota_id": ""}, "qoi_quota_id"),
    ],
)
def test_point_contract_refuses_semantic_unit_and_entity_mismatches(
    batch, change, message
):
    with pytest.raises(ValueError, match=message):
        replace(batch.points[0], **change)


@pytest.mark.parametrize(
    "change",
    [
        "stream",
        "hash",
        "count",
        "type",
        "counter",
        "stream-trace",
        "sample-id",
        "kinds",
    ],
)
def test_batch_contract_checks_population_shape_order_and_identity(batch, change):
    arguments = {}
    points = batch.points
    if change == "stream":
        arguments["stream"] = object()
    elif change == "hash":
        arguments["sampler_stream_hash"] = "0" * 64
    elif change == "count":
        points = points[:-1]
    elif change == "type":
        points = (object(), *points[1:])
    elif change == "counter":
        points = (replace(points[0], counter=1), *points[1:])
    elif change == "stream-trace":
        points = (replace(points[0], sampler_stream_hash="0" * 64), *points[1:])
    elif change == "sample-id":
        points = (replace(points[0], sample_id="another-sample"), *points[1:])
    else:
        points = (
            replace(
                points[0],
                kind=CollocationKind.BOUNDARY,
                semantic_class=SemanticBCClass.FREE,
                source_node_ids=(0, 1),
            ),
            *points[1:],
        )
    with pytest.raises((TypeError, ValueError)):
        replace(batch, points=points, **arguments)


def test_quota_kind_must_match_its_declared_region_support():
    quota = SemanticCollocationQuota("region", CollocationKind.INTERIOR, 1, region_id=0)
    original = sample_collocation(
        snapshot(),
        CASE_ID,
        mesh(),
        (),
        (),
        stream_index=0,
        interior_count=0,
        boundary_count=0,
        interface_count=0,
        qoi_quotas=(quota,),
    )
    point = replace(
        original.points[0],
        kind=CollocationKind.BOUNDARY,
        semantic_class=SemanticBCClass.FREE,
        source_node_ids=(0, 1),
    )
    with pytest.raises(ValueError, match="QoI point counts"):
        replace(original, points=(point,))


@pytest.mark.parametrize("index", [True, 1.5, -1, 100])
def test_boundary_classification_refuses_outside_or_noninteger_facet(index):
    with pytest.raises(ValueError, match="facet_index"):
        classify_boundary_facet(mesh(), index, (), ())


@pytest.mark.parametrize(
    ("conditions", "applied_loads", "message"),
    [
        ((object(),), (), "BoundaryConditionSpec"),
        ((), (object(),), "LoadSpec"),
        (
            (BoundaryConditionSpec("unknown", "left", "unsupported"),),
            (),
            "unsupported boundary",
        ),
        (
            (BoundaryConditionSpec("unknown", "absent", "free"),),
            (),
            "unknown semantic tag",
        ),
        (
            (),
            (LoadSpec("unsupported", "unsupported", "left"),),
            "unsupported boundary load",
        ),
        ((), (LoadSpec("unknown", "traction", "absent"),), "unknown semantic tag"),
    ],
)
def test_semantic_specs_cannot_reference_undeclared_or_unsupported_boundaries(
    conditions, applied_loads, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        classify_boundary_facet(mesh(), 0, conditions, applied_loads)


@pytest.mark.parametrize(
    ("quotas", "message"),
    [
        ((object(),), "SemanticCollocationQuota"),
        (
            (SemanticCollocationQuota("dup", CollocationKind.BOUNDARY, 1, "left"),) * 2,
            "unique",
        ),
        (
            (
                SemanticCollocationQuota(
                    "missing", CollocationKind.BOUNDARY, 1, "absent"
                ),
            ),
            "unknown semantic tag",
        ),
        (
            (
                SemanticCollocationQuota(
                    "missing", CollocationKind.INTERIOR, 1, region_id=99
                ),
            ),
            "unknown region",
        ),
    ],
)
def test_sampler_refuses_unbound_quota_support(quotas, message):
    with pytest.raises((TypeError, ValueError), match=message):
        sample_collocation(
            snapshot(),
            CASE_ID,
            mesh(),
            (),
            (),
            stream_index=0,
            interior_count=1,
            boundary_count=0,
            interface_count=0,
            qoi_quotas=quotas,
        )


def test_sampler_rejects_empty_semantic_quota_region():
    empty = replace(mesh(), semantic_tags={**mesh().semantic_tags, "empty": ()})
    bound = snapshot(mesh_content_hash=hash_spec(empty, "content"))
    quota = SemanticCollocationQuota("empty", CollocationKind.BOUNDARY, 1, "empty")
    with pytest.raises(ValueError, match="no boundary facets"):
        sample_collocation(
            bound,
            CASE_ID,
            empty,
            (),
            (),
            stream_index=0,
            interior_count=1,
            boundary_count=0,
            interface_count=0,
            qoi_quotas=(quota,),
        )


@pytest.mark.parametrize(
    "change", ["case", "mesh", "snapshot", "negative-count", "zero-count", "counter"]
)
def test_sampler_refuses_invalid_contracts_and_population_intervals(change):
    arguments = {
        "snapshot": snapshot(),
        "case_id": CASE_ID,
        "mesh": mesh(),
        "boundary_conditions": (),
        "loads": (),
        "stream_index": 0,
        "interior_count": 1,
        "boundary_count": 0,
        "interface_count": 0,
    }
    if change == "case":
        arguments["case_id"] = ""
    elif change in {"mesh", "snapshot"}:
        arguments[change] = object()
    elif change == "negative-count":
        arguments["boundary_count"] = -1
    elif change == "zero-count":
        arguments["interior_count"] = 0
    else:
        arguments["counter_start"] = -1
    with pytest.raises((TypeError, ValueError)):
        sample_collocation(**arguments)


@pytest.mark.parametrize(
    "operation", ["replay", "verify", "verify-request", "persist", "sample-request"]
)
def test_replay_and_persistence_refuse_untyped_boundaries(batch, operation):
    if operation == "replay":
        arguments = (object(), snapshot(), CASE_ID, mesh(), (), ())
        function = replay_collocation
    elif operation == "verify":
        arguments = (object(), snapshot(), CASE_ID, mesh(), (), ())
        function = verify_collocation_batch
    elif operation == "verify-request":
        arguments = (object(), snapshot(), CASE_ID, request_with_qoi())
        function = verify_request_collocation_batch
    elif operation == "persist":
        arguments = (batch, snapshot(), CASE_ID, request_with_qoi(), object())
        function = persist_collocation_batch
    else:
        with pytest.raises(TypeError, match="FemRequest"):
            sample_request_collocation(
                snapshot(),
                CASE_ID,
                object(),
                stream_index=0,
                interior_count=1,
                boundary_count=0,
                interface_count=0,
                qoi_quotas=(),
            )
        return
    with pytest.raises(TypeError):
        function(*arguments)
