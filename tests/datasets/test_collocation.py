from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from piginot.contracts import (
    CanonicalMesh,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.datasets.collocation import (
    MAX_COLLOCATION_POINTS,
    CollocationKind,
    CollocationPoint,
    SemanticBCClass,
    SemanticCollocationQuota,
    classify_boundary_facet,
    persist_collocation_batch,
    replay_collocation,
    sample_collocation,
    sample_request_collocation,
    verify_collocation_batch,
)
from piginot.doe import EvidenceRole
from piginot.store import ObjectStore
from tests.datasets.collocation_fixtures import (
    CAMPAIGN_HASH as _CAMPAIGN_HASH,
)
from tests.datasets.collocation_fixtures import CASE_ID as _CASE_ID
from tests.datasets.collocation_fixtures import (
    MANIFEST_HASH as _MANIFEST_HASH,
)
from tests.datasets.collocation_fixtures import (
    CollocationCounts as _CollocationCounts,
)
from tests.datasets.collocation_fixtures import (
    boundary_conditions as _boundary_conditions,
)
from tests.datasets.collocation_fixtures import loads as _loads
from tests.datasets.collocation_fixtures import mesh as _mesh
from tests.datasets.collocation_fixtures import request_with_qoi as _request_with_qoi
from tests.datasets.collocation_fixtures import snapshot as _snapshot


def test_collocation_point_rejects_boolean_coordinates() -> None:
    with pytest.raises(ValueError, match="finite 2D point"):
        CollocationPoint(
            sample_id="case-001:collocation:0000000000000000",
            case_id="case-001",
            leakage_group_id="group-001",
            role=EvidenceRole.TRAIN,
            kind=CollocationKind.INTERIOR,
            semantic_class=SemanticBCClass.INTERIOR,
            coordinates=(True, 0.0),
            source_entity_index=0,
            source_node_ids=(0, 1, 2),
            region_ids=(0,),
            semantic_tags=(),
            qoi_quota_id=None,
            counter=0,
            sampler_stream_hash="a" * 64,
        )


def test_boundary_classification_uses_semantic_tags_and_physics_specs() -> None:
    mesh = _mesh()
    conditions = _boundary_conditions()
    loads = _loads()

    assert classify_boundary_facet(mesh, 3, conditions, loads) == (
        SemanticBCClass.DIRICHLET,
        ("left",),
    )
    assert classify_boundary_facet(mesh, 1, conditions, loads) == (
        SemanticBCClass.MIXED,
        ("right",),
    )
    assert classify_boundary_facet(mesh, 2, conditions, loads) == (
        SemanticBCClass.NEUMANN,
        ("top",),
    )
    assert classify_boundary_facet(mesh, 0, conditions, loads) == (
        SemanticBCClass.FREE,
        ("bottom",),
    )


def test_recorded_stream_replays_byte_identically_and_normalizes_spec_order() -> None:
    first = sample_collocation(
        _snapshot(),
        _CASE_ID,
        _mesh(),
        tuple(reversed(_boundary_conditions())),
        tuple(reversed(_loads())),
        stream_index=7,
        interior_count=12,
        boundary_count=24,
        interface_count=8,
        counter_start=17,
    )
    reordered = sample_collocation(
        _snapshot(),
        _CASE_ID,
        _mesh(),
        _boundary_conditions(),
        _loads(),
        stream_index=7,
        interior_count=12,
        boundary_count=24,
        interface_count=8,
        counter_start=17,
    )
    replayed = replay_collocation(
        first.stream,
        _snapshot(),
        _CASE_ID,
        _mesh(),
        _boundary_conditions(),
        _loads(),
    )

    assert first.to_bytes() == canonical_json_bytes(first)
    assert reordered.to_bytes() == first.to_bytes()
    assert replayed.to_bytes() == first.to_bytes()
    assert first.sampler_stream_hash == first.stream.stream_hash
    assert first.stream.algorithm_id == "blake2b-keyed-counter-u53-v1"
    assert first.stream.seed_commitment_algorithm_id.endswith("-v1")
    assert len(first.stream.replay_seed_hex) == 64
    assert len(first.stream.seed_commitment) == 64
    assert first.stream.campaign_hash == _CAMPAIGN_HASH
    assert first.stream.public_case_manifest_hash == _MANIFEST_HASH
    assert first.stream.snapshot_content_hash == _snapshot().content_hash
    assert first.stream.stream_index == 7
    assert first.stream.counter_start == 17
    assert first.stream.counter_stop == 61
    assert tuple(point.counter for point in first.points) == tuple(range(17, 61))
    assert all(
        point.sampler_stream_hash == first.sampler_stream_hash for point in first.points
    )


def test_samples_cover_interior_boundary_and_material_interface() -> None:
    batch = sample_collocation(
        _snapshot(),
        _CASE_ID,
        _mesh(),
        _boundary_conditions(),
        _loads(),
        stream_index=0,
        interior_count=20,
        boundary_count=80,
        interface_count=20,
    )
    by_kind = {
        kind: tuple(point for point in batch.points if point.kind is kind)
        for kind in CollocationKind
    }

    assert len(by_kind[CollocationKind.INTERIOR]) == 20
    assert len(by_kind[CollocationKind.BOUNDARY]) == 80
    assert len(by_kind[CollocationKind.INTERFACE]) == 20
    assert {point.semantic_class for point in by_kind[CollocationKind.BOUNDARY]} == {
        SemanticBCClass.DIRICHLET,
        SemanticBCClass.NEUMANN,
        SemanticBCClass.MIXED,
        SemanticBCClass.FREE,
    }
    assert all(
        point.semantic_class is SemanticBCClass.INTERIOR
        for point in by_kind[CollocationKind.INTERIOR]
    )
    assert all(
        point.semantic_class is SemanticBCClass.INTERFACE
        and point.source_node_ids == (0, 2)
        and point.region_ids == (0, 1)
        and point.coordinates[0] == pytest.approx(point.coordinates[1])
        for point in by_kind[CollocationKind.INTERFACE]
    )


def test_entity_selection_is_weighted_by_physical_measure() -> None:
    mesh = CanonicalMesh(
        mesh_id="unequal-elements",
        nodes=(
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (10.0, 0.0),
            (20.0, 0.0),
            (10.0, 10.0),
        ),
        cells=((0, 1, 2), (3, 4, 5)),
        cell_type="triangle",
        boundary_facets=((0, 1), (3, 4)),
        semantic_tags={"short": (0,), "long": (1,)},
        region_ids=(0, 0),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature="unequal-v1",
        quality_metrics={"minimum_scaled_jacobian": 1.0},
        source_provenance={"fixture": "generic-unequal-elements"},
    )
    batch = sample_collocation(
        _snapshot(
            mesh_id=mesh.mesh_id,
            mesh_content_hash=hash_spec(mesh, "content"),
        ),
        _CASE_ID,
        mesh,
        (),
        (),
        stream_index=0,
        interior_count=300,
        boundary_count=300,
        interface_count=0,
    )
    large_cell = sum(
        point.source_entity_index == 1
        for point in batch.points
        if point.kind is CollocationKind.INTERIOR
    )
    long_facet = sum(
        point.source_entity_index == 1
        for point in batch.points
        if point.kind is CollocationKind.BOUNDARY
    )

    assert large_cell >= 285  # area ratio 100:1, not uniform-by-cell sampling
    assert long_facet >= 260  # length ratio 10:1, not uniform-by-facet sampling


def test_tiny_semantic_qoi_entity_receives_its_measure_independent_quota() -> None:
    mesh = CanonicalMesh(
        mesh_id="tiny-qoi-edge",
        nodes=((0.0, 0.0), (0.001, 0.0), (0.0, 1.0), (100.0, 0.0)),
        cells=((0, 1, 2), (1, 3, 2)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 3), (3, 2), (2, 0)),
        semantic_tags={"qoi_root": (0,), "far_boundary": (1, 2, 3)},
        region_ids=(0, 0),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature="tiny-qoi-edge-v1",
        quality_metrics={"minimum_scaled_jacobian": 0.001},
        source_provenance={"fixture": "generic-tiny-qoi-edge"},
    )
    quota = SemanticCollocationQuota(
        quota_id="peak_region",
        kind=CollocationKind.BOUNDARY,
        count=17,
        semantic_tag="qoi_root",
    )
    batch = sample_collocation(
        _snapshot(
            mesh_id=mesh.mesh_id,
            mesh_content_hash=hash_spec(mesh, "content"),
        ),
        _CASE_ID,
        mesh,
        (),
        (),
        stream_index=0,
        interior_count=0,
        boundary_count=0,
        interface_count=0,
        qoi_quotas=(quota,),
    )

    assert batch.stream.qoi_quotas == (quota,)
    assert batch.stream.qoi_quota_count == 17
    assert len(batch.points) == 17
    assert all(
        point.qoi_quota_id == "peak_region"
        and point.source_entity_index == 0
        and point.semantic_tags == ("qoi_root",)
        for point in batch.points
    )
    assert (
        replay_collocation(
            batch.stream,
            _snapshot(
                mesh_id=mesh.mesh_id,
                mesh_content_hash=hash_spec(mesh, "content"),
            ),
            _CASE_ID,
            mesh,
            (),
            (),
        ).to_bytes()
        == batch.to_bytes()
    )


def test_request_sampler_requires_every_declared_qoi_region_to_have_a_quota() -> None:
    request = _request_with_qoi()
    snapshot = _snapshot(request=request)
    quota = SemanticCollocationQuota(
        quota_id="peak_left",
        kind=CollocationKind.BOUNDARY,
        count=7,
        semantic_tag="left",
    )

    with pytest.raises(ValueError, match="QoI region.*left"):
        sample_request_collocation(
            snapshot,
            _CASE_ID,
            request,
            stream_index=0,
            interior_count=2,
            boundary_count=2,
            interface_count=1,
            qoi_quotas=(),
        )

    batch = sample_request_collocation(
        snapshot,
        _CASE_ID,
        request,
        stream_index=0,
        interior_count=2,
        boundary_count=2,
        interface_count=1,
        qoi_quotas=(quota,),
    )

    assert sum(point.qoi_quota_id == "peak_left" for point in batch.points) == 7
    assert all(point.coordinate_unit is Unit.MM for point in batch.points)

    changed = replace(
        request,
        boundary_conditions=request.boundary_conditions[:-1],
    )
    with pytest.raises(ValueError, match="request"):
        sample_request_collocation(
            snapshot,
            _CASE_ID,
            changed,
            stream_index=0,
            interior_count=1,
            boundary_count=1,
            interface_count=1,
            qoi_quotas=(quota,),
        )


def test_stream_and_batch_constructors_reject_cross_boundary_provenance() -> None:
    batch = sample_collocation(
        _snapshot(),
        _CASE_ID,
        _mesh(),
        _boundary_conditions(),
        _loads(),
        stream_index=0,
        interior_count=1,
        boundary_count=1,
        interface_count=1,
    )

    with pytest.raises(ValueError, match="public role"):
        replace(batch.stream, role=EvidenceRole.HOLDOUT)
    with pytest.raises(ValueError, match="digest"):
        replace(batch.stream, mesh_content_hash="not-a-digest")

    foreign = replace(batch.points[0], role=EvidenceRole.DEVELOPMENT)
    with pytest.raises(ValueError, match="case provenance"):
        replace(batch, points=(foreign, *batch.points[1:]))


def test_sampler_requires_exact_successful_snapshot_case_binding() -> None:
    with pytest.raises(ValueError, match="successful"):
        sample_collocation(
            _snapshot(success=False),
            _CASE_ID,
            _mesh(),
            _boundary_conditions(),
            _loads(),
            stream_index=0,
            interior_count=1,
            boundary_count=1,
            interface_count=1,
        )
    with pytest.raises(TypeError, match="RoleDatasetSnapshot"):
        sample_collocation(
            object(),  # type: ignore[arg-type]
            _CASE_ID,
            _mesh(),
            _boundary_conditions(),
            _loads(),
            stream_index=0,
            interior_count=1,
            boundary_count=1,
            interface_count=1,
        )
    with pytest.raises(ValueError, match="case_id"):
        sample_collocation(
            _snapshot(),
            "unknown-case",
            _mesh(),
            _boundary_conditions(),
            _loads(),
            stream_index=0,
            interior_count=1,
            boundary_count=1,
            interface_count=1,
        )
    with pytest.raises(ValueError, match="mesh"):
        sample_collocation(
            _snapshot(mesh_id="other-mesh"),
            _CASE_ID,
            _mesh(),
            _boundary_conditions(),
            _loads(),
            stream_index=0,
            interior_count=1,
            boundary_count=1,
            interface_count=1,
        )
    changed_nodes = replace(
        _mesh(),
        nodes=((0.0, 0.0), (1.1, 0.0), (1.0, 1.0), (0.0, 1.0)),
    )
    with pytest.raises(ValueError, match="mesh"):
        sample_collocation(
            _snapshot(),
            _CASE_ID,
            changed_nodes,
            _boundary_conditions(),
            _loads(),
            stream_index=0,
            interior_count=1,
            boundary_count=1,
            interface_count=1,
        )
    with pytest.raises(ValueError, match="interface"):
        no_interface_mesh = _mesh(regions=(0, 0))
        sample_collocation(
            _snapshot(mesh_content_hash=hash_spec(no_interface_mesh, "content")),
            _CASE_ID,
            no_interface_mesh,
            _boundary_conditions(),
            _loads(),
            stream_index=0,
            interior_count=0,
            boundary_count=0,
            interface_count=1,
        )


def test_replay_detects_changed_public_lineage_mesh_or_semantic_specs() -> None:
    batch = sample_collocation(
        _snapshot(),
        _CASE_ID,
        _mesh(),
        _boundary_conditions(),
        _loads(),
        stream_index=0,
        interior_count=1,
        boundary_count=1,
        interface_count=1,
    )

    with pytest.raises(ValueError, match="derived public replay seed"):
        replace(batch.stream, replay_seed_hex="abcdef0123456789" * 4)
    with pytest.raises(ValueError, match="derived public replay seed"):
        replace(batch.stream, campaign_hash="f" * 64)
    with pytest.raises(ValueError, match="mesh"):
        replay_collocation(
            batch.stream,
            _snapshot(),
            _CASE_ID,
            replace(_mesh(), mesh_id="changed-mesh"),
            _boundary_conditions(),
            _loads(),
        )
    with pytest.raises(ValueError, match="semantic"):
        replay_collocation(
            batch.stream,
            _snapshot(),
            _CASE_ID,
            _mesh(),
            _boundary_conditions()[:-1],
            _loads(),
        )


def test_public_stream_index_derives_seed_without_caller_supplied_material() -> None:
    common: _CollocationCounts = {
        "interior_count": 2,
        "boundary_count": 2,
        "interface_count": 1,
    }
    first = sample_collocation(
        _snapshot(),
        _CASE_ID,
        _mesh(),
        _boundary_conditions(),
        _loads(),
        stream_index=0,
        **common,
    )
    second = sample_collocation(
        _snapshot(),
        _CASE_ID,
        _mesh(),
        _boundary_conditions(),
        _loads(),
        stream_index=1,
        **common,
    )

    assert first.stream.replay_seed_hex != second.stream.replay_seed_hex
    assert first.sampler_stream_hash != second.sampler_stream_hash
    with pytest.raises(TypeError, match="replay_seed_hex"):
        sample_collocation(
            _snapshot(),
            _CASE_ID,
            _mesh(),
            _boundary_conditions(),
            _loads(),
            stream_index=0,
            replay_seed_hex="00" * 32,  # type: ignore[call-arg]
            **common,
        )


@pytest.mark.parametrize("stream_index", (-1, True))
def test_stream_index_must_be_a_public_non_negative_integer(
    stream_index: int,
) -> None:
    with pytest.raises(ValueError, match="stream_index"):
        sample_collocation(
            _snapshot(),
            _CASE_ID,
            _mesh(),
            _boundary_conditions(),
            _loads(),
            stream_index=stream_index,
            interior_count=1,
            boundary_count=0,
            interface_count=0,
        )


def test_replay_requires_the_exact_bound_snapshot() -> None:
    snapshot = _snapshot()
    batch = sample_collocation(
        snapshot,
        _CASE_ID,
        _mesh(),
        _boundary_conditions(),
        _loads(),
        stream_index=0,
        interior_count=1,
        boundary_count=1,
        interface_count=1,
    )
    other_snapshot = replace(snapshot, campaign_hash="f" * 64)

    with pytest.raises(ValueError, match="snapshot"):
        replay_collocation(
            batch.stream,
            other_snapshot,
            _CASE_ID,
            _mesh(),
            _boundary_conditions(),
            _loads(),
        )


def test_verification_and_persistence_reject_shape_valid_point_tampering(
    tmp_path: Path,
) -> None:
    request = replace(_request_with_qoi(), qois=())
    snapshot = _snapshot(request=request)
    batch = sample_collocation(
        snapshot,
        _CASE_ID,
        request.mesh,
        request.boundary_conditions,
        request.loads,
        stream_index=4,
        interior_count=1,
        boundary_count=1,
        interface_count=1,
    )
    original = batch.points[0]
    forged_points = (
        replace(original, coordinates=(0.125, 0.875)),
        replace(original, source_entity_index=99),
        replace(original, semantic_tags=("forged-tag",)),
    )
    store = ObjectStore.create(tmp_path / "objects")

    for forged_point in forged_points:
        forged = replace(batch, points=(forged_point, *batch.points[1:]))
        with pytest.raises(ValueError, match="deterministic replay"):
            verify_collocation_batch(
                forged,
                snapshot,
                _CASE_ID,
                request.mesh,
                request.boundary_conditions,
                request.loads,
            )
        with pytest.raises(ValueError, match="deterministic replay"):
            persist_collocation_batch(
                forged,
                snapshot,
                _CASE_ID,
                request,
                store,
            )

    object_root = store.root / "objects"
    assert not object_root.exists() or not any(
        path.is_file() for path in object_root.rglob("*")
    )


def test_verified_batch_persists_with_stream_batch_and_store_identities(
    tmp_path: Path,
) -> None:
    request = replace(_request_with_qoi(), qois=())
    snapshot = _snapshot(request=request)
    batch = sample_collocation(
        snapshot,
        _CASE_ID,
        request.mesh,
        request.boundary_conditions,
        request.loads,
        stream_index=2,
        interior_count=2,
        boundary_count=2,
        interface_count=1,
    )
    store = ObjectStore.create(tmp_path / "objects")

    assert (
        verify_collocation_batch(
            batch,
            snapshot,
            _CASE_ID,
            request.mesh,
            request.boundary_conditions,
            request.loads,
        )
        is batch
    )
    reference = persist_collocation_batch(
        batch,
        snapshot,
        _CASE_ID,
        request,
        store,
    )

    assert reference.sampler_stream_hash == batch.sampler_stream_hash
    assert reference.batch_content_hash == batch.content_hash
    assert reference.snapshot_content_hash == snapshot.content_hash
    assert store.get(reference.store_content_hash) == batch.to_bytes()


def test_persistence_requires_semantics_from_the_snapshot_bound_request(
    tmp_path: Path,
) -> None:
    request = replace(_request_with_qoi(), qois=())
    snapshot = _snapshot(request=request)
    batch = sample_collocation(
        snapshot,
        _CASE_ID,
        request.mesh,
        request.boundary_conditions,
        request.loads,
        stream_index=3,
        interior_count=1,
        boundary_count=1,
        interface_count=1,
    )
    changed = replace(
        request,
        boundary_conditions=request.boundary_conditions[:-1],
    )
    store = ObjectStore.create(tmp_path / "objects")

    with pytest.raises(ValueError, match="request"):
        persist_collocation_batch(batch, snapshot, _CASE_ID, changed, store)


def test_persistence_cannot_bypass_request_qoi_quotas(tmp_path: Path) -> None:
    request = _request_with_qoi()
    snapshot = _snapshot(request=request)
    batch = sample_collocation(
        snapshot,
        _CASE_ID,
        request.mesh,
        request.boundary_conditions,
        request.loads,
        stream_index=3,
        interior_count=1,
        boundary_count=1,
        interface_count=1,
    )
    store = ObjectStore.create(tmp_path / "objects")

    with pytest.raises(ValueError, match="QoI region.*left"):
        persist_collocation_batch(batch, snapshot, _CASE_ID, request, store)


def test_sampler_refuses_memory_exhausting_point_population() -> None:
    with pytest.raises(ValueError, match="maximum"):
        sample_collocation(
            _snapshot(),
            _CASE_ID,
            _mesh(),
            _boundary_conditions(),
            _loads(),
            stream_index=0,
            interior_count=MAX_COLLOCATION_POINTS + 1,
            boundary_count=0,
            interface_count=0,
        )
