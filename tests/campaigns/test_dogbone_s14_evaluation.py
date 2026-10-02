from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from campaigns.dogbone_v1.family import compiled_family
from campaigns.dogbone_v1.s10_requests import _load_public_sources
from campaigns.dogbone_v1.s11_contingency_source import _rebuild_request
from campaigns.dogbone_v1.s14_candidate import S14Case
from campaigns.dogbone_v1.s14_evaluation import (
    REVEAL_SCHEMA,
    DevelopmentAccess,
    consume_and_reveal_holdout,
    holdout_cases,
    holdout_commitment,
    open_development,
    parse_reveal,
    s10_request,
)
from piginot.contracts import ExecutionProfile
from piginot.doe import EvidenceRole, HoldoutCommitment
from piginot.doe.commitment import _compute
from piginot.verification.holdout import ConsumptionRecord, HoldoutAlreadyConsumedError

_HASH = "ab" * 32


def _active_cases():  # type: ignore[no-untyped-def]
    _, manifest = _load_public_sources()
    return [case for case in manifest.cases if case.status.value == "ACTIVE"]


def _profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version="3.11.9",
        library_versions={"numpy": "2.2.6"},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
        container_digest=None,
    )


def test_holdout_strata_reproduce_the_public_s9_strata() -> None:
    cases = _active_cases()[:40]
    coordinates = [
        {**case.primitive_coordinates, **case.derived_coordinates} for case in cases
    ]
    rebuilt = holdout_cases(coordinates)

    assert [c.strata for c in rebuilt] == [dict(c.strata) for c in cases]
    assert all(c.role is EvidenceRole.HOLDOUT for c in rebuilt)
    assert rebuilt[3].case_id == "holdout-03"
    with pytest.raises(ValueError):
        holdout_cases([{"L_total": 60.0}])


def test_s10_request_is_the_s10_recipe_for_any_role() -> None:
    case = _active_cases()[0]
    profile = _profile()
    expected = _rebuild_request(case, profile, compiled_family())

    assert s10_request(S14Case.from_published(case), profile) == expected
    holdout = S14Case(
        "holdout-00",
        EvidenceRole.HOLDOUT,
        "holdout-00",
        case.primitive_coordinates,
        case.derived_coordinates,
        case.strata,
    )
    assert s10_request(holdout, profile).mesh == expected.mesh
    outside = S14Case(
        "x",
        EvidenceRole.HOLDOUT,
        "x",
        {**case.primitive_coordinates, "W_gauge": 1.0},
        case.derived_coordinates,
        case.strata,
    )
    with pytest.raises(ValueError):
        s10_request(outside, profile)


def _reveal(coordinates, salt: bytes = bytes(range(7, 39))):  # type: ignore[no-untyped-def]
    commitment = _compute(
        record_type=HoldoutCommitment,
        expected_count=64,
        campaign_hash=_HASH,
        salt=salt,
        holdout_coordinates=coordinates,
        created_utc="2026-10-02T00:00:00Z",
    )
    payload = json.dumps(
        {
            "schema": REVEAL_SCHEMA,
            "campaign_hash": _HASH,
            "holdout_salt_hex": salt.hex(),
            "holdout_coordinates": coordinates,
        }
    ).encode()
    return commitment, payload


def _record() -> ConsumptionRecord:
    return ConsumptionRecord("dogbone-v1", _HASH, _HASH, "2026-10-02T00:00:00Z", "S14")


def test_holdout_is_consumed_before_the_verified_reveal(tmp_path: Path) -> None:
    coordinates = [
        {**c.primitive_coordinates, **c.derived_coordinates}
        for c in _active_cases()[:64]
    ]
    commitment, payload = _reveal(coordinates)
    reveal = tmp_path / "reveal.json"
    reveal.write_bytes(payload)
    record_path = tmp_path / "consumed.json"

    revealed = consume_and_reveal_holdout(
        record_path=record_path,
        record=_record(),
        reveal_path=reveal,
        commitment=commitment,
    )
    assert len(revealed) == 64 and record_path.is_file()
    with pytest.raises(HoldoutAlreadyConsumedError):
        consume_and_reveal_holdout(
            record_path=record_path,
            record=_record(),
            reveal_path=reveal,
            commitment=commitment,
        )

    tampered = json.loads(payload)
    tampered["holdout_coordinates"][0]["L_total"] += 1.0
    with pytest.raises(ValueError):
        parse_reveal(json.dumps(tampered).encode(), commitment)
    with pytest.raises(ValueError):
        parse_reveal(json.dumps({**tampered, "schema": "x"}).encode(), commitment)


def test_a_failed_reveal_still_leaves_the_holdout_consumed(tmp_path: Path) -> None:
    coordinates = [
        {**c.primitive_coordinates, **c.derived_coordinates}
        for c in _active_cases()[:64]
    ]
    commitment, _ = _reveal(coordinates)
    record_path = tmp_path / "consumed.json"
    with pytest.raises(FileNotFoundError):
        consume_and_reveal_holdout(
            record_path=record_path,
            record=_record(),
            reveal_path=tmp_path / "missing.json",
            commitment=commitment,
        )
    assert record_path.is_file()


def test_public_commitment_is_the_cp_envelope_record() -> None:
    commitment = holdout_commitment()
    assert commitment.n_holdout_points == 64
    assert commitment.scheme == "blake2b-256"


def test_development_opens_once_after_its_record(tmp_path: Path) -> None:
    ids = tuple(f"candidate-{i:06d}" for i in range(64))
    loaded: list[str] = []
    source = SimpleNamespace(
        _protocol=SimpleNamespace(development_case_ids=ids),
        _load_case=lambda case_id, role: loaded.append(case_id) or (case_id, role),
    )
    access = DevelopmentAccess(_HASH, "Betim Bahtiri", "t0", "t1", ids)
    record = tmp_path / "access.json"

    cases = list(open_development(source, record, access))
    assert record.is_file() and loaded == list(ids)
    assert cases[0] == (ids[0], EvidenceRole.DEVELOPMENT)
    with pytest.raises(RuntimeError):
        open_development(source, record, access)
    other = DevelopmentAccess(_HASH, "Betim Bahtiri", "t0", "t1", ids[::-1])
    with pytest.raises(ValueError):
        open_development(source, tmp_path / "b.json", other)
    with pytest.raises(ValueError):
        DevelopmentAccess("x", "Betim Bahtiri", "t0", "t1", ids)
    with pytest.raises(ValueError):
        DevelopmentAccess(_HASH, "Betim Bahtiri", "t0", "t1", ids[:3])


def test_free_traction_components_follow_the_boundary_conditions() -> None:
    from campaigns.dogbone_v1.s14_candidate import free_traction_components

    request = s10_request(S14Case.from_published(_active_cases()[0]), _profile())
    facets, free = free_traction_components(request)
    by_tag = {
        tag: {free[index] for index in indices}
        for tag, indices in request.mesh.semantic_tags.items()
    }

    assert len(facets) == len(free) == len(request.mesh.boundary_facets)
    assert by_tag == {
        "bottom": {(0,)},
        "left_symmetry": {(1,)},
        "right_grip": {(1,)},
        "gauge_top": {(0, 1)},
        "right_arc": {(0, 1)},
    }
