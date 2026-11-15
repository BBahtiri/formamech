from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.doe.publication import (
    PublishedCase,
    PublishedCaseManifest,
    PublishedCaseStatus,
)
from piginot.doe.roles import EvidenceRole


def _case(
    case_id: str = "case-002",
    *,
    group_id: str = "group-002",
    role: EvidenceRole = EvidenceRole.TRAIN,
    status: PublishedCaseStatus = PublishedCaseStatus.ACTIVE,
) -> PublishedCase:
    return PublishedCase(
        case_id=case_id,
        leakage_group_id=group_id,
        role=role,
        primitive_coordinates={"height": 2.0, "width": 1.0},
        derived_coordinates={"net_section_ratio": 0.5},
        strata={"net_section_ratio": 1, "width": 0},
        status=status,
    )


def _manifest(**overrides: object) -> PublishedCaseManifest:
    values: dict[str, object] = {
        "manifest_id": "s9-public-cases",
        "primitive_axes": ("width", "height"),
        "derived_axes": ("net_section_ratio",),
        "stratum_axes": ("width", "net_section_ratio"),
        "cases": (
            _case(),
            _case(
                "case-001",
                group_id="group-001",
                role=EvidenceRole.DEVELOPMENT,
                status=PublishedCaseStatus.RESERVE,
            ),
        ),
    }
    values.update(overrides)
    return PublishedCaseManifest(**values)  # type: ignore[arg-type]


def _payload() -> dict[str, object]:
    return json.loads(_manifest().to_bytes())  # type: ignore[no-any-return]


def _noncanonical_key_order() -> bytes:
    payload = _payload()
    manifest_id = payload.pop("manifest_id")
    reordered = {"manifest_id": manifest_id, **payload}
    return json.dumps(reordered, separators=(",", ":")).encode()


def test_public_manifest_is_immutable_deterministic_and_canonical() -> None:
    manifest = _manifest()

    assert tuple(case.case_id for case in manifest.cases) == (
        "case-001",
        "case-002",
    )
    assert tuple(manifest.cases[0].primitive_coordinates) == ("height", "width")
    assert isinstance(manifest.cases[0].primitive_coordinates, MappingProxyType)
    assert manifest.to_bytes() == canonical_json_bytes(manifest)
    assert PublishedCaseManifest.from_bytes(manifest.to_bytes()) == manifest
    assert manifest.content_hash == hash_spec(manifest, "content")
    assert manifest.content_hash == _manifest().content_hash
    assert (
        manifest.content_hash
        == _manifest(
            primitive_axes=("height", "width"),
            stratum_axes=("net_section_ratio", "width"),
        ).content_hash
    )
    assert (
        manifest.content_hash
        != _manifest(
            cases=(replace(_case(), status=PublishedCaseStatus.RESERVE),)
        ).content_hash
    )

    with pytest.raises(FrozenInstanceError):
        manifest.manifest_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        manifest.cases[0].primitive_coordinates["width"] = 3.0  # type: ignore[index]


def test_public_serialization_contains_no_holdout_or_secret_material() -> None:
    payload = _manifest().to_bytes()

    assert b"HOLDOUT" not in payload
    assert b'"seed"' not in payload
    assert b'"salt"' not in payload
    assert b'"commitment_preimage"' not in payload
    assert b'"holdout_coordinates"' not in payload


@pytest.mark.parametrize(
    "role",
    (
        EvidenceRole.PILOT,
        EvidenceRole.TRAIN,
        EvidenceRole.DEVELOPMENT,
        EvidenceRole.LEGACY,
    ),
)
def test_every_non_holdout_evidence_role_may_be_published(
    role: EvidenceRole,
) -> None:
    case = _case(role=role)

    assert case.role is role
    assert PublishedCaseManifest(
        "allowed-role",
        ("width", "height"),
        ("net_section_ratio",),
        ("width", "net_section_ratio"),
        (case,),
    ).cases == (case,)


def test_holdout_cases_are_rejected_at_both_publication_boundaries() -> None:
    with pytest.raises(ValueError, match="HOLDOUT"):
        _case(role=EvidenceRole.HOLDOUT)

    payload = _payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    case = cases[0]
    assert isinstance(case, dict)
    case["role"] = "HOLDOUT"
    with pytest.raises(ValueError, match="HOLDOUT"):
        PublishedCaseManifest.from_bytes(json.dumps(payload))


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"case_id": ""}, "case_id"),
        ({"case_id": "case:node"}, "case-level"),
        ({"leakage_group_id": ""}, "leakage_group_id"),
        ({"role": "UNKNOWN"}, "UNKNOWN"),
        ({"status": "UNKNOWN"}, "UNKNOWN"),
        ({"primitive_coordinates": {}}, "primitive_coordinates"),
        ({"primitive_coordinates": []}, "primitive_coordinates"),
        ({"derived_coordinates": {}}, "derived_coordinates"),
        ({"primitive_coordinates": {"not valid": 1.0}}, "identifier"),
        ({"primitive_coordinates": {"width": True}}, "finite number"),
        ({"primitive_coordinates": {"width": float("nan")}}, "finite number"),
        ({"derived_coordinates": {"ratio": float("inf")}}, "finite number"),
        (
            {
                "primitive_coordinates": {"width": 1.0},
                "derived_coordinates": {"width": 2.0},
            },
            "disjoint",
        ),
        ({"strata": []}, "strata"),
        ({"strata": {"missing": 0}}, "coordinate axes"),
        ({"strata": {"width": True}}, "non-negative integers"),
        ({"strata": {"width": -1}}, "non-negative integers"),
    ),
)
def test_published_case_rejects_malformed_public_data(
    changes: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "case_id": "case",
        "leakage_group_id": "group",
        "role": EvidenceRole.TRAIN,
        "primitive_coordinates": {"width": 1.0},
        "derived_coordinates": {"ratio": 0.5},
        "strata": {"width": 0, "ratio": 1},
        "status": PublishedCaseStatus.ACTIVE,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        PublishedCase(**values)  # type: ignore[arg-type]


def test_manifest_rejects_duplicate_cases_and_role_split_leakage_groups() -> None:
    duplicate = replace(_case(), leakage_group_id="another-group")
    with pytest.raises(ValueError, match="case IDs must be unique"):
        _manifest(cases=(_case(), duplicate))

    split_role = _case(
        "case-003",
        group_id="group-002",
        role=EvidenceRole.DEVELOPMENT,
    )
    with pytest.raises(ValueError, match="leakage group.*one role"):
        _manifest(cases=(_case(), split_role))


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"manifest_id": ""}, "manifest_id"),
        ({"primitive_axes": {}}, "primitive_axes"),
        ({"primitive_axes": ()}, "primitive_axes"),
        ({"derived_axes": ()}, "derived_axes"),
        ({"stratum_axes": ()}, "stratum_axes"),
        ({"primitive_axes": ("width", "width")}, "unique"),
        ({"derived_axes": ("not valid",)}, "identifiers"),
        ({"derived_axes": ("width",)}, "disjoint"),
        ({"stratum_axes": ("unknown",)}, "coordinate axes"),
        ({"cases": ()}, "cases"),
        ({"cases": (object(),)}, "PublishedCase"),
        ({"schema_version": "2.0"}, "schema version"),
        (
            {
                "primitive_axes": ("width",),
            },
            "primitive coordinate schema",
        ),
        (
            {
                "derived_axes": ("other_ratio",),
                "stratum_axes": ("width", "other_ratio"),
            },
            "derived coordinate schema",
        ),
        (
            {
                "stratum_axes": ("height", "net_section_ratio"),
            },
            "strata schema",
        ),
    ),
)
def test_manifest_rejects_incoherent_schemas(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _manifest(**override)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update({"unknown": 1}), "unknown field"),
        (lambda payload: payload.pop("manifest_id"), "missing field"),
        (lambda payload: payload.update({"$type": "wrong.Type"}), "discriminator"),
        (lambda payload: payload.update({"schema_version": "9.0"}), "schema"),
        (lambda payload: payload.update({"cases": {}}), "cases must be an array"),
        (
            lambda payload: payload.update({"primitive_axes": "width"}),
            "primitive_axes must be an array",
        ),
        (
            lambda payload: payload["cases"][0].update({"unknown": 1}),
            "unknown field",
        ),
        (
            lambda payload: payload["cases"][0].pop("case_id"),
            "missing field",
        ),
        (
            lambda payload: payload["cases"][0].update({"$type": "wrong.Type"}),
            "discriminator",
        ),
        (
            lambda payload: payload["cases"][0].update({"primitive_coordinates": []}),
            "primitive_coordinates must be an object",
        ),
        (
            lambda payload: payload["cases"][0].update({"strata": []}),
            "strata must be an object",
        ),
    ),
)
def test_from_bytes_rejects_unknown_missing_or_mistyped_fields(
    mutation: object, message: str
) -> None:
    payload = _payload()
    mutation(payload)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        PublishedCaseManifest.from_bytes(json.dumps(payload))


@pytest.mark.parametrize("document", (b"[]", b"null", b"true"))
def test_from_bytes_requires_a_manifest_object(document: bytes) -> None:
    with pytest.raises(ValueError, match="manifest must be a JSON object"):
        PublishedCaseManifest.from_bytes(document)


def test_from_bytes_rejects_duplicate_json_object_names() -> None:
    document = _manifest().to_bytes().decode()
    duplicate = document.replace(
        '"manifest_id":"s9-public-cases"',
        '"manifest_id":"s9-public-cases","manifest_id":"other"',
    )

    with pytest.raises(ValueError, match="duplicate JSON object name"):
        PublishedCaseManifest.from_bytes(duplicate)


@pytest.mark.parametrize(
    "document",
    (
        b" " + _manifest().to_bytes(),
        _noncanonical_key_order(),
    ),
)
def test_from_bytes_rejects_noncanonical_whitespace_or_key_order(
    document: bytes,
) -> None:
    with pytest.raises(ValueError, match="canonical JSON"):
        PublishedCaseManifest.from_bytes(document)


def test_from_bytes_rejects_non_text_input() -> None:
    with pytest.raises(ValueError, match="str or bytes"):
        PublishedCaseManifest.from_bytes(1)  # type: ignore[arg-type]
