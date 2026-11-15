from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import rfc8785

from piginot.doe.publication import (
    PublishedCase,
    PublishedCaseManifest,
    PublishedCaseStatus,
)
from piginot.doe.roles import EvidenceRole


def _case(case_id: str = "load-001", **changes) -> PublishedCase:
    return PublishedCase(
        **{
            "case_id": case_id,
            "leakage_group_id": case_id,
            "role": EvidenceRole.TRAIN,
            "primitive_coordinates": {"load": 0.01, "elastic_modulus": 760.0},
            "derived_coordinates": {},
            "strata": {},
            "status": PublishedCaseStatus.ACTIVE,
            **changes,
        }
    )


def _manifest(**changes) -> PublishedCaseManifest:
    return PublishedCaseManifest(
        **{
            "manifest_id": "fixed-cad-load-material",
            "primitive_axes": ("load", "elastic_modulus"),
            "derived_axes": (),
            "stratum_axes": (),
            "cases": (
                _case(),
                _case(
                    "load-002",
                    primitive_coordinates={"load": 0.02, "elastic_modulus": 800.0},
                ),
            ),
            **changes,
        }
    )


def test_fixed_geometry_needs_no_fabricated_derived_axis_or_stratum() -> None:
    manifest = _manifest()
    assert manifest.primitive_axes == ("elastic_modulus", "load")
    assert manifest.derived_axes == manifest.stratum_axes == ()
    assert all(
        not case.derived_coordinates and not case.strata for case in manifest.cases
    )
    assert PublishedCaseManifest.from_bytes(manifest.to_bytes()) == manifest
    assert (
        PublishedCaseManifest.from_bytes(manifest.to_bytes().decode()).content_hash
        == manifest.content_hash
    )
    changed = replace(
        manifest,
        cases=(_case(primitive_coordinates={"load": 0.012, "elastic_modulus": 760.0}),),
    )
    assert changed.content_hash != manifest.content_hash


def test_unstratified_publication_still_snapshots_coordinates() -> None:
    coordinates = {"load": 0.01, "elastic_modulus": 760.0}
    case = _case(primitive_coordinates=coordinates)
    coordinates["load"] = 0.99
    assert case.primitive_coordinates["load"] == 0.01
    with pytest.raises(TypeError):
        case.derived_coordinates["later"] = 1.0
    with pytest.raises(TypeError):
        case.strata["later"] = 0


def test_empty_primitive_coordinates_and_axes_are_still_forbidden() -> None:
    with pytest.raises(ValueError, match="primitive_coordinates"):
        _case(primitive_coordinates={})
    with pytest.raises(ValueError, match="primitive_axes"):
        _manifest(primitive_axes=())


@pytest.mark.parametrize(
    "changes",
    [
        {"primitive_axes": ("load",)},
        {"derived_axes": ("ratio",)},
        {"stratum_axes": ("load",)},
        {"stratum_axes": ("missing",)},
    ],
)
def test_empty_optional_axes_do_not_relax_schema_coherence(changes) -> None:
    with pytest.raises(ValueError):
        _manifest(**changes)


def test_nonempty_derived_coordinates_can_still_be_unstratified() -> None:
    case = _case(derived_coordinates={"ratio": 0.25})
    manifest = _manifest(derived_axes=("ratio",), cases=(case,))
    assert not manifest.stratum_axes
    assert PublishedCaseManifest.from_bytes(manifest.to_bytes()) == manifest
    with pytest.raises(ValueError, match="derived"):
        _manifest(cases=(case,))


def test_strata_must_still_refer_to_real_coordinates() -> None:
    with pytest.raises(ValueError, match="coordinate axes"):
        _case(strata={"unknown": 0})
    with pytest.raises(ValueError, match="strata"):
        _manifest(cases=(_case(strata={"load": 0}),))


def test_holdout_and_mixed_leakage_roles_stay_forbidden() -> None:
    with pytest.raises(ValueError, match="HOLDOUT"):
        _case(role=EvidenceRole.HOLDOUT)
    with pytest.raises(ValueError, match="exactly one role"):
        _manifest(
            cases=(
                _case(),
                _case(
                    "load-002",
                    leakage_group_id="load-001",
                    role=EvidenceRole.DEVELOPMENT,
                ),
            )
        )


def test_unknown_payload_fields_and_noncanonical_bytes_stay_forbidden() -> None:
    manifest = _manifest()
    document = json.loads(manifest.to_bytes())
    document["cases"][0]["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        PublishedCaseManifest.from_bytes(rfc8785.dumps(document))
    with pytest.raises(ValueError, match="canonical"):
        PublishedCaseManifest.from_bytes(
            json.dumps(json.loads(manifest.to_bytes()), indent=2)
        )


@pytest.mark.parametrize("with_strata", [False, True])
def test_manifest_bytes_are_unchanged_by_file_roundtrip(
    tmp_path: Path, with_strata: bool
) -> None:
    manifest = _manifest()
    if with_strata:
        manifest = _manifest(
            derived_axes=("ratio",),
            stratum_axes=("load",),
            cases=tuple(
                replace(case, derived_coordinates={"ratio": 0.25}, strata={"load": i})
                for i, case in enumerate(manifest.cases)
            ),
        )
    original = manifest.to_bytes()
    path = tmp_path / "source.json"
    path.write_bytes(original)
    assert PublishedCaseManifest.from_bytes(path.read_bytes()).to_bytes() == original
