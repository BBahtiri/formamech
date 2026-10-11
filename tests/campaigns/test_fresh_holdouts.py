"""G20.holdouts: fresh sealed H1/H2 sets, disjoint, balanced and committed."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from campaigns import fresh_holdouts as fh
from piginot.contracts import canonical_json_bytes
from piginot.doe.commitment import compute_holdout_commitment
from piginot.doe.design import normalized_coordinates, weighted_distance
from piginot.doe.secret_material import generate_campaign_secrets

CAMPAIGN = "notched-v1"


def _inputs(root: Path = fh.REPOSITORY) -> dict[str, Any]:
    adapter = fh._adapters()[CAMPAIGN]
    return {
        "manifest": json.loads(
            (root / adapter.artifacts / "public-case-manifest.json").read_text("utf-8")
        ),
        "committed_envelope": (
            root / adapter.artifacts / "executable-envelope.json"
        ).read_bytes(),
    }


@pytest.fixture(scope="module")
def first_draw() -> tuple[Any, fh.FreshDraw]:
    secrets = generate_campaign_secrets()
    result = fh.draw(
        fh._adapters()[CAMPAIGN],
        secrets=secrets,
        old_holdout=[],
        created_utc="2026-10-11T00:00:00Z",
        **_inputs(),
    )
    return secrets, result


def test_sets_are_full_disjoint_and_balanced(
    first_draw: tuple[Any, fh.FreshDraw],
) -> None:
    _, result = first_draw
    ids = {
        name: [case_id for case_id, _ in members]
        for name, members in result.sets.items()
    }
    assert [len(ids[name]) for name in fh.SETS] == [64, 64]
    assert not set(ids["H1"]) & set(ids["H2"])
    points = [p.all_coordinates for members in result.sets.values() for _, p in members]
    assert len({canonical_json_bytes(dict(p)) for p in points}) == 128
    manifest = _inputs()["manifest"]
    used = {case["case_id"] for case in manifest["cases"]}
    assert not used & {case_id for members in ids.values() for case_id in members}


def test_no_fresh_case_is_a_near_duplicate_of_an_existing_one(
    first_draw: tuple[Any, fh.FreshDraw],
) -> None:
    _, result = first_draw
    adapter = fh._adapters()[CAMPAIGN]
    envelope = adapter.envelope()
    axes = result.spec.axes
    existing = [
        normalized_coordinates(envelope, axes, p)
        for p in fh.existing_points(_inputs()["manifest"], [])
    ]
    tolerance = result.spec.leakage_policy.near_duplicate_tolerance
    for members in result.sets.values():
        for _, point in members:
            normal = normalized_coordinates(envelope, axes, point)
            assert min(weighted_distance(normal, e, axes) for e in existing) > tolerance


def test_spent_holdout_coordinates_are_rejected(
    first_draw: tuple[Any, fh.FreshDraw],
) -> None:
    """Same secrets, so the same design: its own H1 cases now count as spent."""

    secrets, result = first_draw
    spent = [dict(p.all_coordinates) for _, p in result.sets["H1"][:10]]
    again = fh.draw(
        fh._adapters()[CAMPAIGN],
        secrets=secrets,
        old_holdout=spent,
        created_utc="2026-10-11T00:00:00Z",
        **_inputs(),
    )
    assert again.rejected_near_duplicates >= result.rejected_near_duplicates + 10
    fresh = {
        canonical_json_bytes(dict(p.all_coordinates))
        for members in again.sets.values()
        for _, p in members
    }
    assert not fresh & {canonical_json_bytes(c) for c in spent}


def test_seals_verify_and_detect_tampering(
    first_draw: tuple[Any, fh.FreshDraw],
) -> None:
    _, result = first_draw
    public = fh.public_document(result)
    private = json.loads(canonical_json_bytes(fh.private_document(result)))
    for name in fh.SETS:
        assert len(fh.verify_set_reveal(public, private, name)) == 64
    assert "coordinates" not in json.dumps(public)
    tampered = json.loads(json.dumps(private))
    first = tampered["sets"]["H2"][0]["coordinates"]
    key = sorted(first)[0]
    first[key] = first[key] * (1 + 1e-9)
    with pytest.raises(ValueError, match="H2"):
        fh.verify_set_reveal(public, tampered, "H2")
    renamed = json.loads(json.dumps(private))
    renamed["sets"]["H1"][0]["case_id"] = "fresh-h1-999"
    with pytest.raises(ValueError, match="case IDs"):
        fh.verify_set_reveal(public, renamed, "H1")


def test_a_changed_envelope_is_refused() -> None:
    inputs = _inputs()
    inputs["committed_envelope"] = inputs["committed_envelope"] + b" "
    with pytest.raises(ValueError, match="envelope"):
        fh.draw(
            fh._adapters()[CAMPAIGN],
            secrets=generate_campaign_secrets(),
            old_holdout=[],
            created_utc="2026-10-11T00:00:00Z",
            **inputs,
        )


def _scratch_repository(tmp_path: Path) -> tuple[Path, Path]:
    """A Git repo with the campaign's public records and a synthetic S9 seal."""

    root = tmp_path / "repo"
    adapter = fh._adapters()[CAMPAIGN]
    target = root / adapter.artifacts
    target.mkdir(parents=True)
    for name in ("public-case-manifest.json", "executable-envelope.json"):
        shutil.copy(fh.REPOSITORY / adapter.artifacts / name, target / name)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text(".piginot-private/\n", encoding="utf-8")
    old_secrets = generate_campaign_secrets()
    donor = fh.draw(
        adapter,
        secrets=old_secrets,
        old_holdout=[],
        created_utc="2026-10-11T00:00:00Z",
        **_inputs(),
    )
    spent = [dict(p.all_coordinates) for _, p in donor.sets["H1"]]
    campaign_hash = "a" * 64
    commitment = compute_holdout_commitment(
        campaign_hash=campaign_hash,
        campaign_secrets=old_secrets,
        holdout_coordinates=spent,
        created_utc="2026-01-01T00:00:00Z",
    )
    (target / "CP-ENVELOPE.json").write_text(
        json.dumps(
            {
                "campaign_hash": campaign_hash,
                "holdout_commitment": json.loads(canonical_json_bytes(commitment)),
            }
        ),
        encoding="utf-8",
    )
    reveal = tmp_path / "old" / CAMPAIGN / "reveal-v1.json"
    reveal.parent.mkdir(parents=True)
    reveal.write_text(
        json.dumps(
            {
                "campaign_hash": campaign_hash,
                "holdout_salt_hex": old_secrets.holdout_salt.hex(),
                "holdout_coordinates": spent,
            }
        ),
        encoding="utf-8",
    )
    return root, reveal


def test_the_ceremony_writes_private_first_and_is_one_shot(tmp_path: Path) -> None:
    root, reveal = _scratch_repository(tmp_path)
    backup = tmp_path / "backup" / CAMPAIGN / fh.PRIVATE_NAME
    result = fh.execute(
        root=root, campaign_id=CAMPAIGN, old_reveal=reveal, backup=backup
    )
    adapter = fh._adapters()[CAMPAIGN]
    primary = root / adapter.private_path
    public_path = root / adapter.artifacts / fh.PUBLIC_NAME
    assert primary.read_bytes() == backup.read_bytes()
    public = json.loads(public_path.read_text("utf-8"))
    private = json.loads(primary.read_text("utf-8"))
    for name in fh.SETS:
        fh.verify_set_reveal(public, private, name)
    assert public["existing_case_count"] == 216 + 64
    assert result.rejected_near_duplicates >= 0
    with pytest.raises(FileExistsError):
        fh.execute(root=root, campaign_id=CAMPAIGN, old_reveal=reveal, backup=backup)


def test_the_ceremony_refuses_bad_inputs(tmp_path: Path) -> None:
    root, reveal = _scratch_repository(tmp_path)
    good_backup = tmp_path / "backup" / CAMPAIGN / fh.PRIVATE_NAME
    with pytest.raises(ValueError, match="absolute"):
        fh.execute(
            root=root,
            campaign_id=CAMPAIGN,
            old_reveal=reveal,
            backup=Path("relative") / CAMPAIGN / fh.PRIVATE_NAME,
        )
    with pytest.raises(ValueError, match="outside the repository"):
        fh.execute(
            root=root,
            campaign_id=CAMPAIGN,
            old_reveal=reveal,
            backup=root / "inside" / CAMPAIGN / fh.PRIVATE_NAME,
        )
    with pytest.raises(ValueError, match="backup must be"):
        fh.execute(
            root=root,
            campaign_id=CAMPAIGN,
            old_reveal=reveal,
            backup=tmp_path / "backup" / "elsewhere" / fh.PRIVATE_NAME,
        )
    document = json.loads(reveal.read_text("utf-8"))
    wrong_salt = {
        **document,
        "holdout_salt_hex": generate_campaign_secrets().holdout_salt.hex(),
    }
    reveal.write_text(json.dumps(wrong_salt), encoding="utf-8")
    with pytest.raises(ValueError, match="old reveal"):
        fh.execute(
            root=root, campaign_id=CAMPAIGN, old_reveal=reveal, backup=good_backup
        )
    reveal.write_text(
        json.dumps({**document, "campaign_hash": "b" * 64}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="CP-ENVELOPE"):
        fh.execute(
            root=root, campaign_id=CAMPAIGN, old_reveal=reveal, backup=good_backup
        )
    (root / ".gitignore").write_text("", encoding="utf-8")
    reveal.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="ignored"):
        fh.execute(
            root=root, campaign_id=CAMPAIGN, old_reveal=reveal, backup=good_backup
        )
