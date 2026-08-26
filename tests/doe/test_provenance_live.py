from __future__ import annotations

import pytest

from piginot.doe.provenance import (
    ArtifactRecord,
    CommitmentOrderError,
    LiveArtifactEvidence,
    LiveCommitmentOrderEvidence,
)
from piginot.doe.roles import EvidenceRole


def _live_evidence(
    artifacts: tuple[ArtifactRecord, ...] = (),
    *,
    commitment_mtime_ns: int = 1,
    artifact_mtime_ns: int = 2,
) -> LiveCommitmentOrderEvidence:
    active = tuple(sorted(artifacts, key=lambda artifact: artifact.path))
    return LiveCommitmentOrderEvidence(
        commitment_path="holdout.json",
        commitment_mtime_ns=commitment_mtime_ns,
        artifacts=tuple(
            LiveArtifactEvidence(
                path=artifact.path,
                role=artifact.role,
                mtime_ns=artifact_mtime_ns + index,
            )
            for index, artifact in enumerate(active)
        ),
    )


def test_live_evidence_round_trip_is_canonical_and_supports_nanosecond_values() -> None:
    first = ArtifactRecord("train/z.pt", EvidenceRole.TRAIN)
    second = ArtifactRecord("train/a.pt", EvidenceRole.DEVELOPMENT)
    live = _live_evidence(
        (first, second),
        commitment_mtime_ns=1_700_000_000_000_000_000,
        artifact_mtime_ns=1_700_000_000_000_000_001,
    )

    document = live.to_canonical_bytes()
    restored = LiveCommitmentOrderEvidence.from_canonical_bytes(document)

    assert restored == live
    assert restored.to_canonical_bytes() == document
    assert b'"mtime_ns":"1700000000000000001"' in document
    assert tuple(artifact.path for artifact in restored.artifacts) == (
        "train/a.pt",
        "train/z.pt",
    )


def test_tampered_live_mtime_and_noncanonical_document_fail_closed() -> None:
    artifacts = (ArtifactRecord("train/model.pt", EvidenceRole.TRAIN),)
    document = _live_evidence(artifacts).to_canonical_bytes()
    tampered = document.replace(
        b'"commitment_mtime_ns":"1"',
        b'"commitment_mtime_ns":"3"',
    )
    restored = LiveCommitmentOrderEvidence.from_canonical_bytes(tampered)

    with pytest.raises(CommitmentOrderError, match="mtime"):
        restored.verify()
    with pytest.raises(ValueError, match="canonical JSON"):
        LiveCommitmentOrderEvidence.from_canonical_bytes(b" " + document)


@pytest.mark.parametrize(
    "document, message",
    [
        (b"[]", "JSON object"),
        (
            b'{"artifacts":[],"commitment_mtime_ns":"1",'
            b'"commitment_path":"holdout.json","schema":"wrong"}',
            "schema",
        ),
        (
            b'{"artifacts":[],"commitment_mtime_ns":1,'
            b'"commitment_path":"holdout.json",'
            b'"schema":"piginot.s9.live-commitment-order-v1"}',
            "decimal string",
        ),
    ],
)
def test_persisted_record_parser_rejects_schema_and_type_tampering(
    document: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LiveCommitmentOrderEvidence.from_canonical_bytes(document)


@pytest.mark.parametrize("mtime_ns", [-1, False, 1.5, "1"])
def test_live_evidence_rejects_invalid_mtimes(mtime_ns: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        LiveCommitmentOrderEvidence(
            commitment_path="holdout.json",
            commitment_mtime_ns=mtime_ns,  # type: ignore[arg-type]
            artifacts=(),
        )


@pytest.mark.parametrize("role", [EvidenceRole.LEGACY, "UNKNOWN"])
def test_live_artifact_rejects_legacy_and_unknown_roles(role: object) -> None:
    with pytest.raises(ValueError, match="LEGACY|declared evidence Role"):
        LiveArtifactEvidence(
            path="train/model.pt",
            role=role,  # type: ignore[arg-type]
            mtime_ns=2,
        )


def test_live_record_rejects_invalid_duplicate_and_commitment_artifacts() -> None:
    artifact = LiveArtifactEvidence(
        path="train/model.pt",
        role=EvidenceRole.TRAIN,
        mtime_ns=2,
    )
    with pytest.raises(ValueError, match="LiveArtifactEvidence"):
        LiveCommitmentOrderEvidence(
            commitment_path="holdout.json",
            commitment_mtime_ns=1,
            artifacts=(object(),),  # type: ignore[arg-type]
        )
    with pytest.raises(CommitmentOrderError, match="unique"):
        LiveCommitmentOrderEvidence(
            commitment_path="holdout.json",
            commitment_mtime_ns=1,
            artifacts=(artifact, artifact),
        )
    with pytest.raises(CommitmentOrderError, match="also be an artifact"):
        LiveCommitmentOrderEvidence(
            commitment_path="holdout.json",
            commitment_mtime_ns=1,
            artifacts=(
                LiveArtifactEvidence(
                    path="holdout.json",
                    role=EvidenceRole.TRAIN,
                    mtime_ns=2,
                ),
            ),
        )


def test_persisted_parser_accepts_text_and_rejects_non_text() -> None:
    document = _live_evidence().to_canonical_bytes().decode("utf-8")

    assert LiveCommitmentOrderEvidence.from_canonical_bytes(document) == (
        _live_evidence()
    )
    with pytest.raises(ValueError, match="UTF-8"):
        LiveCommitmentOrderEvidence.from_canonical_bytes(7)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "document, message",
    [
        (
            b'{"artifacts":{},"commitment_mtime_ns":"1",'
            b'"commitment_path":"holdout.json",'
            b'"schema":"piginot.s9.live-commitment-order-v1"}',
            "field types",
        ),
        (
            b'{"artifacts":[{}],"commitment_mtime_ns":"1",'
            b'"commitment_path":"holdout.json",'
            b'"schema":"piginot.s9.live-commitment-order-v1"}',
            "invalid fields",
        ),
        (
            b'{"artifacts":[{"mtime_ns":"2","path":7,"role":"TRAIN"}],'
            b'"commitment_mtime_ns":"1","commitment_path":"holdout.json",'
            b'"schema":"piginot.s9.live-commitment-order-v1"}',
            "field types",
        ),
    ],
)
def test_persisted_parser_rejects_malformed_artifact_records(
    document: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LiveCommitmentOrderEvidence.from_canonical_bytes(document)
