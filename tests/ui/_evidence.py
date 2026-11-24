"""Build content-addressed qualification evidence for workbench UI tests."""

from pathlib import Path

from piginot.bundle import load_bundle
from piginot.bundle.inference import ENVELOPE_PATH
from piginot.qualify import (
    QualificationManifest,
    QualificationVerdict,
    default_development_policy,
)
from piginot.verification.status import ClaimLevel


def save_qualification(
    bundle_path: Path, path: Path, *, qualified: bool = True
) -> None:
    bundle = load_bundle(bundle_path)
    policy = default_development_policy()
    manifest = QualificationManifest(
        created_utc="2026-01-01T00:00:00Z",
        bundle_name=bundle.manifest.name,
        bundle_manifest_sha256=bundle.manifest.manifest_sha256,
        approved_envelope_sha256=bundle.manifest.files[ENVELOPE_PATH].sha256,
        problem_hash=None,
        policy_id=policy.policy_id,
        policy_sha256=policy.policy_sha256,
        target_claim_level=ClaimLevel.DEVELOPMENT_VERIFIED,
        granted_claim_level=(
            ClaimLevel.DEVELOPMENT_VERIFIED if qualified else ClaimLevel.NOT_VERIFIED
        ),
        verdict=(
            QualificationVerdict.QUALIFIED
            if qualified
            else QualificationVerdict.NOT_QUALIFIED
        ),
        solver_backend_id="test",
        families=(),
        checks=(),
        dev_metrics={},
    )
    manifest.save(path)
