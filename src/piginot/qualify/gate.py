"""Fail-closed deployment gate for qualified FormaMech bundles.

Verifies bundle integrity, qualification manifest integrity, exact SHA-256
bindings (bundle manifest, approved envelope, optional policy), passing
``QUALIFIED`` verdict, and minimum required :class:`ClaimLevel`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from piginot.contracts.state_machine import (
    ClaimLevel,
    StateMachineSpec,
    TransitionSpec,
    register_state_machine,
)
from piginot.qualify.manifest import (
    QualificationManifest,
    QualificationVerdict,
    claim_rank,
    verify_manifest_document,
)
from piginot.qualify.policy import QualificationPolicy

DEPLOYMENT_GATE_SCHEMA: Final = "formamech.deployment-gate-v1"


class DeploymentGateState(StrEnum):
    """Explicit outcome states of the fail-closed deployment gate."""

    UNCHECKED = "UNCHECKED"
    ALLOWED = "ALLOWED"
    REFUSED = "REFUSED"


DEPLOYMENT_GATE_MACHINE: Final[StateMachineSpec[DeploymentGateState]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.qualify.deployment_gate",
            description=(
                "Fail-closed deployment eligibility gate verifying bundle integrity, "
                "qualification manifest bindings, QUALIFIED verdict, and min ClaimLevel."
            ),
            states=tuple(DeploymentGateState),
            initial_states=(DeploymentGateState.UNCHECKED,),
            terminal_states=(DeploymentGateState.ALLOWED, DeploymentGateState.REFUSED),
            transitions=(
                TransitionSpec(
                    DeploymentGateState.UNCHECKED,
                    DeploymentGateState.ALLOWED,
                    "pass_gate",
                    "All integrity, binding, verdict, and claim-level checks passed.",
                ),
                TransitionSpec(
                    DeploymentGateState.UNCHECKED,
                    DeploymentGateState.REFUSED,
                    "refuse_gate",
                    "One or more fail-closed GateReasonCode checks failed.",
                ),
            ),
        )
    )
)


class GateReasonCode(StrEnum):
    """Stable refusal codes returned by :func:`check_deployment_gate`."""

    BUNDLE_INVALID = "GATE_BUNDLE_INVALID"
    MANIFEST_MISSING = "GATE_MANIFEST_MISSING"
    MANIFEST_INVALID = "GATE_MANIFEST_INVALID"
    BUNDLE_MISMATCH = "GATE_BUNDLE_MISMATCH"
    ENVELOPE_MISMATCH = "GATE_ENVELOPE_MISMATCH"
    POLICY_MISMATCH = "GATE_POLICY_MISMATCH"
    NOT_QUALIFIED = "GATE_NOT_QUALIFIED"
    CLAIM_INSUFFICIENT = "GATE_CLAIM_INSUFFICIENT"


class DeploymentGateError(ValueError):
    """Raised when a bundle fails deployment qualification checks."""

    def __init__(self, reason_codes: tuple[GateReasonCode, ...], detail: str) -> None:
        self.reason_codes = tuple(sorted(set(reason_codes), key=lambda c: c.value))
        super().__init__(f"{', '.join(c.value for c in self.reason_codes)}: {detail}")


@dataclass(frozen=True, slots=True)
class DeploymentGateResult:
    """Auditable outcome of a deployment gate check."""

    allowed: bool
    bundle_name: str
    bundle_manifest_sha256: str | None
    qualification_manifest_sha256: str | None
    granted_claim_level: ClaimLevel
    min_claim_level: ClaimLevel
    reason_codes: tuple[GateReasonCode, ...]
    detail: str

    def __post_init__(self) -> None:
        if self.allowed and self.reason_codes:
            raise ValueError("allowed=True forbids non-empty reason_codes")
        if not self.allowed and not self.reason_codes:
            raise ValueError("allowed=False requires at least one GateReasonCode")

    @property
    def state(self) -> DeploymentGateState:
        return (
            DeploymentGateState.ALLOWED if self.allowed else DeploymentGateState.REFUSED
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema": DEPLOYMENT_GATE_SCHEMA,
            "allowed": self.allowed,
            "bundle_name": self.bundle_name,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "qualification_manifest_sha256": self.qualification_manifest_sha256,
            "granted_claim_level": self.granted_claim_level.value,
            "min_claim_level": self.min_claim_level.value,
            "reason_codes": [c.value for c in self.reason_codes],
            "detail": self.detail,
        }


def default_manifest_path(bundle_path: Path | str) -> Path:
    """Default sidecar location: ``<bundle_parent>/qualification/<bundle>.json``."""
    directory = Path(bundle_path)
    return directory.parent / "qualification" / f"{directory.name}.json"


def check_deployment_gate(
    bundle_path: Path | str,
    manifest_path: Path | str | None = None,
    *,
    policy: QualificationPolicy | None = None,
    min_claim_level: ClaimLevel = ClaimLevel.DEVELOPMENT_VERIFIED,
) -> DeploymentGateResult:
    """Evaluate deployment eligibility fail-closed; never raises on policy refusal."""
    from piginot.bundle import BundleIntegrityError, load_bundle
    from piginot.bundle.inference import ENVELOPE_PATH

    bpath = Path(bundle_path)
    min_level = (
        min_claim_level
        if isinstance(min_claim_level, ClaimLevel)
        else ClaimLevel(str(min_claim_level))
    )
    try:
        bundle = load_bundle(bpath)
    except (BundleIntegrityError, OSError, ValueError) as exc:
        return DeploymentGateResult(
            allowed=False,
            bundle_name=bpath.name,
            bundle_manifest_sha256=None,
            qualification_manifest_sha256=None,
            granted_claim_level=ClaimLevel.NOT_VERIFIED,
            min_claim_level=min_level,
            reason_codes=(GateReasonCode.BUNDLE_INVALID,),
            detail=f"bundle failed integrity check: {exc}",
        )

    bundle_sha = bundle.manifest.manifest_sha256
    envelope_sha = bundle.manifest.files[ENVELOPE_PATH].sha256
    mpath = (
        Path(manifest_path)
        if manifest_path is not None
        else default_manifest_path(bpath)
    )
    if not mpath.is_file():
        return DeploymentGateResult(
            allowed=False,
            bundle_name=bundle.manifest.name,
            bundle_manifest_sha256=bundle_sha,
            qualification_manifest_sha256=None,
            granted_claim_level=ClaimLevel.NOT_VERIFIED,
            min_claim_level=min_level,
            reason_codes=(GateReasonCode.MANIFEST_MISSING,),
            detail=f"qualification manifest not found: {mpath}",
        )

    try:
        raw_doc = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return DeploymentGateResult(
            allowed=False,
            bundle_name=bundle.manifest.name,
            bundle_manifest_sha256=bundle_sha,
            qualification_manifest_sha256=None,
            granted_claim_level=ClaimLevel.NOT_VERIFIED,
            min_claim_level=min_level,
            reason_codes=(GateReasonCode.MANIFEST_INVALID,),
            detail=f"cannot parse qualification manifest: {exc}",
        )

    ok, errors = verify_manifest_document(raw_doc)
    if not ok:
        return DeploymentGateResult(
            allowed=False,
            bundle_name=bundle.manifest.name,
            bundle_manifest_sha256=bundle_sha,
            qualification_manifest_sha256=None,
            granted_claim_level=ClaimLevel.NOT_VERIFIED,
            min_claim_level=min_level,
            reason_codes=(GateReasonCode.MANIFEST_INVALID,),
            detail="; ".join(errors),
        )

    manifest = QualificationManifest.from_document(raw_doc)
    reasons: list[GateReasonCode] = []
    messages: list[str] = []

    if manifest.bundle_manifest_sha256 != bundle_sha:
        reasons.append(GateReasonCode.BUNDLE_MISMATCH)
        messages.append("manifest bundle_manifest_sha256 does not match bundle")
    if manifest.approved_envelope_sha256 != envelope_sha:
        reasons.append(GateReasonCode.ENVELOPE_MISMATCH)
        messages.append(
            "manifest approved_envelope_sha256 does not match bundle envelope"
        )
    if policy is not None and manifest.policy_sha256 != policy.policy_sha256:
        reasons.append(GateReasonCode.POLICY_MISMATCH)
        messages.append("manifest policy_sha256 does not match required policy")
    if manifest.verdict is not QualificationVerdict.QUALIFIED:
        reasons.append(GateReasonCode.NOT_QUALIFIED)
        messages.append("qualification verdict is NOT_QUALIFIED")
    if claim_rank(manifest.granted_claim_level) < claim_rank(min_level):
        reasons.append(GateReasonCode.CLAIM_INSUFFICIENT)
        messages.append(
            f"granted_claim_level={manifest.granted_claim_level.value} "
            f"is below required {min_level.value}"
        )

    if reasons:
        return DeploymentGateResult(
            allowed=False,
            bundle_name=bundle.manifest.name,
            bundle_manifest_sha256=bundle_sha,
            qualification_manifest_sha256=manifest.manifest_sha256,
            granted_claim_level=manifest.granted_claim_level,
            min_claim_level=min_level,
            reason_codes=tuple(reasons),
            detail="; ".join(messages),
        )

    return DeploymentGateResult(
        allowed=True,
        bundle_name=bundle.manifest.name,
        bundle_manifest_sha256=bundle_sha,
        qualification_manifest_sha256=manifest.manifest_sha256,
        granted_claim_level=manifest.granted_claim_level,
        min_claim_level=min_level,
        reason_codes=(),
        detail=(
            f"bundle {bundle.manifest.name!r} qualified at "
            f"{manifest.granted_claim_level.value} under policy {manifest.policy_id!r}"
        ),
    )


def enforce_deployment_gate(
    bundle_path: Path | str,
    manifest_path: Path | str | None = None,
    *,
    policy: QualificationPolicy | None = None,
    min_claim_level: ClaimLevel = ClaimLevel.DEVELOPMENT_VERIFIED,
) -> DeploymentGateResult:
    """Enforce deployment qualification; raise :class:`DeploymentGateError` if refused."""
    result = check_deployment_gate(
        bundle_path,
        manifest_path,
        policy=policy,
        min_claim_level=min_claim_level,
    )
    if not result.allowed:
        raise DeploymentGateError(result.reason_codes, result.detail)
    return result


__all__ = (
    "DEPLOYMENT_GATE_SCHEMA",
    "DeploymentGateError",
    "DeploymentGateResult",
    "GateReasonCode",
    "check_deployment_gate",
    "default_manifest_path",
    "enforce_deployment_gate",
)
