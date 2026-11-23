"""Canonical qualification manifests, integrity checks, and version diffs.

Pure contract module (zero ML/FEM imports). Every :class:`QualificationManifest`
is bound by RFC 8785 canonical JSON SHA-256 to a single bundle manifest digest,
approved input envelope digest, and :class:`QualificationPolicy` digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from piginot.bundle.limitations import REQUIRED_LIMITATION_PARAGRAPH
from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.state_machine import CheckStatus, ClaimLevel

QUALIFICATION_MANIFEST_SCHEMA: Final = "formamech.qualification-manifest-v1"
QUALIFICATION_DIFF_SCHEMA: Final = "formamech.qualification-diff-v1"
EPISTEMIC_SCOPE_STATEMENT: Final = (
    "Numerical model qualification against the declared discrete FEM reference "
    "within the approved parameter envelope. Does not constitute experimental "
    "validation, physical specimen certification, or regulatory approval "
    "(ASME V&V 10)."
)


class QualificationVerdict(StrEnum):
    """Fail-closed outcome of running a qualification policy."""

    QUALIFIED = "QUALIFIED"
    NOT_QUALIFIED = "NOT_QUALIFIED"


class QualificationIntegrityError(ValueError):
    """Raised when a qualification manifest fails schema or SHA-256 verification."""


def claim_rank(level: ClaimLevel) -> int:
    """Monotonic integer rank for comparing :class:`ClaimLevel` values."""
    return level.rank


@dataclass(frozen=True, slots=True)
class QualificationCheckResult:
    """One auditable check inside a verification family."""

    check_id: str
    family_id: str
    status: CheckStatus
    detail: str
    measured: Mapping[str, Any] = ()  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, str) or not self.check_id.strip():
            raise ValueError("check_id must be non-empty text")
        if not isinstance(self.family_id, str) or not self.family_id.strip():
            raise ValueError("family_id must be non-empty text")
        st = (
            self.status
            if isinstance(self.status, CheckStatus)
            else CheckStatus(str(self.status))
        )
        object.__setattr__(self, "status", st)
        measured_dict = (
            dict(self.measured) if isinstance(self.measured, Mapping) else {}
        )
        object.__setattr__(self, "measured", measured_dict)

    def document(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "family_id": self.family_id,
            "status": self.status.value,
            "detail": self.detail,
            "measured": dict(self.measured),
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> QualificationCheckResult:
        return cls(
            check_id=str(doc["check_id"]),
            family_id=str(doc["family_id"]),
            status=CheckStatus(str(doc["status"])),
            detail=str(doc.get("detail", "")),
            measured=dict(doc.get("measured", {})),
        )


@dataclass(frozen=True, slots=True)
class QualificationManifest:
    """Content-addressed numerical qualification evidence for one FormaMech bundle."""

    created_utc: str
    bundle_name: str
    bundle_manifest_sha256: str
    approved_envelope_sha256: str
    problem_hash: str | None
    policy_id: str
    policy_sha256: str
    target_claim_level: ClaimLevel
    granted_claim_level: ClaimLevel
    verdict: QualificationVerdict
    solver_backend_id: str
    families: tuple[Mapping[str, Any], ...]
    checks: tuple[QualificationCheckResult, ...]
    dev_metrics: Mapping[str, Any]
    holdout_metrics: Mapping[str, Any] | None = None
    calibration_sha256: str | None = None
    calibration_path: str | None = None
    amortization: Mapping[str, Any] | None = None
    epistemic_scope: Mapping[str, Any] | None = None
    limitations: str = REQUIRED_LIMITATION_PARAGRAPH

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_name, str) or not self.bundle_name.strip():
            raise ValueError("bundle_name must be non-empty text")
        for label in (
            "bundle_manifest_sha256",
            "approved_envelope_sha256",
            "policy_sha256",
        ):
            val = getattr(self, label)
            if not isinstance(val, str) or len(val) != 64:
                raise ValueError(f"{label} must be a 64-character hex digest")
        target = (
            self.target_claim_level
            if isinstance(self.target_claim_level, ClaimLevel)
            else ClaimLevel(str(self.target_claim_level))
        )
        granted = (
            self.granted_claim_level
            if isinstance(self.granted_claim_level, ClaimLevel)
            else ClaimLevel(str(self.granted_claim_level))
        )
        verdict = (
            self.verdict
            if isinstance(self.verdict, QualificationVerdict)
            else QualificationVerdict(str(self.verdict))
        )
        if (
            verdict is QualificationVerdict.NOT_QUALIFIED
            and granted is not ClaimLevel.NOT_VERIFIED
        ):
            raise ValueError(
                "NOT_QUALIFIED manifest must have granted_claim_level == NOT_VERIFIED"
            )
        if (
            verdict is QualificationVerdict.QUALIFIED
            and granted is ClaimLevel.NOT_VERIFIED
        ):
            raise ValueError(
                "QUALIFIED manifest cannot have granted_claim_level == NOT_VERIFIED"
            )
        if claim_rank(granted) > claim_rank(target):
            raise ValueError("granted_claim_level cannot exceed target_claim_level")
        object.__setattr__(self, "target_claim_level", target)
        object.__setattr__(self, "granted_claim_level", granted)
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "families", tuple(dict(f) for f in self.families))
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "dev_metrics", dict(self.dev_metrics))
        object.__setattr__(
            self,
            "holdout_metrics",
            None if self.holdout_metrics is None else dict(self.holdout_metrics),
        )
        object.__setattr__(
            self,
            "amortization",
            None if self.amortization is None else dict(self.amortization),
        )
        scope = (
            dict(self.epistemic_scope)
            if self.epistemic_scope is not None
            else {
                "numerical_verification_only": True,
                "experimental_validation": False,
                "regulatory_certification": False,
                "statement": EPISTEMIC_SCOPE_STATEMENT,
            }
        )
        if scope.get("experimental_validation") is not False:
            raise ValueError("epistemic_scope.experimental_validation must be False")
        if scope.get("regulatory_certification") is not False:
            raise ValueError("epistemic_scope.regulatory_certification must be False")
        object.__setattr__(self, "epistemic_scope", scope)
        if self.limitations != REQUIRED_LIMITATION_PARAGRAPH:
            raise ValueError("manifest must carry REQUIRED_LIMITATION_PARAGRAPH")

    def document(self) -> dict[str, Any]:
        """Canonical manifest document without ``qualification_manifest_sha256``."""
        return {
            "schema": QUALIFICATION_MANIFEST_SCHEMA,
            "created_utc": self.created_utc,
            "bundle_name": self.bundle_name,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "approved_envelope_sha256": self.approved_envelope_sha256,
            "problem_hash": self.problem_hash,
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "target_claim_level": self.target_claim_level.value,
            "granted_claim_level": self.granted_claim_level.value,
            "verdict": self.verdict.value,
            "solver_backend_id": self.solver_backend_id,
            "calibration_sha256": self.calibration_sha256,
            "calibration_path": self.calibration_path,
            "families": [dict(f) for f in self.families],
            "checks": [c.document() for c in self.checks],
            "dev_metrics": dict(self.dev_metrics),
            "holdout_metrics": (
                None if self.holdout_metrics is None else dict(self.holdout_metrics)
            ),
            "amortization": (
                None if self.amortization is None else dict(self.amortization)
            ),
            "epistemic_scope": dict(self.epistemic_scope or {}),
            "limitations": self.limitations,
        }

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()

    def full_document(self) -> dict[str, Any]:
        return {
            **self.document(),
            "qualification_manifest_sha256": self.manifest_sha256,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.full_document(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> QualificationManifest:
        if doc.get("schema") != QUALIFICATION_MANIFEST_SCHEMA:
            raise QualificationIntegrityError(
                f"document is not a {QUALIFICATION_MANIFEST_SCHEMA} manifest"
            )
        try:
            manifest = cls(
                created_utc=str(doc["created_utc"]),
                bundle_name=str(doc["bundle_name"]),
                bundle_manifest_sha256=str(doc["bundle_manifest_sha256"]),
                approved_envelope_sha256=str(doc["approved_envelope_sha256"]),
                problem_hash=(
                    None
                    if doc.get("problem_hash") is None
                    else str(doc["problem_hash"])
                ),
                policy_id=str(doc["policy_id"]),
                policy_sha256=str(doc["policy_sha256"]),
                target_claim_level=ClaimLevel(str(doc["target_claim_level"])),
                granted_claim_level=ClaimLevel(str(doc["granted_claim_level"])),
                verdict=QualificationVerdict(str(doc["verdict"])),
                solver_backend_id=str(doc["solver_backend_id"]),
                families=tuple(doc.get("families", ())),
                checks=tuple(
                    QualificationCheckResult.from_document(c)
                    for c in doc.get("checks", ())
                ),
                dev_metrics=dict(doc.get("dev_metrics", {})),
                holdout_metrics=(
                    None
                    if doc.get("holdout_metrics") is None
                    else dict(doc["holdout_metrics"])
                ),
                calibration_sha256=(
                    None
                    if doc.get("calibration_sha256") is None
                    else str(doc["calibration_sha256"])
                ),
                calibration_path=(
                    None
                    if doc.get("calibration_path") is None
                    else str(doc["calibration_path"])
                ),
                amortization=(
                    None
                    if doc.get("amortization") is None
                    else dict(doc["amortization"])
                ),
                epistemic_scope=doc.get("epistemic_scope"),
                limitations=str(doc.get("limitations", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QualificationIntegrityError(f"malformed manifest: {exc}") from exc

        recorded_sha = doc.get("qualification_manifest_sha256")
        if recorded_sha is not None and recorded_sha != manifest.manifest_sha256:
            raise QualificationIntegrityError(
                "qualification_manifest_sha256 does not match canonical document"
            )
        return manifest

    @classmethod
    def load(cls, path: Path) -> QualificationManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise QualificationIntegrityError(
                "qualification manifest must be a JSON object"
            )
        return cls.from_document(raw)


def verify_manifest_document(
    doc: Mapping[str, Any],
    *,
    expected_bundle_sha256: str | None = None,
    expected_policy_sha256: str | None = None,
    expected_envelope_sha256: str | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Verify a qualification manifest document fail-closed; return ``(ok, errors)``."""
    errors: list[str] = []
    if not isinstance(doc, Mapping):
        return False, ("manifest must be a JSON object",)
    if "qualification_manifest_sha256" not in doc:
        errors.append("missing qualification_manifest_sha256 digest")
    try:
        manifest = QualificationManifest.from_document(doc)
    except QualificationIntegrityError as exc:
        return False, (*errors, str(exc))

    if (
        expected_bundle_sha256 is not None
        and manifest.bundle_manifest_sha256 != expected_bundle_sha256
    ):
        errors.append("manifest bundle_manifest_sha256 does not match target bundle")
    if (
        expected_policy_sha256 is not None
        and manifest.policy_sha256 != expected_policy_sha256
    ):
        errors.append("manifest policy_sha256 does not match required policy")
    if (
        expected_envelope_sha256 is not None
        and manifest.approved_envelope_sha256 != expected_envelope_sha256
    ):
        errors.append(
            "manifest approved_envelope_sha256 does not match target bundle envelope"
        )
    return len(errors) == 0, tuple(errors)


@dataclass(frozen=True, slots=True)
class QualificationDiff:
    """Version-to-version regression report between two qualification manifests."""

    old_manifest_sha256: str
    new_manifest_sha256: str
    old_verdict: str
    new_verdict: str
    old_claim_level: str
    new_claim_level: str
    metric_deltas: Mapping[str, float]
    regressions: tuple[str, ...]

    @property
    def regressed(self) -> bool:
        return len(self.regressions) > 0

    def document(self) -> dict[str, Any]:
        return {
            "schema": QUALIFICATION_DIFF_SCHEMA,
            "old_manifest_sha256": self.old_manifest_sha256,
            "new_manifest_sha256": self.new_manifest_sha256,
            "old_verdict": self.old_verdict,
            "new_verdict": self.new_verdict,
            "old_claim_level": self.old_claim_level,
            "new_claim_level": self.new_claim_level,
            "metric_deltas": dict(self.metric_deltas),
            "regressed": self.regressed,
            "regressions": list(self.regressions),
        }


def compare_qualifications(
    old: QualificationManifest,
    new: QualificationManifest,
    *,
    relative_regression_tol: float = 0.05,
) -> QualificationDiff:
    """Compare two qualification manifests and flag verdict, claim, or metric regressions."""
    regressions: list[str] = []
    if (
        old.verdict is QualificationVerdict.QUALIFIED
        and new.verdict is QualificationVerdict.NOT_QUALIFIED
    ):
        regressions.append(
            "qualification verdict regressed from QUALIFIED to NOT_QUALIFIED"
        )
    if claim_rank(new.granted_claim_level) < claim_rank(old.granted_claim_level):
        regressions.append(
            f"granted_claim_level regressed from {old.granted_claim_level.value} "
            f"to {new.granted_claim_level.value}"
        )

    deltas: dict[str, float] = {}
    for partition, old_m, new_m in (
        ("dev", old.dev_metrics, new.dev_metrics),
        ("holdout", old.holdout_metrics or {}, new.holdout_metrics or {}),
    ):
        for key in ("disp_rel_l2_quantile", "stress_rel_l2_quantile"):
            old_val = old_m.get(key)
            new_val = new_m.get(key)
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                delta = float(new_val) - float(old_val)
                deltas[f"{partition}.{key}"] = delta
                if old_val > 0.0 and delta > relative_regression_tol * abs(
                    float(old_val)
                ):
                    regressions.append(
                        f"{partition}.{key} regressed: {old_val:.4e} -> {new_val:.4e}"
                    )

    old_families = {str(f["family_id"]): str(f["status"]) for f in old.families}
    new_families = {str(f["family_id"]): str(f["status"]) for f in new.families}
    for fam_id, old_status in sorted(old_families.items()):
        new_status = new_families.get(fam_id, "MISSING")
        if old_status == "PASS" and new_status != "PASS":
            regressions.append(f"family {fam_id!r} regressed from PASS to {new_status}")

    return QualificationDiff(
        old_manifest_sha256=old.manifest_sha256,
        new_manifest_sha256=new.manifest_sha256,
        old_verdict=old.verdict.value,
        new_verdict=new.verdict.value,
        old_claim_level=old.granted_claim_level.value,
        new_claim_level=new.granted_claim_level.value,
        metric_deltas=deltas,
        regressions=tuple(regressions),
    )


__all__ = (
    "EPISTEMIC_SCOPE_STATEMENT",
    "QUALIFICATION_DIFF_SCHEMA",
    "QUALIFICATION_MANIFEST_SCHEMA",
    "QualificationCheckResult",
    "QualificationDiff",
    "QualificationIntegrityError",
    "QualificationManifest",
    "QualificationVerdict",
    "claim_rank",
    "compare_qualifications",
    "verify_manifest_document",
)
