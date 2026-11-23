"""RFC 8785 canonical per-inference Trust Receipt schema (``formamech.trust-receipt-v1``).

Maintains a strict, non-overlapping boundary between four epistemic classes so
weaker checks are never conflated into an ambiguous "verified" flag:

1. **Syntax & artifact integrity**: RFC 8785 JCS SHA-256 digests over model
   weights, mesh, CAD source, request, and solver configuration.
2. **Physical admissibility (Tier 0)**: zero-FEM checks (``det(F)``, Dirichlet
   compliance, PDE/traction residual norms, force balance).
3. **Statistical calibration (Tier 1a)**: finite-sample split-conformal marginal
   coverage over exchangeable in-envelope requests.
4. **Discrete FEM verification (Tier 1b / Tier 2)**: DWR adjoint QoI bounds and
   converged warm-start FEM solves under declared mesh/solver tolerances.

Per-inference :class:`VerificationTier` is strictly orthogonal to the bundle's
:class:`~piginot.verification.status.ClaimLevel` (which stops at
``HOLDOUT_VERIFIED`` per ``G18.claim-ceiling``). Neither ever asserts physical
experimental validation.

This module uses only the Python standard library and :mod:`piginot.contracts`
so ``formamech[runtime]`` and metadata-only installs remain isolated.
"""

from __future__ import annotations

import enum
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from piginot.contracts.canonical import canonical_json_bytes
from piginot.contracts.state_machine import ALLOWED_CLAIM_LEVELS

TRUST_RECEIPT_SCHEMA: Final = "formamech.trust-receipt-v1"


class VerificationTier(enum.StrEnum):
    """Non-overlapping per-inference verification tier (orthogonal to ClaimLevel)."""

    REFUSED = "REFUSED"
    SURROGATE_UNCHECKED = "SURROGATE_UNCHECKED"
    SURROGATE_ADMISSIBLE = "SURROGATE_ADMISSIBLE"
    CONFORMAL_BOUNDED = "CONFORMAL_BOUNDED"
    DWR_BOUNDED = "DWR_BOUNDED"
    DISCRETE_FEM_VERIFIED = "DISCRETE_FEM_VERIFIED"


_EPISTEMIC_SCOPE: Final[dict[VerificationTier, dict[str, bool]]] = {
    VerificationTier.REFUSED: {
        "syntax_integrity": True,
        "physics_admissibility": False,
        "marginal_statistical_coverage": False,
        "discrete_fem_agreement": False,
        "experimental_validation": False,
    },
    VerificationTier.SURROGATE_UNCHECKED: {
        "syntax_integrity": True,
        "physics_admissibility": False,
        "marginal_statistical_coverage": False,
        "discrete_fem_agreement": False,
        "experimental_validation": False,
    },
    VerificationTier.SURROGATE_ADMISSIBLE: {
        "syntax_integrity": True,
        "physics_admissibility": True,
        "marginal_statistical_coverage": False,
        "discrete_fem_agreement": False,
        "experimental_validation": False,
    },
    VerificationTier.CONFORMAL_BOUNDED: {
        "syntax_integrity": True,
        "physics_admissibility": True,
        "marginal_statistical_coverage": True,
        "discrete_fem_agreement": False,
        "experimental_validation": False,
    },
    VerificationTier.DWR_BOUNDED: {
        "syntax_integrity": True,
        "physics_admissibility": True,
        "marginal_statistical_coverage": False,
        "discrete_fem_agreement": True,
        "experimental_validation": False,
    },
    VerificationTier.DISCRETE_FEM_VERIFIED: {
        "syntax_integrity": True,
        "physics_admissibility": True,
        "marginal_statistical_coverage": False,
        "discrete_fem_agreement": True,
        "experimental_validation": False,
    },
}


def epistemic_scope(
    tier: VerificationTier,
    *,
    has_conformal_bound: bool = False,
) -> dict[str, bool]:
    """Return explicit boolean scope flags for ``tier`` (experimental always False)."""
    scope = dict(_EPISTEMIC_SCOPE[tier])
    if has_conformal_bound and tier in (
        VerificationTier.DWR_BOUNDED,
        VerificationTier.DISCRETE_FEM_VERIFIED,
    ):
        scope["marginal_statistical_coverage"] = True
    scope["experimental_validation"] = False
    return scope


@dataclass(frozen=True, slots=True)
class ArtifactDigests:
    """Content digests pinning the exact model, CAD, mesh, and solver configuration."""

    bundle_manifest_sha256: str | None = None
    weights_sha256: str | None = None
    cad_source_sha256: str | None = None
    mesh_id: str | None = None
    problem_hash: str | None = None
    request_hash: str | None = None
    solver_backend_id: str | None = None
    solver_tolerances: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.solver_tolerances is not None:
            if not isinstance(self.solver_tolerances, Mapping):
                raise TypeError("solver_tolerances must be a mapping or None")
            object.__setattr__(
                self,
                "solver_tolerances",
                dict(sorted(self.solver_tolerances.items())),
            )

    def document(self) -> dict[str, Any]:
        return {
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "weights_sha256": self.weights_sha256,
            "cad_source_sha256": self.cad_source_sha256,
            "mesh_id": self.mesh_id,
            "problem_hash": self.problem_hash,
            "request_hash": self.request_hash,
            "solver_backend_id": self.solver_backend_id,
            "solver_tolerances": (
                None if self.solver_tolerances is None else dict(self.solver_tolerances)
            ),
        }


@dataclass(frozen=True, slots=True)
class TrustReceipt:
    """Canonical RFC 8785 per-inference provenance and verification receipt."""

    mode: str
    serving_path: str
    claim_level: str
    verification_tier: VerificationTier
    artifact_digests: ArtifactDigests
    tier0_admissibility: Mapping[str, Any] | None = None
    tier1_conformal: Mapping[str, Any] | None = None
    tier1_dwr: Mapping[str, Any] | None = None
    tier2_fem: Mapping[str, Any] | None = None
    test_time_adaptation: Mapping[str, Any] | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or not self.mode.strip():
            raise ValueError("mode must be non-empty text")
        if not isinstance(self.serving_path, str) or not self.serving_path.strip():
            raise ValueError("serving_path must be non-empty text")
        raw_claim = (
            self.claim_level.value
            if isinstance(self.claim_level, enum.Enum)
            else self.claim_level
        )
        if type(raw_claim) is not str or raw_claim not in ALLOWED_CLAIM_LEVELS:
            raise ValueError(
                f"claim_level must be one of {ALLOWED_CLAIM_LEVELS}, got {raw_claim!r}"
            )
        object.__setattr__(self, "claim_level", raw_claim)
        tier = VerificationTier(self.verification_tier)
        object.__setattr__(self, "verification_tier", tier)
        if not isinstance(self.artifact_digests, ArtifactDigests):
            raise TypeError("artifact_digests must be an ArtifactDigests instance")
        _validate_tier_consistency(
            tier=tier,
            tier0=self.tier0_admissibility,
            tier1_conformal=self.tier1_conformal,
            tier1_dwr=self.tier1_dwr,
            tier2_fem=self.tier2_fem,
        )

    def unsigned_document(self) -> dict[str, Any]:
        has_conformal = bool(
            isinstance(self.tier1_conformal, Mapping)
            and self.tier1_conformal.get("refused") is False
        )
        return {
            "schema": TRUST_RECEIPT_SCHEMA,
            "mode": self.mode,
            "serving_path": self.serving_path,
            "claim_level": self.claim_level,
            "verification_tier": self.verification_tier.value,
            "epistemic_scope": epistemic_scope(
                self.verification_tier,
                has_conformal_bound=has_conformal,
            ),
            "artifact_digests": self.artifact_digests.document(),
            "evidence": {
                "tier0_admissibility": (
                    None
                    if self.tier0_admissibility is None
                    else dict(self.tier0_admissibility)
                ),
                "tier1_conformal": (
                    None if self.tier1_conformal is None else dict(self.tier1_conformal)
                ),
                "tier1_dwr": (None if self.tier1_dwr is None else dict(self.tier1_dwr)),
                "tier2_fem": (None if self.tier2_fem is None else dict(self.tier2_fem)),
                "test_time_adaptation": (
                    None
                    if self.test_time_adaptation is None
                    else dict(self.test_time_adaptation)
                ),
            },
            "reason": self.reason,
        }

    @property
    def canonical_bytes(self) -> bytes:
        """RFC 8785 JCS canonical UTF-8 payload of the receipt body."""
        return canonical_json_bytes(self.unsigned_document())

    @property
    def receipt_sha256(self) -> str:
        """SHA-256 hex digest of :attr:`canonical_bytes`."""
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def document(self) -> dict[str, Any]:
        """Full serialized receipt including its RFC 8785 SHA-256 digest."""
        return {
            **self.unsigned_document(),
            "receipt_sha256": self.receipt_sha256,
        }


def _validate_tier_consistency(
    *,
    tier: VerificationTier,
    tier0: Mapping[str, Any] | None,
    tier1_conformal: Mapping[str, Any] | None,
    tier1_dwr: Mapping[str, Any] | None,
    tier2_fem: Mapping[str, Any] | None,
) -> None:
    if tier is VerificationTier.SURROGATE_ADMISSIBLE:
        if not isinstance(tier0, Mapping) or tier0.get("passed") is not True:
            raise ValueError(
                "SURROGATE_ADMISSIBLE requires passing tier0_admissibility evidence"
            )
    elif tier is VerificationTier.CONFORMAL_BOUNDED:
        if not isinstance(tier0, Mapping) or tier0.get("passed") is not True:
            raise ValueError(
                "CONFORMAL_BOUNDED requires passing tier0_admissibility evidence"
            )
        if (
            not isinstance(tier1_conformal, Mapping)
            or tier1_conformal.get("refused") is not False
        ):
            raise ValueError(
                "CONFORMAL_BOUNDED requires accepted (refused=False) tier1_conformal"
            )
    elif tier is VerificationTier.DWR_BOUNDED:
        if not isinstance(tier1_dwr, Mapping) or tier1_dwr.get("verdict") != "OK":
            raise ValueError("DWR_BOUNDED requires tier1_dwr with verdict='OK'")
    elif tier is VerificationTier.DISCRETE_FEM_VERIFIED:
        if not isinstance(tier2_fem, Mapping) or tier2_fem.get("converged") is not True:
            raise ValueError(
                "DISCRETE_FEM_VERIFIED requires tier2_fem with converged=True"
            )


def verify_receipt_document(
    document: Mapping[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    """Verify RFC 8785 digest, schema, claim ceiling, and tier consistency."""
    errors: list[str] = []
    if not isinstance(document, Mapping):
        return False, ("receipt must be a JSON object",)
    if document.get("schema") != TRUST_RECEIPT_SCHEMA:
        errors.append(
            f"unexpected schema {document.get('schema')!r}; "
            f"expected {TRUST_RECEIPT_SCHEMA!r}"
        )
    claim = document.get("claim_level")
    if claim not in ALLOWED_CLAIM_LEVELS:
        errors.append(
            f"claim_level {claim!r} violates allowed ceiling {ALLOWED_CLAIM_LEVELS}"
        )
    raw_tier = document.get("verification_tier")
    tier: VerificationTier | None = None
    try:
        tier = VerificationTier(str(raw_tier))
    except ValueError:
        errors.append(f"unknown verification_tier {raw_tier!r}")
    scope = document.get("epistemic_scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("experimental_validation") is not False
    ):
        errors.append("epistemic_scope.experimental_validation must be strictly False")
    evidence = document.get("evidence")
    if tier is not None and isinstance(evidence, Mapping):
        try:
            _validate_tier_consistency(
                tier=tier,
                tier0=evidence.get("tier0_admissibility"),
                tier1_conformal=evidence.get("tier1_conformal"),
                tier1_dwr=evidence.get("tier1_dwr"),
                tier2_fem=evidence.get("tier2_fem"),
            )
        except ValueError as exc:
            errors.append(str(exc))
        if isinstance(scope, Mapping):
            tier1_conf = evidence.get("tier1_conformal")
            has_conformal = bool(
                isinstance(tier1_conf, Mapping) and tier1_conf.get("refused") is False
            )
            expected_scope = epistemic_scope(tier, has_conformal_bound=has_conformal)
            if dict(scope) != expected_scope:
                errors.append(
                    "epistemic_scope does not match declared verification_tier "
                    f"{tier.value!r}"
                )
    elif not isinstance(evidence, Mapping):
        errors.append("receipt must contain an evidence object")

    recorded_sha = document.get("receipt_sha256")
    if (
        not isinstance(recorded_sha, str)
        or len(recorded_sha) != 64
        or any(ch not in "0123456789abcdef" for ch in recorded_sha)
    ):
        errors.append("receipt_sha256 must be a 64-character hex digest")
    else:
        body = {k: v for k, v in document.items() if k != "receipt_sha256"}
        computed = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        if computed != recorded_sha:
            errors.append(
                f"receipt_sha256 mismatch: recorded {recorded_sha}, computed {computed}"
            )
    return not errors, tuple(errors)


__all__ = (
    "ALLOWED_CLAIM_LEVELS",
    "TRUST_RECEIPT_SCHEMA",
    "ArtifactDigests",
    "TrustReceipt",
    "VerificationTier",
    "epistemic_scope",
    "verify_receipt_document",
)
