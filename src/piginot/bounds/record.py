"""Calibration sidecars and the bounds they attach to a request (S23, ADR 0104).

A calibration is a canonical-JSON sidecar ``<bundles>/calibration/<bundle>.json``
bound to one bundle by its manifest hash, like the S22 FEM recipes, so the
shipped bundles stay byte-identical. It holds, per bound kind and quantity,
a split-conformal upper bound of the S14 budgeted error:

- ``budget_total``: surrogate + discretisation + projection, for ``fast``;
- ``reference_terms``: discretisation + projection, for ``certified`` and
  ``exact``, next to (never inside) the certificate.

and the Mahalanobis gate that decides whether a request is inside the
calibration envelope.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from piginot.bounds.gate import GateError, MahalanobisGate

SCHEMA = "piginot.calibration-v1"
BOUND_KINDS = ("budget_total", "reference_terms")
IN_CALIBRATION = "IN_CALIBRATION"
OUT_OF_CALIBRATION = "OUT_OF_CALIBRATION"


class CalibrationError(ValueError):
    """A malformed calibration, or one bound to another bundle."""


def case_ids_sha256(case_ids: Sequence[str]) -> str:
    """Digest of a list of case IDs, order kept."""

    payload = json.dumps(list(case_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _bounds(value: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    if set(value) != set(BOUND_KINDS):
        raise CalibrationError(f"bounds must hold exactly {list(BOUND_KINDS)}")
    result: dict[str, dict[str, float]] = {}
    for kind in BOUND_KINDS:
        quantities = value[kind]
        if not quantities:
            raise CalibrationError(f"{kind} bounds must not be empty")
        result[kind] = {}
        for quantity, bound in sorted(quantities.items()):
            number = float(bound)
            if not math.isfinite(number) or number < 0.0:
                raise CalibrationError(f"{kind}.{quantity} must be finite, >= 0")
            result[kind][quantity] = number
    return result


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    """One bundle's calibrated bounds and gate."""

    bundle: str
    bundle_manifest_sha256: str
    campaign_id: str
    protocol_sha256: str
    alpha: float
    n: int
    k: int
    calibration_case_ids_sha256: str
    bounds: Mapping[str, Mapping[str, float]]
    gate: MahalanobisGate

    def __post_init__(self) -> None:
        for label in ("bundle", "campaign_id"):
            if not getattr(self, label):
                raise CalibrationError(f"{label} must not be empty")
        for label in (
            "bundle_manifest_sha256",
            "protocol_sha256",
            "calibration_case_ids_sha256",
        ):
            digest = getattr(self, label)
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise CalibrationError(f"{label} must be a sha256 hex digest")
        if not 0.0 < float(self.alpha) < 1.0:
            raise CalibrationError("alpha must lie strictly between 0 and 1")
        if not 1 <= self.k <= self.n:
            raise CalibrationError("k must lie in [1, n]")
        object.__setattr__(self, "bounds", _bounds(self.bounds))

    def document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "bundle": self.bundle,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "campaign_id": self.campaign_id,
            "protocol_sha256": self.protocol_sha256,
            "alpha": self.alpha,
            "n": self.n,
            "k": self.k,
            "calibration_case_ids_sha256": self.calibration_case_ids_sha256,
            "bounds": {kind: dict(values) for kind, values in self.bounds.items()},
            "gate": self.gate.document(),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> CalibrationRecord:
        if document.get("schema") != SCHEMA:
            raise CalibrationError(f"not a {SCHEMA} document")
        try:
            return cls(
                bundle=document["bundle"],
                bundle_manifest_sha256=document["bundle_manifest_sha256"],
                campaign_id=document["campaign_id"],
                protocol_sha256=document["protocol_sha256"],
                alpha=float(document["alpha"]),
                n=int(document["n"]),
                k=int(document["k"]),
                calibration_case_ids_sha256=document["calibration_case_ids_sha256"],
                bounds=document["bounds"],
                gate=MahalanobisGate.from_document(document["gate"]),
            )
        except (KeyError, TypeError, GateError) as exc:
            raise CalibrationError(f"malformed calibration: {exc}") from exc

    @classmethod
    def load(
        cls, path: Path | str, *, bundle: str, manifest_sha256: str
    ) -> CalibrationRecord:
        """Load a sidecar and check that it belongs to the given bundle."""

        try:
            document = json.loads(Path(path).read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationError(f"cannot read calibration {path}: {exc}") from exc
        record = cls.from_document(document)
        if record.bundle != bundle or record.bundle_manifest_sha256 != manifest_sha256:
            raise CalibrationError(
                f"calibration {path} belongs to another bundle or bundle version"
            )
        return record

    def evaluate(self, inputs: Mapping[str, Any], kind: str) -> dict[str, Any]:
        """The bounds document for one request (inputs already accepted)."""

        if kind not in BOUND_KINDS:
            raise CalibrationError(f"kind must be one of {list(BOUND_KINDS)}")
        distance = self.gate.distance(inputs)
        inside = distance <= self.gate.max_distance
        return {
            "status": IN_CALIBRATION if inside else OUT_OF_CALIBRATION,
            "kind": kind,
            "alpha": self.alpha,
            "coverage": (
                f"marginal: at least {1.0 - self.alpha:.0%} of requests drawn "
                "like the calibration cases lie within the bounds"
            ),
            "distance_to_train": distance,
            "max_calibration_distance": self.gate.max_distance,
            "upper_bounds": dict(self.bounds[kind]) if inside else None,
            "calibration_sha256": self.sha256(),
        }

    def sha256(self) -> str:
        from piginot.contracts import canonical_json_bytes

        return hashlib.sha256(canonical_json_bytes(self.document())).hexdigest()


def calibration_path(bundle_path: Path | str) -> Path:
    """Where a bundle's calibration sidecar lives."""

    directory = Path(bundle_path)
    return directory.parent / "calibration" / f"{directory.name}.json"


__all__ = (
    "BOUND_KINDS",
    "IN_CALIBRATION",
    "OUT_OF_CALIBRATION",
    "SCHEMA",
    "CalibrationError",
    "CalibrationRecord",
    "calibration_path",
    "case_ids_sha256",
)
