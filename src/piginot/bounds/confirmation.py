"""H1 confirmation of calibrated bounds (S23 follow-up, ADR 0104).

A bound is *confirmed* when its single H1 coverage test passed (covered
in-calibration cases at least the preregistered acceptance count). The
confirmation is a sidecar ``<bundles>/h1-results/<bundle>.json`` bound to the
bundle's manifest and to the calibration it tested, built from the committed
H1 reveal; the calibration sidecar itself never changes. Serving withholds an
unconfirmed bound: the quantity is listed under ``not_confirmed`` with no
number.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from piginot.bounds.record import BOUND_KINDS, CalibrationError, CalibrationRecord

SCHEMA = "piginot.calibration-h1-v1"


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise CalibrationError(f"{label} must be a sha256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class H1Confirmation:
    """Which calibrated bounds passed their H1 coverage test."""

    bundle: str
    bundle_manifest_sha256: str
    calibration_sha256: str
    reveal_sha256: str
    confirmed: Mapping[str, Mapping[str, bool]]

    def __post_init__(self) -> None:
        if not self.bundle:
            raise CalibrationError("bundle must not be empty")
        for label in ("bundle_manifest_sha256", "calibration_sha256", "reveal_sha256"):
            _digest(getattr(self, label), label)
        if set(self.confirmed) != set(BOUND_KINDS):
            raise CalibrationError(f"confirmed must hold exactly {list(BOUND_KINDS)}")
        frozen: dict[str, dict[str, bool]] = {}
        for kind in BOUND_KINDS:
            values = self.confirmed[kind]
            if not values or any(type(v) is not bool for v in values.values()):
                raise CalibrationError(f"{kind} confirmations must be booleans")
            frozen[kind] = dict(sorted(values.items()))
        object.__setattr__(self, "confirmed", frozen)

    def unconfirmed(self, kind: str) -> tuple[str, ...]:
        return tuple(q for q, ok in self.confirmed[kind].items() if not ok)

    def document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "bundle": self.bundle,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "calibration_sha256": self.calibration_sha256,
            "reveal_sha256": self.reveal_sha256,
            "confirmed": {k: dict(v) for k, v in self.confirmed.items()},
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> H1Confirmation:
        if document.get("schema") != SCHEMA:
            raise CalibrationError(f"not a {SCHEMA} document")
        try:
            return cls(
                bundle=document["bundle"],
                bundle_manifest_sha256=document["bundle_manifest_sha256"],
                calibration_sha256=document["calibration_sha256"],
                reveal_sha256=document["reveal_sha256"],
                confirmed=document["confirmed"],
            )
        except (KeyError, TypeError) as exc:
            raise CalibrationError(f"malformed H1 confirmation: {exc}") from exc

    @classmethod
    def load(cls, path: Path | str, calibration: CalibrationRecord) -> H1Confirmation:
        """Load a confirmation and check that it tested ``calibration``."""

        try:
            document = json.loads(Path(path).read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CalibrationError(
                f"cannot read H1 confirmation {path}: {exc}"
            ) from exc
        confirmation = cls.from_document(document)
        if (
            confirmation.bundle != calibration.bundle
            or confirmation.bundle_manifest_sha256 != calibration.bundle_manifest_sha256
            or confirmation.calibration_sha256 != calibration.sha256()
        ):
            raise CalibrationError(
                f"H1 confirmation {path} belongs to another calibration"
            )
        return confirmation


@dataclass(frozen=True, slots=True)
class ConfirmedCalibration:
    """A calibration whose unconfirmed bounds are withheld when served."""

    calibration: CalibrationRecord
    confirmation: H1Confirmation

    @property
    def bundle_manifest_sha256(self) -> str:
        return self.calibration.bundle_manifest_sha256

    def evaluate(self, inputs: Mapping[str, Any], kind: str) -> dict[str, Any]:
        report = self.calibration.evaluate(inputs, kind)
        withheld = self.confirmation.unconfirmed(kind)
        upper = report["upper_bounds"]
        if upper is not None:
            report["upper_bounds"] = {
                q: v for q, v in upper.items() if q not in withheld
            }
        report["not_confirmed"] = list(withheld)
        report["h1_confirmation"] = "only bounds that passed their H1 coverage test"
        return report


def confirmation_path(bundle_path: Path | str) -> Path:
    """Where a bundle's H1 confirmation sidecar lives."""

    directory = Path(bundle_path)
    return directory.parent / "h1-results" / f"{directory.name}.json"


__all__ = (
    "ConfirmedCalibration",
    "H1Confirmation",
    "confirmation_path",
)
