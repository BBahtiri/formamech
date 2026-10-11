"""The per-prediction physics certificate (S21, ADR 0102)."""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


class CertificateVerdict(enum.StrEnum):
    """Whether the error estimate may be used."""

    OK = "OK"
    INDEFINITE = "INDEFINITE"
    INADMISSIBLE = "INADMISSIBLE"


@dataclass(frozen=True, slots=True)
class Certificate:
    """One prediction's distance to the converged discrete solution.

    ``eta_energy`` is ``sqrt(d^T K d)`` for the Newton increment ``d`` at the
    surrogate state; ``energy_drop`` the potential-energy decrease along it
    (about ``eta_energy**2 / 2`` near the minimizer). ``indicators`` holds
    relative field and QoI changes along the increment. Wall times are kept
    out, so the certificate's bytes are reproducible.
    """

    verdict: CertificateVerdict
    eta_energy: float | None
    energy_drop: float | None
    potential_energy: float | None
    indicators: Mapping[str, float] = field(default_factory=dict)
    guards: Mapping[str, bool | None] = field(default_factory=dict)
    dirichlet_deviation: float = 0.0
    dof_count: int = 0
    reason: str = ""

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "verdict": self.verdict.value,
            "eta_energy": self.eta_energy,
            "energy_drop": self.energy_drop,
            "potential_energy": self.potential_energy,
            "indicators": dict(sorted(self.indicators.items())),
            "guards": dict(sorted(self.guards.items())),
            "dirichlet_deviation": self.dirichlet_deviation,
            "dof_count": self.dof_count,
            "reason": self.reason,
        }
