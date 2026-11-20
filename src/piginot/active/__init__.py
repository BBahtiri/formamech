"""Certificate-driven active FEM acquisition and multi-fidelity learning (Feature C)."""

from __future__ import annotations

from typing import Any

from piginot.active.acquisition import (
    ACQUISITION_ARMS,
    ACQUISITION_PLAN_SCHEMA,
    FEM_COST_LEDGER_SCHEMA,
    SELECTION_EVIDENCE_SCHEMA,
    TRAINING_MODES,
    AcquisitionCandidate,
    AcquisitionPlan,
    AcquisitionWeights,
    FEMCostLedger,
    FEMCostLedgerEntry,
    SelectionEvidence,
    SelectionScoreBreakdown,
    select_acquisition_batch,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "MULTIFIDELITY_DISCLOSURE": (
        "piginot.active.multifidelity",
        "MULTIFIDELITY_DISCLOSURE",
    ),
    "MultiFidelityConfig": ("piginot.active.multifidelity", "MultiFidelityConfig"),
    "MultiFidelityTerms": ("piginot.active.multifidelity", "MultiFidelityTerms"),
    "multifidelity_energy_loss": (
        "piginot.active.multifidelity",
        "multifidelity_energy_loss",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(mod_name), attr_name)
    globals()[name] = value
    return value


__all__ = (
    "ACQUISITION_ARMS",
    "ACQUISITION_PLAN_SCHEMA",
    "FEM_COST_LEDGER_SCHEMA",
    "MULTIFIDELITY_DISCLOSURE",
    "SELECTION_EVIDENCE_SCHEMA",
    "TRAINING_MODES",
    "AcquisitionCandidate",
    "AcquisitionPlan",
    "AcquisitionWeights",
    "FEMCostLedger",
    "FEMCostLedgerEntry",
    "MultiFidelityConfig",
    "MultiFidelityTerms",
    "SelectionEvidence",
    "SelectionScoreBreakdown",
    "multifidelity_energy_loss",
    "select_acquisition_batch",
)
