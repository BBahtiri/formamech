"""Certificate-driven active FEM acquisition and cost ledger (Feature C / P3).

Ranks candidate queries strictly inside a declared ``TRAIN``-only acquisition
pool using scale-normalized physics indicators, OOD distance, ensemble spread,
and greedy farthest-point diversity:

    a(q) = w_1 * tilde_eta_E(q) + w_2 * tilde_d_OOD(q)
         + w_3 * tilde_sigma_ens(q) + w_4 * tilde_d_div(q)

Hard scientific guardrails:
* Any candidate outside ``partition == "TRAIN"`` is refused immediately (DEV
  and HOLDOUT cases can never be read by acquisition or selection code).
* Deterministic tie-breaking by canonical ``case_id`` ensures exact replay
  with identical ``selection_hash``.
* Every attempted, succeeded, and failed FEM solve is recorded in an immutable
  ``FEMCostLedger`` with full cost accounting.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from piginot.contracts import canonical_json_bytes

ACQUISITION_PLAN_SCHEMA: Final = "piginot.active-fem-plan-v1"
FEM_COST_LEDGER_SCHEMA: Final = "piginot.fem-cost-ledger-v1"
SELECTION_EVIDENCE_SCHEMA: Final = "piginot.active-selection-evidence-v1"
ACQUISITION_ARMS: Final = ("random_doe", "physics_indicator", "physics_diversity")
TRAINING_MODES: Final = ("label_free", "multi_fidelity_label_efficient")


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _finite(
    value: object, label: str, *, nonnegative: bool = False, positive: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{label} must be finite")
    if positive and num <= 0.0:
        raise ValueError(f"{label} must be strictly positive")
    if nonnegative and num < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return num


@dataclass(frozen=True, slots=True)
class AcquisitionCandidate:
    """One TRAIN-only candidate in the active-FEM acquisition pool."""

    case_id: str
    partition: str
    feature_vector: tuple[float, ...]
    eta_energy: float
    ood_distance: float = 0.0
    ensemble_disagreement: float = 0.0
    parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be non-empty text")
        if self.partition != "TRAIN":
            raise ValueError(
                f"acquisition pool strictly forbids non-TRAIN partition {self.partition!r}"
            )
        if not self.feature_vector:
            raise ValueError("feature_vector must be non-empty")
        feats = tuple(
            _finite(v, f"feature_vector[{i}]")
            for i, v in enumerate(self.feature_vector)
        )
        object.__setattr__(self, "feature_vector", feats)
        object.__setattr__(
            self, "eta_energy", _finite(self.eta_energy, "eta_energy", nonnegative=True)
        )
        object.__setattr__(
            self,
            "ood_distance",
            _finite(self.ood_distance, "ood_distance", nonnegative=True),
        )
        object.__setattr__(
            self,
            "ensemble_disagreement",
            _finite(
                self.ensemble_disagreement,
                "ensemble_disagreement",
                nonnegative=True,
            ),
        )
        if not isinstance(self.parameters, Mapping):
            object.__setattr__(self, "parameters", {})
        else:
            object.__setattr__(
                self,
                "parameters",
                {
                    str(k): _finite(v, f"param.{k}")
                    for k, v in sorted(self.parameters.items())
                },
            )


@dataclass(frozen=True, slots=True)
class AcquisitionWeights:
    """Preregistered non-negative weights for the composite acquisition score."""

    eta_weight: float = 0.5
    ood_weight: float = 0.2
    ensemble_weight: float = 0.1
    diversity_weight: float = 0.2

    def __post_init__(self) -> None:
        w1 = _finite(self.eta_weight, "eta_weight", nonnegative=True)
        w2 = _finite(self.ood_weight, "ood_weight", nonnegative=True)
        w3 = _finite(self.ensemble_weight, "ensemble_weight", nonnegative=True)
        w4 = _finite(self.diversity_weight, "diversity_weight", nonnegative=True)
        if w1 + w2 + w3 + w4 <= 0.0:
            raise ValueError("at least one acquisition weight must be positive")
        object.__setattr__(self, "eta_weight", w1)
        object.__setattr__(self, "ood_weight", w2)
        object.__setattr__(self, "ensemble_weight", w3)
        object.__setattr__(self, "diversity_weight", w4)

    def document(self) -> dict[str, float]:
        return {
            "eta_weight": self.eta_weight,
            "ood_weight": self.ood_weight,
            "ensemble_weight": self.ensemble_weight,
            "diversity_weight": self.diversity_weight,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionPlan:
    """Preregistered, content-hashed active-FEM acquisition policy."""

    plan_id: str
    arm_kind: Literal["random_doe", "physics_indicator", "physics_diversity"]
    budget_solves: int
    batch_size: int
    seed: int
    weights: AcquisitionWeights = AcquisitionWeights()
    supervises_training: bool = True
    training_mode: Literal["label_free", "multi_fidelity_label_efficient"] = (
        "multi_fidelity_label_efficient"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            raise ValueError("plan_id must be non-empty text")
        if self.arm_kind not in ACQUISITION_ARMS:
            raise ValueError(f"arm_kind must be one of {ACQUISITION_ARMS}")
        if type(self.budget_solves) is not int or self.budget_solves <= 0:
            raise ValueError("budget_solves must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.batch_size > self.budget_solves:
            raise ValueError("batch_size cannot exceed budget_solves")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.training_mode not in TRAINING_MODES:
            raise ValueError(f"training_mode must be one of {TRAINING_MODES}")
        if self.supervises_training and self.training_mode == "label_free":
            raise ValueError(
                "acquired FEM targets supervising training must be disclosed as "
                "'multi_fidelity_label_efficient', never 'label_free'"
            )

    def document(self) -> dict[str, Any]:
        return {
            "schema": ACQUISITION_PLAN_SCHEMA,
            "plan_id": self.plan_id,
            "arm_kind": self.arm_kind,
            "budget_solves": self.budget_solves,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "weights": self.weights.document(),
            "supervises_training": self.supervises_training,
            "training_mode": self.training_mode,
        }

    @property
    def plan_hash(self) -> str:
        return _sha256_doc(self.document())


@dataclass(frozen=True, slots=True)
class SelectionScoreBreakdown:
    """Auditable per-candidate normalized feature scores at selection time."""

    case_id: str
    rank: int
    total_score: float
    tilde_eta: float
    tilde_ood: float
    tilde_ensemble: float
    tilde_diversity: float

    def document(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "rank": self.rank,
            "total_score": self.total_score,
            "tilde_eta": self.tilde_eta,
            "tilde_ood": self.tilde_ood,
            "tilde_ensemble": self.tilde_ensemble,
            "tilde_diversity": self.tilde_diversity,
        }


@dataclass(frozen=True, slots=True)
class SelectionEvidence:
    """Deterministic, content-hashed record of one active acquisition batch."""

    plan_hash: str
    arm_kind: str
    round_index: int
    pool_size: int
    already_acquired_count: int
    selected_case_ids: tuple[str, ...]
    scores: tuple[SelectionScoreBreakdown, ...]

    def document(self) -> dict[str, Any]:
        return {
            "schema": SELECTION_EVIDENCE_SCHEMA,
            "plan_hash": self.plan_hash,
            "arm_kind": self.arm_kind,
            "round_index": self.round_index,
            "pool_size": self.pool_size,
            "already_acquired_count": self.already_acquired_count,
            "selected_case_ids": list(self.selected_case_ids),
            "scores": [s.document() for s in self.scores],
        }

    @property
    def selection_hash(self) -> str:
        return _sha256_doc(self.document())


def _normalize_scale(values: Sequence[float]) -> tuple[float, ...]:
    """Scale-invariant min-max normalization into [0, 1]; constant inputs -> 0."""
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 1e-15:
        return tuple(0.0 for _ in values)
    return tuple((v - lo) / span for v in values)


def _euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def select_acquisition_batch(
    pool: Sequence[AcquisitionCandidate],
    *,
    already_acquired_ids: Sequence[str],
    plan: AcquisitionPlan,
    round_index: int = 0,
    already_acquired_features: Sequence[tuple[float, ...]] = (),
) -> SelectionEvidence:
    """Deterministically select up to ``plan.batch_size`` TRAIN candidates."""
    if not pool:
        raise ValueError("acquisition pool must be non-empty")
    dim = len(pool[0].feature_vector)
    seen_ids: set[str] = set()
    for cand in pool:
        if cand.partition != "TRAIN":
            raise ValueError("non-TRAIN candidate encountered in acquisition pool")
        if len(cand.feature_vector) != dim:
            raise ValueError("all candidates must share feature_vector dimension")
        if cand.case_id in seen_ids:
            raise ValueError(f"duplicate case_id {cand.case_id!r} in pool")
        seen_ids.add(cand.case_id)

    acquired_set = set(already_acquired_ids)
    remaining_budget = plan.budget_solves - len(acquired_set)
    if remaining_budget <= 0:
        return SelectionEvidence(
            plan_hash=plan.plan_hash,
            arm_kind=plan.arm_kind,
            round_index=round_index,
            pool_size=len(pool),
            already_acquired_count=len(acquired_set),
            selected_case_ids=(),
            scores=(),
        )

    # Canonical sort by case_id ensures order-independent deterministic replay.
    eligible = sorted(
        (c for c in pool if c.case_id not in acquired_set),
        key=lambda c: c.case_id,
    )
    k = min(plan.batch_size, remaining_budget, len(eligible))
    if k == 0:
        return SelectionEvidence(
            plan_hash=plan.plan_hash,
            arm_kind=plan.arm_kind,
            round_index=round_index,
            pool_size=len(pool),
            already_acquired_count=len(acquired_set),
            selected_case_ids=(),
            scores=(),
        )

    if plan.arm_kind == "random_doe":
        rng = random.Random(plan.seed * 1009 + round_index)
        picked = rng.sample(eligible, k)
        picked_ids = tuple(c.case_id for c in picked)
        scores = tuple(
            SelectionScoreBreakdown(c.case_id, rank + 1, 0.0, 0.0, 0.0, 0.0, 0.0)
            for rank, c in enumerate(picked)
        )
        return SelectionEvidence(
            plan_hash=plan.plan_hash,
            arm_kind=plan.arm_kind,
            round_index=round_index,
            pool_size=len(pool),
            already_acquired_count=len(acquired_set),
            selected_case_ids=picked_ids,
            scores=scores,
        )

    # Normalize feature vectors per coordinate for scale-invariant diversity distances.
    col_lo = [min(c.feature_vector[j] for c in eligible) for j in range(dim)]
    col_span = [
        max(max(c.feature_vector[j] for c in eligible) - col_lo[j], 1e-15)
        for j in range(dim)
    ]

    def _norm_feat(vec: tuple[float, ...]) -> tuple[float, ...]:
        return tuple((vec[j] - col_lo[j]) / col_span[j] for j in range(dim))

    norm_feats = {c.case_id: _norm_feat(c.feature_vector) for c in eligible}
    eta_norm = dict(
        zip(
            (c.case_id for c in eligible),
            _normalize_scale([c.eta_energy for c in eligible]),
            strict=True,
        )
    )
    ood_norm = dict(
        zip(
            (c.case_id for c in eligible),
            _normalize_scale([c.ood_distance for c in eligible]),
            strict=True,
        )
    )
    ens_norm = dict(
        zip(
            (c.case_id for c in eligible),
            _normalize_scale([c.ensemble_disagreement for c in eligible]),
            strict=True,
        )
    )

    if plan.arm_kind == "physics_indicator":
        w1, w2, w3, w4 = 1.0, 0.0, 0.0, 0.0
    else:
        w = plan.weights
        w1, w2, w3, w4 = (
            w.eta_weight,
            w.ood_weight,
            w.ensemble_weight,
            w.diversity_weight,
        )

    anchors: list[tuple[float, ...]] = [
        _norm_feat(f) for f in already_acquired_features
    ]
    remaining = list(eligible)
    selected_ids: list[str] = []
    breakdowns: list[SelectionScoreBreakdown] = []

    for rank in range(1, k + 1):
        if anchors:
            raw_divs = [
                min(_euclidean(norm_feats[c.case_id], a) for a in anchors)
                for c in remaining
            ]
            div_norm = dict(
                zip(
                    (c.case_id for c in remaining),
                    _normalize_scale(raw_divs),
                    strict=True,
                )
            )
        else:
            div_norm = {c.case_id: 0.0 for c in remaining}

        best_cand: AcquisitionCandidate | None = None
        best_key: tuple[float, str] | None = None
        best_parts: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

        for cand in remaining:
            cid = cand.case_id
            t_eta = eta_norm[cid]
            t_ood = ood_norm[cid]
            t_ens = ens_norm[cid]
            t_div = div_norm[cid]
            score = w1 * t_eta + w2 * t_ood + w3 * t_ens + w4 * t_div
            # Maximize score, break exact ties deterministically by lexicographic case_id.
            key = (-round(score, 12), cid)
            if best_key is None or key < best_key:
                best_key = key
                best_cand = cand
                best_parts = (score, t_eta, t_ood, t_ens, t_div)

        assert best_cand is not None
        selected_ids.append(best_cand.case_id)
        anchors.append(norm_feats[best_cand.case_id])
        remaining = [c for c in remaining if c.case_id != best_cand.case_id]
        breakdowns.append(
            SelectionScoreBreakdown(
                case_id=best_cand.case_id,
                rank=rank,
                total_score=best_parts[0],
                tilde_eta=best_parts[1],
                tilde_ood=best_parts[2],
                tilde_ensemble=best_parts[3],
                tilde_diversity=best_parts[4],
            )
        )

    return SelectionEvidence(
        plan_hash=plan.plan_hash,
        arm_kind=plan.arm_kind,
        round_index=round_index,
        pool_size=len(pool),
        already_acquired_count=len(acquired_set),
        selected_case_ids=tuple(selected_ids),
        scores=tuple(breakdowns),
    )


@dataclass(frozen=True, slots=True)
class FEMCostLedgerEntry:
    """Auditable record of one charged FEM solve attempt (success or failure)."""

    case_id: str
    partition: str
    attempt_index: int
    succeeded: bool
    wall_seconds: float
    dof_count: int
    result_hash: str | None = None
    failure_reason: str = ""

    def __post_init__(self) -> None:
        if self.partition != "TRAIN":
            raise ValueError("active-FEM cost ledger only records TRAIN acquisitions")
        if type(self.attempt_index) is not int or self.attempt_index < 1:
            raise ValueError("attempt_index must be >= 1")
        _finite(self.wall_seconds, "wall_seconds", nonnegative=True)
        if type(self.dof_count) is not int or self.dof_count < 0:
            raise ValueError("dof_count must be nonnegative")

    def document(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "partition": self.partition,
            "attempt_index": self.attempt_index,
            "succeeded": self.succeeded,
            "wall_seconds": float(self.wall_seconds),
            "dof_count": self.dof_count,
            "result_hash": self.result_hash,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class FEMCostLedger:
    """Immutable ledger of all active-FEM solves, failures, and budget usage."""

    plan_hash: str
    budget_solves: int
    entries: tuple[FEMCostLedgerEntry, ...] = ()

    def append(self, entry: FEMCostLedgerEntry) -> FEMCostLedger:
        if len(self.entries) >= self.budget_solves:
            raise ValueError(
                f"FEM solve budget exhausted ({len(self.entries)}/{self.budget_solves})"
            )
        return FEMCostLedger(
            plan_hash=self.plan_hash,
            budget_solves=self.budget_solves,
            entries=(*self.entries, entry),
        )

    @property
    def attempted_count(self) -> int:
        return len(self.entries)

    @property
    def succeeded_count(self) -> int:
        return sum(1 for e in self.entries if e.succeeded)

    @property
    def failed_count(self) -> int:
        return sum(1 for e in self.entries if not e.succeeded)

    @property
    def total_wall_seconds(self) -> float:
        return math.fsum(e.wall_seconds for e in self.entries)

    def document(self) -> dict[str, Any]:
        return {
            "schema": FEM_COST_LEDGER_SCHEMA,
            "plan_hash": self.plan_hash,
            "budget_solves": self.budget_solves,
            "attempted_count": self.attempted_count,
            "succeeded_count": self.succeeded_count,
            "failed_count": self.failed_count,
            "total_wall_seconds": self.total_wall_seconds,
            "entries": [e.document() for e in self.entries],
        }

    @property
    def ledger_hash(self) -> str:
        return _sha256_doc(self.document())


__all__ = (
    "ACQUISITION_ARMS",
    "ACQUISITION_PLAN_SCHEMA",
    "FEM_COST_LEDGER_SCHEMA",
    "SELECTION_EVIDENCE_SCHEMA",
    "TRAINING_MODES",
    "AcquisitionCandidate",
    "AcquisitionPlan",
    "AcquisitionWeights",
    "FEMCostLedger",
    "FEMCostLedgerEntry",
    "SelectionEvidence",
    "SelectionScoreBreakdown",
    "select_acquisition_batch",
)
