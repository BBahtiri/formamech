"""TRAIN-only eligibility and deterministic checkpoint selection for S13."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import ClassVar

from piginot.contracts import hash_spec
from piginot.contracts.errors import ContractValidationError
from piginot.doe import EvidenceRole
from piginot.training.search import GridSearchSpace

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")

_INCUMBENT_MEDIAN_ABSOLUTE_PEAK_ERROR = 0.08085713846866197
_INCUMBENT_P90_ABSOLUTE_PEAK_ERROR = 0.13634630217960156
_INCUMBENT_MEDIAN_SIGNED_PEAK_ERROR = -0.07762409959121884
_INCUMBENT_U_RELATIVE_L2 = 0.03730866424082414
_INCUMBENT_P_RELATIVE_L2 = 0.1919960359979199
_MINIMUM_UNDERPREDICTION_REDUCTION = 0.10
_MAXIMUM_FIELD_RELATIVE_REGRESSION = 0.02

SELECT_ON_HOLDOUT_MUTANT = "SELECT_ON_HOLDOUT"
CHOOSE_NONRANKED_CHECKPOINT_MUTANT = "CHOOSE_NONRANKED_CHECKPOINT"


class NoEligibleTrialError(ContractValidationError):
    """No registered trial satisfies the frozen checkpoint guardrails."""


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{label} must be non-empty text")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContractValidationError(f"{label} must be a lowercase 256-bit digest")
    return value


def _finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ContractValidationError(f"{label} must be finite")
    return float(value)


def _nonnegative(value: object, label: str) -> float:
    converted = _finite(value, label)
    if converted < 0.0:
        raise ContractValidationError(f"{label} must be non-negative")
    return converted


def _fixed(value: object, expected: float, label: str) -> float:
    converted = _finite(value, label)
    if converted != expected:
        raise ContractValidationError(f"{label} must remain fixed at {expected}")
    return converted


@dataclass(frozen=True, slots=True)
class SelectionMetrics:
    """Preregistered TRAIN-screen metrics for one completed trial."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    median_absolute_peak_error: float
    p90_absolute_peak_error: float
    median_signed_peak_error: float
    p_relative_l2: float
    u_relative_l2: float
    hybrid_loss: float

    def __post_init__(self) -> None:
        for name in (
            "median_absolute_peak_error",
            "p90_absolute_peak_error",
            "p_relative_l2",
            "u_relative_l2",
            "hybrid_loss",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(
            self,
            "median_signed_peak_error",
            _finite(self.median_signed_peak_error, "median_signed_peak_error"),
        )

    @property
    def ranking_key(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.median_absolute_peak_error,
            self.p90_absolute_peak_error,
            max(0.0, -self.median_signed_peak_error),
            self.p_relative_l2,
            self.u_relative_l2,
            self.hybrid_loss,
        )


@dataclass(frozen=True, slots=True)
class CheckpointLineage:
    """Immutable resolution path from a trial to its persisted checkpoint."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    trial_id: str
    run_hash: str
    dataset_snapshot_hash: str
    execution_profile_hash: str
    weights_hash: str
    model_state_hash: str
    checkpoint_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "trial_id", _identifier(self.trial_id, "trial_id"))
        for name in (
            "run_hash",
            "dataset_snapshot_hash",
            "execution_profile_hash",
            "weights_hash",
            "model_state_hash",
            "checkpoint_hash",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class TrialResult:
    """One role-labelled TRAIN-screen result with complete checkpoint lineage."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    lineage: CheckpointLineage
    role: EvidenceRole
    metrics: SelectionMetrics

    def __post_init__(self) -> None:
        if type(self.lineage) is not CheckpointLineage:
            raise ContractValidationError("lineage must be an exact CheckpointLineage")
        if type(self.role) is not EvidenceRole:
            raise ContractValidationError("role must be an exact EvidenceRole")
        if type(self.metrics) is not SelectionMetrics:
            raise ContractValidationError("metrics must be an exact SelectionMetrics")

    @property
    def trial_id(self) -> str:
        return self.lineage.trial_id

    @property
    def run_hash(self) -> str:
        return self.lineage.run_hash

    @property
    def dataset_snapshot_hash(self) -> str:
        return self.lineage.dataset_snapshot_hash

    @property
    def execution_profile_hash(self) -> str:
        return self.lineage.execution_profile_hash

    @property
    def weights_hash(self) -> str:
        return self.lineage.weights_hash

    @property
    def model_state_hash(self) -> str:
        return self.lineage.model_state_hash

    @property
    def checkpoint_hash(self) -> str:
        return self.lineage.checkpoint_hash

    @property
    def ranking_key(self) -> tuple[float, float, float, float, float, float, str]:
        return (*self.metrics.ranking_key, self.trial_id)


@dataclass(frozen=True, slots=True)
class SelectionGuardrails:
    """The immutable S11-A incumbent thresholds preregistered for S13."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    incumbent_median_absolute_peak_error: float = _INCUMBENT_MEDIAN_ABSOLUTE_PEAK_ERROR
    incumbent_p90_absolute_peak_error: float = _INCUMBENT_P90_ABSOLUTE_PEAK_ERROR
    incumbent_median_signed_peak_error: float = _INCUMBENT_MEDIAN_SIGNED_PEAK_ERROR
    incumbent_u_relative_l2: float = _INCUMBENT_U_RELATIVE_L2
    incumbent_p_relative_l2: float = _INCUMBENT_P_RELATIVE_L2
    minimum_underprediction_reduction: float = _MINIMUM_UNDERPREDICTION_REDUCTION
    maximum_field_relative_regression: float = _MAXIMUM_FIELD_RELATIVE_REGRESSION

    def __post_init__(self) -> None:
        for name, expected in (
            (
                "incumbent_median_absolute_peak_error",
                _INCUMBENT_MEDIAN_ABSOLUTE_PEAK_ERROR,
            ),
            (
                "incumbent_p90_absolute_peak_error",
                _INCUMBENT_P90_ABSOLUTE_PEAK_ERROR,
            ),
            (
                "incumbent_median_signed_peak_error",
                _INCUMBENT_MEDIAN_SIGNED_PEAK_ERROR,
            ),
            ("incumbent_u_relative_l2", _INCUMBENT_U_RELATIVE_L2),
            ("incumbent_p_relative_l2", _INCUMBENT_P_RELATIVE_L2),
            (
                "minimum_underprediction_reduction",
                _MINIMUM_UNDERPREDICTION_REDUCTION,
            ),
            (
                "maximum_field_relative_regression",
                _MAXIMUM_FIELD_RELATIVE_REGRESSION,
            ),
        ):
            object.__setattr__(self, name, _fixed(getattr(self, name), expected, name))

    def admits(self, metrics: SelectionMetrics) -> bool:
        """Apply every frozen guardrail without data-dependent relaxation."""

        if type(metrics) is not SelectionMetrics:
            raise TypeError("metrics must be an exact SelectionMetrics")
        incumbent_underprediction = max(
            0.0,
            -self.incumbent_median_signed_peak_error,
        )
        candidate_underprediction = max(0.0, -metrics.median_signed_peak_error)
        return (
            candidate_underprediction
            <= (1.0 - self.minimum_underprediction_reduction)
            * incumbent_underprediction
            and metrics.median_absolute_peak_error
            <= self.incumbent_median_absolute_peak_error
            and metrics.p90_absolute_peak_error
            <= self.incumbent_p90_absolute_peak_error
            and metrics.u_relative_l2
            <= (1.0 + self.maximum_field_relative_regression)
            * self.incumbent_u_relative_l2
            and metrics.p_relative_l2
            <= (1.0 + self.maximum_field_relative_regression)
            * self.incumbent_p_relative_l2
        )


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Frozen guardrails bound to one exact preregistered search space."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    search_space: GridSearchSpace
    guardrails: SelectionGuardrails = field(default_factory=SelectionGuardrails)

    def __post_init__(self) -> None:
        if type(self.search_space) is not GridSearchSpace:
            raise ContractValidationError(
                "search_space must be an exact GridSearchSpace"
            )
        if type(self.guardrails) is not SelectionGuardrails:
            raise ContractValidationError(
                "guardrails must be exact SelectionGuardrails"
            )

    @property
    def registered_trial_ids(self) -> tuple[str, ...]:
        return self.search_space.trial_ids

    @property
    def search_space_hash(self) -> str:
        return self.search_space.search_space_hash

    @property
    def policy_hash(self) -> str:
        return hash_spec(self, "recipe")


def _rank_eligible(
    results: tuple[TrialResult, ...],
    guardrails: SelectionGuardrails,
) -> tuple[TrialResult, ...]:
    return tuple(
        sorted(
            (result for result in results if guardrails.admits(result.metrics)),
            key=lambda result: result.ranking_key,
        )
    )


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    """Auditable decision whose first eligible result is necessarily selected."""

    __canonical_namespace__: ClassVar[str] = "piginot.training"

    policy: SelectionPolicy
    results: tuple[TrialResult, ...]
    ranked_eligible_trial_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.policy) is not SelectionPolicy:
            raise ContractValidationError("policy must be an exact SelectionPolicy")
        canonical_results = _validate_result_matrix(
            self.policy.search_space,
            self.results,
        )
        if canonical_results != self.results:
            raise ContractValidationError("results must be in canonical trial_id order")
        expected_ranking = tuple(
            result.trial_id
            for result in _rank_eligible(self.results, self.policy.guardrails)
        )
        if not expected_ranking:
            raise NoEligibleTrialError(
                "no trial satisfies guardrails; CP-FREEZE must be withheld"
            )
        if type(self.ranked_eligible_trial_ids) is not tuple:
            raise ContractValidationError(
                "ranked_eligible_trial_ids must be an immutable tuple"
            )
        if self.ranked_eligible_trial_ids != expected_ranking:
            raise ContractValidationError(
                "ranked_eligible_trial_ids must match the declared ranking"
            )

    @property
    def selected_result(self) -> TrialResult:
        selected_id = self.ranked_eligible_trial_ids[0]
        return next(result for result in self.results if result.trial_id == selected_id)

    @property
    def selected_lineage(self) -> CheckpointLineage:
        return self.selected_result.lineage

    @property
    def role(self) -> EvidenceRole:
        return EvidenceRole.TRAIN

    @property
    def registered_trial_ids(self) -> tuple[str, ...]:
        return self.policy.registered_trial_ids

    @property
    def search_space_hash(self) -> str:
        return self.policy.search_space_hash

    @property
    def policy_hash(self) -> str:
        return self.policy.policy_hash


def _validate_result_matrix(
    search_space: GridSearchSpace,
    results: tuple[TrialResult, ...],
) -> tuple[TrialResult, ...]:
    if type(search_space) is not GridSearchSpace:
        raise ContractValidationError("search_space must be an exact GridSearchSpace")
    if type(results) is not tuple or any(
        type(result) is not TrialResult for result in results
    ):
        raise ContractValidationError("results must contain exact TrialResult values")
    for result in results:
        if result.role is not EvidenceRole.TRAIN:
            raise PermissionError(
                f"{result.role.value} evidence cannot select an S13 checkpoint"
            )
    registered = search_space.trial_ids
    result_ids = tuple(result.trial_id for result in results)
    if len(result_ids) != len(set(result_ids)):
        raise ContractValidationError("duplicate trial result")
    if set(result_ids) - set(registered):
        raise ContractValidationError(
            "result trial_id is not a registered search trial"
        )
    if len(result_ids) != len(registered) or set(result_ids) != set(registered):
        raise ContractValidationError(
            "results must cover each registered trial exactly once"
        )
    return tuple(sorted(results, key=lambda result: result.trial_id))


def _select_authorized(
    policy: SelectionPolicy,
    results: tuple[TrialResult, ...],
) -> SelectionDecision:
    canonical_results = _validate_result_matrix(policy.search_space, results)
    ranked = _rank_eligible(canonical_results, policy.guardrails)
    if not ranked:
        raise NoEligibleTrialError(
            "no trial satisfies guardrails; CP-FREEZE must be withheld"
        )
    return SelectionDecision(
        policy=policy,
        results=canonical_results,
        ranked_eligible_trial_ids=tuple(result.trial_id for result in ranked),
    )


def select_checkpoint(
    results: tuple[TrialResult, ...],
    *,
    policy: SelectionPolicy,
) -> SelectionDecision:
    """Select the first eligible TRAIN checkpoint under the frozen policy."""

    if type(policy) is not SelectionPolicy:
        raise ContractValidationError("policy must be an exact SelectionPolicy")
    return _select_authorized(policy, results)


def select_training_trial(
    search_space: GridSearchSpace,
    results: tuple[TrialResult, ...],
    *,
    role: EvidenceRole,
    guardrails: SelectionGuardrails | None = None,
) -> SelectionDecision:
    """Select only on TRAIN evidence using the frozen eligibility and rank rule."""

    if type(role) is not EvidenceRole:
        raise TypeError("role must be an EvidenceRole")
    if role is not EvidenceRole.TRAIN:
        raise PermissionError(f"{role.value} evidence cannot select an S13 checkpoint")
    resolved_guardrails = SelectionGuardrails() if guardrails is None else guardrails
    if type(resolved_guardrails) is not SelectionGuardrails:
        raise ContractValidationError("guardrails must be exact SelectionGuardrails")
    return select_checkpoint(
        results,
        policy=SelectionPolicy(search_space, resolved_guardrails),
    )


def _select_on_holdout_mutant_for_test(
    search_space: GridSearchSpace,
    results: tuple[TrialResult, ...],
    *,
    role: EvidenceRole,
) -> TrialResult:
    """Test-only SELECT_ON_HOLDOUT mutant that omits the production role guard."""

    if type(role) is not EvidenceRole or role is not EvidenceRole.HOLDOUT:
        raise ContractValidationError(
            "SELECT_ON_HOLDOUT mutant requires HOLDOUT evidence"
        )
    if type(search_space) is not GridSearchSpace:
        raise ContractValidationError("search_space must be an exact GridSearchSpace")
    if type(results) is not tuple or any(
        type(result) is not TrialResult for result in results
    ):
        raise ContractValidationError("results must contain exact TrialResult values")
    return min(results, key=lambda result: result.ranking_key)


def _choose_nonranked_checkpoint_mutant_for_test(
    decision: SelectionDecision,
) -> TrialResult:
    """Test-only CHOOSE_NONRANKED_CHECKPOINT mutant used by G13 evidence."""

    if type(decision) is not SelectionDecision:
        raise ContractValidationError("decision must be an exact SelectionDecision")
    if len(decision.ranked_eligible_trial_ids) < 2:
        raise ContractValidationError("mutant requires at least two eligible trials")
    mutated_id = decision.ranked_eligible_trial_ids[1]
    return next(result for result in decision.results if result.trial_id == mutated_id)


__all__ = (
    "CHOOSE_NONRANKED_CHECKPOINT_MUTANT",
    "CheckpointLineage",
    "NoEligibleTrialError",
    "SELECT_ON_HOLDOUT_MUTANT",
    "SelectionDecision",
    "SelectionGuardrails",
    "SelectionMetrics",
    "SelectionPolicy",
    "TrialResult",
    "select_checkpoint",
    "select_training_trial",
)
