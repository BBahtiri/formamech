"""Preregistered S11-to-S12 DogBone architecture-contingency screen."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import ClassVar

from piginot.contracts import canonical_json_bytes, hash_spec
from piginot.doe import EvidenceRole, PublishedCase, PublishedCaseManifest
from piginot.doe.publication import PublishedCaseStatus

CONTINGENCY_PROTOCOL_ID = "dogbone-s11c-architecture-screen-v1"
BASELINE_CANDIDATE_ID = "A-s11-baseline"
_MANIFEST_HASH = "ce95c03f5b22308a7aa9bf627966a6b21300a4e7b0272d3affa7dd503af564a9"
_TRAINING_SNAPSHOT_HASH = (
    "af04a059e0efb647a1ba768f722d2fde7500d06d11695d1132a0b26dded24845"
)
_DEVELOPMENT_SNAPSHOT_HASH = (
    "5c6c59c621341d35797bcef3bca8f44510fcbfa90b329858d2e9a3e407824a77"
)
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_MODULE_DIGEST_PATTERN = re.compile(r"blake2b256:[0-9a-f]{64}")
_CANONICAL_NAMESPACE = "piginot.campaigns"

RAW_PARAMETER_NAMES = (
    "L_total",
    "W_grip",
    "W_gauge",
    "R_fillet",
    "E",
    "nu",
    "nominal_axial_strain",
)
DERIVED_PARAMETER_NAMES = (
    "arc_feasibility_ratio",
    "fillet_to_gauge_ratio",
    "grip_to_gauge_ratio",
    "length_to_gauge_ratio",
)
OPERATOR_TENSOR_DTYPE = "torch.float64"
INITIALIZATION_POLICY_ID = "s11c-fresh-shared-state-template-state-hash-v2"
CANDIDATE_SPECIFIC_INITIALIZATION_SEED = 11123
SAMPLING_SCHEDULE_POLICY_ID = "s11c-shared-case-and-query-schedule-v1"
OBJECTIVE_POLICY_ID = "s11c-supervised-architecture-isolation-v1"
SUPERVISED_OBJECTIVE_TERMS = (
    "normalized_u_mse",
    "normalized_in_plane_P_mse",
    "root_qoi_peak_P_mse",
)
DEFERRED_PHYSICS_OBJECTIVE_TERMS = (
    "strong_equilibrium",
    "weak_equilibrium",
    "constitutive_in_plane",
    "angular_momentum",
    "traction_free",
)
DATA_PREPARATION_POLICY_ID = "s11c-s10-p2-vertices-boundary-arclength-v1"
NORMALIZATION_POLICY_ID = (
    "s11c-fit-zscore-u-by-prescribed-displacement-p-by-760mpa-v1"
)
QOI_POLICY_ID = "s11c-right-arc-minus-right-grip-root-peak-v1"
METRIC_POLICY_ID = "s11c-relative-l2-root-frobenius-linear-quantile-v1"
OPTIMIZER_POLICY_ID = "s11c-torch-adam-no-foreach-no-fused-v1"
SOURCE_ACCESS_POLICY_ID = "s11c-verified-fit-screen-durable-development-v1"
DEVELOPMENT_CONFIRMATION_POLICY_ID = (
    "s11c-baseline-vs-train-winner-same-guardrails-v1"
)
BOUNDARY_POINT_COUNT = 320
BOUNDARY_TAG_NAMES = (
    "bottom",
    "gauge_top",
    "left_symmetry",
    "right_arc",
    "right_grip",
)
STRESS_REFERENCE_MPA = 760.0


def _finite(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _nonnegative(value: object, label: str) -> float:
    number = _finite(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{label} must be canonical non-empty text")
    return value


def _digest(value: object, label: str) -> str:
    digest = _identifier(value, label)
    if _HASH_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a lowercase 256-bit digest")
    return digest


def _module_digest(value: object, label: str) -> str:
    digest = _identifier(value, label)
    if _MODULE_DIGEST_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be a blake2b256 module digest")
    return digest


def _identifiers(values: object, label: str, count: int) -> tuple[str, ...]:
    if type(values) is not tuple or len(values) != count:
        raise ValueError(f"{label} must contain exactly {count} case IDs")
    result = tuple(_identifier(value, f"{label} entry") for value in values)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be sorted and unique")
    return result


@dataclass(frozen=True, slots=True)
class ArchitectureCandidate:
    """One finite architecture choice in the preregistered factorial screen."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    candidate_id: str
    parameter_names: tuple[str, ...]
    local_boundary_kernel_scales: tuple[float, ...]
    stress_enrichment_layers: int
    operator_module_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _identifier(self.candidate_id, "candidate_id"),
        )
        names = tuple(self.parameter_names)
        if not names or names != tuple(dict.fromkeys(names)):
            raise ValueError("candidate parameter_names must be non-empty and unique")
        for name in names:
            _identifier(name, "candidate parameter name")
        scales = tuple(
            float(_finite(value, "local kernel scale"))
            for value in self.local_boundary_kernel_scales
        )
        if any(scale <= 0.0 for scale in scales) or scales != tuple(
            sorted(set(scales))
        ):
            raise ValueError(
                "candidate local scales must be unique increasing positives"
            )
        if (
            isinstance(self.stress_enrichment_layers, bool)
            or not isinstance(self.stress_enrichment_layers, int)
            or self.stress_enrichment_layers < 0
        ):
            raise ValueError("candidate stress enrichment depth must be non-negative")
        _module_digest(
            self.operator_module_digest,
            "candidate operator_module_digest",
        )
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "local_boundary_kernel_scales", scales)


_CANDIDATES = (
    ArchitectureCandidate(
        BASELINE_CANDIDATE_ID,
        RAW_PARAMETER_NAMES,
        (),
        0,
        "blake2b256:ac064556f53e573b4f9de63a2610077dfd38141b6eef695731302cea3a560f47",
    ),
    ArchitectureCandidate(
        "B-derived-ratio-film",
        (*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES),
        (),
        0,
        "blake2b256:b0ed86e230926d2b7fbdeac368bb2299bcc1a1f473d20640af3e634944ad76f8",
    ),
    ArchitectureCandidate(
        "C-local-stress-enrichment",
        RAW_PARAMETER_NAMES,
        (0.08, 0.25),
        2,
        "blake2b256:cab946682fca6e1f4f5bb51170b9a1949da900ca8c0db6a118ef250d26c61879",
    ),
    ArchitectureCandidate(
        "D-derived-local-stress",
        (*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES),
        (0.08, 0.25),
        2,
        "blake2b256:8396ce60f458057e9b45173c3657b98bb121399146a5321989bf77967f9c4c1f",
    ),
)


@dataclass(frozen=True, slots=True)
class ArchitectureScreenProtocol:
    """Immutable pre-label contract for the bounded contingency experiment."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    protocol_id: str
    public_case_manifest_hash: str
    training_snapshot_hash: str
    development_snapshot_hash: str
    raw_parameter_names: tuple[str, ...]
    derived_parameter_names: tuple[str, ...]
    fit_case_ids: tuple[str, ...]
    screen_case_ids: tuple[str, ...]
    development_case_ids: tuple[str, ...]
    candidates: tuple[ArchitectureCandidate, ...]
    optimizer_steps: int
    case_batch_size: int
    query_points_per_case: int
    operator_tensor_dtype: str
    initialization_policy_id: str
    common_state_template_candidate_id: str
    candidate_specific_initialization_seed: int
    sampling_schedule_policy_id: str
    objective_policy_id: str
    supervised_objective_terms: tuple[str, ...]
    physics_objectives_deferred_to_step: str
    deferred_physics_objective_terms: tuple[str, ...]
    data_preparation_policy_id: str
    normalization_policy_id: str
    qoi_policy_id: str
    metric_policy_id: str
    optimizer_policy_id: str
    source_access_policy_id: str
    development_confirmation_policy_id: str
    boundary_point_count: int
    boundary_tag_names: tuple[str, ...]
    stress_reference_mpa: float
    qoi_fraction: float
    peak_supervised_p_weight: float
    model_seed: int
    sampler_seed: int
    learning_rate: float
    gradient_clip_norm: float
    minimum_signed_bias_reduction: float
    minimum_median_peak_reduction: float
    maximum_field_relative_regression: float

    def __post_init__(self) -> None:
        if self.protocol_id != CONTINGENCY_PROTOCOL_ID:
            raise ValueError("unexpected contingency protocol_id")
        for value, label in (
            (self.public_case_manifest_hash, "public_case_manifest_hash"),
            (self.training_snapshot_hash, "training_snapshot_hash"),
            (self.development_snapshot_hash, "development_snapshot_hash"),
        ):
            _digest(value, label)
        if tuple(self.raw_parameter_names) != RAW_PARAMETER_NAMES:
            raise ValueError("raw parameter names changed")
        if tuple(self.derived_parameter_names) != DERIVED_PARAMETER_NAMES:
            raise ValueError("derived parameter names changed")
        fit = _identifiers(self.fit_case_ids, "fit_case_ids", 192)
        screen = _identifiers(self.screen_case_ids, "screen_case_ids", 64)
        if (
            type(self.development_case_ids) is not tuple
            or len(self.development_case_ids) != 64
        ):
            raise ValueError(
                "development_case_ids must contain exactly 64 DEVELOPMENT case IDs"
            )
        development = _identifiers(
            self.development_case_ids,
            "development_case_ids",
            64,
        )
        if set(fit) & set(screen):
            raise ValueError("fit and screen case IDs must be disjoint")
        if set(fit) & set(development) or set(screen) & set(development):
            raise ValueError("TRAIN and DEVELOPMENT case IDs must be disjoint")
        if tuple(self.candidates) != _CANDIDATES:
            raise ValueError("candidate matrix changed")
        if self.operator_tensor_dtype != OPERATOR_TENSOR_DTYPE:
            raise ValueError("operator_tensor_dtype changed")
        if self.initialization_policy_id != INITIALIZATION_POLICY_ID:
            raise ValueError("initialization_policy_id changed")
        if self.common_state_template_candidate_id != BASELINE_CANDIDATE_ID:
            raise ValueError("common_state_template_candidate_id changed")
        if self.sampling_schedule_policy_id != SAMPLING_SCHEDULE_POLICY_ID:
            raise ValueError("sampling_schedule_policy_id changed")
        if self.objective_policy_id != OBJECTIVE_POLICY_ID:
            raise ValueError("objective_policy_id changed")
        if tuple(self.supervised_objective_terms) != SUPERVISED_OBJECTIVE_TERMS:
            raise ValueError("supervised_objective_terms changed")
        if self.physics_objectives_deferred_to_step != "S12":
            raise ValueError("physics_objectives_deferred_to_step must remain S12")
        if (
            tuple(self.deferred_physics_objective_terms)
            != DEFERRED_PHYSICS_OBJECTIVE_TERMS
        ):
            raise ValueError("deferred_physics_objective_terms changed")
        for value, expected, label in (
            (
                self.data_preparation_policy_id,
                DATA_PREPARATION_POLICY_ID,
                "data_preparation_policy_id",
            ),
            (
                self.normalization_policy_id,
                NORMALIZATION_POLICY_ID,
                "normalization_policy_id",
            ),
            (self.qoi_policy_id, QOI_POLICY_ID, "qoi_policy_id"),
            (self.metric_policy_id, METRIC_POLICY_ID, "metric_policy_id"),
            (
                self.optimizer_policy_id,
                OPTIMIZER_POLICY_ID,
                "optimizer_policy_id",
            ),
            (
                self.source_access_policy_id,
                SOURCE_ACCESS_POLICY_ID,
                "source_access_policy_id",
            ),
            (
                self.development_confirmation_policy_id,
                DEVELOPMENT_CONFIRMATION_POLICY_ID,
                "development_confirmation_policy_id",
            ),
        ):
            if value != expected:
                raise ValueError(f"{label} changed")
        if self.boundary_point_count != BOUNDARY_POINT_COUNT:
            raise ValueError("boundary_point_count changed")
        if tuple(self.boundary_tag_names) != BOUNDARY_TAG_NAMES:
            raise ValueError("boundary_tag_names changed")
        if self.stress_reference_mpa != STRESS_REFERENCE_MPA:
            raise ValueError("stress_reference_mpa changed")
        for integer, expected_integer, integer_label in (
            (self.optimizer_steps, 1_500, "optimizer_steps"),
            (self.case_batch_size, 4, "case_batch_size"),
            (self.query_points_per_case, 256, "query_points_per_case"),
            (
                self.candidate_specific_initialization_seed,
                CANDIDATE_SPECIFIC_INITIALIZATION_SEED,
                "candidate_specific_initialization_seed",
            ),
        ):
            if type(integer) is not int or integer != expected_integer:
                raise ValueError(
                    f"{integer_label} must remain fixed at {expected_integer}"
                )
        for number, expected_number, number_label in (
            (self.qoi_fraction, 0.25, "qoi_fraction"),
            (self.peak_supervised_p_weight, 1.0, "peak_supervised_p_weight"),
            (self.learning_rate, 1.0e-4, "learning_rate"),
            (self.gradient_clip_norm, 1.0, "gradient_clip_norm"),
            (
                self.minimum_signed_bias_reduction,
                0.20,
                "minimum_signed_bias_reduction",
            ),
            (
                self.minimum_median_peak_reduction,
                0.10,
                "minimum_median_peak_reduction",
            ),
            (
                self.maximum_field_relative_regression,
                0.02,
                "maximum_field_relative_regression",
            ),
        ):
            if _finite(number, number_label) != expected_number:
                raise ValueError(
                    f"{number_label} must remain fixed at {expected_number}"
                )
        for seed, expected_seed, seed_label in (
            (self.model_seed, 11121, "model_seed"),
            (self.sampler_seed, 11122, "sampler_seed"),
        ):
            if type(seed) is not int or seed != expected_seed:
                raise ValueError(f"{seed_label} must remain fixed at {expected_seed}")
        object.__setattr__(self, "fit_case_ids", fit)
        object.__setattr__(self, "screen_case_ids", screen)
        object.__setattr__(self, "development_case_ids", development)
        object.__setattr__(self, "candidates", tuple(self.candidates))

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self)

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")


@dataclass(frozen=True, slots=True)
class ArchitectureScreenMetrics:
    """Selection statistics for one fixed-budget TRAIN-screen candidate."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    candidate_id: str
    median_signed_peak_error: float
    median_absolute_peak_error: float
    p90_absolute_peak_error: float
    median_u_relative_l2: float
    median_p_relative_l2: float
    trainable_parameter_count: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _identifier(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(
            self,
            "median_signed_peak_error",
            _finite(self.median_signed_peak_error, "median_signed_peak_error"),
        )
        for name in (
            "median_absolute_peak_error",
            "p90_absolute_peak_error",
            "median_u_relative_l2",
            "median_p_relative_l2",
            "elapsed_seconds",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if (
            type(self.trainable_parameter_count) is not int
            or self.trainable_parameter_count <= 0
        ):
            raise ValueError("trainable_parameter_count must be positive")


@dataclass(frozen=True, slots=True)
class ArchitectureSelection:
    """Deterministic TRAIN-screen decision made without DEVELOPMENT labels."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    selected_candidate_id: str
    eligible_candidate_ids: tuple[str, ...]
    fallback_to_baseline: bool

    def __post_init__(self) -> None:
        candidate_ids = tuple(candidate.candidate_id for candidate in _CANDIDATES)
        selected = _identifier(self.selected_candidate_id, "selected_candidate_id")
        if selected not in candidate_ids:
            raise ValueError("selected_candidate_id is not a registered candidate")
        if type(self.eligible_candidate_ids) is not tuple:
            raise ValueError("eligible_candidate_ids must be a tuple")
        eligible = tuple(
            _identifier(value, "eligible candidate ID")
            for value in self.eligible_candidate_ids
        )
        expected_order = tuple(
            candidate_id
            for candidate_id in candidate_ids[1:]
            if candidate_id in set(eligible)
        )
        if eligible != expected_order:
            raise ValueError(
                "eligible_candidate_ids must be unique registered candidates in protocol order"
            )
        if type(self.fallback_to_baseline) is not bool:
            raise ValueError("fallback_to_baseline must be boolean")
        if self.fallback_to_baseline != (selected == BASELINE_CANDIDATE_ID):
            raise ValueError("fallback_to_baseline must match the selected candidate")
        if self.fallback_to_baseline and eligible:
            raise ValueError("eligible_candidate_ids must be empty on fallback")
        if not self.fallback_to_baseline and selected not in eligible:
            raise ValueError("selected candidate must be eligible")
        object.__setattr__(self, "selected_candidate_id", selected)
        object.__setattr__(self, "eligible_candidate_ids", eligible)


def _case_rank(case_id: str) -> bytes:
    payload = f"{CONTINGENCY_PROTOCOL_ID}\0{case_id}".encode()
    return hashlib.blake2b(payload, digest_size=32).digest()


def _derived_cell(case: PublishedCase) -> tuple[int, ...]:
    return tuple(case.strata[name] for name in DERIVED_PARAMETER_NAMES)


def _split_train_cases(
    cases: tuple[PublishedCase, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    groups: dict[tuple[int, ...], list[PublishedCase]] = {}
    for case in cases:
        groups.setdefault(_derived_cell(case), []).append(case)
    quotas = {key: len(group) // 4 for key, group in groups.items()}
    remaining = 64 - sum(quotas.values())
    remainders = sorted(
        groups,
        key=lambda key: (-(len(groups[key]) % 4), key),
    )
    for key in remainders[:remaining]:
        quotas[key] += 1

    screen: set[str] = set()
    for key, group in groups.items():
        ranked = sorted(
            group, key=lambda case: (_case_rank(case.case_id), case.case_id)
        )
        screen.update(case.case_id for case in ranked[: quotas[key]])
    all_ids = {case.case_id for case in cases}
    if len(screen) != 64 or len(all_ids) != 256:
        raise ValueError("TRAIN split did not produce the fixed 192/64 population")
    return tuple(sorted(all_ids - screen)), tuple(sorted(screen))


def build_architecture_screen_protocol(
    manifest: PublishedCaseManifest,
) -> ArchitectureScreenProtocol:
    """Compile the exact public case IDs before any FEM label is opened."""

    if type(manifest) is not PublishedCaseManifest:
        raise TypeError("manifest must be an exact PublishedCaseManifest")
    if manifest.content_hash != _MANIFEST_HASH:
        raise ValueError("public DogBone manifest hash changed")
    active_train = tuple(
        case
        for case in manifest.cases
        if case.role is EvidenceRole.TRAIN and case.status is PublishedCaseStatus.ACTIVE
    )
    active_development = tuple(
        case
        for case in manifest.cases
        if case.role is EvidenceRole.DEVELOPMENT
        and case.status is PublishedCaseStatus.ACTIVE
    )
    if len(active_train) != 256 or len(active_development) != 64:
        raise ValueError("manifest must expose 256 TRAIN and 64 DEVELOPMENT cases")
    fit, screen = _split_train_cases(active_train)
    return ArchitectureScreenProtocol(
        protocol_id=CONTINGENCY_PROTOCOL_ID,
        public_case_manifest_hash=manifest.content_hash,
        training_snapshot_hash=_TRAINING_SNAPSHOT_HASH,
        development_snapshot_hash=_DEVELOPMENT_SNAPSHOT_HASH,
        raw_parameter_names=RAW_PARAMETER_NAMES,
        derived_parameter_names=DERIVED_PARAMETER_NAMES,
        fit_case_ids=fit,
        screen_case_ids=screen,
        development_case_ids=tuple(sorted(case.case_id for case in active_development)),
        candidates=_CANDIDATES,
        optimizer_steps=1_500,
        case_batch_size=4,
        query_points_per_case=256,
        operator_tensor_dtype=OPERATOR_TENSOR_DTYPE,
        initialization_policy_id=INITIALIZATION_POLICY_ID,
        common_state_template_candidate_id=BASELINE_CANDIDATE_ID,
        candidate_specific_initialization_seed=CANDIDATE_SPECIFIC_INITIALIZATION_SEED,
        sampling_schedule_policy_id=SAMPLING_SCHEDULE_POLICY_ID,
        objective_policy_id=OBJECTIVE_POLICY_ID,
        supervised_objective_terms=SUPERVISED_OBJECTIVE_TERMS,
        physics_objectives_deferred_to_step="S12",
        deferred_physics_objective_terms=DEFERRED_PHYSICS_OBJECTIVE_TERMS,
        data_preparation_policy_id=DATA_PREPARATION_POLICY_ID,
        normalization_policy_id=NORMALIZATION_POLICY_ID,
        qoi_policy_id=QOI_POLICY_ID,
        metric_policy_id=METRIC_POLICY_ID,
        optimizer_policy_id=OPTIMIZER_POLICY_ID,
        source_access_policy_id=SOURCE_ACCESS_POLICY_ID,
        development_confirmation_policy_id=DEVELOPMENT_CONFIRMATION_POLICY_ID,
        boundary_point_count=BOUNDARY_POINT_COUNT,
        boundary_tag_names=BOUNDARY_TAG_NAMES,
        stress_reference_mpa=STRESS_REFERENCE_MPA,
        qoi_fraction=0.25,
        peak_supervised_p_weight=1.0,
        model_seed=11121,
        sampler_seed=11122,
        learning_rate=1.0e-4,
        gradient_clip_norm=1.0,
        minimum_signed_bias_reduction=0.20,
        minimum_median_peak_reduction=0.10,
        maximum_field_relative_regression=0.02,
    )


def select_architecture_candidate(
    protocol: ArchitectureScreenProtocol,
    metrics: tuple[ArchitectureScreenMetrics, ...],
) -> ArchitectureSelection:
    """Apply the preregistered TRAIN-screen rule, falling back to S11."""

    if type(protocol) is not ArchitectureScreenProtocol:
        raise TypeError("protocol must be an ArchitectureScreenProtocol")
    if type(metrics) is not tuple or any(
        type(metric) is not ArchitectureScreenMetrics for metric in metrics
    ):
        raise TypeError("metrics must contain exact ArchitectureScreenMetrics")
    by_id = {metric.candidate_id: metric for metric in metrics}
    candidate_ids = tuple(candidate.candidate_id for candidate in protocol.candidates)
    if len(by_id) != len(metrics) or set(by_id) != set(candidate_ids):
        raise ValueError("metrics must cover the exact candidate matrix once")
    baseline = by_id[BASELINE_CANDIDATE_ID]
    eligible: list[ArchitectureScreenMetrics] = []
    for candidate_id in candidate_ids[1:]:
        candidate = by_id[candidate_id]
        if abs(baseline.median_signed_peak_error) == 0.0:
            continue
        signed_limit = (1.0 - protocol.minimum_signed_bias_reduction) * abs(
            baseline.median_signed_peak_error
        )
        absolute_limit = (1.0 - protocol.minimum_median_peak_reduction) * (
            baseline.median_absolute_peak_error
        )
        field_limit = 1.0 + protocol.maximum_field_relative_regression
        if (
            abs(candidate.median_signed_peak_error) <= signed_limit
            and candidate.median_absolute_peak_error <= absolute_limit
            and candidate.p90_absolute_peak_error <= baseline.p90_absolute_peak_error
            and candidate.median_u_relative_l2
            <= field_limit * baseline.median_u_relative_l2
            and candidate.median_p_relative_l2
            <= field_limit * baseline.median_p_relative_l2
        ):
            eligible.append(candidate)
    ranked = sorted(
        eligible,
        key=lambda item: (
            item.median_absolute_peak_error,
            item.p90_absolute_peak_error,
            item.median_p_relative_l2,
            item.trainable_parameter_count,
            item.candidate_id,
        ),
    )
    selected = baseline if not ranked else ranked[0]
    return ArchitectureSelection(
        selected_candidate_id=selected.candidate_id,
        eligible_candidate_ids=tuple(
            candidate_id
            for candidate_id in candidate_ids[1:]
            if candidate_id in {item.candidate_id for item in eligible}
        ),
        fallback_to_baseline=not ranked,
    )


__all__ = (
    "ArchitectureCandidate",
    "ArchitectureScreenMetrics",
    "ArchitectureScreenProtocol",
    "ArchitectureSelection",
    "BASELINE_CANDIDATE_ID",
    "BOUNDARY_POINT_COUNT",
    "BOUNDARY_TAG_NAMES",
    "CANDIDATE_SPECIFIC_INITIALIZATION_SEED",
    "CONTINGENCY_PROTOCOL_ID",
    "DATA_PREPARATION_POLICY_ID",
    "DEVELOPMENT_CONFIRMATION_POLICY_ID",
    "DEFERRED_PHYSICS_OBJECTIVE_TERMS",
    "DERIVED_PARAMETER_NAMES",
    "INITIALIZATION_POLICY_ID",
    "METRIC_POLICY_ID",
    "NORMALIZATION_POLICY_ID",
    "OBJECTIVE_POLICY_ID",
    "OPERATOR_TENSOR_DTYPE",
    "OPTIMIZER_POLICY_ID",
    "QOI_POLICY_ID",
    "RAW_PARAMETER_NAMES",
    "SAMPLING_SCHEDULE_POLICY_ID",
    "SOURCE_ACCESS_POLICY_ID",
    "SUPERVISED_OBJECTIVE_TERMS",
    "STRESS_REFERENCE_MPA",
    "build_architecture_screen_protocol",
    "select_architecture_candidate",
)
