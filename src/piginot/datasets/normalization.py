"""Leakage-safe, reversible normalization for S10 dataset fields."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import ClassVar

from piginot.contracts import CanonicalQuantity, Dimension, Unit, hash_spec
from piginot.datasets.snapshots import RoleDatasetSnapshot
from piginot.doe import EvidenceRole, PublishedCaseStatus

_CANONICAL_NAMESPACE = "piginot.datasets"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_PUBLIC_ROLES = frozenset(
    {
        EvidenceRole.PILOT,
        EvidenceRole.TRAIN,
        EvidenceRole.DEVELOPMENT,
        EvidenceRole.LEGACY,
    }
)


class NormalizationError(ValueError):
    """Base class for invalid normalization operations."""


class NormalizationLeakageError(NormalizationError):
    """Statistics were requested from a forbidden role or reserve case."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NormalizationError(f"{label} must be non-empty text")
    return value


def _content_hash(value: object, label: str) -> str:
    content_hash = _text(value, label)
    if _HASH_PATTERN.fullmatch(content_hash) is None:
        raise NormalizationError(f"{label} must be a lowercase BLAKE2b-256 hash")
    return content_hash


def _finite_values(values: object, label: str) -> tuple[float, ...]:
    if not isinstance(values, tuple) or not values:
        raise NormalizationError(f"{label} must be a non-empty tuple")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NormalizationError(f"{label} must contain real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise NormalizationError(f"{label} must contain finite numbers")
        result.append(number)
    return tuple(result)


def _canonical_unit(unit: object, label: str) -> Unit:
    if not isinstance(unit, Unit) or not unit.is_canonical:
        raise NormalizationError(f"{label} must be a canonical Unit")
    return unit


@dataclass(frozen=True, slots=True)
class SampleProvenance:
    """Stable identity linking one statistic back to its source artifact."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    sample_id: str
    case_id: str
    leakage_group_id: str
    source_hash: str
    role: EvidenceRole
    status: PublishedCaseStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _text(self.sample_id, "sample_id"))
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(
            self,
            "leakage_group_id",
            _text(self.leakage_group_id, "leakage_group_id"),
        )
        object.__setattr__(
            self,
            "source_hash",
            _content_hash(self.source_hash, "source_hash"),
        )
        role = EvidenceRole(self.role)
        if role not in _PUBLIC_ROLES:
            raise NormalizationLeakageError(
                "HOLDOUT is outside the public S10 normalization contract"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "status", PublishedCaseStatus(self.status))


@dataclass(frozen=True, slots=True)
class NormalizationRecord:
    """One dimensional vector and the immutable case provenance that produced it."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    sample_id: str
    case_id: str
    leakage_group_id: str
    source_hash: str
    role: EvidenceRole
    status: PublishedCaseStatus
    unit: Unit
    values: tuple[float, ...]
    dimension: Dimension | None = None

    def __post_init__(self) -> None:
        provenance = SampleProvenance(
            self.sample_id,
            self.case_id,
            self.leakage_group_id,
            self.source_hash,
            self.role,
            self.status,
        )
        object.__setattr__(self, "sample_id", provenance.sample_id)
        object.__setattr__(self, "case_id", provenance.case_id)
        object.__setattr__(self, "leakage_group_id", provenance.leakage_group_id)
        object.__setattr__(self, "source_hash", provenance.source_hash)
        object.__setattr__(self, "role", EvidenceRole(self.role))
        object.__setattr__(self, "status", PublishedCaseStatus(self.status))
        unit = _canonical_unit(self.unit, "unit")
        dimension = unit.dimension if self.dimension is None else self.dimension
        if not isinstance(dimension, Dimension) or not unit.supports_dimension(
            dimension
        ):
            raise NormalizationError("dimension must be supported by the field unit")
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "values", _finite_values(self.values, "values"))

    @property
    def provenance(self) -> SampleProvenance:
        return SampleProvenance(
            self.sample_id,
            self.case_id,
            self.leakage_group_id,
            self.source_hash,
            self.role,
            self.status,
        )


@dataclass(frozen=True, slots=True)
class NormalizationStatistics:
    """TRAIN-only statistics in nondimensional reference-scale coordinates."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    unit: Unit
    dimension: Dimension
    reference_scale: CanonicalQuantity
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    training_provenance: tuple[SampleProvenance, ...]
    training_snapshot_hash: str

    def __post_init__(self) -> None:
        unit = _canonical_unit(self.unit, "unit")
        if not isinstance(self.dimension, Dimension):
            raise NormalizationError("dimension must be a Dimension")
        if not isinstance(self.reference_scale, CanonicalQuantity):
            raise NormalizationError("reference_scale must be a CanonicalQuantity")
        if self.reference_scale.value <= 0.0:
            raise NormalizationError("reference scale must be positive")
        if self.reference_scale.unit is not unit:
            raise NormalizationError("reference scale unit must match the field unit")
        if self.reference_scale.dimension is not self.dimension:
            raise NormalizationError("reference scale dimension must match the field")
        mean = _finite_values(self.mean, "mean")
        deviation = _finite_values(
            self.standard_deviation,
            "standard_deviation",
        )
        if len(mean) != len(deviation):
            raise NormalizationError("mean and standard deviation widths must match")
        if any(value <= 0.0 for value in deviation):
            raise NormalizationError("standard deviations must be positive")
        provenance = tuple(self.training_provenance)
        if not provenance or any(
            not isinstance(item, SampleProvenance) for item in provenance
        ):
            raise NormalizationError(
                "training_provenance must contain SampleProvenance values"
            )
        if provenance != tuple(sorted(provenance, key=_provenance_key)):
            raise NormalizationError("training_provenance must be canonically ordered")
        if any(
            item.role is not EvidenceRole.TRAIN
            or item.status is not PublishedCaseStatus.ACTIVE
            for item in provenance
        ):
            raise NormalizationLeakageError(
                "NORMALIZE_ON_ALL_SPLITS: statistics provenance must be ACTIVE TRAIN"
            )
        identities = tuple(_provenance_key(item) for item in provenance)
        if len(identities) != len(set(identities)):
            raise NormalizationError("training provenance must be unique")
        snapshot_hash = _content_hash(
            self.training_snapshot_hash,
            "training_snapshot_hash",
        )

        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", deviation)
        object.__setattr__(self, "training_provenance", provenance)
        object.__setattr__(self, "training_snapshot_hash", snapshot_hash)

    @property
    def content_hash(self) -> str:
        """Return the canonical identity carried by every transformed sample."""

        return hash_spec(self, "content")


@dataclass(frozen=True, slots=True)
class NormalizedSample:
    """A dimensionless sample that retains its complete source provenance."""

    __canonical_namespace__: ClassVar[str] = _CANONICAL_NAMESPACE

    sample_id: str
    case_id: str
    leakage_group_id: str
    source_hash: str
    role: EvidenceRole
    status: PublishedCaseStatus
    source_unit: Unit
    source_dimension: Dimension
    values: tuple[float, ...]
    normalization_hash: str

    def __post_init__(self) -> None:
        provenance = SampleProvenance(
            self.sample_id,
            self.case_id,
            self.leakage_group_id,
            self.source_hash,
            self.role,
            self.status,
        )
        object.__setattr__(self, "sample_id", provenance.sample_id)
        object.__setattr__(self, "case_id", provenance.case_id)
        object.__setattr__(self, "leakage_group_id", provenance.leakage_group_id)
        object.__setattr__(self, "source_hash", provenance.source_hash)
        object.__setattr__(self, "role", EvidenceRole(self.role))
        object.__setattr__(self, "status", PublishedCaseStatus(self.status))
        source_unit = _canonical_unit(self.source_unit, "source_unit")
        if not isinstance(
            self.source_dimension, Dimension
        ) or not source_unit.supports_dimension(self.source_dimension):
            raise NormalizationError(
                "source_dimension must be supported by source_unit"
            )
        object.__setattr__(
            self,
            "source_unit",
            source_unit,
        )
        object.__setattr__(self, "values", _finite_values(self.values, "values"))
        object.__setattr__(
            self,
            "normalization_hash",
            _content_hash(self.normalization_hash, "normalization_hash"),
        )

    @property
    def provenance(self) -> SampleProvenance:
        return SampleProvenance(
            self.sample_id,
            self.case_id,
            self.leakage_group_id,
            self.source_hash,
            self.role,
            self.status,
        )


def _provenance_key(
    item: SampleProvenance,
) -> tuple[str, str, str, str, str, str]:
    return (
        item.sample_id,
        item.case_id,
        item.leakage_group_id,
        item.source_hash,
        item.role.value,
        item.status.value,
    )


def _record_key(
    record: NormalizationRecord,
) -> tuple[str, str, str, str, str, str]:
    return _provenance_key(record.provenance)


def fit_train_normalization(
    records: tuple[NormalizationRecord, ...],
    *,
    reference_scale: CanonicalQuantity,
    training_snapshot: RoleDatasetSnapshot,
) -> NormalizationStatistics:
    """Fit statistics only from records resolved by one TRAIN snapshot."""

    if not isinstance(records, tuple) or not records:
        raise NormalizationError("records must be a non-empty tuple")
    if any(not isinstance(record, NormalizationRecord) for record in records):
        raise NormalizationError("records must contain NormalizationRecord values")
    for record in records:
        if record.role is not EvidenceRole.TRAIN:
            raise NormalizationLeakageError(
                "NORMALIZE_ON_ALL_SPLITS: statistics require TRAIN records only"
            )
        if record.status is not PublishedCaseStatus.ACTIVE:
            raise NormalizationLeakageError(
                "statistics require ACTIVE TRAIN records; reserves are forbidden"
            )
    if type(training_snapshot) is not RoleDatasetSnapshot:
        raise NormalizationError(
            "training_snapshot must be an exact RoleDatasetSnapshot"
        )
    if training_snapshot.role is not EvidenceRole.TRAIN:
        raise NormalizationLeakageError(
            "normalization statistics require a TRAIN role snapshot"
        )

    ordered = tuple(sorted(records, key=_record_key))
    identities = tuple(_record_key(record) for record in ordered)
    if len(identities) != len(set(identities)):
        raise NormalizationError("normalization records must be unique")
    snapshot_cases = {case.case_id: case for case in training_snapshot.cases}
    for record in ordered:
        case = snapshot_cases.get(record.case_id)
        if (
            case is None
            or case.role is not EvidenceRole.TRAIN
            or case.leakage_group_id != record.leakage_group_id
            or not case.success
            or case.sample_trace is None
            or case.sample_trace.fem_result_hash != record.source_hash
        ):
            raise NormalizationLeakageError(
                "every normalization record must resolve to its TRAIN snapshot FEM result"
            )
    unit = ordered[0].unit
    width = len(ordered[0].values)
    if any(record.unit is not unit for record in ordered):
        raise NormalizationError("all normalization records must use one unit")
    dimension = ordered[0].dimension
    if not isinstance(dimension, Dimension):  # narrowed after record validation
        raise NormalizationError("normalization record dimension is unresolved")
    if any(record.dimension is not dimension for record in ordered):
        raise NormalizationError("all normalization records must use one dimension")
    if any(len(record.values) != width for record in ordered):
        raise NormalizationError("all normalization records must have one width")
    if not isinstance(reference_scale, CanonicalQuantity):
        raise NormalizationError("reference scale must be a CanonicalQuantity")
    if reference_scale.unit is not unit or reference_scale.dimension is not dimension:
        raise NormalizationError(
            "reference scale must match the field unit and dimension"
        )
    if reference_scale.value <= 0.0:
        raise NormalizationError("reference scale must be positive")

    rows = tuple(
        tuple(value / reference_scale.value for value in record.values)
        for record in ordered
    )
    count = len(rows)
    mean = tuple(
        math.fsum(row[index] for row in rows) / count for index in range(width)
    )
    deviation = tuple(
        _population_deviation(rows, index, mean[index]) for index in range(width)
    )
    return NormalizationStatistics(
        unit=unit,
        dimension=dimension,
        reference_scale=reference_scale,
        mean=mean,
        standard_deviation=deviation,
        training_provenance=tuple(record.provenance for record in ordered),
        training_snapshot_hash=training_snapshot.content_hash,
    )


def _population_deviation(
    rows: tuple[tuple[float, ...], ...],
    index: int,
    mean: float,
) -> float:
    variance = math.fsum((row[index] - mean) ** 2 for row in rows) / len(rows)
    deviation = math.sqrt(variance)
    return 1.0 if deviation == 0.0 else deviation


def normalize_sample(
    record: NormalizationRecord,
    statistics: NormalizationStatistics,
) -> NormalizedSample:
    """Apply frozen TRAIN statistics without changing source provenance."""

    if not isinstance(record, NormalizationRecord):
        raise NormalizationError("record must be a NormalizationRecord")
    if not isinstance(statistics, NormalizationStatistics):
        raise NormalizationError("statistics must be NormalizationStatistics")
    if record.status is not PublishedCaseStatus.ACTIVE:
        raise NormalizationLeakageError("only ACTIVE records may be normalized")
    if record.unit is not statistics.unit:
        raise NormalizationError("record unit does not match normalization statistics")
    if record.dimension is not statistics.dimension:
        raise NormalizationError(
            "record dimension does not match normalization statistics"
        )
    if len(record.values) != len(statistics.mean):
        raise NormalizationError("record width does not match normalization statistics")
    scaled = tuple(value / statistics.reference_scale.value for value in record.values)
    values = tuple(
        (value - mean) / deviation
        for value, mean, deviation in zip(
            scaled,
            statistics.mean,
            statistics.standard_deviation,
            strict=True,
        )
    )
    return NormalizedSample(
        sample_id=record.sample_id,
        case_id=record.case_id,
        leakage_group_id=record.leakage_group_id,
        source_hash=record.source_hash,
        role=record.role,
        status=record.status,
        source_unit=record.unit,
        source_dimension=record.dimension,
        values=values,
        normalization_hash=statistics.content_hash,
    )


def denormalize_sample(
    sample: NormalizedSample,
    statistics: NormalizationStatistics,
) -> NormalizationRecord:
    """Invert one transformation only with the exact statistics that produced it."""

    if not isinstance(sample, NormalizedSample):
        raise NormalizationError("sample must be a NormalizedSample")
    if not isinstance(statistics, NormalizationStatistics):
        raise NormalizationError("statistics must be NormalizationStatistics")
    if sample.normalization_hash != statistics.content_hash:
        raise NormalizationError("statistics identity does not match normalized sample")
    if sample.source_unit is not statistics.unit:
        raise NormalizationError("sample unit does not match normalization statistics")
    if sample.source_dimension is not statistics.dimension:
        raise NormalizationError(
            "sample dimension does not match normalization statistics"
        )
    if len(sample.values) != len(statistics.mean):
        raise NormalizationError("sample width does not match normalization statistics")
    values = tuple(
        (value * deviation + mean) * statistics.reference_scale.value
        for value, mean, deviation in zip(
            sample.values,
            statistics.mean,
            statistics.standard_deviation,
            strict=True,
        )
    )
    return NormalizationRecord(
        sample_id=sample.sample_id,
        case_id=sample.case_id,
        leakage_group_id=sample.leakage_group_id,
        source_hash=sample.source_hash,
        role=sample.role,
        status=sample.status,
        unit=sample.source_unit,
        values=values,
        dimension=sample.source_dimension,
    )


__all__ = (
    "NormalizationError",
    "NormalizationLeakageError",
    "NormalizationRecord",
    "NormalizationStatistics",
    "NormalizedSample",
    "SampleProvenance",
    "denormalize_sample",
    "fit_train_normalization",
    "normalize_sample",
)
