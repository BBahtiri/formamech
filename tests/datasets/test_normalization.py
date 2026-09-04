from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from piginot.contracts import CanonicalQuantity, Dimension, Unit
from piginot.datasets.normalization import (
    NormalizationError,
    NormalizationLeakageError,
    NormalizationRecord,
    NormalizationStatistics,
    NormalizedSample,
    SampleProvenance,
    denormalize_sample,
    normalize_sample,
)
from piginot.datasets.normalization import (
    fit_train_normalization as _fit_train_normalization,
)
from piginot.datasets.snapshots import (
    DatasetCaseRecord,
    RoleDatasetSnapshot,
    SampleTrace,
)
from piginot.doe import EvidenceRole, PublishedCaseStatus


def _record(
    sample_id: str,
    values: tuple[float, ...],
    *,
    role: EvidenceRole = EvidenceRole.TRAIN,
    status: PublishedCaseStatus = PublishedCaseStatus.ACTIVE,
    unit: Unit = Unit.MPA,
    dimension: Dimension | None = None,
) -> NormalizationRecord:
    suffix = int(sample_id.rsplit("-", 1)[-1])
    return NormalizationRecord(
        sample_id=sample_id,
        case_id=f"case-{suffix}",
        leakage_group_id=f"group-{suffix}",
        source_hash=f"{suffix + 1:064x}",
        role=role,
        status=status,
        unit=unit,
        values=values,
        dimension=dimension,
    )


def _stress_scale() -> CanonicalQuantity:
    return CanonicalQuantity(100.0, Unit.MPA, Dimension.STRESS)


def _training_snapshot(*records: NormalizationRecord) -> RoleDatasetSnapshot:
    cases = {
        record.case_id: DatasetCaseRecord(
            case_id=record.case_id,
            leakage_group_id=record.leakage_group_id,
            role=EvidenceRole.TRAIN,
            primitive_coordinates={"width": 1.0},
            derived_coordinates={"ratio": 1.0},
            strata={"width": 0},
            request_recipe_hash="a" * 64,
            request_content_hash="b" * 64,
            mesh_id=f"mesh-{record.case_id}",
            mesh_content_hash="c" * 64,
            success=True,
            sample_trace=SampleTrace(fem_result_hash=record.source_hash),
            store_content_hash="d" * 64,
            failure_code=None,
            message="succeeded",
        )
        for record in records
        if record.role is EvidenceRole.TRAIN
        and record.status is PublishedCaseStatus.ACTIVE
    }
    if not cases:
        fallback = _record("stress-99", (1.0,))
        return _training_snapshot(fallback)
    return RoleDatasetSnapshot(
        campaign_hash="e" * 64,
        public_case_manifest_hash="f" * 64,
        request_plan_hash="1" * 64,
        role=EvidenceRole.TRAIN,
        cases=tuple(cases.values()),
    )


def fit_train_normalization(
    records: Any,
    *,
    reference_scale: Any,
) -> NormalizationStatistics:
    valid_records = (
        tuple(record for record in records if isinstance(record, NormalizationRecord))
        if isinstance(records, tuple)
        else ()
    )
    return _fit_train_normalization(
        records,
        reference_scale=reference_scale,
        training_snapshot=_training_snapshot(*valid_records),
    )


def test_fit_requires_every_source_to_resolve_in_the_training_snapshot() -> None:
    record = _record("stress-0", (100.0,))
    snapshot = _training_snapshot(record)
    forged = replace(record, source_hash="9" * 64)

    with pytest.raises(NormalizationLeakageError, match="snapshot"):
        _fit_train_normalization(
            (forged,),
            reference_scale=_stress_scale(),
            training_snapshot=snapshot,
        )


def test_fit_requires_an_exact_train_role_snapshot() -> None:
    record = _record("stress-0", (100.0,))
    snapshot = _training_snapshot(record)
    development_case = replace(
        snapshot.cases[0],
        role=EvidenceRole.DEVELOPMENT,
    )
    development_snapshot = replace(
        snapshot,
        role=EvidenceRole.DEVELOPMENT,
        cases=(development_case,),
    )

    with pytest.raises(NormalizationError, match="exact RoleDatasetSnapshot"):
        _fit_train_normalization(
            (record,),
            reference_scale=_stress_scale(),
            training_snapshot=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(NormalizationLeakageError, match="TRAIN role snapshot"):
        _fit_train_normalization(
            (record,),
            reference_scale=_stress_scale(),
            training_snapshot=development_snapshot,
        )


def test_fit_normalize_and_denormalize_are_reversible_with_provenance() -> None:
    records = (
        _record("stress-0", (100.0, 300.0)),
        _record("stress-1", (300.0, 700.0)),
    )

    statistics = fit_train_normalization(records, reference_scale=_stress_scale())
    transformed = normalize_sample(records[0], statistics)
    restored = denormalize_sample(transformed, statistics)

    assert statistics.unit is Unit.MPA
    assert statistics.reference_scale == _stress_scale()
    assert (
        statistics.training_snapshot_hash == _training_snapshot(*records).content_hash
    )
    assert statistics.mean == pytest.approx((2.0, 5.0))
    assert statistics.standard_deviation == pytest.approx((1.0, 2.0))
    assert transformed.values == pytest.approx((-1.0, -1.0))
    assert transformed.sample_id == records[0].sample_id
    assert transformed.case_id == records[0].case_id
    assert transformed.leakage_group_id == records[0].leakage_group_id
    assert transformed.source_hash == records[0].source_hash
    assert transformed.role is EvidenceRole.TRAIN
    assert transformed.source_unit is Unit.MPA
    assert transformed.source_dimension is Dimension.STRESS
    assert transformed.provenance == records[0].provenance
    assert all(
        item.role is EvidenceRole.TRAIN and item.status is PublishedCaseStatus.ACTIVE
        for item in statistics.training_provenance
    )
    assert restored == records[0]


@pytest.mark.parametrize(
    "forbidden_role",
    [
        EvidenceRole.PILOT,
        EvidenceRole.DEVELOPMENT,
        EvidenceRole.LEGACY,
    ],
)
def test_fit_rejects_normalize_on_all_splits_mutant(
    forbidden_role: EvidenceRole,
) -> None:
    records = (
        _record("stress-0", (100.0,)),
        _record("stress-1", (1_000_000.0,), role=forbidden_role),
    )

    with pytest.raises(NormalizationLeakageError, match="NORMALIZE_ON_ALL_SPLITS"):
        fit_train_normalization(records, reference_scale=_stress_scale())


def test_public_normalization_contract_rejects_holdout_at_construction() -> None:
    with pytest.raises(NormalizationLeakageError, match="HOLDOUT"):
        _record(
            "stress-0",
            (100.0,),
            role=EvidenceRole.HOLDOUT,
        )


def test_statistics_contract_rejects_nontrain_provenance_directly() -> None:
    train = _record("stress-0", (100.0,))
    statistics = fit_train_normalization((train,), reference_scale=_stress_scale())
    tainted = replace(
        statistics.training_provenance[0],
        role=EvidenceRole.DEVELOPMENT,
    )

    with pytest.raises(NormalizationLeakageError, match="NORMALIZE_ON_ALL_SPLITS"):
        replace(statistics, training_provenance=(tainted,))


def test_fit_rejects_reserve_records() -> None:
    reserve = _record(
        "stress-0",
        (100.0,),
        status=PublishedCaseStatus.RESERVE,
    )

    with pytest.raises(NormalizationLeakageError, match="ACTIVE TRAIN"):
        fit_train_normalization((reserve,), reference_scale=_stress_scale())


def test_input_order_does_not_change_statistics_or_identity() -> None:
    records = (
        _record("stress-0", (1.0e16, 10.0)),
        _record("stress-1", (1.0, 20.0)),
        _record("stress-2", (-1.0e16, 30.0)),
    )

    forward = fit_train_normalization(records, reference_scale=_stress_scale())
    reverse = fit_train_normalization(
        tuple(reversed(records)),
        reference_scale=_stress_scale(),
    )

    assert forward == reverse
    assert forward.content_hash == reverse.content_hash
    assert tuple(item.sample_id for item in forward.training_provenance) == (
        "stress-0",
        "stress-1",
        "stress-2",
    )


def test_zero_variance_component_uses_unit_scale_and_remains_reversible() -> None:
    record = _record("stress-0", (250.0, 250.0))

    statistics = fit_train_normalization(
        (record,),
        reference_scale=_stress_scale(),
    )
    transformed = normalize_sample(record, statistics)

    assert statistics.standard_deviation == (1.0, 1.0)
    assert transformed.values == (0.0, 0.0)
    assert denormalize_sample(transformed, statistics) == record


def test_fit_rejects_unit_or_dimension_mismatch() -> None:
    length_record = _record("stress-0", (10.0,), unit=Unit.MM)

    with pytest.raises(ValueError, match="reference scale"):
        fit_train_normalization(
            (length_record,),
            reference_scale=_stress_scale(),
        )


def test_ambiguous_canonical_unit_retains_declared_energy_dimension() -> None:
    energy = _record(
        "stress-0",
        (25.0,),
        unit=Unit.N_MM,
        dimension=Dimension.ENERGY,
    )
    reference = CanonicalQuantity(10.0, Unit.N_MM, Dimension.ENERGY)

    statistics = fit_train_normalization((energy,), reference_scale=reference)
    transformed = normalize_sample(energy, statistics)

    assert statistics.dimension is Dimension.ENERGY
    assert transformed.source_dimension is Dimension.ENERGY
    assert denormalize_sample(transformed, statistics) == energy


def test_application_preserves_active_development_sample_provenance() -> None:
    train = _record("stress-0", (100.0,))
    development = _record(
        "stress-1",
        (200.0,),
        role=EvidenceRole.DEVELOPMENT,
    )
    statistics = fit_train_normalization((train,), reference_scale=_stress_scale())

    transformed = normalize_sample(development, statistics)

    assert transformed.role is EvidenceRole.DEVELOPMENT
    assert transformed.source_hash == development.source_hash
    assert denormalize_sample(transformed, statistics) == development


def test_application_rejects_reserve_and_foreign_statistics() -> None:
    train = _record("stress-0", (100.0,))
    statistics = fit_train_normalization((train,), reference_scale=_stress_scale())
    reserve = replace(train, status=PublishedCaseStatus.RESERVE)

    with pytest.raises(NormalizationLeakageError, match="ACTIVE"):
        normalize_sample(reserve, statistics)

    transformed = normalize_sample(train, statistics)
    foreign = replace(statistics, reference_scale=CanonicalQuantity(50.0, Unit.MPA))
    with pytest.raises(ValueError, match="statistics identity"):
        denormalize_sample(transformed, foreign)


@pytest.mark.parametrize("records", [(), [], (object(),)])
def test_fit_requires_nonempty_record_tuple(records: object) -> None:
    with pytest.raises(NormalizationError, match="records"):
        fit_train_normalization(records, reference_scale=_stress_scale())  # type: ignore[arg-type]


def test_fit_rejects_duplicate_provenance_mixed_units_and_widths() -> None:
    first = _record("stress-0", (100.0,))

    with pytest.raises(NormalizationError, match="unique"):
        fit_train_normalization((first, first), reference_scale=_stress_scale())
    with pytest.raises(NormalizationError, match="one unit"):
        fit_train_normalization(
            (first, _record("stress-1", (10.0,), unit=Unit.MM)),
            reference_scale=_stress_scale(),
        )
    with pytest.raises(NormalizationError, match="one width"):
        fit_train_normalization(
            (first, _record("stress-1", (10.0, 20.0))),
            reference_scale=_stress_scale(),
        )


def test_application_rejects_incompatible_unit_and_width() -> None:
    train = _record("stress-0", (100.0,))
    statistics = fit_train_normalization((train,), reference_scale=_stress_scale())

    with pytest.raises(NormalizationError, match="unit"):
        normalize_sample(_record("stress-1", (10.0,), unit=Unit.MM), statistics)
    with pytest.raises(NormalizationError, match="width"):
        normalize_sample(_record("stress-1", (10.0, 20.0)), statistics)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"sample_id": ""}, "non-empty text"),
        ({"source_hash": "A" * 64}, "lowercase BLAKE2b-256"),
        ({"values": ()}, "non-empty tuple"),
        ({"values": (True,)}, "real numbers"),
        ({"values": (float("nan"),)}, "finite numbers"),
        ({"unit": Unit.CM}, "canonical Unit"),
        ({"dimension": Dimension.LENGTH}, "supported by the field unit"),
    ],
)
def test_record_contract_rejects_malformed_boundaries(
    changes: dict[str, object],
    message: str,
) -> None:
    record = _record("stress-0", (100.0,))

    with pytest.raises(NormalizationError, match=message):
        replace(record, **cast(Any, changes))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dimension": "stress"}, "dimension must be"),
        ({"reference_scale": object()}, "CanonicalQuantity"),
        (
            {"reference_scale": CanonicalQuantity(-1.0, Unit.MPA)},
            "reference scale must be positive",
        ),
        ({"unit": Unit.MM}, "reference scale unit"),
        ({"dimension": Dimension.LENGTH}, "reference scale dimension"),
        ({"standard_deviation": (1.0, 1.0)}, "widths must match"),
        ({"standard_deviation": (0.0,)}, "standard deviations must be positive"),
        ({"training_provenance": ()}, "must contain SampleProvenance"),
        ({"training_provenance": (object(),)}, "must contain SampleProvenance"),
    ],
)
def test_statistics_contract_rejects_malformed_boundaries(
    changes: dict[str, object],
    message: str,
) -> None:
    statistics = fit_train_normalization(
        (_record("stress-0", (100.0,)),),
        reference_scale=_stress_scale(),
    )

    with pytest.raises(NormalizationError, match=message):
        replace(statistics, **cast(Any, changes))


def test_statistics_contract_requires_ordered_unique_provenance() -> None:
    statistics = fit_train_normalization(
        (
            _record("stress-0", (100.0,)),
            _record("stress-1", (200.0,)),
        ),
        reference_scale=_stress_scale(),
    )

    with pytest.raises(NormalizationError, match="canonically ordered"):
        replace(
            statistics,
            training_provenance=tuple(reversed(statistics.training_provenance)),
        )
    with pytest.raises(NormalizationError, match="unique"):
        replace(
            statistics,
            training_provenance=(
                statistics.training_provenance[0],
                statistics.training_provenance[0],
            ),
        )


def test_fit_rejects_reference_type_value_and_mixed_dimension() -> None:
    moment = _record(
        "stress-0",
        (10.0,),
        unit=Unit.N_MM,
        dimension=Dimension.MOMENT,
    )
    energy = _record(
        "stress-1",
        (20.0,),
        unit=Unit.N_MM,
        dimension=Dimension.ENERGY,
    )

    with pytest.raises(NormalizationError, match="one dimension"):
        fit_train_normalization(
            (moment, energy),
            reference_scale=CanonicalQuantity(1.0, Unit.N_MM, Dimension.MOMENT),
        )
    with pytest.raises(NormalizationError, match="CanonicalQuantity"):
        fit_train_normalization((moment,), reference_scale=object())  # type: ignore[arg-type]
    with pytest.raises(NormalizationError, match="positive"):
        fit_train_normalization(
            (moment,),
            reference_scale=CanonicalQuantity(
                -1.0,
                Unit.N_MM,
                Dimension.MOMENT,
            ),
        )


def test_transform_functions_reject_wrong_contract_types() -> None:
    record = _record("stress-0", (100.0,))
    statistics = fit_train_normalization((record,), reference_scale=_stress_scale())
    transformed = normalize_sample(record, statistics)

    with pytest.raises(NormalizationError, match="record must be"):
        normalize_sample(object(), statistics)  # type: ignore[arg-type]
    with pytest.raises(NormalizationError, match="statistics must be"):
        normalize_sample(record, object())  # type: ignore[arg-type]
    with pytest.raises(NormalizationError, match="sample must be"):
        denormalize_sample(object(), statistics)  # type: ignore[arg-type]
    with pytest.raises(NormalizationError, match="statistics must be"):
        denormalize_sample(transformed, object())  # type: ignore[arg-type]


def test_transform_functions_reject_dimension_and_inverse_shape_mismatch() -> None:
    energy = _record(
        "stress-0",
        (25.0,),
        unit=Unit.N_MM,
        dimension=Dimension.ENERGY,
    )
    statistics = fit_train_normalization(
        (energy,),
        reference_scale=CanonicalQuantity(10.0, Unit.N_MM, Dimension.ENERGY),
    )
    transformed = normalize_sample(energy, statistics)
    moment = replace(energy, dimension=Dimension.MOMENT)

    with pytest.raises(NormalizationError, match="record dimension"):
        normalize_sample(moment, statistics)
    with pytest.raises(NormalizationError, match="sample unit"):
        denormalize_sample(
            replace(
                transformed,
                source_unit=Unit.MM,
                source_dimension=Dimension.LENGTH,
            ),
            statistics,
        )
    with pytest.raises(NormalizationError, match="sample dimension"):
        denormalize_sample(
            replace(transformed, source_dimension=Dimension.MOMENT),
            statistics,
        )
    with pytest.raises(NormalizationError, match="sample width"):
        denormalize_sample(replace(transformed, values=(0.0, 1.0)), statistics)


def test_direct_contract_constructors_reject_invalid_provenance_and_sample() -> None:
    with pytest.raises(NormalizationError, match="non-empty text"):
        SampleProvenance(
            "",
            "case",
            "group",
            "1" * 64,
            EvidenceRole.TRAIN,
            PublishedCaseStatus.ACTIVE,
        )

    with pytest.raises(NormalizationError, match="source_dimension"):
        NormalizedSample(
            sample_id="sample",
            case_id="case",
            leakage_group_id="group",
            source_hash="1" * 64,
            role=EvidenceRole.TRAIN,
            status=PublishedCaseStatus.ACTIVE,
            source_unit=Unit.MPA,
            source_dimension=Dimension.LENGTH,
            values=(0.0,),
            normalization_hash="2" * 64,
        )
