from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from campaigns.dogbone_v1 import s13_data
from campaigns.dogbone_v1.s11_contingency import (
    DERIVED_PARAMETER_NAMES,
    RAW_PARAMETER_NAMES,
)
from campaigns.dogbone_v1.s11_contingency_data import CaseNormalizationStatistics
from campaigns.dogbone_v1.s11_contingency_source import SelectiveS10Source
from campaigns.dogbone_v1.s13_data import (
    DogboneS13TrainingData,
    materialize_dogbone_s13_training_case,
    materialize_dogbone_s13_training_data,
)
from campaigns.dogbone_v1.s13_initialization import SelectedS11Initialization
from piginot.doe import EvidenceRole
from piginot.training import TrainingPopulation
from tests.campaigns.test_dogbone_s13_objective import _training_case


@pytest.fixture(scope="module")
def valid_training_data() -> DogboneS13TrainingData:
    original = _training_case()
    normalization = CaseNormalizationStatistics(
        parameter_names=(*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES),
        mean=(0.0,) * 11,
        standard_deviation=(1.0,) * 11,
        fit_case_ids=(original.tensors.case_id,),
        training_snapshot_hash="a" * 64,
    )
    training_case = replace(
        original,
        tensors=replace(
            original.tensors,
            normalization_hash=normalization.content_hash,
        ),
    )
    population = TrainingPopulation(
        dataset_snapshot_hash=normalization.training_snapshot_hash,
        public_manifest_hash="b" * 64,
        role=EvidenceRole.TRAIN,
        sample_ids=(training_case.tensors.case_id,),
    )
    return DogboneS13TrainingData(
        cases=(training_case,),
        population=population,
        normalization=normalization,
        source_bindings_hash="c" * 64,
        selected_candidate_id="A-s11-baseline",
        selected_checkpoint_file_hash="d" * 64,
    )


def _unopened_source(tmp_path: Path) -> SelectiveS10Source:
    source = object.__new__(SelectiveS10Source)
    source._development_opened = False
    source._development_access_record = tmp_path / "development-not-opened.json"
    source._bindings = SimpleNamespace(content_hash="c" * 64)  # type: ignore[assignment]
    return source


def _initialization(data: DogboneS13TrainingData) -> SelectedS11Initialization:
    initialization = object.__new__(SelectedS11Initialization)
    object.__setattr__(
        initialization, "source_bindings_hash", data.source_bindings_hash
    )
    object.__setattr__(initialization, "normalization", data.normalization)
    return initialization


def test_training_data_binds_cases_population_and_normalization() -> None:
    original = _training_case()
    normalization = CaseNormalizationStatistics(
        parameter_names=(*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES),
        mean=(0.0,) * 11,
        standard_deviation=(1.0,) * 11,
        fit_case_ids=(original.tensors.case_id,),
        training_snapshot_hash="a" * 64,
    )
    training_case = replace(
        original,
        tensors=replace(
            original.tensors,
            normalization_hash=normalization.content_hash,
        ),
    )
    population = TrainingPopulation(
        dataset_snapshot_hash=normalization.training_snapshot_hash,
        public_manifest_hash="b" * 64,
        role=EvidenceRole.TRAIN,
        sample_ids=(training_case.tensors.case_id,),
    )

    data = DogboneS13TrainingData(
        cases=(training_case,),
        population=population,
        normalization=normalization,
        source_bindings_hash="c" * 64,
        selected_candidate_id="A-s11-baseline",
        selected_checkpoint_file_hash="d" * 64,
    )

    assert data.objective_builder().sample_ids == population.sample_ids


def test_training_data_refuses_population_drift() -> None:
    original = _training_case()
    normalization = CaseNormalizationStatistics(
        parameter_names=(*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES),
        mean=(0.0,) * 11,
        standard_deviation=(1.0,) * 11,
        fit_case_ids=(original.tensors.case_id,),
        training_snapshot_hash="a" * 64,
    )
    training_case = replace(
        original,
        tensors=replace(
            original.tensors,
            normalization_hash=normalization.content_hash,
        ),
    )
    wrong_population = TrainingPopulation(
        dataset_snapshot_hash=normalization.training_snapshot_hash,
        public_manifest_hash="b" * 64,
        role=EvidenceRole.TRAIN,
        sample_ids=("train-other",),
    )

    with pytest.raises(ValueError, match="population"):
        DogboneS13TrainingData(
            cases=(training_case,),
            population=wrong_population,
            normalization=normalization,
            source_bindings_hash="c" * 64,
            selected_candidate_id="A-s11-baseline",
            selected_checkpoint_file_hash="d" * 64,
        )


def test_production_materializer_requires_verified_exact_inputs() -> None:
    with pytest.raises(TypeError, match="SelectiveS10Source"):
        materialize_dogbone_s13_training_data(object(), object())  # type: ignore[arg-type]


@pytest.mark.parametrize("cases", [(), [], ("not-a-TRAIN-case",)])
def test_training_data_rejects_missing_or_unverified_cases(
    valid_training_data: DogboneS13TrainingData,
    cases: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="cases"):
        replace(valid_training_data, cases=cases)


@pytest.mark.parametrize("field", ["population", "normalization"])
def test_training_data_requires_exact_population_and_normalization(
    valid_training_data: DogboneS13TrainingData,
    field: str,
) -> None:
    with pytest.raises(TypeError, match=field):
        replace(valid_training_data, **{field: object()})


def test_training_data_rejects_stale_case_normalization(
    valid_training_data: DogboneS13TrainingData,
) -> None:
    case = valid_training_data.cases[0]
    stale = replace(
        case,
        tensors=replace(case.tensors, normalization_hash="e" * 64),
    )

    with pytest.raises(ValueError, match="frozen normalization"):
        replace(valid_training_data, cases=(stale,))


@pytest.mark.parametrize(
    ("field", "digest"),
    [
        ("source_bindings_hash", "a" * 63),
        ("source_bindings_hash", "g" * 64),
        ("source_bindings_hash", "A" * 64),
        ("selected_checkpoint_file_hash", "a" * 63),
        ("selected_checkpoint_file_hash", "g" * 64),
        ("selected_checkpoint_file_hash", "A" * 64),
    ],
)
def test_training_data_requires_canonical_provenance_digests(
    valid_training_data: DogboneS13TrainingData,
    field: str,
    digest: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        replace(valid_training_data, **{field: digest})


def test_training_data_requires_selected_candidate_identity(
    valid_training_data: DogboneS13TrainingData,
) -> None:
    with pytest.raises(ValueError, match="selected_candidate_id"):
        replace(valid_training_data, selected_candidate_id="")


def test_materializer_rejects_unverified_initialization(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="SelectedS11Initialization"):
        materialize_dogbone_s13_training_data(
            _unopened_source(tmp_path),
            object(),  # type: ignore[arg-type]
        )


def test_materializer_refuses_previously_opened_development(
    tmp_path: Path,
    valid_training_data: DogboneS13TrainingData,
) -> None:
    source = _unopened_source(tmp_path)
    source._development_opened = True

    with pytest.raises(RuntimeError, match="DEVELOPMENT labels must remain closed"):
        materialize_dogbone_s13_training_data(
            source, _initialization(valid_training_data)
        )


def test_materializer_refuses_source_checkpoint_binding_drift(
    tmp_path: Path,
    valid_training_data: DogboneS13TrainingData,
) -> None:
    initialization = _initialization(valid_training_data)
    object.__setattr__(initialization, "source_bindings_hash", "e" * 64)

    with pytest.raises(ValueError, match="S10 source differs"):
        materialize_dogbone_s13_training_data(
            _unopened_source(tmp_path), initialization
        )


def test_materializer_refuses_normalization_fit_population_drift(
    tmp_path: Path,
    valid_training_data: DogboneS13TrainingData,
) -> None:
    source = _unopened_source(tmp_path)
    source._protocol = SimpleNamespace(fit_case_ids=("other-TRAIN-case",))  # type: ignore[assignment]

    with pytest.raises(ValueError, match="registered TRAIN fit set"):
        materialize_dogbone_s13_training_data(
            source, _initialization(valid_training_data)
        )


def test_single_case_materializer_uses_frozen_normalization_without_full_fit(
    tmp_path: Path,
    valid_training_data: DogboneS13TrainingData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = valid_training_data.cases[0]
    case_id = expected.tensors.case_id
    source = _unopened_source(tmp_path)
    source._protocol = SimpleNamespace(  # type: ignore[assignment]
        fit_case_ids=(case_id,),
    )
    prepared = object()
    calls: list[tuple[object, ...]] = []
    verified = SimpleNamespace(
        case=object(),
        request=expected.request,
        result=object(),
    )

    def load_one(
        _source: SelectiveS10Source,
        selected_id: str,
    ) -> tuple[object, object]:
        calls.append(("load", selected_id))
        return verified, expected.collocation

    def prepare(case: object, request: object, result: object) -> object:
        calls.append(("prepare", case, request, result))
        return prepared

    def normalize(case: object, normalization: object) -> object:
        calls.append(("normalize", case, normalization))
        return expected.tensors

    monkeypatch.setattr(SelectiveS10Source, "load_fit_with_collocation", load_one)
    monkeypatch.setattr(s13_data, "prepare_operator_case", prepare)
    monkeypatch.setattr(s13_data, "normalize_case_values", normalize)

    observed = materialize_dogbone_s13_training_case(
        source,
        _initialization(valid_training_data),
        case_id,
    )

    assert observed == expected
    assert calls == [
        ("load", case_id),
        ("prepare", verified.case, expected.request, verified.result),
        ("normalize", prepared, valid_training_data.normalization),
    ]
    assert source.development_opened is False


def test_single_case_materializer_requires_exact_inputs(
    tmp_path: Path,
    valid_training_data: DogboneS13TrainingData,
) -> None:
    source = _unopened_source(tmp_path)
    source._protocol = SimpleNamespace(  # type: ignore[assignment]
        fit_case_ids=valid_training_data.normalization.fit_case_ids,
    )
    initialization = _initialization(valid_training_data)
    case_id = valid_training_data.cases[0].tensors.case_id

    with pytest.raises(TypeError, match="SelectiveS10Source"):
        materialize_dogbone_s13_training_case(  # type: ignore[arg-type]
            object(), initialization, case_id
        )
    with pytest.raises(TypeError, match="SelectedS11Initialization"):
        materialize_dogbone_s13_training_case(  # type: ignore[arg-type]
            source, object(), case_id
        )
    with pytest.raises(TypeError, match="case_id"):
        materialize_dogbone_s13_training_case(  # type: ignore[arg-type]
            source, initialization, 1
        )
    with pytest.raises(ValueError, match="non-empty"):
        materialize_dogbone_s13_training_case(source, initialization, "")


def test_single_case_materializer_refuses_open_or_drifted_sources(
    tmp_path: Path,
    valid_training_data: DogboneS13TrainingData,
) -> None:
    case_id = valid_training_data.cases[0].tensors.case_id
    initialization = _initialization(valid_training_data)
    opened = _unopened_source(tmp_path)
    opened._protocol = SimpleNamespace(  # type: ignore[assignment]
        fit_case_ids=valid_training_data.normalization.fit_case_ids,
    )
    opened._development_opened = True
    with pytest.raises(RuntimeError, match="DEVELOPMENT labels must remain closed"):
        materialize_dogbone_s13_training_case(opened, initialization, case_id)

    binding_drift = _unopened_source(tmp_path)
    binding_drift._protocol = SimpleNamespace(  # type: ignore[assignment]
        fit_case_ids=valid_training_data.normalization.fit_case_ids,
    )
    binding_drift._bindings = SimpleNamespace(  # type: ignore[assignment]
        content_hash="e" * 64,
    )
    with pytest.raises(ValueError, match="S10 source differs"):
        materialize_dogbone_s13_training_case(
            binding_drift,
            initialization,
            case_id,
        )

    fit_drift = _unopened_source(tmp_path)
    fit_drift._protocol = SimpleNamespace(  # type: ignore[assignment]
        fit_case_ids=("different-fit-case",),
    )
    with pytest.raises(ValueError, match="registered TRAIN fit set"):
        materialize_dogbone_s13_training_case(fit_drift, initialization, case_id)


def test_single_case_materializer_rejects_non_fit_id_without_loading(
    tmp_path: Path,
    valid_training_data: DogboneS13TrainingData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _unopened_source(tmp_path)
    source._protocol = SimpleNamespace(  # type: ignore[assignment]
        fit_case_ids=valid_training_data.normalization.fit_case_ids,
    )
    loaded = False

    def load_one(*args: object) -> tuple[object, object]:
        nonlocal loaded
        loaded = True
        return object(), object()

    monkeypatch.setattr(SelectiveS10Source, "load_fit_with_collocation", load_one)

    with pytest.raises(ValueError, match="registered TRAIN fit"):
        materialize_dogbone_s13_training_case(
            source,
            _initialization(valid_training_data),
            "screen-only",
        )
    assert loaded is False


def test_single_case_materializer_detects_development_opened_during_load(
    tmp_path: Path,
    valid_training_data: DogboneS13TrainingData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _unopened_source(tmp_path)
    source._protocol = SimpleNamespace(  # type: ignore[assignment]
        fit_case_ids=valid_training_data.normalization.fit_case_ids,
    )

    def leak(
        current: SelectiveS10Source,
        case_id: str,
    ) -> tuple[object, object]:
        current._development_opened = True
        return object(), object()

    monkeypatch.setattr(SelectiveS10Source, "load_fit_with_collocation", leak)

    with pytest.raises(RuntimeError, match="DEVELOPMENT labels must remain closed"):
        materialize_dogbone_s13_training_case(
            source,
            _initialization(valid_training_data),
            valid_training_data.cases[0].tensors.case_id,
        )
