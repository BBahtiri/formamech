from __future__ import annotations

from dataclasses import replace

import pytest

from campaigns.dogbone_v1.s11_contingency import (
    DERIVED_PARAMETER_NAMES,
    RAW_PARAMETER_NAMES,
)
from campaigns.dogbone_v1.s11_contingency_data import CaseNormalizationStatistics
from campaigns.dogbone_v1.s13_data import (
    DogboneS13TrainingData,
    materialize_dogbone_s13_training_data,
)
from piginot.doe import EvidenceRole
from piginot.training import TrainingPopulation
from tests.campaigns.test_dogbone_s13_objective import _training_case


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
