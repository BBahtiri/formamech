"""Leakage-safe materialization of the registered S13 TRAIN-fit population."""

from __future__ import annotations

from dataclasses import dataclass

from campaigns.dogbone_v1.s11_contingency_data import (
    CaseNormalizationStatistics,
    OperatorCaseTensors,
    fit_case_normalization,
    normalize_case_values,
    prepare_operator_case,
)
from campaigns.dogbone_v1.s11_contingency_source import SelectiveS10Source
from campaigns.dogbone_v1.s13_initialization import SelectedS11Initialization
from campaigns.dogbone_v1.s13_objective import (
    DogboneS13ObjectiveBuilder,
    DogboneS13TrainingCase,
)
from piginot.datasets import CollocationBatch
from piginot.doe import EvidenceRole
from piginot.fem import FemRequest
from piginot.training import TrainingPopulation


@dataclass(frozen=True, slots=True)
class DogboneS13TrainingData:
    """Verified runtime tensors and identities for one S13 TRAIN-fit run."""

    cases: tuple[DogboneS13TrainingCase, ...]
    population: TrainingPopulation
    normalization: CaseNormalizationStatistics
    source_bindings_hash: str
    selected_candidate_id: str
    selected_checkpoint_file_hash: str

    def __post_init__(self) -> None:
        if type(self.cases) is not tuple or not self.cases:
            raise ValueError("cases must be a non-empty tuple")
        if any(type(item) is not DogboneS13TrainingCase for item in self.cases):
            raise TypeError("cases must contain exact DogboneS13TrainingCase values")
        if type(self.population) is not TrainingPopulation:
            raise TypeError("population must be an exact TrainingPopulation")
        if type(self.normalization) is not CaseNormalizationStatistics:
            raise TypeError("normalization must be exact CaseNormalizationStatistics")
        case_ids = tuple(item.tensors.case_id for item in self.cases)
        if case_ids != self.population.sample_ids:
            raise ValueError("materialized cases differ from the training population")
        if any(
            item.tensors.normalization_hash != self.normalization.content_hash
            for item in self.cases
        ):
            raise ValueError("materialized cases differ from the frozen normalization")
        for name in ("source_bindings_hash", "selected_checkpoint_file_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{name} must be a lowercase 256-bit digest")
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(f"{name} must be a lowercase 256-bit digest") from exc
            if value.lower() != value:
                raise ValueError(f"{name} must be a lowercase 256-bit digest")
        if (
            not isinstance(self.selected_candidate_id, str)
            or not self.selected_candidate_id
        ):
            raise ValueError("selected_candidate_id must be non-empty text")

    def objective_builder(self) -> DogboneS13ObjectiveBuilder:
        """Create the exact strong-HYBRID objective adapter for this population."""

        return DogboneS13ObjectiveBuilder(self.cases)


def materialize_dogbone_s13_training_data(
    source: SelectiveS10Source,
    initialization: SelectedS11Initialization,
) -> DogboneS13TrainingData:
    """Resolve only registered TRAIN-fit FEM/collocation material for S13."""

    if type(source) is not SelectiveS10Source:
        raise TypeError("source must be an exact verified SelectiveS10Source")
    if type(initialization) is not SelectedS11Initialization:
        raise TypeError("initialization must be exact SelectedS11Initialization")
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels must remain closed during S13 training")
    if source.bindings.content_hash != initialization.source_bindings_hash:
        raise ValueError("S10 source differs from the selected S11 checkpoint")
    normalization = initialization.normalization
    if normalization.fit_case_ids != source.fit_case_ids:
        raise ValueError("S11 normalization differs from the registered TRAIN fit set")

    prepared: list[OperatorCaseTensors] = []
    requests: list[FemRequest] = []
    collocations: list[CollocationBatch] = []
    for verified, collocation in source.iter_fit_with_collocation():
        prepared.append(
            prepare_operator_case(verified.case, verified.request, verified.result)
        )
        requests.append(verified.request)
        collocations.append(collocation)
    prepared_cases = tuple(prepared)
    if tuple(item.case_id for item in prepared_cases) != source.fit_case_ids:
        raise ValueError("materialized TRAIN fit case order or identity changed")

    recomputed = fit_case_normalization(
        prepared_cases,
        fit_case_ids=source.fit_case_ids,
        training_snapshot_hash=source.train_snapshot.content_hash,
    )
    if recomputed != normalization:
        raise ValueError("recomputed TRAIN normalization differs from S11 evidence")
    cases = tuple(
        DogboneS13TrainingCase(
            tensors=normalize_case_values(prepared_case, normalization),
            request=request,
            collocation=collocation,
        )
        for prepared_case, request, collocation in zip(
            prepared_cases,
            requests,
            collocations,
            strict=True,
        )
    )
    population = TrainingPopulation(
        dataset_snapshot_hash=normalization.training_snapshot_hash,
        public_manifest_hash=source.bindings.public_case_manifest_hash,
        role=EvidenceRole.TRAIN,
        sample_ids=source.fit_case_ids,
    )
    return DogboneS13TrainingData(
        cases=cases,
        population=population,
        normalization=normalization,
        source_bindings_hash=source.bindings.content_hash,
        selected_candidate_id=initialization.candidate_id,
        selected_checkpoint_file_hash=initialization.checkpoint_file_blake2b256,
    )


__all__ = (
    "DogboneS13TrainingData",
    "materialize_dogbone_s13_training_data",
)
