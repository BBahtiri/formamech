"""TRAIN-only S13 selection metrics over the registered DogBone populations."""

from __future__ import annotations

import math
from typing import cast

import torch

from campaigns.dogbone_v1.s10_evidence import DOGBONE_S10_DATASET_POLICY
from campaigns.dogbone_v1.s11_contingency_data import (
    normalize_case_values,
    prepare_operator_case,
)
from campaigns.dogbone_v1.s11_contingency_source import SelectiveS10Source
from campaigns.dogbone_v1.s11_contingency_training import (
    calculate_architecture_screen_metrics,
)
from campaigns.dogbone_v1.s13_data import DogboneS13TrainingData
from campaigns.dogbone_v1.s13_objective import (
    _COORDINATE_CONVERSION,
    DogboneS13ObjectiveBuilder,
    DogboneS13TrainingCase,
    _scales,
    _validated_operator,
)
from campaigns.dogbone_v1.s13_protocol import DogboneS13Protocol
from piginot.datasets import SemanticBCClass
from piginot.doe import EvidenceRole
from piginot.objectives import (
    ObjectiveMode,
    ObjectiveRegistry,
    displacement_gradient,
)
from piginot.operators import OperatorOutput, PiGinotOperator, module_digest
from piginot.physics._validation import DETERMINANT_MARGIN
from piginot.training import SelectionMetrics, TrialConfig
from piginot.training.engine import _scaled_registry


def evaluate_dogbone_s13_train_screen(
    *,
    protocol: DogboneS13Protocol,
    source: SelectiveS10Source,
    training_data: DogboneS13TrainingData,
    model: PiGinotOperator,
    trial: TrialConfig,
) -> SelectionMetrics:
    """Evaluate one registered checkpoint without opening DEVELOPMENT/HOLDOUT.

    Accuracy and fixed-weight HYBRID loss use 64 TRAIN-screen cases. The
    orientation count covers 128 replayed interior points in each of all 256
    TRAIN cases, including the 192 fit cases held in ``training_data``.
    """

    if type(source) is not SelectiveS10Source:
        raise TypeError("source must be an exact verified SelectiveS10Source")
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels must remain closed during S13 screening")
    if type(protocol) is not DogboneS13Protocol:
        raise TypeError("protocol must be an exact DogboneS13Protocol")
    if type(training_data) is not DogboneS13TrainingData:
        raise TypeError("training_data must be exact DogboneS13TrainingData")
    if type(trial) is not TrialConfig:
        raise TypeError("trial must be an exact TrialConfig")
    protocol.search_space.authorize(trial)
    operator = _validated_operator(model)
    if module_digest(operator) != protocol.architecture_module_digest:
        raise ValueError("model architecture differs from the S13 protocol")
    if (
        source.bindings.content_hash != protocol.source_bindings_hash
        or training_data.source_bindings_hash != protocol.source_bindings_hash
        or source.fit_case_ids != protocol.fit_case_ids
        or source.train_screen_case_ids != protocol.screen_case_ids
    ):
        raise ValueError("S10 source or TRAIN population differs from S13 protocol")
    if (
        training_data.population.sample_ids != protocol.fit_case_ids
        or training_data.population.dataset_snapshot_hash
        != protocol.dataset_snapshot_hash
        or training_data.population.public_manifest_hash
        != protocol.public_manifest_hash
        or training_data.normalization.content_hash != protocol.normalization_hash
        or training_data.selected_candidate_id != protocol.selected_candidate_id
        or training_data.selected_checkpoint_file_hash
        != protocol.initial_checkpoint_file_hash
    ):
        raise ValueError("TRAIN-fit data or S11 normalization differs from protocol")
    if source.train_snapshot.content_hash != protocol.dataset_snapshot_hash:
        raise ValueError("S10 TRAIN snapshot differs from S13 protocol")

    screen_cases = tuple(
        DogboneS13TrainingCase(
            tensors=normalize_case_values(
                prepare_operator_case(verified.case, verified.request, verified.result),
                training_data.normalization,
            ),
            request=verified.request,
            collocation=source.replay_train_collocation(verified),
        )
        for verified in source.iter_train_screen()
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during S13 screening")
    return _screen_metrics_from_cases(
        operator,
        fit_cases=training_data.cases,
        screen_cases=screen_cases,
        trial=trial,
        base_registry=protocol.objective_registry,
        expected_fit_ids=protocol.fit_case_ids,
        expected_screen_ids=protocol.screen_case_ids,
        expected_interior_count=DOGBONE_S10_DATASET_POLICY.interior_count,
    )


def _screen_metrics_from_cases(
    model: PiGinotOperator,
    *,
    fit_cases: tuple[DogboneS13TrainingCase, ...],
    screen_cases: tuple[DogboneS13TrainingCase, ...],
    trial: TrialConfig,
    base_registry: ObjectiveRegistry,
    expected_fit_ids: tuple[str, ...],
    expected_screen_ids: tuple[str, ...],
    expected_interior_count: int,
) -> SelectionMetrics:
    """Reduce exact labelled populations; testable with synthetic TRAIN cases."""

    operator = _validated_operator(model)
    if type(trial) is not TrialConfig:
        raise TypeError("trial must be an exact TrialConfig")
    if type(base_registry) is not ObjectiveRegistry or (
        base_registry.mode is not ObjectiveMode.HYBRID
    ):
        raise ValueError("screen requires an exact strong HYBRID registry")
    _require_population(fit_cases, expected_fit_ids, "fit")
    _require_population(screen_cases, expected_screen_ids, "screen")
    if set(expected_fit_ids) & set(expected_screen_ids):
        raise ValueError("fit and screen populations overlap")
    all_cases = (*fit_cases, *screen_cases)
    if len({case.tensors.leakage_group_id for case in all_cases}) != len(all_cases):
        raise ValueError("fit and screen leakage groups overlap")
    if len({case.tensors.normalization_hash for case in all_cases}) != 1:
        raise ValueError("fit and screen normalization identities differ")
    if type(expected_interior_count) is not int or expected_interior_count <= 0:
        raise ValueError("expected_interior_count must be positive")

    was_training = operator.training
    operator.eval()
    try:
        invalid = _count_invalid_collocation_points(
            operator,
            all_cases,
            expected_interior_count=expected_interior_count,
        )
        predictions = tuple(_predict_vertices(operator, case) for case in screen_cases)
        accuracy = calculate_architecture_screen_metrics(
            trial.trial_id,
            tuple(case.tensors for case in screen_cases),
            predictions,
            trainable_parameter_count=sum(
                parameter.numel()
                for parameter in operator.parameters()
                if parameter.requires_grad
            ),
            elapsed_seconds=0.0,
        )
        registry = _scaled_registry(base_registry, trial.physics_weight_scale)
        totals: list[float] = []
        for case in screen_cases:
            residuals = DogboneS13ObjectiveBuilder((case,))(
                operator, (case.tensors.case_id,), 1
            )
            evaluation = registry.evaluate(residuals, role=EvidenceRole.TRAIN)
            total = float(evaluation.total.detach())
            if not math.isfinite(total) or total < 0.0:
                raise ValueError("TRAIN-screen hybrid loss must be finite")
            totals.append(total)
    finally:
        operator.train(was_training)

    return SelectionMetrics(
        median_absolute_peak_error=accuracy.median_absolute_peak_error,
        p90_absolute_peak_error=accuracy.p90_absolute_peak_error,
        median_signed_peak_error=accuracy.median_signed_peak_error,
        p_relative_l2=accuracy.median_p_relative_l2,
        u_relative_l2=accuracy.median_u_relative_l2,
        hybrid_loss=math.fsum(totals) / len(screen_cases),
        invalid_collocation_points=invalid,
    )


def _require_population(
    cases: tuple[DogboneS13TrainingCase, ...],
    expected_ids: tuple[str, ...],
    label: str,
) -> None:
    if type(cases) is not tuple or any(
        type(case) is not DogboneS13TrainingCase for case in cases
    ):
        raise TypeError(f"{label} cases must contain exact DogboneS13TrainingCase")
    if type(expected_ids) is not tuple or not expected_ids:
        raise ValueError(f"{label} expected IDs must be non-empty")
    if tuple(case.tensors.case_id for case in cases) != expected_ids:
        raise ValueError(f"{label} cases differ from registered {label}_case_ids")
    if any(case.tensors.role is not EvidenceRole.TRAIN for case in cases):
        raise PermissionError(f"non-TRAIN {label} evidence cannot enter S13 screening")


def _predict_vertices(
    model: PiGinotOperator, case: DogboneS13TrainingCase
) -> OperatorOutput:
    tensors = case.tensors
    with torch.no_grad():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            return cast(
                OperatorOutput,
                model(
                    boundary_xy=tensors.boundary_xy.unsqueeze(0),
                    boundary_tags=tensors.boundary_tags.unsqueeze(0),
                    boundary_normals=tensors.boundary_normals.unsqueeze(0),
                    boundary_weights=tensors.boundary_weights.unsqueeze(0),
                    boundary_mask=tensors.boundary_mask.unsqueeze(0),
                    query_xy=tensors.query_xy.unsqueeze(0),
                    parameters=tensors.parameters_for(
                        model.config.parameter_names
                    ).unsqueeze(0),
                ),
            )


def _count_invalid_collocation_points(
    model: PiGinotOperator,
    cases: tuple[DogboneS13TrainingCase, ...],
    *,
    expected_interior_count: int,
) -> int:
    invalid = 0
    for case in cases:
        determinant = _determinants_for_case(model, case)
        if determinant.shape != (expected_interior_count,):
            raise ValueError("TRAIN case has an unexpected interior point count")
        if not bool(torch.isfinite(determinant).all().item()):
            raise ValueError("TRAIN interior deformation determinant must be finite")
        invalid += int((determinant <= DETERMINANT_MARGIN).sum().item())
    return invalid


def _determinants_for_case(
    model: PiGinotOperator, case: DogboneS13TrainingCase
) -> torch.Tensor:
    interior = tuple(
        point.coordinates
        for point in case.collocation.points
        if point.semantic_class is SemanticBCClass.INTERIOR
    )
    coordinates = torch.tensor(interior, dtype=torch.float64).unsqueeze(0)
    coordinates.requires_grad_(True)
    tensors = case.tensors
    with torch.enable_grad():
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            output = model(
                boundary_xy=tensors.boundary_xy.unsqueeze(0),
                boundary_tags=tensors.boundary_tags.unsqueeze(0),
                boundary_normals=tensors.boundary_normals.unsqueeze(0),
                boundary_weights=tensors.boundary_weights.unsqueeze(0),
                boundary_mask=tensors.boundary_mask.unsqueeze(0),
                query_xy=coordinates,
                parameters=tensors.parameters_for(
                    model.config.parameter_names
                ).unsqueeze(0),
            )
            gradient = displacement_gradient(
                output.u * _scales(case.request).displacement_mm,
                coordinates,
                _COORDINATE_CONVERSION,
            )
    deformation = torch.eye(2, dtype=gradient.dtype) + gradient
    determinant = (
        deformation[..., 0, 0] * deformation[..., 1, 1]
        - deformation[..., 0, 1] * deformation[..., 1, 0]
    )
    return determinant.detach().reshape(-1)


__all__ = ("evaluate_dogbone_s13_train_screen",)
