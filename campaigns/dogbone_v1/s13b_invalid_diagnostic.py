"""Read-only TRAIN locator for S13b's invalid collocation determinant."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from campaigns.dogbone_v1.s10_evidence import DOGBONE_S10_DATASET_POLICY
from campaigns.dogbone_v1.s11_contingency_data import (
    CaseNormalizationStatistics,
    normalize_case_values,
    prepare_operator_case,
)
from campaigns.dogbone_v1.s11_contingency_source import (
    SelectiveS10Source,
    open_selective_s10_source,
)
from campaigns.dogbone_v1.s13_data import materialize_dogbone_s13_training_data
from campaigns.dogbone_v1.s13_evaluate import _preflight_paths
from campaigns.dogbone_v1.s13_initialization import load_selected_s11_initialization
from campaigns.dogbone_v1.s13_objective import (
    DogboneS13TrainingCase,
    _validated_operator,
)
from campaigns.dogbone_v1.s13_pilot import _verify_training_data
from campaigns.dogbone_v1.s13_screen import _determinants_for_case, _require_population
from campaigns.dogbone_v1.s13b_evaluate import _load_checked_checkpoint
from campaigns.dogbone_v1.s13b_protocol import build_dogbone_s13b_pilot_plan
from campaigns.dogbone_v1.s13b_run import _build_s13b_binding
from piginot.contracts import canonical_json_bytes
from piginot.datasets import SemanticBCClass
from piginot.doe import EvidenceRole
from piginot.operators import PiGinotOperator
from piginot.physics._validation import DETERMINANT_MARGIN

_PEAK_TRAINING_REVISION = "65b091f136d72697698c1209427e96e0668a43e6"


@dataclass(frozen=True, slots=True)
class InvalidCollocationPoint:
    point_index: int  # Index in the full replayed CollocationBatch.points tuple.
    coordinates: tuple[float, float]
    determinant: float


@dataclass(frozen=True, slots=True)
class CaseDeterminants:
    case_id: str
    population: str
    min_determinant: float
    invalid_points: tuple[InvalidCollocationPoint, ...]


@dataclass(frozen=True, slots=True)
class InvalidCollocationReport:
    role: EvidenceRole
    checkpoint_manifest_hash: str
    determinant_margin: float
    interior_points_per_case: int
    cases: tuple[CaseDeterminants, ...]

    @property
    def invalid_point_count(self) -> int:
        return sum(len(case.invalid_points) for case in self.cases)


def _screen_cases(
    source: SelectiveS10Source,
    normalization: CaseNormalizationStatistics,
) -> tuple[DogboneS13TrainingCase, ...]:
    """Replay precisely the S13 TRAIN-screen case/collocation recipe."""

    return tuple(
        DogboneS13TrainingCase(
            tensors=normalize_case_values(
                prepare_operator_case(verified.case, verified.request, verified.result),
                normalization,
            ),
            request=verified.request,
            collocation=source.replay_train_collocation(verified),
        )
        for verified in source.iter_train_screen()
    )


def _locate_invalid_points(
    model: PiGinotOperator,
    *,
    fit_cases: tuple[DogboneS13TrainingCase, ...],
    screen_cases: tuple[DogboneS13TrainingCase, ...],
    expected_fit_ids: tuple[str, ...],
    expected_screen_ids: tuple[str, ...],
    expected_interior_count: int,
) -> tuple[CaseDeterminants, ...]:
    """Use the S13 screen's determinant rule; preserve point identity and minima."""

    operator = _validated_operator(model)
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
        results: list[CaseDeterminants] = []
        for population, cases in (
            ("TRAIN_FIT", fit_cases),
            ("TRAIN_SCREEN", screen_cases),
        ):
            for case in cases:
                interior = tuple(
                    (index, point)
                    for index, point in enumerate(case.collocation.points)
                    if point.semantic_class is SemanticBCClass.INTERIOR
                )
                determinant = _determinants_for_case(operator, case)
                if len(interior) != expected_interior_count or determinant.shape != (
                    expected_interior_count,
                ):
                    raise ValueError(
                        "TRAIN case has an unexpected interior point count"
                    )
                if not bool(torch.isfinite(determinant).all().item()):
                    raise ValueError(
                        "TRAIN interior deformation determinant must be finite"
                    )
                values = tuple(float(value) for value in determinant.tolist())
                invalid = tuple(
                    InvalidCollocationPoint(index, point.coordinates, value)
                    for (index, point), value in zip(interior, values, strict=True)
                    if value <= DETERMINANT_MARGIN
                )
                results.append(
                    CaseDeterminants(
                        case_id=case.tensors.case_id,
                        population=population,
                        min_determinant=min(values),
                        invalid_points=invalid,
                    )
                )
    finally:
        operator.train(was_training)
    return tuple(results)


def locate_s13b_invalid_collocation(
    *,
    s10_run_root: Path,
    s11_checkpoint_root: Path,
    peak_checkpoint_path: Path,
) -> InvalidCollocationReport:
    """Inspect all 192 fit and 64 screen cases without opening protected roles."""

    pilot = build_dogbone_s13b_pilot_plan()
    s10, s11, checkpoint, access_record = _preflight_paths(
        s10_run_root,
        s11_checkpoint_root,
        peak_checkpoint_path,
        optimizer_steps=800,
    )
    initialization = load_selected_s11_initialization(s11)
    binding = _build_s13b_binding(
        pilot,
        model=initialization.model,
        config=pilot.trainer_config,
        plan=pilot.sampler_plan,
        implementation_revision=_PEAK_TRAINING_REVISION,
    )
    manifest = _load_checked_checkpoint(
        checkpoint,
        model=initialization.model,
        config=pilot.trainer_config,
        binding=binding,
        expected_manifest_hash=None,
    )

    source = open_selective_s10_source(s10, development_access_record=access_record)
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened before S13b TRAIN diagnostic")
    data = materialize_dogbone_s13_training_data(source, initialization)
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during TRAIN materialization")
    _verify_training_data(data, pilot.base_protocol, source.bindings.content_hash)
    screen_cases = _screen_cases(source, data.normalization)
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during TRAIN-screen replay")
    cases = _locate_invalid_points(
        initialization.model,
        fit_cases=data.cases,
        screen_cases=screen_cases,
        expected_fit_ids=pilot.base_protocol.fit_case_ids,
        expected_screen_ids=pilot.base_protocol.screen_case_ids,
        expected_interior_count=DOGBONE_S10_DATASET_POLICY.interior_count,
    )
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during TRAIN diagnostic")
    return InvalidCollocationReport(
        role=EvidenceRole.TRAIN,
        checkpoint_manifest_hash=manifest.manifest_hash,
        determinant_margin=DETERMINANT_MARGIN,
        interior_points_per_case=DOGBONE_S10_DATASET_POLICY.interior_count,
        cases=cases,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only, non-gate S13b TRAIN invalid-collocation locator."
    )
    parser.add_argument("--s10-run-root", required=True, type=Path)
    parser.add_argument("--s11-checkpoint-root", required=True, type=Path)
    parser.add_argument("--peak-checkpoint-path", required=True, type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    report = locate_s13b_invalid_collocation(
        s10_run_root=options.s10_run_root,
        s11_checkpoint_root=options.s11_checkpoint_root,
        peak_checkpoint_path=options.peak_checkpoint_path,
    )
    print(
        canonical_json_bytes(
            {
                "non_gate": True,
                "role": report.role,
                "checkpoint_manifest_hash": report.checkpoint_manifest_hash,
                "determinant_margin": report.determinant_margin,
                "interior_points_per_case": report.interior_points_per_case,
                "checked_case_count": len(report.cases),
                "invalid_case_count": sum(
                    bool(case.invalid_points) for case in report.cases
                ),
                "invalid_point_count": report.invalid_point_count,
                "cases": tuple(case for case in report.cases if case.invalid_points),
            }
        ).decode("utf-8"),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())


__all__ = (
    "CaseDeterminants",
    "InvalidCollocationPoint",
    "InvalidCollocationReport",
    "build_parser",
    "locate_s13b_invalid_collocation",
    "main",
)
