"""Deterministic S1 solve/train/evaluate protocol and LEGACY report writer."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

import numpy as np

from experiments.spike.geometry import SpikeMesh

CAMPAIGN_SEED = 20260809
SPLIT_SEED = 991
TRAINING_SEED = 314159
LEARNING_RATES = (1e-3, 3e-4, 1e-4)
LR_SWEEP_EPOCHS = 150
FINAL_TRAINING_EPOCHS = 1500
GEOMETRY_BATCH_SIZE = 4
SUPERVISED_QUERY_COUNT = 96
PEAK_SUPERVISED_QUERY_COUNT = 32
PHYSICS_QUERY_COUNT = 24
SECTION_RESULTANT_SLICE_COUNT = 8
SECTION_RESULTANT_Y_POINT_COUNT = 16
PEAK_SUPERVISED_P_WEIGHT = 0.0
GRAD_CLIP_NORM = 1.0
UPSTREAM_APPLIED_DISPLACEMENT = 1.0
UPSTREAM_REPOSITORY = "BBahtiri/PI-GINOT-DogBone"
UPSTREAM_COMMIT = "9f6794141beca9d3e66433465c3b07039a7e41d2"
UPSTREAM_CHECKPOINT_SHA256 = (
    "6f8dbdfffbb2170dce9aa6fcd156a151736ae49846c2af525c429edca6dd7c85"
)
UPSTREAM_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/BBahtiri/PI-GINOT-DogBone/"
    "9f6794141beca9d3e66433465c3b07039a7e41d2/checkpoints/best.pt"
)
UPSTREAM_CHECKPOINT_EPOCH = 1424
S1_FEM_CONTENT_SHA256 = (
    "a21d4b6ab942e313774eab998e110061bbe3ccc5a9afec1ec15e5d4722eff211"
)
ITERATION3_SCREEN_SHA256 = (
    "c4b58d0956b0d006459ba05ebfe8bbf1097aafe606e62e8505c49e0666d8a7f6"
)
ITERATION5_SCREEN_SHA256 = (
    "ad94e68987591a0b8f7bc8c7c1c91764a73e0bc1e605f11ec5eb050ca9e884f3"
)
ITERATION6_SCREEN_SHA256 = (
    "59fcd20df0a03450b1871560244d982f4281f14ad64ed7521dc0babcd20b01dd"
)
ITERATION7_CONTROL_AUDIT_SHA256 = (
    "45640e9e7216e7e172e901559624317622cf010e78760f550e0cc37031ce56d2"
)
UPSTREAM_ARCHITECTURE_ADAPTATION = (
    "exact vendored upstream PI_GINOT encoder and displacement decoder; "
    "a direct four-component first-Piola conditioned MLP head is added with "
    "y_max/x_max input and E*u_delta/x_max output scaling"
)
STRESS_HEAD_VARIANT = "conditioned_mlp"
ITERATION3_CANDIDATE_VARIANT = "fully_conditioned_mlp"
STRESS_HEAD_CONDITIONING = ("y_max/x_max",)
STRESS_HEAD_OUTPUT_SCALE = "E*u_delta/x_max"
UPSTREAM_SERVED_STRESS = "P_served = P_predicted (direct second head)"
UPSTREAM_MATERIAL_STRESS_ROLE = "in-plane consistency oracle only"
REGULARITY_RELATIVE_CHANGE_LIMIT = 0.10
FEM_CACHE_VERSION = "s1-plane-stress-v2-nx32-ny10-udispl1mm"


CaseT = TypeVar("CaseT")
SolveT = TypeVar("SolveT")
TrainT = TypeVar("TrainT")


@dataclass(frozen=True)
class GeometrySplit:
    train: tuple[SpikeMesh, ...]
    evaluate: tuple[SpikeMesh, ...]


@dataclass(frozen=True)
class ProtocolOutcome:
    attempted: int
    converged: int
    failed: int
    failure_reasons: Mapping[str, int]
    stopped_before_training: bool
    training_result: object | None
    solve_results: tuple[object, ...]


@dataclass(frozen=True)
class TrainingOutcome:
    model: object
    selected_learning_rate: float
    trials: tuple[Mapping[str, Any], ...]
    final_mean_hybrid_loss: float
    training_seconds: float


@dataclass(frozen=True)
class UpstreamCheckpoint:
    path: Path
    sha256: str
    epoch: int
    state_dict: Mapping[str, object]


@dataclass(frozen=True)
class FineTuneStage:
    """One deterministic phase of the iteration-4 fine-tuning schedule."""

    name: str
    epochs: int
    encoder_lr: float | None
    decoder_lr: float | None
    stress_head_lr: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("fine-tune stage name must not be empty")
        if self.epochs < 1:
            raise ValueError("fine-tune stage epochs must be positive")
        learning_rates = (
            self.encoder_lr,
            self.decoder_lr,
            self.stress_head_lr,
        )
        if any(value is not None and value <= 0.0 for value in learning_rates):
            raise ValueError("fine-tune stage learning rates must be positive")


@dataclass(frozen=True)
class ServedStressPhysicsWeights:
    """Weights for optional direct-stress boundary and section constraints."""

    traction_free: float = 0.0
    partial_traction: float = 0.0
    section_resultant: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.traction_free,
            self.partial_traction,
            self.section_resultant,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "served-stress physics weights must be finite and non-negative"
            )

    @property
    def enabled(self) -> bool:
        """Return whether any optional served-stress physics term is active."""
        return any(
            value > 0.0
            for value in (
                self.traction_free,
                self.partial_traction,
                self.section_resultant,
            )
        )


NO_SERVED_STRESS_PHYSICS = ServedStressPhysicsWeights()
ITERATION5_PHYSICS_WEIGHTS = ServedStressPhysicsWeights(
    traction_free=2.0,
    partial_traction=2.0,
    section_resultant=10.0,
)
ITERATION6_PHYSICS_WEIGHTS = ServedStressPhysicsWeights(
    traction_free=0.2,
    partial_traction=0.2,
    section_resultant=1.0,
)


ITERATION4_FINE_TUNE_STAGES = (
    FineTuneStage("head_only", 60, None, None, 1e-3),
    FineTuneStage("decoder_and_head", 120, None, 1e-4, 1e-3),
    FineTuneStage("full_low_lr", 120, 3e-5, 1e-4, 3e-4),
)


def _architecture_iteration_evidence(
    path: Path | None = None,
) -> Mapping[str, Any]:
    """Load the completed training-only screen and reject unselected candidates."""
    uses_sealed_default = path is None
    screen_path = path or (
        Path(__file__).resolve().parent / "artifacts" / "iteration3_screen.json"
    )
    if not screen_path.is_file():
        raise FileNotFoundError(
            f"completed architecture screen not found: {screen_path}"
        )
    if uses_sealed_default and _file_sha256(screen_path) != ITERATION3_SCREEN_SHA256:
        raise ValueError("architecture screen SHA-256 does not match the sealed screen")
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if not isinstance(screen, Mapping):
        raise ValueError("architecture screen must be a JSON object")
    if screen.get("role") != "LEGACY" or screen.get("no_release_evidence") is not True:
        raise ValueError("architecture screen must be quarantined LEGACY evidence")
    if screen.get("upstream_checkpoint_sha256") != UPSTREAM_CHECKPOINT_SHA256:
        raise ValueError("architecture screen uses an incompatible upstream checkpoint")
    if screen.get("fem_content_sha256") != S1_FEM_CONTENT_SHA256:
        raise ValueError("architecture screen uses incompatible FEM content")
    evidence = screen.get("architecture_iteration")
    if not isinstance(evidence, Mapping):
        raise ValueError("architecture screen has no architecture_iteration object")
    selection = evidence.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("architecture screen has no selection object")
    if selection.get("held_out_geometry_accessed") is not False:
        raise ValueError("architecture screen must not access held-out FEM fields")
    variants = selection.get("variants")
    if not isinstance(variants, Mapping):
        raise ValueError("architecture screen has no candidate results")
    if not {STRESS_HEAD_VARIANT, ITERATION3_CANDIDATE_VARIANT}.issubset(variants):
        raise ValueError("architecture screen is missing a predeclared candidate")
    candidate = variants[ITERATION3_CANDIDATE_VARIANT]
    if not isinstance(candidate, Mapping):
        raise ValueError("architecture screen candidate result is invalid")
    _validate_iteration3_loss_contract(selection, variants)
    selected_variant = evidence.get("selected_variant")
    if selected_variant == ITERATION3_CANDIDATE_VARIANT:
        if candidate.get("guardrails_passed") is not True:
            raise ValueError(
                "architecture screen did not promote the selected candidate"
            )
    elif selected_variant != STRESS_HEAD_VARIANT:
        raise ValueError("architecture screen selected an incompatible variant")
    if evidence.get("iteration") != 3 or evidence.get("selected_variant") != (
        STRESS_HEAD_VARIANT
    ):
        raise ValueError("architecture screen selected an incompatible variant")
    normalized = {
        key: value
        for key, value in evidence.items()
        if key not in {"root_cause", "correction"}
    }
    normalized["hypothesis"] = (
        "explicit dimensionless load and normalized query coordinates might improve "
        "the S1 direct-P adapter"
    )
    normalized["decision"] = (
        "candidate rejected by the training-only guardrails; retain "
        f"{STRESS_HEAD_VARIANT}"
    )
    return normalized


def _iteration5_rejection_evidence(path: Path | None = None) -> Mapping[str, Any]:
    """Load the sealed rejected physics-objective screen for iteration 6."""
    uses_sealed_default = path is None
    screen_path = path or (
        Path(__file__).resolve().parent / "artifacts" / "iteration5_screen.json"
    )
    if not screen_path.is_file():
        raise FileNotFoundError(f"iteration-5 screen not found: {screen_path}")
    if uses_sealed_default and _file_sha256(screen_path) != ITERATION5_SCREEN_SHA256:
        raise ValueError("iteration-5 screen SHA-256 does not match sealed evidence")
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if not isinstance(screen, Mapping):
        raise ValueError("iteration-5 screen must be a JSON object")
    iteration = _require_mapping(
        screen.get("physics_objective_iteration"),
        "iteration-5 physics_objective_iteration",
    )
    selection = _require_mapping(iteration.get("selection"), "iteration-5 selection")
    objectives = _require_mapping(selection.get("objectives"), "iteration-5 objectives")
    candidate = _require_mapping(
        objectives.get("natural_bc_and_section_balance"),
        "iteration-5 natural_bc_and_section_balance",
    )
    if (
        screen.get("role") != "LEGACY"
        or screen.get("no_release_evidence") is not True
        or iteration.get("iteration") != 5
        or iteration.get("selected_objective") != "iteration3_joint_adam"
        or selection.get("held_out_geometry_accessed") is not False
        or candidate.get("guardrails_passed") is not False
    ):
        raise ValueError("iteration-5 evidence is not the sealed rejected candidate")
    return iteration


def _iteration6_rejection_evidence(path: Path | None = None) -> Mapping[str, Any]:
    """Load the sealed rejected weak-physics screen for iteration 7."""
    uses_sealed_default = path is None
    screen_path = path or (
        Path(__file__).resolve().parent / "artifacts" / "iteration6_screen.json"
    )
    if not screen_path.is_file():
        raise FileNotFoundError(f"iteration-6 screen not found: {screen_path}")
    if uses_sealed_default and _file_sha256(screen_path) != ITERATION6_SCREEN_SHA256:
        raise ValueError("iteration-6 screen SHA-256 does not match sealed evidence")
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if not isinstance(screen, Mapping):
        raise ValueError("iteration-6 screen must be a JSON object")
    iteration = _require_mapping(
        screen.get("physics_objective_iteration"),
        "iteration-6 physics_objective_iteration",
    )
    selection = _require_mapping(iteration.get("selection"), "iteration-6 selection")
    objectives = _require_mapping(selection.get("objectives"), "iteration-6 objectives")
    candidate = _require_mapping(
        objectives.get("weak_upstream_physics_0p1x"),
        "iteration-6 weak_upstream_physics_0p1x",
    )
    if (
        screen.get("role") != "LEGACY"
        or screen.get("no_release_evidence") is not True
        or iteration.get("iteration") != 6
        or iteration.get("selected_objective") != "iteration3_joint_adam"
        or selection.get("held_out_geometry_accessed") is not False
        or candidate.get("guardrails_passed") is not False
    ):
        raise ValueError("iteration-6 evidence is not the sealed rejected candidate")
    return iteration


def _validate_iteration3_loss_contract(
    selection: Mapping[str, Any], variants: Mapping[str, Any]
) -> None:
    if selection.get("peak_supervised_P_weight") != PEAK_SUPERVISED_P_WEIGHT:
        raise ValueError(
            "peak-supervised architecture screen selection."
            "peak_supervised_P_weight must equal 0.0"
        )
    for variant_name, raw_variant in variants.items():
        variant = _require_mapping(
            raw_variant,
            f"architecture_iteration.selection.variants.{variant_name}",
        )
        if variant.get("peak_supervised_P_weight") != PEAK_SUPERVISED_P_WEIGHT:
            raise ValueError(
                "peak-supervised architecture screen variant "
                f"{variant_name}.peak_supervised_P_weight must equal 0.0"
            )


@dataclass(frozen=True)
class EvaluationOutcome:
    family_metrics: Mapping[str, Mapping[str, Any]]
    per_case_metrics: Mapping[str, Mapping[str, Any]]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_by_geometry(
    cases: Sequence[SpikeMesh],
    *,
    master_seed: int,
) -> GeometrySplit:
    """Preassign exactly 20 train and 10 evaluation geometries per family."""
    train: list[SpikeMesh] = []
    evaluate: list[SpikeMesh] = []
    for family_index, family in enumerate(("dogbone", "centre_notched")):
        family_cases = sorted(
            (case for case in cases if case.family == family),
            key=lambda case: case.geometry_id,
        )
        if len(family_cases) != 30:
            raise ValueError(f"{family} must contain exactly 30 geometries")
        rng = np.random.default_rng(np.random.SeedSequence([master_seed, family_index]))
        order = rng.permutation(len(family_cases))
        shuffled = tuple(family_cases[index] for index in order)
        train.extend(shuffled[:20])
        evaluate.extend(shuffled[20:])
    return GeometrySplit(tuple(train), tuple(evaluate))


def solve_and_maybe_train(
    cases: Sequence[CaseT],
    *,
    fem_solver: Callable[[CaseT], SolveT],
    trainer: Callable[[object], TrainT],
    minimum_converged: int = 50,
) -> ProtocolOutcome:
    """Attempt every FEM case and enforce the <50 early-stop rule."""
    solves = tuple(fem_solver(case) for case in cases)
    converged = sum(bool(result.converged) for result in solves)
    failure_reasons = Counter(
        str(result.failure_reason or "unspecified failure")
        for result in solves
        if not bool(result.converged)
    )
    failed = len(solves) - converged
    if converged < minimum_converged:
        return ProtocolOutcome(
            attempted=len(solves),
            converged=converged,
            failed=failed,
            failure_reasons=dict(failure_reasons),
            stopped_before_training=True,
            training_result=None,
            solve_results=solves,
        )
    training_result = trainer(tuple(zip(cases, solves, strict=True)))
    return ProtocolOutcome(
        attempted=len(solves),
        converged=converged,
        failed=failed,
        failure_reasons=dict(failure_reasons),
        stopped_before_training=False,
        training_result=training_result,
        solve_results=solves,
    )


def _finite_nonnegative(value: object, path: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{path} must be finite and nonnegative")


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def validate_results(document: Mapping[str, Any]) -> None:
    """Validate the required G1 report surface before strict JSON emission."""
    if document.get("role") != "LEGACY":
        raise ValueError('results role must be "LEGACY"')
    if document.get("no_release_evidence") is not True:
        raise ValueError("results must be permanently excluded from release evidence")

    execution_profile = _require_mapping(
        document.get("execution_profile"), "execution_profile"
    )
    if (
        execution_profile.get("applied_displacement_mm")
        != UPSTREAM_APPLIED_DISPLACEMENT
    ):
        raise ValueError(
            "execution_profile.applied_displacement_mm must equal the upstream 1 mm load"
        )
    if "axial_engineering_strain" in execution_profile:
        raise ValueError(
            "execution_profile.axial_engineering_strain is incompatible with the "
            "upstream fixed-displacement protocol"
        )

    model = _require_mapping(document.get("model"), "model")
    expected_model_fields = {
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "architecture_adaptation": UPSTREAM_ARCHITECTURE_ADAPTATION,
        "served_stress": UPSTREAM_SERVED_STRESS,
        "material_stress_role": UPSTREAM_MATERIAL_STRESS_ROLE,
    }
    for field, expected in expected_model_fields.items():
        if model.get(field) != expected:
            raise ValueError(f"model.{field} does not match the corrected S1 protocol")
    checkpoint = _require_mapping(
        model.get("upstream_checkpoint"), "model.upstream_checkpoint"
    )
    expected_checkpoint_fields = {
        "url": UPSTREAM_CHECKPOINT_URL,
        "sha256": UPSTREAM_CHECKPOINT_SHA256,
        "epoch": UPSTREAM_CHECKPOINT_EPOCH,
    }
    for field, expected in expected_checkpoint_fields.items():
        if checkpoint.get(field) != expected:
            raise ValueError(
                f"model.upstream_checkpoint.{field} does not match the pinned artifact"
            )
    stress_head = _require_mapping(model.get("stress_head"), "model.stress_head")
    expected_stress_head_fields = {
        "variant": STRESS_HEAD_VARIANT,
        "conditioning": list(STRESS_HEAD_CONDITIONING),
        "output_scale": STRESS_HEAD_OUTPUT_SCALE,
        "youngs_modulus_MPa": 760.0,
        "output_layer_initialization": "zeros",
    }
    for field, expected in expected_stress_head_fields.items():
        if stress_head.get(field) != expected:
            raise ValueError(
                f"model.stress_head.{field} does not match the corrected S1 protocol"
            )

    artifacts = _require_mapping(document.get("artifacts"), "artifacts")
    checkpoint_artifact = _require_mapping(
        artifacts.get("checkpoint"), "artifacts.checkpoint"
    )
    if (
        not isinstance(checkpoint_artifact.get("sha256"), str)
        or len(checkpoint_artifact["sha256"]) != 64
    ):
        raise ValueError("artifacts.checkpoint.sha256 must be a SHA-256 hex digest")
    if artifacts.get("prediction_npz_count") != 20:
        raise ValueError("artifacts.prediction_npz_count must equal 20")
    if artifacts.get("comparison_plot_svg_count") != 20:
        raise ValueError("artifacts.comparison_plot_svg_count must equal 20")

    lr_sweep = _require_mapping(document.get("lr_sweep"), "lr_sweep")
    if lr_sweep.get("learning_rates") != list(LEARNING_RATES):
        raise ValueError("lr_sweep.learning_rates must match the declared sweep")
    if lr_sweep.get("epochs_per_trial") != LR_SWEEP_EPOCHS:
        raise ValueError("lr_sweep.epochs_per_trial must match the declared sweep")
    if lr_sweep.get("final_training_epochs") != FINAL_TRAINING_EPOCHS:
        raise ValueError(
            "lr_sweep.final_training_epochs must match the full training horizon"
        )
    if lr_sweep.get("gradient_clip_norm") != GRAD_CLIP_NORM:
        raise ValueError("lr_sweep.gradient_clip_norm must match the trainer")
    if lr_sweep.get("supervised_sampling") != (
        "reference-area-proportional nodal sampling with replacement"
    ):
        raise ValueError("lr_sweep.supervised_sampling must match the trainer")
    loss_weights = _require_mapping(
        lr_sweep.get("fixed_loss_weights"), "lr_sweep.fixed_loss_weights"
    )
    if loss_weights.get("equilibrium_on_P_served") != 1.0:
        raise ValueError(
            "lr_sweep.fixed_loss_weights.equilibrium_on_P_served must equal 1.0"
        )

    architecture_iteration = _require_mapping(
        document.get("architecture_iteration"), "architecture_iteration"
    )
    if architecture_iteration.get("iteration") != 3:
        raise ValueError("architecture_iteration.iteration must equal 3")
    if architecture_iteration.get("selected_variant") != STRESS_HEAD_VARIANT:
        raise ValueError(
            "architecture_iteration.selected_variant must match model.stress_head"
        )
    if architecture_iteration.get("held_out_evaluation_status") != (
        "confirmatory_not_pristine"
    ):
        raise ValueError(
            "architecture_iteration.held_out_evaluation_status must disclose reuse"
        )
    selection = _require_mapping(
        architecture_iteration.get("selection"), "architecture_iteration.selection"
    )
    if selection.get("training_geometry_count") != 40:
        raise ValueError(
            "architecture_iteration.selection.training_geometry_count must equal 40"
        )
    if selection.get("held_out_geometry_accessed") is not False:
        raise ValueError(
            "architecture_iteration.selection.held_out_geometry_accessed must be false"
        )
    variants = _require_mapping(
        selection.get("variants"), "architecture_iteration.selection.variants"
    )
    required_variants = {STRESS_HEAD_VARIANT, ITERATION3_CANDIDATE_VARIANT}
    if not required_variants.issubset(set(variants)):
        raise ValueError(
            "architecture_iteration.selection.variants must contain all candidates"
        )
    _validate_iteration3_loss_contract(selection, variants)
    for variant_name, raw_variant in variants.items():
        variant = _require_mapping(
            raw_variant,
            f"architecture_iteration.selection.variants.{variant_name}",
        )
        _finite_nonnegative(
            variant.get("pooled_worst_family_relative_l2_P"),
            "architecture_iteration.selection.variants."
            f"{variant_name}.pooled_worst_family_relative_l2_P",
        )
    candidate = variants[ITERATION3_CANDIDATE_VARIANT]
    if architecture_iteration.get("selected_variant") == ITERATION3_CANDIDATE_VARIANT:
        if candidate.get("guardrails_passed") is not True:
            raise ValueError(
                "architecture_iteration selected candidate must pass every guardrail"
            )

    families = _require_mapping(document.get("families"), "families")
    if set(families) != {"dogbone", "centre_notched"}:
        raise ValueError("families must contain dogbone and centre_notched exactly")

    for family_name, raw_family in families.items():
        family = _require_mapping(raw_family, f"families.{family_name}")
        if family.get("evaluation_count") != 10:
            raise ValueError(f"families.{family_name}.evaluation_count must equal 10")
        metrics = _require_mapping(
            family.get("metrics"), f"families.{family_name}.metrics"
        )
        for metric_name in (
            "relative_l2_u",
            "relative_l2_P",
            "worst_relative_peak_stress_error",
        ):
            _finite_nonnegative(
                metrics.get(metric_name),
                f"families.{family_name}.metrics.{metric_name}",
            )

        fem = _require_mapping(family.get("fem"), f"families.{family_name}.fem")
        attempted = fem.get("attempted")
        converged = fem.get("converged")
        failed = fem.get("failed")
        if (
            attempted != 30
            or not isinstance(converged, int)
            or not isinstance(failed, int)
        ):
            raise ValueError(f"families.{family_name}.fem counts are invalid")
        if converged + failed != attempted:
            raise ValueError(f"families.{family_name}.fem counts do not balance")
        reasons = _require_mapping(
            fem.get("failure_reasons"),
            f"families.{family_name}.fem.failure_reasons",
        )
        if sum(int(value) for value in reasons.values()) != failed:
            raise ValueError(f"families.{family_name}.failure reasons do not balance")

        timing = _require_mapping(
            family.get("timing"), f"families.{family_name}.timing"
        )
        for timing_name in ("mean_fem_seconds", "mean_inference_seconds"):
            _finite_nonnegative(
                timing.get(timing_name),
                f"families.{family_name}.timing.{timing_name}",
            )
        peak_qoi = _require_mapping(
            family.get("peak_qoi"), f"families.{family_name}.peak_qoi"
        )
        if peak_qoi.get("status") not in {"well_posed", "singular_or_diagnostic"}:
            raise ValueError(f"families.{family_name}.peak_qoi.status is invalid")
        if not isinstance(peak_qoi.get("reason"), str) or not peak_qoi["reason"]:
            raise ValueError(f"families.{family_name}.peak_qoi.reason is required")


def build_gate_document(
    results: Mapping[str, Any],
    *,
    isolation_passed: bool,
    verification: Mapping[str, Any],
    timestamp_utc: str | None = None,
) -> Mapping[str, Any]:
    """Build the machine-readable LEGACY G1 artifact from measured S1 results."""
    protocol = _require_mapping(results.get("protocol"), "protocol")
    converged = int(protocol.get("converged", 0))
    attempted = int(protocol.get("attempted", 0))
    split_proof = _require_mapping(protocol.get("split_proof"), "split_proof")
    split_keys = (
        "geometry_ids_disjoint",
        "mesh_ids_disjoint",
        "rng_seeds_disjoint",
        "mesh_qualified_node_ids_disjoint",
        "outcome_independent_preassignment",
    )
    split_passed = all(split_proof.get(key) is True for key in split_keys)
    sanity = _require_mapping(results.get("fem_sanity"), "fem_sanity")
    raw_sanity_error = sanity.get("max_error")
    sanity_error = (
        float(raw_sanity_error)
        if isinstance(raw_sanity_error, (int, float))
        and not isinstance(raw_sanity_error, bool)
        and math.isfinite(float(raw_sanity_error))
        else None
    )
    sanity_threshold = float(sanity.get("threshold", 1e-6))
    try:
        validate_results(results)
        report_passed = True
        report_reason = "results.json contains every required finite family metric"
    except ValueError as error:
        report_passed = False
        report_reason = str(error)

    checks = [
        {
            "id": "G1.solves",
            "status": "PASS" if converged >= 50 and attempted == 60 else "FAIL",
            "measured": {"attempted": attempted, "converged": converged},
            "threshold": ">=50 of exactly 60 FEM cases converged",
            "comparison": "gte",
        },
        {
            "id": "G1.sanity",
            "status": (
                "PASS"
                if sanity_error is not None and sanity_error <= sanity_threshold
                else "FAIL"
            ),
            "measured": {"max_error": sanity_error},
            "threshold": sanity_threshold,
            "comparison": "lte",
        },
        {
            "id": "G1.split",
            "status": "PASS" if split_passed else "FAIL",
            "measured": dict(split_proof),
            "threshold": "no shared geometry, mesh, RNG seed, or mesh-qualified node identity",
            "comparison": "all_true",
        },
        {
            "id": "G1.report",
            "status": "PASS" if report_passed else "FAIL",
            "measured": {"complete": report_passed, "reason": report_reason},
            "threshold": "complete finite held-out metrics for both families",
            "comparison": "eq",
        },
        {
            "id": "G1.isolation",
            "status": "PASS" if isolation_passed else "FAIL",
            "measured": {"tests/test_spike_isolation.py": isolation_passed},
            "threshold": "standalone isolation guard passes",
            "comparison": "eq",
        },
    ]
    verdict = "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
    return {
        "gate_id": "G1",
        "step": "S1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "commit": results.get("commit"),
        "commit_scope": results.get("commit_scope"),
        "worktree_content_sha256": results.get("worktree_content_sha256"),
        "execution_profile_hash": results.get("execution_profile_hash"),
        "execution_profile": results.get("execution_profile"),
        "timestamp_utc": timestamp_utc
        or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "checks": checks,
        "verification": dict(verification),
        "verdict": verdict,
        "verdict_reason": (
            "Every required G1 check passed; interpretation remains human-owned."
            if verdict == "PASS"
            else "One or more required G1 checks failed; interpretation remains human-owned."
        ),
    }


def write_json_strict(target: Path, document: Mapping[str, Any]) -> None:
    """Write standards-compliant JSON, rejecting NaN and infinities."""
    serialized = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"{serialized}\n", encoding="utf-8")


def _fem_content_hash(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in ("experiments/spike/geometry.py", "experiments/spike/fem.py"):
        digest.update(relative.encode())
        digest.update((repo_root / relative).read_bytes())
    digest.update(FEM_CACHE_VERSION.encode())
    digest.update(b"applied_displacement_mm=1.0")
    return digest.hexdigest()


def _cache_path(cache_dir: Path, mesh: SpikeMesh, fem_content_hash: str) -> Path:
    return cache_dir / (
        f"{FEM_CACHE_VERSION}-{fem_content_hash[:16]}-{mesh.mesh_id}.npz"
    )


def _solve_with_cache(
    mesh: SpikeMesh,
    cache_dir: Path,
    fem_content_hash: str,
) -> object:
    from experiments.spike.fem import FEMResult, solve_case

    target = _cache_path(cache_dir, mesh, fem_content_hash)
    if target.exists():
        with np.load(target, allow_pickle=False) as cached:
            if (
                str(cached["fem_content_sha256"].item()) != fem_content_hash
                or str(cached["mesh_id"].item()) != mesh.mesh_id
            ):
                raise RuntimeError(f"invalid FEM cache identity: {target}")
            converged = bool(cached["converged"].item())
            reason_text = str(cached["failure_reason"].item())
            return FEMResult(
                converged=converged,
                failure_reason=reason_text or None,
                displacement=cached["displacement"] if converged else None,
                first_piola=cached["first_piola"] if converged else None,
                solve_seconds=float(cached["solve_seconds"].item()),
                newton_iterations=int(cached["newton_iterations"].item()),
            )

    result = solve_case(mesh)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        role=np.asarray("LEGACY"),
        no_release_evidence=np.asarray(True),
        fem_content_sha256=np.asarray(fem_content_hash),
        mesh_id=np.asarray(mesh.mesh_id),
        converged=np.asarray(result.converged),
        failure_reason=np.asarray(result.failure_reason or ""),
        displacement=(
            result.displacement if result.displacement is not None else np.empty((0, 2))
        ),
        first_piola=(
            result.first_piola
            if result.first_piola is not None
            else np.empty((0, 2, 2))
        ),
        solve_seconds=np.asarray(result.solve_seconds),
        newton_iterations=np.asarray(result.newton_iterations),
    )
    return result


def _interior_node_indices(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    x_max = float(np.max(x))
    top_by_x = {float(value): float(np.max(y[x == value])) for value in np.unique(x)}
    top = np.asarray([top_by_x[float(value)] for value in x])
    tolerance = 1e-10 * max(x_max, float(np.max(y)), 1.0)
    return np.flatnonzero(
        (x > tolerance)
        & (x < x_max - tolerance)
        & (y > tolerance)
        & (y < top - tolerance)
    )


def _lumped_nodal_areas(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Return P1 lumped reference-area weights for a triangular mesh."""
    triangles = points[cells]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    twice_area = np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    areas = 0.5 * twice_area
    weights = np.zeros(len(points), dtype=np.float64)
    for local_vertex in range(3):
        np.add.at(weights, cells[:, local_vertex], areas / 3.0)
    return weights


def _element_centroid_fields(
    points: np.ndarray,
    cells: np.ndarray,
    displacement: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct exact P1 element-centroid fields from a nodal FEM solution."""
    from experiments.spike.fem import first_piola_plane_stress

    coordinates = np.asarray(points, dtype=np.float64)
    connectivity = np.asarray(cells, dtype=np.int64)
    nodal_u = np.asarray(displacement, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("points must have shape (nodes, 2)")
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError("cells must have shape (elements, 3)")
    if nodal_u.shape != coordinates.shape:
        raise ValueError("displacement must have shape (nodes, 2)")
    if (
        not np.isfinite(coordinates).all()
        or not np.isfinite(nodal_u).all()
        or connectivity.size == 0
        or np.min(connectivity) < 0
        or np.max(connectivity) >= len(coordinates)
    ):
        raise ValueError("element field inputs must be finite and index valid")

    triangles = coordinates[connectivity]
    reference_edges = np.stack(
        (triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=-1,
    )
    twice_area = np.linalg.det(reference_edges)
    scale = max(float(np.ptp(coordinates, axis=0).max()), 1.0)
    if np.any(np.abs(twice_area) <= np.finfo(np.float64).eps * scale**2):
        raise ValueError("element fields require positive-area triangles")

    element_u = nodal_u[connectivity]
    displacement_edges = np.stack(
        (element_u[:, 1] - element_u[:, 0], element_u[:, 2] - element_u[:, 0]),
        axis=-1,
    )
    gradient = displacement_edges @ np.linalg.inv(reference_edges)
    deformation = np.eye(2)[None] + gradient
    stress = first_piola_plane_stress(deformation)
    return (
        triangles.mean(axis=1),
        element_u.mean(axis=1),
        stress,
        0.5 * np.abs(twice_area),
    )


def _patch_average_element_stress(
    cells: np.ndarray,
    element_stress: np.ndarray,
    *,
    node_count: int,
) -> np.ndarray:
    """Apply the legacy unweighted element-to-node recovery exactly."""
    recovered = np.zeros((node_count, 2, 2), dtype=np.float64)
    counts = np.zeros(node_count, dtype=np.float64)
    for local_vertex in range(3):
        vertices = cells[:, local_vertex]
        np.add.at(recovered, vertices, element_stress)
        np.add.at(counts, vertices, 1.0)
    if np.any(counts <= 0.0):
        raise ValueError("every mesh node must belong to at least one element")
    return recovered / counts[:, None, None]


def _weighted_squared_norm(field: np.ndarray, nodal_areas: np.ndarray) -> float:
    """Approximate the spatial L2 norm squared using lumped P1 quadrature."""
    component_axes = tuple(range(1, field.ndim))
    pointwise_squared = np.sum(field**2, axis=component_axes)
    return float(np.sum(nodal_areas * pointwise_squared))


def _top_profile_points(mesh: SpikeMesh) -> np.ndarray:
    """Return the ordered top profile of a mapped quarter-domain mesh."""
    x_coordinates = np.unique(mesh.points[:, 0])
    return np.asarray(
        [
            mesh.points[np.isclose(mesh.points[:, 0], x), :][
                np.argmax(mesh.points[np.isclose(mesh.points[:, 0], x), 1])
            ]
            for x in x_coordinates
        ],
        dtype=np.float64,
    )


def _top_free_boundary_quadrature(
    mesh: SpikeMesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return midpoint quadrature on the physical traction-free top boundary."""
    top = _top_profile_points(mesh)
    tangents = np.diff(top, axis=0)
    lengths = np.linalg.norm(tangents, axis=1)
    if len(lengths) == 0 or not np.all(np.isfinite(lengths)):
        raise ValueError("top boundary must contain finite segments")
    if np.any(lengths <= 0.0):
        raise ValueError("top boundary segments must have positive length")
    midpoints = 0.5 * (top[:-1] + top[1:])
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0])) / lengths[:, None]
    if np.any(normals[:, 1] <= 0.0):
        raise ValueError("top boundary normals must point out of the quarter domain")
    return midpoints, normals, lengths / np.sum(lengths)


def _partial_traction_boundary_quadrature(
    mesh: SpikeMesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return complementary-traction quadrature on essential boundaries."""
    x_coordinates = np.unique(mesh.points[:, 0])
    bottom_nodes = np.column_stack((x_coordinates, np.zeros_like(x_coordinates)))
    left_nodes = mesh.points[np.isclose(mesh.points[:, 0], np.min(mesh.points[:, 0]))]
    right_nodes = mesh.points[np.isclose(mesh.points[:, 0], np.max(mesh.points[:, 0]))]
    left_nodes = left_nodes[np.argsort(left_nodes[:, 1])]
    right_nodes = right_nodes[np.argsort(right_nodes[:, 1])]
    edges = (
        (bottom_nodes, np.asarray([0.0, -1.0]), 0),
        (left_nodes, np.asarray([-1.0, 0.0]), 1),
        (right_nodes, np.asarray([1.0, 0.0]), 1),
    )
    point_groups: list[np.ndarray] = []
    normal_groups: list[np.ndarray] = []
    direction_groups: list[np.ndarray] = []
    weight_groups: list[np.ndarray] = []
    for nodes, normal, direction in edges:
        tangents = np.diff(nodes, axis=0)
        lengths = np.linalg.norm(tangents, axis=1)
        if len(lengths) == 0 or np.any(lengths <= 0.0):
            raise ValueError("essential boundary segments must have positive length")
        point_groups.append(0.5 * (nodes[:-1] + nodes[1:]))
        normal_groups.append(np.broadcast_to(normal, (len(lengths), 2)).copy())
        direction_groups.append(np.full(len(lengths), direction, dtype=np.int64))
        weight_groups.append(lengths / (3.0 * np.sum(lengths)))
    return (
        np.concatenate(point_groups),
        np.concatenate(normal_groups),
        np.concatenate(direction_groups),
        np.concatenate(weight_groups),
    )


def _section_resultant_quadrature(
    mesh: SpikeMesh,
    *,
    slice_count: int,
    y_point_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return midpoint quadrature for interior vertical reference sections."""
    if slice_count < 2:
        raise ValueError("section resultant requires at least two slices")
    if y_point_count < 1:
        raise ValueError("section resultant requires at least one y point")
    top = _top_profile_points(mesh)
    x_max = float(top[-1, 0])
    fractions = (np.arange(slice_count, dtype=np.float64) + 0.5) / slice_count
    section_x = fractions * x_max
    section_height = np.interp(section_x, top[:, 0], top[:, 1])
    if np.any(section_height <= 0.0):
        raise ValueError("section heights must be positive")
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(y_point_count)
    eta = 0.5 * (gauss_nodes + 1.0)
    points = np.empty((slice_count, y_point_count, 2), dtype=np.float64)
    points[..., 0] = section_x[:, None]
    points[..., 1] = section_height[:, None] * eta[None, :]
    weights = 0.5 * section_height[:, None] * gauss_weights[None, :]
    return points, weights


def _traction_loss(
    served_stress: object,
    normals: object,
    quadrature_weights: object,
    p_reference: object,
) -> object:
    """Return a reference-frame traction loss for direct served stress."""
    import torch

    traction = torch.einsum("bqij,bqj->bqi", served_stress, normals)
    squared_norm = traction.square().sum(dim=-1)
    weighted = (quadrature_weights * squared_norm).sum(dim=1)
    return torch.mean(weighted / p_reference.clamp_min(1e-6))


def _partial_traction_loss(
    served_stress: object,
    normals: object,
    directions: object,
    quadrature_weights: object,
    p_reference: object,
) -> object:
    """Penalize only natural components complementary to essential conditions."""
    import torch

    traction = torch.einsum("bqij,bqj->bqi", served_stress, normals)
    selected = torch.gather(traction, dim=2, index=directions[..., None]).squeeze(-1)
    weighted = (quadrature_weights * selected.square()).sum(dim=1)
    return torch.mean(weighted / p_reference.clamp_min(1e-6))


def _section_resultant_loss(
    served_stress: object,
    quadrature_weights: object,
    p_reference: object,
    y_max: object,
) -> object:
    """Penalize variation of the model-predicted axial section resultant."""
    import torch

    axial_resultant = (served_stress[..., 0, 0] * quadrature_weights).sum(dim=-1)
    mean_resultant = axial_resultant.mean(dim=1, keepdim=True)
    force_reference = (p_reference * y_max.square()).clamp_min(1e-6)
    variance = (axial_resultant - mean_resultant).square().mean(dim=1)
    return torch.mean(variance / force_reference)


def _equilibrium_residual_prefix(
    served_stress: object,
    coordinates: object,
    *,
    count: int,
) -> object:
    """Compute ``Div(P)`` for a query prefix without differentiating later outputs."""
    import torch

    if count < 1 or count > int(coordinates.shape[1]):  # type: ignore[union-attr]
        raise ValueError("equilibrium prefix count is outside the query range")
    divergence_components: list[torch.Tensor] = []
    for spatial_component in range(2):
        component_divergence = torch.zeros_like(coordinates[:, :count, 0])
        for reference_direction in range(2):
            stress_component = served_stress[
                :, :count, spatial_component, reference_direction
            ]
            if not stress_component.requires_grad:
                derivative = torch.zeros_like(coordinates)
            else:
                derivative_result = torch.autograd.grad(
                    stress_component.sum(),
                    coordinates,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                derivative = (
                    derivative_result
                    if derivative_result is not None
                    else torch.zeros_like(coordinates)
                )
            component_divergence = (
                component_divergence + derivative[:, :count, reference_direction]
            )
        divergence_components.append(component_divergence)
    return torch.stack(divergence_components, dim=-1)


def _tensor_dataset(
    pairs: Sequence[tuple[SpikeMesh, object]],
) -> Mapping[str, object]:
    import torch

    meshes = tuple(mesh for mesh, _ in pairs)
    results = tuple(result for _, result in pairs)
    if any(not bool(result.converged) for result in results):
        raise ValueError("training requires every preassigned geometry to converge")
    points = np.stack([mesh.points for mesh in meshes]).astype(np.float32)
    boundary = np.stack([mesh.boundary_points for mesh in meshes]).astype(np.float32)
    displacement = np.stack(
        [np.asarray(result.displacement) for result in results]
    ).astype(np.float32)
    stress = np.stack([np.asarray(result.first_piola) for result in results]).astype(
        np.float32
    )
    interior = np.stack([_interior_node_indices(mesh.points) for mesh in meshes])
    nodal_areas = np.stack(
        [_lumped_nodal_areas(mesh.points, mesh.cells) for mesh in meshes]
    ).astype(np.float32)
    qoi_weights = np.zeros_like(nodal_areas)
    for row, mesh in enumerate(meshes):
        qoi_weights[row, mesh.qoi_node_ids] = nodal_areas[row, mesh.qoi_node_ids]
    top_free = tuple(_top_free_boundary_quadrature(mesh) for mesh in meshes)
    partial_traction = tuple(
        _partial_traction_boundary_quadrature(mesh) for mesh in meshes
    )
    section_resultant = tuple(
        _section_resultant_quadrature(
            mesh,
            slice_count=SECTION_RESULTANT_SLICE_COUNT,
            y_point_count=SECTION_RESULTANT_Y_POINT_COUNT,
        )
        for mesh in meshes
    )
    x_max = np.asarray([np.max(mesh.points[:, 0]) for mesh in meshes], dtype=np.float32)
    y_max = np.asarray([np.max(mesh.points[:, 1]) for mesh in meshes], dtype=np.float32)
    return {
        "meshes": meshes,
        "points": torch.from_numpy(points),
        "boundary": torch.from_numpy(boundary),
        "displacement": torch.from_numpy(displacement),
        "stress": torch.from_numpy(stress),
        "interior": torch.from_numpy(interior.astype(np.int64)),
        "nodal_areas": torch.from_numpy(nodal_areas),
        "qoi_weights": torch.from_numpy(qoi_weights),
        "top_free_points": torch.from_numpy(
            np.stack([item[0] for item in top_free]).astype(np.float32)
        ),
        "top_free_normals": torch.from_numpy(
            np.stack([item[1] for item in top_free]).astype(np.float32)
        ),
        "top_free_weights": torch.from_numpy(
            np.stack([item[2] for item in top_free]).astype(np.float32)
        ),
        "partial_traction_points": torch.from_numpy(
            np.stack([item[0] for item in partial_traction]).astype(np.float32)
        ),
        "partial_traction_normals": torch.from_numpy(
            np.stack([item[1] for item in partial_traction]).astype(np.float32)
        ),
        "partial_traction_directions": torch.from_numpy(
            np.stack([item[2] for item in partial_traction]).astype(np.int64)
        ),
        "partial_traction_weights": torch.from_numpy(
            np.stack([item[3] for item in partial_traction]).astype(np.float32)
        ),
        "section_points": torch.from_numpy(
            np.stack([item[0] for item in section_resultant]).astype(np.float32)
        ),
        "section_weights": torch.from_numpy(
            np.stack([item[1] for item in section_resultant]).astype(np.float32)
        ),
        "x_max": torch.from_numpy(x_max),
        "y_max": torch.from_numpy(y_max),
        "u_delta": torch.full(
            (len(meshes),), UPSTREAM_APPLIED_DISPLACEMENT, dtype=torch.float32
        ),
    }


def _element_tensor_dataset(
    pairs: Sequence[tuple[SpikeMesh, object]],
    *,
    validate_patch_recovery: bool = True,
) -> Mapping[str, object]:
    """Add positive interior-quadrature stress labels to the nodal dataset."""
    import torch

    base = dict(_tensor_dataset(pairs))
    barycentric = np.asarray(
        (
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
            (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
        ),
        dtype=np.float64,
    )
    quadrature_points: list[np.ndarray] = []
    quadrature_displacement: list[np.ndarray] = []
    quadrature_targets: list[np.ndarray] = []
    quadrature_weights: list[np.ndarray] = []
    element_stresses: list[np.ndarray] = []
    element_areas: list[np.ndarray] = []
    recovery_errors: list[float] = []
    projection_gaps: list[float] = []
    for mesh, result in pairs:
        _, _, stress, areas = _element_centroid_fields(
            mesh.points,
            mesh.cells,
            np.asarray(result.displacement),
        )
        triangles = np.asarray(mesh.points, dtype=np.float64)[mesh.cells]
        points = np.einsum("qv,evd->eqd", barycentric, triangles).reshape(-1, 2)
        displacement = np.asarray(result.displacement, dtype=np.float64)
        query_u = np.einsum(
            "qv,evd->eqd", barycentric, displacement[mesh.cells]
        ).reshape(-1, 2)
        targets = np.repeat(stress[:, None, :, :], 3, axis=1).reshape(-1, 2, 2)
        weights = np.repeat((areas / 3.0)[:, None], 3, axis=1).reshape(-1)
        if not np.isfinite(points).all() or not np.isfinite(targets).all():
            raise ValueError("stress quadrature fields must be finite")
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("stress quadrature weights must be finite and positive")
        if not np.isclose(
            float(np.sum(weights)),
            float(np.sum(areas)),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("stress quadrature weights must sum to mesh area")

        cached_stress = np.asarray(result.first_piola, dtype=np.float64)
        recovered = _patch_average_element_stress(
            mesh.cells,
            stress,
            node_count=len(mesh.points),
        )
        recovery_error = float(
            np.max(np.abs(recovered - cached_stress))
            / max(float(np.max(np.abs(cached_stress))), 1.0)
        )
        if validate_patch_recovery and recovery_error > 1e-10:
            raise ValueError(
                "element stress does not reproduce cached nodal patch recovery"
            )
        patch_at_quadrature = np.einsum(
            "qv,evij->eqij", barycentric, cached_stress[mesh.cells]
        ).reshape(-1, 2, 2)
        projection_gaps.append(
            math.sqrt(
                float(
                    np.sum(
                        weights
                        * np.sum((patch_at_quadrature - targets) ** 2, axis=(1, 2))
                    )
                )
                / max(
                    float(np.sum(weights * np.sum(targets**2, axis=(1, 2)))),
                    1e-24,
                )
            )
        )
        quadrature_points.append(points)
        quadrature_displacement.append(query_u)
        quadrature_targets.append(targets)
        quadrature_weights.append(weights)
        element_stresses.append(stress)
        element_areas.append(areas)
        recovery_errors.append(recovery_error)

    base.update(
        {
            "stress_quadrature_points": torch.from_numpy(
                np.stack(quadrature_points).astype(np.float32)
            ),
            "stress_quadrature_targets": torch.from_numpy(
                np.stack(quadrature_targets).astype(np.float32)
            ),
            "stress_quadrature_displacement": torch.from_numpy(
                np.stack(quadrature_displacement).astype(np.float32)
            ),
            "stress_quadrature_weights": torch.from_numpy(
                np.stack(quadrature_weights).astype(np.float32)
            ),
            "element_stress": torch.from_numpy(
                np.stack(element_stresses).astype(np.float32)
            ),
            "element_areas": torch.from_numpy(
                np.stack(element_areas).astype(np.float32)
            ),
            "element_patch_recovery_max_relative_error": max(recovery_errors),
            "raw_to_nodal_projection_relative_l2": tuple(projection_gaps),
            "stress_quadrature_rule": "positive three-point barycentric P1",
        }
    )
    return base


def _new_pretrained_model(
    upstream_state_dict: Mapping[str, object],
    *,
    config: object | None = None,
) -> object:
    """Create the exact upstream model and load its displacement checkpoint."""
    from experiments.spike.model import MixedPhysicsOperator

    model = MixedPhysicsOperator(config)
    model.load_upstream_state_dict(upstream_state_dict)
    return model


def _load_upstream_checkpoint(
    path: Path,
    *,
    expected_sha256: str = UPSTREAM_CHECKPOINT_SHA256,
) -> UpstreamCheckpoint:
    """Load the pinned checkpoint only after exact hash verification."""
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"upstream checkpoint not found: {path}")
    measured_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if measured_sha256 != expected_sha256:
        raise ValueError(
            "upstream checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, measured {measured_sha256}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("upstream checkpoint payload must be a mapping")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("upstream checkpoint has no model_state_dict")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("upstream checkpoint epoch must be a non-negative integer")
    return UpstreamCheckpoint(
        path=path.resolve(),
        sha256=measured_sha256,
        epoch=epoch,
        state_dict=dict(state_dict),
    )


def _resolve_upstream_checkpoint(
    explicit_path: Path | None = None,
) -> UpstreamCheckpoint:
    """Resolve an explicit/local checkpoint or download the pinned public artifact."""
    configured = os.environ.get("PI_GINOT_UPSTREAM_CHECKPOINT")
    if explicit_path is not None and configured:
        raise ValueError(
            "provide either upstream_checkpoint_path or "
            "PI_GINOT_UPSTREAM_CHECKPOINT, not both"
        )
    if explicit_path is not None:
        return _load_upstream_checkpoint(explicit_path)
    if configured:
        return _load_upstream_checkpoint(Path(configured))

    cache_directory = (
        Path(tempfile.gettempdir()) / "pi-ginot-upstream-9f679414-checkpoint"
    )
    target = cache_directory / "best.pt"
    if target.exists():
        return _load_upstream_checkpoint(target)

    cache_directory.mkdir(parents=True, exist_ok=True)
    temporary = cache_directory / "best.pt.download"
    with urllib.request.urlopen(UPSTREAM_CHECKPOINT_URL, timeout=120) as response:
        temporary.write_bytes(response.read())
    checkpoint = _load_upstream_checkpoint(temporary)
    temporary.replace(target)
    return UpstreamCheckpoint(
        path=target.resolve(),
        sha256=checkpoint.sha256,
        epoch=checkpoint.epoch,
        state_dict=checkpoint.state_dict,
    )


def _gather_queries(
    values: object,
    geometry_indices: object,
    query_indices: object,
) -> object:
    import torch

    selected = values[geometry_indices]  # type: ignore[index]
    trailing = selected.shape[2:]
    index = query_indices.reshape(*query_indices.shape, *([1] * len(trailing)))
    index = index.expand(*query_indices.shape, *trailing)
    return torch.gather(selected, dim=1, index=index)


def _sample_supervised_indices(
    nodal_areas: object,
    *,
    count: int,
    generator: object,
) -> object:
    """Sample reference nodes according to the evaluation's area measure."""
    import torch

    if count <= 0:
        raise ValueError("supervised sample count must be positive")
    if nodal_areas.ndim != 2:  # type: ignore[union-attr]
        raise ValueError("nodal_areas must have shape (batch, nodes)")
    if not bool(torch.isfinite(nodal_areas).all()):
        raise ValueError("nodal_areas must be finite")
    if bool((nodal_areas < 0).any()):
        raise ValueError("nodal_areas must be non-negative")
    totals = nodal_areas.sum(dim=1, keepdim=True)
    if bool((totals <= 0).any()):
        raise ValueError("each geometry must have positive total nodal area")
    return torch.multinomial(
        nodal_areas / totals,
        count,
        replacement=True,
        generator=generator,
    )


def _hybrid_batch_loss(
    model: object,
    data: Mapping[str, object],
    geometry_indices: object,
    generator: object,
    *,
    peak_supervised_p_weight: float = PEAK_SUPERVISED_P_WEIGHT,
    served_stress_physics: ServedStressPhysicsWeights = NO_SERVED_STRESS_PHYSICS,
    stress_supervision: str = "nodal_patch",
    stress_generator: object | None = None,
) -> tuple[object, Mapping[str, float]]:
    import torch

    from experiments.spike.model import material_first_piola_from_displacement

    batch_size = int(geometry_indices.numel())
    supervised_indices = _sample_supervised_indices(
        data["nodal_areas"][geometry_indices],  # type: ignore[index]
        count=SUPERVISED_QUERY_COUNT,
        generator=generator,
    )
    query_u = _gather_queries(data["points"], geometry_indices, supervised_indices)
    target_u = _gather_queries(
        data["displacement"], geometry_indices, supervised_indices
    )
    legacy_target_p = _gather_queries(
        data["stress"], geometry_indices, supervised_indices
    )
    if stress_supervision == "nodal_patch":
        query = query_u
        target_p = legacy_target_p
        p_output_slice = slice(None)
    elif stress_supervision == "element_quadrature":
        if stress_generator is None:
            raise ValueError(
                "element-quadrature stress supervision requires a separate generator"
            )
        stress_indices = _sample_supervised_indices(
            data["stress_quadrature_weights"][geometry_indices],  # type: ignore[index]
            count=SUPERVISED_QUERY_COUNT,
            generator=stress_generator,
        )
        query_p = _gather_queries(
            data["stress_quadrature_points"], geometry_indices, stress_indices
        )
        target_p = _gather_queries(
            data["stress_quadrature_targets"], geometry_indices, stress_indices
        )
        query = torch.cat((query_u, query_p), dim=1)
        p_output_slice = slice(SUPERVISED_QUERY_COUNT, None)
    else:
        raise ValueError(
            "stress_supervision must be 'nodal_patch' or 'element_quadrature'"
        )
    boundary = data["boundary"][geometry_indices]  # type: ignore[index]
    x_max = data["x_max"][geometry_indices]  # type: ignore[index]
    y_max = data["y_max"][geometry_indices]  # type: ignore[index]
    u_delta = data["u_delta"][geometry_indices]  # type: ignore[index]
    output = model(query, boundary, u_delta, x_max, y_max)

    u_reference = target_u.square().mean(dim=(1, 2)).detach().clamp_min(1e-8)
    p_reference = target_p.square().mean(dim=(1, 2, 3)).detach().clamp_min(1e-6)
    physics_p_reference = (
        legacy_target_p.square().mean(dim=(1, 2, 3)).detach().clamp_min(1e-6)
    )
    supervised_u = torch.mean(
        (output.u[:, :SUPERVISED_QUERY_COUNT] - target_u).square().mean(dim=(1, 2))
        / u_reference
    )
    supervised_p = torch.mean(
        (output.P_served[:, p_output_slice] - target_p).square().mean(dim=(1, 2, 3))
        / p_reference
    )
    if peak_supervised_p_weight < 0.0:
        raise ValueError("peak supervised P weight must be non-negative")
    peak_supervised_p = torch.zeros(
        (), dtype=supervised_p.dtype, device=supervised_p.device
    )
    if peak_supervised_p_weight > 0.0:
        peak_indices = _sample_supervised_indices(
            data["qoi_weights"][geometry_indices],  # type: ignore[index]
            count=PEAK_SUPERVISED_QUERY_COUNT,
            generator=generator,
        )
        peak_query = _gather_queries(data["points"], geometry_indices, peak_indices)
        peak_target_p = _gather_queries(data["stress"], geometry_indices, peak_indices)
        peak_output = model(peak_query, boundary, u_delta, x_max, y_max)
        peak_supervised_p = torch.mean(
            (peak_output.P_served - peak_target_p).square().mean(dim=(1, 2, 3))
            / physics_p_reference
        )

    interior = data["interior"][geometry_indices]  # type: ignore[index]
    interior_choices = torch.randint(
        int(interior.shape[1]),
        (batch_size, PHYSICS_QUERY_COUNT),
        generator=generator,
    )
    physics_indices = torch.gather(interior, dim=1, index=interior_choices)
    interior_query = _gather_queries(
        data["points"], geometry_indices, physics_indices
    ).detach()
    physics_query = interior_query
    top_count = 0
    partial_count = 0
    section_shape = (0, 0)
    if served_stress_physics.enabled:
        top_query = data["top_free_points"][geometry_indices]  # type: ignore[index]
        partial_query = data["partial_traction_points"][  # type: ignore[index]
            geometry_indices
        ]
        section_query = data["section_points"][geometry_indices]  # type: ignore[index]
        top_count = int(top_query.shape[1])
        partial_count = int(partial_query.shape[1])
        section_shape = (int(section_query.shape[1]), int(section_query.shape[2]))
        physics_query = torch.cat(
            (
                interior_query,
                top_query,
                partial_query,
                section_query.reshape(batch_size, -1, 2),
            ),
            dim=1,
        )
    physics_query = physics_query.detach().requires_grad_(True)
    physics_output = model(physics_query, boundary, u_delta, x_max, y_max)
    divergence = _equilibrium_residual_prefix(
        physics_output.P_served,
        physics_query,
        count=PHYSICS_QUERY_COUNT,
    )
    material_p = material_first_piola_from_displacement(
        physics_output.u[:, :PHYSICS_QUERY_COUNT],
        physics_query,
    )[:, :PHYSICS_QUERY_COUNT]
    interior_stress = physics_output.P_served[:, :PHYSICS_QUERY_COUNT]
    characteristic_length = torch.sqrt(x_max * y_max).detach().clamp_min(1e-6)
    equilibrium_reference = (
        torch.sqrt(physics_p_reference) / characteristic_length
    ).square()
    equilibrium = torch.mean(
        divergence.square().mean(dim=(1, 2)) / equilibrium_reference.clamp_min(1e-8)
    )
    constitutive = torch.mean(
        (interior_stress - material_p).square().mean(dim=(1, 2, 3))
        / physics_p_reference
    )
    traction_free = torch.zeros(
        (), dtype=constitutive.dtype, device=constitutive.device
    )
    partial_traction = torch.zeros_like(traction_free)
    section_resultant = torch.zeros_like(traction_free)
    if served_stress_physics.enabled:
        top_start = PHYSICS_QUERY_COUNT
        partial_start = top_start + top_count
        section_start = partial_start + partial_count
        top_stress = physics_output.P_served[:, top_start:partial_start]
        partial_stress = physics_output.P_served[:, partial_start:section_start]
        section_stress = physics_output.P_served[:, section_start:].reshape(
            batch_size,
            section_shape[0],
            section_shape[1],
            2,
            2,
        )
        traction_free = _traction_loss(
            top_stress,
            data["top_free_normals"][geometry_indices],  # type: ignore[index]
            data["top_free_weights"][geometry_indices],  # type: ignore[index]
            physics_p_reference,
        )
        partial_traction = _partial_traction_loss(
            partial_stress,
            data["partial_traction_normals"][geometry_indices],  # type: ignore[index]
            data["partial_traction_directions"][  # type: ignore[index]
                geometry_indices
            ],
            data["partial_traction_weights"][geometry_indices],  # type: ignore[index]
            physics_p_reference,
        )
        section_resultant = _section_resultant_loss(
            section_stress,
            data["section_weights"][geometry_indices],  # type: ignore[index]
            physics_p_reference,
            y_max,
        )
    total = (
        supervised_u
        + supervised_p
        + peak_supervised_p_weight * peak_supervised_p
        + equilibrium
        + constitutive
        + served_stress_physics.traction_free * traction_free
        + served_stress_physics.partial_traction * partial_traction
        + served_stress_physics.section_resultant * section_resultant
    )
    components = {
        "supervised_u": float(supervised_u.detach()),
        "supervised_P": float(supervised_p.detach()),
        "peak_supervised_P": float(peak_supervised_p.detach()),
        "equilibrium": float(equilibrium.detach()),
        "constitutive_in_plane": float(constitutive.detach()),
        "traction_free_on_P_served": float(traction_free.detach()),
        "partial_traction_on_P_served": float(partial_traction.detach()),
        "section_resultant_on_P_served": float(section_resultant.detach()),
        "total": float(total.detach()),
    }
    return total, components


def _train_fixed_epochs(
    model: object,
    data: Mapping[str, object],
    selected_geometry_indices: Sequence[int],
    *,
    learning_rate: float,
    epochs: int,
    random_seed: int,
    label: str,
    peak_supervised_p_weight: float = PEAK_SUPERVISED_P_WEIGHT,
    served_stress_physics: ServedStressPhysicsWeights = NO_SERVED_STRESS_PHYSICS,
    stress_supervision: str = "nodal_patch",
    stress_random_seed: int | None = None,
    max_seconds: float | None = None,
) -> float:
    import torch

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(random_seed)
    if stress_supervision == "element_quadrature":
        if stress_random_seed is None:
            raise ValueError(
                "element-quadrature stress supervision requires stress_random_seed"
            )
        stress_generator = torch.Generator().manual_seed(stress_random_seed)
    elif stress_supervision == "nodal_patch":
        if stress_random_seed is not None:
            raise ValueError("nodal stress supervision does not use stress_random_seed")
        stress_generator = None
    else:
        raise ValueError(
            "stress_supervision must be 'nodal_patch' or 'element_quadrature'"
        )
    selected = torch.tensor(selected_geometry_indices, dtype=torch.long)
    final_epoch_loss = math.inf
    started = perf_counter()
    model.train()
    for epoch in range(epochs):
        permutation = selected[torch.randperm(len(selected), generator=generator)]
        epoch_losses: list[float] = []
        for start in range(0, len(permutation), GEOMETRY_BATCH_SIZE):
            batch = permutation[start : start + GEOMETRY_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss_arguments: dict[str, object] = {
                "peak_supervised_p_weight": peak_supervised_p_weight,
                "served_stress_physics": served_stress_physics,
            }
            if stress_generator is not None:
                loss_arguments.update(
                    {
                        "stress_supervision": stress_supervision,
                        "stress_generator": stress_generator,
                    }
                )
            loss, _ = _hybrid_batch_loss(
                model,
                data,
                batch,
                generator,
                **loss_arguments,
            )
            if not bool(torch.isfinite(loss).detach()):
                raise RuntimeError(f"non-finite hybrid loss in {label}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP_NORM
            )
            if not bool(torch.isfinite(gradient_norm).detach()):
                raise RuntimeError(f"non-finite gradient in {label}")
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            if max_seconds is not None and perf_counter() - started > max_seconds:
                raise TimeoutError(f"training time cap exceeded in {label}")
        final_epoch_loss = float(np.mean(epoch_losses))
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"{label}: epoch {epoch + 1}/{epochs}, "
                f"hybrid_loss={final_epoch_loss:.6g}",
                flush=True,
            )
    return final_epoch_loss


def _staged_optimizer_groups(model: object) -> list[dict[str, Any]]:
    """Partition the active operator path and freeze unrelated upstream heads."""
    modules = {
        "encoder": getattr(model, "encoder", None),
        "decoder": getattr(model, "decoder", None),
        "p_head": getattr(model, "p_head", None),
    }
    if any(module is None for module in modules.values()):
        raise ValueError("staged fine-tuning requires encoder, decoder, and p_head")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    seen: set[int] = set()
    groups: list[dict[str, Any]] = []
    for group_name, module in modules.items():
        parameters = list(module.parameters())
        if not parameters:
            raise ValueError(f"staged fine-tuning group {group_name} has no parameters")
        parameter_ids = {id(parameter) for parameter in parameters}
        if seen.intersection(parameter_ids):
            raise ValueError("staged fine-tuning parameter groups must be disjoint")
        seen.update(parameter_ids)
        groups.append(
            {
                "params": parameters,
                "lr": 0.0,
                "group_name": group_name,
            }
        )
    return groups


def _apply_fine_tune_stage(
    model: object,
    optimizer: object,
    stage: FineTuneStage,
) -> tuple[object, ...]:
    """Apply trainability, module modes, and learning rates for one phase."""
    modules = {
        "encoder": model.encoder,
        "decoder": model.decoder,
        "p_head": model.p_head,
    }
    learning_rates = {
        "encoder": stage.encoder_lr,
        "decoder": stage.decoder_lr,
        "p_head": stage.stress_head_lr,
    }
    model.train()
    trainable: list[object] = []
    for group_name, module in modules.items():
        enabled = learning_rates[group_name] is not None
        # Keep the exact upstream training-time point-sampling behavior even when
        # its weights are frozen. The pinned model has no stateful buffers and all
        # configured dropout probabilities are zero.
        module.train()
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)
            if enabled:
                trainable.append(parameter)

    optimizer_groups = {
        group.get("group_name"): group for group in optimizer.param_groups
    }
    if set(optimizer_groups) != set(modules):
        raise ValueError("staged optimizer groups do not match the operator path")
    for group_name, learning_rate in learning_rates.items():
        optimizer_groups[group_name]["lr"] = (
            0.0 if learning_rate is None else learning_rate
        )
    if not trainable:
        raise ValueError("fine-tune stage must enable at least one parameter")
    return tuple(trainable)


def _train_staged_epochs(
    model: object,
    data: Mapping[str, object],
    selected_geometry_indices: Sequence[int],
    *,
    stages: Sequence[FineTuneStage],
    random_seed: int,
    label: str,
    peak_supervised_p_weight: float = PEAK_SUPERVISED_P_WEIGHT,
) -> Mapping[str, Any]:
    """Fine-tune one model with a single Adam state across declared phases."""
    import torch

    declared_stages = tuple(stages)
    if not declared_stages:
        raise ValueError("staged fine-tuning requires at least one stage")
    names = tuple(stage.name for stage in declared_stages)
    if len(set(names)) != len(names):
        raise ValueError("fine-tune stage names must be unique")
    selected = torch.tensor(selected_geometry_indices, dtype=torch.long)
    if selected.numel() == 0:
        raise ValueError("staged fine-tuning requires selected geometries")
    optimizer = torch.optim.Adam(_staged_optimizer_groups(model))
    generator = torch.Generator().manual_seed(random_seed)
    phase_records: list[Mapping[str, Any]] = []
    final_epoch_loss = math.inf
    completed_epochs = 0
    for stage in declared_stages:
        trainable = _apply_fine_tune_stage(model, optimizer, stage)
        for stage_epoch in range(stage.epochs):
            permutation = selected[torch.randperm(len(selected), generator=generator)]
            epoch_losses: list[float] = []
            for start in range(0, len(permutation), GEOMETRY_BATCH_SIZE):
                batch = permutation[start : start + GEOMETRY_BATCH_SIZE]
                optimizer.zero_grad(set_to_none=True)
                loss, _ = _hybrid_batch_loss(
                    model,
                    data,
                    batch,
                    generator,
                    peak_supervised_p_weight=peak_supervised_p_weight,
                )
                if not bool(torch.isfinite(loss).detach()):
                    raise RuntimeError(
                        f"non-finite hybrid loss in {label} {stage.name}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            final_epoch_loss = float(np.mean(epoch_losses))
            completed_epochs += 1
            if completed_epochs % 10 == 0 or stage_epoch + 1 == stage.epochs:
                print(
                    f"{label}: stage={stage.name} epoch "
                    f"{stage_epoch + 1}/{stage.epochs} total={completed_epochs} "
                    f"hybrid_loss={final_epoch_loss:.6g}",
                    flush=True,
                )
        phase_records.append(
            {
                "name": stage.name,
                "epochs": stage.epochs,
                "encoder_lr": stage.encoder_lr,
                "decoder_lr": stage.decoder_lr,
                "stress_head_lr": stage.stress_head_lr,
                "final_hybrid_loss": final_epoch_loss,
            }
        )
    return {
        "final_hybrid_loss": final_epoch_loss,
        "total_epochs": completed_epochs,
        "phases": phase_records,
        "optimizer_state_preserved_across_phases": True,
    }


def _supervised_validation_score(
    model: object,
    data: Mapping[str, object],
    selected_geometry_indices: Sequence[int],
) -> float:
    import torch

    model.eval()
    family_error = {"dogbone": 0.0, "centre_notched": 0.0}
    family_reference = {"dogbone": 0.0, "centre_notched": 0.0}
    meshes = data["meshes"]
    with torch.no_grad():
        for index in selected_geometry_indices:
            geometry = torch.tensor([index], dtype=torch.long)
            output = model(
                data["points"][geometry],  # type: ignore[index]
                data["boundary"][geometry],  # type: ignore[index]
                data["u_delta"][geometry],  # type: ignore[index]
                data["x_max"][geometry],  # type: ignore[index]
                data["y_max"][geometry],  # type: ignore[index]
            )
            target_p = data["stress"][geometry]  # type: ignore[index]
            weights = data["nodal_areas"][geometry]  # type: ignore[index]
            squared_error = torch.sum(
                weights * (output.P_served - target_p).square().sum(dim=(-2, -1))
            )
            squared_reference = torch.sum(weights * target_p.square().sum(dim=(-2, -1)))
            family = meshes[index].family  # type: ignore[index,union-attr]
            family_error[family] += float(squared_error)
            family_reference[family] += float(squared_reference)

    missing = tuple(
        family for family, reference in family_reference.items() if reference <= 0.0
    )
    if missing:
        raise ValueError(
            "development stress score requires positive reference energy for "
            f"both families; missing={missing}"
        )
    return max(
        math.sqrt(family_error[family] / family_reference[family])
        for family in family_reference
    )


def _select_learning_rate(trials: Sequence[Mapping[str, Any]]) -> float:
    viable = tuple(trial for trial in trials if trial.get("status") == "completed")
    if not viable:
        raise RuntimeError("every learning-rate trial failed")
    selected = min(
        viable,
        key=lambda trial: float(trial["training_only_development_score"]),
    )
    return float(selected["learning_rate"])


def _fit_and_select_model(
    training_pairs: Sequence[tuple[SpikeMesh, object]],
    upstream_state_dict: Mapping[str, object],
) -> TrainingOutcome:
    import torch

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(TRAINING_SEED)
    data = _tensor_dataset(training_pairs)
    meshes = data["meshes"]
    fit_indices: list[int] = []
    development_indices: list[int] = []
    for family in ("dogbone", "centre_notched"):
        family_indices = [
            index for index, mesh in enumerate(meshes) if mesh.family == family
        ]
        fit_indices.extend(family_indices[:16])
        development_indices.extend(family_indices[16:])

    trials: list[Mapping[str, Any]] = []
    viable_trials: list[Mapping[str, Any]] = []
    started = perf_counter()
    for trial_index, learning_rate in enumerate(LEARNING_RATES):
        torch.manual_seed(TRAINING_SEED)
        model = _new_pretrained_model(upstream_state_dict)
        try:
            final_loss = _train_fixed_epochs(
                model,
                data,
                fit_indices,
                learning_rate=learning_rate,
                epochs=LR_SWEEP_EPOCHS,
                random_seed=TRAINING_SEED + 10,
                label=f"lr={learning_rate:g}",
            )
            validation = _supervised_validation_score(model, data, development_indices)
            trial: Mapping[str, Any] = {
                "learning_rate": learning_rate,
                "status": "completed",
                "final_fit_hybrid_loss": final_loss,
                "training_only_development_score": validation,
            }
            trials.append(trial)
            viable_trials.append(trial)
            print(
                f"lr trial {trial_index + 1}/{len(LEARNING_RATES)}: "
                f"development_score={validation:.6g}",
                flush=True,
            )
        except (RuntimeError, ValueError) as error:
            trials.append(
                {
                    "learning_rate": learning_rate,
                    "status": "failed",
                    "failure_reason": str(error),
                }
            )
            print(
                f"lr trial {trial_index + 1}/{len(LEARNING_RATES)} failed: {error}",
                flush=True,
            )
        del model

    selected = _select_learning_rate(viable_trials)
    torch.manual_seed(TRAINING_SEED)
    final_model = _new_pretrained_model(upstream_state_dict)
    final_loss = _train_fixed_epochs(
        final_model,
        data,
        range(len(training_pairs)),
        learning_rate=selected,
        epochs=FINAL_TRAINING_EPOCHS,
        random_seed=TRAINING_SEED + 20,
        label=f"final lr={selected:g}",
    )
    elapsed = perf_counter() - started
    return TrainingOutcome(
        model=final_model,
        selected_learning_rate=float(selected),
        trials=tuple(trials),
        final_mean_hybrid_loss=final_loss,
        training_seconds=elapsed,
    )


def _stress_magnitude(stress: np.ndarray) -> np.ndarray:
    return np.linalg.norm(stress, axis=(-2, -1))


def _evaluate_model(
    model: object,
    evaluation_pairs: Sequence[tuple[SpikeMesh, object]],
) -> EvaluationOutcome:
    import torch

    model.eval()
    family_metrics: dict[str, Mapping[str, Any]] = {}
    per_case_metrics: dict[str, Mapping[str, Any]] = {}
    for family in ("dogbone", "centre_notched"):
        family_pairs = tuple(
            pair for pair in evaluation_pairs if pair[0].family == family
        )
        squared_u_error = 0.0
        squared_u_reference = 0.0
        squared_p_error = 0.0
        squared_p_reference = 0.0
        peak_errors: list[float] = []
        inference_times: list[float] = []
        for mesh, result in family_pairs:
            points = torch.from_numpy(mesh.points.astype(np.float32))[None]
            boundary = torch.from_numpy(mesh.boundary_points.astype(np.float32))[None]
            x_max = torch.tensor([float(np.max(mesh.points[:, 0]))])
            y_max = torch.tensor([float(np.max(mesh.points[:, 1]))])
            u_delta = torch.full_like(x_max, UPSTREAM_APPLIED_DISPLACEMENT)
            with torch.no_grad():
                model(points, boundary, u_delta, x_max, y_max)
                repetitions: list[float] = []
                for _ in range(5):
                    started = perf_counter()
                    output = model(points, boundary, u_delta, x_max, y_max)
                    repetitions.append(perf_counter() - started)
            predicted_u = output.u[0].numpy().astype(np.float64)
            predicted_p = output.P_served[0].numpy().astype(np.float64)
            target_u = np.asarray(result.displacement)
            target_p = np.asarray(result.first_piola)
            nodal_areas = _lumped_nodal_areas(mesh.points, mesh.cells)
            u_error = _weighted_squared_norm(predicted_u - target_u, nodal_areas)
            u_reference = _weighted_squared_norm(target_u, nodal_areas)
            p_error = _weighted_squared_norm(predicted_p - target_p, nodal_areas)
            p_reference = _weighted_squared_norm(target_p, nodal_areas)
            squared_u_error += u_error
            squared_u_reference += u_reference
            squared_p_error += p_error
            squared_p_reference += p_reference
            target_peak = float(np.max(_stress_magnitude(target_p[mesh.qoi_node_ids])))
            predicted_peak = float(
                np.max(_stress_magnitude(predicted_p[mesh.qoi_node_ids]))
            )
            peak_error = abs(predicted_peak - target_peak) / max(target_peak, 1e-12)
            peak_errors.append(peak_error)
            inference_times.append(float(np.mean(repetitions)))
            per_case_metrics[mesh.geometry_id] = {
                "relative_l2_u": math.sqrt(u_error / max(u_reference, 1e-24)),
                "relative_l2_P": math.sqrt(p_error / max(p_reference, 1e-24)),
                "l2_quadrature": "P1 lumped reference-area weights",
                "relative_peak_stress_error": peak_error,
                "inference_seconds": float(np.mean(repetitions)),
            }
        family_metrics[family] = {
            "l2_quadrature": "P1 lumped reference-area weights",
            "relative_l2_u": math.sqrt(
                squared_u_error / max(squared_u_reference, 1e-24)
            ),
            "relative_l2_P": math.sqrt(
                squared_p_error / max(squared_p_reference, 1e-24)
            ),
            "worst_relative_peak_stress_error": max(peak_errors),
            "mean_inference_seconds": float(np.mean(inference_times)),
        }
    return EvaluationOutcome(family_metrics, per_case_metrics)


def _evaluate_model_element_quadrature(
    model: object,
    evaluation_pairs: Sequence[tuple[SpikeMesh, object]],
) -> EvaluationOutcome:
    """Evaluate raw P1 stress on positive element quadrature and legacy nodes."""
    import torch

    pairs = tuple(evaluation_pairs)
    legacy = _evaluate_model(model, pairs)
    data = _element_tensor_dataset(pairs)
    family_error = {family: 0.0 for family in ("dogbone", "centre_notched")}
    family_reference = {family: 0.0 for family in family_error}
    per_case_metrics: dict[str, Mapping[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for index, (mesh, _result) in enumerate(pairs):
            geometry = torch.tensor([index], dtype=torch.long)
            query = data["stress_quadrature_points"][geometry]  # type: ignore[index]
            output = model(
                query,
                data["boundary"][geometry],  # type: ignore[index]
                data["u_delta"][geometry],  # type: ignore[index]
                data["x_max"][geometry],  # type: ignore[index]
                data["y_max"][geometry],  # type: ignore[index]
            )
            target_p = data["stress_quadrature_targets"][geometry]  # type: ignore[index]
            target_u = data["stress_quadrature_displacement"][  # type: ignore[index]
                geometry
            ]
            weights = data["stress_quadrature_weights"][geometry]  # type: ignore[index]
            p_error = float(
                torch.sum(
                    weights * (output.P_served - target_p).square().sum(dim=(-2, -1))
                )
            )
            p_reference = float(
                torch.sum(weights * target_p.square().sum(dim=(-2, -1)))
            )
            u_error = float(
                torch.sum(weights * (output.u - target_u).square().sum(dim=-1))
            )
            u_reference = float(torch.sum(weights * target_u.square().sum(dim=-1)))
            family_error[mesh.family] += p_error
            family_reference[mesh.family] += p_reference
            legacy_case = legacy.per_case_metrics[mesh.geometry_id]
            per_case_metrics[mesh.geometry_id] = {
                **dict(legacy_case),
                "legacy_nodal_relative_l2_P": legacy_case["relative_l2_P"],
                "relative_l2_P": math.sqrt(p_error / max(p_reference, 1e-24)),
                "quadrature_relative_l2_u": math.sqrt(
                    u_error / max(u_reference, 1e-24)
                ),
                "P_l2_quadrature": "positive three-point element area",
            }

    family_metrics: dict[str, Mapping[str, Any]] = {}
    for family in family_error:
        legacy_family = legacy.family_metrics[family]
        family_metrics[family] = {
            **dict(legacy_family),
            "legacy_nodal_relative_l2_P": legacy_family["relative_l2_P"],
            "relative_l2_P": math.sqrt(
                family_error[family] / max(family_reference[family], 1e-24)
            ),
            "P_l2_quadrature": "positive three-point element area",
        }
    return EvaluationOutcome(family_metrics, per_case_metrics)


def _summarize_screen_folds(
    folds: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Aggregate paired training-only fold metrics without touching held-out data."""
    families = ("dogbone", "centre_notched")
    mean_p = {
        family: float(
            np.mean([fold["family_metrics"][family]["relative_l2_P"] for fold in folds])
        )
        for family in families
    }
    mean_u = {
        family: float(
            np.mean([fold["family_metrics"][family]["relative_l2_u"] for fold in folds])
        )
        for family in families
    }
    peak_errors = [
        case["relative_peak_stress_error"]
        for fold in folds
        for case in fold["per_case_metrics"].values()
    ]
    displacement_errors = [
        case["relative_l2_u"]
        for fold in folds
        for case in fold["per_case_metrics"].values()
    ]
    inference_seconds = [
        fold["family_metrics"][family]["mean_inference_seconds"]
        for fold in folds
        for family in families
    ]
    return {
        "mean_relative_l2_P": mean_p,
        "mean_relative_l2_u": mean_u,
        "pooled_worst_family_relative_l2_P": max(mean_p.values()),
        "overall_worst_peak_stress_error": max(peak_errors),
        "median_peak_stress_error": float(np.median(peak_errors)),
        "worst_single_case_relative_l2_u": max(displacement_errors),
        "mean_inference_seconds": float(np.mean(inference_seconds)),
    }


def _screen_guardrail_checks(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    stress_head_parameter_count: int,
    total_parameter_growth: float,
) -> Mapping[str, bool]:
    """Apply the predeclared iteration-3 promotion rule to fold summaries."""
    baseline_score = float(baseline["pooled_worst_family_relative_l2_P"])
    candidate_score = float(candidate["pooled_worst_family_relative_l2_P"])
    baseline_p = baseline["mean_relative_l2_P"]
    candidate_p = candidate["mean_relative_l2_P"]
    baseline_u = baseline["mean_relative_l2_u"]
    candidate_u = candidate["mean_relative_l2_u"]
    families = ("dogbone", "centre_notched")
    checks = {
        "primary_score_improved_by_at_least_10_percent": (
            candidate_score <= 0.90 * baseline_score
        ),
        "family_P_within_absolute_0_005": all(
            float(candidate_p[family]) <= float(baseline_p[family]) + 0.005
            for family in families
        ),
        "family_u_within_absolute_0_005": all(
            float(candidate_u[family]) <= float(baseline_u[family]) + 0.005
            for family in families
        ),
        "worst_case_u_within_absolute_0_01": (
            float(candidate["worst_single_case_relative_l2_u"])
            <= float(baseline["worst_single_case_relative_l2_u"]) + 0.01
        ),
        "worst_peak_within_absolute_0_05": (
            float(candidate["overall_worst_peak_stress_error"])
            <= float(baseline["overall_worst_peak_stress_error"]) + 0.05
        ),
        "median_peak_within_absolute_0_02": (
            float(candidate["median_peak_stress_error"])
            <= float(baseline["median_peak_stress_error"]) + 0.02
        ),
        "stress_head_parameters_at_most_10000": (stress_head_parameter_count <= 10_000),
        "total_model_growth_below_1_percent": total_parameter_growth < 0.01,
        "mean_inference_within_1_25x": (
            float(candidate["mean_inference_seconds"])
            <= 1.25 * float(baseline["mean_inference_seconds"])
        ),
    }
    return {**checks, "all_passed": all(checks.values())}


def _served_stress_physics_diagnostics(
    model: object,
    data: Mapping[str, object],
    geometry_indices: Sequence[int],
) -> Mapping[str, Any]:
    """Measure label-free boundary and section consistency on selected geometries."""
    import torch

    by_family: dict[str, dict[str, list[float]]] = {
        family: {
            "traction_free_normalized_rms": [],
            "partial_traction_normalized_rms": [],
            "section_resultant_normalized_rms": [],
            "section_resultant_coefficient_of_variation": [],
        }
        for family in ("dogbone", "centre_notched")
    }
    model.eval()
    with torch.no_grad():
        for index in geometry_indices:
            geometry_index = torch.tensor([index], dtype=torch.long)
            top = data["top_free_points"][geometry_index]  # type: ignore[index]
            partial = data["partial_traction_points"][geometry_index]  # type: ignore[index]
            section = data["section_points"][geometry_index]  # type: ignore[index]
            query = torch.cat(
                (top, partial, section.reshape(1, -1, 2)),
                dim=1,
            )
            output = model(
                query,
                data["boundary"][geometry_index],  # type: ignore[index]
                data["u_delta"][geometry_index],  # type: ignore[index]
                data["x_max"][geometry_index],  # type: ignore[index]
                data["y_max"][geometry_index],  # type: ignore[index]
            )
            target_stress = data["stress"][geometry_index]  # type: ignore[index]
            p_reference = target_stress.square().mean(dim=(1, 2, 3)).clamp_min(1e-6)
            top_count = int(top.shape[1])
            partial_count = int(partial.shape[1])
            top_end = top_count
            partial_end = top_end + partial_count
            section_stress = output.P_served[:, partial_end:].reshape(
                1,
                SECTION_RESULTANT_SLICE_COUNT,
                SECTION_RESULTANT_Y_POINT_COUNT,
                2,
                2,
            )
            top_loss = _traction_loss(
                output.P_served[:, :top_end],
                data["top_free_normals"][geometry_index],  # type: ignore[index]
                data["top_free_weights"][geometry_index],  # type: ignore[index]
                p_reference,
            )
            partial_loss = _partial_traction_loss(
                output.P_served[:, top_end:partial_end],
                data["partial_traction_normals"][geometry_index],  # type: ignore[index]
                data["partial_traction_directions"][geometry_index],  # type: ignore[index]
                data["partial_traction_weights"][geometry_index],  # type: ignore[index]
                p_reference,
            )
            section_weights = data["section_weights"][  # type: ignore[index]
                geometry_index
            ]
            y_max = data["y_max"][geometry_index]  # type: ignore[index]
            section_loss = _section_resultant_loss(
                section_stress,
                section_weights,
                p_reference,
                y_max,
            )
            axial = (section_stress[..., 0, 0] * section_weights).sum(dim=-1)
            axial_mean = axial.mean(dim=1, keepdim=True)
            coefficient_of_variation = (axial - axial_mean).square().mean(
                dim=1
            ).sqrt() / axial_mean.abs().squeeze(1).clamp_min(1e-6)
            family = data["meshes"][index].family  # type: ignore[index,union-attr]
            by_family[family]["traction_free_normalized_rms"].append(
                float(torch.sqrt(top_loss))
            )
            by_family[family]["partial_traction_normalized_rms"].append(
                float(torch.sqrt(partial_loss))
            )
            by_family[family]["section_resultant_normalized_rms"].append(
                float(torch.sqrt(section_loss))
            )
            by_family[family]["section_resultant_coefficient_of_variation"].append(
                float(coefficient_of_variation)
            )
    return {
        family: {name: float(np.mean(values)) for name, values in family_values.items()}
        for family, family_values in by_family.items()
    }


def run_iteration3_screen(
    upstream_checkpoint_path: Path,
    output_path: Path | None = None,
    *,
    epochs: int = 300,
    peak_supervised_p_weight: float = PEAK_SUPERVISED_P_WEIGHT,
) -> Mapping[str, Any]:
    """Compare the incumbent and one new head on training-only geometry folds."""
    import torch

    from experiments.spike.geometry import sample_geometry_cases
    from experiments.spike.model import ModelConfig

    if epochs < 1:
        raise ValueError("screen epochs must be positive")
    if peak_supervised_p_weight < 0.0:
        raise ValueError("screen peak supervised P weight must be non-negative")
    repo_root = Path(__file__).resolve().parents[2]
    target = output_path or (
        Path(__file__).resolve().parent / "artifacts" / "iteration3_screen.json"
    )
    checkpoint = _load_upstream_checkpoint(upstream_checkpoint_path)
    fem_content_hash = _fem_content_hash(repo_root)
    cache_dir = Path(tempfile.gettempdir()) / "pi-ginot-s1-legacy-cache-019fe6e9"
    cases = sample_geometry_cases(master_seed=CAMPAIGN_SEED, count_per_family=30)
    split = split_by_geometry(cases, master_seed=SPLIT_SEED)
    training_pairs: list[tuple[SpikeMesh, object]] = []
    for mesh in split.train:
        result = _solve_with_cache(mesh, cache_dir, fem_content_hash)
        if not result.converged:
            raise RuntimeError(
                f"training-only screen FEM failed for {mesh.geometry_id}: "
                f"{result.failure_reason}"
            )
        training_pairs.append((mesh, result))
    pairs = tuple(training_pairs)
    data = _tensor_dataset(pairs)
    meshes = tuple(data["meshes"])
    family_indices = {
        family: tuple(
            index for index, mesh in enumerate(meshes) if mesh.family == family
        )
        for family in ("dogbone", "centre_notched")
    }
    fold_slices = (("tail", slice(16, 20)), ("head", slice(0, 4)))
    folds: list[Mapping[str, Any]] = []
    for fold_name, development_slice in fold_slices:
        development = tuple(
            index
            for family in ("dogbone", "centre_notched")
            for index in family_indices[family][development_slice]
        )
        development_set = set(development)
        fit = tuple(
            index for index in range(len(meshes)) if index not in development_set
        )
        folds.append({"name": fold_name, "fit": fit, "development": development})

    variants: dict[str, dict[str, Any]] = {}
    parameter_counts: dict[str, tuple[int, int]] = {}
    for variant in (STRESS_HEAD_VARIANT, ITERATION3_CANDIDATE_VARIANT):
        fold_results: list[Mapping[str, Any]] = []
        for fold_index, fold in enumerate(folds):
            torch.manual_seed(TRAINING_SEED)
            model = _new_pretrained_model(
                checkpoint.state_dict,
                config=ModelConfig(stress_head_variant=variant),
            )
            total_parameters = sum(
                parameter.numel() for parameter in model.parameters()
            )
            stress_head_parameters = sum(
                parameter.numel() for parameter in model.p_head.parameters()
            )
            parameter_counts[variant] = (total_parameters, stress_head_parameters)
            started = perf_counter()
            final_loss = _train_fixed_epochs(
                model,
                data,
                fold["fit"],
                learning_rate=1e-3,
                epochs=epochs,
                random_seed=TRAINING_SEED + 300 + fold_index,
                label=f"iteration3 {variant} fold={fold['name']}",
                peak_supervised_p_weight=peak_supervised_p_weight,
            )
            training_seconds = perf_counter() - started
            development_pairs = tuple(pairs[index] for index in fold["development"])
            evaluation = _evaluate_model(model, development_pairs)
            fold_result = {
                "fold": fold["name"],
                "fit_geometry_ids": [
                    meshes[index].geometry_id for index in fold["fit"]
                ],
                "development_geometry_ids": [
                    meshes[index].geometry_id for index in fold["development"]
                ],
                "final_hybrid_loss": final_loss,
                "training_seconds": training_seconds,
                "family_metrics": dict(evaluation.family_metrics),
                "per_case_metrics": dict(evaluation.per_case_metrics),
            }
            fold_results.append(fold_result)
            print(
                f"SCREEN_RESULT variant={variant} fold={fold['name']} "
                f"score={max(metric['relative_l2_P'] for metric in evaluation.family_metrics.values()):.8f}",
                flush=True,
            )
        summary = dict(_summarize_screen_folds(fold_results))
        variants[variant] = {
            **summary,
            "folds": fold_results,
            "stress_head_parameter_count": parameter_counts[variant][1],
            "total_parameter_count": parameter_counts[variant][0],
            "peak_supervised_P_weight": peak_supervised_p_weight,
            "guardrails_passed": None,
        }

    baseline = variants[STRESS_HEAD_VARIANT]
    candidate = variants[ITERATION3_CANDIDATE_VARIANT]
    baseline_total = parameter_counts[STRESS_HEAD_VARIANT][0]
    candidate_total, candidate_head = parameter_counts[ITERATION3_CANDIDATE_VARIANT]
    checks = dict(
        _screen_guardrail_checks(
            baseline,
            candidate,
            stress_head_parameter_count=candidate_head,
            total_parameter_growth=(candidate_total - baseline_total) / baseline_total,
        )
    )
    paired_p_improvement = all(
        candidate["folds"][fold_index]["family_metrics"][family]["relative_l2_P"]
        < baseline["folds"][fold_index]["family_metrics"][family]["relative_l2_P"]
        for fold_index in range(len(folds))
        for family in ("dogbone", "centre_notched")
    )
    checks["P_improves_in_every_family_fold"] = paired_p_improvement
    checks["all_passed"] = all(
        value for name, value in checks.items() if name != "all_passed"
    )
    candidate["guardrail_checks"] = checks
    candidate["guardrails_passed"] = checks["all_passed"]
    selected_variant = (
        ITERATION3_CANDIDATE_VARIANT if checks["all_passed"] else STRESS_HEAD_VARIANT
    )
    created_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    source_hashes = {
        name: _file_sha256(Path(__file__).resolve().parent / name)
        for name in ("model.py", "run.py")
    }
    screen_document: dict[str, Any] = {
        "schema_version": "1.0",
        "step": "S1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "created_utc": created_utc,
        "training_code_sha256": source_hashes,
        "upstream_checkpoint_sha256": checkpoint.sha256,
        "fem_content_sha256": fem_content_hash,
        "architecture_iteration": {
            "iteration": 3,
            "hypothesis": (
                "explicit dimensionless load and normalized query coordinates might "
                "improve the S1 direct-P adapter"
            ),
            "decision": (
                "promote fully_conditioned_mlp"
                if selected_variant == ITERATION3_CANDIDATE_VARIANT
                else "candidate rejected by the training-only guardrails; retain "
                f"{STRESS_HEAD_VARIANT}"
            ),
            "selected_variant": selected_variant,
            "selection": {
                "data_policy": "existing 40 training geometries only",
                "training_geometry_count": 40,
                "held_out_geometry_accessed": False,
                "fold_count": len(folds),
                "development_geometries_per_family_per_fold": 4,
                "epochs_per_candidate_fold": epochs,
                "learning_rate": 1e-3,
                "peak_supervised_P_weight": peak_supervised_p_weight,
                "primary_score": "worst two-fold-mean family relative_l2_P",
                "variants": variants,
                "guardrails": (
                    ">=10% primary-score improvement; P improves in every family/fold; "
                    "family P/u, worst u, peak, parameter, and inference guardrails"
                ),
            },
            "iteration_2_held_out_already_observed": {
                "dogbone": {
                    "relative_l2_u": 0.032613624872333394,
                    "relative_l2_P": 0.10918577843692369,
                    "worst_relative_peak_stress_error": 0.4201781577070775,
                },
                "centre_notched": {
                    "relative_l2_u": 0.03956596460515004,
                    "relative_l2_P": 0.09439925011793354,
                    "worst_relative_peak_stress_error": 0.4539310459455932,
                },
            },
            "held_out_evaluation_status": "confirmatory_not_pristine",
            "held_out_caveat": (
                "prior iterations already observed the 20-case aggregate; this screen "
                "uses no held-out FEM fields and later evaluation remains confirmatory"
            ),
        },
    }
    write_json_strict(target, screen_document)
    print(f"SCREEN_WROTE {target}", flush=True)
    return screen_document


def run_iteration4_screen(
    upstream_checkpoint_path: Path,
    output_path: Path | None = None,
    *,
    stages: Sequence[FineTuneStage] = ITERATION4_FINE_TUNE_STAGES,
) -> Mapping[str, Any]:
    """Compare staged fine-tuning with sealed iteration-3 training-only evidence."""
    import torch

    from experiments.spike.geometry import sample_geometry_cases
    from experiments.spike.model import ModelConfig

    declared_stages = tuple(stages)
    staged_epochs = sum(stage.epochs for stage in declared_stages)
    if output_path is None and declared_stages != ITERATION4_FINE_TUNE_STAGES:
        raise ValueError("official iteration-4 screen requires preregistered stages")
    baseline_evidence = _architecture_iteration_evidence()
    baseline_selection = _require_mapping(
        baseline_evidence.get("selection"), "iteration-3 selection"
    )
    if baseline_selection.get("held_out_geometry_accessed") is not False:
        raise ValueError("sealed baseline must not access held-out FEM fields")
    baseline_variants = _require_mapping(
        baseline_selection.get("variants"), "iteration-3 variants"
    )
    baseline = dict(
        _require_mapping(
            baseline_variants.get(STRESS_HEAD_VARIANT),
            f"iteration-3 variants.{STRESS_HEAD_VARIANT}",
        )
    )
    repo_root = Path(__file__).resolve().parents[2]
    target = output_path or (
        Path(__file__).resolve().parent / "artifacts" / "iteration4_screen.json"
    )
    checkpoint = _load_upstream_checkpoint(upstream_checkpoint_path)
    fem_content_hash = _fem_content_hash(repo_root)
    cache_dir = Path(tempfile.gettempdir()) / "pi-ginot-s1-legacy-cache-019fe6e9"
    cases = sample_geometry_cases(master_seed=CAMPAIGN_SEED, count_per_family=30)
    split = split_by_geometry(cases, master_seed=SPLIT_SEED)
    training_pairs: list[tuple[SpikeMesh, object]] = []
    for mesh in split.train:
        result = _solve_with_cache(mesh, cache_dir, fem_content_hash)
        if not result.converged:
            raise RuntimeError(
                f"iteration-4 training FEM failed for {mesh.geometry_id}: "
                f"{result.failure_reason}"
            )
        training_pairs.append((mesh, result))
    pairs = tuple(training_pairs)
    data = _tensor_dataset(pairs)
    meshes = tuple(data["meshes"])
    family_indices = {
        family: tuple(
            index for index, mesh in enumerate(meshes) if mesh.family == family
        )
        for family in ("dogbone", "centre_notched")
    }
    folds: list[Mapping[str, Any]] = []
    for fold_name, development_slice in (
        ("tail", slice(16, 20)),
        ("head", slice(0, 4)),
    ):
        development = tuple(
            index
            for family in ("dogbone", "centre_notched")
            for index in family_indices[family][development_slice]
        )
        development_set = set(development)
        fit = tuple(
            index for index in range(len(meshes)) if index not in development_set
        )
        folds.append({"name": fold_name, "fit": fit, "development": development})

    candidate_folds: list[Mapping[str, Any]] = []
    parameter_counts: tuple[int, int] | None = None
    for fold_index, fold in enumerate(folds):
        torch.manual_seed(TRAINING_SEED)
        model = _new_pretrained_model(
            checkpoint.state_dict,
            config=ModelConfig(stress_head_variant=STRESS_HEAD_VARIANT),
        )
        measured_counts = (
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in model.p_head.parameters()),
        )
        if parameter_counts is None:
            parameter_counts = measured_counts
        elif parameter_counts != measured_counts:
            raise RuntimeError("iteration-4 folds changed model parameter counts")
        started = perf_counter()
        training_trace = _train_staged_epochs(
            model,
            data,
            fold["fit"],
            stages=declared_stages,
            random_seed=TRAINING_SEED + 300 + fold_index,
            label=f"iteration4 staged_adam fold={fold['name']}",
            peak_supervised_p_weight=0.0,
        )
        training_seconds = perf_counter() - started
        development_pairs = tuple(pairs[index] for index in fold["development"])
        evaluation = _evaluate_model(model, development_pairs)
        candidate_folds.append(
            {
                "fold": fold["name"],
                "fit_geometry_ids": [
                    meshes[index].geometry_id for index in fold["fit"]
                ],
                "development_geometry_ids": [
                    meshes[index].geometry_id for index in fold["development"]
                ],
                "training": dict(training_trace),
                "training_seconds": training_seconds,
                "family_metrics": dict(evaluation.family_metrics),
                "per_case_metrics": dict(evaluation.per_case_metrics),
            }
        )
        print(
            f"ITERATION4_RESULT protocol=staged_adam fold={fold['name']} "
            f"score={max(metric['relative_l2_P'] for metric in evaluation.family_metrics.values()):.8f}",
            flush=True,
        )
    if parameter_counts is None:
        raise RuntimeError("iteration-4 screen created no models")
    candidate: dict[str, Any] = {
        **dict(_summarize_screen_folds(candidate_folds)),
        "folds": candidate_folds,
        "total_parameter_count": parameter_counts[0],
        "stress_head_parameter_count": parameter_counts[1],
        "peak_supervised_P_weight": 0.0,
        "guardrails_passed": None,
    }
    if len(baseline.get("folds", ())) != len(folds):
        raise ValueError("sealed baseline has incompatible development folds")
    for fold_index, fold in enumerate(folds):
        baseline_fold = _require_mapping(
            baseline["folds"][fold_index], f"sealed baseline fold {fold_index}"
        )
        expected_ids = [meshes[index].geometry_id for index in fold["development"]]
        if (
            baseline_fold.get("fold") != fold["name"]
            or baseline_fold.get("development_geometry_ids") != expected_ids
        ):
            raise ValueError(
                "sealed baseline development folds do not match iteration 4"
            )
    protocols = {"joint_adam": baseline, "staged_adam": candidate}
    checks = dict(
        _screen_guardrail_checks(
            baseline,
            candidate,
            stress_head_parameter_count=parameter_counts[1],
            total_parameter_growth=0.0,
        )
    )
    checks["P_improves_in_every_family_fold"] = all(
        candidate["folds"][fold_index]["family_metrics"][family]["relative_l2_P"]
        < baseline["folds"][fold_index]["family_metrics"][family]["relative_l2_P"]
        for fold_index in range(len(folds))
        for family in ("dogbone", "centre_notched")
    )
    checks["all_passed"] = all(
        value for name, value in checks.items() if name != "all_passed"
    )
    candidate["guardrail_checks"] = checks
    candidate["guardrails_passed"] = checks["all_passed"]
    selected_protocol = "staged_adam" if checks["all_passed"] else "joint_adam"
    created_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    screen_document: dict[str, Any] = {
        "schema_version": "1.0",
        "step": "S1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "created_utc": created_utc,
        "training_code_sha256": {
            name: _file_sha256(Path(__file__).resolve().parent / name)
            for name in ("model.py", "run.py")
        },
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_checkpoint_sha256": checkpoint.sha256,
        "fem_content_sha256": fem_content_hash,
        "training_protocol_iteration": {
            "iteration": 4,
            "hypothesis": (
                "joint high-learning-rate fine-tuning damages useful pretrained "
                "PI-GINOT representations before the new direct-P head adapts"
            ),
            "selected_protocol": selected_protocol,
            "decision": (
                "promote staged_adam"
                if selected_protocol == "staged_adam"
                else "candidate rejected by training-only guardrails; retain joint_adam"
            ),
            "upstream_implementation_audit": {
                "public_head_matches_pinned_commit": True,
                "vendored_source_semantically_identical": True,
                "displacement_uses_upstream_forward": True,
                "checkpoint_runtime_loaded": True,
                "s1_only_extension": "direct first-Piola stress head",
            },
            "selection": {
                "data_policy": "existing 40 training geometries only",
                "training_geometry_count": 40,
                "held_out_geometry_accessed": False,
                "fold_count": len(folds),
                "development_geometries_per_family_per_fold": 4,
                "baseline_epochs": 300,
                "staged_epochs": staged_epochs,
                "same_epoch_budget": staged_epochs == 300,
                "common_random_seed_and_sampling_stream": True,
                "architecture": STRESS_HEAD_VARIANT,
                "loss_weights_changed": False,
                "fem_labels_changed": False,
                "protocols": protocols,
                "guardrails": (
                    ">=10% primary-score improvement; P improves in every family/fold; "
                    "family P/u, worst u, peak, parameter, and inference guardrails"
                ),
            },
            "held_out_evaluation_status": "not_accessed_for_iteration4_selection",
        },
    }
    write_json_strict(target, screen_document)
    print(f"ITERATION4_WROTE {target}", flush=True)
    return screen_document


def _run_natural_physics_screen(
    upstream_checkpoint_path: Path,
    output_path: Path,
    *,
    epochs: int,
    iteration: int,
    objective_name: str,
    physics_weights: ServedStressPhysicsWeights,
    hypothesis: str,
    enforce_hashes: bool,
) -> Mapping[str, Any]:
    """Screen upstream-style natural-boundary physics on training folds only."""
    if epochs < 1:
        raise ValueError("screen epochs must be positive")
    import torch

    from experiments.spike.geometry import sample_geometry_cases
    from experiments.spike.model import ModelConfig

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    screen_started = perf_counter()
    baseline_evidence = _architecture_iteration_evidence()
    baseline_selection = _require_mapping(
        baseline_evidence.get("selection"), "iteration-3 selection"
    )
    if baseline_selection.get("held_out_geometry_accessed") is not False:
        raise ValueError("sealed baseline must not access held-out FEM fields")
    baseline_variants = _require_mapping(
        baseline_selection.get("variants"), "iteration-3 variants"
    )
    baseline = dict(
        _require_mapping(
            baseline_variants.get(STRESS_HEAD_VARIANT),
            f"iteration-3 variants.{STRESS_HEAD_VARIANT}",
        )
    )
    baseline_folds = baseline.get("folds")
    if not isinstance(baseline_folds, list) or len(baseline_folds) != 2:
        raise ValueError("sealed baseline must contain exactly two folds")

    repo_root = Path(__file__).resolve().parents[2]
    target = output_path
    checkpoint = _load_upstream_checkpoint(upstream_checkpoint_path)
    fem_content_hash = _fem_content_hash(repo_root)
    if enforce_hashes:
        if checkpoint.sha256 != UPSTREAM_CHECKPOINT_SHA256:
            raise ValueError(
                f"iteration-{iteration} checkpoint does not match preregistration"
            )
        if fem_content_hash != S1_FEM_CONTENT_SHA256:
            raise ValueError(
                f"iteration-{iteration} FEM content does not match preregistration"
            )
    cache_dir = Path(tempfile.gettempdir()) / "pi-ginot-s1-legacy-cache-019fe6e9"
    cases = sample_geometry_cases(master_seed=CAMPAIGN_SEED, count_per_family=30)
    split = split_by_geometry(cases, master_seed=SPLIT_SEED)
    training_pairs: list[tuple[SpikeMesh, object]] = []
    for mesh in split.train:
        result = _solve_with_cache(mesh, cache_dir, fem_content_hash)
        if not result.converged:
            raise RuntimeError(
                f"iteration-{iteration} training FEM failed for {mesh.geometry_id}: "
                f"{result.failure_reason}"
            )
        training_pairs.append((mesh, result))
    pairs = tuple(training_pairs)
    data = _tensor_dataset(pairs)
    meshes = tuple(data["meshes"])
    index_by_geometry_id = {
        mesh.geometry_id: index for index, mesh in enumerate(meshes)
    }
    if len(index_by_geometry_id) != 40:
        raise ValueError(
            f"iteration-{iteration} requires 40 unique training geometries"
        )

    folds: list[Mapping[str, Any]] = []
    for expected_name, raw_fold in zip(("tail", "head"), baseline_folds, strict=True):
        baseline_fold = _require_mapping(raw_fold, f"sealed baseline {expected_name}")
        fit_ids = baseline_fold.get("fit_geometry_ids")
        development_ids = baseline_fold.get("development_geometry_ids")
        if (
            baseline_fold.get("fold") != expected_name
            or not isinstance(fit_ids, list)
            or not isinstance(development_ids, list)
        ):
            raise ValueError("sealed baseline fold definition is invalid")
        if (
            len(fit_ids) != 32
            or len(development_ids) != 8
            or set(fit_ids).intersection(development_ids)
            or set(fit_ids).union(development_ids) != set(index_by_geometry_id)
        ):
            raise ValueError("sealed baseline fold geometry IDs do not partition train")
        folds.append(
            {
                "name": expected_name,
                "fit": tuple(index_by_geometry_id[item] for item in fit_ids),
                "development": tuple(
                    index_by_geometry_id[item] for item in development_ids
                ),
                "baseline": baseline_fold,
            }
        )

    candidate_folds: list[Mapping[str, Any]] = []
    parameter_counts: tuple[int, int] | None = None
    early_stop_reason: str | None = None
    for fold_index, fold in enumerate(folds):
        torch.manual_seed(TRAINING_SEED)
        model = _new_pretrained_model(
            checkpoint.state_dict,
            config=ModelConfig(stress_head_variant=STRESS_HEAD_VARIANT),
        )
        measured_counts = (
            sum(parameter.numel() for parameter in model.parameters()),
            sum(parameter.numel() for parameter in model.p_head.parameters()),
        )
        if parameter_counts is None:
            parameter_counts = measured_counts
        elif parameter_counts != measured_counts:
            raise RuntimeError(
                f"iteration-{iteration} folds changed model parameter counts"
            )
        baseline_counts = (
            int(baseline["total_parameter_count"]),
            int(baseline["stress_head_parameter_count"]),
        )
        if measured_counts != baseline_counts:
            raise ValueError(
                f"iteration-{iteration} changed the sealed model parameter counts"
            )
        started = perf_counter()
        final_loss = _train_fixed_epochs(
            model,
            data,
            fold["fit"],
            learning_rate=1e-3,
            epochs=epochs,
            random_seed=TRAINING_SEED + 300 + fold_index,
            label=f"iteration{iteration} {objective_name} fold={fold['name']}",
            peak_supervised_p_weight=0.0,
            served_stress_physics=physics_weights,
            max_seconds=3600.0,
        )
        training_seconds = perf_counter() - started
        development_pairs = tuple(pairs[index] for index in fold["development"])
        evaluation = _evaluate_model(model, development_pairs)
        diagnostics = _served_stress_physics_diagnostics(
            model,
            data,
            fold["development"],
        )
        fold_result = {
            "fold": fold["name"],
            "fit_geometry_ids": [meshes[index].geometry_id for index in fold["fit"]],
            "development_geometry_ids": [
                meshes[index].geometry_id for index in fold["development"]
            ],
            "final_hybrid_loss": final_loss,
            "training_seconds": training_seconds,
            "family_metrics": dict(evaluation.family_metrics),
            "per_case_metrics": dict(evaluation.per_case_metrics),
            "served_stress_physics_diagnostics": diagnostics,
        }
        candidate_folds.append(fold_result)
        print(
            f"ITERATION{iteration}_RESULT "
            f"objective={objective_name} fold={fold['name']} "
            f"score={max(metric['relative_l2_P'] for metric in evaluation.family_metrics.values()):.8f}",
            flush=True,
        )
        if fold_index == 0:
            baseline_fold = fold["baseline"]
            paired_p_failed = any(
                evaluation.family_metrics[family]["relative_l2_P"]
                >= baseline_fold["family_metrics"][family]["relative_l2_P"]
                for family in ("dogbone", "centre_notched")
            )
            worst_u_failed = any(
                case["relative_l2_u"]
                > float(baseline["worst_single_case_relative_l2_u"]) + 0.01
                for case in evaluation.per_case_metrics.values()
            )
            worst_peak_failed = any(
                case["relative_peak_stress_error"]
                > float(baseline["overall_worst_peak_stress_error"]) + 0.05
                for case in evaluation.per_case_metrics.values()
            )
            if paired_p_failed or worst_u_failed or worst_peak_failed:
                early_stop_reason = (
                    "tail fold makes promotion impossible under preregistered gates"
                )
                break
        if perf_counter() - screen_started > 9000.0:
            raise TimeoutError(
                f"iteration-{iteration} end-to-end screen time cap exceeded"
            )

    if parameter_counts is None:
        raise RuntimeError(f"iteration-{iteration} screen created no models")
    candidate: dict[str, Any]
    checks: dict[str, bool]
    if early_stop_reason is not None:
        checks = {
            "primary_score_improved_by_at_least_10_percent": False,
            "family_P_within_absolute_0_005": False,
            "family_u_within_absolute_0_005": False,
            "worst_case_u_within_absolute_0_01": False,
            "worst_peak_within_absolute_0_05": False,
            "median_peak_within_absolute_0_02": False,
            "stress_head_parameters_at_most_10000": parameter_counts[1] <= 10_000,
            "total_model_growth_below_1_percent": True,
            "mean_inference_within_1_25x": False,
            "P_improves_in_every_family_fold": False,
            "parameter_counts_unchanged": parameter_counts
            == (
                int(baseline["total_parameter_count"]),
                int(baseline["stress_head_parameter_count"]),
            ),
            "all_passed": False,
        }
        candidate = {
            "folds": candidate_folds,
            "early_stop_reason": early_stop_reason,
            "total_parameter_count": parameter_counts[0],
            "stress_head_parameter_count": parameter_counts[1],
            "peak_supervised_P_weight": 0.0,
            "guardrail_checks": checks,
            "guardrails_passed": False,
        }
    else:
        candidate = {
            **dict(_summarize_screen_folds(candidate_folds)),
            "folds": candidate_folds,
            "total_parameter_count": parameter_counts[0],
            "stress_head_parameter_count": parameter_counts[1],
            "peak_supervised_P_weight": 0.0,
            "guardrails_passed": None,
        }
        checks = dict(
            _screen_guardrail_checks(
                baseline,
                candidate,
                stress_head_parameter_count=parameter_counts[1],
                total_parameter_growth=0.0,
            )
        )
        checks["P_improves_in_every_family_fold"] = all(
            candidate["folds"][fold_index]["family_metrics"][family]["relative_l2_P"]
            < baseline["folds"][fold_index]["family_metrics"][family]["relative_l2_P"]
            for fold_index in range(2)
            for family in ("dogbone", "centre_notched")
        )
        checks["parameter_counts_unchanged"] = parameter_counts == (
            int(baseline["total_parameter_count"]),
            int(baseline["stress_head_parameter_count"]),
        )
        checks["all_passed"] = all(
            value for name, value in checks.items() if name != "all_passed"
        )
        candidate["guardrail_checks"] = checks
        candidate["guardrails_passed"] = checks["all_passed"]

    diagnostic_limits = {
        "dogbone": {"traction_free_normalized_rms": 0.10},
        "centre_notched": {"traction_free_normalized_rms": 0.18},
    }
    diagnostic_checks = {
        "top_traction_within_training_label_floor_envelope": all(
            fold["served_stress_physics_diagnostics"][family][
                "traction_free_normalized_rms"
            ]
            <= diagnostic_limits[family]["traction_free_normalized_rms"]
            for fold in candidate_folds
            for family in ("dogbone", "centre_notched")
        ),
        "section_resultant_coefficient_of_variation_at_most_0_05": all(
            fold["served_stress_physics_diagnostics"][family][
                "section_resultant_coefficient_of_variation"
            ]
            <= 0.05
            for fold in candidate_folds
            for family in ("dogbone", "centre_notched")
        ),
    }
    candidate["physics_diagnostic_checks"] = diagnostic_checks
    selected_objective = (
        objective_name if checks["all_passed"] else "iteration3_joint_adam"
    )
    preregistration = {
        "architecture": STRESS_HEAD_VARIANT,
        "optimizer": "Adam(lr=1e-3, betas=(0.9,0.999), eps=1e-8)",
        "epochs_per_fold": epochs,
        "supervised_queries": SUPERVISED_QUERY_COUNT,
        "interior_queries": PHYSICS_QUERY_COUNT,
        "top_free_queries": int(data["top_free_points"].shape[1]),  # type: ignore[union-attr]
        "partial_traction_queries": int(
            data["partial_traction_points"].shape[1]  # type: ignore[union-attr]
        ),
        "section_slices": SECTION_RESULTANT_SLICE_COUNT,
        "section_y_points": SECTION_RESULTANT_Y_POINT_COUNT,
        "loss_weights": {
            "supervised_u": 1.0,
            "supervised_P": 1.0,
            "equilibrium_on_P_served": 1.0,
            "constitutive_in_plane": 1.0,
            "traction_free_on_P_served": physics_weights.traction_free,
            "partial_traction_on_P_served": physics_weights.partial_traction,
            "section_resultant_on_P_served": physics_weights.section_resultant,
            "peak_supervised_P": 0.0,
        },
        "no_fem_reaction_or_elastic_force_anchor": True,
    }
    preregistration_sha256 = hashlib.sha256(
        json.dumps(preregistration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    screen_document: dict[str, Any] = {
        "schema_version": "1.0",
        "step": "S1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "training_code_sha256": {
            name: _file_sha256(Path(__file__).resolve().parent / name)
            for name in ("model.py", "run.py")
        },
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_checkpoint_sha256": checkpoint.sha256,
        "fem_content_sha256": fem_content_hash,
        "physics_objective_iteration": {
            "iteration": iteration,
            "hypothesis": hypothesis,
            "selected_objective": selected_objective,
            "decision": (
                f"promote {objective_name}"
                if checks["all_passed"]
                else "candidate rejected by training-only guardrails; retain iteration 3"
            ),
            "upstream_implementation_review": {
                "encoder_and_displacement_decoder_exact_from_pinned_upstream": True,
                "served_stress_is_direct_per_binding_S1_requirement": True,
                "top_and_complementary_traction_match_upstream_reference_form": True,
                "section_force_is_label_free": True,
                "dogbone_specific_elastic_anchor_excluded_for_cross_family_validity": True,
                "adaptive_weighting_excluded_to_isolate_one_objective_change": True,
                "physics_weight_scale_relative_to_iteration5": (
                    0.1 if iteration == 6 else 1.0
                ),
                **(
                    {
                        "training_only_label_postmortem": {
                            "dogbone_top_traction_normalized_rms": 0.08127,
                            "centre_notched_top_traction_normalized_rms": 0.17464,
                            "dogbone_section_normalized_rms": 0.004435,
                            "centre_notched_section_normalized_rms": 0.002497,
                        }
                    }
                    if iteration == 6
                    else {}
                ),
            },
            "preregistration": preregistration,
            "preregistration_sha256": preregistration_sha256,
            "selection": {
                "data_policy": "existing 40 training geometries only",
                "training_geometry_count": 40,
                "held_out_geometry_accessed": False,
                "fold_count": len(candidate_folds),
                "development_geometries_per_family_per_fold": 4,
                "sealed_baseline_sha256": ITERATION3_SCREEN_SHA256,
                **(
                    {
                        "prior_iteration_evidence_sealed": enforce_hashes,
                        **(
                            {
                                "sealed_rejected_iteration5_sha256": (
                                    ITERATION5_SCREEN_SHA256
                                )
                            }
                            if enforce_hashes
                            else {}
                        ),
                    }
                    if iteration == 6
                    else {}
                ),
                "objectives": {
                    "iteration3_joint_adam": baseline,
                    objective_name: candidate,
                },
                "guardrails": checks,
            },
            "held_out_evaluation_status": (
                f"not_accessed_for_iteration{iteration}_selection"
            ),
        },
    }
    write_json_strict(target, screen_document)
    print(f"ITERATION{iteration}_WROTE {target}", flush=True)
    return screen_document


def run_iteration5_screen(
    upstream_checkpoint_path: Path,
    output_path: Path | None = None,
    *,
    epochs: int = 300,
) -> Mapping[str, Any]:
    """Run the sealed strong upstream-physics screen."""
    if epochs < 1:
        raise ValueError("screen epochs must be positive")
    if output_path is None and epochs != 300:
        raise ValueError(
            "official iteration-5 screen requires preregistered 300 epochs"
        )
    target = output_path or (
        Path(__file__).resolve().parent / "artifacts" / "iteration5_screen.json"
    )
    return _run_natural_physics_screen(
        upstream_checkpoint_path,
        target,
        epochs=epochs,
        iteration=5,
        objective_name="natural_bc_and_section_balance",
        physics_weights=ITERATION5_PHYSICS_WEIGHTS,
        hypothesis=(
            "missing upstream reference-frame natural-boundary and axial-section "
            "constraints cause the remaining direct served-stress error"
        ),
        enforce_hashes=output_path is None,
    )


def run_iteration6_screen(
    upstream_checkpoint_path: Path,
    output_path: Path | None = None,
    *,
    epochs: int = 300,
) -> Mapping[str, Any]:
    """Run one weak-physics correction after the rejected strong objective."""
    if epochs < 1:
        raise ValueError("screen epochs must be positive")
    if output_path is None and epochs != 300:
        raise ValueError(
            "official iteration-6 screen requires preregistered 300 epochs"
        )
    evidence_grade = output_path is None or epochs == 300
    if evidence_grade:
        _iteration5_rejection_evidence()
    target = output_path or (
        Path(__file__).resolve().parent / "artifacts" / "iteration6_screen.json"
    )
    return _run_natural_physics_screen(
        upstream_checkpoint_path,
        target,
        epochs=epochs,
        iteration=6,
        objective_name="weak_upstream_physics_0p1x",
        physics_weights=ITERATION6_PHYSICS_WEIGHTS,
        hypothesis=(
            "uniformly reducing the upstream-style physics terms by ten restores "
            "compatibility with supervised nodal recovered stress while retaining "
            "useful boundary and section regularization"
        ),
        enforce_hashes=evidence_grade,
    )


def _historical_control_metric_drift(
    control: Mapping[str, Any],
    sealed_fold: Mapping[str, Any],
) -> float:
    """Return diagnostic metric drift across the historical code-hash boundary."""
    sealed_metrics = _require_mapping(
        sealed_fold.get("family_metrics"), "sealed iteration-3 family metrics"
    )
    measured_metrics = _require_mapping(
        control.get("family_metrics"), "iteration-7 control family metrics"
    )
    differences: list[float] = []
    for family in ("dogbone", "centre_notched"):
        sealed = _require_mapping(sealed_metrics.get(family), f"sealed {family}")
        measured = _require_mapping(measured_metrics.get(family), f"control {family}")
        differences.extend(
            (
                abs(
                    float(measured["legacy_nodal_relative_l2_P"])
                    - float(sealed["relative_l2_P"])
                ),
                abs(float(measured["relative_l2_u"]) - float(sealed["relative_l2_u"])),
                abs(
                    float(measured["worst_relative_peak_stress_error"])
                    - float(sealed["worst_relative_peak_stress_error"])
                ),
            )
        )
    return max(differences)


def _tail_element_screen_passes(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    """Fail closed before the second fold when promotion is already impossible."""
    control_family = control["family_metrics"]
    candidate_family = candidate["family_metrics"]
    family_checks = all(
        float(candidate_family[family]["relative_l2_P"])
        < float(control_family[family]["relative_l2_P"])
        and float(candidate_family[family]["legacy_nodal_relative_l2_P"])
        <= float(control_family[family]["legacy_nodal_relative_l2_P"]) + 0.005
        and float(candidate_family[family]["relative_l2_u"])
        <= float(control_family[family]["relative_l2_u"]) + 0.005
        for family in ("dogbone", "centre_notched")
    )
    candidate_cases = tuple(candidate["per_case_metrics"].values())
    control_cases = tuple(control["per_case_metrics"].values())
    worst_u = max(float(case["relative_l2_u"]) for case in candidate_cases)
    control_worst_u = max(float(case["relative_l2_u"]) for case in control_cases)
    peak_errors = tuple(
        float(case["relative_peak_stress_error"]) for case in candidate_cases
    )
    control_peak_errors = tuple(
        float(case["relative_peak_stress_error"]) for case in control_cases
    )
    control_inference = np.mean(
        [
            float(control_family[family]["mean_inference_seconds"])
            for family in ("dogbone", "centre_notched")
        ]
    )
    candidate_inference = np.mean(
        [
            float(candidate_family[family]["mean_inference_seconds"])
            for family in ("dogbone", "centre_notched")
        ]
    )
    return bool(
        family_checks
        and worst_u <= control_worst_u + 0.01
        and max(peak_errors) <= max(control_peak_errors) + 0.05
        and float(np.median(peak_errors))
        <= float(np.median(control_peak_errors)) + 0.02
        and candidate_inference <= 1.25 * control_inference
    )


def run_iteration7_screen(
    upstream_checkpoint_path: Path,
    output_path: Path | None = None,
    *,
    epochs: int = 300,
) -> Mapping[str, Any]:
    """Compare nodal-patch and raw-element P supervision on training folds."""
    import torch

    from experiments.spike.geometry import sample_geometry_cases
    from experiments.spike.model import ModelConfig

    if epochs < 1:
        raise ValueError("screen epochs must be positive")
    if output_path is None and epochs != 300:
        raise ValueError(
            "official iteration-7 screen requires preregistered 300 epochs"
        )
    evidence_grade = output_path is None or epochs == 300
    if evidence_grade:
        _iteration6_rejection_evidence()
        control_audit_path = (
            Path(__file__).resolve().parent
            / "artifacts"
            / "iteration7_control_audit.json"
        )
        if (
            not control_audit_path.is_file()
            or _file_sha256(control_audit_path) != ITERATION7_CONTROL_AUDIT_SHA256
        ):
            raise ValueError(
                "iteration-7 control audit SHA-256 does not match sealed evidence"
            )

    screen_started = perf_counter()
    repo_root = Path(__file__).resolve().parents[2]
    target = output_path or (
        Path(__file__).resolve().parent / "artifacts" / "iteration7_screen.json"
    )
    checkpoint = _load_upstream_checkpoint(upstream_checkpoint_path)
    baseline_evidence = _architecture_iteration_evidence()
    baseline_selection = _require_mapping(
        baseline_evidence.get("selection"), "sealed iteration-3 selection"
    )
    baseline_variants = _require_mapping(
        baseline_selection.get("variants"), "sealed iteration-3 variants"
    )
    sealed_baseline = _require_mapping(
        baseline_variants.get(STRESS_HEAD_VARIANT), "sealed conditioned_mlp"
    )
    sealed_folds = sealed_baseline.get("folds")
    if not isinstance(sealed_folds, list) or len(sealed_folds) != 2:
        raise ValueError("sealed iteration-3 baseline must contain two folds")

    fem_content_hash = _fem_content_hash(repo_root)
    if evidence_grade and fem_content_hash != S1_FEM_CONTENT_SHA256:
        raise ValueError("iteration-7 FEM content differs from the sealed S1 snapshot")
    cache_dir = Path(tempfile.gettempdir()) / "pi-ginot-s1-legacy-cache-019fe6e9"
    cases = sample_geometry_cases(master_seed=CAMPAIGN_SEED, count_per_family=30)
    split = split_by_geometry(cases, master_seed=SPLIT_SEED)
    training_pairs: list[tuple[SpikeMesh, object]] = []
    for mesh in split.train:
        result = _solve_with_cache(mesh, cache_dir, fem_content_hash)
        if not result.converged:
            raise RuntimeError(
                f"iteration-7 training FEM failed for {mesh.geometry_id}: "
                f"{result.failure_reason}"
            )
        training_pairs.append((mesh, result))
    pairs = tuple(training_pairs)
    data = _element_tensor_dataset(pairs)
    meshes = tuple(data["meshes"])
    index_by_geometry_id = {
        mesh.geometry_id: index for index, mesh in enumerate(meshes)
    }
    if len(index_by_geometry_id) != 40:
        raise ValueError("iteration-7 requires 40 unique training geometries")

    folds: list[Mapping[str, Any]] = []
    for expected_name, raw_fold in zip(("tail", "head"), sealed_folds, strict=True):
        sealed_fold = _require_mapping(raw_fold, f"sealed {expected_name} fold")
        fit_ids = sealed_fold.get("fit_geometry_ids")
        development_ids = sealed_fold.get("development_geometry_ids")
        if (
            sealed_fold.get("fold") != expected_name
            or not isinstance(fit_ids, list)
            or not isinstance(development_ids, list)
            or len(fit_ids) != 32
            or len(development_ids) != 8
            or set(fit_ids).intersection(development_ids)
            or set(fit_ids).union(development_ids) != set(index_by_geometry_id)
        ):
            raise ValueError("sealed iteration-3 fold does not partition training data")
        folds.append(
            {
                "name": expected_name,
                "fit": tuple(index_by_geometry_id[item] for item in fit_ids),
                "development": tuple(
                    index_by_geometry_id[item] for item in development_ids
                ),
                "sealed": sealed_fold,
            }
        )

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    objective_names = ("nodal_patch_control", "element_quadrature_P_supervision")
    fold_results: dict[str, list[Mapping[str, Any]]] = {
        name: [] for name in objective_names
    }
    historical_control_drifts: list[float] = []
    parameter_counts: tuple[int, int] | None = None
    early_stop_reason: str | None = None
    for fold_index, fold in enumerate(folds):
        for objective_name in objective_names:
            torch.manual_seed(TRAINING_SEED)
            model = _new_pretrained_model(
                checkpoint.state_dict,
                config=ModelConfig(stress_head_variant=STRESS_HEAD_VARIANT),
            )
            measured_counts = (
                sum(parameter.numel() for parameter in model.parameters()),
                sum(parameter.numel() for parameter in model.p_head.parameters()),
            )
            if parameter_counts is None:
                parameter_counts = measured_counts
            elif parameter_counts != measured_counts:
                raise RuntimeError("iteration-7 changed parameter counts between runs")
            expected_counts = (
                int(sealed_baseline["total_parameter_count"]),
                int(sealed_baseline["stress_head_parameter_count"]),
            )
            if measured_counts != expected_counts:
                raise ValueError("iteration-7 model differs from the sealed incumbent")

            training_arguments: dict[str, Any] = {}
            if objective_name == "element_quadrature_P_supervision":
                training_arguments = {
                    "stress_supervision": "element_quadrature",
                    "stress_random_seed": TRAINING_SEED + 700 + fold_index,
                }
            started = perf_counter()
            final_loss = _train_fixed_epochs(
                model,
                data,
                fold["fit"],
                learning_rate=1e-3,
                epochs=epochs,
                random_seed=TRAINING_SEED + 300 + fold_index,
                label=f"iteration7 {objective_name} fold={fold['name']}",
                peak_supervised_p_weight=0.0,
                max_seconds=3600.0,
                **training_arguments,
            )
            training_seconds = perf_counter() - started
            development_pairs = tuple(pairs[index] for index in fold["development"])
            evaluation = _evaluate_model_element_quadrature(model, development_pairs)
            diagnostics = _served_stress_physics_diagnostics(
                model, data, fold["development"]
            )
            result = {
                "fold": fold["name"],
                "fit_geometry_ids": [
                    meshes[index].geometry_id for index in fold["fit"]
                ],
                "development_geometry_ids": [
                    meshes[index].geometry_id for index in fold["development"]
                ],
                "final_hybrid_loss": final_loss,
                "training_seconds": training_seconds,
                "family_metrics": dict(evaluation.family_metrics),
                "per_case_metrics": dict(evaluation.per_case_metrics),
                "served_stress_physics_diagnostics": diagnostics,
            }
            fold_results[objective_name].append(result)
            print(
                "ITERATION7_RESULT "
                f"objective={objective_name} fold={fold['name']} "
                f"score={max(metric['relative_l2_P'] for metric in evaluation.family_metrics.values()):.8f}",
                flush=True,
            )

            if objective_name == "nodal_patch_control":
                historical_control_drifts.append(
                    _historical_control_metric_drift(result, fold["sealed"])
                )
            elif fold_index == 0 and not _tail_element_screen_passes(
                fold_results["nodal_patch_control"][0],
                result,
            ):
                early_stop_reason = (
                    "tail fold makes element-supervision promotion impossible"
                )
                break
        if early_stop_reason is not None:
            break
        if perf_counter() - screen_started > 12_000.0:
            raise TimeoutError("iteration-7 end-to-end screen time cap exceeded")

    if parameter_counts is None:
        raise RuntimeError("iteration-7 screen created no models")
    control_folds = fold_results["nodal_patch_control"]
    candidate_folds = fold_results["element_quadrature_P_supervision"]
    control_summary = (
        dict(_summarize_screen_folds(control_folds)) if control_folds else {}
    )
    candidate_summary = (
        dict(_summarize_screen_folds(candidate_folds)) if candidate_folds else {}
    )
    complete = len(control_folds) == 2 and len(candidate_folds) == 2
    checks: dict[str, bool]
    if complete:
        checks = dict(
            _screen_guardrail_checks(
                control_summary,
                candidate_summary,
                stress_head_parameter_count=parameter_counts[1],
                total_parameter_growth=0.0,
            )
        )
        checks["P_improves_in_every_family_fold"] = all(
            candidate_folds[fold_index]["family_metrics"][family]["relative_l2_P"]
            < control_folds[fold_index]["family_metrics"][family]["relative_l2_P"]
            for fold_index in range(2)
            for family in ("dogbone", "centre_notched")
        )
        checks["legacy_nodal_family_P_within_absolute_0_005"] = all(
            float(
                np.mean(
                    [
                        fold["family_metrics"][family]["legacy_nodal_relative_l2_P"]
                        for fold in candidate_folds
                    ]
                )
            )
            <= float(
                np.mean(
                    [
                        fold["family_metrics"][family]["legacy_nodal_relative_l2_P"]
                        for fold in control_folds
                    ]
                )
            )
            + 0.005
            for family in ("dogbone", "centre_notched")
        )
        checks["parameter_counts_unchanged"] = parameter_counts == (
            int(sealed_baseline["total_parameter_count"]),
            int(sealed_baseline["stress_head_parameter_count"]),
        )
        checks["paired_current_code_control_completed"] = len(control_folds) == 2
        checks["all_passed"] = all(
            value for name, value in checks.items() if name != "all_passed"
        )
    else:
        checks = {
            "primary_score_improved_by_at_least_10_percent": False,
            "P_improves_in_every_family_fold": False,
            "legacy_nodal_family_P_within_absolute_0_005": False,
            "paired_current_code_control_completed": False,
            "parameter_counts_unchanged": parameter_counts
            == (
                int(sealed_baseline["total_parameter_count"]),
                int(sealed_baseline["stress_head_parameter_count"]),
            ),
            "all_passed": False,
        }

    objectives = {
        "nodal_patch_control": {
            **control_summary,
            "folds": control_folds,
            "stress_supervision": "legacy unweighted nodal patch recovery",
            "total_parameter_count": parameter_counts[0],
            "stress_head_parameter_count": parameter_counts[1],
        },
        "element_quadrature_P_supervision": {
            **candidate_summary,
            "folds": candidate_folds,
            "stress_supervision": "positive three-point raw P1 element quadrature",
            "total_parameter_count": parameter_counts[0],
            "stress_head_parameter_count": parameter_counts[1],
            "early_stop_reason": early_stop_reason,
            "guardrail_checks": checks,
            "guardrails_passed": checks["all_passed"],
        },
    }
    selected_objective = (
        "element_quadrature_P_supervision"
        if checks["all_passed"]
        else "iteration3_joint_adam"
    )
    preregistration = {
        "architecture": STRESS_HEAD_VARIANT,
        "optimizer": "Adam(lr=1e-3, betas=(0.9,0.999), eps=1e-8)",
        "epochs_per_model_fold": epochs,
        "fold_order": ["tail", "head"],
        "nodal_displacement_queries": SUPERVISED_QUERY_COUNT,
        "stress_queries": SUPERVISED_QUERY_COUNT,
        "candidate_stress_quadrature": "three positive barycentric points per P1 triangle",
        "loss_weights": {
            "supervised_u": 1.0,
            "supervised_P": 1.0,
            "equilibrium_on_P_served": 1.0,
            "constitutive_in_plane": 1.0,
            "all_additional_boundary_peak_section_terms": 0.0,
        },
        "primary_score": "worst two-fold-mean family raw-element relative_l2_P",
    }
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "step": "S1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "training_code_sha256": {
            name: _file_sha256(Path(__file__).resolve().parent / name)
            for name in ("model.py", "run.py")
        },
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_checkpoint_sha256": checkpoint.sha256,
        "fem_content_sha256": fem_content_hash,
        "stress_representation_iteration": {
            "iteration": 7,
            "hypothesis": (
                "raw element stress supervision removes the nodal patch-recovery "
                "conflict without changing the PI-GINOT architecture"
            ),
            "selected_objective": selected_objective,
            "decision": (
                "promote element_quadrature_P_supervision"
                if checks["all_passed"]
                else "candidate rejected by training-only guardrails; retain iteration 3"
            ),
            "preregistration": preregistration,
            "preregistration_sha256": hashlib.sha256(
                json.dumps(
                    preregistration, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "data_validation": {
                "element_patch_recovery_max_relative_error": data[
                    "element_patch_recovery_max_relative_error"
                ],
                "raw_to_nodal_projection_relative_l2": list(
                    data["raw_to_nodal_projection_relative_l2"]
                ),
                "stress_quadrature_rule": data["stress_quadrature_rule"],
                "minimum_quadrature_weight": float(
                    data["stress_quadrature_weights"].min()  # type: ignore[union-attr]
                ),
                "all_quadrature_weights_positive": bool(
                    (data["stress_quadrature_weights"] > 0).all()  # type: ignore[operator,union-attr]
                ),
            },
            "selection": {
                "data_policy": "existing 40 training geometries only",
                "training_geometry_count": 40,
                "held_out_geometry_accessed": False,
                "fold_count": len(candidate_folds),
                "development_geometries_per_family_per_fold": 4,
                "historical_control_metric_drift_diagnostic_only": (
                    max(historical_control_drifts)
                    if historical_control_drifts
                    else None
                ),
                "historical_control_drift_reason": (
                    "sealed iteration-3 and current iteration-7 training code hashes "
                    "differ; promotion uses the paired current-code control"
                ),
                "prior_iteration_evidence_sealed": evidence_grade,
                **(
                    {
                        "sealed_rejected_iteration6_sha256": ITERATION6_SCREEN_SHA256,
                        "sealed_iteration7_control_audit_sha256": (
                            ITERATION7_CONTROL_AUDIT_SHA256
                        ),
                    }
                    if evidence_grade
                    else {}
                ),
                "objectives": objectives,
                "guardrails": checks,
            },
            "held_out_evaluation_status": "not_accessed_for_iteration7_selection",
        },
    }
    write_json_strict(target, document)
    print(f"ITERATION7_WROTE {target}", flush=True)
    return document


def _write_evaluation_artifacts(
    model: object,
    evaluation_pairs: Sequence[tuple[SpikeMesh, object]],
    artifact_dir: Path,
) -> Mapping[str, Any]:
    """Persist the final model and held-out FEM-vs-PI-GINOT fields/plots."""
    import torch

    checkpoint_dir = artifact_dir / "checkpoints"
    prediction_dir = artifact_dir / "predictions"
    plot_dir = artifact_dir / "plots"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "final_model.pt"
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "role": "LEGACY",
            "model_state_dict": model.state_dict(),
            "stress_head_variant": STRESS_HEAD_VARIANT,
            "stress_head_conditioning": STRESS_HEAD_CONDITIONING,
            "stress_head_output_scale": STRESS_HEAD_OUTPUT_SCALE,
        },
        temporary_checkpoint,
    )
    temporary_checkpoint.replace(checkpoint_path)

    prediction_count = 0
    plot_count = 0
    model.eval()
    with torch.no_grad():
        for mesh, result in evaluation_pairs:
            points = torch.from_numpy(mesh.points.astype(np.float32))[None]
            boundary = torch.from_numpy(mesh.boundary_points.astype(np.float32))[None]
            x_max = torch.tensor([float(np.max(mesh.points[:, 0]))])
            y_max = torch.tensor([float(np.max(mesh.points[:, 1]))])
            u_delta = torch.full_like(x_max, UPSTREAM_APPLIED_DISPLACEMENT)
            output = model(points, boundary, u_delta, x_max, y_max)

            predicted_u = output.u[0].numpy().astype(np.float64)
            predicted_p = output.P_served[0].numpy().astype(np.float64)
            target_u = np.asarray(result.displacement)
            target_p = np.asarray(result.first_piola)
            stress_error = _stress_magnitude(predicted_p - target_p)
            target_stress = _stress_magnitude(target_p)
            predicted_stress = _stress_magnitude(predicted_p)
            displacement_error = np.linalg.norm(predicted_u - target_u, axis=-1)

            np.savez_compressed(
                prediction_dir / f"{mesh.geometry_id}.npz",
                role="LEGACY",
                geometry_id=mesh.geometry_id,
                family=mesh.family,
                points=mesh.points,
                cells=mesh.cells,
                qoi_node_ids=mesh.qoi_node_ids,
                displacement_fem=target_u,
                displacement_pi_ginot=predicted_u,
                first_piola_fem=target_p,
                first_piola_pi_ginot=predicted_p,
                displacement_error=displacement_error,
                stress_frobenius_error=stress_error,
            )
            prediction_count += 1

            shared_stress_bounds = (
                min(float(np.min(target_stress)), float(np.min(predicted_stress))),
                max(float(np.max(target_stress)), float(np.max(predicted_stress))),
            )
            fields = (
                ("FEM |P|", target_stress, shared_stress_bounds),
                ("PI-GINOT |P|", predicted_stress, shared_stress_bounds),
                ("|P error|", stress_error, None),
                ("|u error|", displacement_error, None),
            )
            panels: list[str] = []
            x = mesh.points[:, 0]
            y = mesh.points[:, 1]
            x_range = max(float(np.ptp(x)), 1e-12)
            y_range = max(float(np.ptp(y)), 1e-12)
            for panel_index, (title, values, declared_bounds) in enumerate(fields):
                v_min, v_max = declared_bounds or (
                    float(np.min(values)),
                    float(np.max(values)),
                )
                span = max(v_max - v_min, 1e-12)
                offset_x = 260 * (panel_index % 2)
                offset_y = 190 * (panel_index // 2)
                circles = []
                for point, value in zip(mesh.points, values, strict=True):
                    normalized = float((value - v_min) / span)
                    red = int(255 * normalized)
                    blue = int(255 * (1.0 - normalized))
                    cx = offset_x + 20 + 220 * float((point[0] - np.min(x)) / x_range)
                    cy = offset_y + 155 - 130 * float((point[1] - np.min(y)) / y_range)
                    circles.append(
                        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="1.4" '
                        f'fill="rgb({red},32,{blue})" />'
                    )
                panels.append(
                    f'<text x="{offset_x + 20}" y="{offset_y + 18}" '
                    f'font-size="12">{title}: {v_min:.3g}..{v_max:.3g}</text>'
                    + "".join(circles)
                )
            svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="380" '
                'viewBox="0 0 520 380">'
                f"<title>{mesh.family} {mesh.geometry_id}</title>"
                + "".join(panels)
                + "</svg>\n"
            )
            (plot_dir / f"{mesh.geometry_id}_fields.svg").write_text(
                svg,
                encoding="utf-8",
            )
            plot_count += 1

    return {
        "directory": str(artifact_dir.resolve()),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _file_sha256(checkpoint_path),
        },
        "prediction_npz_count": prediction_count,
        "comparison_plot_svg_count": plot_count,
    }


def _regularity_check(
    evaluation_meshes: Sequence[SpikeMesh],
) -> Mapping[str, Mapping[str, Any]]:
    from experiments.spike.fem import solve_case
    from experiments.spike.geometry import generate_geometry

    output: dict[str, Mapping[str, Any]] = {}
    levels = ((16, 5), (32, 10), (64, 20))
    for family in ("dogbone", "centre_notched"):
        selected = tuple(mesh for mesh in evaluation_meshes if mesh.family == family)[
            :3
        ]
        case_records: list[Mapping[str, Any]] = []
        changes: list[float] = []
        all_converged = True
        for original in selected:
            peaks: list[float] = []
            level_records: list[Mapping[str, Any]] = []
            for nx, ny in levels:
                refined = generate_geometry(
                    family,
                    original.parameters,
                    original.seed,
                    nx=nx,
                    ny=ny,
                )
                result = solve_case(refined)
                all_converged = all_converged and result.converged
                if result.converged and result.first_piola is not None:
                    peak = float(
                        np.max(
                            _stress_magnitude(result.first_piola[refined.qoi_node_ids])
                        )
                    )
                    peaks.append(peak)
                    level_records.append(
                        {"nx": nx, "ny": ny, "peak_first_piola_mpa": peak}
                    )
                else:
                    level_records.append(
                        {
                            "nx": nx,
                            "ny": ny,
                            "failure_reason": result.failure_reason,
                        }
                    )
            relative_change: float | None = (
                abs(peaks[-1] - peaks[-2]) / max(abs(peaks[-1]), 1e-12)
                if len(peaks) == len(levels)
                else None
            )
            if relative_change is not None:
                changes.append(relative_change)
            case_records.append(
                {
                    "geometry_id": original.geometry_id,
                    "levels": level_records,
                    "last_relative_change": relative_change,
                }
            )
        worst_change = max(changes) if changes else None
        well_posed = (
            all_converged
            and len(changes) == len(selected)
            and worst_change is not None
            and worst_change <= REGULARITY_RELATIVE_CHANGE_LIMIT
        )
        output[family] = {
            "status": "well_posed" if well_posed else "singular_or_diagnostic",
            "reason": (
                "all three smooth-root samples stabilized below the declared "
                "10% last-refinement threshold"
                if well_posed
                else "one or more smooth-root samples did not stabilize below the "
                "declared 10% last-refinement threshold; treat the raw peak as diagnostic"
            ),
            "qoi_definition": (
                "Frobenius norm of in-plane first-Piola stress on the smooth root "
                "region; the dogbone loaded/free endpoint is masked"
            ),
            "relative_change_threshold": REGULARITY_RELATIVE_CHANGE_LIMIT,
            "worst_last_relative_change": worst_change,
            "samples": case_records,
        }
    return output


def _split_proof(split: GeometrySplit) -> Mapping[str, Any]:
    train_geometry = {mesh.geometry_id for mesh in split.train}
    eval_geometry = {mesh.geometry_id for mesh in split.evaluate}
    train_meshes = {mesh.mesh_id for mesh in split.train}
    eval_meshes = {mesh.mesh_id for mesh in split.evaluate}
    train_seeds = {mesh.seed for mesh in split.train}
    eval_seeds = {mesh.seed for mesh in split.evaluate}
    train_nodes = {node for mesh in split.train for node in mesh.node_ids}
    eval_nodes = {node for mesh in split.evaluate for node in mesh.node_ids}
    return {
        "geometry_ids_disjoint": train_geometry.isdisjoint(eval_geometry),
        "mesh_ids_disjoint": train_meshes.isdisjoint(eval_meshes),
        "rng_seeds_disjoint": train_seeds.isdisjoint(eval_seeds),
        "mesh_qualified_node_ids_disjoint": train_nodes.isdisjoint(eval_nodes),
        "node_identity_definition": "mesh_id plus local node index",
        "outcome_independent_preassignment": True,
    }


def _worktree_content_hash(repo_root: Path) -> str:
    paths = (
        "experiments/spike/geometry.py",
        "experiments/spike/fem.py",
        "experiments/spike/model.py",
        "experiments/spike/run.py",
        "experiments/spike/tests/test_geometry.py",
        "experiments/spike/tests/test_fem.py",
        "experiments/spike/tests/test_model.py",
        "experiments/spike/tests/test_run.py",
        "experiments/spike/upstream_pi_ginot/__init__.py",
        "experiments/spike/upstream_pi_ginot/LICENSE",
        "experiments/spike/upstream_pi_ginot/config.py",
        "experiments/spike/upstream_pi_ginot/models/__init__.py",
        "experiments/spike/upstream_pi_ginot/models/geometry_encoder.py",
        "experiments/spike/upstream_pi_ginot/models/physics_decoder.py",
        "experiments/spike/upstream_pi_ginot/models/pi_ginot.py",
        "experiments/spike/upstream_pi_ginot/models/modules/__init__.py",
        "experiments/spike/upstream_pi_ginot/models/modules/point_encoding.py",
        "experiments/spike/upstream_pi_ginot/models/modules/point_position_embedding.py",
        "experiments/spike/upstream_pi_ginot/models/modules/pointnet2_utils.py",
        "experiments/spike/upstream_pi_ginot/models/modules/transformer.py",
    )
    digest = hashlib.sha256()
    for relative in paths:
        path = repo_root / relative
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def run_spike(
    upstream_checkpoint_path: Path | None = None,
) -> Mapping[str, Any]:
    """Execute S1 end to end and write the required strict LEGACY report."""
    import scipy
    import skfem
    import torch

    from experiments.spike.fem import solve_uniaxial_square
    from experiments.spike.geometry import sample_geometry_cases
    from experiments.spike.model import ModelConfig

    repo_root = Path(__file__).resolve().parents[2]
    upstream_checkpoint = _resolve_upstream_checkpoint(upstream_checkpoint_path)
    cache_dir = Path(tempfile.gettempdir()) / "pi-ginot-s1-legacy-cache-019fe6e9"
    fem_content_hash = _fem_content_hash(repo_root)
    cases = sample_geometry_cases(master_seed=CAMPAIGN_SEED, count_per_family=30)
    split = split_by_geometry(cases, master_seed=SPLIT_SEED)
    split_label = {mesh.geometry_id: "train" for mesh in split.train} | {
        mesh.geometry_id: "evaluate" for mesh in split.evaluate
    }

    solve_results: list[object] = []
    for index, mesh in enumerate(cases, start=1):
        result = _solve_with_cache(mesh, cache_dir, fem_content_hash)
        solve_results.append(result)
        print(
            f"FEM {index}/60 {mesh.family} {mesh.geometry_id}: "
            f"{'converged' if result.converged else result.failure_reason} "
            f"({result.solve_seconds:.3f}s)",
            flush=True,
        )
    result_by_geometry = {
        mesh.geometry_id: result
        for mesh, result in zip(cases, solve_results, strict=True)
    }
    converged_count = sum(result.converged for result in solve_results)
    if converged_count < 50:
        raise RuntimeError(
            f"S1 must stop before training: only {converged_count}/60 FEM cases converged"
        )
    if any(not result_by_geometry[mesh.geometry_id].converged for mesh in split.train):
        raise RuntimeError(
            "a preassigned training geometry failed; no outcome reshuffle allowed"
        )
    if any(
        not result_by_geometry[mesh.geometry_id].converged for mesh in split.evaluate
    ):
        raise RuntimeError(
            "a preassigned evaluation geometry failed; G1 report is incomplete"
        )

    training_pairs = tuple(
        (mesh, result_by_geometry[mesh.geometry_id]) for mesh in split.train
    )
    evaluation_pairs = tuple(
        (mesh, result_by_geometry[mesh.geometry_id]) for mesh in split.evaluate
    )
    training = _fit_and_select_model(
        training_pairs,
        upstream_checkpoint.state_dict,
    )
    evaluation = _evaluate_model(training.model, evaluation_pairs)
    created_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    artifact_directory = (
        repo_root
        / "experiments"
        / "spike"
        / "artifacts"
        / f"s1_iteration3_{created_utc.replace(':', '').replace('-', '')}"
    )
    saved_artifacts = _write_evaluation_artifacts(
        training.model,
        evaluation_pairs,
        artifact_directory,
    )
    regularity = _regularity_check(split.evaluate)
    sanity = solve_uniaxial_square(axial_stretch=1.20, nx=6, ny=6)

    cases_document: list[Mapping[str, Any]] = []
    for mesh, result in zip(cases, solve_results, strict=True):
        case_record: dict[str, Any] = {
            "geometry_id": mesh.geometry_id,
            "mesh_id": mesh.mesh_id,
            "family": mesh.family,
            "parameters_mm": dict(mesh.parameters),
            "rng_seed": mesh.seed,
            "split": split_label[mesh.geometry_id],
            "node_count": len(mesh.points),
            "element_count": len(mesh.cells),
            "solve": {
                "converged": result.converged,
                "failure_reason": result.failure_reason,
                "seconds": result.solve_seconds,
                "newton_iterations": result.newton_iterations,
            },
        }
        if mesh.geometry_id in evaluation.per_case_metrics:
            case_record["held_out_metrics"] = evaluation.per_case_metrics[
                mesh.geometry_id
            ]
        cases_document.append(case_record)

    families_document: dict[str, Mapping[str, Any]] = {}
    for family in ("dogbone", "centre_notched"):
        family_solve_pairs = tuple(
            (mesh, result)
            for mesh, result in zip(cases, solve_results, strict=True)
            if mesh.family == family
        )
        failures = Counter(
            result.failure_reason or "unspecified failure"
            for _, result in family_solve_pairs
            if not result.converged
        )
        metrics = evaluation.family_metrics[family]
        families_document[family] = {
            "evaluation_count": 10,
            "metrics": {
                "l2_quadrature": metrics["l2_quadrature"],
                "relative_l2_u": metrics["relative_l2_u"],
                "relative_l2_P": metrics["relative_l2_P"],
                "worst_relative_peak_stress_error": metrics[
                    "worst_relative_peak_stress_error"
                ],
            },
            "fem": {
                "attempted": 30,
                "converged": sum(result.converged for _, result in family_solve_pairs),
                "failed": sum(not result.converged for _, result in family_solve_pairs),
                "failure_reasons": dict(failures),
            },
            "timing": {
                "mean_fem_seconds": float(
                    np.mean([result.solve_seconds for _, result in family_solve_pairs])
                ),
                "mean_inference_seconds": metrics["mean_inference_seconds"],
            },
            "peak_qoi": regularity[family],
        }

    model_config = ModelConfig()
    execution_profile = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_fem": skfem.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "mesh": {"nx": 32, "ny": 10, "fixed": True},
        "material": {"E_MPa": 760.0, "nu": 0.23},
        "applied_displacement_mm": UPSTREAM_APPLIED_DISPLACEMENT,
    }
    document: dict[str, Any] = {
        "schema_version": "1.0",
        "step": "S1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "created_utc": created_utc,
        "commit": _git_head(repo_root),
        "commit_scope": "base commit plus uncommitted local S1 worktree; S1 commit prohibited",
        "worktree_content_sha256": _worktree_content_hash(repo_root),
        "fem_content_sha256": fem_content_hash,
        "execution_profile": execution_profile,
        "execution_profile_hash": hashlib.sha256(
            json.dumps(
                execution_profile, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "assumptions": {
            "centre_notched_plate": (
                "50 mm fixed length, upper-right quarter of a smooth symmetric "
                "double-edge notch; notch_radius is root curvature radius"
            ),
            "centre_notched_parameter_ranges_mm": {
                "width": [16.0, 30.0],
                "notch_radius": [2.0, 8.0],
                "notch_depth": [1.0, 5.0],
            },
            "failed_case_resolution": (
                "20/10 split preassigned before solves; no outcome-dependent reshuffle; "
                "all held-out cases required for a complete report"
            ),
        },
        "architecture_iteration": _architecture_iteration_evidence(),
        "model": {
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_checkpoint": {
                "url": UPSTREAM_CHECKPOINT_URL,
                "sha256": upstream_checkpoint.sha256,
                "epoch": upstream_checkpoint.epoch,
            },
            "architecture_adaptation": UPSTREAM_ARCHITECTURE_ADAPTATION,
            "stress_head": {
                "variant": STRESS_HEAD_VARIANT,
                "conditioning": list(STRESS_HEAD_CONDITIONING),
                "output_scale": STRESS_HEAD_OUTPUT_SCALE,
                "youngs_modulus_MPa": 760.0,
                "output_layer_initialization": "zeros",
            },
            "dimensions": model_config.__dict__,
            "served_stress": UPSTREAM_SERVED_STRESS,
            "material_stress_role": UPSTREAM_MATERIAL_STRESS_ROLE,
            "known_legacy_limitations": [
                (
                    "the upstream hard-BC displacement correction is not multiplied by "
                    "physical x/y scales, so the displacement path remains not "
                    "dimensionally equivariant under uniform geometry scaling"
                ),
                (
                    "the conditioned direct stress head is calibrated only for the fixed "
                    "1 mm S1 load and does not establish generalization to unseen loads"
                ),
                (
                    "FEM nodal stresses use element-to-node count averaging, so the "
                    "small-area peak-stress QoI remains recovery-sensitive"
                ),
            ],
        },
        "artifacts": saved_artifacts,
        "protocol": {
            "attempted": 60,
            "converged": converged_count,
            "failed": 60 - converged_count,
            "split_seed": SPLIT_SEED,
            "campaign_seed": CAMPAIGN_SEED,
            "train_per_family": 20,
            "evaluate_per_family": 10,
            "split_proof": _split_proof(split),
        },
        "fem_sanity": {
            "axial_stretch": 1.20,
            "converged": sanity.converged,
            "failure_reason": sanity.failure_reason,
            "lateral_stretch": sanity.lateral_stretch,
            "relative_l2_u": sanity.relative_l2_u,
            "relative_l2_P": sanity.relative_l2_p,
            "max_error": sanity.max_error,
            "threshold": 1e-6,
        },
        "lr_sweep": {
            "initialization": (
                "exact upstream best.pt encoder/displacement weights at epoch "
                f"{upstream_checkpoint.epoch}; direct P head output layer initialized "
                "to zero"
            ),
            "learning_rates": list(LEARNING_RATES),
            "selection_rule": (
                "minimum worst-family reference-area relative L2 P error on 8 "
                "geometries reserved only from the 40-geometry training split; "
                "held-out evaluation untouched"
            ),
            "fit_geometries_per_family": 16,
            "development_geometries_per_family": 4,
            "epochs_per_trial": LR_SWEEP_EPOCHS,
            "trials": list(training.trials),
            "selected_learning_rate": training.selected_learning_rate,
            "final_training_epochs": FINAL_TRAINING_EPOCHS,
            "final_mean_hybrid_loss": training.final_mean_hybrid_loss,
            "training_seconds": training.training_seconds,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "supervised_sampling": (
                "reference-area-proportional nodal sampling with replacement"
            ),
            "fixed_loss_weights": {
                "supervised_u": 1.0,
                "supervised_P": 1.0,
                "equilibrium_on_P_served": 1.0,
                "constitutive_in_plane": 1.0,
            },
            "loss_normalization": (
                "per-geometry RMS targets; equilibrium uses sqrt(x_max*y_max) as "
                "the fixed characteristic length"
            ),
        },
        "families": families_document,
        "cases": cases_document,
    }
    validate_results(document)
    results_path = repo_root / "experiments" / "spike" / "results.json"
    write_json_strict(results_path, document)
    print(f"wrote {results_path}", flush=True)
    return document


def main() -> None:
    document = run_spike()
    summary = {
        family: document["families"][family]["metrics"]
        for family in ("dogbone", "centre_notched")
    }
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
