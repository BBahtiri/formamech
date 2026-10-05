"""TRAIN-screen evaluation, G11, G13 and CP-FREEZE of the notched campaign.

Every measurement is the DogBone S13c measurement on this family:

- ``screen``: the registered checkpoint on the 32 TRAIN-screen FEM cases
  (vertex displacement over the prescribed displacement, in-plane recovered
  first-Piola stress over 760 MPa, signed peak over the notch vertices), the
  deformation determinant at the vertices and cell centroids of all 128 TRAIN
  meshes, and the absolute-guardrail selection;
- ``g11``: the amended G11 checks on the first TRAIN-screen geometry and the
  exact lifting over the whole four-input box (16 corners, centre, 47 draws);
- ``probe``: an independent replay of the first 400 steps, interrupted at 200;
- ``g13`` and ``freeze``: the G13 gate and the CP-FREEZE record.

DEVELOPMENT and HOLDOUT are never opened.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch

from campaigns.dogbone_v1.s10_closeout import _load_run_execution_profile
from campaigns.dogbone_v1.s10_production import (
    ProductionFemRunLedger,
    load_production_attempts,
)
from campaigns.dogbone_v1.s13c_evaluate import (
    S13cCaseDiagnostics,
    _case_diagnostics,
    _served_vertex_output,
    load_run_binding,
)
from campaigns.dogbone_v1.s13c_g11 import (
    _DeserializationMarker,
    _heads,
    _node_order,
    _query_batch,
    _reachable_parameters,
)
from campaigns.dogbone_v1.s13c_trial import latest_checkpoint
from campaigns.notched_v1.family import BOUNDS, HALF_LENGTH_MM, NOTCH_TAG
from campaigns.notched_v1.gate_io import (
    check_document,
    gate_document,
    git_head,
    plain,
    run_pytest_with_coverage,
    utc_now,
    write_gate_document,
)
from campaigns.notched_v1.pilot import ARTIFACTS
from campaigns.notched_v1.s9 import GATES, PUBLIC_MANIFEST_PATH
from campaigns.notched_v1.s10 import public_requests, request_plan
from campaigns.notched_v1.s13 import (
    CONDITIONING,
    ENERGY_EVALUATION_SEED,
    PROTOCOL_PATH,
    RUN_BINDING_NAME,
    STRESS_REFERENCE_MPA,
    TRAINING_LOG_NAME,
    NotchedGeometryCase,
    NotchedS13Protocol,
    build_initial_operator,
    case_energy,
    load_committed_protocol,
    published_cases,
    sampler_plan,
    train_geometry,
    trainer_config,
)
from piginot.contracts import canonical_json_bytes
from piginot.doe import EvidenceRole
from piginot.fem import FemRequest, FemResult
from piginot.operators import (
    ENERGY_G11_LIFTING_POINTS_PER_EDGE,
    ENERGY_G11_MINIMUM_PARAMETER_VECTORS,
    AffineDirichletLifting,
    AxisAlignedTensionLifting,
    EnergyFormOperator,
    EnergyG11Measurements,
    QueryFrame,
    energy_module_digest,
    evaluate_energy_g11_gate,
)
from piginot.physics._validation import DETERMINANT_MARGIN
from piginot.store import ObjectStore
from piginot.training import (
    CheckpointLineage,
    GridSearchSpace,
    NoEligibleTrialError,
    OutOfSearchSpaceError,
    SelectionMetrics,
    SelectionPolicy,
    TrainingCheckpointManifest,
    TrainingRunBinding,
    TrialResult,
    load_training_checkpoint,
    require_trial_in_space,
    select_checkpoint,
    select_training_trial,
)
from piginot.training.selection import (
    _choose_nonranked_checkpoint_mutant_for_test,
    _select_on_holdout_mutant_for_test,
)

REPOSITORY = Path(__file__).resolve().parents[2]
SCREEN_PATH = ARTIFACTS / "S13-TRAIN-SCREEN.json"
PROBE_PATH = ARTIFACTS / "S13-G13-REPLAY-PROBE.json"
CP_FREEZE_PATH = ARTIFACTS / "CP-FREEZE.json"
G11_PATH = GATES / "S11-ENERGY.json"
G13_PATH = GATES / "S13.json"
S9_GATE_PATH = GATES / "S9.json"
FINAL_STEP = 2_400
PROBE_STEP = 400
_ENERGY_GAP_POINTS = 12
_INTERIOR_DRAW_SEED = 13_105
_INTERIOR_DRAWS = 47
_MUTANT_POINTS_PER_EDGE = 320


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_vertices(mesh: Any) -> tuple[int, ...]:
    """Vertices of the notch profile (the root of the stress concentration)."""

    facets = mesh.boundary_facets
    return tuple(
        sorted(
            {
                vertex
                for index in mesh.semantic_tags[NOTCH_TAG]
                for vertex in facets[index]
            }
        )
    )


def _verify_binding(binding: TrainingRunBinding, protocol: NotchedS13Protocol) -> None:
    expected = {
        "dataset_snapshot_hash": protocol.dataset_snapshot_hash,
        "public_manifest_hash": protocol.public_manifest_hash,
        "normalization_hash": protocol.normalization_hash,
        "architecture_id": protocol.architecture_id,
        "architecture_module_digest": protocol.architecture_module_digest,
        "objective_registry_hash": protocol.objective_hash,
        "trainer_config_hash": trainer_config(protocol).config_hash,
        "sampler_plan_hash": sampler_plan(protocol).plan_hash,
        "initial_weights_hash": protocol.initial_weights_hash,
        "seed": protocol.training_seed,
    }
    for name, value in expected.items():
        if getattr(binding, name) != value:
            raise ValueError(f"run binding {name} differs from the S13 protocol")
    profile = binding.execution_profile
    if (profile.dtype, profile.blas_threads, profile.device) != (
        protocol.execution_dtype,
        protocol.execution_blas_threads,
        "cpu",
    ):
        raise ValueError("run binding execution profile differs from the protocol")


def load_registered_checkpoint(
    protocol: NotchedS13Protocol, checkpoint_path: Path, *, require_final: bool = True
) -> tuple[EnergyFormOperator, TrainingRunBinding, TrainingCheckpointManifest]:
    """Validate a checkpoint against its run binding and the protocol."""

    binding = load_run_binding(checkpoint_path.parent / RUN_BINDING_NAME)
    _verify_binding(binding, protocol)
    config = trainer_config(protocol)
    model = build_initial_operator()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.schedule.learning_rate(0),
        betas=config.adam_betas,
        eps=config.adam_eps,
        weight_decay=config.adam_weight_decay,
        foreach=False,
        fused=False,
    )
    state = torch.random.get_rng_state()
    try:
        loaded = load_training_checkpoint(
            checkpoint_path, model=model, optimizer=optimizer, binding=binding
        )
    finally:
        torch.random.set_rng_state(state)
    manifest = loaded.manifest
    if manifest.run_hash != binding.run_hash:
        raise ValueError("checkpoint run_hash differs from its run binding")
    if require_final and manifest.step != config.trial.optimizer_steps:
        raise ValueError("checkpoint is not the final registered checkpoint")
    return model, binding, manifest


def fem_results(
    s10_run_root: Path, case_ids: Sequence[str]
) -> Mapping[str, tuple[FemRequest, FemResult]]:
    """Requests and FEM results of the named public cases from the S10 run."""

    profile = _load_run_execution_profile(s10_run_root)
    plan = request_plan(profile)
    ledger = ProductionFemRunLedger.from_bytes(
        (s10_run_root / "ledger.json").read_bytes()
    )
    attempts = load_production_attempts(
        request_plan=plan,
        requests={r.request_id: r for r in public_requests(profile)},
        ledger=ledger,
        store=ObjectStore.open(s10_run_root / "object-store"),
    )
    output = {}
    for case_id in case_ids:
        attempt = attempts[case_id]
        if not attempt.success or attempt.result is None:
            raise ValueError(f"{case_id} has no successful FEM result")
        output[case_id] = (attempt.request, attempt.result)
    return output


def vertex_targets(
    case_id: str, strain: float, request: FemRequest, result: FemResult
) -> Any:
    """Vertex targets with the S14 arithmetic: u / (strain * 25), P / 760."""

    if result.request_id != case_id or result.mesh_id != request.mesh.mesh_id:
        raise ValueError("case, request and result identities do not match")
    if request.config.element_order != 2 or result.element_order != 2:
        raise ValueError("scoring requires the verified P2 FEM recipe")
    count = len(request.mesh.nodes)
    query = torch.as_tensor(request.mesh.nodes, dtype=torch.float64)
    coordinates = torch.as_tensor(
        result.displacement_dof_coordinates[:count], dtype=torch.float64
    )
    if not torch.equal(coordinates, query):
        raise ValueError("P2 vertex coordinates do not align with mesh vertices")
    u = torch.as_tensor(result.displacement[:count], dtype=torch.float64)
    stress = torch.as_tensor(
        result.recovered_first_piola_stress[:count], dtype=torch.float64
    )
    if not bool(torch.isfinite(u).all()) or not bool(torch.isfinite(stress).all()):
        raise ValueError("P2 vertex fields must be finite")
    from types import SimpleNamespace

    return SimpleNamespace(
        case_id=case_id,
        query_xy=query,
        target_u=u / (strain * HALF_LENGTH_MM),
        target_P=stress[:, :2, :2] / STRESS_REFERENCE_MPA,
        qoi_indices=torch.tensor(root_vertices(request.mesh), dtype=torch.int64),
    )


def _orientation_points(case: NotchedGeometryCase) -> torch.Tensor:
    vertices = case.domain.vertices.to(torch.float64)
    centroids = vertices[case.domain.cells].mean(dim=1)
    return torch.cat((vertices, centroids))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


@dataclass(frozen=True, slots=True)
class ScreenEvaluation:
    protocol: NotchedS13Protocol
    binding: TrainingRunBinding
    manifest: TrainingCheckpointManifest
    result: TrialResult
    selected: bool
    failed_guardrails: tuple[str, ...]
    diagnostics: tuple[S13cCaseDiagnostics, ...]
    invalid_case_count: int
    minimum_orientation_determinant: float


def failed_guardrails(
    protocol: NotchedS13Protocol, metrics: SelectionMetrics
) -> tuple[str, ...]:
    limits = protocol.selection_policy.guardrails
    assert hasattr(limits, "maximum_median_absolute_peak_error")
    checks = (
        ("invalid_collocation_points", metrics.invalid_collocation_points == 0),
        (
            "median_absolute_peak_error",
            metrics.median_absolute_peak_error
            <= limits.maximum_median_absolute_peak_error,
        ),
        ("p_relative_l2", metrics.p_relative_l2 <= limits.maximum_p_relative_l2),
        ("u_relative_l2", metrics.u_relative_l2 <= limits.maximum_u_relative_l2),
    )
    return tuple(name for name, passed in checks if not passed)


def evaluate_screen(
    *, s10_run_root: Path, checkpoint_path: Path, require_final: bool = True
) -> ScreenEvaluation:
    protocol = load_committed_protocol()
    model, binding, manifest = load_registered_checkpoint(
        protocol, checkpoint_path, require_final=require_final
    )
    served = copy.deepcopy(model).to(torch.float64).eval()
    screen_cases = train_geometry(
        protocol.screen_case_ids, protocol.normalization, dtype=torch.float64
    )
    labelled = fem_results(s10_run_root, protocol.screen_case_ids)
    coordinates = {
        case.case_id: case.primitive_coordinates
        for case in published_cases(protocol.screen_case_ids)
    }
    generator = torch.Generator().manual_seed(ENERGY_EVALUATION_SEED)
    diagnostics, energies = [], []
    for geometry in screen_cases:
        request, result = labelled[geometry.case_id]
        if request.recipe_hash != geometry.request.recipe_hash:
            raise ValueError(f"{geometry.case_id} FEM request differs from the plan")
        strain = float(coordinates[geometry.case_id]["nominal_axial_strain"])
        targets = vertex_targets(geometry.case_id, strain, request, result)
        u, stress, determinant = _served_vertex_output(
            served, geometry, targets.query_xy
        )  # type: ignore[arg-type]
        if not bool(torch.isfinite(stress).all().item()):
            raise ValueError(f"{geometry.case_id} has inadmissible vertex deformation")
        fine = case_energy(
            served,
            geometry,
            replace(protocol.objective.energy, points_per_cell=_ENERGY_GAP_POINTS),
            generator=generator,
            create_graph=False,
        )
        diagnostics.append(
            replace(
                _case_diagnostics(
                    targets,
                    u / geometry.prescribed_displacement_mm,
                    stress / STRESS_REFERENCE_MPA,
                    determinant,
                ),
                model_strain_energy=float(fine.energy)
                * float(geometry.material.properties["elastic_modulus"].value)
                * geometry.domain.area,
                fem_strain_energy=float(result.total_strain_energy),
            )
        )
        energies.append(
            float(
                case_energy(
                    served,
                    geometry,
                    protocol.objective.energy,
                    generator=generator,
                    create_graph=False,
                ).total
            )
        )
    invalid = invalid_cases = 0
    minimum = math.inf
    all_train = (*protocol.fit_case_ids, *protocol.screen_case_ids)
    for geometry in train_geometry(
        all_train, protocol.normalization, dtype=torch.float64
    ):
        _, _, determinant = _served_vertex_output(
            served, geometry, _orientation_points(geometry)
        )  # type: ignore[arg-type]
        count = int((determinant <= DETERMINANT_MARGIN).sum().item())
        invalid += count
        invalid_cases += int(count > 0)
        minimum = min(minimum, float(determinant.min()))
    signed = [item.signed_peak_error for item in diagnostics]
    absolute = [abs(value) for value in signed]
    metrics = SelectionMetrics(
        median_absolute_peak_error=_quantile(absolute, 0.5),
        p90_absolute_peak_error=_quantile(absolute, 0.9),
        median_signed_peak_error=_quantile(signed, 0.5),
        p_relative_l2=_quantile([item.p_relative_l2 for item in diagnostics], 0.5),
        u_relative_l2=_quantile([item.u_relative_l2 for item in diagnostics], 0.5),
        hybrid_loss=statistics.fmean(energies),
        invalid_collocation_points=invalid,
    )
    (trial,) = protocol.search_space.trials
    result = TrialResult(
        lineage=CheckpointLineage(
            trial_id=trial.trial_id,
            run_hash=manifest.run_hash,
            dataset_snapshot_hash=manifest.dataset_snapshot_hash,
            execution_profile_hash=manifest.execution_profile_hash,
            weights_hash=manifest.model_export_hash,
            model_state_hash=manifest.model_state_hash,
            checkpoint_hash=manifest.manifest_hash,
        ),
        role=EvidenceRole.TRAIN,
        metrics=metrics,
    )
    try:
        select_checkpoint((result,), policy=protocol.selection_policy)
        selected = True
    except NoEligibleTrialError:
        selected = False
    return ScreenEvaluation(
        protocol=protocol,
        binding=binding,
        manifest=manifest,
        result=result,
        selected=selected,
        failed_guardrails=failed_guardrails(protocol, metrics),
        diagnostics=tuple(diagnostics),
        invalid_case_count=invalid_cases,
        minimum_orientation_determinant=minimum,
    )


def _training_summary(run_root: Path) -> dict[str, Any]:
    raw = [
        json.loads(line)
        for line in (run_root / TRAINING_LOG_NAME).read_text("utf-8").splitlines()
        if line.strip()
    ]
    by_step = {record["step"]: record for record in raw}
    if sorted(by_step) != list(range(1, len(by_step) + 1)):
        raise ValueError("training log must cover every optimizer step")
    records = [by_step[step] for step in sorted(by_step)]
    return {
        "logged_steps": len(records),
        "replayed_log_records": len(raw) - len(records),
        "mean_energy_first_100_steps": statistics.fmean(
            r["energy"] for r in records[:100]
        ),
        "mean_energy_last_100_steps": statistics.fmean(
            r["energy"] for r in records[-100:]
        ),
        "maximum_barrier": max(r["barrier"] for r in records),
        "minimum_determinant": min(r["minimum_determinant"] for r in records),
        "inadmissible_points_total": sum(r["inadmissible_points"] for r in records),
        "final_learning_rate": records[-1]["learning_rate"],
        "training_log_sha256": _sha256(run_root / TRAINING_LOG_NAME),
    }


def screen_report(evaluation: ScreenEvaluation, run_root: Path) -> dict[str, Any]:
    diagnostics = evaluation.diagnostics
    return {
        "schema_version": "1.0",
        "step": "S13",
        "campaign_id": "notched-v1",
        "evidence_role": "TRAIN",
        "protocol_hash": evaluation.protocol.protocol_hash,
        "run_hash": evaluation.binding.run_hash,
        "training_revision": evaluation.binding.implementation_revision,
        "evaluator_revision": git_head(REPOSITORY),
        "execution_profile_hash": evaluation.binding.execution_profile_hash,
        "checkpoint_step": evaluation.manifest.step,
        "checkpoint_manifest_hash": evaluation.manifest.manifest_hash,
        "trial_result": evaluation.result,
        "guardrails": evaluation.protocol.selection_policy.guardrails,
        "failed_guardrails": list(evaluation.failed_guardrails),
        "selected_checkpoint": evaluation.result.lineage
        if evaluation.selected
        else None,
        "invalid_orientation_case_count": evaluation.invalid_case_count,
        "minimum_orientation_determinant": evaluation.minimum_orientation_determinant,
        "screen_diagnostics": {
            "case_count": len(diagnostics),
            "underpredicted_case_count": sum(
                item.signed_peak_error < 0 for item in diagnostics
            ),
            "peak_at_reference_vertex_count": sum(
                item.predicted_peak_vertex == item.reference_peak_vertex
                for item in diagnostics
            ),
            "median_root_p_relative_l2": statistics.median(
                item.root_p_relative_l2 for item in diagnostics
            ),
            "median_non_root_p_relative_l2": statistics.median(
                item.non_root_p_relative_l2 for item in diagnostics
            ),
            "median_relative_energy_gap_to_fem": statistics.median(
                item.relative_energy_gap for item in diagnostics
            ),
            "cases": [
                {**asdict(item), "relative_energy_gap": item.relative_energy_gap}
                for item in diagnostics
            ],
        },
        "training_summary": _training_summary(run_root),
        "selection_status": "ELIGIBLE" if evaluation.selected else "NO_ELIGIBLE_TRIAL",
        "non_train_evidence_used": False,
    }


def compare_replay(registered_run_root: Path, probe_run_root: Path) -> dict[str, Any]:
    """Independent replay of the first segment, interrupted and resumed at half."""

    registered = registered_run_root / f"checkpoint-step-{PROBE_STEP:04d}"
    reference = TrainingCheckpointManifest.from_bytes(
        (registered / "checkpoint.json").read_bytes()
    )
    half = PROBE_STEP // 2
    if not (probe_run_root / f"checkpoint-step-{half:04d}").is_dir():
        raise RuntimeError("replay probe was not interrupted at half of the segment")
    latest, step = latest_checkpoint(
        probe_run_root, expected_run_hash=reference.run_hash
    )
    if latest is None or step != PROBE_STEP:
        raise RuntimeError("replay probe did not reach the target step")
    replay = TrainingCheckpointManifest.from_bytes(
        (latest / "checkpoint.json").read_bytes()
    )
    files = ("model.safetensors", "state.safetensors", "checkpoint.json")
    identical = {
        name: (latest / name).read_bytes() == (registered / name).read_bytes()
        for name in files
    }
    return {
        "schema_version": "1.0",
        "step": "S13",
        "campaign_id": "notched-v1",
        "evidence_role": "TRAIN",
        "run_hash": reference.run_hash,
        "target_step": PROBE_STEP,
        "replay_interrupted_and_resumed_at_step": half,
        "registered_model_state_hash": reference.model_state_hash,
        "replay_model_state_hash": replay.model_state_hash,
        "registered_weights_hash": reference.model_export_hash,
        "replay_weights_hash": replay.model_export_hash,
        "byte_identical_files": identical,
        "g13_determinism": "PASS"
        if replay.model_export_hash == reference.model_export_hash
        else "FAIL",
        "g13_resume": "PASS"
        if all(identical.values())
        and replay.model_state_hash == reference.model_state_hash
        else "FAIL",
    }


# --- G11 --------------------------------------------------------------------


def lifting_parameter_vectors() -> tuple[dict[str, float], ...]:
    """The box corners, its centre and seeded interior draws (64 vectors)."""

    corners = [
        dict(zip(CONDITIONING, values, strict=True))
        for values in itertools.product(*(BOUNDS[name] for name in CONDITIONING))
    ]
    centre = {name: 0.5 * (BOUNDS[name][0] + BOUNDS[name][1]) for name in CONDITIONING}
    generator = torch.Generator().manual_seed(_INTERIOR_DRAW_SEED)
    draws = torch.rand(
        (_INTERIOR_DRAWS, len(CONDITIONING)), generator=generator, dtype=torch.float64
    )
    interior = [
        {
            name: BOUNDS[name][0]
            + float(draws[row, column]) * (BOUNDS[name][1] - BOUNDS[name][0])
            for column, name in enumerate(CONDITIONING)
        }
        for row in range(_INTERIOR_DRAWS)
    ]
    return (*corners, centre, *interior)


def _lifting_error(
    model: EnergyFormOperator,
    primitives: Mapping[str, float],
    protocol: NotchedS13Protocol,
    count: int,
    *,
    mutant: bool,
) -> float:
    half_width = 0.5 * primitives["width"]
    ligament = half_width - primitives["notch_depth"]
    displacement = primitives["nominal_axial_strain"] * HALF_LENGTH_MM
    extent = torch.tensor([[HALF_LENGTH_MM, half_width]], dtype=torch.float64)
    frame = QueryFrame(torch.zeros((1, 2), dtype=torch.float64), extent)
    lifting = AxisAlignedTensionLifting(
        extent, torch.tensor([displacement], dtype=torch.float64)
    )

    def identity(query: torch.Tensor) -> AffineDirichletLifting:
        return AffineDirichletLifting(torch.zeros_like(query), torch.ones_like(query))

    line = torch.linspace(0.0, 1.0, count, dtype=torch.float64)
    zeros = torch.zeros(count, dtype=torch.float64)
    left = torch.stack((zeros, ligament * line), dim=-1)
    bottom = torch.stack((HALF_LENGTH_MM * line, zeros), dim=-1)
    loaded = torch.stack((zeros + HALF_LENGTH_MM, half_width * line), dim=-1)
    query = torch.cat((left, bottom, loaded)).unsqueeze(0)
    parameters = protocol.normalization.vector(primitives).unsqueeze(0)
    with torch.no_grad():
        u = model.displacement(
            query_xy=query,
            parameters=parameters,
            frame=frame,
            lifting=identity if mutant else lifting,
        )[0]
    errors = torch.cat(
        (
            torch.abs(u[:count, 0]),
            torch.abs(u[count : 2 * count, 1]),
            torch.abs(u[2 * count :, 0] - displacement),
        )
    )
    return float(errors.max().item()) / displacement


def _pickle_refused(protocol: NotchedS13Protocol, checkpoint_path: Path) -> bool:
    import pickle
    import shutil

    for name in ("model.safetensors", "state.safetensors"):
        with tempfile.TemporaryDirectory(prefix="piginot-notched-g11-") as directory:
            root = Path(directory)
            copied = root / checkpoint_path.name
            shutil.copytree(checkpoint_path, copied)
            shutil.copy2(
                checkpoint_path.parent / RUN_BINDING_NAME, root / RUN_BINDING_NAME
            )
            marker = root / "executed.txt"
            (copied / name).write_bytes(pickle.dumps(_DeserializationMarker(marker)))
            try:
                load_registered_checkpoint(protocol, copied)
            except ValueError:
                if marker.exists():
                    return False
                continue
            return False
    return True


def g11_gate(checkpoint_path: Path) -> dict[str, Any]:
    protocol = load_committed_protocol()
    model, binding, manifest = load_registered_checkpoint(protocol, checkpoint_path)
    served = copy.deepcopy(model).to(torch.float64).eval()
    probe_id = protocol.screen_case_ids[0]
    (case,) = train_geometry((probe_id,), protocol.normalization, dtype=torch.float64)
    query = case.domain.vertices.unsqueeze(0)
    u_shape, p_shape, displacement_only, stress_identity = _heads(served, case, query)  # type: ignore[arg-type]
    vectors = lifting_parameter_vectors()
    if len(vectors) < ENERGY_G11_MINIMUM_PARAMETER_VECTORS:
        raise RuntimeError("the lifting population is smaller than ADR 0018 fixes")
    summary, coverage, passed = run_pytest_with_coverage(
        REPOSITORY, ("tests/operators",), ("piginot.operators",)
    )
    measurements = EnergyG11Measurements(
        node_order_relative_change=_node_order(served, case, query),  # type: ignore[arg-type]
        query_batch_relative_change=_query_batch(served, case, query),  # type: ignore[arg-type]
        reachable_parameter_count=_reachable_parameters(served, case, query),  # type: ignore[arg-type]
        required_parameter_count=len(CONDITIONING),
        pickle_refused_before_deserialization=_pickle_refused(
            protocol, checkpoint_path
        ),
        u_shape_verified=u_shape,
        p_shape_verified=p_shape,
        displacement_only_output=displacement_only,
        served_stress_energy_gradient_relative_difference=stress_identity,
        lifting_parameter_vector_count=len(vectors),
        lifting_points_per_edge=ENERGY_G11_LIFTING_POINTS_PER_EDGE,
        lifting_max_relative_error=max(
            _lifting_error(served, vector, protocol, count, mutant=False)
            for vector in vectors
            for count in ENERGY_G11_LIFTING_POINTS_PER_EDGE
        ),
        lifting_mutant_control_relative_error=max(
            _lifting_error(served, v, protocol, _MUTANT_POINTS_PER_EDGE, mutant=False)
            for v in vectors
        ),
        lifting_mutant_relative_error=max(
            _lifting_error(served, v, protocol, _MUTANT_POINTS_PER_EDGE, mutant=True)
            for v in vectors
        ),
        code_coverage_percent=coverage if passed else None,
    )
    evaluation = evaluate_energy_g11_gate(measurements)
    checks = [
        check_document(
            c.check_id, c.status.value == "PASS", c.measured, c.threshold, c.comparison
        )
        for c in evaluation.checks
    ]
    return gate_document(
        "G11",
        "S11",
        REPOSITORY,
        checks,
        decision_record="docs/adr/0018-s13c-energy-form-operator.md",
        architecture={
            "architecture_id": served.architecture_id,
            "module_digest": energy_module_digest(model),
            "conditioning_parameters": list(CONDITIONING),
        },
        measured_checkpoint={
            "run_hash": binding.run_hash,
            "checkpoint_manifest_hash": manifest.manifest_hash,
            "model_state_hash": manifest.model_state_hash,
            "weights_hash": manifest.model_export_hash,
        },
        probe={
            "train_screen_case_id": probe_id,
            "query_points": int(query.shape[1]),
            "lifting_parameter_vectors": len(vectors),
            "interior_draw_seed": _INTERIOR_DRAW_SEED,
        },
        measurements=measurements,
        test_evidence={"command": "pytest tests/operators", "result": summary},
        scope={
            "claim": "structural evidence on the registered checkpoint; no accuracy claim",
            "fem_labels_read": False,
            "development_or_holdout_read": False,
        },
    )


# --- G13 and CP-FREEZE --------------------------------------------------------


def _load(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text("utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return document


def _strip(document: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "$type"}


def screen_result(screen: Mapping[str, Any]) -> TrialResult:
    document = screen["trial_result"]
    return TrialResult(
        lineage=CheckpointLineage(**_strip(document["lineage"])),
        role=EvidenceRole(document["role"]),
        metrics=SelectionMetrics(**_strip(document["metrics"])),
    )


def _is_ancestor(revision: str) -> bool:
    import subprocess

    from piginot.git_environment import isolated_git_environment

    return (
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", revision, "HEAD"),
            cwd=REPOSITORY,
            env=isolated_git_environment(),
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def g13_checks(run_root: Path) -> list[dict[str, Any]]:
    protocol = load_committed_protocol()
    screen = _load(REPOSITORY / SCREEN_PATH)
    probe = _load(REPOSITORY / PROBE_PATH)
    result = screen_result(screen)
    binding = load_run_binding(run_root / RUN_BINDING_NAME)
    registered = TrainingCheckpointManifest.from_bytes(
        (
            run_root / f"checkpoint-step-{PROBE_STEP:04d}" / "checkpoint.json"
        ).read_bytes()
    )
    checks = []
    determinism = {
        "probe_run_hash_matches": probe["run_hash"] == binding.run_hash,
        "registered_weights_match_run_root": probe["registered_weights_hash"]
        == registered.model_export_hash,
        "replay_weights_equal_registered": probe["replay_weights_hash"]
        == probe["registered_weights_hash"],
    }
    checks.append(
        check_document(
            "G13.determinism",
            all(determinism.values()) and probe["g13_determinism"] == "PASS",
            determinism,
            "independent replay from the training revision has the registered weights",
            "all",
        )
    )
    resume = {
        "interrupted_at_step": probe["replay_interrupted_and_resumed_at_step"],
        "byte_identical_files": probe["byte_identical_files"],
        "registered_state_matches_run_root": probe["registered_model_state_hash"]
        == registered.model_state_hash,
    }
    checks.append(
        check_document(
            "G13.resume",
            probe["g13_resume"] == "PASS"
            and len(probe["byte_identical_files"]) == 3
            and all(probe["byte_identical_files"].values())
            and resume["registered_state_matches_run_root"] is True
            and probe["replay_interrupted_and_resumed_at_step"] * 2 == PROBE_STEP,
            resume,
            "interrupt at half of the 400-step segment and resume: bitwise identical",
            "all",
        )
    )
    space = protocol.search_space
    (trial,) = space.trials
    outside = (
        replace(trial, learning_rate=trial.learning_rate * 10.0),
        replace(trial, optimizer_steps=trial.optimizer_steps + 1),
        replace(trial, case_batch_size=trial.case_batch_size + 1),
    )
    refused = 0
    for candidate in outside:
        try:
            require_trial_in_space(candidate, space)
        except OutOfSearchSpaceError:
            refused += 1
    authorized = require_trial_in_space(trial, space) == trial
    checks.append(
        check_document(
            "G13.search-space",
            authorized and refused == 3 and trial.trial_id == result.trial_id,
            {"registered_trials": len(space.trials), "out_of_space_refused": refused},
            "every out-of-space trial refuses; the registered trial is authorized",
            "all",
        )
    )
    holdout = replace(result, role=EvidenceRole.HOLDOUT)
    production_refused = 0
    for attempt in (
        lambda: select_training_trial(space, (holdout,), role=EvidenceRole.HOLDOUT),
        lambda: select_checkpoint((holdout,), policy=protocol.selection_policy),
    ):
        try:
            attempt()
        except PermissionError:
            production_refused += 1
    mutant = _select_on_holdout_mutant_for_test(
        space, (holdout,), role=EvidenceRole.HOLDOUT
    )
    manifest = _load(REPOSITORY / PUBLIC_MANIFEST_PATH)
    development_id = next(
        c["case_id"] for c in manifest["cases"] if c["role"] == "DEVELOPMENT"
    )
    try:
        train_geometry((development_id,), protocol.normalization)
        development_refused = False
    except PermissionError:
        development_refused = True
    visibility = _load(REPOSITORY / S9_GATE_PATH).get("visibility", {})
    holdout_measured = {
        "production_selection_refusals": production_refused,
        "select_on_holdout_mutant_selected_holdout": mutant.role
        is EvidenceRole.HOLDOUT,
        "development_case_refused_by_training_source": development_refused,
        "g9_holdout_visibility_violations": sum(visibility.values())
        if visibility
        else None,
        "screen_used_non_train_evidence": screen["non_train_evidence_used"],
    }
    checks.append(
        check_document(
            "G13.holdout",
            production_refused == 2
            and holdout_measured["select_on_holdout_mutant_selected_holdout"] is True
            and development_refused
            and holdout_measured["g9_holdout_visibility_violations"] == 0
            and screen["non_train_evidence_used"] is False,
            holdout_measured,
            "HOLDOUT is unreachable from training and selection; the mutant fires",
            "all",
        )
    )
    profile = binding.execution_profile
    variants = {
        "data": replace(binding, dataset_snapshot_hash="1" * 64),
        "config": replace(binding, trainer_config_hash="2" * 64),
        "code": replace(binding, implementation_revision="3" * 40),
        "profile": replace(
            binding,
            execution_profile=replace(profile, blas_threads=profile.blas_threads + 1),
        ),
    }
    distinct = len({binding.run_hash, *(v.run_hash for v in variants.values())})
    checks.append(
        check_document(
            "G13.identity",
            distinct == 5,
            {"distinct_run_hashes": distinct},
            "data, config, code and profile changes each create a new run_hash",
            "eq",
        )
    )
    final = run_root / f"checkpoint-step-{FINAL_STEP:04d}"
    _, final_binding, final_manifest = load_registered_checkpoint(protocol, final)
    lineage = result.lineage
    lineage_measured = {
        "run_hash": final_manifest.run_hash
        == final_binding.run_hash
        == screen["run_hash"]
        == lineage.run_hash,
        "checkpoint_manifest_hash": final_manifest.manifest_hash
        == screen["checkpoint_manifest_hash"]
        == lineage.checkpoint_hash,
        "dataset_snapshot_hash": final_manifest.dataset_snapshot_hash
        == protocol.dataset_snapshot_hash
        == lineage.dataset_snapshot_hash,
        "execution_profile_hash": final_manifest.execution_profile_hash
        == screen["execution_profile_hash"]
        == lineage.execution_profile_hash,
        "weights_hash": final_manifest.model_export_hash == lineage.weights_hash,
        "model_state_hash": final_manifest.model_state_hash == lineage.model_state_hash,
        "protocol_hash": protocol.protocol_hash == screen["protocol_hash"],
        "training_revision_in_history": _is_ancestor(
            final_binding.implementation_revision
        ),
        "evaluator_revision_in_history": _is_ancestor(screen["evaluator_revision"]),
    }
    checks.append(
        check_document(
            "G13.lineage",
            all(lineage_measured.values()),
            lineage_measured,
            "the selected checkpoint resolves to its data, run, profile, protocol and revisions",
            "all",
        )
    )
    decision = select_checkpoint((result,), policy=protocol.selection_policy)
    synthetic_space = GridSearchSpace(
        learning_rates=(1.0e-4, 2.0e-4),
        physics_weight_scales=(1.0,),
        optimizer_steps=1,
        case_batch_size=1,
        maximum_trials=2,
    )
    better, worse = synthetic_space.trials
    synthetic = select_checkpoint(
        (
            TrialResult(
                replace(lineage, trial_id=better.trial_id),
                EvidenceRole.TRAIN,
                result.metrics,
            ),
            TrialResult(
                replace(lineage, trial_id=worse.trial_id),
                EvidenceRole.TRAIN,
                replace(
                    result.metrics, hybrid_loss=2.0 * result.metrics.hybrid_loss + 1.0
                ),
            ),
        ),
        policy=SelectionPolicy(synthetic_space, protocol.selection_policy.guardrails),
    )
    nonranked = _choose_nonranked_checkpoint_mutant_for_test(synthetic)
    recorded = (
        CheckpointLineage(**_strip(screen["selected_checkpoint"]))
        if screen["selected_checkpoint"]
        else None
    )
    selection_measured = {
        "selected_trial_id": decision.selected_lineage.trial_id,
        "selected_matches_screen_report": decision.selected_lineage == recorded,
        "screen_selection_status": screen["selection_status"],
        "failed_guardrails": screen["failed_guardrails"],
        "nonranked_mutant_detected": nonranked.trial_id
        != synthetic.selected_result.trial_id
        and synthetic.ranked_eligible_trial_ids == (better.trial_id, worse.trial_id),
    }
    checks.append(
        check_document(
            "G13.selection",
            decision.selected_lineage == recorded
            and screen["selection_status"] == "ELIGIBLE"
            and not screen["failed_guardrails"]
            and selection_measured["nonranked_mutant_detected"] is True,
            selection_measured,
            "the chosen checkpoint is the first ranked eligible trial under the frozen "
            "absolute guardrails; the CHOOSE_NONRANKED_CHECKPOINT mutant is detected",
            "all",
        )
    )
    summary, coverage, passed = run_pytest_with_coverage(
        REPOSITORY, ("tests/training",), ("piginot.training",)
    )
    checks.append(
        check_document(
            "G13.coverage",
            passed and coverage >= 80.0,
            coverage,
            80.0,
            "gte",
            tests=summary,
        )
    )
    return checks


def cp_freeze(run_root: Path) -> dict[str, Any]:
    g11 = _load(REPOSITORY / G11_PATH)
    g13 = _load(REPOSITORY / G13_PATH)
    if g11.get("verdict") != "PASS" or g13.get("verdict") != "PASS":
        raise RuntimeError("CP-FREEZE requires passing G11 and G13 gates")
    protocol = load_committed_protocol()
    screen = _load(REPOSITORY / SCREEN_PATH)
    final = run_root / f"checkpoint-step-{FINAL_STEP:04d}"
    model, binding, manifest = load_registered_checkpoint(protocol, final)
    if manifest.manifest_hash != screen["checkpoint_manifest_hash"]:
        raise ValueError("the checkpoint differs from the gated candidate")
    return {
        "schema_version": "1.0",
        "checkpoint_id": "CP-FREEZE",
        "campaign_id": "notched-v1",
        "step": "S13",
        "created_utc": utc_now(),
        "implementation_source_revision": git_head(REPOSITORY),
        "decision_record": "docs/adr/0025-s17-examples-and-second-family.md",
        "frozen_candidate": {
            "architecture_id": protocol.architecture_id,
            "architecture_module_digest": protocol.architecture_module_digest,
            "trial_id": screen["selected_checkpoint"]["trial_id"],
            "run_hash": binding.run_hash,
            "checkpoint_step": manifest.step,
            "checkpoint_manifest_hash": manifest.manifest_hash,
            "model_state_hash": manifest.model_state_hash,
            "weights_hash": manifest.model_export_hash,
            "model_safetensors_sha256": _sha256(final / "model.safetensors"),
            "protocol_hash": protocol.protocol_hash,
            "objective_hash": protocol.objective_hash,
            "dataset_snapshot_hash": binding.dataset_snapshot_hash,
            "normalization_hash": binding.normalization_hash,
            "execution_profile_hash": binding.execution_profile_hash,
            "initial_weights_hash": binding.initial_weights_hash,
            "training_revision": binding.implementation_revision,
            "evaluator_revision": screen["evaluator_revision"],
            "parameter_count": sum(p.numel() for p in model.parameters()),
        },
        "selection": {
            "policy": "preregistered absolute guardrails (ADR 0025 decision 6)",
            "guardrails": plain(protocol.selection_policy.guardrails),
            "status": screen["selection_status"],
            "train_screen_metrics": _strip(screen["trial_result"]["metrics"]),
        },
        "gates": {
            "G11": {
                "artifact": G11_PATH.as_posix(),
                "sha256": _sha256(REPOSITORY / G11_PATH),
            },
            "G13": {
                "artifact": G13_PATH.as_posix(),
                "sha256": _sha256(REPOSITORY / G13_PATH),
            },
        },
        "evidence": {
            "S13-PROTOCOL.json": _sha256(REPOSITORY / PROTOCOL_PATH),
            "S13-TRAIN-SCREEN.json": _sha256(REPOSITORY / SCREEN_PATH),
            "S13-G13-REPLAY-PROBE.json": _sha256(REPOSITORY / PROBE_PATH),
        },
        "authorization": {
            "development": "S14 may open DEVELOPMENT for exactly this frozen candidate",
            "holdout": "S14 may consume HOLDOUT once, for exactly this frozen candidate, "
            "after DEVELOPMENT calibration is fixed; consumption is recorded before reveal",
            "change_policy": "any change to weights, architecture, objective, data or "
            "conditioning voids this record",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.notched_v1.s13_close")
    sub = parser.add_subparsers(dest="command", required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("--s10-run-root", type=Path, required=True)
    screen.add_argument("--run-root", type=Path, required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--run-root", type=Path, required=True)
    probe.add_argument("--probe-run-root", type=Path, required=True)
    g11 = sub.add_parser("g11")
    g11.add_argument("--run-root", type=Path, required=True)
    g13 = sub.add_parser("g13")
    g13.add_argument("--run-root", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--run-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    final = arguments.run_root / f"checkpoint-step-{FINAL_STEP:04d}"
    if arguments.command == "screen":
        evaluation = evaluate_screen(
            s10_run_root=arguments.s10_run_root, checkpoint_path=final
        )
        report = screen_report(evaluation, arguments.run_root)
        (REPOSITORY / SCREEN_PATH).write_bytes(canonical_json_bytes(report))
        print(json.dumps(plain(evaluation.result.metrics), indent=1))
        print(report["selection_status"], report["failed_guardrails"])
        return 0 if evaluation.selected else 1
    if arguments.command == "probe":
        report = compare_replay(arguments.run_root, arguments.probe_run_root)
        (REPOSITORY / PROBE_PATH).write_bytes(canonical_json_bytes(report))
        print(report["g13_determinism"], report["g13_resume"])
        return 0
    if arguments.command == "g11":
        document = g11_gate(final)
        write_gate_document(REPOSITORY / G11_PATH, document)
    elif arguments.command == "g13":
        document = gate_document(
            "G13",
            "S13",
            REPOSITORY,
            g13_checks(arguments.run_root),
            evidence={
                "protocol_sha256": _sha256(REPOSITORY / PROTOCOL_PATH),
                "train_screen_sha256": _sha256(REPOSITORY / SCREEN_PATH),
                "replay_probe_sha256": _sha256(REPOSITORY / PROBE_PATH),
                "g11_sha256": _sha256(REPOSITORY / G11_PATH),
            },
        )
        write_gate_document(REPOSITORY / G13_PATH, document)
    else:
        payload = cp_freeze(arguments.run_root)
        (REPOSITORY / CP_FREEZE_PATH).write_bytes(canonical_json_bytes(payload))
        print(json.dumps(payload["frozen_candidate"], indent=1))
        return 0
    print(document["verdict"], document["verdict_reason"])
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = (
    "CP_FREEZE_PATH",
    "G11_PATH",
    "G13_PATH",
    "SCREEN_PATH",
    "compare_replay",
    "cp_freeze",
    "evaluate_screen",
    "fem_results",
    "g11_gate",
    "g13_checks",
    "lifting_parameter_vectors",
    "load_registered_checkpoint",
    "root_vertices",
    "screen_report",
    "vertex_targets",
)
