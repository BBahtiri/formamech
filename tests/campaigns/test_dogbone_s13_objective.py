from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from campaigns.dogbone_v1.s11_contingency import (
    BOUNDARY_TAG_NAMES,
    DERIVED_PARAMETER_NAMES,
    RAW_PARAMETER_NAMES,
)
from campaigns.dogbone_v1.s11_contingency_tensors import ArchitectureCaseTensors
from campaigns.dogbone_v1.s13_objective import (
    DOGBONE_S13_OBJECTIVE_TERMS,
    DogboneS13ObjectiveBuilder,
    DogboneS13TrainingCase,
    _constitutive_with_orientation_penalty,
    _physics_material,
)
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    CanonicalQuantity,
    ExecutionProfile,
    MaterialSpec,
    Unit,
    hash_spec,
)
from piginot.datasets import (
    CollocationKind,
    DatasetCaseRecord,
    RoleDatasetSnapshot,
    SampleTrace,
    SemanticCollocationQuota,
    sample_request_collocation,
)
from piginot.doe import EvidenceRole, PublishedCaseStatus
from piginot.fem import FemRequest, FemSolverConfig
from piginot.objectives import (
    EquilibriumForm,
    NondimensionalScales,
    ObjectiveMode,
    ObjectiveRegistry,
    ObjectiveTerm,
    constitutive_residual,
)
from piginot.operators import OperatorConfig, PiGinotOperator
from piginot.training.adaptive import AdaptiveWeightConfig
from piginot.training.engine import (
    DeterministicTrainerConfig,
    TrainingPopulation,
    model_state_hash,
    train_to_checkpoint,
)
from piginot.training.identity import TrainingRunBinding
from piginot.training.scheduler import PlateauSchedulerConfig
from piginot.training.search import GridSearchSpace, TrialConfig
from piginot.training.state import DeterministicSamplerPlan

_DTYPE = torch.float64


def _mesh() -> CanonicalMesh:
    return CanonicalMesh(
        mesh_id="s13-objective-square-v1",
        nodes=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        cells=((0, 1, 2), (0, 2, 3)),
        cell_type="triangle",
        boundary_facets=((0, 1), (1, 2), (2, 3), (3, 0)),
        semantic_tags={
            "bottom": (0,),
            "gauge_top": (2,),
            "left_symmetry": (3,),
            "right_arc": (2,),
            "right_grip": (1,),
        },
        region_ids=(1, 1),
        unit=Unit.MM,
        orientation="counterclockwise",
        topology_signature="s13-objective-square-v1",
        quality_metrics={"minimum_scaled_jacobian": 1.0},
        source_provenance={"fixture": "s13-objective"},
    )


def _request(case_id: str = "train-001") -> FemRequest:
    return FemRequest(
        request_id=case_id,
        mesh=_mesh(),
        material=MaterialSpec(
            name="dogbone-test-material",
            model="finite_strain_plane_stress_neo_hookean",
            properties={
                "elastic_modulus": CanonicalQuantity(760.0, Unit.MPA),
                "poisson_ratio": CanonicalQuantity(0.23, Unit.ONE),
            },
        ),
        boundary_conditions=(
            BoundaryConditionSpec(
                "left_x_symmetry", "left_symmetry", "symmetry", ("x",)
            ),
            BoundaryConditionSpec("bottom_y_symmetry", "bottom", "symmetry", ("y",)),
            BoundaryConditionSpec(
                "right_grip_displacement",
                "right_grip",
                "dirichlet",
                ("x",),
                (CanonicalQuantity(0.1, Unit.MM),),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=2),
        execution_profile=ExecutionProfile(
            python_version="3.11.9",
            library_versions={"torch": torch.__version__},
            blas_threads=1,
            device="cpu",
            dtype="float64",
            rng_algorithm="none",
            seed=0,
        ),
    )


def _snapshot(request: FemRequest) -> RoleDatasetSnapshot:
    record = DatasetCaseRecord(
        case_id=request.request_id,
        leakage_group_id="group-train-001",
        role=EvidenceRole.TRAIN,
        primitive_coordinates={"L_total": 2.0},
        derived_coordinates={"length_to_gauge_ratio": 2.0},
        strata={"length_to_gauge_ratio": 0},
        request_recipe_hash=request.recipe_hash,
        request_content_hash=hash_spec(request, "content"),
        mesh_id=request.mesh.mesh_id,
        mesh_content_hash=hash_spec(request.mesh, "content"),
        success=True,
        sample_trace=SampleTrace(fem_result_hash="d" * 64),
        store_content_hash="e" * 64,
        failure_code=None,
        message="succeeded",
    )
    return RoleDatasetSnapshot(
        campaign_hash="a" * 64,
        public_case_manifest_hash="b" * 64,
        request_plan_hash="c" * 64,
        role=EvidenceRole.TRAIN,
        cases=(record,),
    )


def _collocation(request: FemRequest):  # type: ignore[no-untyped-def]
    return sample_request_collocation(
        _snapshot(request),
        request.request_id,
        request,
        stream_index=0,
        interior_count=8,
        boundary_count=64,
        interface_count=0,
        qoi_quotas=(
            SemanticCollocationQuota(
                quota_id="stress-critical-right-arc-v1",
                kind=CollocationKind.BOUNDARY,
                count=16,
                semantic_tag="right_arc",
            ),
        ),
    )


def _tensor_case(
    request: FemRequest, *, role: EvidenceRole = EvidenceRole.TRAIN
) -> ArchitectureCaseTensors:
    boundary_xy = torch.tensor(
        [[0.5, 0.0], [0.25, 1.0], [0.0, 0.5], [0.75, 1.0], [1.0, 0.5]],
        dtype=_DTYPE,
    )
    boundary_tags = torch.eye(len(BOUNDARY_TAG_NAMES), dtype=_DTYPE)
    return ArchitectureCaseTensors(
        case_id=request.request_id,
        leakage_group_id="group-train-001",
        role=role,
        status=PublishedCaseStatus.ACTIVE,
        parameter_names=(*RAW_PARAMETER_NAMES, *DERIVED_PARAMETER_NAMES),
        normalized_parameters=torch.zeros(11, dtype=_DTYPE),
        normalization_hash="f" * 64,
        boundary_tag_names=BOUNDARY_TAG_NAMES,
        boundary_xy=boundary_xy,
        boundary_tags=boundary_tags,
        boundary_normals=torch.tensor(
            [[0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            dtype=_DTYPE,
        ),
        boundary_weights=torch.ones(5, dtype=_DTYPE),
        boundary_mask=torch.ones(5, dtype=torch.bool),
        query_xy=torch.tensor(request.mesh.nodes, dtype=_DTYPE),
        target_u=torch.zeros((4, 2), dtype=_DTYPE),
        target_P=torch.zeros((4, 2, 2), dtype=_DTYPE),
        qoi_indices=torch.tensor([2, 3], dtype=torch.int64),
    )


def _training_case(
    *, role: EvidenceRole = EvidenceRole.TRAIN
) -> DogboneS13TrainingCase:
    request = _request()
    return DogboneS13TrainingCase(
        tensors=_tensor_case(request, role=role),
        request=request,
        collocation=_collocation(request),
    )


def _model() -> PiGinotOperator:
    torch.manual_seed(1234)
    model = PiGinotOperator(
        OperatorConfig(
            parameter_names=RAW_PARAMETER_NAMES,
            boundary_tag_names=BOUNDARY_TAG_NAMES,
            hidden_dim=8,
            latent_token_count=4,
            num_attention_heads=2,
            num_fourier_frequencies=2,
            encoder_self_attention_layers=1,
            decoder_cross_attention_layers=1,
        )
    ).to(dtype=_DTYPE)
    with torch.no_grad():
        model.u_head.weight.mul_(0.01)
        model.u_head.bias.fill_(0.05)
        model.p_head.weight.mul_(0.01)
        model.p_head.bias.fill_(0.02)
    return model


def test_real_operator_objective_builds_all_strong_hybrid_terms_and_backpropagates() -> (
    None
):
    training_case = _training_case()
    builder = DogboneS13ObjectiveBuilder((training_case,))
    model = _model()

    residuals = builder(model, (training_case.tensors.case_id,), 1)

    assert tuple(residuals.values) == DOGBONE_S13_OBJECTIVE_TERMS
    assert all(value.requires_grad for value in residuals.values.values())
    assert all(
        bool(torch.isfinite(value).all().item()) for value in residuals.values.values()
    )
    registry = ObjectiveRegistry(
        mode=ObjectiveMode.HYBRID,
        equilibrium_form=EquilibriumForm.STRONG,
    )
    evaluation = registry.evaluate(residuals, role=EvidenceRole.TRAIN)
    torch.autograd.backward(evaluation.total)
    gradients = tuple(
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    )
    assert gradients
    assert all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    assert any(bool((gradient != 0.0).any().item()) for gradient in gradients)


def test_objective_uses_physical_stress_and_displacement_scales() -> None:
    training_case = _training_case()
    builder = DogboneS13ObjectiveBuilder((training_case,))
    model = _model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.u_head.bias.copy_(torch.tensor([0.5, 0.25], dtype=_DTYPE))
        model.p_head.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=_DTYPE))

    residuals = builder(model, (training_case.tensors.case_id,), 1)

    expected_constitutive = torch.tensor(
        [0.1, 0.2, 0.3, 0.4],
        dtype=_DTYPE,
    ).expand_as(residuals.values[ObjectiveTerm.CONSTITUTIVE])
    torch.testing.assert_close(
        residuals.values[ObjectiveTerm.CONSTITUTIVE],
        expected_constitutive,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    dirichlet = residuals.values[ObjectiveTerm.DIRICHLET]
    assert bool(torch.isclose(dirichlet, torch.tensor(0.5, dtype=_DTYPE)).any())
    assert bool(torch.isclose(dirichlet, torch.tensor(0.25, dtype=_DTYPE)).any())
    assert bool(torch.isclose(dirichlet, torch.tensor(-0.5, dtype=_DTYPE)).any())


def test_orientation_penalty_preserves_exact_constitutive_residual_when_valid() -> None:
    material = _physics_material(_request().material)
    scales = NondimensionalScales(1.0, 0.1, 760.0)
    stress = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=_DTYPE)
    gradient = torch.tensor([[[[0.1, 0.0], [0.0, 0.2]]]], dtype=_DTYPE)

    actual = _constitutive_with_orientation_penalty(
        stress,
        gradient,
        material,
        scales,
    )
    expected = constitutive_residual(stress, gradient, material, scales)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_orientation_penalty_counts_every_folded_point_and_backpropagates() -> None:
    material = _physics_material(_request().material)
    scales = NondimensionalScales(1.0, 0.1, 760.0)
    stress = torch.zeros((1, 3, 2, 2), dtype=_DTYPE, requires_grad=True)
    gradient = torch.tensor(
        [
            [
                [[-1.5, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[-1.25, 0.0], [0.0, 0.0]],
            ]
        ],
        dtype=_DTYPE,
        requires_grad=True,
    )

    residual = _constitutive_with_orientation_penalty(
        stress,
        gradient,
        material,
        scales,
    )

    assert residual.shape == (1, 3, 2, 2)
    assert torch.isfinite(residual).all()
    assert torch.all(residual[0, 0] > 0.0)
    assert torch.all(residual[0, 2] > 0.0)
    assert torch.all(residual[0, 1] == 0.0)
    torch.autograd.backward(residual.square().mean())
    assert gradient.grad is not None
    assert torch.isfinite(gradient.grad).all()
    assert gradient.grad[0, 0, 0, 0] != 0.0
    assert gradient.grad[0, 2, 0, 0] != 0.0


def test_orientation_penalty_keeps_s12_result_on_valid_rows_of_mixed_batch() -> None:
    material = _physics_material(_request().material)
    scales = NondimensionalScales(1.0, 0.1, 760.0)
    stress = torch.tensor(
        [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]],
        dtype=_DTYPE,
    )
    gradient = torch.tensor(
        [[[[0.1, 0.0], [0.0, 0.2]], [[-1.5, 0.0], [0.0, 0.0]]]],
        dtype=_DTYPE,
    )

    actual = _constitutive_with_orientation_penalty(
        stress, gradient, material, scales
    )
    expected = constitutive_residual(stress[:, :1], gradient[:, :1], material, scales)

    torch.testing.assert_close(actual[:, :1], expected, rtol=0.0, atol=0.0)
    assert torch.all(actual[:, 1:] > 0.0)


def test_orientation_penalty_is_nonzero_at_singular_and_margin_values() -> None:
    material = _physics_material(_request().material)
    scales = NondimensionalScales(1.0, 0.1, 760.0)
    stress = torch.zeros((1, 3, 2, 2), dtype=_DTYPE)
    gradient = torch.tensor(
        [
            [
                [[-1.0, 0.0], [0.0, 0.0]],
                [[-1.0 + 1.0e-13, 0.0], [0.0, 0.0]],
                [[-1.0 + 1.0e-12, 0.0], [0.0, 0.0]],
            ]
        ],
        dtype=_DTYPE,
        requires_grad=True,
    )

    residual = _constitutive_with_orientation_penalty(
        stress,
        gradient,
        material,
        scales,
    )

    assert residual.shape == (1, 3, 2, 2)
    assert torch.all(residual >= 10.0)
    torch.autograd.backward(residual.square().sum())
    assert gradient.grad is not None
    assert torch.all(gradient.grad[0, :, 0, 0] < 0.0)


def test_objective_refuses_non_train_cases_before_model_execution() -> None:
    request = _request()

    with pytest.raises(PermissionError, match="DEVELOPMENT"):
        DogboneS13TrainingCase(
            tensors=_tensor_case(request, role=EvidenceRole.DEVELOPMENT),
            request=request,
            collocation=_collocation(request),
        )


def test_objective_refuses_unknown_material_semantics() -> None:
    training_case = _training_case()
    bad_request = replace(
        training_case.request,
        material=replace(training_case.request.material, model="unknown-model"),
    )

    with pytest.raises(ValueError, match="material model"):
        DogboneS13TrainingCase(
            tensors=training_case.tensors,
            request=bad_request,
            collocation=training_case.collocation,
        )


def test_objective_refuses_unknown_or_duplicate_sample_ids() -> None:
    training_case = _training_case()
    builder = DogboneS13ObjectiveBuilder((training_case,))
    model = _model()

    with pytest.raises(ValueError, match="unique"):
        builder(model, (training_case.tensors.case_id,) * 2, 1)
    with pytest.raises(ValueError, match="unknown"):
        builder(model, ("train-unknown",), 1)


def test_objective_executes_one_real_trainer_step(tmp_path: Path) -> None:
    training_case = _training_case()
    builder = DogboneS13ObjectiveBuilder((training_case,))
    model = _model()
    population = TrainingPopulation(
        dataset_snapshot_hash=_snapshot(training_case.request).content_hash,
        public_manifest_hash="a" * 64,
        role=EvidenceRole.TRAIN,
        sample_ids=builder.sample_ids,
    )
    plan = DeterministicSamplerPlan(
        sampler_id="dogbone-s13-objective-one-step-v1",
        population_size=1,
        batches=((0,),),
    )
    trial = TrialConfig(
        learning_rate=1.0e-4,
        physics_weight_scale=1.0,
        optimizer_steps=1,
        case_batch_size=1,
    )
    config = DeterministicTrainerConfig(
        search_space=GridSearchSpace(
            learning_rates=(trial.learning_rate,),
            physics_weight_scales=(trial.physics_weight_scale,),
            optimizer_steps=trial.optimizer_steps,
            case_batch_size=trial.case_batch_size,
            maximum_trials=1,
        ),
        trial=trial,
        adaptive=AdaptiveWeightConfig(
            beta=0.1,
            warmup_steps=2,
            update_period=2,
        ),
        scheduler=PlateauSchedulerConfig(
            patience=2,
            reduction_factor=0.5,
            absolute_threshold=0.0,
            cooldown_steps=1,
            minimum_learning_rate=1.0e-8,
        ),
        gradient_clip_norm=10.0,
    )
    registry = ObjectiveRegistry(
        mode=ObjectiveMode.HYBRID,
        equilibrium_form=EquilibriumForm.STRONG,
    )
    seed = 20260921
    binding = TrainingRunBinding(
        dataset_snapshot_hash=population.dataset_snapshot_hash,
        public_manifest_hash=population.public_manifest_hash,
        normalization_hash=training_case.tensors.normalization_hash,
        architecture_id="dogbone-s13-objective-fixture-v1",
        architecture_module_digest="blake2b256:" + "b" * 64,
        objective_registry_hash=hash_spec(registry, "recipe"),
        trainer_config_hash=config.config_hash,
        sampler_plan_hash=plan.plan_hash,
        initial_weights_hash=model_state_hash(model),
        implementation_revision="c" * 40,
        seed=seed,
        execution_profile=ExecutionProfile(
            python_version="3.11.9",
            library_versions={"torch": torch.__version__},
            blas_threads=1,
            device="cpu",
            dtype="float64",
            rng_algorithm="torch-cpu-mt19937",
            seed=seed,
        ),
    )

    result = train_to_checkpoint(
        tmp_path / "checkpoint",
        model=model,
        population=population,
        plan=plan,
        binding=binding,
        config=config,
        base_registry=registry,
        objective=builder,
    )

    assert result.step == 1
    assert result.sampler_state.next_step == 1
    assert result.checkpoint.path.is_dir()
    assert set(result.final_losses) == set(DOGBONE_S13_OBJECTIVE_TERMS)
