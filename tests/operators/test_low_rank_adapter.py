"""Tests for Feature B: low-rank per-geometry adapters (Delta-theta_g)."""

from __future__ import annotations

import pytest
import torch

from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from piginot.lifting import AffineDirichletLifting
from piginot.objectives import PotentialEnergyConfig, TriangleDomain
from piginot.operators import (
    GEOMETRY_FEATURE_NAMES,
    AdaptationPlan,
    AdaptationStatus,
    AdaptedBoundaryEnergyOperator,
    BoundaryEnergyConfig,
    BoundaryEnergyOperator,
    BoundaryRoleMap,
    EnergyOperatorConfig,
    GeometryOODDetector,
    GeometryTokenizer,
    LowRankAdapterConfig,
    adapt_boundary_operator_energy,
    boundary_energy_module_digest,
    compute_geometry_descriptor,
)


def _neo_hookean() -> MaterialSpec:
    return MaterialSpec(
        name="nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(100.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.3, Unit.ONE).to_canonical(),
        },
    )


def _base_operator(seed: int = 7) -> BoundaryEnergyOperator:
    torch.manual_seed(seed)
    cfg = BoundaryEnergyConfig(
        energy=EnergyOperatorConfig(
            parameter_names=("strain",),
            hidden_dim=16,
            latent_token_count=6,
            parameter_hidden_dim=16,
            num_attention_heads=4,
            decoder_layers=2,
            positional_frequencies=3,
        ),
        boundary_feature_names=GEOMETRY_FEATURE_NAMES,
    )
    return BoundaryEnergyOperator(cfg).to(torch.float64)


def _rect_mesh(w: float = 2.0, h: float = 1.0):
    return canonicalize_mesh(
        MeshPayload(
            ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h)),
            ((0, 1, 2), (0, 2, 3)),
            CellType.TRIANGLE,
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            {"bottom": (0,), "right": (1,), "top": (2,), "left": (3,)},
            (1, 1),
            Unit.MM,
            {"kind": f"rect-{w}x{h}"},
        )
    )


def _roles() -> BoundaryRoleMap:
    return BoundaryRoleMap(
        {
            "left": "dirichlet_xy",
            "right": "dirichlet_x",
            "bottom": "symmetry",
            "top": "traction_free",
        }
    )


def _exact_uniaxial_lifting(length: float, strain: float):
    def lifting(query_xy: torch.Tensor) -> AffineDirichletLifting:
        x = query_xy[..., 0]
        y = query_xy[..., 1]
        # Exact BC: u_x(0, y) = 0, u_x(L, y) = strain * L, u_y(x, 0) = 0.
        offset_x = strain * x
        offset_y = torch.zeros_like(y)
        scale_x = (x / length) * (1.0 - x / length)
        scale_y = y
        return AffineDirichletLifting(
            offset=torch.stack((offset_x, offset_y), dim=-1),
            scale=torch.stack((scale_x, scale_y), dim=-1),
        )

    return lifting


def test_adapter_budget_guard_rejects_excessive_steps() -> None:
    with pytest.raises(ValueError, match="exceeds budget fraction"):
        LowRankAdapterConfig(
            rank=4,
            alpha=4.0,
            max_steps=25,
            from_scratch_reference_steps=200,
            max_step_budget_fraction=0.10,
        )


def test_adapted_operator_is_zero_shot_identical_at_step_zero_and_freezes_base() -> (
    None
):
    base = _base_operator()
    cfg = LowRankAdapterConfig(
        rank=4, alpha=4.0, max_steps=10, from_scratch_reference_steps=100, seed=1
    )
    adapted = AdaptedBoundaryEnergyOperator(base, cfg)
    assert adapted.adapter_parameter_count() > 0
    assert adapted.adapter_parameter_count() < adapted.base_parameter_count()

    mesh = _rect_mesh()
    batch = GeometryTokenizer().tokenize_batch([(mesh, _roles())], dtype=torch.float64)
    queries = torch.tensor([[[0.5, 0.3], [1.4, 0.7]]], dtype=torch.float64)
    params = torch.tensor([[0.1]], dtype=torch.float64)
    lift = _exact_uniaxial_lifting(2.0, 0.1)

    enc_base = base.encode_boundary(
        boundary_xy=batch.boundary_xy,
        boundary_normals=batch.boundary_normals,
        boundary_features=batch.boundary_features,
        weights=batch.weights,
        mask=batch.mask,
        frame=batch.frame,
    )
    enc_adapt = adapted.encode_boundary(
        boundary_xy=batch.boundary_xy,
        boundary_normals=batch.boundary_normals,
        boundary_features=batch.boundary_features,
        weights=batch.weights,
        mask=batch.mask,
        frame=batch.frame,
    )
    u_base = base.displacement(
        encoding=enc_base, query_xy=queries, parameters=params, lifting=lift
    )
    u_adapt = adapted.displacement(
        encoding=enc_adapt, query_xy=queries, parameters=params, lifting=lift
    )
    assert torch.equal(u_base, u_adapt)


def test_energy_adaptation_is_deterministic_preserves_bcs_and_freezes_base() -> None:
    base = _base_operator()
    base_snapshot = {k: v.clone() for k, v in base.state_dict().items()}
    mesh = _rect_mesh(2.0, 1.0)
    batch = GeometryTokenizer().tokenize_batch([(mesh, _roles())], dtype=torch.float64)
    domain = TriangleDomain(
        vertices=torch.tensor(mesh.nodes, dtype=torch.float64),
        cells=torch.tensor(mesh.cells, dtype=torch.int64),
    )
    cfg = LowRankAdapterConfig(
        rank=4,
        alpha=4.0,
        max_steps=8,
        from_scratch_reference_steps=100,
        learning_rate=2e-2,
        seed=42,
    )
    plan = AdaptationPlan(
        base_model_digest=boundary_energy_module_digest(base),
        target_mesh_id=mesh.mesh_id,
        adapter_config=cfg,
        energy_config=PotentialEnergyConfig(points_per_cell=3),
    )
    params = torch.tensor([[0.15]], dtype=torch.float64)
    lift = _exact_uniaxial_lifting(2.0, 0.15)
    material = _neo_hookean()

    adapted1, res1 = adapt_boundary_operator_energy(
        base,
        plan=plan,
        domain=domain,
        boundary_xy=batch.boundary_xy,
        boundary_normals=batch.boundary_normals,
        boundary_features=batch.boundary_features,
        boundary_weights=batch.boundary_weights
        if hasattr(batch, "boundary_weights")
        else batch.weights,
        boundary_mask=batch.mask,
        frame=batch.frame,
        parameters=params,
        lifting=lift,
        material=material,
    )
    adapted2, res2 = adapt_boundary_operator_energy(
        base,
        plan=plan,
        domain=domain,
        boundary_xy=batch.boundary_xy,
        boundary_normals=batch.boundary_normals,
        boundary_features=batch.boundary_features,
        boundary_weights=batch.weights,
        boundary_mask=batch.mask,
        frame=batch.frame,
        parameters=params,
        lifting=lift,
        material=material,
    )
    assert adapted1 is not None and adapted2 is not None
    assert res1.status is AdaptationStatus.COMPLETED
    assert res1.claim_level == "NOT_VERIFIED"
    assert res1.steps_executed == 8
    assert res1.step_budget_ratio == pytest.approx(0.08)
    assert res1.adapter_digest == res2.adapter_digest
    assert res1.energy_history == res2.energy_history

    # Base operator weights were never mutated.
    for key, orig in base_snapshot.items():
        assert torch.equal(base.state_dict()[key], orig)

    # Exact Dirichlet BCs remain satisfied after adaptation.
    bc_pts = torch.tensor(
        [[[0.0, 0.25], [0.0, 0.8], [2.0, 0.5], [1.0, 0.0]]], dtype=torch.float64
    )
    enc = adapted1.encode_boundary(
        boundary_xy=batch.boundary_xy,
        boundary_normals=batch.boundary_normals,
        boundary_features=batch.boundary_features,
        weights=batch.weights,
        mask=batch.mask,
        frame=batch.frame,
    )
    with torch.no_grad():
        u_bc = adapted1.displacement(
            encoding=enc, query_xy=bc_pts, parameters=params, lifting=lift
        )
    assert float(u_bc[0, 0, 0]) == pytest.approx(0.0, abs=1e-14)
    assert float(u_bc[0, 1, 0]) == pytest.approx(0.0, abs=1e-14)
    assert float(u_bc[0, 2, 0]) == pytest.approx(0.30, abs=1e-14)
    assert float(u_bc[0, 3, 1]) == pytest.approx(0.0, abs=1e-14)

    # Adapter-only state dict roundtrips identically into a fresh wrapper.
    fresh = AdaptedBoundaryEnergyOperator(base, cfg)
    fresh.load_adapter_state_dict(adapted1.adapter_state_dict())
    assert fresh.adapter_digest() == adapted1.adapter_digest()

    # Full forward() serves displacement and constitutive plane-stress first Piola.
    served = fresh(
        encoding=enc,
        query_xy=bc_pts,
        parameters=params,
        lifting=lift,
        materials=(material,),
    )
    assert served.u.shape == (1, 4, 2)
    assert served.P.shape == (1, 4, 2, 2)
    assert res1.document()["claim_level"] == "NOT_VERIFIED"

    with pytest.raises(ValueError, match="keys"):
        fresh.load_adapter_state_dict({})
    with pytest.raises(ValueError, match="rank"):
        LowRankAdapterConfig(rank=0)
    with pytest.raises(ValueError, match="alpha"):
        LowRankAdapterConfig(alpha=-1.0)


def test_energy_adaptation_refuses_unsupported_topology() -> None:
    base = _base_operator()
    train_meshes = [_rect_mesh(2.0, 1.0), _rect_mesh(2.2, 1.0)]
    detector = GeometryOODDetector.fit(
        [compute_geometry_descriptor(m, _roles()) for m in train_meshes]
    )
    # Use a plate-with-hole mesh whose hole count is unsupported by detector.
    from tests.operators.test_geometry_tokenizer import (
        _hole_roles,
        _plate_with_hole_mesh,
    )

    hole_mesh = _plate_with_hole_mesh()
    ood = detector.score(hole_mesh, _hole_roles())
    plan = AdaptationPlan(
        base_model_digest=boundary_energy_module_digest(base),
        target_mesh_id=hole_mesh.mesh_id,
        adapter_config=LowRankAdapterConfig(
            rank=4, alpha=4.0, max_steps=5, from_scratch_reference_steps=100
        ),
        energy_config=PotentialEnergyConfig(points_per_cell=2),
        ood_report=ood,
    )
    batch = GeometryTokenizer().tokenize_batch(
        [(hole_mesh, _hole_roles())], dtype=torch.float64
    )
    domain = TriangleDomain(
        vertices=torch.tensor(hole_mesh.nodes, dtype=torch.float64),
        cells=torch.tensor(hole_mesh.cells, dtype=torch.int64),
    )
    adapted, result = adapt_boundary_operator_energy(
        base,
        plan=plan,
        domain=domain,
        boundary_xy=batch.boundary_xy,
        boundary_normals=batch.boundary_normals,
        boundary_features=batch.boundary_features,
        boundary_weights=batch.weights,
        boundary_mask=batch.mask,
        frame=batch.frame,
        parameters=torch.tensor([[0.1]], dtype=torch.float64),
        lifting=_exact_uniaxial_lifting(4.0, 0.1),
        material=_neo_hookean(),
    )
    assert adapted is None
    assert result.status is AdaptationStatus.REFUSED_UNSUPPORTED_TOPOLOGY
    assert result.steps_executed == 0
    assert result.adapter_digest is None


def test_lora_adaptation_machine_invariants_and_save_refusal(tmp_path) -> None:
    from piginot.operators import (
        LORA_ADAPTATION_MACHINE,
        AdaptationPhase,
        AdaptationResult,
        save_geometry_adapter,
    )

    assert LORA_ADAPTATION_MACHINE.can_transition(
        AdaptationPhase.PLANNED,
        AdaptationPhase.RUNNING,
        trigger="start_energy_adaptation",
    )
    assert LORA_ADAPTATION_MACHINE.is_terminal(AdaptationPhase.FAILED_INADMISSIBLE)

    # COMPLETED requires adapter_digest and minimum_determinant > 0
    with pytest.raises(ValueError, match="adapter_digest"):
        AdaptationResult(
            plan_hash="a" * 64,
            status=AdaptationStatus.COMPLETED,
            claim_level="NOT_VERIFIED",
            steps_executed=5,
            step_budget_ratio=0.05,
            adapter_parameter_count=10,
            base_parameter_count=100,
            initial_energy=1.0,
            final_energy=0.5,
            minimum_determinant=0.9,
            energy_history=(1.0, 0.5),
            adapter_digest=None,
            reason="missing digest",
        )
    with pytest.raises(ValueError, match="minimum_determinant > 0"):
        AdaptationResult(
            plan_hash="a" * 64,
            status=AdaptationStatus.COMPLETED,
            claim_level="NOT_VERIFIED",
            steps_executed=5,
            step_budget_ratio=0.05,
            adapter_parameter_count=10,
            base_parameter_count=100,
            initial_energy=1.0,
            final_energy=0.5,
            minimum_determinant=-0.1,
            energy_history=(1.0, 0.5),
            adapter_digest="d" * 64,
            reason="inverted",
        )
    # Non-COMPLETED forbids non-None adapter_digest
    with pytest.raises(ValueError, match="adapter_digest=None"):
        AdaptationResult(
            plan_hash="a" * 64,
            status=AdaptationStatus.FAILED_INADMISSIBLE,
            claim_level="NOT_VERIFIED",
            steps_executed=2,
            step_budget_ratio=0.02,
            adapter_parameter_count=10,
            base_parameter_count=100,
            initial_energy=1.0,
            final_energy=1.0,
            minimum_determinant=-0.2,
            energy_history=(1.0,),
            adapter_digest="d" * 64,
            reason="should not carry digest",
        )

    base = _base_operator()
    cfg = LowRankAdapterConfig(
        rank=4, alpha=4.0, max_steps=5, from_scratch_reference_steps=100
    )
    adapted = AdaptedBoundaryEnergyOperator(base, cfg)
    failed_res = AdaptationResult(
        plan_hash="a" * 64,
        status=AdaptationStatus.FAILED_INADMISSIBLE,
        claim_level="NOT_VERIFIED",
        steps_executed=2,
        step_budget_ratio=0.02,
        adapter_parameter_count=adapted.adapter_parameter_count(),
        base_parameter_count=adapted.base_parameter_count(),
        initial_energy=1.0,
        final_energy=1.0,
        minimum_determinant=-0.2,
        energy_history=(1.0,),
        adapter_digest=None,
        reason="failed",
    )
    with pytest.raises(ValueError, match="non-COMPLETED"):
        save_geometry_adapter(
            tmp_path / "bad-adapter",
            adapted_operator=adapted,
            result=failed_res,
        )
