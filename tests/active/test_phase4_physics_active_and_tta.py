"""Phase 4 tests: BC/load geometry tokenization, DWR active learning, feedback ledger & TTA."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from formamech.active import (
    FALLBACK_LEDGER_SCHEMA,
    AcquisitionCandidate,
    AcquisitionPlan,
    AcquisitionWeights,
    FallbackFeedbackLedger,
    FallbackReturnRecord,
    select_acquisition_batch,
)

from piginot.bundle import load_bundle
from piginot.certify import verify_receipt_document
from piginot.cli import parse_request
from piginot.contracts import CanonicalMesh, CanonicalQuantity, MaterialSpec, Unit
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from piginot.objectives.energy import TriangleDomain
from piginot.operators import (
    GEOMETRY_FEATURE_NAMES,
    GEOMETRY_LOAD_FEATURE_NAMES,
    AxisAlignedTensionLifting,
    BoundaryEnergyConfig,
    BoundaryEnergyOperator,
    BoundaryLoadMap,
    BoundaryRoleMap,
    EnergyFormOperator,
    EnergyOperatorConfig,
    GeometryTokenizer,
    QueryFrame,
)
from piginot.serving import (
    ServingMode,
    TestTimeLoraConfig,
    adapt_operator_at_test_time,
    load_recipe,
    serve,
)
from tests.platform_support import windows_safe_environment

ROOT = Path(__file__).resolve().parents[2]
NOTCHED_BUNDLE = ROOT / "bundles" / "notched-v1-cp-freeze"
NOTCHED_REQUEST = ROOT / "bundles" / "requests" / "notched-in-domain.json"


def _rect_mesh() -> CanonicalMesh:
    return canonicalize_mesh(
        MeshPayload(
            ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
            ((0, 1, 2), (0, 2, 3)),
            CellType.TRIANGLE,
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            {"bottom": (0,), "right": (1,), "top": (2,), "left": (3,)},
            (1, 1),
            Unit.MM,
            {"kind": "rect"},
        )
    )


def test_geometry_tokenizer_load_embeddings_and_v1_backward_compat() -> None:
    mesh = _rect_mesh()
    roles = BoundaryRoleMap(
        roles={
            "left": "dirichlet_xy",
            "right": "dead_traction",
            "top": "follower_pressure",
            "bottom": "traction_free",
        }
    )
    loads = BoundaryLoadMap(
        loads={
            "right": (0.5, -0.125, 0.0),
            "top": (0.0, 0.0, 0.25),
        }
    )

    # Default (v1): exactly 7 features, backward-compatible
    tok_v1 = GeometryTokenizer(subdivisions_per_facet=2)
    assert tok_v1.feature_names == GEOMETRY_FEATURE_NAMES
    batch_v1 = tok_v1.tokenize_batch([(mesh, roles)])
    assert batch_v1.boundary_features.shape[-1] == 7

    # Extended (v2): 12 features including dead_traction, follower_pressure, normalized loads
    tok_v2 = GeometryTokenizer(
        subdivisions_per_facet=2,
        include_load_embeddings=True,
    )
    assert tok_v2.feature_names == GEOMETRY_LOAD_FEATURE_NAMES
    assert len(GEOMETRY_LOAD_FEATURE_NAMES) == 12
    batch_v2 = tok_v2.tokenize_batch([(mesh, roles)], load_maps=[loads])
    feats = batch_v2.boundary_features[0]
    assert feats.shape[-1] == 12

    # Channels 7..11 must be populated for right (dead_traction) and top (follower_pressure)
    dead_mask = feats[:, 7] > 0.5
    follower_mask = feats[:, 8] > 0.5
    assert bool(dead_mask.any().item())
    assert bool(follower_mask.any().item())
    # Normalized traction (10/20 = 0.5, -2.5/20 = -0.125)
    assert torch.allclose(
        feats[dead_mask, 9],
        torch.full((int(dead_mask.sum()),), 0.5, dtype=feats.dtype),
    )
    assert torch.allclose(
        feats[dead_mask, 10],
        torch.full((int(dead_mask.sum()),), -0.125, dtype=feats.dtype),
    )
    # Normalized pressure (5/20 = 0.25)
    assert torch.allclose(
        feats[follower_mask, 11],
        torch.full((int(follower_mask.sum()),), 0.25, dtype=feats.dtype),
    )


def test_physics_dwr_diversity_arm_single_model_and_legacy_hash_stability() -> None:
    pool = [
        AcquisitionCandidate(
            case_id="c1",
            partition="TRAIN",
            feature_vector=(0.1, 1.0),
            eta_energy=0.05,
            ood_distance=0.2,
            dwr_sensitivity=0.90,
            geometry_token_vector=(1.0, 0.0, 0.0),
        ),
        AcquisitionCandidate(
            case_id="c2",
            partition="TRAIN",
            feature_vector=(0.11, 1.01),
            eta_energy=0.049,
            ood_distance=0.21,
            dwr_sensitivity=0.88,
            geometry_token_vector=(0.99, 0.01, 0.0),
        ),
        AcquisitionCandidate(
            case_id="c3",
            partition="TRAIN",
            feature_vector=(0.12, 1.02),
            eta_energy=0.048,
            ood_distance=0.22,
            dwr_sensitivity=0.85,
            geometry_token_vector=(0.0, 1.0, 0.0),
        ),
    ]
    weights = AcquisitionWeights(
        eta_weight=0.4,
        ood_weight=0.2,
        ensemble_weight=0.0,
        dwr_weight=0.4,
        diversity_weight=0.35,
    )
    plan = AcquisitionPlan(
        plan_id="dwr-single-model-01",
        arm_kind="physics_dwr_diversity",
        budget_solves=4,
        batch_size=2,
        seed=7,
        weights=weights,
        supervises_training=True,
        training_mode="multi_fidelity_label_efficient",
    )
    ev_a = select_acquisition_batch(pool, already_acquired_ids=(), plan=plan)
    ev_b = select_acquisition_batch(pool, already_acquired_ids=(), plan=plan)
    assert ev_a.selection_hash == ev_b.selection_hash
    # c1 has highest DWR; c3 is far in geometry_token_vector space whereas c2 is redundant
    assert ev_a.selected_case_ids == ("c1", "c3")
    assert ev_a.scores[0].tilde_dwr == pytest.approx(1.0)
    assert "tilde_dwr" in ev_a.scores[0].document()

    # Legacy arm with default dwr_weight=0.0 must not emit dwr_weight or tilde_dwr
    legacy_pool = [
        AcquisitionCandidate(
            case_id="l1",
            partition="TRAIN",
            feature_vector=(0.1, 1.0),
            eta_energy=0.05,
            ood_distance=0.3,
        ),
        AcquisitionCandidate(
            case_id="l2",
            partition="TRAIN",
            feature_vector=(0.5, 2.0),
            eta_energy=0.09,
            ood_distance=0.8,
        ),
    ]
    legacy_plan = AcquisitionPlan(
        plan_id="legacy-01",
        arm_kind="physics_diversity",
        budget_solves=2,
        batch_size=1,
        seed=0,
    )
    legacy_ev = select_acquisition_batch(
        legacy_pool, already_acquired_ids=(), plan=legacy_plan
    )
    assert "dwr_weight" not in legacy_plan.weights.document()
    assert "tilde_dwr" not in legacy_ev.scores[0].document()


def test_fallback_feedback_ledger_enforces_train_only_and_exports(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="TRAIN"):
        FallbackReturnRecord(
            case_id="leak-1",
            partition="HOLDOUT",
            request_hash="req-1",
            result_hash="res-1",
            serving_path="warm_newton",
            feature_vector=(0.1, 100.0),
            eta_energy=0.05,
            dwr_sensitivity=0.02,
            ood_distance=0.4,
            wall_seconds=1.2,
            newton_iterations=4,
        )

    ledger = FallbackFeedbackLedger(campaign_id="cad-loop-01")
    rec1 = FallbackReturnRecord(
        case_id="fb-01",
        partition="TRAIN",
        request_hash="req-01",
        result_hash="res-01",
        serving_path="repaired_warm_newton",
        feature_vector=(0.1, 100.0),
        eta_energy=0.08,
        dwr_sensitivity=0.04,
        ood_distance=0.5,
        geometry_token_vector=(0.2, 0.8),
        newton_iterations=5,
        dof_count=200,
        wall_seconds=1.4,
    )
    rec2 = FallbackReturnRecord(
        case_id="fb-02",
        partition="TRAIN",
        request_hash="req-02",
        result_hash="res-02",
        serving_path="warm_newton",
        feature_vector=(0.2, 120.0),
        eta_energy=0.03,
        dwr_sensitivity=0.09,
        ood_distance=0.7,
        geometry_token_vector=(0.9, 0.1),
        newton_iterations=3,
        dof_count=200,
        wall_seconds=0.9,
    )
    ledger = ledger.append(rec1)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.append(rec1)
    ledger = ledger.append(rec2)

    candidates = ledger.to_acquisition_pool()
    assert len(candidates) == 2
    assert candidates[0].dwr_sensitivity == pytest.approx(0.04)
    assert candidates[0].geometry_token_vector == (0.2, 0.8)

    cost_entry = rec1.to_cost_entry(attempt_index=1)
    assert cost_entry.wall_seconds == pytest.approx(1.4)
    assert cost_entry.dof_count == 200

    out_path = tmp_path / "fallback-ledger.json"
    ledger.save(out_path)
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["schema"] == FALLBACK_LEDGER_SCHEMA
    assert doc["count"] == 2
    assert doc["ledger_hash"] == ledger.ledger_hash
    assert len(ledger.ledger_hash) == 64

    # Verify formamech.active still imports with zero ML/FEM libraries
    code = (
        "import sys, formamech.active; "
        "banned = {'torch', 'scipy', 'skfem', 'gmsh', 'sympy', 'meshio'}; "
        "found = banned.intersection(sys.modules); "
        "assert not found, f'unexpected imports: {found}'"
    )
    env = windows_safe_environment({"PYTHONPATH": str(ROOT / "src")})
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_test_time_lora_reduces_energy_on_both_operators_and_rolls_back() -> None:
    dtype = torch.float64
    dom = TriangleDomain(
        vertices=torch.tensor(
            [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0], [1.0, 0.5]], dtype=dtype
        ),
        cells=torch.tensor(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=torch.int64
        ),
    )
    mat = MaterialSpec(
        name="neo-hookean",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": CanonicalQuantity(10.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.3, Unit.ONE),
        },
    )
    frame = QueryFrame(
        origin=torch.zeros((1, 2), dtype=dtype),
        extent=torch.tensor([[2.0, 1.0]], dtype=dtype),
    )
    lifting = AxisAlignedTensionLifting(
        torch.tensor([[2.0, 1.0]], dtype=dtype),
        torch.tensor([0.15], dtype=dtype),
    )
    params = torch.tensor([[0.1, -0.2]], dtype=dtype)

    # 1) EnergyFormOperator
    torch.manual_seed(11)
    op = EnergyFormOperator(
        EnergyOperatorConfig(
            parameter_names=("a", "b"),
            hidden_dim=12,
            latent_token_count=4,
            parameter_hidden_dim=8,
            num_attention_heads=2,
            decoder_layers=2,
            positional_frequencies=2,
            output_init_std=0.05,
        )
    ).to(dtype)
    base_snapshot = {k: v.detach().clone() for k, v in op.state_dict().items()}

    adapted, report = adapt_operator_at_test_time(
        op,
        domain=dom,
        parameters=params,
        frame=frame,
        lifting=lifting,
        material=mat,
        config=TestTimeLoraConfig(steps=12, rank=4, learning_rate=1e-2, seed=3),
    )
    assert report.steps_taken == 12
    assert report.improved is True
    assert report.rolled_back is False
    assert report.final_energy < report.initial_energy
    assert report.adapter_parameter_count > 0
    assert report.adapter_digest.startswith("blake2b256:")
    # Original operator must be untouched
    for k, v in op.state_dict().items():
        assert torch.equal(v, base_snapshot[k])
    assert isinstance(adapted, EnergyFormOperator)

    # 2) Rollback when min_det_f ceiling is impossible (e.g. 5.0)
    _, rb_report = adapt_operator_at_test_time(
        op,
        domain=dom,
        parameters=params,
        frame=frame,
        lifting=lifting,
        material=mat,
        config=TestTimeLoraConfig(steps=3, rank=2, min_det_f=5.0),
    )
    assert rb_report.improved is False
    assert rb_report.rolled_back is True
    assert rb_report.final_energy == pytest.approx(rb_report.initial_energy)

    # 3) BoundaryEnergyOperator
    torch.manual_seed(19)
    bop = BoundaryEnergyOperator(
        BoundaryEnergyConfig(
            energy=EnergyOperatorConfig(
                parameter_names=("a", "b"),
                hidden_dim=12,
                latent_token_count=4,
                parameter_hidden_dim=8,
                num_attention_heads=2,
                decoder_layers=2,
                positional_frequencies=2,
                output_init_std=0.05,
            ),
            boundary_feature_names=("fixed_x", "fixed_y", "traction"),
        )
    ).to(dtype)
    b_xy = torch.tensor([[[0.0, 0.5], [1.0, 0.0], [2.0, 0.5], [1.0, 1.0]]], dtype=dtype)
    b_n = torch.tensor(
        [[[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]]], dtype=dtype
    )
    b_f = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]],
        dtype=dtype,
    )
    enc = bop.encode_boundary(
        boundary_xy=b_xy,
        boundary_normals=b_n,
        boundary_features=b_f,
        weights=torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=dtype),
        mask=torch.ones((1, 4), dtype=torch.bool),
        frame=frame,
    )
    b_adapted, b_report = adapt_operator_at_test_time(
        bop,
        domain=dom,
        parameters=params,
        frame=frame,
        lifting=lifting,
        material=mat,
        boundary_encoding=enc,
        config=TestTimeLoraConfig(steps=10, rank=4, learning_rate=1e-2, seed=5),
    )
    assert isinstance(b_adapted, BoundaryEnergyOperator)
    assert b_report.steps_taken == 10
    assert b_report.improved is True
    assert b_report.final_energy < b_report.initial_energy


def test_serve_with_tier15_tta_records_provenance_in_receipt() -> None:
    bundle = load_bundle(NOTCHED_BUNDLE)
    recipe = load_recipe(bundle, NOTCHED_BUNDLE)
    inputs, queries = parse_request(json.loads(NOTCHED_REQUEST.read_text("utf-8")))
    # Tight tolerance triggers Tier 1.5 TTA before escalation
    result = serve(
        bundle,
        inputs,
        queries,
        mode=ServingMode.CERTIFIED,
        recipe=recipe,
        tolerance=1.0e-12,
        escalate=False,
        tta_config=TestTimeLoraConfig(steps=3, rank=2, learning_rate=2e-3, seed=0),
        emit_receipt=True,
    )
    assert result.test_time_adaptation is not None
    assert result.test_time_adaptation["steps_taken"] == 3
    doc = result.document()
    assert doc["trust_receipt"] is not None
    ok, errs = verify_receipt_document(doc["trust_receipt"])
    assert ok is True and errs == ()
    assert doc["trust_receipt"]["evidence"]["test_time_adaptation"]["steps_taken"] == 3
