"""Tests for P1 compiler hierarchical wiring, P4 CLI --trust-report, and F1 operator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from formamech.cli import main as formamech_main
from piginot.compiler.config_types import EnergyObjectiveConfig
from piginot.contracts import Unit
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from piginot.lifting import AffineDirichletLifting
from piginot.objectives import TriangleDomain
from piginot.operators import LowRankAdapterConfig, QueryFrame
from piginot.stateful import (
    GeneralizedMaxwellConfig,
    GeneralizedMaxwellViscoelastic,
    HistoryAwareViscoelasticOperator,
    MaxwellBranch,
    ViscoelasticOperatorConfig,
    extract_local_viscous_features,
    project_cell_state_to_nodes,
    train_viscoelastic_operator_energy,
)

ROOT = Path(__file__).resolve().parents[1]


def test_hierarchical_energy_objective_config_preserves_default_and_activates() -> None:
    default_cfg = EnergyObjectiveConfig(
        points_per_cell=3,
        barrier_weight=1000.0,
        barrier_threshold=0.05,
        admissibility_floor=1e-6,
    )
    assert default_cfg.is_hierarchical is False

    hier_cfg = EnergyObjectiveConfig(
        points_per_cell=3,
        barrier_weight=1000.0,
        barrier_threshold=0.05,
        admissibility_floor=1e-6,
        cross_mesh_weight=0.25,
        local_physics_weight=0.1,
        quadrature_degree=2,
        use_deterministic_quadrature=True,
    )
    assert hier_cfg.is_hierarchical is True


def test_cli_predict_trust_report_attaches_versioned_report(tmp_path: Path) -> None:
    out_file = tmp_path / "pred_with_trust.json"
    code = formamech_main(
        [
            "predict",
            str(ROOT / "bundles" / "notched-v1-cp-freeze"),
            "--request",
            str(ROOT / "bundles" / "requests" / "notched-in-domain.json"),
            "--mode",
            "fast",
            "--no-confidence",
            "--trust-report",
            "--output",
            str(out_file),
        ]
    )
    assert code == 0
    doc = json.loads(out_file.read_text(encoding="utf-8"))
    assert doc["verdict"] == "IN_DOMAIN"
    assert "trust_report" in doc
    assert doc["trust_report"]["schema"] == "piginot.cad-trust-report-v1"
    assert doc["trust_report"]["overall_status"] == "DIAGNOSTIC_ONLY"
    assert len(doc["trust_report_sha256"]) == 64

    # Out-of-envelope request produces OUT_OF_ENVELOPE trust status.
    out_refused = tmp_path / "pred_refused.json"
    code_ref = formamech_main(
        [
            "predict",
            str(ROOT / "bundles" / "notched-v1-cp-freeze"),
            "--request",
            str(ROOT / "bundles" / "requests" / "notched-out-of-envelope.json"),
            "--mode",
            "fast",
            "--no-confidence",
            "--trust-report",
            "--output",
            str(out_refused),
        ]
    )
    assert code_ref == 2
    ref_doc = json.loads(out_refused.read_text(encoding="utf-8"))
    assert ref_doc["trust_report"]["overall_status"] == "OUT_OF_ENVELOPE"


def test_history_aware_viscoelastic_operator_trains_label_free_and_preserves_bcs() -> (
    None
):
    mesh = canonicalize_mesh(
        MeshPayload(
            ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)),
            ((0, 1, 2), (0, 2, 3)),
            CellType.TRIANGLE,
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            {"bottom": (0,), "right": (1,), "top": (2,), "left": (3,)},
            (1, 1),
            Unit.MM,
            {"kind": "visco-unit"},
        )
    )
    domain = TriangleDomain(
        torch.tensor(mesh.nodes, dtype=torch.float64),
        torch.tensor(mesh.cells, dtype=torch.int64),
    )
    frame = QueryFrame(
        origin=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        extent=torch.tensor([[2.0, 1.0]], dtype=torch.float64),
    )
    visco = GeneralizedMaxwellViscoelastic(
        GeneralizedMaxwellConfig(
            equilibrium_modulus_mpa=100.0,
            poisson_ratio=0.3,
            branches=(MaxwellBranch(0.5, 1.0),),
        )
    )
    op = HistoryAwareViscoelasticOperator(
        ViscoelasticOperatorConfig(
            load_dim=1, n_branches=1, hidden_dim=16, num_layers=2
        )
    ).to(torch.float64)

    def lift_factory(load_vec: torch.Tensor):
        eps = float(load_vec[0].item())

        def lift(q: torch.Tensor) -> AffineDirichletLifting:
            x, y = q[..., 0], q[..., 1]
            return AffineDirichletLifting(
                offset=torch.stack((eps * x, torch.zeros_like(y)), dim=-1),
                scale=torch.stack(((x / 2.0) * (1.0 - x / 2.0), y), dim=-1),
            )

        return lift

    trajs = [
        [
            (torch.tensor([0.05], dtype=torch.float64), 0.25),
            (torch.tensor([0.10], dtype=torch.float64), 0.25),
            (torch.tensor([0.10], dtype=torch.float64), 0.50),
        ]
    ]
    summary = train_viscoelastic_operator_energy(
        op,
        material=visco,
        domain=domain,
        frame=frame,
        load_trajectories=trajs,
        lifting_factory=lift_factory,
        epochs=6,
        learning_rate=1e-2,
        quadrature_degree=2,
        seed=0,
    )
    assert summary.epochs == 6
    assert summary.total_steps == 18
    assert summary.claim_level == "NOT_VERIFIED"
    assert summary.min_det_f_seen > 0.0

    # Verify exact essential BCs at x=0, x=2, and y=0 after training.
    bc_pts = torch.tensor([[[0.0, 0.4], [2.0, 0.5], [1.0, 0.0]]], dtype=torch.float64)
    load_v = torch.tensor([0.10], dtype=torch.float64)
    with torch.no_grad():
        u = op.displacement(
            query_xy=bc_pts,
            frame=frame,
            load_vector=load_v,
            dt=0.25,
            history_features=torch.zeros(4, dtype=torch.float64),
            lifting=lift_factory(load_v),
        )
    assert float(u[0, 0, 0]) == pytest.approx(0.0, abs=1e-14)
    assert float(u[0, 1, 0]) == pytest.approx(0.20, abs=1e-14)
    assert float(u[0, 2, 1]) == pytest.approx(0.0, abs=1e-14)

    # Verify point-local viscous strain feature projection matches node count and responds to local state.
    cell_ev = torch.zeros((domain.cells.shape[0], 1, 3, 3), dtype=torch.float64)
    cell_ev[0, 0, 0, 0] = 0.05
    cell_ev[1, 0, 0, 0] = 0.15
    assert extract_local_viscous_features(cell_ev).shape == (2, 4)
    node_feats = project_cell_state_to_nodes(domain, cell_ev)
    assert node_feats.shape == (domain.vertices.shape[0], 4)
    with torch.no_grad():
        u_local = op.displacement(
            query_xy=domain.vertices.unsqueeze(0),
            frame=frame,
            load_vector=load_v,
            dt=0.25,
            history_features=torch.zeros(4, dtype=torch.float64),
            lifting=lift_factory(load_v),
            local_state_features=node_feats.unsqueeze(0),
        )
    assert u_local.shape == (1, domain.vertices.shape[0], 2)


def test_low_rank_adapter_config_cosine_and_grad_clip_validation() -> None:
    cfg = LowRankAdapterConfig(use_cosine_schedule=True, grad_clip_norm=0.5)
    doc = cfg.document()
    assert doc["use_cosine_schedule"] is True
    assert doc["grad_clip_norm"] == 0.5
    with pytest.raises(ValueError, match="grad_clip_norm"):
        LowRankAdapterConfig(grad_clip_norm=0.0)
