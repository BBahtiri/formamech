"""Unit tests for certificate-driven active FEM acquisition & multi-fidelity loss (P3)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from formamech.active import (
    MULTIFIDELITY_DISCLOSURE,
    AcquisitionCandidate,
    AcquisitionPlan,
    AcquisitionWeights,
    FEMCostLedger,
    FEMCostLedgerEntry,
    MultiFidelityConfig,
    multifidelity_energy_loss,
    select_acquisition_batch,
)

from piginot.contracts import MaterialSpec, Quantity, Unit
from tests.platform_support import windows_safe_environment

ROOT = Path(__file__).resolve().parents[2]


def _make_pool() -> list[AcquisitionCandidate]:
    return [
        AcquisitionCandidate(
            case_id="train-01",
            partition="TRAIN",
            feature_vector=(0.05, 100.0),
            eta_energy=0.01,
            ood_distance=0.2,
            ensemble_disagreement=0.01,
            parameters={"strain": 0.05},
        ),
        AcquisitionCandidate(
            case_id="train-02",
            partition="TRAIN",
            feature_vector=(0.10, 100.0),
            eta_energy=0.08,
            ood_distance=0.4,
            ensemble_disagreement=0.02,
            parameters={"strain": 0.10},
        ),
        AcquisitionCandidate(
            case_id="train-03",
            partition="TRAIN",
            feature_vector=(0.11, 101.0),
            eta_energy=0.079,
            ood_distance=0.41,
            ensemble_disagreement=0.02,
            parameters={"strain": 0.11},
        ),
        AcquisitionCandidate(
            case_id="train-04",
            partition="TRAIN",
            feature_vector=(0.20, 150.0),
            eta_energy=0.075,
            ood_distance=0.95,
            ensemble_disagreement=0.05,
            parameters={"strain": 0.20},
        ),
        AcquisitionCandidate(
            case_id="train-05",
            partition="TRAIN",
            feature_vector=(0.18, 60.0),
            eta_energy=0.05,
            ood_distance=0.6,
            ensemble_disagreement=0.03,
            parameters={"strain": 0.18},
        ),
    ]


def test_active_acquisition_imports_without_ml_or_fem() -> None:
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


def test_partition_guardrails_reject_dev_and_holdout() -> None:
    for bad_partition in ("DEVELOPMENT", "HOLDOUT", "RESERVE", "PILOT"):
        with pytest.raises(ValueError, match="non-TRAIN"):
            AcquisitionCandidate(
                case_id="bad-1",
                partition=bad_partition,
                feature_vector=(0.1,),
                eta_energy=0.05,
            )
        with pytest.raises(ValueError, match="TRAIN"):
            FEMCostLedgerEntry(
                case_id="bad-1",
                partition=bad_partition,
                attempt_index=1,
                succeeded=True,
                wall_seconds=0.1,
                dof_count=100,
            )


def test_plan_forbids_label_free_claim_when_supervising_training() -> None:
    with pytest.raises(ValueError, match="multi_fidelity_label_efficient"):
        AcquisitionPlan(
            plan_id="bad-disclosure",
            arm_kind="physics_diversity",
            budget_solves=4,
            batch_size=2,
            seed=0,
            supervises_training=True,
            training_mode="label_free",
        )


def test_deterministic_order_independent_replay_and_diversity() -> None:
    pool = _make_pool()
    plan_div = AcquisitionPlan(
        plan_id="div-plan",
        arm_kind="physics_diversity",
        budget_solves=4,
        batch_size=2,
        seed=7,
        weights=AcquisitionWeights(
            eta_weight=0.4, ood_weight=0.2, ensemble_weight=0.1, diversity_weight=0.3
        ),
    )
    ev1 = select_acquisition_batch(pool, already_acquired_ids=(), plan=plan_div)
    ev2 = select_acquisition_batch(
        list(reversed(pool)), already_acquired_ids=(), plan=plan_div
    )
    assert ev1.selected_case_ids == ev2.selected_case_ids
    assert ev1.selection_hash == ev2.selection_hash
    assert len(ev1.selected_case_ids) == 2

    # physics_indicator picks the top-2 eta_energy cases (train-02, train-03),
    # whereas physics_diversity separates far-apart points (includes train-04).
    plan_eta = AcquisitionPlan(
        plan_id="eta-plan",
        arm_kind="physics_indicator",
        budget_solves=4,
        batch_size=2,
        seed=7,
    )
    ev_eta = select_acquisition_batch(pool, already_acquired_ids=(), plan=plan_eta)
    assert ev_eta.selected_case_ids == ("train-02", "train-03")
    assert "train-04" in ev1.selected_case_ids

    # random_doe is reproducible per seed and changes with round_index.
    plan_rand = AcquisitionPlan(
        plan_id="rand-plan",
        arm_kind="random_doe",
        budget_solves=4,
        batch_size=2,
        seed=42,
    )
    ev_r1 = select_acquisition_batch(
        pool, already_acquired_ids=(), plan=plan_rand, round_index=0
    )
    ev_r2 = select_acquisition_batch(
        list(reversed(pool)), already_acquired_ids=(), plan=plan_rand, round_index=0
    )
    assert ev_r1.selection_hash == ev_r2.selection_hash


def test_budget_exhaustion_and_fem_cost_ledger_accounting() -> None:
    pool = _make_pool()
    plan = AcquisitionPlan(
        plan_id="budget-plan",
        arm_kind="physics_indicator",
        budget_solves=2,
        batch_size=2,
        seed=0,
    )
    ev_done = select_acquisition_batch(
        pool, already_acquired_ids=("train-01", "train-02"), plan=plan
    )
    assert ev_done.selected_case_ids == ()

    ledger = FEMCostLedger(plan_hash=plan.plan_hash, budget_solves=2)
    ledger = ledger.append(
        FEMCostLedgerEntry(
            case_id="train-02",
            partition="TRAIN",
            attempt_index=1,
            succeeded=True,
            wall_seconds=0.35,
            dof_count=240,
            result_hash="a" * 64,
        )
    )
    ledger = ledger.append(
        FEMCostLedgerEntry(
            case_id="train-03",
            partition="TRAIN",
            attempt_index=2,
            succeeded=False,
            wall_seconds=0.15,
            dof_count=240,
            failure_reason="newton divergence",
        )
    )
    assert ledger.attempted_count == 2
    assert ledger.succeeded_count == 1
    assert ledger.failed_count == 1
    assert ledger.total_wall_seconds == pytest.approx(0.50)
    assert len(ledger.ledger_hash) == 64

    with pytest.raises(ValueError, match="exhausted"):
        ledger.append(
            FEMCostLedgerEntry(
                case_id="train-04",
                partition="TRAIN",
                attempt_index=3,
                succeeded=True,
                wall_seconds=0.2,
                dof_count=240,
            )
        )


def test_multifidelity_energy_loss_enforces_train_only_and_flows_gradients() -> None:
    mat = MaterialSpec(
        name="nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(100.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.3, Unit.ONE).to_canonical(),
        },
    )
    cfg = MultiFidelityConfig(fem_displacement_weight=0.25, fem_stress_weight=0.05)
    assert cfg.disclosure == MULTIFIDELITY_DISCLOSURE

    with pytest.raises(ValueError, match="disclosure"):
        MultiFidelityConfig(fem_displacement_weight=0.1, disclosure="label-free")

    u_pred = torch.tensor(
        [[0.05, 0.0], [0.10, -0.01]], dtype=torch.float64, requires_grad=True
    )
    grad_pred = torch.tensor(
        [[[0.10, 0.0], [0.0, -0.03]], [[0.10, 0.01], [0.0, -0.03]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    wts = torch.tensor([0.5, 0.5], dtype=torch.float64)
    u_ref = torch.tensor([[0.06, 0.0], [0.11, -0.015]], dtype=torch.float64)
    p_ref = torch.tensor(
        [[[10.0, 0.0], [0.0, 0.0]], [[11.0, 0.2], [0.2, 0.0]]], dtype=torch.float64
    )

    with pytest.raises(ValueError, match="TRAIN"):
        multifidelity_energy_loss(
            u_pred=u_pred,
            grad_pred=grad_pred,
            weights=wts,
            material=mat,
            reference_area=1.0,
            config=cfg,
            u_fem_target=u_ref,
            partition="DEVELOPMENT",
        )

    terms = multifidelity_energy_loss(
        u_pred=u_pred,
        grad_pred=grad_pred,
        weights=wts,
        material=mat,
        reference_area=1.0,
        config=cfg,
        u_fem_target=u_ref,
        p_fem_target=p_ref,
        partition="TRAIN",
    )
    assert terms.disclosure == MULTIFIDELITY_DISCLOSURE
    assert float(terms.displacement_distillation.item()) > 0.0
    assert float(terms.stress_distillation.item()) > 0.0
    terms.total.backward()  # type: ignore[no-untyped-call]
    assert u_pred.grad is not None and torch.isfinite(u_pred.grad).all()
    assert grad_pred.grad is not None and torch.isfinite(grad_pred.grad).all()
