"""Gate G-V5-4c tests: Population-centric PLM / EU AI Act Annex IV & HITL Signoff FSM."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from piginot.contracts.state_machine import InvalidTransitionError
from piginot.geometry.mesh3d import make_brick_tet4_mesh
from piginot.governance import (
    HITL_SIGNOFF_MACHINE,
    DesignPopulationPackage,
    DesignVariantSummary,
    HandCalculationSanityCheck,
    HitlSignoffState,
    ReviewerRole,
    create_hitl_signoff,
    export_plm_compliance_package,
    verify_plm_compliance_package,
)
from piginot.serving.audit_ledger import AuditEventKind, SignedAuditLedger
from piginot.store.dataset_3d import (
    DatasetSplit,
    ShardedMeshDataset3D,
    ShardedRecord3D,
)
from piginot.training.continual_clpm import (
    ContinualLoopState,
    ContinualQualificationConfig,
    ContinualQualificationLoop,
)


def _sample_population_package() -> DesignPopulationPackage:
    return DesignPopulationPackage(
        package_id="eco-bracket-2026-q4",
        intended_purpose="Preliminary lightweighting & stress screening for Ti-6Al-4V mount brackets",
        operational_design_domain="Linear-elastic/small-plasticity static loads <= 25 kN, 20-120 C",
        surrogate_model_id="surface-3d-multi-topo-v5",
        training_archive_sha256="a" * 64,
        numerical_claim_level="HOLDOUT_VERIFIED",
        experimental_validation_tier="HOLDOUT_EXPERIMENTALLY_VALIDATED",
        mean_physical_error_summary={"displacement_mm": 0.012, "von_mises_mpa": 1.8},
        worst_case_physical_error_summary={
            "displacement_mm": 0.041,
            "von_mises_mpa": 6.4,
        },
        variants=(
            DesignVariantSummary(
                variant_id="v01",
                parameters={"rib_mm": 4.0},
                mass_kg=0.48,
                compliance_mj=12.4,
                peak_von_mises_mpa=145.0,
                feasible=True,
                pareto_optimal=True,
                trust_receipt_sha256="1" * 64,
            ),
            DesignVariantSummary(
                variant_id="v02",
                parameters={"rib_mm": 2.5},
                mass_kg=0.31,
                compliance_mj=24.0,
                peak_von_mises_mpa=420.0,
                feasible=False,
                ood_refused=False,
                trust_receipt_sha256="2" * 64,
            ),
            DesignVariantSummary(
                variant_id="v03",
                parameters={"rib_mm": 9.9},
                mass_kg=0.95,
                compliance_mj=7.0,
                peak_von_mises_mpa=90.0,
                feasible=False,
                ood_refused=True,
                trust_receipt_sha256="3" * 64,
            ),
        ),
        promoted_variant_id="v01",
        promoted_brep_step_sha256="b" * 64,
        gdt_robustness_passed=True,
        break_even_n_star=42.5,
    )


def test_plm_dossier_export_requires_valid_hitl_signoff_and_verifies_hash_chain(
    tmp_path: Path,
) -> None:
    """Gate G-V5-4c: PLM export refuses missing/rejected/tampered HITL; valid signoff verifies 100%."""
    pkg = _sample_population_package()
    ledger = SignedAuditLedger(tmp_path / "audit.jsonl")

    # 1) Missing signoff fails closed with InvalidTransitionError
    with pytest.raises(InvalidTransitionError, match="HITL signoff required"):
        export_plm_compliance_package(
            pkg, tmp_path / "out_missing", hitl_signoff=None, require_hitl_signoff=True
        )

    # 2) Rejected signoff fails closed
    rejected = create_hitl_signoff(
        artifact_id=pkg.package_id,
        artifact_sha256=pkg.package_sha256,
        reviewer_id="eng-42",
        reviewer_role=ReviewerRole.LEAD_STRESS_ENGINEER,
        approved=False,
        decision_note="Fillet radius R1.5 mm below shop-floor minimum",
        audit_ledger=ledger,
    )
    assert rejected.state is HitlSignoffState.REJECTED_WITH_FEEDBACK
    assert rejected.snapshot().machine_id == HITL_SIGNOFF_MACHINE.machine_id
    with pytest.raises(InvalidTransitionError, match="not approved"):
        export_plm_compliance_package(
            pkg, tmp_path / "out_rej", hitl_signoff=rejected, require_hitl_signoff=True
        )

    # 3) Valid approved signoff with hand-calc sanity check succeeds & records to ledger
    hand_calc = HandCalculationSanityCheck(
        hand_calc_note="Roark cantilever bending sigma = M*c/I = 140 MPa vs model 145 MPa",
        hand_calc_value=140.0,
        model_predicted_value=145.0,
        unit="MPa",
    )
    approved = create_hitl_signoff(
        artifact_id=pkg.package_id,
        artifact_sha256=pkg.package_sha256,
        reviewer_id="eng-07",
        reviewer_role=ReviewerRole.VALIDITY_OWNER,
        approved=True,
        decision_note="Hand-calc within 3.6%, GD&T verified, cleared for ECO release.",
        hand_calc=hand_calc,
        audit_ledger=ledger,
    )
    assert approved.state is HitlSignoffState.APPROVED
    assert approved.verify_signature() is True

    # 4) Tampered signoff fails closed
    tampered = dataclasses.replace(approved, reviewer_id="spoofed-reviewer")
    assert tampered.verify_signature() is False
    with pytest.raises(InvalidTransitionError, match="tampered"):
        export_plm_compliance_package(
            pkg,
            tmp_path / "out_tampered",
            hitl_signoff=tampered,
            require_hitl_signoff=True,
        )

    # 5) Legitimate export produces valid JSON-LD, ReqIF XML, EU AI Act Annex IV MD & verified manifest
    exported = export_plm_compliance_package(
        pkg, tmp_path / "out_valid", hitl_signoff=approved, require_hitl_signoff=True
    )
    assert exported.jsonld_path.is_file()
    assert exported.reqif_path.is_file()
    assert exported.markdown_path.is_file()
    # ReqIF parses as valid XML
    reqif_root = ET.fromstring(exported.reqif_path.read_bytes())
    assert reqif_root.tag.endswith("REQ-IF")
    # Hash-chain verification passes 100%
    ver = verify_plm_compliance_package(exported.output_dir)
    assert ver["ok"] is True
    assert ver["file_count"] == 3
    chain_ok, _ = ledger.verify_chain()
    assert chain_ok is True

    # Tampering an exported compliance file fails verification
    exported.markdown_path.write_text("tampered content", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_plm_compliance_package(exported.output_dir)


def test_continual_qualification_loop_enforces_hitl_signoff_gate(
    tmp_path: Path,
) -> None:
    """Gate G-V5-4c: ContinualQualificationLoop(require_hitl_signoff=True) blocks until signed off."""
    active_ckpt = tmp_path / "active_ckpt"
    active_ckpt.mkdir()
    (active_ckpt / "weights.json").write_text('{"v": 1}', encoding="utf-8")

    ledger = SignedAuditLedger(tmp_path / "clpm_ledger.jsonl")
    ledger.append(
        event_kind=AuditEventKind.FEM_FALLBACK,
        bundle_name="bracket-3d",
        mode="certified",
        verdict="ESCALATED",
        inputs={"case": "fb-1"},
    )

    ds = ShardedMeshDataset3D(tmp_path / "clpm_ds")
    m_tr = make_brick_tet4_mesh(10.0, 5.0, 5.0, nx=1, ny=1, nz=1)
    m_ho = make_brick_tet4_mesh(12.0, 5.0, 5.0, nx=1, ny=1, nz=1)
    m_fb = make_brick_tet4_mesh(14.0, 5.0, 5.0, nx=1, ny=1, nz=1)
    ds.append_records(
        [
            ShardedRecord3D(
                "tr-0",
                "m-tr",
                DatasetSplit.TRAIN,
                (1.0,),
                np.asarray(m_tr.nodes),
                np.asarray(m_tr.cells),
                np.zeros((len(m_tr.nodes), 3)),
                {},
            ),
            ShardedRecord3D(
                "ho-0",
                "m-ho",
                DatasetSplit.HOLDOUT,
                (1.2,),
                np.asarray(m_ho.nodes),
                np.asarray(m_ho.cells),
                np.zeros((len(m_ho.nodes), 3)),
                {},
            ),
        ]
    )
    fb_rec = ShardedRecord3D(
        "fb-0",
        "m-fb",
        DatasetSplit.TRAIN,
        (1.4,),
        np.asarray(m_fb.nodes),
        np.asarray(m_fb.cells),
        np.zeros((len(m_fb.nodes), 3)),
        {},
    )

    loop = ContinualQualificationLoop(
        active_checkpoint_dir=active_ckpt,
        audit_ledger=ledger,
        dataset=ds,
        workdir=tmp_path / "clpm_work",
        config=ContinualQualificationConfig(require_hitl_signoff=True),
    )

    def _eval(ckpt: Path, records: object) -> float:
        del records
        return 0.02 if ckpt.name != "active_ckpt" else 0.10

    # Without signoff -> fails closed with InvalidTransitionError!
    with pytest.raises(InvalidTransitionError, match="HITL signoff required"):
        loop.run_cycle("cycle-01", solved_fallbacks=[fb_rec], evaluator=_eval)

    # With valid senior-engineer signoff -> atomically promotes!
    signoff = create_hitl_signoff(
        artifact_id="cycle-01",
        artifact_sha256="c" * 64,
        reviewer_id="chief-eng-1",
        reviewer_role=ReviewerRole.CHIEF_ENGINEER,
        approved=True,
        decision_note="Shadow metrics verified; approved for promotion.",
        hand_calc=HandCalculationSanityCheck(
            hand_calc_note="Tip deflection beam estimate 0.021 mm vs shadow 0.020 mm",
            hand_calc_value=0.021,
            model_predicted_value=0.020,
            unit="mm",
        ),
        audit_ledger=ledger,
    )
    res = loop.run_cycle(
        "cycle-02",
        solved_fallbacks=[fb_rec],
        evaluator=_eval,
        hitl_signoff=signoff,
    )
    assert res.promoted is True
    assert res.state is ContinualLoopState.PROMOTED
