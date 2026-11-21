"""Phase 1 tests: RFC 8785 Trust Receipt v1.0 and unsymmetric coercive inertia."""

from __future__ import annotations

import copy
import subprocess
import sys

import numpy as np
import pytest
from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

from piginot.certify import (
    TRUST_RECEIPT_SCHEMA,
    ArtifactDigests,
    TrustReceipt,
    VerificationTier,
    coercive_inertia,
    epistemic_scope,
    verify_receipt_document,
)


def _digests() -> ArtifactDigests:
    return ArtifactDigests(
        bundle_manifest_sha256="a" * 64,
        weights_sha256="b" * 64,
        cad_source_sha256="c" * 64,
        mesh_id="mesh-001",
        problem_hash="d" * 64,
        request_hash="e" * 64,
        solver_backend_id="scikit-fem-2d-v1",
        solver_tolerances={"solver_tolerance": 1e-10, "load_steps": 16},
    )


def test_trust_receipt_rfc8785_determinism_and_tamper_detection() -> None:
    receipt = TrustReceipt(
        mode="exact",
        serving_path="warm_newton",
        claim_level="NOT_VERIFIED",
        verification_tier=VerificationTier.DISCRETE_FEM_VERIFIED,
        artifact_digests=_digests(),
        tier0_admissibility={"passed": True, "min_det_f": 0.94},
        tier1_conformal={"refused": False, "nominal_coverage": 0.95},
        tier1_dwr={"verdict": "OK", "eta_energy": 0.01},
        tier2_fem={"converged": True, "newton_iterations": 2},
        reason="converged via warm_newton",
    )
    doc = receipt.document()
    assert doc["schema"] == TRUST_RECEIPT_SCHEMA
    assert doc["claim_level"] == "NOT_VERIFIED"
    assert doc["verification_tier"] == "DISCRETE_FEM_VERIFIED"
    assert doc["epistemic_scope"]["discrete_fem_agreement"] is True
    assert doc["epistemic_scope"]["marginal_statistical_coverage"] is True
    assert doc["epistemic_scope"]["experimental_validation"] is False

    ok, errors = verify_receipt_document(doc)
    assert ok is True
    assert errors == ()

    tampered = copy.deepcopy(doc)
    tampered["reason"] = "edited"
    ok_bad, errors_bad = verify_receipt_document(tampered)
    assert ok_bad is False
    assert any("receipt_sha256 mismatch" in msg for msg in errors_bad)


def test_trust_receipt_enforces_claim_ceiling_and_tier_evidence() -> None:
    for tier in VerificationTier:
        assert epistemic_scope(tier)["experimental_validation"] is False

    with pytest.raises(ValueError, match="claim_level"):
        TrustReceipt(
            mode="fast",
            serving_path="operator",
            claim_level="CERTIFIED",
            verification_tier=VerificationTier.SURROGATE_UNCHECKED,
            artifact_digests=_digests(),
        )

    with pytest.raises(ValueError, match="SURROGATE_ADMISSIBLE"):
        TrustReceipt(
            mode="fast",
            serving_path="operator",
            claim_level="NOT_VERIFIED",
            verification_tier=VerificationTier.SURROGATE_ADMISSIBLE,
            artifact_digests=_digests(),
            tier0_admissibility={"passed": False},
        )

    with pytest.raises(ValueError, match="CONFORMAL_BOUNDED"):
        TrustReceipt(
            mode="fast",
            serving_path="operator",
            claim_level="NOT_VERIFIED",
            verification_tier=VerificationTier.CONFORMAL_BOUNDED,
            artifact_digests=_digests(),
            tier0_admissibility={"passed": True},
            tier1_conformal={"refused": True},
        )

    with pytest.raises(ValueError, match="DWR_BOUNDED"):
        TrustReceipt(
            mode="certified",
            serving_path="operator+certificate",
            claim_level="NOT_VERIFIED",
            verification_tier=VerificationTier.DWR_BOUNDED,
            artifact_digests=_digests(),
            tier1_dwr={"verdict": "INDEFINITE"},
        )

    with pytest.raises(ValueError, match="DISCRETE_FEM_VERIFIED"):
        TrustReceipt(
            mode="exact",
            serving_path="solve_failed",
            claim_level="NOT_VERIFIED",
            verification_tier=VerificationTier.DISCRETE_FEM_VERIFIED,
            artifact_digests=_digests(),
            tier2_fem={"converged": False},
        )


def test_receipt_import_preserves_runtime_isolation() -> None:
    program = (
        "import sys; import piginot.certify.receipt; "
        "assert 'torch' not in sys.modules; "
        "assert 'numpy' not in sys.modules; "
        "assert 'skfem' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_coercive_inertia_handles_symmetric_and_unsymmetric_tangents() -> None:
    spd = csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    res_spd = coercive_inertia(spd)
    assert res_spd.tangent_symmetric is True
    assert res_spd.positive_definite is True
    assert res_spd.relative_asymmetry == pytest.approx(0.0)

    indef = csc_matrix(np.array([[1.0, 3.0], [3.0, 1.0]]))
    res_indef = coercive_inertia(indef)
    assert res_indef.tangent_symmetric is True
    assert res_indef.positive_definite is False

    # Unsymmetric follower-load-like tangent whose symmetric part is SPD:
    # K = [[4, 2], [0, 3]] -> K_sym = [[4, 1], [1, 3]] (eigenvalues > 0).
    unsym_coercive = csc_matrix(np.array([[4.0, 2.0], [0.0, 3.0]]))
    res_uc = coercive_inertia(unsym_coercive)
    assert res_uc.tangent_symmetric is False
    assert res_uc.relative_asymmetry > 0.1
    assert res_uc.positive_definite is True

    # Unsymmetric tangent whose symmetric part is indefinite:
    unsym_bad = csc_matrix(np.array([[1.0, 5.0], [1.0, 1.0]]))
    res_ub = coercive_inertia(unsym_bad)
    assert res_ub.tangent_symmetric is False
    assert res_ub.positive_definite is False
