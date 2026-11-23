"""Unit tests for Tier 1.5 test-time physics LoRA and Tier 2 spectrally safe repair."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from campaigns.vvuq_suite import rectangular_strip_mesh
from piginot.contracts import (
    BoundaryConditionSpec,
    ExecutionProfile,
    MaterialSpec,
    Quantity,
    Unit,
)
from piginot.fem import FemRequest, FemSolverConfig, ScikitFemBackend
from piginot.fem.linearization import FemLinearization
from piginot.lifting import AffineDirichletLifting
from piginot.objectives import TriangleDomain
from piginot.operators import (
    GEOMETRY_FEATURE_NAMES,
    BoundaryEnergyConfig,
    BoundaryEnergyOperator,
    BoundaryRoleMap,
    EnergyOperatorConfig,
    GeometryTokenizer,
    QueryFrame,
)
from piginot.operators.energy_operator import EnergyFormOperator
from piginot.serving import (
    SpectralRepairConfig,
    TestTimeLoraConfig,
    adapt_operator_at_test_time,
    spectrally_safe_repair,
)


def _neo_hookean(e_mpa: float = 10.0, nu: float = 0.32) -> MaterialSpec:
    return MaterialSpec(
        name="nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(e_mpa, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(nu, Unit.ONE).to_canonical(),
        },
    )


def test_test_time_lora_energy_and_boundary_operators_and_rollback() -> None:
    mat = _neo_hookean()
    mesh = rectangular_strip_mesh(length_mm=10.0, height_mm=4.0, nx=4, ny=2)
    dom = TriangleDomain(
        torch.tensor(mesh.nodes, dtype=torch.float64),
        torch.tensor(mesh.cells, dtype=torch.int64),
    )
    frame = QueryFrame(
        origin=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        extent=torch.tensor([[10.0, 4.0]], dtype=torch.float64),
    )

    def _lift(q: torch.Tensor) -> AffineDirichletLifting:
        s = (q[..., 0] / 10.0).clamp(0.0, 1.0)
        offset = torch.stack((1.2 * s, torch.zeros_like(s)), dim=-1)
        scale = (s * (1.0 - s) * 4.0).unsqueeze(-1).expand_as(q)
        return AffineDirichletLifting(offset=offset, scale=scale)

    # 1) EnergyFormOperator TTA improves potential energy label-free
    torch.manual_seed(0)
    op = EnergyFormOperator(
        EnergyOperatorConfig(
            parameter_names=("strain",),
            hidden_dim=16,
            latent_token_count=8,
            parameter_hidden_dim=16,
            num_attention_heads=2,
            decoder_layers=2,
            positional_frequencies=2,
        )
    ).to(torch.float64)
    params = torch.tensor([[0.12]], dtype=torch.float64)
    cfg = TestTimeLoraConfig(steps=12, rank=4, learning_rate=1.0e-2, seed=0)
    assert cfg.document()["steps"] == 12
    adapted, rep = adapt_operator_at_test_time(
        op,
        domain=dom,
        parameters=params,
        frame=frame,
        lifting=_lift,
        material=mat,
        config=cfg,
    )
    assert rep.improved is True and rep.rolled_back is False
    assert rep.energy_drop > 0.0
    assert rep.document()["improved"] is True

    # 2) Rollback when min_det_f is impossibly high
    _, rep_rb = adapt_operator_at_test_time(
        op,
        domain=dom,
        parameters=params,
        frame=frame,
        lifting=_lift,
        material=mat,
        config=TestTimeLoraConfig(steps=4, rank=2, min_det_f=100.0),
    )
    assert rep_rb.rolled_back is True and rep_rb.improved is False
    assert rep_rb.final_energy == pytest.approx(rep_rb.initial_energy)

    # 3) BoundaryEnergyOperator TTA + missing boundary_encoding guard
    tok = GeometryTokenizer()
    roles = BoundaryRoleMap(
        roles={
            "left": "dirichlet_xy",
            "right": "dirichlet_x",
            "top": "traction_free",
            "bottom": "traction_free",
        }
    )
    batch = tok.tokenize_batch([(mesh, roles)], dtype=torch.float64)
    b_op = BoundaryEnergyOperator(
        BoundaryEnergyConfig(
            energy=EnergyOperatorConfig(
                parameter_names=("strain",),
                hidden_dim=16,
                latent_token_count=8,
                parameter_hidden_dim=16,
                num_attention_heads=2,
                decoder_layers=2,
                positional_frequencies=2,
            ),
            boundary_feature_names=GEOMETRY_FEATURE_NAMES,
        )
    ).to(torch.float64)
    with torch.no_grad():
        enc = b_op.encode_boundary(
            boundary_xy=batch.boundary_xy,
            boundary_normals=batch.boundary_normals,
            boundary_features=batch.boundary_features,
            weights=batch.weights,
            mask=batch.mask,
            frame=batch.frame,
        )
    _, rep_b = adapt_operator_at_test_time(
        b_op,
        domain=dom,
        parameters=params,
        frame=frame,
        lifting=_lift,
        material=mat,
        boundary_encoding=enc,
        config=TestTimeLoraConfig(steps=8, rank=4, learning_rate=1.0e-2),
    )
    assert rep_b.steps_taken == 8
    with pytest.raises(ValueError, match="boundary_encoding"):
        adapt_operator_at_test_time(
            b_op,
            domain=dom,
            parameters=params,
            frame=frame,
            lifting=_lift,
            material=mat,
            boundary_encoding=None,
            config=TestTimeLoraConfig(steps=2),
        )
    with pytest.raises(ValueError, match="no nn.Linear"):
        adapt_operator_at_test_time(
            op,
            domain=dom,
            parameters=params,
            frame=frame,
            lifting=_lift,
            material=mat,
            config=TestTimeLoraConfig(steps=2, target_substrings=("nonexistent",)),
        )

    # Config validation guards
    with pytest.raises(ValueError, match="steps"):
        TestTimeLoraConfig(steps=0)
    with pytest.raises(ValueError, match="rank"):
        TestTimeLoraConfig(rank=0)
    with pytest.raises(ValueError, match="alpha"):
        TestTimeLoraConfig(alpha=0.0)
    with pytest.raises(ValueError, match="learning_rate"):
        TestTimeLoraConfig(learning_rate=-1.0)
    with pytest.raises(ValueError, match="min_det_f"):
        TestTimeLoraConfig(min_det_f=0.0)


def test_spectrally_safe_repair_paths_and_config_guards() -> None:
    mat = _neo_hookean()
    mesh = rectangular_strip_mesh(length_mm=10.0, height_mm=4.0, nx=4, ny=2)
    req = FemRequest(
        request_id="spectral-repair",
        mesh=mesh,
        material=mat,
        boundary_conditions=(
            BoundaryConditionSpec(
                name="left_fix",
                boundary_tag="left",
                kind="dirichlet",
                components=("x", "y"),
                values=(
                    Quantity(0.0, Unit.MM).to_canonical(),
                    Quantity(0.0, Unit.MM).to_canonical(),
                ),
            ),
            BoundaryConditionSpec(
                name="right_pull",
                boundary_tag="right",
                kind="dirichlet",
                components=("x",),
                values=(Quantity(0.8, Unit.MM).to_canonical(),),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=1, solver_tolerance=1.0e-10),
        execution_profile=ExecutionProfile(
            python_version="3.13",
            library_versions={"numpy": np.__version__},
            blas_threads=1,
            device="cpu",
            dtype="float64",
            rng_algorithm="none",
            seed=0,
        ),
    )
    sk = ScikitFemBackend()
    att = sk.solve_many((req,))[0]
    assert att.success and att.result is not None
    lin = FemLinearization(req)
    u_good = lin.vector(np.asarray(att.result.displacement, dtype=np.float64))

    # 1) Already admissible & SPD -> zero steps
    _, rep_ok = spectrally_safe_repair(lin, u_good)
    assert rep_ok.needed_repair is False and rep_ok.succeeded is True
    assert rep_ok.steps_taken == 0
    assert rep_ok.document()["succeeded"] is True

    # 2) Folded/inverted start -> admissibility relaxation restores SPD
    u_folded = u_good.copy()
    u_folded[lin.constraints.free] = 15.0
    repaired, rep_fix = spectrally_safe_repair(lin, u_folded)
    assert rep_fix.needed_repair is True and rep_fix.succeeded is True
    assert rep_fix.final_positive_definite is True
    assert rep_fix.final_min_det_f is not None and rep_fix.final_min_det_f > 0.05

    # 3) Indefinite tangent (compression buckling / large distortion) triggers LM stage 2
    req_comp = FemRequest(
        request_id="spectral-repair-comp",
        mesh=mesh,
        material=mat,
        boundary_conditions=(
            BoundaryConditionSpec(
                name="left_fix",
                boundary_tag="left",
                kind="dirichlet",
                components=("x", "y"),
                values=(
                    Quantity(0.0, Unit.MM).to_canonical(),
                    Quantity(0.0, Unit.MM).to_canonical(),
                ),
            ),
            BoundaryConditionSpec(
                name="right_push",
                boundary_tag="right",
                kind="dirichlet",
                components=("x",),
                values=(Quantity(-2.0, Unit.MM).to_canonical(),),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=1),
        execution_profile=req.execution_profile,
    )
    lin_c = FemLinearization(req_comp)
    # Unrecoverable min_det_f floor exercises failure path cleanly
    _, rep_fail = spectrally_safe_repair(
        lin_c,
        np.zeros(lin_c.dof_count, dtype=np.float64),
        config=SpectralRepairConfig(min_det_f=5.0, admissibility_relax_steps=2),
    )
    assert rep_fail.succeeded is False

    # Config validation guards
    with pytest.raises(ValueError, match="max_steps"):
        SpectralRepairConfig(max_steps=0)
    with pytest.raises(ValueError, match="min_det_f"):
        SpectralRepairConfig(min_det_f=0.0)
    with pytest.raises(ValueError, match="initial_damping"):
        SpectralRepairConfig(initial_damping=0.0)
    with pytest.raises(ValueError, match="max_damping"):
        SpectralRepairConfig(initial_damping=10.0, max_damping=1.0)


def test_spectrally_safe_repair_stage2_lm_descent_and_exhaustion() -> None:
    """Exercise Stage 2 shifted Levenberg-Marquardt repair on an indefinite tangent."""
    from dataclasses import dataclass

    from scipy.sparse import csc_matrix  # type: ignore[import-untyped]

    @dataclass
    class _Constraints:
        free: np.ndarray

    @dataclass
    class _Assembly:
        deformation_gradient: np.ndarray

    @dataclass
    class _State:
        residual: np.ndarray
        potential_energy: float
        tangent: csc_matrix
        assembly: _Assembly

    class _IndefiniteLin:
        """Minimal 2-DOF toy energy E(x) = -x0^2 + x1^2 near x0=0, convex for |x0|>=0.5."""

        def __init__(self, *, always_indefinite: bool = False) -> None:
            self.constraints = _Constraints(free=np.array([0, 1], dtype=np.int64))
            self.always_indefinite = always_indefinite

        def with_exact_dirichlet(self, vec: np.ndarray) -> np.ndarray:
            return np.asarray(vec, dtype=np.float64).copy()

        def state(self, vec: np.ndarray) -> _State:
            x0, x1 = float(vec[0]), float(vec[1])
            # Smooth double-well in x0: E = 0.25*(x0^2 - 1)^2 + 0.5*x1^2
            # dE/dx0 = x0*(x0^2 - 1), d2E/dx0^2 = 3*x0^2 - 1 (negative at |x0| < 1/sqrt(3)!)
            energy = 0.25 * (x0 * x0 - 1.0) ** 2 + 0.5 * x1 * x1
            res = np.array([x0 * (x0 * x0 - 1.0), x1], dtype=np.float64)
            k00 = -2.0 if self.always_indefinite else (3.0 * x0 * x0 - 1.0)
            k = csc_matrix(np.diag([k00, 1.0]))
            f3d = np.broadcast_to(np.eye(3, dtype=np.float64), (1, 1, 3, 3)).copy()
            return _State(
                residual=res,
                potential_energy=energy,
                tangent=k,
                assembly=_Assembly(deformation_gradient=f3d),
            )

        def free_tangent(self, st: _State) -> csc_matrix:
            return st.tangent

    # Start in the non-convex dome (x0 = 0.2 -> k00 = -0.88 < 0); LM shifts & restores SPD!
    repaired, rep_lm = spectrally_safe_repair(
        _IndefiniteLin(),  # type: ignore[arg-type]
        np.array([0.2, 0.5], dtype=np.float64),
        config=SpectralRepairConfig(max_steps=15, initial_damping=0.5),
    )
    assert rep_lm.needed_repair is True
    assert rep_lm.initial_positive_definite is False
    assert rep_lm.succeeded is True
    assert rep_lm.final_positive_definite is True
    assert abs(float(repaired[0])) > 0.577

    # Budget exhaustion when tangent stays indefinite
    _, rep_ex = spectrally_safe_repair(
        _IndefiniteLin(always_indefinite=True),  # type: ignore[arg-type]
        np.array([0.2, 0.5], dtype=np.float64),
        config=SpectralRepairConfig(max_steps=2, initial_damping=0.5),
    )
    assert rep_ex.succeeded is False
    assert "exhausted" in rep_ex.reason


def test_serve_triggers_tier15_tta_when_certified_tolerance_tight() -> None:
    """Exercise hybrid.py Tier 1.5 TTA integration when p_l2_change exceeds tolerance."""
    import json
    from pathlib import Path

    from piginot.bundle import load_bundle
    from piginot.cli import parse_request
    from piginot.serving import ServingMode, load_recipe, serve

    root = Path(__file__).resolve().parents[2]
    bundle_dir = root / "bundles" / "notched-v1-cp-freeze"
    bundle = load_bundle(bundle_dir)
    recipe = load_recipe(bundle, bundle_dir)
    raw_req = json.loads(
        (root / "bundles" / "requests" / "notched-in-domain.json").read_text(
            encoding="utf-8"
        )
    )
    inputs, queries = parse_request(raw_req)
    out = serve(
        bundle,
        inputs,
        queries,
        mode=ServingMode.CERTIFIED,
        recipe=recipe,
        tolerance=1.0e-9,
        tta_config=TestTimeLoraConfig(steps=3, rank=2, learning_rate=5.0e-3, seed=0),
        emit_receipt=True,
    )
    assert out.document()["test_time_adaptation"] is not None
    assert out.document()["trust_receipt"] is not None


def test_serving_fsm_invariants_statuses_and_repaired_ladder() -> None:
    from piginot.certify import VerificationTier
    from piginot.serving import (
        ServingMode,
        ServingPath,
        SpectralRepairReport,
        SpectralRepairStatus,
        TestTimeLoraReport,
        TestTimeLoraStatus,
        VerifiedPrediction,
        solve_ladder,
    )
    from piginot.serving.hybrid import _build_trust_receipt

    # 1) VerifiedPrediction invariants
    with pytest.raises(ValueError, match="FAST mode"):
        VerifiedPrediction(ServingMode.FAST, ServingPath.COLD_FEM, prediction={})
    with pytest.raises(ValueError, match="converged exact FemResult"):
        VerifiedPrediction(ServingMode.EXACT, ServingPath.WARM_NEWTON, prediction={})
    with pytest.raises(ValueError, match="SOLVE_FAILED"):
        VerifiedPrediction(
            ServingMode.EXACT,
            ServingPath.SOLVE_FAILED,
            prediction={},
            exact=object(),  # type: ignore[arg-type]
        )

    # 2) SpectralRepairReport invariants & all 5 status classifications
    base_kwargs: dict[str, object] = {
        "steps_taken": 0,
        "initial_admissible": True,
        "initial_positive_definite": True,
        "final_positive_definite": True,
        "initial_min_det_f": 1.0,
        "final_min_det_f": 1.0,
        "initial_residual_norm": 0.0,
        "final_residual_norm": 0.0,
        "wall_seconds": 0.0,
        "reason": "ok",
    }
    assert (
        SpectralRepairReport(
            needed_repair=False,
            succeeded=True,
            **base_kwargs,  # type: ignore[arg-type]
        ).status
        is SpectralRepairStatus.ALREADY_SAFE
    )
    assert (
        SpectralRepairReport(
            needed_repair=True,
            succeeded=True,
            **{**base_kwargs, "initial_admissible": False, "steps_taken": 0},  # type: ignore[arg-type]
        ).status
        is SpectralRepairStatus.REPAIRED_HOMOTOPY
    )
    assert (
        SpectralRepairReport(
            needed_repair=True,
            succeeded=True,
            **{**base_kwargs, "steps_taken": 3},  # type: ignore[arg-type]
        ).status
        is SpectralRepairStatus.REPAIRED_LM
    )
    assert (
        SpectralRepairReport(
            needed_repair=True,
            succeeded=False,
            **{**base_kwargs, "final_positive_definite": None},  # type: ignore[arg-type]
        ).status
        is SpectralRepairStatus.FAILED_ADMISSIBILITY
    )
    assert (
        SpectralRepairReport(
            needed_repair=True,
            succeeded=False,
            **{**base_kwargs, "final_positive_definite": False},  # type: ignore[arg-type]
        ).status
        is SpectralRepairStatus.FAILED_LM_BUDGET
    )
    with pytest.raises(ValueError, match="non-negative"):
        SpectralRepairReport(
            needed_repair=False,
            succeeded=True,
            **{**base_kwargs, "steps_taken": -1},  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="needed_repair=False"):
        SpectralRepairReport(
            needed_repair=False,
            succeeded=False,
            **base_kwargs,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="final_positive_definite=False"):
        SpectralRepairReport(
            needed_repair=True,
            succeeded=True,
            **{**base_kwargs, "final_positive_definite": False},  # type: ignore[arg-type]
        )

    # 3) TestTimeLoraReport status & opposite-bool guard
    tta_base: dict[str, object] = {
        "steps_taken": 2,
        "adapted_layers": ("l1",),
        "adapter_parameter_count": 4,
        "initial_energy": 1.0,
        "final_energy": 0.9,
        "energy_drop": 0.1,
        "final_min_det_f": 0.9,
        "adapter_digest": "blake2b256:" + "0" * 64,
        "wall_seconds": 0.01,
    }
    assert (
        TestTimeLoraReport(
            improved=True,
            rolled_back=False,
            **tta_base,  # type: ignore[arg-type]
        ).status
        is TestTimeLoraStatus.APPLIED
    )
    assert (
        TestTimeLoraReport(
            improved=False,
            rolled_back=True,
            **tta_base,  # type: ignore[arg-type]
        ).status
        is TestTimeLoraStatus.ROLLED_BACK
    )
    with pytest.raises(ValueError, match="strictly opposite"):
        TestTimeLoraReport(
            improved=True,
            rolled_back=True,
            **tta_base,  # type: ignore[arg-type]
        )

    # 4) Trust Receipt tier precedence for REFUSED, SOLVE_FAILED, SURROGATE_ADMISSIBLE
    class _StubBundle:
        claim_level = "NOT_VERIFIED"
        manifest = type("M", (), {"manifest_sha256": "a" * 64})()
        identities = {"weights_sha256": "b" * 64}
        reference_mesh = type("Mesh", (), {"mesh_id": "m1"})()
        model = type("Model", (), {"problem": None})()

    rc_ref, _ = _build_trust_receipt(
        bundle=_StubBundle(),
        mode=ServingMode.FAST,
        path=ServingPath.REFUSED,
        request=None,
        solver=None,
        tier0=None,
        conformal_doc=None,
        certificate_doc=None,
        trust_report_doc=None,
        attempts=(),
        exact_converged=False,
        spectral_repair_doc=None,
        tta_doc=None,
        reason="out of envelope",
    )
    assert rc_ref["verification_tier"] == VerificationTier.REFUSED.value

    rc_fail, _ = _build_trust_receipt(
        bundle=_StubBundle(),
        mode=ServingMode.EXACT,
        path=ServingPath.SOLVE_FAILED,
        request=None,
        solver=None,
        tier0={"passed": True},
        conformal_doc=None,
        certificate_doc={"verdict": "OK"},
        trust_report_doc=None,
        attempts=(),
        exact_converged=False,
        spectral_repair_doc=None,
        tta_doc=None,
        reason="ladder failed",
    )
    assert rc_fail["verification_tier"] == VerificationTier.SURROGATE_UNCHECKED.value

    rc_t0, _ = _build_trust_receipt(
        bundle=_StubBundle(),
        mode=ServingMode.FAST,
        path=ServingPath.OPERATOR,
        request=None,
        solver=None,
        tier0={"passed": True},
        conformal_doc=None,
        certificate_doc=None,
        trust_report_doc=None,
        attempts=(),
        exact_converged=False,
        spectral_repair_doc=None,
        tta_doc=None,
        reason="",
    )
    assert rc_t0["verification_tier"] == VerificationTier.SURROGATE_ADMISSIBLE.value

    # 5) solve_ladder with inline spectral_repair records REPAIRED_WARM_NEWTON
    mat = _neo_hookean()
    mesh = rectangular_strip_mesh(length_mm=10.0, height_mm=4.0, nx=4, ny=2)
    req = FemRequest(
        request_id="ladder-repair",
        mesh=mesh,
        material=mat,
        boundary_conditions=(
            BoundaryConditionSpec(
                name="left_fix",
                boundary_tag="left",
                kind="dirichlet",
                components=("x", "y"),
                values=(
                    Quantity(0.0, Unit.MM).to_canonical(),
                    Quantity(0.0, Unit.MM).to_canonical(),
                ),
            ),
            BoundaryConditionSpec(
                name="right_pull",
                boundary_tag="right",
                kind="dirichlet",
                components=("x",),
                values=(Quantity(0.8, Unit.MM).to_canonical(),),
            ),
        ),
        loads=(),
        qois=(),
        config=FemSolverConfig(element_order=1, solver_tolerance=1.0e-10),
        execution_profile=ExecutionProfile(
            python_version="3.13",
            library_versions={"numpy": np.__version__},
            blas_threads=1,
            device="cpu",
            dtype="float64",
            rng_algorithm="none",
            seed=0,
        ),
    )
    sk = ScikitFemBackend()
    lin = FemLinearization(req)
    u_folded = np.zeros(lin.dof_count, dtype=np.float64)
    u_folded[lin.constraints.free] = 15.0
    u_folded = lin.with_exact_dirichlet(u_folded)
    att_r, attempts_r = solve_ladder(
        sk, req, u_folded, linearization=lin, spectral_repair=True
    )
    assert att_r is not None
    assert attempts_r[0].path is ServingPath.REPAIRED_WARM_NEWTON

    # 6) Trigger improved TTA inside serve() so post-TTA Tier 0 evaluation runs
    import json
    from pathlib import Path

    from piginot.bundle import load_bundle
    from piginot.cli import parse_request
    from piginot.serving import load_recipe, serve

    root = Path(__file__).resolve().parents[2]
    bundle_dir = root / "bundles" / "notched-v1-cp-freeze"
    bundle = load_bundle(bundle_dir)
    recipe = load_recipe(bundle, bundle_dir)
    raw_req = json.loads(
        (root / "bundles" / "requests" / "notched-in-domain.json").read_text(
            encoding="utf-8"
        )
    )
    inputs, queries = parse_request(raw_req)
    out_tta = serve(
        bundle,
        inputs,
        queries,
        mode=ServingMode.CERTIFIED,
        recipe=recipe,
        tolerance=1.0e-9,
        tta_config=TestTimeLoraConfig(steps=10, rank=4, learning_rate=2.0e-3, seed=0),
        tier0_check=True,
        emit_receipt=True,
    )
    assert out_tta.test_time_adaptation is not None
    assert out_tta.tier0_admissibility is not None
