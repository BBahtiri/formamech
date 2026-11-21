"""Verified serving: ``fast``, ``certified`` and ``exact`` (S22, ADR 0103).

- ``fast``: the bundle's prediction, unchanged from v0.1.0.
- ``certified``: the prediction plus the S21 certificate, evaluated on the FEM
  problem the bundle's recipe rebuilds for the request; with ``corrected`` also
  the one-step corrected displacement at the FEM nodes. With ``escalate``, a
  certificate that is not ``OK`` (or whose stress change exceeds
  ``tolerance``) promotes the request to ``exact``.
- ``exact``: the converged FEM solution, started from the prediction. The
  solve ladder is warm Newton at full load, then a short continuation from the
  half-scaled prediction, then the cold load-stepped solve, then an explicit
  failure. Every response names the path that produced it.

``fast`` needs only the runtime extra. The other modes import the FEM stack
lazily and raise ``MissingExtraError`` without it.
"""

from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from piginot.bounds import (
    CadCalibrationRecord,
    CalibrationError,
    CalibrationRecord,
    ConfirmedCalibration,
)
from piginot.bundle import predict
from piginot.bundle.problem import PortableProblem
from piginot.certify.receipt import (
    ArtifactDigests,
    TrustReceipt,
    VerificationTier,
)

if TYPE_CHECKING:
    from piginot.certify.admissibility import (
        AdmissibilityThresholds,
        Tier0AdmissibilityResult,
    )
    from piginot.fem.models import FemRequest, FemResult
    from piginot.serving.recipe import FemRecipe
    from piginot.serving.spectral_repair import SpectralRepairConfig
    from piginot.serving.test_time_lora import TestTimeLoraConfig


class ServingMode(enum.StrEnum):
    FAST = "fast"
    CERTIFIED = "certified"
    EXACT = "exact"


class ServingPath(enum.StrEnum):
    OPERATOR = "operator"
    OPERATOR_CERTIFICATE = "operator+certificate"
    WARM_NEWTON = "warm_newton"
    REPAIRED_WARM_NEWTON = "repaired_warm_newton"
    CONTINUATION = "continuation"
    COLD_FEM = "cold_fem"
    SOLVE_FAILED = "solve_failed"
    REFUSED = "refused"


class MissingExtraError(ImportError):
    """A certified or exact request without the FEM dependencies installed."""

    def __init__(self, mode: str) -> None:
        super().__init__(
            f"mode {mode!r} needs the FEM extra: pip install 'formamech[verify]'"
        )


@dataclass(frozen=True, slots=True)
class SolveAttempt:
    """One rung of the solve ladder."""

    path: ServingPath
    success: bool
    newton_iterations: int
    failure_code: str | None
    message: str

    def document(self) -> dict[str, Any]:
        return {
            "path": self.path.value,
            "success": self.success,
            "newton_iterations": self.newton_iterations,
            "failure_code": self.failure_code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class VerifiedPrediction:
    """A served prediction and how far it was verified."""

    mode: ServingMode
    path: ServingPath
    prediction: Any
    certificate: Mapping[str, Any] | None = None
    corrected_displacement: np.ndarray[Any, Any] | None = None
    exact: FemResult | None = None
    exact_result_hash: str | None = None
    attempts: tuple[SolveAttempt, ...] = ()
    escalated: bool = False
    request_hash: str | None = None
    reason: str = ""
    bounds: Mapping[str, Any] | None = None
    trust_report: Mapping[str, Any] | None = None
    trust_report_sha256: str | None = None
    tier0_admissibility: Mapping[str, Any] | None = None
    spectral_repair: Mapping[str, Any] | None = None
    test_time_adaptation: Mapping[str, Any] | None = None
    trust_receipt: Mapping[str, Any] | None = None
    trust_receipt_sha256: str | None = None

    def document(self) -> dict[str, Any]:
        """JSON form; ``fast`` without bounds is exactly the v0.1.0 document.

        Calibrated bounds (S23) appear under ``bounds`` only when requested, so
        every output without them is unchanged.
        """

        if self.mode is ServingMode.FAST:
            prediction: dict[str, Any] = self.prediction.document()
            if (
                self.bounds is None
                and self.trust_report is None
                and self.tier0_admissibility is None
                and self.trust_receipt is None
            ):
                return prediction
            doc_fast: dict[str, Any] = {
                "mode": self.mode.value,
                "path": self.path.value,
                "prediction": prediction,
            }
            if self.bounds is not None:
                doc_fast["bounds"] = dict(self.bounds)
            if self.tier0_admissibility is not None:
                doc_fast["tier0_admissibility"] = dict(self.tier0_admissibility)
            if self.trust_report is not None:
                doc_fast["trust_report"] = dict(self.trust_report)
                doc_fast["trust_report_sha256"] = self.trust_report_sha256
            if self.trust_receipt is not None:
                doc_fast["trust_receipt"] = dict(self.trust_receipt)
                doc_fast["trust_receipt_sha256"] = self.trust_receipt_sha256
            return doc_fast
        document: dict[str, Any] = {
            "mode": self.mode.value,
            "path": self.path.value,
            "escalated": self.escalated,
            "reason": self.reason,
            "request_hash": self.request_hash,
            "prediction": self.prediction.document(),
            "certificate": None if self.certificate is None else dict(self.certificate),
            "corrected_displacement": None
            if self.corrected_displacement is None
            else self.corrected_displacement.tolist(),
            "exact": None
            if self.exact is None
            else {
                "result_hash": self.exact_result_hash,
                "total_potential_energy": self.exact.total_potential_energy,
                "displacement": [list(v) for v in self.exact.displacement],
                "displacement_dof_coordinates": [
                    list(v) for v in self.exact.displacement_dof_coordinates
                ],
            },
            "attempts": [attempt.document() for attempt in self.attempts],
        }
        if self.bounds is not None:
            document["bounds"] = dict(self.bounds)
        if self.tier0_admissibility is not None:
            document["tier0_admissibility"] = dict(self.tier0_admissibility)
        if self.spectral_repair is not None:
            document["spectral_repair"] = dict(self.spectral_repair)
        if self.test_time_adaptation is not None:
            document["test_time_adaptation"] = dict(self.test_time_adaptation)
        if self.trust_report is not None:
            document["trust_report"] = dict(self.trust_report)
            document["trust_report_sha256"] = self.trust_report_sha256
        if self.trust_receipt is not None:
            document["trust_receipt"] = dict(self.trust_receipt)
            document["trust_receipt_sha256"] = self.trust_receipt_sha256
        return document


def default_request_id(inputs: Mapping[str, Any]) -> str:
    """A request ID derived from the inputs (canonical values and units)."""

    payload = {
        name: [float(q.value), str(getattr(q.unit, "value", q.unit))]
        for name, q in sorted(inputs.items())
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return "served-" + digest.hexdigest()[:16]


def operator_displacement(
    bundle: Any, inputs: Mapping[str, Any], mesh: Any, points: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """The operator's displacement at ``points``, without stresses (S22).

    Bit-identical to the ``u`` that :func:`piginot.bundle.predict` serves at the
    same query points, at a fraction of the cost.
    """

    import torch

    from piginot.bundle.inference import _Request

    request = _Request(bundle, inputs, mesh)
    u = bundle.serving.displacement(
        torch.tensor(points, dtype=torch.float64),
        parameters=request.parameters,
        frame=request.frame,
        lifting=request.lifting,
        chunk=bundle.model.query_chunk,
    )
    result: np.ndarray[Any, Any] = u.numpy().astype(np.float64)
    return result


def recipe_path(bundle_path: Any) -> Any:
    """Where a bundle's recipe sidecar lives: ``<bundles>/fem-recipes/<name>.json``."""

    from pathlib import Path

    directory = Path(bundle_path)
    return directory.parent / "fem-recipes" / f"{directory.name}.json"


def load_recipe(bundle: Any, bundle_path: Any, path: Any = None) -> FemRecipe:
    """Load and bind the recipe of a loaded bundle (needs the FEM extra)."""

    from pathlib import Path

    try:
        from piginot.serving.recipe import FemRecipe
    except ImportError as exc:
        raise MissingExtraError("certified") from exc
    location = Path(path) if path is not None else recipe_path(bundle_path)
    return FemRecipe.load(
        location,
        bundle=Path(bundle_path).name,
        manifest_sha256=bundle.manifest.manifest_sha256,
    )


def _fem(mode: ServingMode) -> tuple[Any, Any, Any]:
    try:
        from piginot.certify import certify_with_correction
        from piginot.fem import ScikitFemBackend
        from piginot.fem.linearization import FemLinearization
    except ImportError as exc:
        raise MissingExtraError(mode.value) from exc
    return certify_with_correction, ScikitFemBackend, FemLinearization


def _resolve_solver(backend: Any, default_type: Any) -> Any:
    if backend is None:
        return default_type()
    if isinstance(backend, str):
        from piginot.fem.registry import resolve_backend

        return resolve_backend(backend)
    return backend


def _build_portable_fem_request(
    problem: PortableProblem,
    inputs: Mapping[str, Any],
    mesh: Any,
    request_id: str,
) -> FemRequest:
    import sys
    from importlib.metadata import PackageNotFoundError, version

    from piginot.contracts import ExecutionProfile
    from piginot.problem.requests import fem_request

    libs: dict[str, str] = {}
    for dist in ("numpy", "scikit-fem", "scipy", "sympy", "torch"):
        try:
            libs[dist] = version(dist)
        except PackageNotFoundError:
            libs[dist] = "unknown"
    profile = ExecutionProfile(
        python_version=(
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        library_versions=libs,
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="none",
        seed=0,
    )
    return fem_request(problem.spec, inputs, mesh, request_id, profile)


def _evaluate_tier0(
    prediction: Any,
    *,
    thresholds: AdmissibilityThresholds | None = None,
    dirichlet_deviation: float | None = None,
) -> Tier0AdmissibilityResult:
    from piginot.certify.admissibility import (
        AdmissibilityThresholds as ThresholdsCls,
    )
    from piginot.certify.admissibility import (
        evaluate_tier0_admissibility,
    )

    fields: Mapping[str, Any] = {}
    vertices = getattr(prediction, "vertices", None)
    if vertices is not None:
        extracted: dict[str, Any] = {}
        for name in ("u", "F_2D", "F33", "J_3D", "P_served", "sigma"):
            val = getattr(vertices, name, None)
            if val is not None:
                extracted[name] = (
                    val.detach().cpu().numpy() if hasattr(val, "detach") else val
                )
        fields = extracted
    else:
        pred_doc = prediction.document()
        if isinstance(pred_doc, Mapping):
            raw = pred_doc.get("fields") or pred_doc.get("vertices")
            if isinstance(raw, Mapping):
                fields = raw
    return evaluate_tier0_admissibility(
        fields,
        thresholds=ThresholdsCls() if thresholds is None else thresholds,
        dirichlet_deviation=dirichlet_deviation,
    )


def _build_trust_receipt(
    *,
    bundle: Any,
    mode: ServingMode,
    path: ServingPath,
    request: FemRequest | None,
    solver: Any | None,
    tier0: Mapping[str, Any] | None,
    conformal_doc: Mapping[str, Any] | None,
    certificate_doc: Mapping[str, Any] | None,
    trust_report_doc: Mapping[str, Any] | None,
    attempts: tuple[SolveAttempt, ...],
    exact_converged: bool,
    spectral_repair_doc: Mapping[str, Any] | None,
    tta_doc: Mapping[str, Any] | None = None,
    reason: str,
) -> tuple[dict[str, Any], str]:
    manifest = getattr(bundle, "manifest", None)
    files = getattr(manifest, "files", {}) if manifest is not None else {}
    weights_entry = (
        files.get("weights/model.safetensors") if isinstance(files, Mapping) else None
    )
    problem = getattr(getattr(bundle, "model", None), "problem", None)
    mesh = getattr(bundle, "reference_mesh", None)
    solver_tols: dict[str, Any] | None = None
    if request is not None:
        solver_tols = {
            "element_order": request.config.element_order,
            "load_steps": request.config.load_steps,
            "max_newton_iterations": request.config.max_newton_iterations,
            "solver_tolerance": request.config.solver_tolerance,
        }
    digests = ArtifactDigests(
        bundle_manifest_sha256=(
            None if manifest is None else getattr(manifest, "manifest_sha256", None)
        ),
        weights_sha256=(
            None if weights_entry is None else getattr(weights_entry, "sha256", None)
        ),
        mesh_id=(
            request.mesh.mesh_id
            if request is not None
            else (None if mesh is None else getattr(mesh, "mesh_id", None))
        ),
        problem_hash=(
            problem.problem_hash if isinstance(problem, PortableProblem) else None
        ),
        request_hash=None if request is None else str(request.recipe_hash),
        solver_backend_id=(
            None
            if solver is None
            else str(
                getattr(
                    solver, "backend_id", getattr(solver, "backend_name", "unknown")
                )
            )
        ),
        solver_tolerances=solver_tols,
    )

    claim_raw = getattr(bundle, "claim_level", "NOT_VERIFIED")
    claim_str = claim_raw.value if isinstance(claim_raw, enum.Enum) else str(claim_raw)

    dwr_evidence: dict[str, Any] | None = None
    if certificate_doc is not None:
        dwr_evidence = {
            "verdict": certificate_doc.get("verdict"),
            "eta_energy": certificate_doc.get("eta_energy"),
            "dirichlet_deviation": certificate_doc.get("dirichlet_deviation"),
            "trust_report_overall_status": (
                None
                if trust_report_doc is None
                else trust_report_doc.get("overall_status")
            ),
        }

    tier2_evidence: dict[str, Any] | None = None
    if attempts or exact_converged:
        tier2_evidence = {
            "converged": exact_converged,
            "serving_path": path.value,
            "attempts": [a.document() for a in attempts],
            "spectral_repair": (
                None if spectral_repair_doc is None else dict(spectral_repair_doc)
            ),
        }

    conformal_accepted = bool(
        isinstance(conformal_doc, Mapping) and conformal_doc.get("refused") is False
    )
    tier0_passed = bool(isinstance(tier0, Mapping) and tier0.get("passed") is True)
    if path is ServingPath.REFUSED:
        tier = VerificationTier.REFUSED
    elif exact_converged:
        tier = VerificationTier.DISCRETE_FEM_VERIFIED
    elif dwr_evidence is not None and dwr_evidence.get("verdict") == "OK":
        tier = VerificationTier.DWR_BOUNDED
    elif tier0_passed and conformal_accepted:
        tier = VerificationTier.CONFORMAL_BOUNDED
    elif tier0_passed:
        tier = VerificationTier.SURROGATE_ADMISSIBLE
    else:
        tier = VerificationTier.SURROGATE_UNCHECKED

    receipt = TrustReceipt(
        mode=mode.value,
        serving_path=path.value,
        claim_level=claim_str,
        verification_tier=tier,
        artifact_digests=digests,
        tier0_admissibility=tier0,
        tier1_conformal=conformal_doc,
        tier1_dwr=dwr_evidence,
        tier2_fem=tier2_evidence,
        test_time_adaptation=tta_doc,
        reason=reason,
    )
    return receipt.document(), receipt.receipt_sha256


def solve_ladder(
    backend: Any,
    request: FemRequest,
    initial: np.ndarray[Any, Any],
    *,
    linearization: Any | None = None,
    spectral_repair: bool = False,
    spectral_repair_config: SpectralRepairConfig | None = None,
) -> tuple[Any | None, tuple[SolveAttempt, ...]]:
    """Warm Newton (optionally spectrally repaired), continuation, cold solve."""

    start_vec = initial
    first_path = ServingPath.WARM_NEWTON
    if spectral_repair and linearization is not None:
        from piginot.serving.spectral_repair import (
            SpectralRepairConfig as RepairCfg,
        )
        from piginot.serving.spectral_repair import (
            spectrally_safe_repair,
        )

        repaired, report = spectrally_safe_repair(
            linearization,
            initial,
            config=RepairCfg()
            if spectral_repair_config is None
            else spectral_repair_config,
        )
        if report.needed_repair and report.succeeded:
            start_vec = repaired
            first_path = ServingPath.REPAIRED_WARM_NEWTON

    ladder: Sequence[
        tuple[ServingPath, np.ndarray[Any, Any] | None, tuple[float, ...]]
    ] = (
        (first_path, start_vec, (1.0,)),
        (ServingPath.CONTINUATION, 0.5 * start_vec, (0.5, 0.75, 1.0)),
        (ServingPath.COLD_FEM, None, ()),
    )
    attempts: list[SolveAttempt] = []
    for path, start, factors in ladder:
        attempt = (
            backend.solve_many((request,))[0]
            if start is None
            else backend.solve_from(request, start, factors)
        )
        iterations = (
            len(attempt.result.convergence_history) if attempt.result is not None else 0
        )
        attempts.append(
            SolveAttempt(
                path,
                bool(attempt.success),
                iterations,
                None if attempt.failure_code is None else str(attempt.failure_code),
                str(attempt.message),
            )
        )
        if attempt.success:
            return attempt, tuple(attempts)
    return None, tuple(attempts)


def _build_live_cad_trust_report(
    *,
    bundle: Any,
    request: FemRequest,
    surrogate: np.ndarray[Any, Any],
    trust_sidecar: Mapping[str, Any] | None = None,
    solve_failed: bool = False,
    backend: Any | None = None,
    statistical_by_qoi: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Evaluate live DWR adjoint QoI error budgets and assemble ``CADTrustReport``."""
    from piginot.certify import (
        DiscretizationErrorRecord,
        GeometryErrorRecord,
        ReactionForceQoi,
        RegionalAverageStressQoi,
        TotalStrainEnergyQoi,
        evaluate_cad_trust,
    )

    qois: list[Any] = []
    for bc in request.boundary_conditions:
        if bc.kind == "dirichlet" and any(abs(v.value) > 0.0 for v in bc.values):
            qois.append(ReactionForceQoi(f"reaction_{bc.name}", bc.name))
    qois.append(RegionalAverageStressQoi("avg_p11", component=(0, 0)))
    qois.append(TotalStrainEnergyQoi("total_strain_energy"))

    raw_geom = (
        trust_sidecar.get("geometry") if isinstance(trust_sidecar, Mapping) else None
    )
    geom_doc: Mapping[str, Any] = raw_geom if isinstance(raw_geom, Mapping) else {}
    raw_disc = (
        trust_sidecar.get("discretization")
        if isinstance(trust_sidecar, Mapping)
        else None
    )
    disc_doc: Mapping[str, Any] = raw_disc if isinstance(raw_disc, Mapping) else {}
    geom_rec = GeometryErrorRecord(
        cad_source_hash=geom_doc.get("cad_source_hash"),
        declared_curve_tolerance_mm=geom_doc.get("declared_curve_tolerance_mm"),
        achieved_max_chord_error_mm=geom_doc.get("achieved_max_chord_error_mm"),
        remesh_status=str(geom_doc.get("remesh_status", "FIXED_BUNDLE_MESH")),
        excluded_weight_fraction=geom_doc.get("excluded_weight_fraction"),
    )
    disc_rec = DiscretizationErrorRecord(
        reference_status=str(disc_doc.get("reference_status", "MEASURED_CHANGES_ONLY")),
        element_order=int(disc_doc.get("element_order", request.config.element_order)),
        dof_count=int(disc_doc.get("dof_count", surrogate.shape[0] * 2)),
        m1_to_m2_u_rel_l2=disc_doc.get("m1_to_m2_u_rel_l2"),
        m1_to_m2_p_rel_l2=disc_doc.get("m1_to_m2_p_rel_l2"),
        m1_to_m2_qoi_change=disc_doc.get("m1_to_m2_qoi_change"),
    )
    indicators_qualified = bool(
        isinstance(trust_sidecar, Mapping)
        and trust_sidecar.get("indicators_qualified") is True
    )
    report = evaluate_cad_trust(
        request,
        surrogate,
        qois,
        problem_hash=str(bundle.manifest.manifest_sha256),
        geometry_error=geom_rec,
        discretization_error=disc_rec,
        bundle_digest=str(bundle.manifest.manifest_sha256),
        claim_level=bundle.claim_level,
        in_parameter_envelope=True,
        in_geometry_envelope=True,
        statistical_by_qoi=statistical_by_qoi,
        indicators_qualified=indicators_qualified,
        solve_failed=solve_failed,
        backend=backend,
    )
    return report.document(), report.report_hash


def serve(
    bundle: Any,
    inputs: Mapping[str, Any],
    queries: Sequence[Sequence[float]] | None = None,
    *,
    mode: ServingMode | str = ServingMode.FAST,
    recipe: FemRecipe | None = None,
    corrected: bool = False,
    escalate: bool = False,
    tolerance: float | None = None,
    request_id: str | None = None,
    backend: Any = None,
    confidence: bool | None = None,
    calibration: CalibrationRecord | ConfirmedCalibration | None = None,
    cad_calibration: CadCalibrationRecord | None = None,
    trust_report: bool = False,
    trust_sidecar: Mapping[str, Any] | None = None,
    tier0_check: bool = False,
    tier0_thresholds: AdmissibilityThresholds | None = None,
    spectral_repair: bool = False,
    spectral_repair_config: SpectralRepairConfig | None = None,
    tta_config: TestTimeLoraConfig | None = None,
    emit_receipt: bool = False,
) -> VerifiedPrediction:
    """Serve one request in the chosen mode (see the module docstring).

    ``confidence`` controls the S14 pointwise bound in the returned prediction:
    on by default in ``fast`` (as in v0.1.0), off by default otherwise, where
    the certificate supersedes it. ``calibration`` (S23) attaches calibrated
    upper bounds to a served request; ``cad_calibration`` binds split-conformal
    and functional-band bounds to arbitrary user-compiled CAD bundles.
    """

    mode = ServingMode(mode)
    portable_problem = getattr(bundle.model, "problem", None)
    is_real_portable = isinstance(portable_problem, PortableProblem)
    if portable_problem is not None:
        if not is_real_portable and mode is not ServingMode.FAST:
            raise ValueError("a portable candidate supports only fast serving")
        if recipe is not None:
            raise ValueError("a portable candidate has no bound FEM recipe")
        if calibration is not None or confidence is True:
            raise CalibrationError("a portable candidate has no matching calibration")
    if calibration is not None and (
        calibration.bundle_manifest_sha256 != bundle.manifest.manifest_sha256
    ):
        raise CalibrationError("the calibration belongs to another bundle")
    if cad_calibration is not None and (
        cad_calibration.bundle_manifest_sha256 != bundle.manifest.manifest_sha256
    ):
        raise CalibrationError("the CAD calibration belongs to another bundle")

    if mode is ServingMode.FAST:
        prediction = (
            predict(bundle, inputs, queries)
            if confidence is None
            else predict(bundle, inputs, queries, confidence=confidence)
        )
        verdict = getattr(getattr(prediction, "verdict", None), "value", "IN_DOMAIN")
        bounds: Mapping[str, Any] | None = None
        if calibration is not None and verdict != "REFUSED":
            bounds = calibration.evaluate(inputs, "budget_total")
        elif cad_calibration is not None and verdict != "REFUSED":
            bounds = cad_calibration.evaluate(inputs)
        tier0_doc: dict[str, Any] | None = None
        if (tier0_check or emit_receipt) and verdict != "REFUSED":
            tier0_doc = _evaluate_tier0(
                prediction, thresholds=tier0_thresholds
            ).document()
        receipt_doc: dict[str, Any] | None = None
        receipt_sha: str | None = None
        path_fast = (
            ServingPath.REFUSED if verdict == "REFUSED" else ServingPath.OPERATOR
        )
        if emit_receipt:
            receipt_doc, receipt_sha = _build_trust_receipt(
                bundle=bundle,
                mode=mode,
                path=path_fast,
                request=None,
                solver=None,
                tier0=tier0_doc,
                conformal_doc=bounds,
                certificate_doc=None,
                trust_report_doc=None,
                attempts=(),
                exact_converged=False,
                spectral_repair_doc=None,
                reason="",
            )
        return VerifiedPrediction(
            mode,
            path_fast,
            prediction,
            bounds=bounds,
            tier0_admissibility=tier0_doc,
            trust_receipt=receipt_doc,
            trust_receipt_sha256=receipt_sha,
        )

    prediction = predict(bundle, inputs, queries, confidence=bool(confidence))
    if prediction.mesh is None or prediction.verdict.value == "REFUSED":
        refused_reason = "the bundle refused the request: " + ", ".join(
            str(code) for code in prediction.reason_codes
        )
        r_doc: dict[str, Any] | None = None
        r_sha: str | None = None
        if emit_receipt:
            r_doc, r_sha = _build_trust_receipt(
                bundle=bundle,
                mode=mode,
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
                reason=refused_reason,
            )
        return VerifiedPrediction(
            mode,
            ServingPath.REFUSED,
            prediction,
            reason=refused_reason,
            trust_receipt=r_doc,
            trust_receipt_sha256=r_sha,
        )

    if recipe is None and not is_real_portable:
        raise ValueError("certified and exact modes need the bundle's FEM recipe")

    certify_with_correction, backend_type, linearization_type = _fem(mode)
    solver = _resolve_solver(backend, backend_type)
    req_id = request_id or default_request_id(inputs)
    if recipe is not None:
        request = recipe.build_request(inputs, prediction.mesh, req_id)
    else:
        assert isinstance(portable_problem, PortableProblem)
        request = _build_portable_fem_request(
            portable_problem, inputs, prediction.mesh, req_id
        )

    linearization = (
        solver.linearize(request)
        if hasattr(solver, "linearize")
        else linearization_type(request)
    )
    surrogate = operator_displacement(
        bundle, inputs, prediction.mesh, linearization.dof_coordinates
    )
    certificate, corrected_vector = certify_with_correction(linearization, surrogate)
    p_change = certificate.indicators.get("p_l2_change", float("inf"))
    tta_doc: dict[str, Any] | None = None
    if tta_config is not None and (
        certificate.verdict.value != "OK"
        or (tolerance is not None and p_change > tolerance)
    ):
        import torch

        from piginot.bundle.inference import _Request
        from piginot.objectives.energy import TriangleDomain
        from piginot.operators.energy_operator import EnergyFormOperator
        from piginot.serving.test_time_lora import adapt_operator_at_test_time

        base_op = getattr(bundle.serving, "_model", None)
        if isinstance(base_op, EnergyFormOperator):
            req_ctx = _Request(bundle, inputs, prediction.mesh)
            dtype = next(base_op.parameters()).dtype
            dom = TriangleDomain(
                vertices=torch.tensor(prediction.mesh.nodes, dtype=dtype),
                cells=torch.tensor(prediction.mesh.cells, dtype=torch.int64),
            )
            adapted_op, tta_report = adapt_operator_at_test_time(
                base_op,
                domain=dom,
                parameters=req_ctx.parameters.to(dtype=dtype),
                frame=req_ctx.frame,
                lifting=req_ctx.lifting,
                material=req_ctx.material,
                config=tta_config,
            )
            tta_doc = tta_report.document()
            if tta_report.improved:
                with torch.no_grad():
                    pts_t = torch.tensor(
                        linearization.dof_coordinates, dtype=dtype
                    ).unsqueeze(0)
                    u_adapted = adapted_op.displacement(
                        query_xy=pts_t,
                        parameters=req_ctx.parameters.to(dtype=dtype),
                        frame=req_ctx.frame,
                        lifting=req_ctx.lifting,
                    )[0]
                surrogate = u_adapted.detach().cpu().numpy().astype(np.float64)
                certificate, corrected_vector = certify_with_correction(
                    linearization, surrogate
                )
                p_change = certificate.indicators.get("p_l2_change", float("inf"))

    tier0_v_doc: dict[str, Any] | None = None
    if tier0_check or emit_receipt:
        tier0_v_doc = _evaluate_tier0(
            prediction,
            thresholds=tier0_thresholds,
            dirichlet_deviation=certificate.dirichlet_deviation,
        ).document()

    corrected_nodal = (
        linearization.nodal(corrected_vector)
        if corrected and corrected_vector is not None
        else None
    )
    tier0_failed = bool(tier0_v_doc is not None and tier0_v_doc.get("passed") is False)
    needs_exact = mode is ServingMode.EXACT or (
        escalate
        and (
            tier0_failed
            or certificate.verdict.value != "OK"
            or (tolerance is not None and p_change > tolerance)
        )
    )
    request_hash = str(request.recipe_hash)
    bounds_v: Mapping[str, Any] | None = None
    stat_by_qoi: dict[str, Any] | None = None
    if calibration is not None:
        bounds_v = calibration.evaluate(inputs, "reference_terms")
    elif cad_calibration is not None:
        bounds_v = cad_calibration.evaluate(inputs)
        in_cal_env = cad_calibration.in_envelope(inputs)
        stat_by_qoi = {
            name: rec
            for name in cad_calibration.qoi_calibrations
            if (rec := cad_calibration.statistical_record(name, in_envelope=in_cal_env))
            is not None
        }

    if not needs_exact:
        tr_doc: dict[str, Any] | None = None
        tr_sha: str | None = None
        if trust_report:
            tr_doc, tr_sha = _build_live_cad_trust_report(
                bundle=bundle,
                request=request,
                surrogate=surrogate,
                trust_sidecar=trust_sidecar,
                solve_failed=False,
                backend=solver,
                statistical_by_qoi=stat_by_qoi,
            )
        rc_doc: dict[str, Any] | None = None
        rc_sha: str | None = None
        if emit_receipt:
            rc_doc, rc_sha = _build_trust_receipt(
                bundle=bundle,
                mode=mode,
                path=ServingPath.OPERATOR_CERTIFICATE,
                request=request,
                solver=solver,
                tier0=tier0_v_doc,
                conformal_doc=bounds_v,
                certificate_doc=certificate.document(),
                trust_report_doc=tr_doc,
                attempts=(),
                exact_converged=False,
                spectral_repair_doc=None,
                tta_doc=tta_doc,
                reason="",
            )
        return VerifiedPrediction(
            mode,
            ServingPath.OPERATOR_CERTIFICATE,
            prediction,
            certificate.document(),
            corrected_nodal,
            request_hash=request_hash,
            bounds=bounds_v,
            trust_report=tr_doc,
            trust_report_sha256=tr_sha,
            tier0_admissibility=tier0_v_doc,
            test_time_adaptation=tta_doc,
            trust_receipt=rc_doc,
            trust_receipt_sha256=rc_sha,
        )

    start = linearization.with_exact_dirichlet(linearization.vector(surrogate))
    repair_doc: dict[str, Any] | None = None
    if spectral_repair:
        from piginot.serving.spectral_repair import (
            SpectralRepairConfig as RepairCfg,
        )
        from piginot.serving.spectral_repair import (
            spectrally_safe_repair,
        )

        repaired_vec, repair_report = spectrally_safe_repair(
            linearization,
            start,
            config=RepairCfg()
            if spectral_repair_config is None
            else spectral_repair_config,
        )
        repair_doc = repair_report.document()
        if repair_report.needed_repair and repair_report.succeeded:
            start = repaired_vec
    attempt, attempts = solve_ladder(
        solver,
        request,
        start,
        linearization=None,
        spectral_repair=False,
    )
    final_path = (
        ServingPath.REPAIRED_WARM_NEWTON
        if (
            attempt is not None
            and attempts[-1].path is ServingPath.WARM_NEWTON
            and repair_doc is not None
            and repair_doc.get("needed_repair") is True
            and repair_doc.get("succeeded") is True
        )
        else (attempts[-1].path if attempt is not None else ServingPath.SOLVE_FAILED)
    )
    tr_doc_ex: dict[str, Any] | None = None
    tr_sha_ex: str | None = None
    if trust_report:
        tr_doc_ex, tr_sha_ex = _build_live_cad_trust_report(
            bundle=bundle,
            request=request,
            surrogate=surrogate,
            trust_sidecar=trust_sidecar,
            solve_failed=attempt is None,
            backend=solver,
            statistical_by_qoi=stat_by_qoi,
        )
    reason_ex = "" if attempt is not None else "every rung of the solve ladder failed"
    rc_doc_ex: dict[str, Any] | None = None
    rc_sha_ex: str | None = None
    if emit_receipt:
        rc_doc_ex, rc_sha_ex = _build_trust_receipt(
            bundle=bundle,
            mode=mode,
            path=final_path,
            request=request,
            solver=solver,
            tier0=tier0_v_doc,
            conformal_doc=bounds_v,
            certificate_doc=certificate.document(),
            trust_report_doc=tr_doc_ex,
            attempts=attempts,
            exact_converged=attempt is not None,
            spectral_repair_doc=repair_doc,
            tta_doc=tta_doc,
            reason=reason_ex,
        )
    return VerifiedPrediction(
        mode,
        final_path,
        prediction,
        certificate.document(),
        corrected_nodal,
        None if attempt is None else attempt.result,
        None if attempt is None else attempt.result_hash,
        attempts,
        escalated=mode is ServingMode.CERTIFIED,
        request_hash=request_hash,
        reason=reason_ex,
        bounds=bounds_v,
        trust_report=tr_doc_ex,
        trust_report_sha256=tr_sha_ex,
        tier0_admissibility=tier0_v_doc,
        spectral_repair=repair_doc,
        test_time_adaptation=tta_doc,
        trust_receipt=rc_doc_ex,
        trust_receipt_sha256=rc_sha_ex,
    )
