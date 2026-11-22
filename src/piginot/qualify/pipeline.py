"""Continuous numerical model qualification pipeline for FormaMech bundles.

Unifies bundle verification, partition leakage hygiene, Tier 0 admissibility,
authoritative FEM reference comparisons, DEV-only split-conformal + functional
band calibration, live DWR adjoint effectivity checks, optional HOLDOUT
verification, fail-closed ``reduce_families`` claim reduction, break-even
amortization ``N*``, and optional hash-consistent promoted bundle emission.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from piginot.contracts import CanonicalQuantity, Quantity, Unit, canonical_json_bytes
from piginot.qualify.manifest import (
    QualificationCheckResult,
    QualificationManifest,
    QualificationVerdict,
)
from piginot.qualify.policy import QualificationPolicy
from piginot.qualify.report import render_qualification_report_markdown
from piginot.verification.amortization import amortized_break_even_queries
from piginot.verification.reducer import VerificationFamily, reduce_families
from piginot.verification.status import CheckStatus, ClaimLevel

if TYPE_CHECKING:
    from piginot.bounds.cad_calibration import CadCalibrationRecord
    from piginot.bundle import Bundle
    from piginot.certify.adjoint import DifferentiableQoi

PartitionRole = Literal["DEV", "HOLDOUT"]


@dataclass(frozen=True, slots=True)
class QualificationCase:
    """One named, canonicalized qualification case belonging to DEV or HOLDOUT."""

    case_id: str
    partition: PartitionRole
    inputs: Mapping[str, CanonicalQuantity]

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be non-empty text")
        if self.partition not in ("DEV", "HOLDOUT"):
            raise ValueError("partition must be 'DEV' or 'HOLDOUT'")
        cleaned: dict[str, CanonicalQuantity] = {}
        for k, v in sorted(self.inputs.items()):
            if not isinstance(v, CanonicalQuantity):
                raise TypeError(f"input {k!r} must be a CanonicalQuantity")
            cleaned[str(k)] = v
        object.__setattr__(self, "inputs", cleaned)

    @property
    def fingerprint(self) -> str:
        return input_fingerprint(self.inputs)


@dataclass(frozen=True, slots=True)
class _EvaluatedCase:
    case_id: str
    partition: PartitionRole
    fingerprint: str
    in_domain: bool
    admissible: bool
    min_det_f: float | None
    dirichlet_deviation: float
    fem_converged: bool
    disp_rel_l2: float | None
    stress_rel_l2: float | None
    qoi_pred: Mapping[str, float]
    qoi_fem: Mapping[str, float]
    qoi_units: Mapping[str, str]
    dwr_estimates: Mapping[str, float | None]
    dwr_verdicts: Mapping[str, str]
    vertex_disp_residuals: list[float] | None
    inference_seconds: float
    fem_seconds: float
    dof_count: int | None


@dataclass(frozen=True, slots=True)
class QualificationRunResult:
    """Complete output of :func:`qualify_bundle`."""

    manifest: QualificationManifest
    calibration: CadCalibrationRecord | None
    report_markdown: str
    manifest_path: Path | None
    report_path: Path | None
    calibration_path: Path | None
    promoted_bundle_path: Path | None

    @property
    def qualified(self) -> bool:
        return self.manifest.verdict is QualificationVerdict.QUALIFIED


def input_fingerprint(inputs: Mapping[str, CanonicalQuantity]) -> str:
    """Canonical SHA-256 fingerprint of a parameter vector for leakage checks."""
    payload = {
        name: [round(float(q.value), 12), q.unit.value]
        for name, q in sorted(inputs.items())
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def parse_cases_document(
    document: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    default_partition: PartitionRole = "DEV",
) -> tuple[QualificationCase, ...]:
    """Parse JSON case definitions into canonical :class:`QualificationCase` tuples."""
    raw_items: Sequence[Any]
    if isinstance(document, Mapping):
        if "cases" not in document or not isinstance(document["cases"], list):
            raise ValueError("cases document object must contain a 'cases' list")
        raw_items = document["cases"]
    elif isinstance(document, Sequence) and not isinstance(document, (str, bytes)):
        raw_items = document
    else:
        raise ValueError("cases document must be a JSON object or list")

    cases: list[QualificationCase] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            raise ValueError(f"case {idx} must be a JSON object")
        cid = str(item.get("case_id", f"{default_partition.lower()}-{idx:04d}"))
        part_raw = str(item.get("partition", default_partition)).upper()
        if part_raw not in ("DEV", "HOLDOUT"):
            raise ValueError(f"case {cid!r}: invalid partition {part_raw!r}")
        part: PartitionRole = "HOLDOUT" if part_raw == "HOLDOUT" else "DEV"
        raw_inputs = item.get("inputs")
        if not isinstance(raw_inputs, Mapping):
            raise ValueError(f"case {cid!r}: 'inputs' must be a JSON object")
        parsed: dict[str, CanonicalQuantity] = {}
        for name, val in raw_inputs.items():
            if not isinstance(val, Mapping) or "value" not in val or "unit" not in val:
                raise ValueError(
                    f"case {cid!r} input {name!r} must have 'value' and 'unit'"
                )
            parsed[str(name)] = Quantity(
                float(val["value"]), Unit(str(val["unit"]))
            ).to_canonical()
        cases.append(QualificationCase(case_id=cid, partition=part, inputs=parsed))
    return tuple(cases)


def load_cases_file(
    path: Path, *, default_partition: PartitionRole = "DEV"
) -> tuple[QualificationCase, ...]:
    """Load canonical qualification cases from a JSON file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return parse_cases_document(raw, default_partition=default_partition)


def sample_qualification_cases(
    bundle: Bundle,
    *,
    dev_count: int,
    holdout_count: int = 0,
    seed: int = 0,
    exclude_fingerprints: frozenset[str] = frozenset(),
) -> tuple[tuple[QualificationCase, ...], tuple[QualificationCase, ...]]:
    """Deterministically sample disjoint in-domain DEV and HOLDOUT cases from envelope."""
    if dev_count < 0 or holdout_count < 0:
        raise ValueError("dev_count and holdout_count must be non-negative")
    total = dev_count + holdout_count
    if total == 0:
        return (), ()

    import numpy as np

    bounds = bundle.envelope.primitive_bounds
    rng = np.random.default_rng(seed)
    seen = set(exclude_fingerprints)
    accepted: list[dict[str, CanonicalQuantity]] = []

    # Latin-hypercube strata per dimension, oversampling up to 20x if derived bounds reject.
    max_attempts = max(total * 20, 100)
    strata_order = [rng.permutation(max(total, 1)) for _ in bounds]
    attempt = 0
    while len(accepted) < total and attempt < max_attempts:
        candidate: dict[str, CanonicalQuantity] = {}
        for dim_idx, bound in enumerate(bounds):
            lo = float(bound.lower_bound.value)
            hi = float(bound.upper_bound.value)
            unit = bound.lower_bound.unit
            if hi <= lo:
                val = lo
            elif attempt < total:
                stratum = int(strata_order[dim_idx][attempt])
                u = (stratum + float(rng.uniform(0.05, 0.95))) / float(total)
                val = lo + u * (hi - lo)
            else:
                val = float(rng.uniform(lo, hi))
            candidate[bound.name] = Quantity(val, unit).to_canonical()
        attempt += 1

        fp = input_fingerprint(candidate)
        if fp in seen:
            continue
        decision = bundle.checker.check(candidate)
        if decision.accepted and bundle.geometry(candidate).mesh is not None:
            seen.add(fp)
            accepted.append(candidate)

    if len(accepted) < total:
        raise ValueError(
            f"sampled only {len(accepted)}/{total} admissible in-envelope cases "
            f"after {max_attempts} attempts"
        )

    dev = tuple(
        QualificationCase(
            case_id=f"dev-{i:04d}",
            partition="DEV",
            inputs=accepted[i],
        )
        for i in range(dev_count)
    )
    holdout = tuple(
        QualificationCase(
            case_id=f"holdout-{i:04d}",
            partition="HOLDOUT",
            inputs=accepted[dev_count + i],
        )
        for i in range(holdout_count)
    )
    return dev, holdout


def _build_qois(request: Any) -> tuple[DifferentiableQoi, ...]:
    from piginot.certify.adjoint import (
        ReactionForceQoi,
        RegionalAverageStressQoi,
        TotalStrainEnergyQoi,
    )

    qois: list[DifferentiableQoi] = []
    for bc in request.boundary_conditions:
        if bc.kind == "dirichlet" and any(abs(v.value) > 0.0 for v in bc.values):
            qois.append(ReactionForceQoi(f"reaction_{bc.name}", bc.name))
    qois.append(RegionalAverageStressQoi("avg_p11", component=(0, 0)))
    qois.append(TotalStrainEnergyQoi("total_strain_energy"))
    return tuple(qois)


def _quantile_stat(values: Sequence[float], q: float) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    if q >= 1.0:
        return float(np.max(arr))
    return float(np.quantile(arr, q, method="higher"))


def _evaluate_case(
    *,
    case: QualificationCase,
    bundle: Bundle,
    recipe: Any | None,
    solver: Any,
    policy: QualificationPolicy,
    warm_start_fem: bool = True,
) -> _EvaluatedCase:
    import numpy as np

    from piginot.bundle import predict
    from piginot.bundle.inference import Verdict
    from piginot.bundle.problem import PortableProblem
    from piginot.certify.adjoint import estimate_qoi_dwr
    from piginot.certify.admissibility import (
        AdmissibilityThresholds,
        evaluate_tier0_admissibility,
    )
    from piginot.serving.hybrid import (
        _build_portable_fem_request,
        operator_displacement,
        solve_ladder,
    )

    t0 = time.perf_counter()
    pred = predict(bundle, case.inputs, None, confidence=False)
    infer_s = max(time.perf_counter() - t0, 1.0e-9)

    if pred.mesh is None or pred.verdict is Verdict.REFUSED:
        return _EvaluatedCase(
            case_id=case.case_id,
            partition=case.partition,
            fingerprint=case.fingerprint,
            in_domain=False,
            admissible=False,
            min_det_f=None,
            dirichlet_deviation=float("inf"),
            fem_converged=False,
            disp_rel_l2=None,
            stress_rel_l2=None,
            qoi_pred={},
            qoi_fem={},
            qoi_units={},
            dwr_estimates={},
            dwr_verdicts={},
            vertex_disp_residuals=None,
            inference_seconds=infer_s,
            fem_seconds=0.0,
            dof_count=None,
        )

    # Tier 0 admissibility on predicted vertex fields
    vertices = getattr(pred, "vertices", None)
    fields: dict[str, Any] = {}
    if vertices is not None:
        for fname in ("u", "F_2D", "F33", "J_3D", "P_served", "sigma"):
            val = getattr(vertices, fname, None)
            if val is not None:
                fields[fname] = (
                    val.detach().cpu().numpy() if hasattr(val, "detach") else val
                )

    portable_problem = getattr(bundle.model, "problem", None)
    req_id = f"qual-{case.case_id}"
    if recipe is not None:
        fem_req = recipe.build_request(case.inputs, pred.mesh, req_id)
    elif isinstance(portable_problem, PortableProblem):
        fem_req = _build_portable_fem_request(
            portable_problem, case.inputs, pred.mesh, req_id
        )
    else:
        raise ValueError(
            "bundle has neither a PortableProblem nor a bound FemRecipe for FEM evaluation"
        )

    linearization = solver.linearize(fem_req)
    surrogate_nodal = operator_displacement(
        bundle, case.inputs, pred.mesh, linearization.dof_coordinates
    )
    surrogate_vec = linearization.vector(surrogate_nodal)
    dirichlet_dev = float(linearization.dirichlet_deviation(surrogate_vec))

    tier0 = evaluate_tier0_admissibility(
        fields,
        thresholds=AdmissibilityThresholds(
            min_det_f_2d=policy.field_tolerances.min_det_f_floor,
            min_det_f_3d=policy.field_tolerances.min_det_f_floor,
            max_dirichlet_deviation=policy.field_tolerances.max_dirichlet_residual,
        ),
        dirichlet_deviation=dirichlet_dev,
    )

    t_fem0 = time.perf_counter()
    if warm_start_fem:
        start_vec = linearization.with_exact_dirichlet(surrogate_vec)
        attempt, _ = solve_ladder(
            solver,
            fem_req,
            start_vec,
            linearization=linearization,
            spectral_repair=True,
        )
    else:
        attempt = solver.solve_many((fem_req,))[0]
    fem_s = max(time.perf_counter() - t_fem0, 1.0e-9)

    if attempt is None or not attempt.success or attempt.result is None:
        return _EvaluatedCase(
            case_id=case.case_id,
            partition=case.partition,
            fingerprint=case.fingerprint,
            in_domain=True,
            admissible=tier0.passed,
            min_det_f=tier0.min_det_f_2d,
            dirichlet_deviation=dirichlet_dev,
            fem_converged=False,
            disp_rel_l2=None,
            stress_rel_l2=None,
            qoi_pred={},
            qoi_fem={},
            qoi_units={},
            dwr_estimates={},
            dwr_verdicts={},
            vertex_disp_residuals=None,
            inference_seconds=infer_s,
            fem_seconds=fem_s,
            dof_count=int(linearization.dof_count),
        )

    u_fem_nodal = np.asarray(attempt.result.displacement, dtype=np.float64)
    u_diff = surrogate_nodal - u_fem_nodal
    disp_rel_l2 = float(
        np.linalg.norm(u_diff) / max(float(np.linalg.norm(u_fem_nodal)), 1.0e-12)
    )
    vertex_res = np.linalg.norm(u_diff, axis=-1).tolist()

    pred_state = linearization.state(surrogate_vec)
    fem_state = linearization.state(linearization.vector(u_fem_nodal))

    p_pred = np.asarray(linearization.recovered_stress(pred_state), dtype=np.float64)
    p_fem = np.asarray(attempt.result.recovered_first_piola_stress, dtype=np.float64)
    stress_rel_l2 = float(
        np.linalg.norm(p_pred - p_fem) / max(float(np.linalg.norm(p_fem)), 1.0e-12)
    )

    qois = _build_qois(fem_req)
    q_pred_map: dict[str, float] = {}
    q_fem_map: dict[str, float] = {}
    q_unit_map: dict[str, str] = {}
    dwr_est_map: dict[str, float | None] = {}
    dwr_verdict_map: dict[str, str] = {}
    for qoi in qois:
        q_pred_map[qoi.name] = float(qoi.evaluate(linearization, pred_state))
        q_fem_map[qoi.name] = float(qoi.evaluate(linearization, fem_state))
        q_unit_map[qoi.name] = qoi.unit
        if policy.dwr.enabled and case.partition == "DEV":
            dwr = estimate_qoi_dwr(linearization, surrogate_nodal, qoi)
            dwr_est_map[qoi.name] = dwr.dwr_estimate
            dwr_verdict_map[qoi.name] = dwr.verdict.value

    return _EvaluatedCase(
        case_id=case.case_id,
        partition=case.partition,
        fingerprint=case.fingerprint,
        in_domain=True,
        admissible=tier0.passed,
        min_det_f=tier0.min_det_f_2d,
        dirichlet_deviation=dirichlet_dev,
        fem_converged=True,
        disp_rel_l2=disp_rel_l2,
        stress_rel_l2=stress_rel_l2,
        qoi_pred=q_pred_map,
        qoi_fem=q_fem_map,
        qoi_units=q_unit_map,
        dwr_estimates=dwr_est_map,
        dwr_verdicts=dwr_verdict_map,
        vertex_disp_residuals=vertex_res,
        inference_seconds=infer_s,
        fem_seconds=fem_s,
        dof_count=int(linearization.dof_count),
    )


def _summarize_partition(
    cases: Sequence[_EvaluatedCase], policy: QualificationPolicy
) -> dict[str, Any]:
    q = policy.field_tolerances.quantile
    disp_errs = [c.disp_rel_l2 for c in cases if c.disp_rel_l2 is not None]
    stress_errs = [c.stress_rel_l2 for c in cases if c.stress_rel_l2 is not None]
    det_fs = [c.min_det_f for c in cases if c.min_det_f is not None]
    dir_devs = [
        c.dirichlet_deviation for c in cases if math.isfinite(c.dirichlet_deviation)
    ]

    qoi_names = sorted({name for c in cases for name in c.qoi_fem})
    qoi_summary: dict[str, dict[str, float]] = {}
    for name in qoi_names:
        tol = policy.qoi_tolerances.get(name)
        q_qoi = tol.quantile if tol is not None else q
        rel_floor = tol.rel_floor if tol is not None else 1.0e-12
        abs_errs: list[float] = []
        rel_errs: list[float] = []
        for c in cases:
            if name in c.qoi_fem and name in c.qoi_pred:
                ae = abs(c.qoi_fem[name] - c.qoi_pred[name])
                re = ae / max(abs(c.qoi_fem[name]), rel_floor)
                abs_errs.append(ae)
                rel_errs.append(re)
        if abs_errs:
            qoi_summary[name] = {
                "abs_quantile": _quantile_stat(abs_errs, q_qoi),
                "abs_max": float(max(abs_errs)),
                "rel_quantile": _quantile_stat(rel_errs, q_qoi),
                "rel_max": float(max(rel_errs)),
            }

    return {
        "case_count": len(cases),
        "in_domain_count": sum(1 for c in cases if c.in_domain),
        "admissible_count": sum(1 for c in cases if c.admissible),
        "fem_converged_count": sum(1 for c in cases if c.fem_converged),
        "disp_rel_l2_quantile": _quantile_stat(disp_errs, q) if disp_errs else None,
        "disp_rel_l2_max": float(max(disp_errs)) if disp_errs else None,
        "stress_rel_l2_quantile": (
            _quantile_stat(stress_errs, q) if stress_errs else None
        ),
        "stress_rel_l2_max": float(max(stress_errs)) if stress_errs else None,
        "min_det_f": float(min(det_fs)) if det_fs else None,
        "max_dirichlet_deviation": float(max(dir_devs)) if dir_devs else None,
        "qoi_errors": qoi_summary,
    }


def _accuracy_checks(
    *,
    prefix: str,
    family_id: str,
    cases: Sequence[_EvaluatedCase],
    summary: Mapping[str, Any],
    policy: QualificationPolicy,
) -> list[QualificationCheckResult]:
    out: list[QualificationCheckResult] = []
    if not cases:
        out.append(
            QualificationCheckResult(
                check_id=f"{prefix}.non_empty",
                family_id=family_id,
                status=CheckStatus.FAIL,
                detail=f"{prefix} partition has zero evaluated cases",
                measured={"case_count": 0},
            )
        )
        return out

    failed_fem = sum(1 for c in cases if not c.fem_converged)
    fem_ok = (failed_fem == 0) or (not policy.require_zero_fem_failures)
    out.append(
        QualificationCheckResult(
            check_id=f"{prefix}.fem_convergence",
            family_id=family_id,
            status=CheckStatus.PASS if fem_ok else CheckStatus.FAIL,
            detail=(
                f"{len(cases) - failed_fem}/{len(cases)} reference FEM solves converged"
            ),
            measured={"converged": len(cases) - failed_fem, "failed": failed_fem},
        )
    )

    disp_q = summary.get("disp_rel_l2_quantile")
    disp_ok = (
        isinstance(disp_q, float)
        and math.isfinite(disp_q)
        and disp_q <= policy.field_tolerances.disp_rel_l2_max
    )
    out.append(
        QualificationCheckResult(
            check_id=f"{prefix}.disp_rel_l2",
            family_id=family_id,
            status=CheckStatus.PASS if disp_ok else CheckStatus.FAIL,
            detail=(
                f"disp rel-L2 q={policy.field_tolerances.quantile}: "
                f"{disp_q} <= {policy.field_tolerances.disp_rel_l2_max}"
            ),
            measured={
                "value": disp_q,
                "limit": policy.field_tolerances.disp_rel_l2_max,
                "quantile": policy.field_tolerances.quantile,
            },
        )
    )

    stress_q = summary.get("stress_rel_l2_quantile")
    stress_ok = (
        isinstance(stress_q, float)
        and math.isfinite(stress_q)
        and stress_q <= policy.field_tolerances.stress_rel_l2_max
    )
    out.append(
        QualificationCheckResult(
            check_id=f"{prefix}.stress_rel_l2",
            family_id=family_id,
            status=CheckStatus.PASS if stress_ok else CheckStatus.FAIL,
            detail=(
                f"stress rel-L2 q={policy.field_tolerances.quantile}: "
                f"{stress_q} <= {policy.field_tolerances.stress_rel_l2_max}"
            ),
            measured={
                "value": stress_q,
                "limit": policy.field_tolerances.stress_rel_l2_max,
                "quantile": policy.field_tolerances.quantile,
            },
        )
    )

    qoi_summary = summary.get("qoi_errors", {})
    for qoi_name, tol in sorted(policy.qoi_tolerances.items()):
        entry = qoi_summary.get(qoi_name) if isinstance(qoi_summary, Mapping) else None
        if not isinstance(entry, Mapping):
            out.append(
                QualificationCheckResult(
                    check_id=f"{prefix}.qoi.{qoi_name}",
                    family_id=family_id,
                    status=CheckStatus.FAIL,
                    detail=f"required QoI {qoi_name!r} was not evaluated",
                    measured={},
                )
            )
            continue
        ok_rel = (
            tol.max_rel_error is None
            or float(entry["rel_quantile"]) <= tol.max_rel_error
        )
        ok_abs = (
            tol.max_abs_error is None
            or float(entry["abs_quantile"]) <= tol.max_abs_error
        )
        out.append(
            QualificationCheckResult(
                check_id=f"{prefix}.qoi.{qoi_name}",
                family_id=family_id,
                status=CheckStatus.PASS if (ok_rel and ok_abs) else CheckStatus.FAIL,
                detail=(
                    f"QoI {qoi_name}: rel={entry['rel_quantile']:.4e} "
                    f"(limit={tol.max_rel_error}), abs={entry['abs_quantile']:.4e} "
                    f"(limit={tol.max_abs_error})"
                ),
                measured={
                    "rel_quantile": float(entry["rel_quantile"]),
                    "abs_quantile": float(entry["abs_quantile"]),
                    "max_rel_error": tol.max_rel_error,
                    "max_abs_error": tol.max_abs_error,
                },
            )
        )
    return out


def _emit_promoted_bundle(
    *,
    source_bundle: Bundle,
    destination: Path,
    manifest: QualificationManifest,
) -> Path:
    """Emit a new hash-consistent bundle carrying the granted numerical claim."""
    from piginot.bundle import build_bundle
    from piginot.bundle.inference import WEIGHTS_PATH

    if manifest.verdict is not QualificationVerdict.QUALIFIED:
        raise ValueError("cannot emit a promoted bundle for a NOT_QUALIFIED manifest")

    weights_bytes = (source_bundle.path / WEIGHTS_PATH).read_bytes()
    existing_items = list(source_bundle.limitations.get("items", ()))
    qual_item = {
        "id": "numerical-qualification",
        "statement": (
            f"Numerically qualified at {manifest.granted_claim_level.value} under "
            f"policy {manifest.policy_id} (manifest {manifest.manifest_sha256[:16]}). "
            "Numerical verification against FEM only; no experimental validation."
        ),
    }
    filtered_items = [
        item for item in existing_items if item.get("id") != "numerical-qualification"
    ]
    filtered_items.append(qual_item)

    verification_doc: dict[str, Any] = {
        "gate_id": "G-QUALIFY",
        "verdict": "PASS",
        "claim_level": manifest.granted_claim_level.value,
        "policy_id": manifest.policy_id,
        "policy_sha256": manifest.policy_sha256,
        "qualification_manifest_sha256": manifest.manifest_sha256,
        "source_bundle_manifest_sha256": source_bundle.manifest.manifest_sha256,
        "families": [dict(f) for f in manifest.families],
        "outstanding_work": [],
    }
    identities = {
        **dict(source_bundle.manifest.identities),
        "source_bundle_manifest_sha256": source_bundle.manifest.manifest_sha256,
        "qualification_manifest_sha256": manifest.manifest_sha256,
        "qualification_policy_sha256": manifest.policy_sha256,
    }
    build_bundle(
        destination,
        name=source_bundle.manifest.name,
        created_utc=manifest.created_utc,
        claim_level=manifest.granted_claim_level.value,
        reference_mesh=source_bundle.reference_mesh,
        morph_map=source_bundle.morph_map,
        envelope=source_bundle.envelope,
        model=source_bundle.model,
        weights=weights_bytes,
        verification=verification_doc,
        limitation_items=filtered_items,
        identities=identities,
    )
    return destination


def qualify_bundle(
    bundle_path: Path | str,
    policy: QualificationPolicy,
    *,
    dev_cases: Sequence[QualificationCase],
    holdout_cases: Sequence[QualificationCase] = (),
    train_fingerprints: frozenset[str] = frozenset(),
    recipe_path: Path | str | None = None,
    backend: Any = "scikit-fem",
    warm_start_fem: bool = True,
    output_dir: Path | str | None = None,
    emit_qualified_bundle: Path | str | None = None,
    created_utc: str | None = None,
) -> QualificationRunResult:
    """Execute the full fail-closed numerical qualification pipeline on a bundle."""
    import numpy as np

    from piginot.bounds.cad_calibration import build_cad_calibration
    from piginot.bundle import load_bundle
    from piginot.bundle.inference import ENVELOPE_PATH
    from piginot.bundle.problem import PortableProblem
    from piginot.serving.hybrid import _resolve_solver, load_recipe

    bpath = Path(bundle_path)
    bundle = load_bundle(bpath)
    bundle_sha = bundle.manifest.manifest_sha256
    envelope_sha = bundle.manifest.files[ENVELOPE_PATH].sha256
    portable_problem = getattr(bundle.model, "problem", None)
    problem_hash = (
        portable_problem.problem_hash
        if isinstance(portable_problem, PortableProblem)
        else None
    )

    from piginot.fem.backend import ScikitFemBackend

    solver = _resolve_solver(backend, ScikitFemBackend)
    solver_id = str(
        getattr(solver, "backend_id", getattr(solver, "backend_name", "unknown"))
    )

    recipe = None
    if recipe_path is not None:
        recipe = load_recipe(bundle, bpath, Path(recipe_path))
    elif not isinstance(portable_problem, PortableProblem):
        recipe = load_recipe(bundle, bpath)

    checks: list[QualificationCheckResult] = []
    family_specs: list[VerificationFamily] = []

    # 1. Integrity family
    checks.append(
        QualificationCheckResult(
            check_id="integrity.bundle_verified",
            family_id="integrity",
            status=CheckStatus.PASS,
            detail=f"bundle manifest {bundle_sha[:16]} verified",
            measured={"bundle_manifest_sha256": bundle_sha},
        )
    )
    family_specs.append(
        VerificationFamily("integrity", True, ("integrity.bundle_verified",))
    )

    # 2. Partition hygiene (zero leakage between TRAIN, DEV, HOLDOUT)
    dev_fps = [c.fingerprint for c in dev_cases]
    holdout_fps = [c.fingerprint for c in holdout_cases]
    dev_set = set(dev_fps)
    holdout_set = set(holdout_fps)
    overlap_dev_holdout = dev_set & holdout_set
    overlap_train = (dev_set | holdout_set) & set(train_fingerprints)
    dup_dev = len(dev_fps) - len(dev_set)
    dup_holdout = len(holdout_fps) - len(holdout_set)

    hygiene_ok = (
        len(dev_cases) > 0
        and len(overlap_dev_holdout) == 0
        and len(overlap_train) == 0
        and dup_dev == 0
        and dup_holdout == 0
    )
    checks.append(
        QualificationCheckResult(
            check_id="partition.disjointness",
            family_id="partition_hygiene",
            status=CheckStatus.PASS if hygiene_ok else CheckStatus.FAIL,
            detail=(
                f"dev={len(dev_cases)}, holdout={len(holdout_cases)}, "
                f"dev_holdout_overlap={len(overlap_dev_holdout)}, "
                f"train_overlap={len(overlap_train)}, "
                f"duplicates={dup_dev + dup_holdout}"
            ),
            measured={
                "dev_count": len(dev_cases),
                "holdout_count": len(holdout_cases),
                "dev_holdout_overlap": len(overlap_dev_holdout),
                "train_overlap": len(overlap_train),
                "duplicates": dup_dev + dup_holdout,
            },
        )
    )
    family_specs.append(
        VerificationFamily(
            "partition_hygiene",
            policy.require_partition_disjointness,
            ("partition.disjointness",),
        )
    )

    # Evaluate DEV & HOLDOUT cases against authoritative FEM
    eval_dev = tuple(
        _evaluate_case(
            case=c,
            bundle=bundle,
            recipe=recipe,
            solver=solver,
            policy=policy,
            warm_start_fem=warm_start_fem,
        )
        for c in dev_cases
    )
    eval_holdout = tuple(
        _evaluate_case(
            case=c,
            bundle=bundle,
            recipe=recipe,
            solver=solver,
            policy=policy,
            warm_start_fem=warm_start_fem,
        )
        for c in holdout_cases
    )
    all_eval = (*eval_dev, *eval_holdout)

    # 3. Tier 0 admissibility family
    in_domain_fail = sum(1 for c in all_eval if not c.in_domain)
    inadmissible_count = sum(1 for c in all_eval if not c.admissible)
    adm_ok = (in_domain_fail == 0) and (
        inadmissible_count == 0 or not policy.require_zero_inadmissible
    )
    checks.append(
        QualificationCheckResult(
            check_id="admissibility.tier0_sweep",
            family_id="admissibility",
            status=CheckStatus.PASS if adm_ok else CheckStatus.FAIL,
            detail=(
                f"{len(all_eval) - inadmissible_count}/{len(all_eval)} cases admissible; "
                f"refused={in_domain_fail}"
            ),
            measured={
                "total": len(all_eval),
                "inadmissible": inadmissible_count,
                "out_of_domain": in_domain_fail,
            },
        )
    )
    family_specs.append(
        VerificationFamily("admissibility", True, ("admissibility.tier0_sweep",))
    )

    # 4. DEV FEM accuracy family
    dev_summary = _summarize_partition(eval_dev, policy)
    dev_acc_checks = _accuracy_checks(
        prefix="dev",
        family_id="fem_accuracy_dev",
        cases=eval_dev,
        summary=dev_summary,
        policy=policy,
    )
    checks.extend(dev_acc_checks)
    family_specs.append(
        VerificationFamily(
            "fem_accuracy_dev",
            True,
            tuple(c.check_id for c in dev_acc_checks),
        )
    )

    # 5. DEV-only split-conformal & functional-band calibration family
    calibration_record: CadCalibrationRecord | None = None
    cal_checks: list[QualificationCheckResult] = []
    if policy.calibration.enabled:
        converged_dev = [c for c in eval_dev if c.fem_converged]
        n_cal = len(converged_dev)
        enough_samples = n_cal >= policy.calibration.min_calibration_cases
        if enough_samples:
            qoi_scores: dict[str, tuple[str, list[float]]] = {}
            qoi_names = sorted({k for c in converged_dev for k in c.qoi_fem})
            for qname in qoi_names:
                unit = next(
                    (c.qoi_units[qname] for c in converged_dev if qname in c.qoi_units),
                    "1",
                )
                scores = [
                    abs(c.qoi_fem[qname] - c.qoi_pred[qname])
                    for c in converged_dev
                    if qname in c.qoi_fem and qname in c.qoi_pred
                ]
                if len(scores) >= policy.calibration.min_calibration_cases:
                    qoi_scores[qname] = (unit, scores)

            func_res: dict[str, tuple[str, list[list[float]]]] | None = None
            if policy.calibration.require_functional_band:
                disp_rows = [
                    c.vertex_disp_residuals
                    for c in converged_dev
                    if c.vertex_disp_residuals is not None
                ]
                if len(disp_rows) >= policy.calibration.min_calibration_cases:
                    func_res = {"displacement_mm": ("mm", disp_rows)}

            param_bounds = {
                b.name: (float(b.lower_bound.value), float(b.upper_bound.value))
                for b in bundle.envelope.primitive_bounds
            }
            if qoi_scores:
                calibration_record = build_cad_calibration(
                    bundle_name=bundle.manifest.name,
                    bundle_manifest_sha256=bundle_sha,
                    problem_hash=problem_hash,
                    qoi_scores=qoi_scores,
                    alpha=policy.calibration.alpha,
                    functional_residuals=func_res,
                    parameter_bounds=param_bounds,
                )

        cal_built_ok = calibration_record is not None and (
            (not policy.calibration.require_functional_band)
            or bool(calibration_record.functional_bands)
        )
        cal_checks.append(
            QualificationCheckResult(
                check_id="calibration.dev_fit",
                family_id="calibration",
                status=CheckStatus.PASS if cal_built_ok else CheckStatus.FAIL,
                detail=(
                    f"calibrated on {n_cal} DEV cases "
                    f"(min={policy.calibration.min_calibration_cases}, "
                    f"alpha={policy.calibration.alpha})"
                ),
                measured={
                    "sample_count": n_cal,
                    "min_required": policy.calibration.min_calibration_cases,
                    "alpha": policy.calibration.alpha,
                    "functional_bands": (
                        sorted(calibration_record.functional_bands)
                        if calibration_record is not None
                        else []
                    ),
                },
            )
        )

        # Empirical holdout coverage check if requested
        if (
            policy.calibration.min_holdout_coverage is not None
            and policy.target_claim_level is ClaimLevel.HOLDOUT_VERIFIED
        ):
            converged_ho = [c for c in eval_holdout if c.fem_converged]
            if calibration_record is None or not converged_ho:
                cal_checks.append(
                    QualificationCheckResult(
                        check_id="calibration.holdout_coverage",
                        family_id="calibration",
                        status=CheckStatus.FAIL,
                        detail="cannot measure holdout coverage without calibration and holdout cases",
                        measured={},
                    )
                )
            else:
                coverages: dict[str, float] = {}
                for qname, qcal in calibration_record.qoi_calibrations.items():
                    covered = sum(
                        1
                        for c in converged_ho
                        if qname in c.qoi_fem
                        and abs(c.qoi_fem[qname] - c.qoi_pred[qname])
                        <= qcal.quantile + 1.0e-12
                    )
                    coverages[qname] = covered / float(len(converged_ho))
                min_cov = min(coverages.values()) if coverages else 0.0
                cov_ok = min_cov >= policy.calibration.min_holdout_coverage
                cal_checks.append(
                    QualificationCheckResult(
                        check_id="calibration.holdout_coverage",
                        family_id="calibration",
                        status=CheckStatus.PASS if cov_ok else CheckStatus.FAIL,
                        detail=(
                            f"min empirical holdout coverage {min_cov:.3f} "
                            f"(limit >= {policy.calibration.min_holdout_coverage:.3f})"
                        ),
                        measured={
                            "per_qoi_coverage": coverages,
                            "min_coverage": min_cov,
                            "limit": policy.calibration.min_holdout_coverage,
                        },
                    )
                )
        checks.extend(cal_checks)
        family_specs.append(
            VerificationFamily(
                "calibration",
                True,
                tuple(c.check_id for c in cal_checks),
            )
        )

    # 6. DWR adjoint effectivity family
    if policy.dwr.enabled:
        eff_values: list[float] = []
        valid_count = 0
        total_tested = 0
        for c in eval_dev:
            if not c.fem_converged:
                continue
            for qname, est in c.dwr_estimates.items():
                if qname not in c.qoi_fem or qname not in c.qoi_pred:
                    continue
                true_err = abs(c.qoi_fem[qname] - c.qoi_pred[qname])
                if true_err <= policy.dwr.error_floor:
                    continue
                total_tested += 1
                if est is not None and math.isfinite(est):
                    eff = abs(est) / true_err
                    eff_values.append(eff)
                    if policy.dwr.effectivity_min <= eff <= policy.dwr.effectivity_max:
                        valid_count += 1
        valid_frac = 1.0 if total_tested == 0 else valid_count / float(total_tested)
        dwr_ok = valid_frac >= policy.dwr.min_valid_fraction
        checks.append(
            QualificationCheckResult(
                check_id="dwr.effectivity_band",
                family_id="dwr_trust",
                status=CheckStatus.PASS if dwr_ok else CheckStatus.FAIL,
                detail=(
                    f"{valid_count}/{total_tested} DWR estimates within "
                    f"[{policy.dwr.effectivity_min}, {policy.dwr.effectivity_max}] "
                    f"(fraction={valid_frac:.3f} >= {policy.dwr.min_valid_fraction:.3f})"
                ),
                measured={
                    "valid_count": valid_count,
                    "total_tested": total_tested,
                    "valid_fraction": valid_frac,
                    "median_effectivity": (
                        float(np.median(eff_values)) if eff_values else None
                    ),
                },
            )
        )
        family_specs.append(
            VerificationFamily("dwr_trust", True, ("dwr.effectivity_band",))
        )

    # 7. HOLDOUT verification family (required when targeting HOLDOUT_VERIFIED)
    holdout_summary: dict[str, Any] | None = None
    need_holdout = policy.target_claim_level is ClaimLevel.HOLDOUT_VERIFIED
    if need_holdout or eval_holdout:
        holdout_summary = _summarize_partition(eval_holdout, policy)
        ho_checks = _accuracy_checks(
            prefix="holdout",
            family_id="holdout_verification",
            cases=eval_holdout,
            summary=holdout_summary,
            policy=policy,
        )
        checks.extend(ho_checks)
        family_specs.append(
            VerificationFamily(
                "holdout_verification",
                need_holdout,
                tuple(c.check_id for c in ho_checks),
            )
        )

    # 8. Reduce families fail-closed using existing S14 reducer
    statuses = {c.check_id: c.status for c in checks}
    reduction = reduce_families(
        tuple(family_specs),
        statuses,
        evidence_level=policy.target_claim_level,
    )
    verdict = (
        QualificationVerdict.QUALIFIED
        if reduction.verified
        else QualificationVerdict.NOT_QUALIFIED
    )

    # 9. Break-even amortization economics
    amort_doc: dict[str, Any] | None = None
    if policy.report_breakeven and eval_dev:
        mean_fem = float(np.mean([c.fem_seconds for c in eval_dev]))
        mean_inf = float(np.mean([c.inference_seconds for c in eval_dev]))
        dof = next((c.dof_count for c in eval_dev if c.dof_count is not None), None)
        if mean_fem > mean_inf > 0.0:
            amort = amortized_break_even_queries(
                training_wall_seconds=policy.training_wall_seconds,
                calibration_cases=len(eval_dev),
                fem_seconds=mean_fem,
                inference_seconds=mean_inf,
                dof_count=dof,
            )
            amort_doc = amort.document()

    # 10. Persist calibration sidecar, manifest, report, and optional promoted bundle
    out_path = None if output_dir is None else Path(output_dir)
    cal_file_path: Path | None = None
    if out_path is not None and calibration_record is not None:
        cal_file_path = out_path / "cad-calibration.json"
        calibration_record.save(cal_file_path)

    ts = (
        created_utc
        if created_utc is not None
        else datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    manifest = QualificationManifest(
        created_utc=ts,
        bundle_name=bundle.manifest.name,
        bundle_manifest_sha256=bundle_sha,
        approved_envelope_sha256=envelope_sha,
        problem_hash=problem_hash,
        policy_id=policy.policy_id,
        policy_sha256=policy.policy_sha256,
        target_claim_level=policy.target_claim_level,
        granted_claim_level=reduction.claim_level,
        verdict=verdict,
        solver_backend_id=solver_id,
        families=tuple(
            {
                "family_id": fam.family_id,
                "required": fam.required,
                "status": fam.status.value,
                "missing_checks": list(fam.missing_checks),
            }
            for fam in reduction.families
        ),
        checks=tuple(checks),
        dev_metrics=dev_summary,
        holdout_metrics=holdout_summary,
        calibration_sha256=(
            None
            if calibration_record is None
            else calibration_record.calibration_sha256
        ),
        calibration_path=None if cal_file_path is None else cal_file_path.name,
        amortization=amort_doc,
    )
    report_md = render_qualification_report_markdown(manifest)

    manifest_file_path: Path | None = None
    report_file_path: Path | None = None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        manifest_file_path = out_path / "qualification-manifest.json"
        manifest.save(manifest_file_path)
        report_file_path = out_path / "qualification-report.md"
        report_file_path.write_text(report_md, encoding="utf-8")

    promoted_path: Path | None = None
    if emit_qualified_bundle is not None:
        if verdict is QualificationVerdict.QUALIFIED:
            promoted_path = _emit_promoted_bundle(
                source_bundle=bundle,
                destination=Path(emit_qualified_bundle),
                manifest=manifest,
            )

    return QualificationRunResult(
        manifest=manifest,
        calibration=calibration_record,
        report_markdown=report_md,
        manifest_path=manifest_file_path,
        report_path=report_file_path,
        calibration_path=cal_file_path,
        promoted_bundle_path=promoted_path,
    )


__all__ = (
    "PartitionRole",
    "QualificationCase",
    "QualificationRunResult",
    "input_fingerprint",
    "load_cases_file",
    "parse_cases_document",
    "qualify_bundle",
    "sample_qualification_cases",
)
