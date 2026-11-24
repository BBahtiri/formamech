"""``piginot`` command line.

Framework commands work on any bundle or store: ``bundle verify``, ``predict``,
``inspect`` and ``store prune``. Campaign commands (``family compile``,
``pilot``, ``campaign``, ``simulate``, ``train``, ``verify``,
``holdout-reveal`` and ``bundle build``) are provided by a campaign plugin
named with ``--campaign``: an entry point in the ``piginot.campaigns`` group,
a ``module:attribute`` path, or a bare name resolved as
``campaigns.<name>.cli:COMMANDS``. A plugin maps command names to callables
taking the remaining arguments and returning an exit code.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, cast

from piginot import __version__
from piginot.contracts.canonical import canonical_json_bytes
from piginot.store import ObjectStore, StoreError

CAMPAIGN_COMMANDS = (
    "family compile",
    "pilot",
    "campaign",
    "simulate",
    "train",
    "verify",
    "holdout-reveal",
    "bundle build",
)
PLUGIN_GROUP = "piginot.campaigns"
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2
EXIT_SCIENTIFIC_REFUSED = 3

CampaignCommand = Callable[[list[str]], int]


class PluginError(LookupError):
    """The campaign plugin or one of its commands cannot be resolved."""


def _program(args: argparse.Namespace) -> str:
    return str(getattr(args, "program", "piginot"))


def resolve_campaign(name: str, search_path: Path | None = None) -> Mapping[str, Any]:
    """The command table of a campaign plugin."""

    if search_path is not None and str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
    target: str | None = None
    if ":" in name:
        target = name
    else:
        for entry in entry_points(group=PLUGIN_GROUP):
            if entry.name == name:
                return cast(Mapping[str, Any], entry.load())
        target = f"campaigns.{name}.cli:COMMANDS"
    module_name, _, attribute = target.partition(":")
    try:
        module = importlib.import_module(module_name)
        table = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise PluginError(f"campaign plugin {name!r} is not importable: {exc}") from exc
    if not isinstance(table, Mapping):
        raise PluginError(f"campaign plugin {name!r} is not a command table")
    return table


def _emit(document: object, output: Path | None) -> None:
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    if output is None:
        print(text)
    else:
        output.write_text(text + "\n", encoding="utf-8")


def _bundle_refusal(exc: Exception) -> dict[str, object]:
    from piginot.bundle import REQUIRED_LIMITATION_PARAGRAPH, BundleIntegrityError

    codes = (
        [code.value for code in exc.reason_codes]
        if isinstance(exc, BundleIntegrityError)
        else ["BUNDLE_UNREADABLE"]
    )
    return {
        "verdict": "BUNDLE_REFUSED",
        "reason_codes": codes,
        "detail": str(exc),
        "claim_level": "NOT_VERIFIED",
        "limitations": REQUIRED_LIMITATION_PARAGRAPH,
    }


def parse_request(document: object) -> tuple[dict[str, object], list[object] | None]:
    """Inputs as canonical quantities; anything malformed is kept for refusal."""

    from piginot.contracts import Quantity, Unit

    if not isinstance(document, dict):
        return {}, None
    raw_inputs = document.get("inputs")
    inputs: dict[str, object] = {}
    if isinstance(raw_inputs, dict):
        for name, value in raw_inputs.items():
            try:
                quantity = Quantity(value["value"], Unit(value["unit"]))
                inputs[name] = quantity.to_canonical()
            except (KeyError, TypeError, ValueError):
                inputs[name] = value
    queries = document.get("queries")
    if queries is None or isinstance(queries, list):
        return inputs, queries
    return inputs, [queries]


def _predict_option_error(args: argparse.Namespace) -> str | None:
    """Reject option combinations that would otherwise be silently ignored."""

    if args.mode != "certified" and (args.escalate or args.tolerance is not None):
        return "--escalate and --tolerance apply to --mode certified only"
    if args.tolerance is not None:
        if not args.escalate:
            return "--tolerance needs --escalate"
        if not math.isfinite(args.tolerance) or args.tolerance <= 0.0:
            return "--tolerance must be a finite positive number"
    return None


def _attach_trust_report(
    document: dict[str, Any], bundle: Any, *, mode: str
) -> dict[str, Any]:
    """Attach a versioned ``CADTrustReport`` without altering serving semantics."""
    from piginot.certify.trust_report import (
        DiscretizationErrorRecord,
        GeometryErrorRecord,
        QoITrustResult,
        SurrogateErrorRecord,
        build_cad_trust_report,
        reduce_qoi_trust_status,
    )

    in_env = document.get("verdict") == "IN_DOMAIN"
    cert = (
        document.get("certificate")
        if isinstance(document.get("certificate"), dict)
        else None
    )
    solve_failed = document.get("serving_path") == "SOLVE_FAILED"
    surr = SurrogateErrorRecord(
        verdict=str(cert.get("verdict", "UNAVAILABLE")) if cert else "UNAVAILABLE",
        eta_energy=cert.get("eta_energy") if cert else None,
        dwr_estimate=None,
        newton_qoi_change=None,
        dirichlet_deviation=float(cert.get("dirichlet_deviation", 0.0))
        if cert
        else 0.0,
        guards=dict(cert.get("guards", {})) if cert else {},
    )
    disc = DiscretizationErrorRecord(
        reference_status="MEASURED_CHANGES_ONLY" if mode != "fast" else "NOT_EVALUATED",
        element_order=2,
        dof_count=int(cert.get("dof_count", 0)) if cert else 0,
    )
    geom = GeometryErrorRecord(
        cad_source_hash=None,
        declared_curve_tolerance_mm=None,
        achieved_max_chord_error_mm=None,
        remesh_status="FIXED_BUNDLE_MESH",
    )
    status, reason = reduce_qoi_trust_status(
        in_parameter_envelope=in_env,
        in_geometry_envelope=in_env,
        solve_failed=solve_failed,
        surrogate=surr,
        discretization=disc,
        singular_or_reentrant=False,
        indicators_qualified=False,
    )
    qoi_res = QoITrustResult(
        qoi_name="bundle_field_summary",
        qoi_kind="regional_average_stress",
        unit="MPa",
        predicted_value=None,
        corrected_value=None,
        status=status,
        reason=reason,
        geometry=geom,
        discretization=disc,
        surrogate=surr,
    )
    claim = str(document.get("claim_level", "NOT_VERIFIED"))
    report = build_cad_trust_report(
        problem_hash=str(getattr(bundle.manifest, "manifest_sha256", "unknown")),
        mesh_id=str(getattr(bundle.reference_mesh, "mesh_id", "unknown")),
        cad_source_hash=None,
        bundle_digest=str(getattr(bundle.manifest, "manifest_sha256", "")),
        claim_level=claim,
        qoi_results=(qoi_res,),
    )
    return {
        **document,
        "trust_report": report.document(),
        "trust_report_sha256": report.report_hash,
    }


def _predict(args: argparse.Namespace) -> int:
    from piginot.bundle import BundleIntegrityError, load_bundle, predict
    from piginot.cad.parametric_adapter import (
        PARAMETRIC_QUALIFICATION_SCHEMA,
        is_parametric_bundle_dir,
        load_parametric_bundle,
        predict_parametric_cad,
    )

    if getattr(args, "cad_params", None) is not None or is_parametric_bundle_dir(
        args.bundle
    ):
        problem = _predict_option_error(args)
        if problem is not None:
            print(f"{_program(args)}: {problem}", file=sys.stderr)
            return EXIT_ERROR
        if getattr(args, "cad_params", None) is None:
            print(
                f"{_program(args)} predict: parametric bundle requires --cad-params",
                file=sys.stderr,
            )
            return EXIT_ERROR
        try:
            p_bundle = load_parametric_bundle(args.bundle)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"{_program(args)} predict: {exc}", file=sys.stderr)
            return EXIT_ERROR
        if getattr(args, "require_qualified", False):
            from piginot.contracts.state_machine import ClaimLevel as ContractClaimLevel

            try:
                p_min_claim = ContractClaimLevel(
                    str(
                        getattr(
                            args,
                            "min_claim",
                            ContractClaimLevel.DEVELOPMENT_VERIFIED.value,
                        )
                    )
                )
                granted = ContractClaimLevel(p_bundle.claim_level)
                q_path = getattr(args, "qualification_manifest", None)
                if q_path is None:
                    cand = Path(args.bundle) / "parametric-qualification.json"
                    if cand.is_file():
                        q_path = cand
                if q_path is not None:
                    q_doc = json.loads(Path(q_path).read_text(encoding="utf-8"))
                    if (
                        not isinstance(q_doc, dict)
                        or q_doc.get("schema") != PARAMETRIC_QUALIFICATION_SCHEMA
                    ):
                        raise ValueError(
                            f"expected {PARAMETRIC_QUALIFICATION_SCHEMA!r} manifest"
                        )
                    if q_doc.get("bundle_hash") != p_bundle.bundle_hash:
                        raise ValueError(
                            "parametric qualification manifest bundle_hash mismatch"
                        )
                    if q_doc.get("passed") is not True:
                        raise ValueError(
                            "parametric qualification manifest did not pass"
                        )
                    granted = ContractClaimLevel(
                        str(q_doc.get("claim_level", "NOT_VERIFIED"))
                    )
                if not granted.satisfies(p_min_claim):
                    raise ValueError(
                        f"parametric bundle claim_level {granted.value!r} does not satisfy "
                        f"required minimum {p_min_claim.value!r}"
                    )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                print(f"{_program(args)} predict: {exc}", file=sys.stderr)
                return EXIT_REFUSED
        try:
            raw_params = json.loads(args.cad_params.read_text(encoding="utf-8"))
            if not isinstance(raw_params, dict):
                raise ValueError("--cad-params JSON must be a parameter mapping")
            report = predict_parametric_cad(
                p_bundle,
                parameters={str(k): float(v) for k, v in raw_params.items()},
                mode=args.mode,
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"{_program(args)} predict: {exc}", file=sys.stderr)
            return EXIT_ERROR
        _emit({**report.document(), "report_hash": report.report_hash}, args.output)
        return EXIT_OK if report.accepted else EXIT_REFUSED

    if args.request is None:
        print(f"{_program(args)} predict: --request is required", file=sys.stderr)
        return EXIT_ERROR

    problem = _predict_option_error(args)
    if problem is not None:
        print(f"{_program(args)}: {problem}", file=sys.stderr)
        return EXIT_ERROR
    if getattr(args, "require_qualified", False):
        from piginot.qualify import DeploymentGateError, enforce_deployment_gate
        from piginot.verification.status import ClaimLevel

        try:
            min_claim = ClaimLevel(
                str(getattr(args, "min_claim", ClaimLevel.DEVELOPMENT_VERIFIED.value))
            )
            enforce_deployment_gate(
                args.bundle,
                getattr(args, "qualification_manifest", None),
                min_claim_level=min_claim,
            )
        except (DeploymentGateError, ValueError) as exc:
            print(f"{_program(args)} predict: {exc}", file=sys.stderr)
            return EXIT_REFUSED
    try:
        bundle = load_bundle(args.bundle)
    except (BundleIntegrityError, OSError) as exc:
        _emit(_bundle_refusal(exc), args.output)
        return EXIT_ERROR
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{_program(args)}: cannot read request: {exc}", file=sys.stderr)
        return EXIT_ERROR
    inputs, queries = parse_request(request)
    calibration = None
    if args.bounds or args.calibration is not None:
        from piginot.bounds import (
            CalibrationError,
            CalibrationRecord,
            ConfirmedCalibration,
            H1Confirmation,
            calibration_path,
            confirmation_path,
        )

        try:
            record = CalibrationRecord.load(
                args.calibration or calibration_path(args.bundle),
                bundle=Path(args.bundle).name,
                manifest_sha256=bundle.manifest.manifest_sha256,
            )
            h1_results = confirmation_path(args.bundle)
            calibration = (
                ConfirmedCalibration(record, H1Confirmation.load(h1_results, record))
                if args.calibration is None and h1_results.is_file()
                else record
            )
        except CalibrationError as exc:
            print(f"{_program(args)}: {exc}", file=sys.stderr)
            return EXIT_ERROR
    cad_calibration = None
    if getattr(args, "cad_calibration", None) is not None:
        from piginot.bounds import CadCalibrationRecord

        try:
            cad_calibration = CadCalibrationRecord.load(
                args.cad_calibration,
                expected_manifest_sha256=bundle.manifest.manifest_sha256,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"{_program(args)}: {exc}", file=sys.stderr)
            return EXIT_ERROR
    if args.mode == "fast":
        use_serve = (
            calibration is not None
            or cad_calibration is not None
            or bool(getattr(args, "tier0", False))
            or bool(getattr(args, "receipt", False))
        )
        if use_serve:
            from piginot.serving import serve

            served = serve(
                bundle,
                inputs,
                cast(Any, queries),
                confidence=not args.no_confidence,
                calibration=calibration,
                cad_calibration=cad_calibration,
                tier0_check=bool(getattr(args, "tier0", False)),
                emit_receipt=bool(getattr(args, "receipt", False)),
            )
            doc = served.document()
            if getattr(args, "trust_report", False):
                doc = _attach_trust_report(doc, bundle, mode="fast")
            _emit(doc, args.output)
            return EXIT_OK if served.prediction.in_domain else EXIT_REFUSED
        prediction = predict(
            bundle,
            inputs,
            queries,
            confidence=not args.no_confidence,
        )
        doc = prediction.document()
        if getattr(args, "trust_report", False):
            doc = _attach_trust_report(doc, bundle, mode="fast")
        _emit(doc, args.output)
        return EXIT_OK if prediction.in_domain else EXIT_REFUSED
    return _serve_verified(args, bundle, inputs, queries, calibration, cad_calibration)


def _load_trust_sidecar(
    bundle_path: Path, explicit: Path | None
) -> dict[str, Any] | None:
    path = (
        explicit
        if explicit is not None
        else bundle_path.parent / "trust" / f"{bundle_path.name}.json"
    )
    if not path.is_file():
        if explicit is not None:
            raise OSError(f"trust sidecar not found: {path}")
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("trust sidecar must be a JSON object")
    return loaded


def _serve_verified(
    args: argparse.Namespace,
    bundle: Any,
    inputs: Any,
    queries: Any,
    calibration: Any = None,
    cad_calibration: Any = None,
) -> int:
    """``--mode certified|exact`` (S22): certificate, correction, FEM solve, live DWR."""

    from piginot.bundle.problem import PortableProblem
    from piginot.serving import MissingExtraError, ServingPath, load_recipe, serve

    try:
        is_portable = isinstance(
            getattr(bundle.model, "problem", None), PortableProblem
        )
        recipe = (
            None
            if (is_portable and args.recipe is None)
            else load_recipe(bundle, args.bundle, args.recipe)
        )
        sidecar = (
            _load_trust_sidecar(args.bundle, getattr(args, "trust_sidecar", None))
            if getattr(args, "trust_report", False)
            else None
        )
        tta_steps = int(getattr(args, "tta_steps", 0) or 0)
        tta_cfg = None
        if tta_steps > 0:
            from piginot.serving.test_time_lora import TestTimeLoraConfig

            tta_cfg = TestTimeLoraConfig(steps=tta_steps)
        served = serve(
            bundle,
            inputs,
            queries,
            mode=args.mode,
            recipe=recipe,
            corrected=args.corrected,
            escalate=args.escalate,
            tolerance=args.tolerance,
            backend=getattr(args, "backend", None),
            calibration=calibration,
            cad_calibration=cad_calibration,
            trust_report=bool(getattr(args, "trust_report", False)),
            trust_sidecar=sidecar,
            tier0_check=bool(getattr(args, "tier0", False)),
            spectral_repair=bool(getattr(args, "spectral_repair", False)),
            tta_config=tta_cfg,
            emit_receipt=bool(getattr(args, "receipt", False)),
        )
    except MissingExtraError as exc:
        print(f"{_program(args)}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (LookupError, OSError, ValueError) as exc:
        print(f"{_program(args)}: cannot serve {args.mode}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    doc = served.document()
    if getattr(args, "trust_report", False) and "trust_report" not in doc:
        doc = _attach_trust_report(doc, bundle, mode=args.mode)
    _emit(doc, args.output)
    if served.path is ServingPath.REFUSED:
        return EXIT_REFUSED
    return EXIT_ERROR if served.path is ServingPath.SOLVE_FAILED else EXIT_OK


def _calibrate(args: argparse.Namespace) -> int:
    """Build a bundle-bound per-CAD split-conformal calibration sidecar."""
    from piginot.bounds import build_cad_calibration
    from piginot.bundle import BundleIntegrityError, load_bundle
    from piginot.bundle.problem import PortableProblem

    try:
        bundle = load_bundle(args.bundle)
    except (BundleIntegrityError, OSError) as exc:
        _emit(_bundle_refusal(exc), None)
        return EXIT_ERROR
    try:
        raw = json.loads(args.scores.read_text(encoding="utf-8"))
        qoi_scores = {
            str(name): (str(entry["unit"]), [float(v) for v in entry["scores"]])
            for name, entry in dict(raw.get("qoi_scores", {})).items()
        }
        func_res = {
            str(name): (
                str(entry["unit"]),
                [[float(x) for x in row] for row in entry["residuals"]],
            )
            for name, entry in dict(raw.get("functional_residuals", {})).items()
        }
        pbounds = {
            bound.name: (
                float(bound.lower_bound.value),
                float(bound.upper_bound.value),
            )
            for bound in bundle.envelope.primitive_bounds
        }
        problem = getattr(bundle.model, "problem", None)
        record = build_cad_calibration(
            bundle_name=bundle.manifest.name,
            bundle_manifest_sha256=bundle.manifest.manifest_sha256,
            problem_hash=(
                problem.problem_hash if isinstance(problem, PortableProblem) else None
            ),
            qoi_scores=qoi_scores,
            alpha=float(args.alpha),
            exchangeability_scope=str(
                raw.get(
                    "exchangeability_scope",
                    "in-envelope-exchangeable-cad-calibration",
                )
            ),
            functional_residuals=func_res or None,
            parameter_bounds=pbounds,
        )
        record.save(args.output)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"{_program(args)} calibrate: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    _emit(
        {
            **record.document(),
            "calibration_sha256": record.calibration_sha256,
            "output": str(args.output),
        },
        None,
    )
    return EXIT_OK


def _inspect(args: argparse.Namespace) -> int:
    from piginot.bundle import BundleIntegrityError, inspect, load_bundle

    try:
        bundle = load_bundle(args.bundle)
    except (BundleIntegrityError, OSError) as exc:
        _emit(_bundle_refusal(exc), args.output)
        return EXIT_ERROR
    _emit(inspect(bundle), args.output)
    return EXIT_OK


def _bundle_verify(args: argparse.Namespace) -> int:
    from piginot.bundle import BundleIntegrityError, load_bundle

    try:
        bundle = load_bundle(args.bundle)
    except (BundleIntegrityError, OSError) as exc:
        _emit(_bundle_refusal(exc), None)
        return EXIT_ERROR
    _emit(
        {
            "verdict": "VERIFIED",
            "manifest_sha256": bundle.manifest.manifest_sha256,
            "claim_level": bundle.claim_level.value,
            "limitations": bundle.limitations["paragraph"],
        },
        None,
    )
    return EXIT_OK


def _store_prune(args: argparse.Namespace) -> int:
    try:
        result = ObjectStore.open(args.root).prune(dry_run=args.dry_run)
    except (OSError, StoreError) as exc:
        print(f"{_program(args)}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    report = {
        "dry_run": result.dry_run,
        "object_count": result.object_count,
        "pruned_bytes": result.pruned_bytes,
        "pruned_hashes": list(result.pruned_hashes),
    }
    print(canonical_json_bytes(report).decode("utf-8"))
    return EXIT_OK


def _compile(args: argparse.Namespace) -> int:
    try:
        from piginot.compiler.cli import compile_command
    except ImportError as error:
        print(
            f"{_program(args)} compile: compiler dependencies unavailable: {error}",
            file=sys.stderr,
        )
        return 2
    return compile_command(
        args.spec,
        args.config,
        args.output,
        plan_only=args.plan_only,
        resume=args.resume,
        prog=_program(args),
        qualify_policy=getattr(args, "qualify_policy", None),
        qualify_dev_samples=int(getattr(args, "qualify_dev_samples", 9)),
        qualify_holdout_samples=int(getattr(args, "qualify_holdout_samples", 0)),
        qualify_backend=str(getattr(args, "qualify_backend", "scikit-fem")),
        qualify_seed=int(getattr(args, "qualify_seed", 0)),
    )


def _cad(args: argparse.Namespace) -> int:
    from piginot.cad.cli import run

    return run(args)


def _active_select(args: argparse.Namespace) -> int:
    """Pure-stdlib deterministic TRAIN-only active-FEM batch selection (P3)."""
    from piginot.active import (
        AcquisitionCandidate,
        AcquisitionPlan,
        AcquisitionWeights,
        select_acquisition_batch,
    )

    try:
        plan_doc = json.loads(args.plan.read_text(encoding="utf-8"))
        pool_doc = json.loads(args.pool.read_text(encoding="utf-8"))
        w_doc = plan_doc.get("weights", {})
        plan = AcquisitionPlan(
            plan_id=str(plan_doc["plan_id"]),
            arm_kind=plan_doc["arm_kind"],
            budget_solves=int(plan_doc["budget_solves"]),
            batch_size=int(plan_doc["batch_size"]),
            seed=int(plan_doc.get("seed", 0)),
            weights=AcquisitionWeights(
                eta_weight=float(w_doc.get("eta_weight", 0.5)),
                ood_weight=float(w_doc.get("ood_weight", 0.2)),
                ensemble_weight=float(w_doc.get("ensemble_weight", 0.1)),
                diversity_weight=float(w_doc.get("diversity_weight", 0.2)),
            ),
            supervises_training=bool(plan_doc.get("supervises_training", True)),
            training_mode=plan_doc.get(
                "training_mode", "multi_fidelity_label_efficient"
            ),
        )
        candidates = [
            AcquisitionCandidate(
                case_id=str(row["case_id"]),
                partition=str(row["partition"]),
                feature_vector=tuple(float(v) for v in row["feature_vector"]),
                eta_energy=float(row["eta_energy"]),
                ood_distance=float(row.get("ood_distance", 0.0)),
                ensemble_disagreement=float(row.get("ensemble_disagreement", 0.0)),
                parameters=dict(row.get("parameters", {})),
            )
            for row in pool_doc
        ]
        acquired_ids: list[str] = []
        acquired_feats: list[tuple[float, ...]] = []
        if args.acquired is not None:
            acq_doc = json.loads(args.acquired.read_text(encoding="utf-8"))
            if isinstance(acq_doc, list):
                acquired_ids = [str(v) for v in acq_doc]
            elif isinstance(acq_doc, dict):
                acquired_ids = [str(v) for v in acq_doc.get("case_ids", [])]
                acquired_feats = [
                    tuple(float(x) for x in f) for f in acq_doc.get("features", [])
                ]
        evidence = select_acquisition_batch(
            candidates,
            already_acquired_ids=acquired_ids,
            plan=plan,
            round_index=int(args.round),
            already_acquired_features=acquired_feats,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"{_program(args)} active select: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    out_doc = {
        **evidence.document(),
        "selection_hash": evidence.selection_hash,
        "training_mode": plan.training_mode,
    }
    _emit(out_doc, args.output)
    return EXIT_OK


def _optimize(args: argparse.Namespace) -> int:
    """Run FEM-guarded differentiable design optimization on a verified bundle (P5 / M4)."""
    from piginot.bundle import BundleIntegrityError, load_bundle
    from piginot.cad.parametric_adapter import (
        is_parametric_bundle_dir,
        load_parametric_bundle,
    )
    from piginot.optimization import (
        BundleDesignBridge,
        DesignOptimizationStatus,
        ParametricBundleDesignBridge,
        parse_bundle_design_document,
        parse_parametric_design_document,
    )
    from piginot.serving import MissingExtraError, load_recipe

    if is_parametric_bundle_dir(args.bundle):
        try:
            p_bundle = load_parametric_bundle(args.bundle)
            prob_doc = json.loads(args.problem.read_text(encoding="utf-8"))
            p_spec = parse_parametric_design_document(prob_doc, bundle=p_bundle)
            p_bridge = ParametricBundleDesignBridge(p_bundle, p_spec)
            report = p_bridge.run()
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"{_program(args)} optimize: {exc}", file=sys.stderr)
            return EXIT_ERROR
        out_doc = {**report.document(), "report_hash": report.report_hash}
        _emit(out_doc, args.output)
        if report.status is DesignOptimizationStatus.REFUSED_UNSUPPORTED_GEOMETRY:
            return EXIT_REFUSED
        if report.status is DesignOptimizationStatus.FAILED_NO_FEASIBLE_FEM_DESIGN:
            return EXIT_ERROR
        return EXIT_OK

    try:
        bundle = load_bundle(args.bundle)
    except (BundleIntegrityError, OSError) as exc:
        _emit(_bundle_refusal(exc), args.output)
        return EXIT_ERROR
    try:
        recipe = load_recipe(bundle, args.bundle, args.recipe)
        prob_doc = json.loads(args.problem.read_text(encoding="utf-8"))
        spec = parse_bundle_design_document(prob_doc, bundle=bundle)
        bridge = BundleDesignBridge(bundle, recipe, spec)
        report = bridge.run(geometry_supported=not bool(args.refuse_unsupported))
    except MissingExtraError as exc:
        print(f"{_program(args)} optimize: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"{_program(args)} optimize: {exc}", file=sys.stderr)
        return EXIT_ERROR
    out_doc = {**report.document(), "report_hash": report.report_hash}
    _emit(out_doc, args.output)
    if report.status is DesignOptimizationStatus.REFUSED_UNSUPPORTED_GEOMETRY:
        return EXIT_REFUSED
    if report.status is DesignOptimizationStatus.FAILED_NO_FEASIBLE_FEM_DESIGN:
        return EXIT_ERROR
    return EXIT_OK


def _stateful_rollout(args: argparse.Namespace) -> int:
    """Run guarded viscoelastic trajectory rollout from a saved P6 artifact."""
    from piginot.contracts.expressions import DerivedExpression
    from piginot.contracts.stateful import LoadSegment, TimeGrid
    from piginot.stateful import load_viscoelastic_artifact, run_viscoelastic_rollout

    try:
        artifact = load_viscoelastic_artifact(args.artifact)
        lp_doc = json.loads(args.load_path.read_text(encoding="utf-8"))
        grid_doc = lp_doc["time_grid"]
        grid = TimeGrid(
            t0=float(grid_doc["t0"]),
            t1=float(grid_doc["t1"]),
            dt_initial=float(grid_doc["dt_initial"]),
            dt_min=float(grid_doc["dt_min"]),
            dt_max=float(grid_doc["dt_max"]),
        )
        segments = tuple(
            LoadSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                kind=seg["kind"],
                boundary_values={
                    str(k): DerivedExpression.parse(str(v))
                    for k, v in seg["boundary_values"].items()
                },
            )
            for seg in lp_doc["segments"]
        )
        params = {
            str(k): float(v) for k, v in dict(lp_doc.get("parameters", {})).items()
        }
        result = run_viscoelastic_rollout(
            artifact,
            time_grid=grid,
            segments=segments,
            parameters=params,
            load_key=str(lp_doc.get("load_key", "ux")),
            residual_tolerance=float(args.residual_tolerance),
            resume_checkpoint=args.resume,
            save_checkpoint_to=args.checkpoint_out,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"{_program(args)} stateful rollout: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    _emit(result, args.output)
    return EXIT_OK if result.get("accepted") else EXIT_REFUSED


def _stateful_verify_tangent(args: argparse.Namespace) -> int:
    """Verify J2 elastoplastic algorithmic consistent tangent against central FD."""
    import torch

    from piginot.stateful import (
        J2IsotropicPlasticity,
        J2PlasticityConfig,
        verify_algorithmic_tangent,
        verify_plane_stress_tangent,
    )

    try:
        raw = json.loads(args.material.read_text(encoding="utf-8"))
        cfg = J2PlasticityConfig.from_document(raw)
        model = J2IsotropicPlasticity(cfg)
        yield_strain = cfg.yield_stress_mpa / cfg.youngs_modulus_mpa
        mode = str(args.mode)
        checks: dict[str, dict[str, Any]] = {}
        eye3 = torch.eye(3, dtype=torch.float64)
        eye2 = torch.eye(2, dtype=torch.float64)

        regimes = (
            ("elastic", 0.5 * yield_strain, 0.25 * yield_strain),
            ("active_plastic", 2.5 * yield_strain, 1.0 * yield_strain),
            ("unloading", 1.5 * yield_strain, 0.4 * yield_strain),
        )
        state_3d = model.initial_state((), dtype=torch.float64)
        state_2d = model.initial_state((), dtype=torch.float64)
        f3_prev = eye3.clone()
        f2_prev = eye2.clone()
        all_passed = True

        for label, exx, exy in regimes:
            if mode == "plane-stress":
                f2 = eye2.clone()
                f2[0, 0] = 1.0 + exx
                f2[1, 1] = 1.0 - 0.3 * exx
                f2[0, 1] = exy
                f2[1, 0] = exy
                res = verify_plane_stress_tangent(
                    model, f2_prev, f2, state_2d, dt=1.0, rel_tol=float(args.rel_tol)
                )
                upd = model.plane_stress_update(f2_prev, f2, state_2d, dt=1.0)
                state_2d = dict(upd.update_3d.state_new)
                f2_prev = f2
            else:
                f3 = eye3.clone()
                f3[0, 0] = 1.0 + exx
                f3[1, 1] = 1.0 - (0.3 * exx if mode == "3d" else 0.2 * exx)
                f3[2, 2] = 1.0 if mode == "plane-strain" else 1.0 - 0.1 * exx
                f3[0, 1] = exy
                f3[1, 0] = exy
                res = verify_algorithmic_tangent(
                    model, f3_prev, f3, state_3d, dt=1.0, rel_tol=float(args.rel_tol)
                )
                upd3 = model.update(f3_prev, f3, state_3d, dt=1.0)
                state_3d = dict(upd3.state_new)
                f3_prev = f3
            all_passed = all_passed and res.passed
            checks[label] = {
                "passed": res.passed,
                "relative_error": res.relative_error,
                "max_abs_error": res.max_abs_error,
                "tolerance": res.tolerance,
            }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"{_program(args)} stateful verify-tangent: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    doc = {
        "model_id": cfg.document()["model_id"],
        "config_hash": cfg.config_hash,
        "mode": mode,
        "passed": all_passed,
        "regimes": checks,
    }
    _emit(doc, args.output)
    return EXIT_OK if all_passed else EXIT_SCIENTIFIC_REFUSED


def _ui(args: argparse.Namespace) -> int:
    """Launch the Verification & Trust Inspector Streamlit app (`formamech[ui]`)."""
    import importlib.util
    import subprocess

    if importlib.util.find_spec("streamlit") is None:
        print(
            f"{_program(args)} ui: requires optional extra {_program(args)}[ui] "
            "(install with `pip install 'formamech[ui]'`)",
            file=sys.stderr,
        )
        return EXIT_ERROR
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(int(args.port)),
    ]
    if args.headless:
        cmd.extend(["--server.headless", "true"])
    extra_args: list[str] = []
    if args.bundle is not None:
        extra_args.extend(["--bundle", str(args.bundle)])
    if getattr(args, "workspace", None) is not None:
        extra_args.extend(["--workspace", str(args.workspace)])
    if extra_args:
        cmd.append("--")
        cmd.extend(extra_args)
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _mcp(args: argparse.Namespace) -> int:
    """Run or inspect the FormaMech Model Context Protocol (MCP) server."""
    from piginot.mcp_server import FormaMechMcpServer

    try:
        server = FormaMechMcpServer(
            args.workspace,
            max_dof_limit=int(args.max_dofs),
            require_qualified=bool(getattr(args, "require_qualified", False)),
        )
    except ValueError as exc:
        print(f"{_program(args)} mcp: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.mcp_command == "list-tools":
        _emit({"tools": server.list_tools()}, args.output)
        return EXIT_OK
    server.serve_stdio()
    return EXIT_OK


def _qualify(args: argparse.Namespace) -> int:
    """Run, verify, or diff continuous numerical model qualification (Feature 5)."""
    from piginot.bundle import BundleIntegrityError, load_bundle
    from piginot.qualify import (
        QualificationIntegrityError,
        QualificationManifest,
        QualificationPolicy,
        check_deployment_gate,
        compare_qualifications,
        default_development_policy,
        default_holdout_policy,
        load_cases_file,
        qualify_bundle,
        sample_qualification_cases,
    )
    from piginot.verification.status import ClaimLevel

    cmd = args.qualify_command
    if cmd == "verify":
        try:
            policy = (
                QualificationPolicy.load(args.policy)
                if args.policy is not None
                else None
            )
            min_claim = ClaimLevel(str(args.min_claim))
            result = check_deployment_gate(
                args.bundle,
                args.manifest,
                policy=policy,
                min_claim_level=min_claim,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"{_program(args)} qualify verify: {exc}", file=sys.stderr)
            return EXIT_REFUSED
        _emit(result.document(), args.output)
        return EXIT_OK if result.allowed else EXIT_REFUSED

    if cmd == "diff":
        try:
            old_m = QualificationManifest.load(args.old_manifest)
            new_m = QualificationManifest.load(args.new_manifest)
            diff = compare_qualifications(
                old_m,
                new_m,
                relative_regression_tol=float(args.regression_tol),
            )
        except (
            OSError,
            json.JSONDecodeError,
            QualificationIntegrityError,
            ValueError,
        ) as exc:
            print(f"{_program(args)} qualify diff: {exc}", file=sys.stderr)
            return EXIT_REFUSED
        _emit(diff.document(), args.output)
        return EXIT_SCIENTIFIC_REFUSED if diff.regressed else EXIT_OK

    # cmd == "run"
    try:
        if args.policy is not None:
            policy = QualificationPolicy.load(args.policy)
        elif args.preset == "holdout":
            policy = default_holdout_policy()
        else:
            policy = default_development_policy()
        bundle = load_bundle(args.bundle)

        train_fps: set[str] = set()
        if args.train_cases is not None:
            for tc in load_cases_file(args.train_cases, default_partition="DEV"):
                train_fps.add(tc.fingerprint)

        if args.dev_cases is not None:
            dev_cases = load_cases_file(args.dev_cases, default_partition="DEV")
            holdout_cases = (
                load_cases_file(args.holdout_cases, default_partition="HOLDOUT")
                if args.holdout_cases is not None
                else ()
            )
        else:
            ho_count = int(args.sample_holdout)
            if (
                ho_count == 0
                and args.holdout_cases is None
                and policy.target_claim_level is ClaimLevel.HOLDOUT_VERIFIED
            ):
                ho_count = max(int(args.sample_dev), 5)
            dev_cases, sampled_ho = sample_qualification_cases(
                bundle,
                dev_count=int(args.sample_dev),
                holdout_count=0 if args.holdout_cases is not None else ho_count,
                seed=int(args.seed),
                exclude_fingerprints=frozenset(train_fps),
            )
            holdout_cases = (
                load_cases_file(args.holdout_cases, default_partition="HOLDOUT")
                if args.holdout_cases is not None
                else sampled_ho
            )

        run_res = qualify_bundle(
            args.bundle,
            policy,
            dev_cases=dev_cases,
            holdout_cases=holdout_cases,
            train_fingerprints=frozenset(train_fps),
            recipe_path=args.recipe,
            backend=args.backend,
            output_dir=args.output,
            emit_qualified_bundle=args.emit_qualified_bundle,
        )
    except (
        BundleIntegrityError,
        OSError,
        json.JSONDecodeError,
        LookupError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"{_program(args)} qualify run: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    summary_doc = {
        **run_res.manifest.full_document(),
        "output_dir": None if args.output is None else str(args.output),
        "promoted_bundle_path": (
            None
            if run_res.promoted_bundle_path is None
            else str(run_res.promoted_bundle_path)
        ),
    }
    _emit(summary_doc, None)
    return EXIT_OK if run_res.qualified else EXIT_SCIENTIFIC_REFUSED


def _export(args: argparse.Namespace) -> int:
    """Export a verified FormaMech bundle to ONNX or FMI 3.0 Co-Simulation FMU."""
    from piginot.bundle import BundleIntegrityError
    from piginot.export import export_bundle_to_fmu, export_bundle_to_onnx
    from piginot.serving.hybrid import MissingExtraError

    if getattr(args, "require_qualified", False):
        from piginot.qualify import DeploymentGateError, enforce_deployment_gate
        from piginot.verification.status import ClaimLevel

        try:
            min_claim = ClaimLevel(
                str(getattr(args, "min_claim", ClaimLevel.DEVELOPMENT_VERIFIED.value))
            )
            enforce_deployment_gate(
                args.bundle,
                getattr(args, "qualification_manifest", None),
                min_claim_level=min_claim,
            )
        except (DeploymentGateError, ValueError) as exc:
            print(f"{_program(args)} export: {exc}", file=sys.stderr)
            return EXIT_REFUSED

    try:
        if args.export_command == "onnx":
            onnx_res = export_bundle_to_onnx(
                args.bundle,
                args.output,
                opset_version=int(args.opset),
                verify_parity=not bool(args.no_verify),
            )
            _emit(onnx_res.document(), None)
            return EXIT_OK
        fmu_res = export_bundle_to_fmu(
            args.bundle,
            args.output,
            model_name=args.model_name,
            include_onnx=bool(args.include_onnx),
        )
        _emit(fmu_res.document(), None)
        return EXIT_OK
    except MissingExtraError as exc:
        print(f"{_program(args)} export: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (BundleIntegrityError, OSError, ValueError) as exc:
        print(f"{_program(args)} export: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _bench(args: argparse.Namespace) -> int:
    """Run reproducible 3D scaling, foundation transfer, OOD & parity benchmarks (V4-2)."""
    from piginot.benchmarks import run_benchmark_suite_3d

    try:
        rep = run_benchmark_suite_3d(
            args.workdir,
            profile=args.profile,
            foundation_dir=args.foundation_dir,
        )
        doc = {**rep.document(), "report_hash": rep.report_hash}
        _emit(doc, args.output)
        if args.markdown is not None:
            Path(args.markdown).write_text(rep.to_markdown(), encoding="utf-8")
        return EXIT_OK
    except Exception as exc:
        print(f"{_program(args)} bench: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _serve_api(args: argparse.Namespace) -> int:
    """Launch the production verified HTTP/JSON + Prometheus serving API (V4-1)."""
    from piginot.harness import HarnessBudget
    from piginot.serving.api_server import FormaMechApiApplication, FormaMechHttpServer

    try:
        app = FormaMechApiApplication(
            workspace_root=args.workspace,
            bundle_path=args.bundle,
            ledger_path=args.ledger,
            budget=HarnessBudget(
                max_fem_solves=int(args.max_fem_solves),
                max_dofs=int(args.max_dofs),
            ),
        )
        srv = FormaMechHttpServer(app, host=str(args.host), port=int(args.port))
        print(
            json.dumps(
                {
                    "status": "listening",
                    "url": srv.base_url,
                    "workspace": str(app.workspace_root),
                    "ledger": str(app.ledger.path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        srv.serve_forever()
        return EXIT_OK
    except KeyboardInterrupt:
        return EXIT_OK
    except Exception as exc:
        print(f"{_program(args)} serve-api: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _agent(args: argparse.Namespace) -> int:
    """AI-ready Universal Agent Harness CLI commands (V4-0)."""
    from piginot.harness import (
        FormaMechAgentHarness,
        HarnessBudget,
        build_default_registry,
        export_tool_schemas,
    )

    cmd = str(args.agent_command)
    try:
        if cmd == "capabilities":
            reg = build_default_registry()
            items = reg.list_capabilities(
                domain=args.domain,
                cost_class=args.cost_class,
                search=args.search,
            )
            _emit(
                {
                    "count": len(items),
                    "domains": reg.domains_summary(),
                    "capabilities": items,
                },
                args.output,
            )
            return EXIT_OK
        if cmd == "describe":
            reg = build_default_registry()
            _emit(reg.describe(str(args.capability)), args.output)
            return EXIT_OK
        if cmd == "export-tools":
            reg = build_default_registry()
            tools = export_tool_schemas(
                reg,
                provider=args.provider,
                mode=args.mode,
                domain=args.domain,
            )
            _emit(
                {
                    "provider": args.provider,
                    "mode": args.mode,
                    "count": len(tools),
                    "tools": tools,
                },
                args.output,
            )
            return EXIT_OK

        ws = Path(args.workspace)
        if cmd == "invoke":
            raw_args = json.loads(args.args_json) if args.args_json else {}
            sfile = ws / ".formamech-agent" / "session.json"
            if sfile.is_file():
                harness = FormaMechAgentHarness.resume(ws)
            else:
                harness = FormaMechAgentHarness(
                    ws,
                    budget=HarnessBudget(
                        max_fem_solves=int(args.max_fem_solves),
                        max_dofs=int(args.max_dofs),
                    ),
                )
            env = harness.invoke(
                str(args.capability), raw_args, dry_run=bool(args.dry_run)
            )
            _emit(env.document(), args.output)
            return EXIT_OK if env.ok else EXIT_REFUSED
        if cmd == "session-status":
            harness = FormaMechAgentHarness.resume(ws)
            ok, errs = harness.verify_ledger()
            _emit(
                {
                    "session_id": harness.session_id,
                    "snapshot": harness.snapshot().document(),
                    "budget": harness.budget_status(),
                    "ledger_verified": ok,
                    "ledger_errors": list(errs),
                },
                args.output,
            )
            return EXIT_OK if ok else EXIT_REFUSED
        if cmd == "verify-ledger":
            harness = FormaMechAgentHarness.resume(ws)
            ok, errs = harness.verify_ledger()
            _emit(
                {
                    "session_id": harness.session_id,
                    "verified": ok,
                    "entry_count": len(harness.read_ledger()),
                    "errors": list(errs),
                },
                args.output,
            )
            return EXIT_OK if ok else EXIT_REFUSED
        return EXIT_ERROR
    except Exception as exc:
        print(f"{_program(args)} agent {cmd}: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _ingest_cae(args: argparse.Namespace) -> int:
    """Ingest external VTU/XDMF/CalculiX/Abaqus/Nastran CAE files & run outlier screening."""
    from piginot.ingest import (
        read_abaqus_inp_and_fields,
        read_calculix_inp_frd,
        read_nastran_bdf_and_fields,
        read_vtu_or_xdmf,
        screen_cae_archive,
    )

    try:
        samples = []
        src = Path(args.source)
        fmt = str(args.format)
        if src.is_dir():
            files = sorted(src.glob("*.vtu")) + sorted(src.glob("*.xdmf"))
            for fp in files:
                samples.append(read_vtu_or_xdmf(fp))
        elif fmt == "calculix-frd":
            samples.append(read_calculix_inp_frd(src, args.fields))
        elif fmt == "abaqus-inp":
            samples.append(read_abaqus_inp_and_fields(src, args.fields))
        elif fmt == "nastran-bdf":
            samples.append(read_nastran_bdf_and_fields(src, args.fields))
        else:
            samples.append(read_vtu_or_xdmf(src))
        report = screen_cae_archive(
            samples,
            zscore_threshold=float(args.zscore_threshold),
            output_dir=args.output,
        )
        _emit({**report.document(), "report_hash": report.report_hash}, None)
        return EXIT_OK if report.accepted_count > 0 else EXIT_REFUSED
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"{_program(args)} ingest: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _validate_experiment(args: argparse.Namespace) -> int:
    """Calibrate constitutive parameters & run ASME V&V 10 coupon holdout validation."""
    from piginot.validation import (
        CouponRole,
        ExperimentalCouponDataset,
        ExperimentalTestMode,
        InverseCalibrationConfig,
        calibrate_material_from_experiments,
    )

    try:
        raw = json.loads(Path(args.coupons).read_text(encoding="utf-8"))
        coupons = [
            ExperimentalCouponDataset(
                specimen_id=str(item["specimen_id"]),
                role=CouponRole(str(item["role"])),
                test_mode=ExperimentalTestMode(str(item["test_mode"])),
                stretches_or_strains=tuple(
                    float(v) for v in item["stretches_or_strains"]
                ),
                measured_stress_mpa=tuple(
                    float(v) for v in item["measured_stress_mpa"]
                ),
                measurement_std_mpa=float(item["measurement_std_mpa"]),
            )
            for item in raw["coupons"]
        ]
        cfg = InverseCalibrationConfig(
            model_family=args.model_family,
            u_num_mpa=float(args.u_num_mpa),
        )
        dossier = calibrate_material_from_experiments(coupons, config=cfg)
        doc = {**dossier.document(), "dossier_hash": dossier.dossier_hash}
        if args.output is not None:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_bytes(canonical_json_bytes(doc) + b"\n")
        _emit(doc, None)
        return EXIT_OK if dossier.passed else EXIT_SCIENTIFIC_REFUSED
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        print(f"{_program(args)} validate: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _fleet(args: argparse.Namespace) -> int:
    """Dispatch or generate distributed 3D FEM fleet worker batches (V4-5)."""
    from piginot.training.fleet import (
        FleetTaskSpec,
        KubernetesJobFleet,
        LocalMultiprocessFleet,
        SlurmScriptFleet,
        WorkerBackend,
    )

    cmd = str(args.fleet_command)
    try:
        raw = (
            json.loads(args.tasks_json)
            if getattr(args, "tasks_json", None)
            else [{"lx_mm": 18.0}, {"lx_mm": 22.0}]
        )
        tasks = [
            FleetTaskSpec(task_id=f"task-{i:03d}", payload=dict(item))
            for i, item in enumerate(raw)
        ]
        backend: WorkerBackend
        if cmd == "generate-slurm":
            backend = SlurmScriptFleet(partition=str(args.partition))
        elif cmd == "generate-k8s":
            backend = KubernetesJobFleet(image=str(args.image))
        else:
            backend = LocalMultiprocessFleet(
                max_workers=int(args.max_workers),
                use_threads=bool(args.use_threads),
            )
        rep = backend.submit_batch(str(args.job_id), tasks, workdir=args.workdir)
        _emit(rep.document(), args.output)
        return EXIT_OK if rep.failed_count == 0 else EXIT_REFUSED
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"{_program(args)} fleet {cmd}: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _continual(args: argparse.Namespace) -> int:
    """Inspect Continual Qualification Loop & SignedAuditLedger status (V4-5)."""
    from piginot.serving.audit_ledger import SignedAuditLedger
    from piginot.store.dataset_3d import ShardedMeshDataset3D

    try:
        ledger = SignedAuditLedger(args.ledger)
        chain_valid, chain_errors = ledger.verify_chain()
        ds = ShardedMeshDataset3D(args.dataset)
        doc = {
            "ledger_valid": chain_valid,
            "ledger_errors": list(chain_errors),
            "ledger_entry_count": len(ledger.read_entries()),
            "fallback_count": len(ledger.query_fallbacks()),
            "dataset": ds.summary(),
        }
        _emit(doc, args.output)
        return EXIT_OK if chain_valid else EXIT_REFUSED
    except (OSError, ValueError) as exc:
        print(f"{_program(args)} continual: {exc}", file=sys.stderr)
        return EXIT_ERROR


def _campaign(command: str, args: argparse.Namespace) -> int:
    try:
        table = resolve_campaign(args.campaign, args.campaign_path)
        handler = table.get(command)
        if handler is None:
            raise PluginError(f"campaign {args.campaign!r} has no {command!r} command")
    except PluginError as exc:
        print(f"{_program(args)}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    rest = list(args.arguments)
    if rest[:1] == ["--"]:
        rest = rest[1:]
    return int(handler(rest))


def _campaign_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    command: str,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name, help=f"campaign command: {command}")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--campaign-path", type=Path, default=None)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parser.set_defaults(campaign_command=command)
    return parser


def build_parser(*, prog: str = "piginot") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.set_defaults(program=prog)
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")

    compile_parser = subparsers.add_parser(
        "compile", help="plan and execute a declared campaign"
    )
    compile_parser.add_argument("--spec", required=True, type=Path)
    compile_parser.add_argument("--config", required=True, type=Path)
    compile_parser.add_argument("--output", required=True, type=Path)
    action = compile_parser.add_mutually_exclusive_group()
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--resume", action="store_true")
    compile_parser.add_argument(
        "--qualify-policy",
        type=Path,
        default=None,
        help="optional QualificationPolicy JSON to run automatically after export",
    )
    compile_parser.add_argument("--qualify-dev-samples", type=int, default=9)
    compile_parser.add_argument("--qualify-holdout-samples", type=int, default=0)
    compile_parser.add_argument("--qualify-backend", type=str, default="scikit-fem")
    compile_parser.add_argument("--qualify-seed", type=int, default=0)
    compile_parser.set_defaults(handler=_compile)

    cad = subparsers.add_parser("cad", help="import and inspect a fixed CAD geometry")
    cad_commands = cad.add_subparsers(dest="cad_command", required=True)
    cad_import = cad_commands.add_parser("import", help="mesh a STEP or DXF profile")
    cad_import.add_argument("source", type=Path)
    cad_import.add_argument("--unit", choices=("mm", "cm", "m"), required=True)
    sizing = cad_import.add_mutually_exclusive_group(required=True)
    sizing.add_argument("--mesh-size-mm", type=float)
    sizing.add_argument(
        "--mesh-policy", type=Path, help="explicit local sizing and curve-accuracy JSON"
    )
    cad_import.add_argument("--output", type=Path, required=True)
    cad_import.add_argument(
        "--hierarchy",
        action="store_true",
        help="also emit a verified M0/M1/M2 MeshHierarchy sidecar (hierarchy.json)",
    )
    cad_import.set_defaults(handler=_cad)
    cad_inspect = cad_commands.add_parser(
        "inspect", help="inspect verified geometry data"
    )
    cad_inspect.add_argument("project", type=Path)
    cad_inspect.set_defaults(handler=_cad)
    cad_prepare = cad_commands.add_parser(
        "prepare", help="prepare a fixed-CAD training campaign"
    )
    cad_prepare.add_argument("--project", type=Path, required=True)
    cad_prepare.add_argument("--settings", type=Path, required=True)
    cad_prepare.add_argument("--output", type=Path, required=True)
    cad_prepare.add_argument("--execution-profile", action="store_true")
    cad_prepare.set_defaults(handler=_cad)
    cad_hier = cad_commands.add_parser(
        "hierarchy", help="build a verified M0/M1/M2 MeshHierarchy for a CAD project"
    )
    cad_hier.add_argument("project", type=Path)
    cad_hier.add_argument("--output", type=Path, default=None)
    cad_hier.add_argument("--geometry-tolerance-mm", type=float, default=None)
    cad_hier.set_defaults(handler=_cad)
    cad_conv = cad_commands.add_parser(
        "convergence",
        help="remesh a STEP or DXF source at multiple sizes and check geometry convergence",
    )
    cad_conv.add_argument("source", type=Path)
    cad_conv.add_argument("--unit", choices=("mm", "cm", "m"), required=True)
    cad_conv.add_argument("--mesh-sizes", nargs="+", type=float, required=True)
    cad_conv.add_argument("--max-area-change", type=float, default=0.01)
    cad_conv.add_argument("--max-perimeter-change", type=float, default=0.01)
    cad_conv.add_argument("--max-chord-error-mm", type=float, default=0.05)
    cad_conv.add_argument("--output", type=Path, default=None)
    cad_conv.set_defaults(handler=_cad)
    cad_adapt = cad_commands.add_parser(
        "adapt",
        help="adapt a pretrained BoundaryEnergyOperator to a CAD project via low-rank deltas",
    )
    cad_adapt.add_argument("--project", type=Path, required=True)
    cad_adapt.add_argument("--checkpoint", type=Path, required=True)
    cad_adapt.add_argument("--config", type=Path, required=True)
    cad_adapt.add_argument("--output", type=Path, required=True)
    cad_adapt.add_argument("--ood-detector", type=Path, default=None)
    cad_adapt.set_defaults(handler=_cad)

    cad_param = cad_commands.add_parser(
        "parametric",
        help="parametric CAD family init, DOE sampling, training, serving, and qualification (M4)",
    )
    param_commands = cad_param.add_subparsers(dest="parametric_command", required=True)
    p_init = param_commands.add_parser(
        "init", help="initialize a hash-bound parametric CAD project"
    )
    p_init.add_argument("--spec", type=Path, default=None)
    p_init.add_argument(
        "--template",
        choices=("notched-mounting-bracket", "holed-plate"),
        default="notched-mounting-bracket",
    )
    p_init.add_argument("--mesh-size-mm", type=float, default=2.5)
    p_init.add_argument("--no-verify-nominal", action="store_true")
    p_init.add_argument("--output", type=Path, required=True)
    p_init.set_defaults(handler=_cad)

    p_sample = param_commands.add_parser(
        "sample", help="sample topology-verified DOE variants across the parameter box"
    )
    p_sample.add_argument("project", type=Path)
    p_sample.add_argument("--count", type=int, default=16)
    p_sample.add_argument("--seed-offset", type=int, default=1)
    p_sample.add_argument("--write-steps", action="store_true")
    p_sample.add_argument("--output", type=Path, required=True)
    p_sample.set_defaults(handler=_cad)

    p_train = param_commands.add_parser(
        "train",
        help="train a geometry-aware BoundaryEnergyOperator + GeometryOODDetector bundle",
    )
    p_train.add_argument("project", type=Path)
    p_train.add_argument("--physics", type=Path, default=None)
    p_train.add_argument("--count", type=int, default=12)
    p_train.add_argument("--seed-offset", type=int, default=1)
    p_train.add_argument("--epochs", type=int, default=60)
    p_train.add_argument("--learning-rate", type=float, default=3.0e-3)
    p_train.add_argument("--points-per-cell", type=int, default=2)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--supervise", action="store_true")
    p_train.add_argument("--output", type=Path, required=True)
    p_train.set_defaults(handler=_cad)

    p_pred = param_commands.add_parser(
        "predict", help="predict and OOD-gate one parametric CAD variant"
    )
    p_pred.add_argument("bundle", type=Path)
    p_pred.add_argument("--cad-params", type=Path, required=True)
    p_pred.add_argument(
        "--mode", choices=("fast", "certified", "exact"), default="certified"
    )
    p_pred.add_argument("--output", type=Path, default=None)
    p_pred.set_defaults(handler=_cad)

    p_qual = param_commands.add_parser(
        "qualify",
        help="qualify a parametric CAD bundle on held-out unseen geometries and OOD probes",
    )
    p_qual.add_argument("bundle", type=Path)
    p_qual.add_argument("--count", type=int, default=10)
    p_qual.add_argument("--seed-offset", type=int, default=101)
    p_qual.add_argument("--max-disp-error", type=float, default=0.25)
    p_qual.add_argument("--max-reaction-error", type=float, default=0.25)
    p_qual.add_argument("--output", type=Path, default=None)
    p_qual.set_defaults(handler=_cad)

    cad_imp3d = cad_commands.add_parser(
        "import-3d",
        help="import a single-solid 3D STEP file into a TET10/TET4 CanonicalMesh3D project (M5)",
    )
    cad_imp3d.add_argument("source", type=Path)
    cad_imp3d.add_argument("--unit", choices=("mm", "cm", "m"), default="mm")
    cad_imp3d.add_argument("--mesh-size-mm", type=float, default=3.0)
    cad_imp3d.add_argument("--element", choices=("tet10", "tet4"), default="tet10")
    cad_imp3d.add_argument("--poisson-ratio", type=float, default=None)
    cad_imp3d.add_argument("--output", type=Path, required=True)
    cad_imp3d.set_defaults(handler=_cad)

    cad_insp3d = cad_commands.add_parser(
        "inspect-3d",
        help="verify integrity and inspect a prepared 3D CAD geometry project (M5)",
    )
    cad_insp3d.add_argument("project", type=Path)
    cad_insp3d.set_defaults(handler=_cad)

    cad_samp3d = cad_commands.add_parser(
        "sample-3d",
        help="generate a built-in OpenCASCADE 3D STEP solid (box, l-bracket, mounting-lug)",
    )
    cad_samp3d.add_argument(
        "--template",
        choices=("box", "l-bracket", "mounting-lug"),
        default="mounting-lug",
    )
    cad_samp3d.add_argument("--output", type=Path, required=True)
    cad_samp3d.set_defaults(handler=_cad)

    cad_p3d = cad_commands.add_parser(
        "parametric-3d",
        help="3D parametric CAD DOE sampling, GALE training, serving, and qualification (V3-2)",
    )
    p3d_cmds = cad_p3d.add_subparsers(dest="parametric_3d_command", required=True)
    p3d_sample = p3d_cmds.add_parser(
        "sample", help="sample topology-verified 3D CAD DOE"
    )
    p3d_sample.add_argument(
        "--template",
        choices=("holed-block", "l-bracket", "mounting-lug"),
        default="holed-block",
    )
    p3d_sample.add_argument("--mesh-size-mm", type=float, default=6.0)
    p3d_sample.add_argument("--count", type=int, default=4)
    p3d_sample.add_argument("--seed-offset", type=int, default=1)
    p3d_sample.add_argument("--output", type=Path, required=True)
    p3d_sample.set_defaults(handler=_cad)

    p3d_train = p3d_cmds.add_parser(
        "train", help="train a multi-CAD 3D GALE surrogate bundle"
    )
    p3d_train.add_argument(
        "--template",
        choices=("holed-block", "l-bracket", "mounting-lug"),
        default="holed-block",
    )
    p3d_train.add_argument("--mesh-size-mm", type=float, default=6.0)
    p3d_train.add_argument("--count", type=int, default=4)
    p3d_train.add_argument("--epochs", type=int, default=30)
    p3d_train.add_argument("--supervise", action="store_true")
    p3d_train.add_argument("--output", type=Path, required=True)
    p3d_train.set_defaults(handler=_cad)

    p3d_pred = p3d_cmds.add_parser(
        "predict", help="predict & OOD-gate one 3D CAD variant"
    )
    p3d_pred.add_argument("bundle", type=Path)
    p3d_pred.add_argument("--cad-params", type=Path, required=True)
    p3d_pred.add_argument(
        "--mode", choices=("fast", "certified", "exact"), default="fast"
    )
    p3d_pred.add_argument("--output", type=Path, default=None)
    p3d_pred.set_defaults(handler=_cad)

    p3d_qual = p3d_cmds.add_parser(
        "qualify", help="qualify a 3D parametric bundle on holdouts + OOD"
    )
    p3d_qual.add_argument("bundle", type=Path)
    p3d_qual.add_argument("--count", type=int, default=3)
    p3d_qual.add_argument("--max-disp-error", type=float, default=0.35)
    p3d_qual.add_argument("--output", type=Path, default=None)
    p3d_qual.set_defaults(handler=_cad)

    active = subparsers.add_parser(
        "active", help="certificate-driven TRAIN-only active FEM acquisition (P3)"
    )
    active_commands = active.add_subparsers(dest="active_command", required=True)
    active_sel = active_commands.add_parser(
        "select", help="deterministically rank and select the next TRAIN FEM batch"
    )
    active_sel.add_argument("--plan", type=Path, required=True)
    active_sel.add_argument("--pool", type=Path, required=True)
    active_sel.add_argument("--acquired", type=Path, default=None)
    active_sel.add_argument("--round", type=int, default=0)
    active_sel.add_argument("--output", type=Path, default=None)
    active_sel.set_defaults(handler=_active_select)

    optimize = subparsers.add_parser(
        "optimize",
        help="FEM-guarded differentiable design optimization on a verified bundle (P5)",
    )
    optimize.add_argument("bundle", type=Path)
    optimize.add_argument("--problem", type=Path, required=True)
    optimize.add_argument("--recipe", type=Path, default=None)
    optimize.add_argument("--output", type=Path, default=None)
    optimize.add_argument("--refuse-unsupported", action="store_true")
    optimize.set_defaults(handler=_optimize)

    stateful = subparsers.add_parser(
        "stateful", help="stateful nonlinear mechanics rollout and checkpointing (P6)"
    )
    stateful_commands = stateful.add_subparsers(dest="stateful_command", required=True)
    st_roll = stateful_commands.add_parser(
        "rollout", help="run a guarded finite-strain viscoelastic trajectory rollout"
    )
    st_roll.add_argument("--artifact", type=Path, required=True)
    st_roll.add_argument("--load-path", type=Path, required=True)
    st_roll.add_argument("--resume", type=Path, default=None)
    st_roll.add_argument("--checkpoint-out", type=Path, default=None)
    st_roll.add_argument("--residual-tolerance", type=float, default=5.0)
    st_roll.add_argument("--output", type=Path, default=None)
    st_roll.set_defaults(handler=_stateful_rollout)
    st_tan = stateful_commands.add_parser(
        "verify-tangent",
        help="verify J2 elastoplastic consistent tangent against central finite differences",
    )
    st_tan.add_argument("--material", type=Path, required=True)
    st_tan.add_argument(
        "--mode",
        choices=("3d", "plane-strain", "plane-stress"),
        default="plane-stress",
    )
    st_tan.add_argument("--rel-tol", type=float, default=1.0e-5)
    st_tan.add_argument("--output", type=Path, default=None)
    st_tan.set_defaults(handler=_stateful_verify_tangent)

    store = subparsers.add_parser("store")
    store_commands = store.add_subparsers(dest="store_command", required=True)
    prune = store_commands.add_parser("prune")
    prune.add_argument("--root", required=True, type=Path)
    prune.add_argument("--dry-run", action="store_true")
    prune.set_defaults(handler=_store_prune)

    predict = subparsers.add_parser("predict", help="serve one request from a bundle")
    predict.add_argument("bundle", type=Path)
    predict.add_argument("--request", default=None, type=Path)
    predict.add_argument(
        "--cad-params",
        default=None,
        type=Path,
        help="parametric CAD parameter JSON when serving a formamech.parametric-bundle-v1",
    )
    predict.add_argument("--output", type=Path, default=None)
    predict.add_argument(
        "--no-confidence",
        action="store_true",
        help="fast: leave out the S14 pointwise stress bound "
        "(certified and exact leave it out by default)",
    )
    predict.add_argument(
        "--mode",
        choices=("fast", "certified", "exact"),
        default="fast",
        help="fast: the bundle alone; certified: plus the physics certificate "
        "against the bundle's FEM problem; exact: the converged FEM solution "
        f"started from the prediction (certified and exact need {prog}[verify])",
    )
    predict.add_argument(
        "--corrected",
        action="store_true",
        help="certified or exact: add the one-step corrected displacement "
        "at the FEM nodes",
    )
    predict.add_argument(
        "--escalate",
        action="store_true",
        help="certified: solve exactly when the certificate is not OK "
        "or exceeds --tolerance",
    )
    predict.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="certified with --escalate: the largest accepted relative L2 change "
        "of the first Piola stress along the certificate's Newton step "
        "(0.01 = 1%%); a larger change is solved exactly",
    )
    predict.add_argument(
        "--recipe",
        type=Path,
        default=None,
        help="FEM recipe (default: beside the bundle)",
    )
    predict.add_argument(
        "--bounds",
        action="store_true",
        help="attach the calibrated 90%% upper bounds of the bundle's "
        "calibration sidecar (fast: S14 budget total; certified and exact: "
        "FEM reference terms)",
    )
    predict.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="calibration sidecar to use with --bounds (default: "
        "<bundles>/calibration/<bundle>.json; bounds not confirmed on H1, listed "
        "in <bundles>/h1-results/<bundle>.json, are then withheld)",
    )
    predict.add_argument(
        "--trust-report",
        action="store_true",
        help="attach a versioned CADTrustReport separating geometry, "
        "discretization, surrogate (with live DWR adjoints in certified/exact "
        "mode) and statistical status",
    )
    predict.add_argument(
        "--trust-sidecar",
        type=Path,
        default=None,
        help="optional geometry/discretization convergence sidecar JSON for --trust-report",
    )
    predict.add_argument(
        "--backend",
        type=str,
        default=None,
        help="FEM solver backend name registered under formamech.fem.backends "
        "(default: scikit-fem)",
    )
    predict.add_argument(
        "--tier0",
        action="store_true",
        help="evaluate Tier 0 zero-FEM physics admissibility checks on served fields",
    )
    predict.add_argument(
        "--spectral-repair",
        action="store_true",
        help="exact/escalated: run bounded discrete spectral repair before warm Newton "
        "when initial predictions are inadmissible or indefinite",
    )
    predict.add_argument(
        "--cad-calibration",
        type=Path,
        default=None,
        help="path to a per-CAD split-conformal calibration sidecar JSON",
    )
    predict.add_argument(
        "--tta-steps",
        type=int,
        default=0,
        help="certified: optional zero-shot test-time physics LoRA steps (1..200) "
        "before escalating to Tier 2 FEM",
    )
    predict.add_argument(
        "--receipt",
        action="store_true",
        help="emit a canonical RFC 8785 Trust Receipt (formamech.trust-receipt-v1)",
    )
    predict.add_argument(
        "--require-qualified",
        action="store_true",
        help="fail closed unless the bundle passes the qualification deployment gate",
    )
    predict.add_argument(
        "--qualification-manifest",
        type=Path,
        default=None,
        help="path to qualification-manifest.json used with --require-qualified",
    )
    predict.add_argument(
        "--min-claim",
        choices=("DEVELOPMENT_VERIFIED", "HOLDOUT_VERIFIED"),
        default="DEVELOPMENT_VERIFIED",
        help="minimum ClaimLevel enforced by --require-qualified",
    )
    predict.set_defaults(handler=_predict)

    qualify = subparsers.add_parser(
        "qualify",
        help="continuous numerical model qualification, verification gates, and diffs",
    )
    qualify_commands = qualify.add_subparsers(dest="qualify_command", required=True)
    qual_run = qualify_commands.add_parser(
        "run", help="qualify a bundle against a declarative QualificationPolicy"
    )
    qual_run.add_argument("bundle", type=Path)
    qual_run.add_argument("--policy", type=Path, default=None)
    qual_run.add_argument(
        "--preset",
        choices=("dev", "holdout"),
        default="dev",
        help="built-in policy preset when --policy is omitted",
    )
    qual_run.add_argument("--dev-cases", type=Path, default=None)
    qual_run.add_argument("--holdout-cases", type=Path, default=None)
    qual_run.add_argument("--train-cases", type=Path, default=None)
    qual_run.add_argument("--sample-dev", type=int, default=9)
    qual_run.add_argument("--sample-holdout", type=int, default=0)
    qual_run.add_argument("--seed", type=int, default=0)
    qual_run.add_argument("--recipe", type=Path, default=None)
    qual_run.add_argument("--backend", type=str, default="scikit-fem")
    qual_run.add_argument("--output", type=Path, required=True)
    qual_run.add_argument(
        "--emit-qualified-bundle",
        type=Path,
        default=None,
        help="optional destination for a newly promoted bundle copy when QUALIFIED",
    )
    qual_run.set_defaults(handler=_qualify)

    qual_verify = qualify_commands.add_parser(
        "verify", help="enforce the deployment gate for a bundle and manifest"
    )
    qual_verify.add_argument("bundle", type=Path)
    qual_verify.add_argument("--manifest", type=Path, default=None)
    qual_verify.add_argument("--policy", type=Path, default=None)
    qual_verify.add_argument(
        "--min-claim",
        choices=("DEVELOPMENT_VERIFIED", "HOLDOUT_VERIFIED"),
        default="DEVELOPMENT_VERIFIED",
    )
    qual_verify.add_argument("--output", type=Path, default=None)
    qual_verify.set_defaults(handler=_qualify)

    qual_diff = qualify_commands.add_parser(
        "diff", help="compare two qualification manifests for CI regression detection"
    )
    qual_diff.add_argument("old_manifest", type=Path)
    qual_diff.add_argument("new_manifest", type=Path)
    qual_diff.add_argument("--regression-tol", type=float, default=0.05)
    qual_diff.add_argument("--output", type=Path, default=None)
    qual_diff.set_defaults(handler=_qualify)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="build a per-CAD split-conformal & functional-band calibration sidecar",
    )
    calibrate.add_argument("bundle", type=Path)
    calibrate.add_argument("--scores", type=Path, required=True)
    calibrate.add_argument("--alpha", type=float, default=0.05)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.set_defaults(handler=_calibrate)

    ui = subparsers.add_parser(
        "ui",
        help="launch the Interactive Engineering Workbench & Trust Inspector (needs [ui] extra)",
    )
    ui.add_argument("--bundle", type=Path, default=None)
    ui.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="canonical workbench project directory (default: .formamech-runs/workbench/demo-project)",
    )
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument("--headless", action="store_true")
    ui.set_defaults(handler=_ui)

    mcp = subparsers.add_parser(
        "mcp",
        help="run or inspect the sandboxed FormaMech MCP server",
    )
    mcp_commands = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_serve = mcp_commands.add_parser(
        "serve", help="serve JSON-RPC 2.0 MCP tools over stdio"
    )
    mcp_serve.add_argument("--workspace", type=Path, default=Path("."))
    mcp_serve.add_argument("--max-dofs", type=int, default=100_000)
    mcp_serve.add_argument(
        "--require-qualified",
        action="store_true",
        help="refuse predict_verified unless a valid QUALIFIED manifest is supplied",
    )
    mcp_serve.set_defaults(handler=_mcp)
    mcp_list = mcp_commands.add_parser(
        "list-tools", help="print declared MCP tool descriptors as JSON"
    )
    mcp_list.add_argument("--workspace", type=Path, default=Path("."))
    mcp_list.add_argument("--max-dofs", type=int, default=100_000)
    mcp_list.add_argument("--output", type=Path, default=None)
    mcp_list.set_defaults(handler=_mcp)

    agent = subparsers.add_parser(
        "agent",
        help="AI-ready Universal Agent Harness (capabilities, schema export, sandboxed invocation, ledger audit)",
    )
    agent_commands = agent.add_subparsers(dest="agent_command", required=True)
    ag_caps = agent_commands.add_parser(
        "capabilities", help="list registered FormaMech agent capabilities"
    )
    ag_caps.add_argument("--domain", type=str, default=None)
    ag_caps.add_argument("--cost-class", type=str, default=None)
    ag_caps.add_argument("--search", type=str, default=None)
    ag_caps.add_argument("--output", type=Path, default=None)
    ag_caps.set_defaults(handler=_agent)

    ag_desc = agent_commands.add_parser(
        "describe", help="print full JSON Schema descriptor for one capability"
    )
    ag_desc.add_argument("capability", type=str)
    ag_desc.add_argument("--output", type=Path, default=None)
    ag_desc.set_defaults(handler=_agent)

    ag_inv = agent_commands.add_parser(
        "invoke",
        help="invoke (or dry-run) any registered capability in a workspace session",
    )
    ag_inv.add_argument("capability", type=str)
    ag_inv.add_argument("--workspace", type=Path, required=True)
    ag_inv.add_argument("--args-json", type=str, default="{}")
    ag_inv.add_argument("--dry-run", action="store_true")
    ag_inv.add_argument("--max-fem-solves", type=int, default=32)
    ag_inv.add_argument("--max-dofs", type=int, default=150_000)
    ag_inv.add_argument("--output", type=Path, default=None)
    ag_inv.set_defaults(handler=_agent)

    ag_stat = agent_commands.add_parser(
        "session-status", help="inspect persisted agent session state and budget"
    )
    ag_stat.add_argument("--workspace", type=Path, required=True)
    ag_stat.add_argument("--output", type=Path, default=None)
    ag_stat.set_defaults(handler=_agent)

    ag_led = agent_commands.add_parser(
        "verify-ledger", help="verify SHA-256 hash chain of session ledger.jsonl"
    )
    ag_led.add_argument("--workspace", type=Path, required=True)
    ag_led.add_argument("--output", type=Path, default=None)
    ag_led.set_defaults(handler=_agent)

    ag_exp = agent_commands.add_parser(
        "export-tools",
        help="export native LLM tool schemas (openai, anthropic, gemini, mcp)",
    )
    ag_exp.add_argument(
        "--provider",
        choices=("openai", "anthropic", "gemini", "mcp"),
        default="openai",
    )
    ag_exp.add_argument(
        "--mode",
        choices=("progressive", "full"),
        default="progressive",
    )
    ag_exp.add_argument("--domain", type=str, default=None)
    ag_exp.add_argument("--output", type=Path, default=None)
    ag_exp.set_defaults(handler=_agent)

    sapi = subparsers.add_parser(
        "serve-api",
        help="run the production verified HTTP/JSON & Prometheus serving API (V4-1)",
    )
    sapi.add_argument("--workspace", type=Path, default=Path(".formamech-runs/api"))
    sapi.add_argument("--bundle", type=Path, default=None)
    sapi.add_argument("--ledger", type=Path, default=None)
    sapi.add_argument("--host", type=str, default="127.0.0.1")
    sapi.add_argument("--port", type=int, default=8080)
    sapi.add_argument("--max-fem-solves", type=int, default=32)
    sapi.add_argument("--max-dofs", type=int, default=150_000)
    sapi.set_defaults(handler=_serve_api)

    bench = subparsers.add_parser(
        "bench",
        help="run reproducible 3D solver, foundation transfer, OOD & parity benchmarks (V4-2)",
    )
    bench_cmds = bench.add_subparsers(dest="bench_command", required=True)
    b_run = bench_cmds.add_parser("run", help="execute B1-B4 3D benchmark suite")
    b_run.add_argument("--profile", choices=("quick", "full"), default="quick")
    b_run.add_argument("--workdir", type=Path, default=Path(".formamech-runs/bench"))
    b_run.add_argument("--foundation-dir", type=Path, default=None)
    b_run.add_argument("--output", type=Path, default=None)
    b_run.add_argument("--markdown", type=Path, default=None)
    b_run.set_defaults(handler=_bench)

    fleet = subparsers.add_parser(
        "fleet",
        help="distributed 3D FEM fleet workers (local multiprocess, Slurm, Kubernetes) (V4-5)",
    )
    fleet_cmds = fleet.add_subparsers(dest="fleet_command", required=True)
    fl_loc = fleet_cmds.add_parser(
        "submit-local", help="run parallel local 3D FEM batch"
    )
    fl_loc.add_argument("--job-id", type=str, default="local-batch")
    fl_loc.add_argument("--workdir", type=Path, required=True)
    fl_loc.add_argument("--tasks-json", type=str, default=None)
    fl_loc.add_argument("--max-workers", type=int, default=2)
    fl_loc.add_argument("--use-threads", action="store_true")
    fl_loc.add_argument("--output", type=Path, default=None)
    fl_loc.set_defaults(handler=_fleet)

    fl_sl = fleet_cmds.add_parser("generate-slurm", help="generate Slurm array script")
    fl_sl.add_argument("--job-id", type=str, default="slurm-batch")
    fl_sl.add_argument("--workdir", type=Path, required=True)
    fl_sl.add_argument("--tasks-json", type=str, default=None)
    fl_sl.add_argument("--partition", type=str, default="compute")
    fl_sl.add_argument("--output", type=Path, default=None)
    fl_sl.set_defaults(handler=_fleet)

    fl_k8s = fleet_cmds.add_parser(
        "generate-k8s", help="generate Kubernetes Indexed Job YAML"
    )
    fl_k8s.add_argument("--job-id", type=str, default="k8s-batch")
    fl_k8s.add_argument("--workdir", type=Path, required=True)
    fl_k8s.add_argument("--tasks-json", type=str, default=None)
    fl_k8s.add_argument("--image", type=str, default="formamech/worker:v0.4")
    fl_k8s.add_argument("--output", type=Path, default=None)
    fl_k8s.set_defaults(handler=_fleet)

    cont = subparsers.add_parser(
        "continual",
        help="inspect continual qualification loop, audit ledger & sharded 3D dataset (V4-5)",
    )
    cont_cmds = cont.add_subparsers(dest="continual_command", required=True)
    ct_st = cont_cmds.add_parser(
        "status", help="verify audit ledger & inspect continual backlog"
    )
    ct_st.add_argument("--ledger", type=Path, required=True)
    ct_st.add_argument("--dataset", type=Path, required=True)
    ct_st.add_argument("--output", type=Path, default=None)
    ct_st.set_defaults(handler=_continual)

    export = subparsers.add_parser(
        "export",
        help="export a verified bundle to ONNX or FMI 3.0 Co-Simulation FMU",
    )
    export_commands = export.add_subparsers(dest="export_command", required=True)
    exp_onnx = export_commands.add_parser(
        "onnx",
        help="export surrogate operator + lifting to ONNX (needs [export] extra)",
    )
    exp_onnx.add_argument("bundle", type=Path)
    exp_onnx.add_argument("--output", type=Path, required=True)
    exp_onnx.add_argument("--opset", type=int, default=17)
    exp_onnx.add_argument("--no-verify", action="store_true")
    exp_onnx.add_argument("--require-qualified", action="store_true")
    exp_onnx.add_argument("--qualification-manifest", type=Path, default=None)
    exp_onnx.add_argument(
        "--min-claim",
        choices=("DEVELOPMENT_VERIFIED", "HOLDOUT_VERIFIED"),
        default="DEVELOPMENT_VERIFIED",
    )
    exp_onnx.set_defaults(handler=_export)
    exp_fmu = export_commands.add_parser(
        "fmu", help="package an FMI 3.0 Co-Simulation .fmu archive"
    )
    exp_fmu.add_argument("bundle", type=Path)
    exp_fmu.add_argument("--output", type=Path, required=True)
    exp_fmu.add_argument("--model-name", type=str, default=None)
    exp_fmu.add_argument("--include-onnx", action="store_true")
    exp_fmu.add_argument("--require-qualified", action="store_true")
    exp_fmu.add_argument("--qualification-manifest", type=Path, default=None)
    exp_fmu.add_argument(
        "--min-claim",
        choices=("DEVELOPMENT_VERIFIED", "HOLDOUT_VERIFIED"),
        default="DEVELOPMENT_VERIFIED",
    )
    exp_fmu.set_defaults(handler=_export)

    ingest = subparsers.add_parser(
        "ingest",
        help="ingest external CAE archives (VTU/XDMF/CalculiX/Abaqus/Nastran) with outlier screening (V3-3)",
    )
    ingest.add_argument("source", type=Path)
    ingest.add_argument(
        "--format",
        choices=("auto", "vtu", "xdmf", "calculix-frd", "abaqus-inp", "nastran-bdf"),
        default="auto",
    )
    ingest.add_argument("--fields", type=Path, default=None)
    ingest.add_argument("--zscore-threshold", type=float, default=3.0)
    ingest.add_argument("--output", type=Path, default=None)
    ingest.set_defaults(handler=_ingest_cae)

    validate = subparsers.add_parser(
        "validate",
        help="differentiable inverse material calibration & ASME V&V 10 coupon validation (V3-3)",
    )
    validate.add_argument("coupons", type=Path)
    validate.add_argument(
        "--model-family",
        choices=(
            "incompressible_neo_hookean",
            "incompressible_mooney_rivlin",
            "incompressible_yeoh",
            "j2_voce_1d",
        ),
        default="incompressible_mooney_rivlin",
    )
    validate.add_argument("--u-num-mpa", type=float, default=0.01)
    validate.add_argument("--output", type=Path, default=None)
    validate.set_defaults(handler=_validate_experiment)

    inspect = subparsers.add_parser("inspect", help="describe a bundle")
    inspect.add_argument("bundle", type=Path)
    inspect.add_argument("--output", type=Path, default=None)
    inspect.set_defaults(handler=_inspect)

    bundle = subparsers.add_parser("bundle", help="verify or build a bundle")
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    verify = bundle_commands.add_parser("verify")
    verify.add_argument("bundle", type=Path)
    verify.set_defaults(handler=_bundle_verify)
    _campaign_parser(bundle_commands, "build", "bundle build")

    family = subparsers.add_parser("family", help="campaign family commands")
    family_commands = family.add_subparsers(dest="family_command", required=True)
    _campaign_parser(family_commands, "compile", "family compile")
    for command in (
        "pilot",
        "campaign",
        "simulate",
        "train",
        "verify",
        "holdout-reveal",
    ):
        _campaign_parser(subparsers, command, command)
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str = "piginot") -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    if hasattr(args, "campaign_command"):
        return _campaign(args.campaign_command, args)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_OK
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
