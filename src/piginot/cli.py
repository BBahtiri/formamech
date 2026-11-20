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

    problem = _predict_option_error(args)
    if problem is not None:
        print(f"{_program(args)}: {problem}", file=sys.stderr)
        return EXIT_ERROR
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
    if args.mode == "fast":
        if calibration is not None:
            from piginot.serving import serve

            served = serve(
                bundle,
                inputs,
                cast(Any, queries),
                confidence=not args.no_confidence,
                calibration=calibration,
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
    return _serve_verified(args, bundle, inputs, queries, calibration)


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
) -> int:
    """``--mode certified|exact`` (S22): certificate, correction, FEM solve, live DWR."""

    from piginot.serving import MissingExtraError, ServingPath, load_recipe, serve

    try:
        recipe = load_recipe(bundle, args.bundle, args.recipe)
        sidecar = (
            _load_trust_sidecar(args.bundle, getattr(args, "trust_sidecar", None))
            if getattr(args, "trust_report", False)
            else None
        )
        served = serve(
            bundle,
            inputs,
            queries,
            mode=args.mode,
            recipe=recipe,
            corrected=args.corrected,
            escalate=args.escalate,
            tolerance=args.tolerance,
            calibration=calibration,
            trust_report=bool(getattr(args, "trust_report", False)),
            trust_sidecar=sidecar,
        )
    except MissingExtraError as exc:
        print(f"{_program(args)}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (OSError, ValueError) as exc:
        print(f"{_program(args)}: cannot serve {args.mode}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    doc = served.document()
    if getattr(args, "trust_report", False) and "trust_report" not in doc:
        doc = _attach_trust_report(doc, bundle, mode=args.mode)
    _emit(doc, args.output)
    if served.path is ServingPath.REFUSED:
        return EXIT_REFUSED
    return EXIT_ERROR if served.path is ServingPath.SOLVE_FAILED else EXIT_OK


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
    """Run FEM-guarded differentiable design optimization on a verified bundle (P5)."""
    from piginot.bundle import BundleIntegrityError, load_bundle
    from piginot.optimization import (
        BundleDesignBridge,
        DesignOptimizationStatus,
        parse_bundle_design_document,
    )
    from piginot.serving import MissingExtraError, load_recipe

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

    store = subparsers.add_parser("store")
    store_commands = store.add_subparsers(dest="store_command", required=True)
    prune = store_commands.add_parser("prune")
    prune.add_argument("--root", required=True, type=Path)
    prune.add_argument("--dry-run", action="store_true")
    prune.set_defaults(handler=_store_prune)

    predict = subparsers.add_parser("predict", help="serve one request from a bundle")
    predict.add_argument("bundle", type=Path)
    predict.add_argument("--request", required=True, type=Path)
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
    predict.set_defaults(handler=_predict)

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
