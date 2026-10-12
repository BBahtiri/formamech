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


def _predict(args: argparse.Namespace) -> int:
    from piginot.bundle import BundleIntegrityError, load_bundle, predict

    problem = _predict_option_error(args)
    if problem is not None:
        print(f"piginot: {problem}", file=sys.stderr)
        return EXIT_ERROR
    try:
        bundle = load_bundle(args.bundle)
    except (BundleIntegrityError, OSError) as exc:
        _emit(_bundle_refusal(exc), args.output)
        return EXIT_ERROR
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"piginot: cannot read request: {exc}", file=sys.stderr)
        return EXIT_ERROR
    inputs, queries = parse_request(request)
    calibration = None
    if args.bounds or args.calibration is not None:
        from piginot.bounds import CalibrationError, CalibrationRecord, calibration_path

        try:
            calibration = CalibrationRecord.load(
                args.calibration or calibration_path(args.bundle),
                bundle=Path(args.bundle).name,
                manifest_sha256=bundle.manifest.manifest_sha256,
            )
        except CalibrationError as exc:
            print(f"piginot: {exc}", file=sys.stderr)
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
            _emit(served.document(), args.output)
            return EXIT_OK if served.prediction.in_domain else EXIT_REFUSED
        prediction = predict(
            bundle,
            inputs,
            queries,
            confidence=not args.no_confidence,
        )
        _emit(prediction.document(), args.output)
        return EXIT_OK if prediction.in_domain else EXIT_REFUSED
    return _serve_verified(args, bundle, inputs, queries, calibration)


def _serve_verified(
    args: argparse.Namespace,
    bundle: Any,
    inputs: Any,
    queries: Any,
    calibration: Any = None,
) -> int:
    """``--mode certified|exact`` (S22): certificate, correction, FEM solve."""

    from piginot.serving import MissingExtraError, ServingPath, load_recipe, serve

    try:
        recipe = load_recipe(bundle, args.bundle, args.recipe)
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
        )
    except MissingExtraError as exc:
        print(f"piginot: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except (OSError, ValueError) as exc:
        print(f"piginot: cannot serve {args.mode}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    _emit(served.document(), args.output)
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
        print(f"piginot: {exc}", file=sys.stderr)
        return EXIT_ERROR
    report = {
        "dry_run": result.dry_run,
        "object_count": result.object_count,
        "pruned_bytes": result.pruned_bytes,
        "pruned_hashes": list(result.pruned_hashes),
    }
    print(canonical_json_bytes(report).decode("utf-8"))
    return EXIT_OK


def _campaign(command: str, args: argparse.Namespace) -> int:
    try:
        table = resolve_campaign(args.campaign, args.campaign_path)
        handler = table.get(command)
        if handler is None:
            raise PluginError(f"campaign {args.campaign!r} has no {command!r} command")
    except PluginError as exc:
        print(f"piginot: {exc}", file=sys.stderr)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="piginot")
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")

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
        "started from the prediction (certified and exact need piginot[verify])",
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
        help="calibration sidecar to use with --bounds "
        "(default: <bundles>/calibration/<bundle>.json)",
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
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
