"""S10 for the notched campaign: request plan, FEM production, close-out, G10.

The FEM recipe is the DogBone S10 recipe on the morphed reference mesh. The
production runner, the materialiser and the G10 reducer are the framework and
DogBone-campaign code paths, called with the notched requests and policy.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from campaigns.dogbone_v1.s10_closeout import (
    _clean_source_revision,
    _load_run_execution_profile,
    gate_measurements_from_report,
    materialize_s10_dataset,
    write_s10_closeout_report,
)
from campaigns.dogbone_v1.s10_evidence import (
    S10CloseoutReport,
    S10DatasetPolicy,
    S10MutationEvidence,
)
from campaigns.dogbone_v1.s10_production import (
    ProductionFemRunLedger,
    ProductionFemRunRecord,
    _assert_runtime_source,
    _assert_single_thread_environment,
    _exclusive_run_lock,
    _progress,
    default_s10_execution_profile,
    load_production_attempts,
    persist_execution_profile,
    persist_request_plan,
    run_public_fem_requests,
)
from campaigns.notched_v1.family import (
    GEOMETRY,
    INPUTS,
    NOTCH_TAG,
    boundary_conditions,
    compiled_family,
    material,
    parameter_point,
    solver_config,
)
from campaigns.notched_v1.gate_io import (
    check_document,
    gate_document,
    run_pytest_with_coverage,
    write_gate_document,
)
from campaigns.notched_v1.s9 import CP_ENVELOPE_PATH, GATES, PUBLIC_MANIFEST_PATH
from piginot.contracts import (
    CanonicalQuantity,
    ExecutionProfile,
    MaterialSpec,
    Unit,
    hash_spec,
)
from piginot.datasets import PublicFemRequestPlan, evaluate_s10_gate
from piginot.datasets.collocation import CollocationKind, SemanticCollocationQuota
from piginot.datasets.request_plan import build_public_fem_request_plan
from piginot.doe import CPEnvelope, PublishedCase, PublishedCaseManifest
from piginot.doe.publication import PublishedCaseStatus
from piginot.fem import FemBackend, FemRequest
from piginot.fem.backend import ScikitFemBackend
from piginot.morphing import apply_morph
from piginot.store import ObjectStore

RUN_ID = "notched-s10-public-production-v1"
REPOSITORY = Path(__file__).resolve().parents[2]
GATE_PATH = GATES / "S10.json"
_LEDGER, _PLAN, _PROFILE, _STORE, _LOCK = (
    "ledger.json",
    "request-plan.json",
    "execution-profile.json",
    "object-store",
    "run.lock",
)
_PRIMITIVES = frozenset((*INPUTS, "elastic_modulus", "poisson_ratio"))
# Inside every family member: the ligament at the notch root is at least 3 mm.
QUERY_POINTS = tuple(
    (x, y) for x in (1.0, 5.0, 10.0, 15.0, 20.0) for y in (0.75, 1.5, 2.25)
)
DATASET_POLICY = S10DatasetPolicy(
    query_points=QUERY_POINTS,
    interior_count=128,
    boundary_count=64,
    interface_count=0,
    qoi_quotas=(
        SemanticCollocationQuota(
            quota_id="stress-critical-notch-v1",
            kind=CollocationKind.BOUNDARY,
            count=64,
            semantic_tag=NOTCH_TAG,
        ),
    ),
    reference_scale=CanonicalQuantity(760.0, Unit.MPA),
)
ROLE_COUNTS = {"PILOT": 8, "TRAIN": 128, "DEVELOPMENT": 64}
G10_MUTANTS = {
    "DROP_PROJECTION_ERROR": "G10.projection-error",
    "STRESS_NODE_PAIRING": "G10.cross-mesh",
    "NORMALIZE_ON_ALL_SPLITS": "G10.normalization",
}


def load_public_sources(
    root: Path = REPOSITORY,
) -> tuple[CPEnvelope, PublishedCaseManifest]:
    """The committed CP-ENVELOPE and manifest, bound to the runtime family."""

    checkpoint = CPEnvelope.from_bytes((root / CP_ENVELOPE_PATH).read_bytes())
    manifest = PublishedCaseManifest.from_bytes(
        (root / PUBLIC_MANIFEST_PATH).read_bytes()
    )
    if manifest.content_hash != checkpoint.public_case_manifest_hash:
        raise RuntimeError("notched public manifest does not match CP-ENVELOPE")
    if hash_spec(compiled_family(), "content") != checkpoint.compiled_family_hash:
        raise RuntimeError("runtime notched family differs from CP-ENVELOPE")
    return checkpoint, manifest


def number(case: PublishedCase, name: str) -> float:
    value = float(case.primitive_coordinates[name])
    if not math.isfinite(value):
        raise RuntimeError(f"case {case.case_id!r} has non-finite {name}")
    return value


def case_material(case: PublishedCase) -> MaterialSpec:
    return material(number(case, "elastic_modulus"), number(case, "poisson_ratio"))


def case_request(case: PublishedCase, profile: ExecutionProfile) -> FemRequest:
    """The exact S10 request of one public case."""

    if frozenset(case.primitive_coordinates) != _PRIMITIVES:
        raise RuntimeError(f"case {case.case_id!r} primitive axes changed")
    family = compiled_family()
    theta = parameter_point(*(number(case, name) for name in GEOMETRY))
    morph = apply_morph(
        family.reference_mesh, family.morph_map, family.membership, theta
    )
    if not morph.accepted or morph.mesh is None:
        raise RuntimeError(f"case {case.case_id!r} failed morphing")
    return FemRequest(
        request_id=case.case_id,
        mesh=morph.mesh,
        material=case_material(case),
        boundary_conditions=boundary_conditions(number(case, "nominal_axial_strain")),
        loads=(),
        qois=(),
        config=solver_config(),
        execution_profile=profile,
    )


@lru_cache(maxsize=4)
def public_requests(profile: ExecutionProfile) -> tuple[FemRequest, ...]:
    _, manifest = load_public_sources()
    active = sorted(
        (case for case in manifest.cases if case.status is PublishedCaseStatus.ACTIVE),
        key=lambda case: case.case_id,
    )
    return tuple(case_request(case, profile) for case in active)


def request_plan(profile: ExecutionProfile) -> PublicFemRequestPlan:
    checkpoint, manifest = load_public_sources()
    requests = public_requests(profile)
    return build_public_fem_request_plan(
        checkpoint, manifest, {request.request_id: request for request in requests}
    )


def run_production(
    run_root: Path,
    *,
    source_revision: str,
    execution_profile: ExecutionProfile | None = None,
    backend: FemBackend | None = None,
    max_new_attempts: int | None = None,
    dry_run: bool = False,
    on_record: Callable[[ProductionFemRunLedger, ProductionFemRunRecord], None]
    | None = None,
) -> ProductionFemRunLedger | PublicFemRequestPlan:
    """Execute or resume the notched S10 FEM campaign (one case at a time)."""

    profile = (
        default_s10_execution_profile()
        if execution_profile is None
        else execution_profile
    )
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    with _exclusive_run_lock(root / _LOCK):
        store = ObjectStore.create(root / _STORE)
        solver = ScikitFemBackend(store=store) if backend is None else backend
        plan = request_plan(profile)
        persist_request_plan(plan, root / _PLAN)
        persist_execution_profile(profile, root / _PROFILE)
        if dry_run:
            return plan
        return run_public_fem_requests(
            request_plan=plan,
            requests={
                request.request_id: request for request in public_requests(profile)
            },
            backend=solver,
            store=store,
            ledger_path=root / _LEDGER,
            source_revision=source_revision,
            clock=perf_counter,
            max_new_attempts=max_new_attempts,
            on_record=on_record,
            run_id=RUN_ID,
        )


def closeout(
    run_root: Path,
    *,
    materializer_source_revision: str,
    on_case: Callable[[int, int, str], None] | None = None,
) -> S10CloseoutReport:
    """Materialise the complete notched S10 dataset evidence."""

    root = Path(run_root)
    profile = _load_run_execution_profile(root)
    checkpoint, manifest = load_public_sources()
    plan = request_plan(profile)
    if (
        PublicFemRequestPlan.from_bytes((root / _PLAN).read_bytes()).to_bytes()
        != plan.to_bytes()
    ):
        raise ValueError("persisted request plan differs from the runtime plan")
    ledger = ProductionFemRunLedger.from_bytes((root / _LEDGER).read_bytes())
    if ledger.run_id != RUN_ID or not ledger.is_complete:
        raise ValueError("the notched production ledger is not complete")
    store = ObjectStore.open(root / _STORE)
    attempts = load_production_attempts(
        request_plan=plan,
        requests={request.request_id: request for request in public_requests(profile)},
        ledger=ledger,
        store=store,
    )
    report = materialize_s10_dataset(
        checkpoint=checkpoint,
        manifest=manifest,
        request_plan=plan,
        attempts=attempts,
        store=store,
        policy=DATASET_POLICY,
        run_id=ledger.run_id,
        fem_source_revision=ledger.source_revision,
        materializer_source_revision=materializer_source_revision,
        on_case=on_case,
    )
    observed = {item.role.value: item.case_count for item in report.role_snapshots}
    if observed != ROLE_COUNTS:
        raise ValueError(f"notched S10 role population changed: {observed}")
    return report


def mutant_evidence() -> dict[str, Any]:
    """Run the three G10 mutants of the S16 registry against their detectors."""

    from campaigns.dogbone_v1.s16_detectors import CACHES, DETECTORS, cleanup
    from campaigns.dogbone_v1.s16_mutants import MUTANTS
    from piginot.mutation import run_suite

    mutants = tuple(mutant for mutant in MUTANTS if mutant.name in G10_MUTANTS)
    detectors = tuple(d for d in DETECTORS if d.check_id in set(G10_MUTANTS.values()))
    try:
        report = run_suite(mutants, detectors, caches=CACHES)
    finally:
        cleanup()
    fired = {run.mutant.name: run.primary_fired for run in report.runs}
    controls = report.control_passed
    return {"fired": fired, "controls_pass": controls}


def write_gate(report: S10CloseoutReport, report_path: Path) -> dict[str, Any]:
    """Reduce the close-out report, the G10 mutants and the coverage run."""

    mutants = mutant_evidence()
    summary, coverage, passed = run_pytest_with_coverage(
        REPOSITORY, ("tests/datasets",), ("piginot.datasets",)
    )
    fired = mutants["fired"]
    measurements = gate_measurements_from_report(
        report,
        mutation_evidence=S10MutationEvidence(
            drop_projection_error_mutant_fired=fired["DROP_PROJECTION_ERROR"]
            and mutants["controls_pass"],
            stress_node_pairing_mutant_fired=fired["STRESS_NODE_PAIRING"]
            and mutants["controls_pass"],
            normalize_on_all_splits_mutant_fired=fired["NORMALIZE_ON_ALL_SPLITS"]
            and mutants["controls_pass"],
            code_coverage_percent=coverage if passed else None,
        ),
    )
    evaluation = evaluate_s10_gate(measurements)
    checks = [
        check_document(
            check.check_id,
            check.status.value == "PASS",
            check.measured,
            check.threshold,
            check.comparison,
        )
        for check in evaluation.checks
    ]
    document = gate_document(
        "G10",
        "S10",
        REPOSITORY,
        checks,
        closeout_report=report_path.name,
        closeout_report_sha256=__import__("hashlib")
        .sha256(report_path.read_bytes())
        .hexdigest(),
        run_id=report.run_id,
        fem_source_revision=report.fem_source_revision,
        mutation_evidence=mutants,
        test_evidence={"command": "pytest tests/datasets", "result": summary},
    )
    write_gate_document(REPOSITORY / GATE_PATH, document)
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.notched_v1.s10")
    sub = parser.add_subparsers(dest="command", required=True)
    produce = sub.add_parser("produce")
    produce.add_argument("--run-root", type=Path, required=True)
    produce.add_argument("--max-new-attempts", type=int)
    produce.add_argument("--dry-run", action="store_true")
    close = sub.add_parser("closeout")
    close.add_argument("--run-root", type=Path, required=True)
    close.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "produce":
        _assert_runtime_source()
        _assert_single_thread_environment()
        revision = _clean_source_revision()
        result = run_production(
            arguments.run_root.resolve(),
            source_revision=revision,
            max_new_attempts=arguments.max_new_attempts,
            dry_run=arguments.dry_run,
            on_record=_progress,
        )
        if isinstance(result, PublicFemRequestPlan):
            print(
                f"dry run: plan {result.content_hash} with {len(result.entries)} cases"
            )
        else:
            print(
                f"completed {result.completed_case_count}/{result.planned_case_count} "
                f"({result.success_count} successes, {result.failure_count} failures)"
            )
        return 0
    if arguments.command == "closeout":
        revision = _clean_source_revision()
        report = closeout(
            arguments.run_root.resolve(),
            materializer_source_revision=revision,
            on_case=lambda index, total, case_id: print(
                f"[{index:03d}/{total:03d}] {case_id}", flush=True
            ),
        )
        output = arguments.output.resolve()
        write_s10_closeout_report(report, output)
        print(f"wrote {output}")
    document = write_gate(report, output)
    print(document["verdict"], document["verdict_reason"])
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = (
    "DATASET_POLICY",
    "QUERY_POINTS",
    "RUN_ID",
    "case_request",
    "closeout",
    "load_public_sources",
    "public_requests",
    "request_plan",
    "run_production",
)
