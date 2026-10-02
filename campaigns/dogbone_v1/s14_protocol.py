"""The frozen S14 protocol (``S14-PROTOCOL.json``) and its sign-off check.

The protocol fixes, before DEVELOPMENT is opened, every S14 decision of
ADR 0019 together with the methods of ADR 0020: tolerances, slices and their
statistic, the budget component rules, the confidence variant chosen by the
TRAIN pilot, the refusal fixture set, the latency protocol, the G14 check list
and the digests of every evaluation source file. Step 3 refuses to run unless
the user's sign-off record names the exact protocol digest and every
evaluation file still has its frozen digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from campaigns.dogbone_v1.s14_candidate import (
    CP_FREEZE_PATH,
    INDICATOR_LENGTH_SCALE_MM,
    frozen_candidate_record,
)
from campaigns.dogbone_v1.s14_envelope import FIXTURE_COUNTS
from campaigns.dogbone_v1.s14_evaluation import (
    DEVELOPMENT_ACCESS_NAME,
    HOLDOUT_CONSUMPTION_NAME,
    REVEAL_RELATIVE_PATH,
    holdout_commitment,
)
from campaigns.dogbone_v1.s14_gate import (
    COVERAGE_MINIMUM,
    FALSE_REFUSAL_MAXIMUM,
    REPORTED_CHECKS,
    REQUIRED_CHECKS,
    SPEARMAN_MINIMUM,
    SPEEDUP_TARGET,
)
from campaigns.dogbone_v1.s14_reference import (
    RECOVERY_SAFETY_FACTOR,
    REFINEMENT_RATIO,
    RICHARDSON_ORDER,
    RICHARDSON_SAFETY_FACTOR,
)
from campaigns.dogbone_v1.s14_rehearsal import (
    CALIBRATION_BINS,
    CALIBRATION_QUANTILE,
    GATED_SLICES,
    REPORTED_SLICES,
    SCORING_THREADS,
    TOLERANCES,
)
from piginot.verification import INDICATOR_VARIANTS
from piginot.verification.budget import MINIMUM_QUANTILE_CASES, QUANTILE

_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = _ROOT / "artifacts" / "campaigns" / "dogbone-v1"
PROTOCOL_PATH = ARTIFACTS / "S14-PROTOCOL.json"
SIGNOFF_PATH = ARTIFACTS / "S14-PROTOCOL-SIGNOFF.json"
REHEARSAL_PATH = ARTIFACTS / "S14-TRAIN-REHEARSAL.json"
PROTOCOL_ID = "dogbone-v1-s14-protocol-v1"
AMENDMENT_GLOB = "S14-PROTOCOL-AMENDMENT-*.json"
SIGNATORY = "Betim Bahtiri"
LATENCY_CASES = 8
LATENCY_REPEATS = 20
EVALUATION_FILES = (
    "campaigns/dogbone_v1/s14_candidate.py",
    "campaigns/dogbone_v1/s14_envelope.py",
    "campaigns/dogbone_v1/s14_evaluation.py",
    "campaigns/dogbone_v1/s14_gate.py",
    "campaigns/dogbone_v1/s14_protocol.py",
    "campaigns/dogbone_v1/s14_reference.py",
    "campaigns/dogbone_v1/s14_rehearsal.py",
    "campaigns/dogbone_v1/s14_validate.py",
    "src/piginot/verification/__init__.py",
    "src/piginot/verification/budget.py",
    "src/piginot/verification/confidence.py",
    "src/piginot/verification/envelope.py",
    "src/piginot/verification/holdout.py",
    "src/piginot/verification/latency.py",
    "src/piginot/verification/reducer.py",
    "src/piginot/verification/report.py",
    "src/piginot/verification/status.py",
)
_QUANTITY_DEFINITIONS = {
    "p_relative_l2": "relative L2 of the in-plane first-Piola stress over the "
    "case vertices (reference: recovered P2 stress)",
    "absolute_peak_error": "absolute relative error of the largest in-plane "
    "stress norm over the root vertices (right_arc without right_grip)",
    "u_relative_l2": "relative L2 of the displacement over the case vertices",
}


def text_sha256(path: Path) -> str:
    """SHA-256 of a text file with line endings normalised to LF."""

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_digests(root: Path = _ROOT) -> dict[str, str]:
    """Digests of every evaluation source file."""

    return {name: text_sha256(root / name) for name in EVALUATION_FILES}


def _rehearsal_summary(rehearsal: Mapping[str, object]) -> dict[str, object]:
    budget = rehearsal["budget"]
    pilot = rehearsal["confidence_pilot"]
    assert isinstance(budget, Mapping) and isinstance(pilot, Mapping)
    worst: dict[str, float] = {}
    for row in budget["gated"]:
        quantity = row["quantity"]
        worst[quantity] = max(worst.get(quantity, 0.0), row["statistic"])
    return {
        "case_count": rehearsal["case_count"],
        "gated_status": budget["gated_status"],
        "largest_gated_slice_statistic": worst,
        "pooled_spearman": pilot["pooled_spearman"],
        "selected_variant": pilot["selected_variant"],
    }


def build_protocol(
    *,
    rehearsal_bytes: bytes,
    revision: str,
    created_utc: str,
    rehearsal_revision: str | None = None,
) -> dict[str, object]:
    """The protocol document; the indicator variant comes from the pilot."""

    rehearsal = json.loads(rehearsal_bytes)
    if rehearsal.get("evidence_role") != "TRAIN" or rehearsal.get("development_opened"):
        raise ValueError("the rehearsal must be TRAIN-only")
    if rehearsal.get("case_count") != 64:
        raise ValueError("the rehearsal must cover the 64 TRAIN-screen cases")
    pilot = rehearsal["confidence_pilot"]
    variant = pilot["selected_variant"]
    if variant not in INDICATOR_VARIANTS:
        raise ValueError("the pilot selected an unregistered variant")
    refusal = rehearsal["refusal"]
    candidate = frozen_candidate_record()
    commitment = holdout_commitment()
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": "1.0",
        "status": "FROZEN_PENDING_USER_SIGN_OFF",
        "created_utc": created_utc,
        "decision_records": [
            "docs/adr/0019-s14-verification-decisions.md",
            "docs/adr/0020-s14-protocol-methods.md",
        ],
        "frozen_candidate": {
            "cp_freeze_sha256": text_sha256(CP_FREEZE_PATH),
            "checkpoint_manifest_hash": candidate["checkpoint_manifest_hash"],
            "weights_hash": candidate["weights_hash"],
            "run_hash": candidate["run_hash"],
        },
        "evaluation_code": {
            "revision": revision,
            "file_sha256": file_digests(),
            "line_endings": "LF-normalised before hashing",
        },
        "populations": {
            "development": {
                "count": 64,
                "source": "pinned S10 selective source, DEVELOPMENT case list",
                "access_record": DEVELOPMENT_ACCESS_NAME,
                "opened_only_after": "user sign-off of this protocol",
            },
            "holdout": {
                "count": 64,
                "commitment": commitment.commitment,
                "campaign_hash": commitment.campaign_hash,
                "consumption_record": HOLDOUT_CONSUMPTION_NAME,
                "reveal": f"pi_ginot_private_reveals/{REVEAL_RELATIVE_PATH.as_posix()}",
                "order": "consumption record written and fsynced, then the "
                "reveal is read and must open the CP-ENVELOPE commitment",
                "solver": "unchanged S10 P2 request recipe and execution profile",
                "opened_only_after": "DEVELOPMENT results and calibration are "
                "written, and the DEVELOPMENT budget passes on every gated slice "
                "and the pooled Spearman correlation meets the threshold; "
                "otherwise HOLDOUT stays unspent",
            },
        },
        "budget": {
            "quantities": {
                t.quantity: {
                    "tolerance": t.tolerance,
                    "definition": _QUANTITY_DEFINITIONS[t.quantity],
                }
                for t in TOLERANCES
            },
            "gated_population": "DEVELOPMENT",
            "claim_population": "HOLDOUT (sets HOLDOUT_VERIFIED or "
            "DEVELOPMENT_VERIFIED when every required check passes)",
            "gated_slices": list(GATED_SLICES),
            "reported_slices": list(REPORTED_SLICES[1:]),
            "slice_statistic": {
                "quantile": QUANTILE,
                "interpolation": "linear between order statistics",
                "minimum_cases_for_quantile": MINIMUM_QUANTILE_CASES,
                "otherwise": "maximum",
            },
            "per_case_total": "surrogate + discretisation + projection",
            "missing_component": "NOT_RUN, never zero",
            "components": {
                "surrogate": "the unchanged S13c metric code at the vertices",
                "discretisation": {
                    "p_relative_l2": "max(2.61 eta_all, 1.97 d_P)",
                    "absolute_peak_error": "max(2.61 eta_root, 1.97 d_peak)",
                    "u_relative_l2": "1.97 d_u",
                },
                "projection": {
                    "p_relative_l2": "eta_all",
                    "absolute_peak_error": "eta_root",
                    "u_relative_l2": "0 (vertex values are P2 degrees of freedom)",
                },
                "eta": "|P_recovered - P_raw| / |P_raw|, quadrature L2 of the "
                "in-plane block; all cells (eta_all) or the cells incident to a "
                "root vertex (eta_root)",
                "d": "relative difference between the S10 solve and one solve on "
                "the half-resolution level mesh, at the coarse vertices",
                "recovery_safety_factor": RECOVERY_SAFETY_FACTOR,
                "richardson_safety_factor": RICHARDSON_SAFETY_FACTOR,
                "richardson_order": RICHARDSON_ORDER,
                "refinement_ratio": REFINEMENT_RATIO,
                "coarse_level": "18 + 10 columns, 8 rows (S10: 36 + 20, 16)",
            },
        },
        "confidence": {
            "indicator": variant,
            "variants": list(INDICATOR_VARIANTS),
            "residual": "|Div P| * length_scale / E of the served stress, by "
            "automatic differentiation",
            "length_scale_mm": INDICATOR_LENGTH_SCALE_MM,
            "error": "Frobenius norm of the in-plane first-Piola error at the "
            "vertex divided by 760 MPa",
            "statistic": "Spearman correlation, all DEVELOPMENT vertices pooled",
            "threshold": SPEARMAN_MINIMUM,
            "calibration": {
                "method": "equal-count bins, per-bin quantile, pool-adjacent-"
                "violators isotonic fit",
                "quantile": CALIBRATION_QUANTILE,
                "bins": CALIBRATION_BINS,
                "fitted_on": "DEVELOPMENT",
                "coverage_reported_on": "HOLDOUT",
            },
            "train_pilot": {
                "pooled_spearman": pilot["pooled_spearman"],
                "median_per_case_spearman": pilot["median_per_case_spearman"],
            },
        },
        "refusal": {
            "fixture_set_hash": refusal["fixture_set_hash"],
            "fixture_counts": dict(FIXTURE_COUNTS),
            "required_refusal": "100% with the expected reason code",
            "maximum_false_refusal_rate": FALSE_REFUSAL_MAXIMUM,
            "in_domain_count": refusal["in_domain_count"],
            "reason_codes": "identical on two consecutive runs",
        },
        "latency": {
            "cases": f"the first {LATENCY_CASES} DEVELOPMENT cases",
            "query_set": "every mesh vertex of the case",
            "threads": 1,
            "model": "displacement only, and displacement with stress",
            "repeats": LATENCY_REPEATS,
            "warmup": 3,
            "fem": "re-solve of the same S10 request on one thread",
            "statistics": "median and p99 of the operator; median FEM time; "
            "speed-up of the median",
            "speedup_target": SPEEDUP_TARGET,
            "decision": "measured and reported",
        },
        "gate": {
            "required": list(REQUIRED_CHECKS),
            "reported": list(REPORTED_CHECKS),
            "coverage_minimum_percent": COVERAGE_MINIMUM,
            "verdict": "PASS only if every required check passes; FAIL, ERROR "
            "and NOT_RUN all leave the claim at NOT_VERIFIED",
            "claim_ceiling": "HOLDOUT_VERIFIED",
        },
        "execution": {
            "scoring_threads": SCORING_THREADS,
            "fem_threads": 1,
            "dtype": "float64",
        },
        "train_rehearsal": {
            "artifact": REHEARSAL_PATH.relative_to(_ROOT).as_posix(),
            "sha256": hashlib.sha256(rehearsal_bytes).hexdigest(),
            "code_revision": rehearsal_revision or revision,
            "summary": _rehearsal_summary(rehearsal),
        },
        "out_of_scope": [
            "retraining or any change to the frozen candidate",
            "threshold or method changes after DEVELOPMENT is opened",
            "any claim above HOLDOUT_VERIFIED",
        ],
    }


def protocol_bytes(document: Mapping[str, object]) -> bytes:
    return (json.dumps(document, indent=1, sort_keys=True) + "\n").encode()


def verify_frozen_files(
    document: Mapping[str, object], root: Path = _ROOT
) -> tuple[str, ...]:
    """Evaluation files whose digest differs from the frozen protocol."""

    code = document["evaluation_code"]
    assert isinstance(code, Mapping)
    frozen = code["file_sha256"]
    assert isinstance(frozen, Mapping)
    current = file_digests(root)
    return tuple(
        name
        for name in sorted(set(frozen) | set(current))
        if frozen.get(name) != current.get(name)
    )


def amendment_paths(directory: Path | None = None) -> tuple[Path, ...]:
    """Every signed amendment of the protocol, in order."""

    root = ARTIFACTS if directory is None else directory
    return tuple(sorted(root.glob(AMENDMENT_GLOB)))


def _verified_amendments(
    paths: tuple[Path, ...], base_digest: str
) -> tuple[dict[str, object], ...]:
    amendments: list[dict[str, object]] = []
    for path in paths:
        amendment = json.loads(path.read_text(encoding="utf-8"))
        if amendment.get("base_protocol_sha256") != base_digest:
            raise PermissionError(f"{path.name} amends a different protocol")
        if amendment.get("signed_off_by") != SIGNATORY:
            raise PermissionError(f"{path.name} is not signed by the decision owner")
        if not isinstance(amendment.get("evaluation_file_sha256"), dict):
            raise PermissionError(f"{path.name} does not freeze the evaluation files")
        amendments.append(amendment)
    return tuple(amendments)


def load_signed_protocol(
    protocol_path: Path | None = None,
    signoff_path: Path | None = None,
    amendments: tuple[Path, ...] | None = None,
) -> tuple[dict[str, object], str]:
    """The protocol and its digest, only if signed off and still frozen.

    Signed amendments (``S14-PROTOCOL-AMENDMENT-*.json``) name the base
    protocol digest and re-freeze the evaluation files; the latest one sets
    the frozen digests. The returned digest stays the base protocol digest, to
    which every recorded phase is bound, and the document lists the
    amendments under ``amendments``.
    """

    protocol_path = PROTOCOL_PATH if protocol_path is None else protocol_path
    signoff_path = SIGNOFF_PATH if signoff_path is None else signoff_path
    if not signoff_path.is_file():
        raise PermissionError("S14-PROTOCOL.json has not been signed off")
    digest = text_sha256(protocol_path)
    signoff = json.loads(signoff_path.read_text(encoding="utf-8"))
    if signoff.get("protocol_sha256") != digest:
        raise PermissionError("the sign-off names a different protocol digest")
    if signoff.get("signed_off_by") != SIGNATORY:
        raise PermissionError("the protocol is not signed off by the decision owner")
    document = json.loads(protocol_path.read_text(encoding="utf-8"))
    verified = _verified_amendments(
        amendment_paths() if amendments is None else amendments, digest
    )
    frozen = (
        {"evaluation_code": {"file_sha256": verified[-1]["evaluation_file_sha256"]}}
        if verified
        else document
    )
    changed = verify_frozen_files(frozen)
    if changed:
        raise PermissionError(
            "evaluation files changed after the freeze: " + ", ".join(changed)
        )
    document["amendments"] = list(verified)
    return document, digest


__all__ = (
    "AMENDMENT_GLOB",
    "EVALUATION_FILES",
    "LATENCY_CASES",
    "LATENCY_REPEATS",
    "PROTOCOL_ID",
    "PROTOCOL_PATH",
    "REHEARSAL_PATH",
    "SIGNATORY",
    "SIGNOFF_PATH",
    "amendment_paths",
    "build_protocol",
    "file_digests",
    "load_signed_protocol",
    "protocol_bytes",
    "text_sha256",
    "verify_frozen_files",
)
