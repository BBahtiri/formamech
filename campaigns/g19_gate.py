"""G19: the S19+ fork is isolated and reproduces the v0.1.0 baseline.

Checks (``STEPS-v2.0.md`` §S19):

- ``G19.isolation``: no remote, no object alternates, ``.git`` is a directory,
  exactly one worktree;
- ``G19.ancestry``: ``HEAD`` descends from the fork base, and ``v0.1.0`` and
  ``v0.2.0-base`` both resolve to it;
- ``G19.identity``: every commit after the fork base, and the configured
  committer, is the decision owner;
- ``G19.quality``: ``ruff check``, ``ruff format --check`` and
  ``mypy --strict src/`` are clean;
- ``G19.suite``: the recorded full-suite run on the fork-base tree passed at
  least as many tests as S18 and failed none;
- ``G19.coverage``: a recorded full-suite coverage run on the fork-base tree
  keeps the S18 branch coverage of ``src/piginot`` (91%, to the integer);
- ``G19.manifests``: ``bundles/`` and ``artifacts/`` equal the fork base, in Git
  and byte for byte in the working tree (BLAKE2b manifest);
- ``G19.bundles``: both shipped bundles load with their recorded manifests and
  claim levels, and serve the example requests byte-identically twice and
  against the golden recorded for this execution profile;
- ``G19.resume``: a clean-checkout S13c run stopped at 200 and resumed to 400
  is byte-identical to a straight 0 -> 400 run on the same execution profile
  (ADR 0100 decision 5); the registered probe hash is reported alongside;
- ``G19.s13c-rescore`` and ``G19.resume`` need the ignored S10 run root and the
  registered S13c run. Without them they are ``NOT_RUN`` and block the gate.

Run from the repository root::

    python -m campaigns.g19_gate --write
    python -m campaigns.g19_gate --write \\
        --s13c-evaluation rescore.json \\
        --replay-checkpoint <replay>/checkpoint-step-0400 \\
        --straight-checkpoint <straight>/checkpoint-step-0400
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from campaigns.notched_v1.gate_io import (
    check_document,
    git_head,
    plain,
    utc_now,
    write_gate_document,
)
from piginot.git_environment import isolated_git_environment

REPOSITORY = Path(__file__).resolve().parents[1]
BASE = "399a14659cec6812e24980ece8f81ab2d94a4753"
BASE_TAGS = ("v0.1.0", "v0.2.0-base")
SIGNATORY = "Betim Bahtiri"
GATE_PATH = Path("artifacts/gates/S19.json")
FORK_DIR = Path("artifacts/fork")
SUITE_PATH = FORK_DIR / "S19-SUITE.json"
COVERAGE_PATH = FORK_DIR / "S19-COVERAGE.json"
MANIFEST_PATH = FORK_DIR / "S19-BASE-MANIFEST.json"
GOLDEN_PATH = FORK_DIR / "S19-BUNDLE-OUTPUTS.json"
MANIFEST_INCLUDE = ("artifacts", "bundles")
# S18 recorded "3,075 passed, 1 skipped, 91%" (coverage rounded to the integer).
S18_SUITE = {"passed": 3075, "coverage_percent": 90.5}
SCREEN_PATH = Path("artifacts/campaigns/dogbone-v1/S13C-TRAIN-SCREEN.json")
PROBE_PATH = Path("artifacts/campaigns/dogbone-v1/S13C-G13-REPLAY-PROBE.json")
EXAMPLES = (
    ("dogbone-v1-s13c-cp-freeze", "dogbone-in-domain.json"),
    ("notched-v1-cp-freeze", "notched-in-domain.json"),
    ("notched-v1-cp-freeze", "notched-out-of-envelope.json"),
)

Runner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


def _run(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=isolated_git_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


def _git(root: Path, *arguments: str, runner: Runner = _run) -> str:
    return runner(["git", "-C", str(root), *arguments], root).stdout.strip()


def _fork_added(path: str) -> bool:
    return path == GATE_PATH.as_posix() or path.startswith(FORK_DIR.as_posix() + "/")


def read_text_any(path: Path) -> str:
    """Text of a file written by any shell (PowerShell 5.1 writes UTF-16)."""

    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


def not_run(check_id: str, reason: str, threshold: object) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "NOT_RUN",
        "measured": {"reason": reason},
        "threshold": plain(threshold),
        "comparison": "all",
    }


def check_isolation(root: Path, runner: Runner = _run) -> dict[str, Any]:
    remotes = _git(root, "remote", runner=runner)
    worktrees = [
        line
        for line in _git(
            root, "worktree", "list", "--porcelain", runner=runner
        ).splitlines()
        if line.startswith("worktree ")
    ]
    measured = {
        "remotes": remotes.split(),
        "alternates": (root / ".git/objects/info/alternates").exists(),
        "git_is_directory": (root / ".git").is_dir(),
        "worktrees": len(worktrees),
    }
    passed = (
        not measured["remotes"]
        and not measured["alternates"]
        and measured["git_is_directory"]
        and measured["worktrees"] == 1
    )
    return check_document(
        "G19.isolation",
        passed,
        measured,
        "no remote, no alternates, .git a directory, one worktree",
        "all",
    )


def check_ancestry(root: Path, runner: Runner = _run) -> dict[str, Any]:
    ancestor = (
        runner(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", BASE, "HEAD"], root
        ).returncode
        == 0
    )
    tags = {
        tag: _git(root, "rev-parse", f"{tag}^{{commit}}", runner=runner)
        for tag in BASE_TAGS
    }
    passed = ancestor and all(value == BASE for value in tags.values())
    return check_document(
        "G19.ancestry",
        passed,
        {"head_descends_from_base": ancestor, "tags": tags},
        {"base": BASE, "tags_resolve_to_base": list(BASE_TAGS)},
        "all",
    )


def check_identity(root: Path, runner: Runner = _run) -> dict[str, Any]:
    authors = sorted(
        set(
            _git(root, "log", "--format=%an", f"{BASE}..HEAD", runner=runner).split(
                "\n"
            )
        )
        - {""}
    )
    committer = _git(root, "config", "user.name", runner=runner)
    passed = committer == SIGNATORY and authors in ([], [SIGNATORY])
    return check_document(
        "G19.identity",
        passed,
        {"authors_after_base": authors, "configured_user": committer},
        f"every commit after the base and the configured user are {SIGNATORY}",
        "all",
    )


def check_quality(root: Path, runner: Runner = _run) -> dict[str, Any]:
    python = sys.executable
    commands = {
        "ruff_check": [python, "-m", "ruff", "check", "."],
        "ruff_format": [python, "-m", "ruff", "format", "--check", "."],
        "mypy_strict": [python, "-m", "mypy", "--strict", "src/"],
    }
    results = {}
    for name, command in commands.items():
        completed = runner(command, root)
        tail = (completed.stdout.strip().splitlines() or [""])[-1]
        results[name] = {"returncode": completed.returncode, "summary": tail}
    passed = all(result["returncode"] == 0 for result in results.values())
    return check_document("G19.quality", passed, results, "all three exit 0", "all")


def check_suite(root: Path) -> dict[str, Any]:
    path = root / SUITE_PATH
    threshold = {
        "passed_at_least": S18_SUITE["passed"],
        "failed": 0,
        "errors": 0,
        "tree_of": BASE,
    }
    if not path.is_file():
        return not_run("G19.suite", f"{SUITE_PATH} not recorded", threshold)
    suite = json.loads(path.read_text("utf-8"))
    passed = (
        suite["tree"] == _git(root, "rev-parse", f"{BASE}^{{tree}}")
        and suite["passed"] >= S18_SUITE["passed"]
        and suite["failed"] == 0
        and suite["errors"] == 0
    )
    return check_document(
        "G19.suite", passed, suite, threshold, "all", evidence=SUITE_PATH.as_posix()
    )


def check_coverage(root: Path) -> dict[str, Any]:
    path = root / COVERAGE_PATH
    threshold = {
        "branch_coverage_percent_at_least": S18_SUITE["coverage_percent"],
        "package": "src/piginot",
        "tree_of": BASE,
    }
    if not path.is_file():
        return not_run(
            "G19.coverage",
            f"{COVERAGE_PATH} not recorded (run the full suite with coverage "
            "on a multi-core machine, see docs/fork/S19-FORK.md)",
            threshold,
        )
    coverage = json.loads(path.read_text("utf-8"))
    passed = (
        coverage["tree"] == _git(root, "rev-parse", f"{BASE}^{{tree}}")
        and coverage["clean"] is True
        and coverage["branch_coverage"] is True
        and coverage["coverage_percent"] >= S18_SUITE["coverage_percent"]
        and coverage["failed"] == 0
        and coverage["errors"] == 0
    )
    return check_document(
        "G19.coverage",
        passed,
        coverage,
        threshold,
        "all",
        evidence=COVERAGE_PATH.as_posix(),
    )


def _verify_fork_tool(root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_fork", root / "scripts" / "verify_fork.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_manifests(root: Path, runner: Runner = _run) -> dict[str, Any]:
    changed = [
        path
        for path in _git(
            root,
            "diff",
            "--name-only",
            BASE,
            "HEAD",
            "--",
            *MANIFEST_INCLUDE,
            runner=runner,
        ).splitlines()
        if path and not _fork_added(path)
    ]
    tool = _verify_fork_tool(root)
    recorded = json.loads((root / MANIFEST_PATH).read_text("utf-8"))
    current = tool.build_manifest(root, list(MANIFEST_INCLUDE))
    current["files"] = {
        path: entry for path, entry in current["files"].items() if not _fork_added(path)
    }
    difference = tool.compare_manifests(recorded, current)
    passed = not changed and not any(difference.values())
    return check_document(
        "G19.manifests",
        passed,
        {
            "git_changes_since_base": changed,
            "working_tree_difference": difference,
            "recorded_table_blake2b": recorded["table_blake2b"],
            "recorded_file_count": recorded["file_count"],
        },
        "bundles/ and artifacts/ equal the fork base, apart from the fork's own records",
        "all",
        evidence=MANIFEST_PATH.as_posix(),
    )


def execution_profile() -> dict[str, str]:
    import torch

    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cpu_capability": str(torch.backends.cpu.get_cpu_capability()),
    }


def profile_key(profile: Mapping[str, str]) -> str:
    return "|".join(profile[name] for name in sorted(profile))


def serve_examples(root: Path) -> dict[str, Any]:
    """Load both bundles and serve every example request twice."""

    from piginot.bundle import load_bundle, predict
    from piginot.cli import parse_request

    bundles: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    for name, request_file in EXAMPLES:
        bundle = load_bundle(root / "bundles" / name)
        bundles[name] = {
            "manifest_sha256": bundle.manifest.manifest_sha256,
            "claim_level": bundle.claim_level.value,
        }
        document = json.loads(
            (root / "bundles" / "requests" / request_file).read_text("utf-8")
        )
        digests = []
        verdict = ""
        for _ in range(2):
            inputs, queries = parse_request(document)
            prediction = predict(bundle, inputs, queries)
            verdict = prediction.verdict.value
            text = json.dumps(prediction.document(), sort_keys=True, allow_nan=False)
            digests.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
        outputs[f"{name}/{request_file}"] = {
            "verdict": verdict,
            "sha256": digests[0],
            "repeat_identical": digests[0] == digests[1],
        }
    return {"bundles": bundles, "outputs": outputs}


def check_bundles(
    root: Path,
    served: Mapping[str, Any],
    profile: Mapping[str, str],
) -> dict[str, Any]:
    recorded_manifests = {
        "dogbone-v1-s13c-cp-freeze": (
            "009db9d379d6f94ccda5395762e22c9c810c466a686626d500f1575d45b1bb4c",
            "HOLDOUT_VERIFIED",
        ),
        "notched-v1-cp-freeze": (
            "3749f4be4761a55582dd07f90e3cc5950b8231357162366d79dc47463dddcc99",
            "DEVELOPMENT_VERIFIED",
        ),
    }
    identity = all(
        served["bundles"][name] == {"manifest_sha256": manifest, "claim_level": claim}
        for name, (manifest, claim) in recorded_manifests.items()
    )
    repeat = all(output["repeat_identical"] for output in served["outputs"].values())
    expected = {
        "dogbone-v1-s13c-cp-freeze/dogbone-in-domain.json": "IN_DOMAIN",
        "notched-v1-cp-freeze/notched-in-domain.json": "IN_DOMAIN",
        "notched-v1-cp-freeze/notched-out-of-envelope.json": "REFUSED",
    }
    verdicts = all(
        served["outputs"][key]["verdict"] == verdict
        for key, verdict in expected.items()
    )
    goldens = (
        json.loads((root / GOLDEN_PATH).read_text("utf-8"))["profiles"]
        if (root / GOLDEN_PATH).is_file()
        else {}
    )
    key = profile_key(profile)
    golden = goldens.get(key)
    measured = {
        **served,
        "profile": dict(profile),
        "golden_for_profile": golden is not None,
    }
    if golden is None:
        return {
            "id": "G19.bundles",
            "status": "NOT_RUN",
            "measured": plain(measured),
            "threshold": "golden outputs recorded for this execution profile",
            "comparison": "all",
        }
    matches = all(
        served["outputs"][name]["sha256"] == digest for name, digest in golden.items()
    )
    measured["golden_match"] = matches
    return check_document(
        "G19.bundles",
        identity and repeat and verdicts and matches,
        measured,
        "recorded manifests and claim levels; example verdicts; repeat and golden "
        "outputs byte-identical",
        "all",
        evidence=GOLDEN_PATH.as_posix(),
    )


def record_golden(
    root: Path, served: Mapping[str, Any], profile: Mapping[str, str]
) -> None:
    path = root / GOLDEN_PATH
    document = (
        json.loads(path.read_text("utf-8"))
        if path.is_file()
        else {"schema_version": "1.0", "profiles": {}, "recorded": {}}
    )
    key = profile_key(profile)
    document["profiles"][key] = {
        name: output["sha256"] for name, output in served["outputs"].items()
    }
    document["recorded"][key] = {"profile": dict(profile), "commit": git_head(root)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def record_coverage(
    root: Path, base_clone: Path, coverage_json: Path, pytest_log: Path
) -> dict[str, Any]:
    """Write ``COVERAGE_PATH`` from a coverage run in a clean clone of the base."""

    report = json.loads(read_text_any(coverage_json))
    summary = ""
    for line in read_text_any(pytest_log).splitlines()[::-1]:
        if re.search(r"\d+ (passed|failed)", line):
            summary = line.strip("= ").strip()
            break
    counts = {
        key: int(value)
        for value, key in re.findall(r"(\d+) (passed|failed|skipped|errors?)", summary)
    }
    dirty = _git(base_clone, "status", "--porcelain")
    document = {
        "schema_version": "1.0",
        "tree": _git(base_clone, "rev-parse", "HEAD^{tree}"),
        "commit": _git(base_clone, "rev-parse", "HEAD"),
        "clean": dirty == "",
        "coverage_percent": round(float(report["totals"]["percent_covered"]), 2),
        "branch_coverage": bool(report.get("meta", {}).get("branch_coverage", False)),
        "passed": counts.get("passed", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "errors": counts.get("errors", counts.get("error", 0)),
        "summary": summary,
        "execution_profile": execution_profile(),
    }
    path = root / COVERAGE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def _differences(
    recorded: object, current: object, path: str = ""
) -> list[tuple[str, object, object]]:
    """Every leaf where two JSON documents differ, with its path."""

    if isinstance(recorded, dict) and isinstance(current, dict):
        return [
            difference
            for key in sorted(set(recorded) | set(current))
            for difference in _differences(
                recorded.get(key), current.get(key), f"{path}.{key}"
            )
        ]
    if (
        isinstance(recorded, list)
        and isinstance(current, list)
        and len(recorded) == len(current)
    ):
        return [
            difference
            for index, (left, right) in enumerate(zip(recorded, current, strict=True))
            for difference in _differences(left, right, f"{path}[{index}]")
        ]
    return [] if recorded == current else [(path, recorded, current)]


def _relative(left: object, right: object) -> float | None:
    numbers = (int, float)
    if (
        isinstance(left, numbers)
        and isinstance(right, numbers)
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        scale = max(abs(float(left)), abs(float(right)), 1e-300)
        return abs(float(left) - float(right)) / scale
    return None


def check_rescore(root: Path, evaluation: Path | None) -> dict[str, Any]:
    """The printed ``TrialResult`` equals the recorded ``trial_result`` exactly.

    ``campaigns.dogbone_v1.s13c_evaluate`` prints the canonical ``TrialResult``
    (lineage, metrics, role) and then ``selected`` or the failed guardrails.
    ``S13C-TRAIN-SCREEN.json`` holds the same object as its ``trial_result``.
    """

    threshold = {
        "compared": f"printed TrialResult vs {SCREEN_PATH.as_posix()} trial_result",
        "equality": "exact (every leaf)",
        "decision": "selected, and the recorded selection_status is ELIGIBLE",
    }
    if evaluation is None:
        return not_run(
            "G19.s13c-rescore",
            "needs the ignored S10 run root and the registered S13c checkpoint "
            "(run campaigns.dogbone_v1.s13c_evaluate locally, pass --s13c-evaluation)",
            threshold,
        )
    recorded = json.loads((root / SCREEN_PATH).read_text("utf-8"))
    lines = [line.strip() for line in read_text_any(evaluation).splitlines()]
    payloads = [line for line in lines if line.startswith("{")]
    decisions = [line for line in lines if line and not line.startswith("{")]
    result = json.loads(payloads[0]) if payloads else None
    differences = _differences(recorded["trial_result"], result)
    relatives = [
        value
        for value in (_relative(left, right) for _, left, right in differences)
        if value is not None
    ]
    decision = decisions[0] if decisions else None
    measured = {
        "decision": decision,
        "recorded_selection_status": recorded.get("selection_status"),
        "difference_count": len(differences),
        "numeric_difference_count": len(relatives),
        "max_relative_numeric_difference": max(relatives) if relatives else 0.0,
        "differences": [
            {"path": path, "recorded": plain(left), "rescored": plain(right)}
            for path, left, right in differences[:25]
        ],
    }
    passed = (
        result is not None
        and not differences
        and decision == "selected"
        and recorded.get("selection_status") == "ELIGIBLE"
    )
    return check_document(
        "G19.s13c-rescore",
        passed,
        measured,
        threshold,
        "all",
        evidence=evaluation.name,
    )


def _checkpoint_files(checkpoint: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(checkpoint.iterdir())
        if path.is_file()
    }


def check_resume(
    root: Path, replay: Path | None, straight: Path | None = None
) -> dict[str, Any]:
    """Interrupted-and-resumed equals uninterrupted, byte for byte (ADR 0100 §5).

    Trained weights depend on the CPU instruction set (AVX2 and AVX512 diverge),
    so the check is made on the profile that runs it. The registered replay
    probe (Linux, AVX512) is compared alongside and reported, not required.
    """

    from piginot.training.checkpoint import TrainingCheckpointManifest

    probe = json.loads((root / PROBE_PATH).read_text("utf-8"))
    target = int(probe["target_step"])
    threshold = {
        "procedure": "clean checkout; stop after 200; resume to 400; and a "
        "straight 0 -> 400 run on the same profile",
        "step": target,
        "resumed_equals_straight": "every checkpoint file byte-identical",
        "registered_model_state_hash": "reported, required only on the "
        "registered profile (ADR 0100 decision 5)",
    }
    if replay is None or straight is None:
        return not_run(
            "G19.resume",
            "needs the ignored S10 run root (run campaigns.dogbone_v1.s13c_trial "
            "locally, pass --replay-checkpoint and --straight-checkpoint)",
            threshold,
        )
    manifests = {
        name: TrainingCheckpointManifest.from_bytes(
            (path / "checkpoint.json").read_bytes()
        )
        for name, path in (("replay", replay), ("straight", straight))
    }
    files = {
        "replay": _checkpoint_files(replay),
        "straight": _checkpoint_files(straight),
    }
    interrupted = (replay.parent / f"checkpoint-step-{target // 2:04d}").is_dir()
    registered = probe["registered_model_state_hash"]
    measured = {
        "replay": {
            "step": manifests["replay"].step,
            "model_state_hash": manifests["replay"].model_state_hash,
            "manifest_hash": manifests["replay"].manifest_hash,
            "run_hash": manifests["replay"].run_hash,
            "interrupted_at": target // 2 if interrupted else None,
        },
        "straight": {
            "step": manifests["straight"].step,
            "model_state_hash": manifests["straight"].model_state_hash,
            "manifest_hash": manifests["straight"].manifest_hash,
            "run_hash": manifests["straight"].run_hash,
        },
        "byte_identical_files": {
            name: files["replay"].get(name) == digest
            for name, digest in files["straight"].items()
        },
        "file_sets_equal": sorted(files["replay"]) == sorted(files["straight"]),
        "registered_profile_comparison": {
            "registered_model_state_hash": registered,
            "matches_registered": manifests["replay"].model_state_hash == registered,
            "expected_to_match": False,
            "reason": "the registered probe ran on Linux/AVX512; trained weights "
            "depend on the CPU instruction set",
        },
    }
    passed = (
        interrupted
        and manifests["replay"].step == target
        and manifests["straight"].step == target
        and measured["file_sets_equal"]
        and all(measured["byte_identical_files"].values())
    )
    return check_document("G19.resume", passed, measured, threshold, "all")


def build_gate(
    root: Path,
    *,
    s13c_evaluation: Path | None = None,
    replay_checkpoint: Path | None = None,
    straight_checkpoint: Path | None = None,
    record: bool = False,
    runner: Runner = _run,
) -> dict[str, Any]:
    profile = execution_profile()
    served = serve_examples(root)
    if record:
        record_golden(root, served, profile)
    checks = [
        check_isolation(root, runner),
        check_ancestry(root, runner),
        check_identity(root, runner),
        check_quality(root, runner),
        check_suite(root),
        check_coverage(root),
        check_manifests(root, runner),
        check_bundles(root, served, profile),
        check_rescore(root, s13c_evaluation),
        check_resume(root, replay_checkpoint, straight_checkpoint),
    ]
    blockers = [check["id"] for check in checks if check["status"] != "PASS"]
    return {
        "gate_id": "G19",
        "step": "S19",
        "commit": git_head(root),
        "timestamp_utc": utc_now(),
        "fork_base": BASE,
        "execution_profile": profile,
        "checks": checks,
        "decision_record": "docs/adr/0100-s19-fork-for-the-s19-line.md",
        "verdict": "FAIL" if blockers else "PASS",
        "verdict_reason": (
            f"{len(blockers)} required check(s) not PASS: {', '.join(blockers)}"
            if blockers
            else "every required check PASS"
        ),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate gate G19.")
    parser.add_argument("--write", action="store_true", help=f"write {GATE_PATH}")
    parser.add_argument("--record-golden", action="store_true")
    parser.add_argument("--s13c-evaluation", type=Path)
    parser.add_argument("--replay-checkpoint", type=Path)
    parser.add_argument("--straight-checkpoint", type=Path)
    parser.add_argument(
        "--record-coverage",
        nargs=3,
        type=Path,
        metavar=("BASE_CLONE", "COVERAGE_JSON", "PYTEST_LOG"),
        help=f"record {COVERAGE_PATH} from a coverage run in a clean base clone",
    )
    options = parser.parse_args(arguments)
    if options.record_coverage is not None:
        recorded = record_coverage(REPOSITORY, *options.record_coverage)
        print(json.dumps(recorded, indent=2))
    document = build_gate(
        REPOSITORY,
        s13c_evaluation=options.s13c_evaluation,
        replay_checkpoint=options.replay_checkpoint,
        straight_checkpoint=options.straight_checkpoint,
        record=options.record_golden,
    )
    if options.write:
        write_gate_document(REPOSITORY / GATE_PATH, document)
    for check in document["checks"]:
        print(f"{check['id']}: {check['status']}")
    print(f"G19: {document['verdict']} ({document['verdict_reason']})")
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
