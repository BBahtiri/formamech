"""Benchmark tables of both campaigns and their reproduction (G17.example2-3).

``table`` collects one benchmark table per family from the committed campaign
artifacts: the TRAIN-screen metrics of the frozen candidate, the DEVELOPMENT
and HOLDOUT error budgets (all-cases slice), the confidence statistics and the
latency. It writes ``artifacts/benchmarks/BENCHMARKS.json`` and
``docs/benchmarks.md``.

``reproduce`` re-derives a family's benchmark from its pinned seeds:

- **replay**: the registered trial is re-run for its first 400 optimizer
  steps from the pinned seeds in a checkout of the training revision,
  interrupted and resumed at step 200; the model, state and manifest files of
  step 400 must equal the registered run's bit for bit;
- **re-scoring**: the frozen candidate is scored again on the TRAIN-screen
  cases with the committed evaluation code; every TRAIN-screen metric of the
  table must reproduce within a relative tolerance of 1e-9.

DEVELOPMENT and HOLDOUT are not re-opened: HOLDOUT is consumed once, so its
numbers are reported from the committed result and not re-measured.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from campaigns.history import resolve_revision
from piginot.verification.amortization import amortized_break_even_queries

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = Path("artifacts/benchmarks")
TABLE_PATH = OUTPUT / "BENCHMARKS.json"
DOCS_PATH = Path("docs/benchmarks.md")
TOLERANCE = 1.0e-9
REPLAY_STEP = 400
METRICS = (
    "median_absolute_peak_error",
    "p90_absolute_peak_error",
    "median_signed_peak_error",
    "p_relative_l2",
    "u_relative_l2",
    "invalid_collocation_points",
)
FAMILIES: dict[str, dict[str, str]] = {
    "dogbone-v1": {
        "artifacts": "artifacts/campaigns/dogbone-v1",
        "screen": "S13C-TRAIN-SCREEN.json",
        "rehearsal": "S14-TRAIN-REHEARSAL.json",
        "development": "S14-DEVELOPMENT.json",
        "holdout": "S14-HOLDOUT.json",
        "gate": "artifacts/gates/S14.json",
        "bundle_gate": "artifacts/gates/S15.json",
        "train_module": "campaigns.dogbone_v1.s13c_trial",
    },
    "notched-v1": {
        "artifacts": "artifacts/campaigns/notched-v1",
        "screen": "S13-TRAIN-SCREEN.json",
        "rehearsal": "S14-TRAIN-REHEARSAL.json",
        "development": "S14-DEVELOPMENT.json",
        "holdout": "S14-HOLDOUT.json",
        "gate": "artifacts/campaigns/notched-v1/S14-GATE.json",
        "bundle_gate": "artifacts/campaigns/notched-v1/gates/S15.json",
        "train_module": "campaigns.notched_v1.s13",
    },
}


def _load(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return document


def _all_rows(result: Mapping[str, Any]) -> dict[str, float]:
    return {
        row["quantity"]: row["statistic"]
        for row in result["budget"]["gated"]
        if row["slice"] == "all"
    }


def family_table(family: str) -> dict[str, Any]:
    """The benchmark row set of one family from its committed artifacts.

    HOLDOUT is ``None`` when it was never opened (the protocol keeps it
    unspent when DEVELOPMENT fails); the row then says so instead of a number.
    """

    spec = FAMILIES[family]
    artifacts = REPOSITORY / spec["artifacts"]
    screen = _load(artifacts / spec["screen"])
    metrics = {
        k: v for k, v in screen["trial_result"]["metrics"].items() if k in METRICS
    }
    rehearsal = _load(artifacts / spec["rehearsal"])
    development = _load(artifacts / spec["development"])
    holdout_path = artifacts / spec["holdout"]
    holdout = _load(holdout_path) if holdout_path.is_file() else None
    gate = _load(REPOSITORY / spec["gate"])
    latency = development["latency"]
    amortization = amortized_break_even_queries(
        training_wall_seconds=None,
        calibration_cases=int(development["case_count"]),
        fem_seconds=float(latency["fem_median_seconds"]),
        inference_seconds=float(latency["model_u_and_p_median_seconds"]),
        threads=int(latency.get("threads", 1)),
        precision="float64",
        device="cpu",
    ).document()
    amortization["query_set"] = latency.get("query_set")
    return {
        "family": family,
        "train_screen": {
            "cases": screen.get("screen_diagnostics", {}).get("case_count"),
            **metrics,
        },
        "train_rehearsal": {
            "cases": rehearsal["case_count"],
            "p90_including_fem_terms": _all_rows(rehearsal),
            "gated_status": rehearsal["budget"]["gated_status"],
        },
        "development": {
            "cases": development["case_count"],
            "p90_including_fem_terms": _all_rows(development),
            "gated_status": development["budget"]["gated_status"],
            "pooled_spearman": development["confidence"]["pooled_spearman"],
            "latency_u_and_p_median_seconds": latency["model_u_and_p_median_seconds"],
            "fem_median_seconds": latency["fem_median_seconds"],
            "speedup": latency["speedup_u_and_p"],
            "amortization": amortization,
        },
        "holdout": None
        if holdout is None
        else {
            "cases": holdout["case_count"],
            "p90_including_fem_terms": _all_rows(holdout),
            "gated_status": holdout["budget"]["gated_status"],
            "pooled_spearman": holdout["confidence"]["pooled_spearman"],
            "calibration_coverage": holdout["confidence"]["calibration_coverage"],
        },
        "claim_level": gate.get("claim_level")
        or gate.get("report", {}).get("claim_level"),
        "development_budget_decision": _budget_decision(gate),
        "sources": {
            "screen": f"{spec['artifacts']}/{spec['screen']}",
            "gate": spec["gate"],
        },
    }


def _budget_decision(gate: Mapping[str, Any]) -> str | None:
    """The recorded reason of the G14 budget check (for example a signed waiver)."""

    checks = gate.get("checks") or gate.get("report", {}).get("checks", [])
    for check in checks:
        if check.get("id") == "G14.budget" and check.get("reason"):
            return str(check["reason"])
    return None


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render(table: Mapping[str, Any]) -> str:
    lines = [
        "# Benchmarks",
        "",
        "One table per family, generated by `python -m campaigns.benchmarks table` "
        "from the committed campaign artifacts. Errors are relative; the "
        "TRAIN-rehearsal, DEVELOPMENT and HOLDOUT rows are the p90 over all cases of the S14 "
        "per-case budget (surrogate + FEM discretisation + projection), against "
        "the tolerances 5% (stress field), 5% (peak) and 2% (displacement). The "
        "Budget column is the status over every gated slice (all cases, and each "
        "stratum of each input), not only the all-cases numbers shown.",
        "",
    ]
    for row in table["families"]:
        screen, rehearsal, development, holdout = (
            row["train_screen"],
            row["train_rehearsal"],
            row["development"],
            row["holdout"],
        )
        lines += [
            f"## {row['family']}",
            "",
            f"Claim level: `{row['claim_level']}`.",
            "",
            "| Population | Stress field | Peak | Displacement | Spearman | Budget |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| TRAIN-screen ({screen['cases']} cases, surrogate medians) | "
            f"{_percent(screen['p_relative_l2'])} | "
            f"{_percent(screen['median_absolute_peak_error'])} | "
            f"{_percent(screen['u_relative_l2'])} | — | — |",
        ]
        populations = (
            ("TRAIN rehearsal", rehearsal),
            ("DEVELOPMENT", development),
            ("HOLDOUT", holdout),
        )
        for name, population in populations:
            if population is None:
                lines.append(
                    f"| {name} | not opened: DEVELOPMENT failed the gated budget, "
                    "so HOLDOUT stays unspent | — | — | — | — |"
                )
                continue
            p90 = population["p90_including_fem_terms"]
            spearman = population.get("pooled_spearman")
            lines.append(
                f"| {name} ({population['cases']} cases, p90 with FEM terms) | "
                f"{_percent(p90['p_relative_l2'])} | "
                f"{_percent(p90['absolute_peak_error'])} | "
                f"{_percent(p90['u_relative_l2'])} | "
                f"{'—' if spearman is None else f'{spearman:.3f}'} | "
                f"{population['gated_status']} |"
            )
        coverage = (
            "HOLDOUT was not opened, so the confidence bound has no HOLDOUT "
            "calibration measurement. "
            if holdout is None
            else "HOLDOUT calibration coverage of the 90% confidence bound: "
            f"{_percent(holdout['calibration_coverage'])}. "
        )
        decision = row.get("development_budget_decision")
        if decision:
            lines += ["", f"DEVELOPMENT budget: {decision}."]
        lines += [
            "",
            f"{coverage}Online latency (u and P at every "
            f"vertex, one thread, float64 CPU): "
            f"{development['latency_u_and_p_median_seconds']:.3f} s "
            f"against {development['fem_median_seconds']:.1f} s for FEM, an "
            f"unamortized online speed-up of {development['speedup']:.0f}x.",
        ]
        amort = development.get("amortization")
        if isinstance(amort, Mapping):
            cal_lb = float(amort["calibration_only_break_even_queries"])
            n_cal = int(amort["calibration_cases"])
            n_star = amort.get("break_even_queries")
            qset = amort.get("query_set") or "mesh vertices"
            if n_star is None:
                lines.append(
                    f"Amortized break-even query count `N* = (T_train + N_cal * T_FEM) "
                    f"/ (T_FEM - T_infer)` ({qset}, {amort.get('precision', 'float64')} "
                    f"{amort.get('device', 'cpu')}, {amort.get('threads', 1)} thread): "
                    f"at least **{cal_lb:.1f} queries** to amortize the {n_cal} "
                    f"DEVELOPMENT calibration FEM solves alone (legacy S13 artifacts "
                    f"did not record offline training wall time `T_train`, which adds "
                    f"`T_train / (T_FEM - T_infer)` additional queries)."
                )
            else:
                lines.append(
                    f"Amortized break-even query count `N*` ({qset}, "
                    f"{amort.get('precision', 'float64')} {amort.get('device', 'cpu')}, "
                    f"{amort.get('threads', 1)} thread): **{float(n_star):.1f} queries** "
                    f"(including {float(amort['training_wall_seconds']):.1f} s training "
                    f"and {n_cal} calibration FEM solves)."
                )
        lines.append("")
    lines += [
        "## Reproduction",
        "",
        "`python -m campaigns.benchmarks reproduce --family F ...` replays the first "
        f"{REPLAY_STEP} optimizer steps of the registered trial from its pinned seeds "
        "in a checkout of the training revision (interrupted and resumed at step "
        f"{REPLAY_STEP // 2}) and compares the step-{REPLAY_STEP} files with the "
        "registered run bit for bit; it then re-scores the frozen candidate on the "
        "TRAIN-screen cases and compares every TRAIN-screen metric with the table "
        f"(relative tolerance {TOLERANCE:g}). DEVELOPMENT and HOLDOUT are not "
        "re-opened: HOLDOUT is consumed once, so its row is reported, not re-measured.",
        "",
    ]
    reproduction = table.get("reproduction") or {}
    if reproduction:
        lines += [
            "| Family | Replay bitwise | TRAIN-screen within tolerance |",
            "| --- | --- | --- |",
        ]
        for family, result in sorted(reproduction.items()):
            lines.append(
                f"| {family} | {result['replay']['bitwise']} | {result['rescore']['within_tolerance']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _rescore(
    family: str, s10_run_root: Path, registered_run_root: Path
) -> dict[str, float]:
    checkpoint = registered_run_root / "checkpoint-step-2400"
    if family == "dogbone-v1":
        from campaigns.dogbone_v1.s13c_evaluate import evaluate_s13c_checkpoint

        result = evaluate_s13c_checkpoint(
            s10_run_root=s10_run_root, checkpoint_path=checkpoint
        ).result
    else:
        from campaigns.notched_v1.s13_close import evaluate_screen

        result = evaluate_screen(
            s10_run_root=s10_run_root, checkpoint_path=checkpoint
        ).result
    return {name: float(getattr(result.metrics, name)) for name in METRICS}


def _replay(
    family: str,
    worktree: Path,
    s10_run_root: Path,
    registered_run_root: Path,
    scratch: Path,
) -> dict[str, Any]:
    from campaigns.dogbone_v1.s13c_evaluate import load_run_binding

    binding = load_run_binding(registered_run_root / "run-binding.json")
    head = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    expected = resolve_revision(binding.implementation_revision, worktree)
    if head != expected:
        raise RuntimeError(
            f"the replay checkout is at {head}, not {expected} "
            f"(recorded as {binding.implementation_revision})"
        )
    run_root = scratch / f"replay-{family}"
    command = [sys.executable, "-m", FAMILIES[family]["train_module"]]
    if family == "dogbone-v1":
        command += ["--s10-run-root", str(s10_run_root)]
    else:
        command += ["train"]
    command += [
        "--run-root",
        str(run_root),
        "--segment-steps",
        str(REPLAY_STEP // 2),
        "--stop-after",
        str(REPLAY_STEP),
    ]
    environment = {**os.environ, "PYTHONPATH": "src:."}
    completed = subprocess.run(
        command, cwd=worktree, env=environment, capture_output=True, text=True
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr[-2000:])
    files = ("model.safetensors", "state.safetensors", "checkpoint.json")
    registered = registered_run_root / f"checkpoint-step-{REPLAY_STEP:04d}"
    replayed = run_root / f"checkpoint-step-{REPLAY_STEP:04d}"
    identical = {
        name: (registered / name).read_bytes() == (replayed / name).read_bytes()
        for name in files
    }
    return {
        "training_revision": head,
        "run_hash": binding.run_hash,
        "steps": REPLAY_STEP,
        "interrupted_and_resumed_at": REPLAY_STEP // 2,
        "byte_identical_files": identical,
        "bitwise": all(identical.values()),
    }


def reproduce(
    family: str, *, worktree: Path, s10_run_root: Path, registered_run_root: Path
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="piginot-benchmark-") as directory:
        replay = _replay(
            family, worktree, s10_run_root, registered_run_root, Path(directory)
        )
    spec = FAMILIES[family]
    committed = _load(REPOSITORY / spec["artifacts"] / spec["screen"])["trial_result"][
        "metrics"
    ]
    measured = _rescore(family, s10_run_root, registered_run_root)
    differences = {
        name: abs(measured[name] - float(committed[name]))
        / max(abs(float(committed[name])), 1e-300)
        if committed[name]
        else abs(measured[name])
        for name in METRICS
    }
    return {
        "family": family,
        "replay": replay,
        "rescore": {
            "measured": measured,
            "committed": {name: committed[name] for name in METRICS},
            "relative_difference": differences,
            "tolerance": TOLERANCE,
            "within_tolerance": all(
                math.isfinite(v) and v <= TOLERANCE for v in differences.values()
            ),
        },
        "passed": replay["bitwise"]
        and all(math.isfinite(v) and v <= TOLERANCE for v in differences.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("table")
    again = sub.add_parser("reproduce")
    again.add_argument("--family", choices=sorted(FAMILIES), required=True)
    again.add_argument("--worktree", type=Path, required=True)
    again.add_argument("--s10-run-root", type=Path, required=True)
    again.add_argument("--registered-run-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    (REPOSITORY / OUTPUT).mkdir(parents=True, exist_ok=True)
    if arguments.command == "reproduce":
        result = reproduce(
            arguments.family,
            worktree=arguments.worktree.resolve(),
            s10_run_root=arguments.s10_run_root.resolve(),
            registered_run_root=arguments.registered_run_root.resolve(),
        )
        target = REPOSITORY / OUTPUT / f"REPRODUCTION-{arguments.family}.json"
        target.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"family": arguments.family, "passed": result["passed"]}))
        return 0 if result["passed"] else 1
    document: dict[str, Any] = {
        "families": [
            family_table("dogbone-v1"),
            family_table("notched-v1"),
        ],
        "reproduction": {
            family: _load(REPOSITORY / OUTPUT / f"REPRODUCTION-{family}.json")
            for family in FAMILIES
            if (REPOSITORY / OUTPUT / f"REPRODUCTION-{family}.json").exists()
        },
    }
    (REPOSITORY / TABLE_PATH).write_text(
        json.dumps(document, indent=1, sort_keys=True) + "\n"
    )
    (REPOSITORY / DOCS_PATH).write_text(render(document))
    print(f"wrote {TABLE_PATH} and {DOCS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
