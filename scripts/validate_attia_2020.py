"""Reproducible real-world ASME V&V 10 validation #1: Attia et al. (Nature 2020).

Downloads (if absent) and SHA-256 verifies the CC-BY-4.0 Attia et al. (2020)
battery fast-charging dataset strictly into gitignored
``.formamech-runs/datasets/matr-attia-clo/``, runs FormaMech's V5-2 hybrid
simulation + physical-test grounding and 10-seed active physical test-plan
reduction benchmarks, and emits canonical reports under
``.formamech-runs/validation/``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from piginot.active.test_planner import (
    PhysicalTestPlanConfig,
    run_retrospective_test_reduction_benchmark,
)
from piginot.benchmarks.external_data import (
    EXTERNAL_DATASET_MANIFEST_SCHEMA,
    resolve_dataset_spec,
)
from piginot.contracts import canonical_json_bytes
from piginot.validation.component_test_contracts import (
    ComponentTestRecord,
    SensorObservationKind,
)
from piginot.validation.contracts import CouponRole
from piginot.validation.hybrid_grounding import (
    HybridGroundingConfig,
    ground_simulation_with_experiments,
)

CACHE_DIR = Path(".formamech-runs/datasets/matr-attia-clo").resolve()
REPORT_DIR = Path(".formamech-runs/validation").resolve()
UPSTREAM_BASE = (
    "https://raw.githubusercontent.com/chueh-ermon/"
    "battery-fast-charging-optimization/master/"
)
UPSTREAM_FILES: dict[str, str] = {
    "policies_all.csv": "data/policies_all.csv",
    "pred_round_0.csv": "data/pred/0.csv",
    "pred_round_1.csv": "data/pred/1.csv",
    "pred_round_2.csv": "data/pred/2.csv",
    "pred_round_3.csv": "data/pred/3.csv",
    "oed_estimated_cycle_lives.csv": (
        "figures/SI/cycle%20life%20table/oed_estimated_cycle_lives.csv"
    ),
    "fig4_final_results.csv": "figures/fig4/final_results.csv",
    "fig4_predictions.csv": "figures/fig4/predictions.csv",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_local_attia_cache() -> dict[str, Any]:
    """Ensure CC-BY-4.0 Attia et al. CSVs are cached and SHA-256 verified locally."""
    spec = resolve_dataset_spec("matr-attia-clo")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE_DIR / "cache_manifest.json"
    if manifest_path.is_file() and all(
        (CACHE_DIR / name).is_file() for name in UPSTREAM_FILES
    ):
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        for fname, expected in manifest["file_sha256"].items():
            actual = _sha256_file(CACHE_DIR / fname)
            if actual != expected:
                raise ValueError(f"SHA-256 mismatch for cached file {fname}")
        return manifest

    file_hashes: dict[str, str] = {}
    for local_name, remote_rel in UPSTREAM_FILES.items():
        dest = CACHE_DIR / local_name
        with urllib.request.urlopen(UPSTREAM_BASE + remote_rel, timeout=30) as resp:
            payload = resp.read()
        dest.write_bytes(payload)
        file_hashes[local_name] = hashlib.sha256(payload).hexdigest()

    manifest = {
        "schema": EXTERNAL_DATASET_MANIFEST_SCHEMA,
        "dataset": spec.document(),
        "source_mode": "verified_local",
        "source_repo": "https://github.com/chueh-ermon/battery-fast-charging-optimization",
        "citation": (
            "Attia et al., Nature 578, 397-402 (2020), Closed-loop optimization "
            "of fast-charging protocols for batteries with machine learning"
        ),
        "license": "CC-BY-4.0",
        "file_sha256": file_hashes,
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest


def wang_2011_lfp_electrothermal_prior(
    c1: float,
    c2: float,
    c3: float,
    c4: float,
    *,
    dt_s: float = 1.0,
) -> tuple[float, float]:
    """Frozen A123 APR18650M1A electro-thermal ODE + Wang et al. (2011) SEI prior."""
    q_nom_ah = 1.1
    r0_ohm = 0.020
    m_cp_j_per_k = 39.0
    h_a_w_per_k = 0.055
    t_amb_k = 303.15
    r_gas = 8.314462618

    def _wang_b(c_rate: float) -> float:
        return float(
            30330.0 * math.exp(-0.14 * min(c_rate, 6.0))
            + 280.0 * max(c_rate - 6.0, 0.0) ** 2
        )

    t_k = t_amb_k
    t_peak_k = t_amb_k
    weighted_rate_acc = 0.0
    for step_idx, c_rate in enumerate((c1, c2, c3, c4)):
        step_dur_s = (0.20 / max(c_rate, 0.1)) * 3600.0
        n_sub = max(1, int(math.ceil(step_dur_s / dt_s)))
        h_dt = step_dur_s / n_sub
        i_amp = c_rate * q_nom_ah
        for _ in range(n_sub):
            r_eff = r0_ohm * math.exp(1200.0 * (1.0 / t_k - 1.0 / t_amb_k))
            t_k += (
                h_dt
                * ((i_amp * i_amp) * r_eff - h_a_w_per_k * (t_k - t_amb_k))
                / m_cp_j_per_k
            )
            t_peak_k = max(t_peak_k, t_k)
            ea_j = 31700.0 - 370.3 * c_rate
            k_inst = (_wang_b(c_rate) * math.exp(-ea_j / (r_gas * t_k))) ** (1.0 / 0.55)
            weighted_rate_acc += (
                (1.0 + 0.08 * step_idx) * k_inst * (i_amp * h_dt / 3600.0)
            )

    c_dis = 4.0
    ea_dis = 31700.0 - 370.3 * c_dis
    k_dis = (_wang_b(c_dis) * math.exp(-ea_dis / (r_gas * 0.5 * (t_k + t_amb_k)))) ** (
        1.0 / 0.55
    )
    mean_k_per_ah = (weighted_rate_acc + k_dis * (1.32 * q_nom_ah)) / (2.20 * q_nom_ah)
    ah_eol = (20.0 ** (1.0 / 0.55)) / max(mean_k_per_ah, 1.0e-12)
    return float(ah_eol / (2.0 * q_nom_ah)), float(t_peak_k - 273.15)


def _make_record(
    specimen_id: str,
    role: CouponRole,
    c1: float,
    c2: float,
    c3: float,
    c4: float,
    y_obs: float,
    y_sim: float,
    u_exp: float,
    source: str,
) -> ComponentTestRecord:
    return ComponentTestRecord(
        specimen_id=specimen_id,
        role=role,
        sensor_kind=SensorObservationKind.DEGRADATION_METRIC,
        condition_vector=(c1, c2, c3, c4),
        observed_values=(y_obs,),
        simulated_prior_values=(y_sim,),
        measurement_std=u_exp,
        unit="cycles",
        metadata={"source": source},
    )


def main() -> None:
    manifest = ensure_local_attia_cache()
    by_key: dict[tuple[float, float, float], list[tuple[float, float]]] = defaultdict(
        list
    )
    for r_idx in range(4):
        rows = list(
            csv.reader(
                (CACHE_DIR / f"pred_round_{r_idx}.csv")
                .read_text(encoding="utf-8-sig")
                .splitlines()
            )
        )
        for row in rows:
            c1, c2, c3, c4, life = (float(v) for v in row[:5])
            by_key[(round(c1, 3), round(c2, 3), round(c3, 3))].append((c4, life))

    rep_vars = [
        float(np.var([x[1] for x in vals], ddof=1))
        for vals in by_key.values()
        if len(vals) >= 2
    ]
    pred_single_std = math.sqrt(float(np.mean(rep_vars)))

    pred_items = []
    for key in sorted(by_key):
        vals = by_key[key]
        c1, c2, c3 = key
        c4 = float(np.mean([x[0] for x in vals]))
        y_sim, _ = wang_2011_lfp_electrothermal_prior(c1, c2, c3, c4)
        pred_items.append(
            {
                "key": key,
                "c1": c1,
                "c2": c2,
                "c3": c3,
                "c4": c4,
                "obs": float(np.mean([x[1] for x in vals])),
                "u_exp": pred_single_std / math.sqrt(len(vals)),
                "y_sim": y_sim,
            }
        )

    u_num = float(
        np.sqrt(
            np.mean(
                [
                    (
                        wang_2011_lfp_electrothermal_prior(
                            it["c1"], it["c2"], it["c3"], it["c4"], dt_s=1.0
                        )[0]
                        - wang_2011_lfp_electrothermal_prior(
                            it["c1"], it["c2"], it["c3"], it["c4"], dt_s=0.25
                        )[0]
                    )
                    ** 2
                    for it in pred_items[:20]
                ]
            )
        )
    )

    s1_cal, s1_val = [], []
    for idx, it in enumerate(pred_items):
        is_val = (idx % 5) == 2
        rec = _make_record(
            f"attia-proto-{idx:03d}",
            CouponRole.VALIDATION_HOLDOUT if is_val else CouponRole.CALIBRATION,
            it["c1"],
            it["c2"],
            it["c3"],
            it["c4"],
            it["obs"],
            it["y_sim"],
            it["u_exp"],
            "attia2020_pred_rounds",
        )
        (s1_val if is_val else s1_cal).append(rec)

    cfg = HybridGroundingConfig(
        fit_prior_scale_rho=True,
        rho_prior_strength=1.0e-2,
        length_scale=1.5,
        coverage_factor_k=2.0,
        u_num=u_num,
        numerical_claim_level="NOT_VERIFIED",
    )
    _, d1 = ground_simulation_with_experiments([*s1_cal, *s1_val], config=cfg)
    print(f"[Study 1 - 107 Protocols] {d1.reason} | state={d1.state.value}")

    # Load 9x5 true run-to-failure cells
    c4_map = {
        (round(float(r[0]), 3), round(float(r[1]), 3), round(float(r[2]), 3)): float(
            r[3]
        )
        for r in csv.reader(
            (CACHE_DIR / "policies_all.csv")
            .read_text(encoding="utf-8-sig")
            .splitlines()
        )
    }
    early_map: dict[tuple[float, float, float], float] = {}
    with (CACHE_DIR / "fig4_predictions.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            k = (
                round(float(row["C1"]), 3),
                round(float(row["C2"]), 3),
                round(float(row["C3"]), 3),
            )
            early_map[k] = float(
                np.mean(
                    [float(row[f"P{i}"]) for i in range(1, 6) if row[f"P{i}"].strip()]
                )
            )
    fig4_items = []
    with (CACHE_DIR / "fig4_final_results.csv").open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            k = (
                round(float(row["C1"]), 3),
                round(float(row["C2"]), 3),
                round(float(row["C3"]), 3),
            )
            rvals = [float(row[f"R{i}"]) for i in range(1, 6)]
            y_sim, _ = wang_2011_lfp_electrothermal_prior(k[0], k[1], k[2], c4_map[k])
            fig4_items.append(
                {
                    "key": k,
                    "c4": c4_map[k],
                    "obs": float(np.mean(rvals)),
                    "u_exp": float(np.std(rvals, ddof=1) / math.sqrt(5.0)),
                    "y_sim": y_sim,
                    "early": early_map[k],
                }
            )

    # Study 2A: cross-batch stress test (strictly disjoint protocols)
    f4_keys = {x["key"] for x in fig4_items}
    s2a = [
        _make_record(
            f"early-{i:03d}",
            CouponRole.CALIBRATION,
            it["c1"],
            it["c2"],
            it["c3"],
            it["c4"],
            it["obs"],
            it["y_sim"],
            it["u_exp"],
            "attia2020_disjoint_early",
        )
        for i, it in enumerate(pred_items)
        if it["key"] not in f4_keys
    ] + [
        _make_record(
            f"rtf-{i:02d}",
            CouponRole.VALIDATION_HOLDOUT,
            it["key"][0],
            it["key"][1],
            it["key"][2],
            it["c4"],
            it["obs"],
            it["y_sim"],
            it["u_exp"],
            "attia2020_fig4_rtf",
        )
        for i, it in enumerate(fig4_items)
    ]
    _, d2a = ground_simulation_with_experiments(s2a, config=cfg)
    print(f"[Study 2A - Cross-Batch]  {d2a.reason} | state={d2a.state.value}")

    # Study 2B: 5 cal / 4 blind true run-to-failure
    cfg_2b = HybridGroundingConfig(
        length_scale=1.8, u_num=u_num, numerical_claim_level="NOT_VERIFIED"
    )
    s2b_w = [
        _make_record(
            f"rtf-w-{i}",
            CouponRole.CALIBRATION if (i % 2 == 0) else CouponRole.VALIDATION_HOLDOUT,
            it["key"][0],
            it["key"][1],
            it["key"][2],
            it["c4"],
            it["obs"],
            it["y_sim"],
            it["u_exp"],
            "attia2020_rtf_wang",
        )
        for i, it in enumerate(fig4_items)
    ]
    s2b_e = [
        _make_record(
            f"rtf-e-{i}",
            CouponRole.CALIBRATION if (i % 2 == 0) else CouponRole.VALIDATION_HOLDOUT,
            it["key"][0],
            it["key"][1],
            it["key"][2],
            it["c4"],
            it["obs"],
            it["early"],
            it["u_exp"],
            "attia2020_rtf_early",
        )
        for i, it in enumerate(fig4_items)
    ]
    _, d2bw = ground_simulation_with_experiments(s2b_w, config=cfg_2b)
    _, d2be = ground_simulation_with_experiments(s2b_e, config=cfg_2b)
    print(f"[Study 2B - RTF Wang]     {d2bw.reason} | state={d2bw.state.value}")
    print(f"[Study 2B - RTF Early]    {d2be.reason} | state={d2be.state.value}")

    # Active test-plan reduction on 107 real protocols (10 seeds, target <= 120 cyc)
    plan_cfg = PhysicalTestPlanConfig(
        batch_size=1, sim_uncertainty_weight=0.0, length_scale=1.5, noise_std=0.25
    )
    bench = run_retrospective_test_reduction_benchmark(
        s1_cal,
        s1_val,
        initial_seed_count=4,
        max_rounds=len(s1_cal) - 4,
        target_holdout_rmse=120.0,
        random_seeds=tuple(range(10)),
        grounding_config=cfg,
        plan_config=plan_cfg,
    )
    print(
        f"[Active Reduction <=120]  active={bench.active_tests_needed} vs "
        f"random_mean={bench.random_mean_tests_needed:.1f} -> "
        f"saved={100.0 * bench.tests_saved_fraction:+.1f}% (hash={bench.benchmark_hash[:12]})"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "attia_2020_cli_run.json"
    out.write_bytes(
        canonical_json_bytes(
            {
                "dataset_sha256": manifest["file_sha256"],
                "study_1": d1.document(),
                "study_2a": d2a.document(),
                "study_2b_wang": d2bw.document(),
                "study_2b_early": d2be.document(),
                "active_120": bench.document(),
            }
        )
    )
    print(f"Wrote run receipt: {out}")


if __name__ == "__main__":
    main()
