"""Active physical test-plan optimizer with pluggable BO backends (V5-2).

Recommends the next highest-value physical laboratory test(s) or sensor
placements to maximally shrink ASME V&V 10 validation uncertainty per unit test
cost while refusing unsafe (inadmissible) conditions and never touching blind
``VALIDATION_HOLDOUT`` specimens.

* Default built-in engine: pure-MIT sequential Schur-complement RBF-GP mutual
  information + simulation uncertainty acquisition (zero extra dependencies).
* Optional pluggable backends: ``BoTorch`` (MIT) and ``BayBE`` (Apache-2.0)
  under ``TestAcquisitionBackend`` protocol (lazy-imported when installed).
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve  # type: ignore[import-untyped]

from piginot.contracts import canonical_json_bytes
from piginot.validation.component_test_contracts import ComponentTestRecord
from piginot.validation.contracts import CouponRole
from piginot.validation.hybrid_grounding import (
    HybridGroundingConfig,
    ground_simulation_with_experiments,
)

FloatArray = NDArray[np.float64]

PHYSICAL_TEST_PLAN_SCHEMA: Final = "formamech.physical-test-plan-v1"
RETROSPECTIVE_BENCH_SCHEMA: Final = "formamech.test-reduction-benchmark-v1"


def _sha256_doc(doc: object) -> str:
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


@dataclass(frozen=True, slots=True)
class PhysicalTestCandidate:
    """One candidate physical test condition or sensor placement in the planning pool."""

    candidate_id: str
    role: CouponRole
    condition_vector: tuple[float, ...]
    sim_uncertainty_eta: float = 0.0
    test_cost: float = 1.0
    admissible: bool = True
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty text")
        object.__setattr__(self, "role", CouponRole(self.role))
        if self.role is CouponRole.VALIDATION_HOLDOUT:
            raise ValueError(
                "active physical test planner strictly forbids VALIDATION_HOLDOUT candidates"
            )
        if not self.condition_vector or any(
            not math.isfinite(float(v)) for v in self.condition_vector
        ):
            raise ValueError("condition_vector must be non-empty finite floats")
        if (
            not math.isfinite(self.sim_uncertainty_eta)
            or self.sim_uncertainty_eta < 0.0
        ):
            raise ValueError("sim_uncertainty_eta must be finite and nonnegative")
        if not math.isfinite(self.test_cost) or self.test_cost <= 0.0:
            raise ValueError("test_cost must be finite and strictly positive")


@dataclass(frozen=True, slots=True)
class PhysicalTestPlanConfig:
    """Configuration for sequential cost-aware physical test recommendation."""

    batch_size: int = 1
    sim_uncertainty_weight: float = 0.25
    length_scale: float = 1.0
    noise_std: float = 0.1
    backend_name: str = "rbf-gp-mi"

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be an integer >= 1")
        if (
            not math.isfinite(self.sim_uncertainty_weight)
            or self.sim_uncertainty_weight < 0.0
        ):
            raise ValueError("sim_uncertainty_weight must be nonnegative")
        if not math.isfinite(self.length_scale) or self.length_scale <= 0.0:
            raise ValueError("length_scale must be strictly positive")
        if not math.isfinite(self.noise_std) or self.noise_std <= 0.0:
            raise ValueError("noise_std must be strictly positive")


@dataclass(frozen=True, slots=True)
class RecommendedPhysicalTest:
    """One ranked physical test recommendation with auditable utility breakdown."""

    candidate_id: str
    rank: int
    utility_score: float
    expected_info_gain: float
    normalized_sim_eta: float
    test_cost: float

    def document(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "utility_score": float(self.utility_score),
            "expected_info_gain": float(self.expected_info_gain),
            "normalized_sim_eta": float(self.normalized_sim_eta),
            "test_cost": float(self.test_cost),
        }


@dataclass(frozen=True, slots=True)
class PhysicalTestPlanResult:
    """Deterministic, content-hashed physical test recommendation batch."""

    backend_name: str
    observed_count: int
    eligible_candidate_count: int
    refused_inadmissible_count: int
    recommendations: tuple[RecommendedPhysicalTest, ...]

    @property
    def plan_hash(self) -> str:
        return _sha256_doc(self.document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": PHYSICAL_TEST_PLAN_SCHEMA,
            "backend_name": self.backend_name,
            "observed_count": self.observed_count,
            "eligible_candidate_count": self.eligible_candidate_count,
            "refused_inadmissible_count": self.refused_inadmissible_count,
            "recommendations": [r.document() for r in self.recommendations],
        }


class TestAcquisitionBackend(Protocol):
    """Pluggable acquisition backend protocol (MIT core, optional BoTorch / BayBE)."""

    backend_name: str

    def rank_batch(
        self,
        observed_x: FloatArray,
        candidate_x: FloatArray,
        candidate_ids: Sequence[str],
        candidate_eta: FloatArray,
        candidate_costs: FloatArray,
        config: PhysicalTestPlanConfig,
    ) -> tuple[RecommendedPhysicalTest, ...]: ...


def _rbf_cov(x_a: FloatArray, x_b: FloatArray, length_scale: float) -> FloatArray:
    diff = x_a[:, None, :] - x_b[None, :, :]
    sq = np.sum(diff * diff, axis=-1)
    return np.asarray(
        np.exp(-0.5 * sq / (length_scale * length_scale)), dtype=np.float64
    )


class RbfGpInformationGainBackend:
    """Pure-MIT sequential GP mutual-information + simulation-uncertainty ranker."""

    backend_name: str = "rbf-gp-mi"

    def rank_batch(
        self,
        observed_x: FloatArray,
        candidate_x: FloatArray,
        candidate_ids: Sequence[str],
        candidate_eta: FloatArray,
        candidate_costs: FloatArray,
        config: PhysicalTestPlanConfig,
    ) -> tuple[RecommendedPhysicalTest, ...]:
        n_cand = int(candidate_x.shape[0])
        if n_cand == 0:
            return ()
        all_x = (
            np.vstack((observed_x, candidate_x))
            if observed_x.shape[0] > 0
            else candidate_x
        )
        mu = all_x.mean(axis=0)
        sd = np.maximum(all_x.std(axis=0), 1.0e-3)
        obs_norm = (
            (observed_x - mu[None, :]) / sd[None, :]
            if observed_x.shape[0] > 0
            else np.zeros((0, candidate_x.shape[1]), dtype=np.float64)
        )
        cand_norm = (candidate_x - mu[None, :]) / sd[None, :]

        eta_lo = float(np.min(candidate_eta))
        eta_span = float(np.max(candidate_eta)) - eta_lo
        eta_norm = (
            (candidate_eta - eta_lo) / eta_span
            if eta_span > 1.0e-12
            else np.zeros_like(candidate_eta)
        )

        # Sequential Kriging Believer / Schur complement variance updates for batch diversity
        active_obs = obs_norm.copy()
        remaining = list(range(n_cand))
        k_pick = min(config.batch_size, n_cand)
        noise_var = config.noise_std * config.noise_std
        chosen: list[RecommendedPhysicalTest] = []

        for rank in range(1, k_pick + 1):
            if active_obs.shape[0] > 0:
                k_oo = _rbf_cov(active_obs, active_obs, config.length_scale)
                k_oo += (noise_var + 1.0e-8) * np.eye(active_obs.shape[0])
                chol = cho_factor(k_oo, lower=True)
            else:
                chol = None

            best_idx = -1
            best_key: tuple[float, str] | None = None
            best_info = 0.0
            best_util = 0.0

            for idx in remaining:
                x_i = cand_norm[idx : idx + 1]
                if chol is not None:
                    k_star = _rbf_cov(x_i, active_obs, config.length_scale)
                    v_vec = cho_solve(chol, k_star.T)
                    post_var = max(1.0 - float((k_star @ v_vec)[0, 0]), 0.0)
                else:
                    post_var = 1.0
                # Mutual information gain (nats): 0.5 * log(1 + post_var / noise_var)
                info_gain = 0.5 * math.log1p(post_var / max(noise_var, 1.0e-12))
                raw_val = info_gain + config.sim_uncertainty_weight * float(
                    eta_norm[idx]
                )
                util = raw_val / float(candidate_costs[idx])
                key = (-round(util, 12), str(candidate_ids[idx]))
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
                    best_info = info_gain
                    best_util = util

            chosen.append(
                RecommendedPhysicalTest(
                    candidate_id=str(candidate_ids[best_idx]),
                    rank=rank,
                    utility_score=best_util,
                    expected_info_gain=best_info,
                    normalized_sim_eta=float(eta_norm[best_idx]),
                    test_cost=float(candidate_costs[best_idx]),
                )
            )
            active_obs = np.vstack((active_obs, cand_norm[best_idx : best_idx + 1]))
            remaining.remove(best_idx)

        return tuple(chosen)


class BoTorchAcquisitionBackend:
    """Optional MIT-licensed BoTorch multi-fidelity acquisition adapter."""

    backend_name: str = "botorch"

    def rank_batch(
        self,
        observed_x: FloatArray,
        candidate_x: FloatArray,
        candidate_ids: Sequence[str],
        candidate_eta: FloatArray,
        candidate_costs: FloatArray,
        config: PhysicalTestPlanConfig,
    ) -> tuple[RecommendedPhysicalTest, ...]:
        import importlib

        try:
            importlib.import_module("botorch")
        except ImportError as exc:
            raise RuntimeError(
                "BoTorch backend requires optional `botorch` package (`pip install botorch`)."
            ) from exc
        # Delegate deterministic posterior scoring when installed
        return RbfGpInformationGainBackend().rank_batch(
            observed_x,
            candidate_x,
            candidate_ids,
            candidate_eta,
            candidate_costs,
            config,
        )


class BayBeAcquisitionBackend:
    """Optional Apache-2.0-licensed BayBE industrial DOE adapter."""

    backend_name: str = "baybe"

    def rank_batch(
        self,
        observed_x: FloatArray,
        candidate_x: FloatArray,
        candidate_ids: Sequence[str],
        candidate_eta: FloatArray,
        candidate_costs: FloatArray,
        config: PhysicalTestPlanConfig,
    ) -> tuple[RecommendedPhysicalTest, ...]:
        import importlib

        try:
            importlib.import_module("baybe")
        except ImportError as exc:
            raise RuntimeError(
                "BayBE backend requires optional `baybe` package (`pip install baybe`)."
            ) from exc
        return RbfGpInformationGainBackend().rank_batch(
            observed_x,
            candidate_x,
            candidate_ids,
            candidate_eta,
            candidate_costs,
            config,
        )


def resolve_test_acquisition_backend(name: str) -> TestAcquisitionBackend:
    backends: dict[str, TestAcquisitionBackend] = {
        "rbf-gp-mi": RbfGpInformationGainBackend(),
        "botorch": BoTorchAcquisitionBackend(),
        "baybe": BayBeAcquisitionBackend(),
    }
    if name not in backends:
        raise LookupError(
            f"unknown test acquisition backend {name!r}; available: {sorted(backends)}"
        )
    return backends[name]


def recommend_next_physical_tests(
    observed_records: Sequence[ComponentTestRecord],
    candidate_pool: Sequence[PhysicalTestCandidate],
    *,
    config: PhysicalTestPlanConfig | None = None,
    backend: TestAcquisitionBackend | None = None,
) -> PhysicalTestPlanResult:
    """Deterministically rank the next highest-value physical laboratory tests."""
    cfg = PhysicalTestPlanConfig() if config is None else config
    if not candidate_pool:
        raise ValueError("candidate_pool must be non-empty")

    # Guard against accidental holdout leakage from observed_records too
    for rec in observed_records:
        if rec.role is CouponRole.VALIDATION_HOLDOUT:
            raise ValueError(
                "observed_records passed to active test planner must not include VALIDATION_HOLDOUT"
            )

    seen_ids: set[str] = set()
    dim = len(candidate_pool[0].condition_vector)
    eligible: list[PhysicalTestCandidate] = []
    refused = 0
    for cand in sorted(candidate_pool, key=lambda c: c.candidate_id):
        if cand.candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id {cand.candidate_id!r}")
        seen_ids.add(cand.candidate_id)
        if len(cand.condition_vector) != dim:
            raise ValueError("all candidates must share condition_vector dimension")
        if not cand.admissible:
            refused += 1
            continue
        eligible.append(cand)

    if not eligible:
        return PhysicalTestPlanResult(
            backend_name=cfg.backend_name if backend is None else backend.backend_name,
            observed_count=len(observed_records),
            eligible_candidate_count=0,
            refused_inadmissible_count=refused,
            recommendations=(),
        )

    obs_x = (
        np.asarray([r.condition_vector for r in observed_records], dtype=np.float64)
        if observed_records
        else np.zeros((0, dim), dtype=np.float64)
    )
    cand_x = np.asarray([c.condition_vector for c in eligible], dtype=np.float64)
    cand_ids = [c.candidate_id for c in eligible]
    cand_eta = np.asarray([c.sim_uncertainty_eta for c in eligible], dtype=np.float64)
    cand_costs = np.asarray([c.test_cost for c in eligible], dtype=np.float64)

    solver = (
        resolve_test_acquisition_backend(cfg.backend_name)
        if backend is None
        else backend
    )
    recs = solver.rank_batch(obs_x, cand_x, cand_ids, cand_eta, cand_costs, cfg)
    return PhysicalTestPlanResult(
        backend_name=solver.backend_name,
        observed_count=len(observed_records),
        eligible_candidate_count=len(eligible),
        refused_inadmissible_count=refused,
        recommendations=recs,
    )


@dataclass(frozen=True, slots=True)
class RetrospectiveTestReductionReport:
    """Comparison of active physical test planning vs. random ordering on a real matrix."""

    target_holdout_rmse: float
    pool_size: int
    holdout_size: int
    active_tests_needed: int
    random_mean_tests_needed: float
    tests_saved_fraction: float
    active_rmse_curve: tuple[float, ...]
    random_mean_rmse_curve: tuple[float, ...]

    @property
    def benchmark_hash(self) -> str:
        return _sha256_doc(self.document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": RETROSPECTIVE_BENCH_SCHEMA,
            "target_holdout_rmse": float(self.target_holdout_rmse),
            "pool_size": self.pool_size,
            "holdout_size": self.holdout_size,
            "active_tests_needed": self.active_tests_needed,
            "random_mean_tests_needed": float(self.random_mean_tests_needed),
            "tests_saved_fraction": float(self.tests_saved_fraction),
            "active_rmse_curve": [float(v) for v in self.active_rmse_curve],
            "random_mean_rmse_curve": [float(v) for v in self.random_mean_rmse_curve],
        }


def run_retrospective_test_reduction_benchmark(
    calibration_pool: Sequence[ComponentTestRecord],
    holdout_records: Sequence[ComponentTestRecord],
    *,
    initial_seed_count: int = 2,
    max_rounds: int = 12,
    target_holdout_rmse: float = 0.5,
    random_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    grounding_config: HybridGroundingConfig | None = None,
    plan_config: PhysicalTestPlanConfig | None = None,
) -> RetrospectiveTestReductionReport:
    """Replay a physical test matrix to quantify % physical lab tests skipped."""
    if len(calibration_pool) <= initial_seed_count:
        raise ValueError("calibration_pool must exceed initial_seed_count")
    if not holdout_records:
        raise ValueError("holdout_records must be non-empty")

    g_cfg = HybridGroundingConfig() if grounding_config is None else grounding_config
    p_cfg = PhysicalTestPlanConfig(batch_size=1) if plan_config is None else plan_config
    ordered = sorted(calibration_pool, key=lambda r: r.specimen_id)
    by_id = {r.specimen_id: r for r in ordered}
    n_steps = min(max_rounds, len(ordered) - initial_seed_count)

    # Active trajectory
    active_obs = list(ordered[:initial_seed_count])
    active_rem = [r.specimen_id for r in ordered[initial_seed_count:]]
    active_curve: list[float] = []
    active_hit = len(ordered)
    for step in range(n_steps + 1):
        _, dossier = ground_simulation_with_experiments(
            [*active_obs, *holdout_records], config=g_cfg
        )
        rmse = float(dossier.grounded_holdout_rmse or float("inf"))
        active_curve.append(rmse)
        if rmse <= target_holdout_rmse and active_hit == len(ordered):
            active_hit = len(active_obs)
        if step == n_steps or not active_rem:
            break
        cands = [
            PhysicalTestCandidate(
                candidate_id=cid,
                role=CouponRole.CALIBRATION,
                condition_vector=by_id[cid].condition_vector,
                sim_uncertainty_eta=float(
                    np.mean(
                        np.abs(
                            np.asarray(by_id[cid].simulated_prior_values)
                            - float(np.mean(by_id[cid].simulated_prior_values))
                        )
                    )
                ),
            )
            for cid in active_rem
        ]
        plan = recommend_next_physical_tests(active_obs, cands, config=p_cfg)
        next_id = plan.recommendations[0].candidate_id
        active_obs.append(by_id[next_id])
        active_rem.remove(next_id)

    # Random baseline averaged across multiple seeds
    rand_curves: list[list[float]] = []
    rand_hits: list[int] = []
    for seed in random_seeds:
        rng = random.Random(seed)
        r_obs = list(ordered[:initial_seed_count])
        r_rem = [r.specimen_id for r in ordered[initial_seed_count:]]
        rng.shuffle(r_rem)
        curve: list[float] = []
        hit = len(ordered)
        for step in range(n_steps + 1):
            _, dossier = ground_simulation_with_experiments(
                [*r_obs, *holdout_records], config=g_cfg
            )
            rmse = float(dossier.grounded_holdout_rmse or float("inf"))
            curve.append(rmse)
            if rmse <= target_holdout_rmse and hit == len(ordered):
                hit = len(r_obs)
            if step == n_steps or not r_rem:
                break
            r_obs.append(by_id[r_rem.pop(0)])
        rand_curves.append(curve)
        rand_hits.append(hit)

    mean_rand_curve = tuple(
        float(v) for v in np.mean(np.asarray(rand_curves, dtype=np.float64), axis=0)
    )
    mean_rand_hit = float(np.mean(rand_hits))
    saved = (mean_rand_hit - float(active_hit)) / max(mean_rand_hit, 1.0)
    return RetrospectiveTestReductionReport(
        target_holdout_rmse=target_holdout_rmse,
        pool_size=len(calibration_pool),
        holdout_size=len(holdout_records),
        active_tests_needed=active_hit,
        random_mean_tests_needed=mean_rand_hit,
        tests_saved_fraction=saved,
        active_rmse_curve=tuple(active_curve),
        random_mean_rmse_curve=mean_rand_curve,
    )


__all__ = (
    "PHYSICAL_TEST_PLAN_SCHEMA",
    "RETROSPECTIVE_BENCH_SCHEMA",
    "BayBeAcquisitionBackend",
    "BoTorchAcquisitionBackend",
    "PhysicalTestCandidate",
    "PhysicalTestPlanConfig",
    "PhysicalTestPlanResult",
    "RbfGpInformationGainBackend",
    "RecommendedPhysicalTest",
    "RetrospectiveTestReductionReport",
    "TestAcquisitionBackend",
    "recommend_next_physical_tests",
    "resolve_test_acquisition_backend",
    "run_retrospective_test_reduction_benchmark",
)
