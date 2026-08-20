from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import experiments.spike.iteration8 as iteration8
from experiments.spike.confirm import build_fresh_campaign_manifest
from experiments.spike.fem import SanityResult
from experiments.spike.geometry import sample_geometry_cases
from experiments.spike.iteration8 import (
    CONFIRMATION_REPORT_NAME,
    CORE_CONFIRMATION_REPORT_NAME,
    PREREGISTRATION_NAME,
    REPRODUCTION_CHECKPOINT_NAME,
    REPRODUCTION_EPOCHS,
    REPRODUCTION_LEARNING_RATE,
    REPRODUCTION_MODEL_SEED,
    REPRODUCTION_REPORT_NAME,
    REPRODUCTION_SAMPLER_SEED,
    _confirmation_regularity,
    build_iteration8_preregistration,
    reproduce_iteration3_protocol,
    run_iteration8_confirmation,
    select_reproduction_training_cases,
)


def test_iteration3_protocol_reproduction_contract_is_frozen() -> None:
    assert REPRODUCTION_MODEL_SEED == 314159
    assert REPRODUCTION_SAMPLER_SEED == 314179
    assert REPRODUCTION_LEARNING_RATE == 1e-3
    assert REPRODUCTION_EPOCHS == 1500


def test_reproduction_selects_only_the_original_40_training_geometries() -> None:
    historical = sample_geometry_cases(
        master_seed=20260809,
        count_per_family=30,
        nx=6,
        ny=3,
    )

    selected = select_reproduction_training_cases(historical)

    assert len(selected) == 40
    assert {
        family: sum(case.family == family for case in selected)
        for family in (
            "dogbone",
            "centre_notched",
        )
    } == {"dogbone": 20, "centre_notched": 20}
    selected_geometry_ids = {case.geometry_id for case in selected}
    evaluation = tuple(
        case for case in historical if case.geometry_id not in selected_geometry_ids
    )
    assert {case.geometry_id for case in selected}.isdisjoint(
        case.geometry_id for case in evaluation
    )
    assert {case.mesh_id for case in selected}.isdisjoint(
        case.mesh_id for case in evaluation
    )


def test_preregistration_is_complete_and_contains_no_post_solve_results() -> None:
    historical = sample_geometry_cases(
        master_seed=20260809,
        count_per_family=30,
        nx=6,
        ny=3,
    )
    fresh = build_fresh_campaign_manifest(
        historical_cases=historical,
        master_seed=2026081301,
        nx=6,
        ny=3,
    )
    checkpoint_provenance = {
        "schema_version": "iteration3-protocol-reproduction-v1",
        "architecture": "conditioned_mlp",
        "epochs": 1500,
    }

    document = build_iteration8_preregistration(
        checkpoint_sha256="a" * 64,
        checkpoint_provenance=checkpoint_provenance,
        fresh_campaign=fresh,
        fem_content_sha256="b" * 64,
        created_utc="2026-08-13T00:00:00Z",
    )

    assert document["schema_version"] == "iteration8-preregistration-v1"
    assert document["role"] == "LEGACY"
    assert document["no_release_evidence"] is True
    assert document["created_before_fem_solves"] is True
    assert document["no_tuning"] is True
    assert document["frozen_model"]["sha256"] == "a" * 64
    assert document["fresh_campaign"]["manifest_sha256"] == fresh.sha256
    assert document["fresh_campaign"]["manifest"] == fresh.manifest
    assert document["gates"]["family_relative_l2_P_max"] == 0.15
    assert document["gates"]["family_relative_l2_P_bootstrap_upper_max"] == 0.15
    assert "metrics" not in document
    assert "fem_results" not in document
    json.dumps(document, allow_nan=False)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    assert len(hashlib.sha256(canonical.encode()).hexdigest()) == 64


class _FakeReproductionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Parameter(torch.full((610_455,), 0.01))
        self.p_head = torch.nn.Module()
        self.p_head.output = torch.nn.Parameter(torch.full((4_484,), 0.02))

    def forward(
        self,
        points: torch.Tensor,
        _boundary: torch.Tensor,
        _u_delta: torch.Tensor,
        _x_max: torch.Tensor,
        _y_max: torch.Tensor,
    ) -> SimpleNamespace:
        scale = self.p_head.output[0]
        return SimpleNamespace(
            u=torch.ones_like(points) * scale,
            P_served=torch.ones((*points.shape[:2], 2, 2)) * scale,
        )


def _converged_result(mesh: object, *, stress_scale: float = 1.0) -> SimpleNamespace:
    point_count = len(mesh.points)
    return SimpleNamespace(
        converged=True,
        failure_reason=None,
        displacement=np.zeros((point_count, 2), dtype=np.float64),
        first_piola=np.full(
            (point_count, 2, 2),
            stress_scale,
            dtype=np.float64,
        ),
        solve_seconds=0.01,
        newton_iterations=3,
    )


def test_reproduce_iteration3_protocol_orchestrates_the_frozen_all_40_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = sample_geometry_cases(
        master_seed=20260809,
        count_per_family=30,
        nx=6,
        ny=3,
    )
    expected_training = select_reproduction_training_cases(historical)
    expected_ids = [mesh.geometry_id for mesh in expected_training]
    evaluation_ids = {
        mesh.geometry_id for mesh in historical if mesh.geometry_id not in expected_ids
    }
    artifact_root = tmp_path / "iteration8"
    upstream_path = tmp_path / "upstream.pt"
    fem_hash = "b" * 64
    checkpoint_sha = "c" * 64
    events: list[str] = []
    sample_calls: list[dict[str, int]] = []
    solved_ids: list[str] = []
    model_seeds: list[int] = []
    training_call: dict[str, object] = {}
    checkpoint_call: dict[str, object] = {}
    reloaded_model = _FakeReproductionModel()
    upstream_state = {"upstream": object()}

    def sample(**kwargs: int) -> tuple[object, ...]:
        sample_calls.append(kwargs)
        return historical

    def solve(mesh: object, cache_dir: Path, received_hash: str) -> object:
        assert cache_dir.name == "pi-ginot-s1-legacy-cache-019fe6e9"
        assert received_hash == fem_hash
        solved_ids.append(mesh.geometry_id)
        return _converged_result(mesh)

    def new_model(state: object) -> _FakeReproductionModel:
        assert state is upstream_state
        events.append("model")
        return _FakeReproductionModel()

    dataset = object()

    def tensor_dataset(pairs: object) -> object:
        assert [mesh.geometry_id for mesh, _result in pairs] == expected_ids
        events.append("dataset")
        return dataset

    def train(
        model: object,
        received_dataset: object,
        indices: object,
        **kwargs: object,
    ) -> float:
        assert received_dataset is dataset
        assert list(indices) == list(range(40))
        training_call.update(kwargs)
        events.append("train")
        with torch.no_grad():
            model.p_head.output.fill_(0.123)
        return 0.0125

    def save(
        path: Path,
        model: _FakeReproductionModel,
        *,
        provenance: object,
    ) -> str:
        assert events[-1] == "train"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frozen iteration3 reproduction")
        checkpoint_call.update(path=path, model=model, provenance=provenance)
        reloaded_model.load_state_dict(model.state_dict())
        events.append("save")
        return checkpoint_sha

    def load(
        path: Path,
        *,
        expected_sha256: str,
        expected_provenance: object,
    ) -> _FakeReproductionModel:
        assert events[-1] == "save"
        assert path == checkpoint_call["path"]
        assert expected_sha256 == checkpoint_sha
        assert expected_provenance == checkpoint_call["provenance"]
        events.append("reload")
        return reloaded_model

    monkeypatch.setattr(iteration8, "sample_geometry_cases", sample)
    monkeypatch.setattr(iteration8, "_fem_content_hash", lambda _root: fem_hash)
    monkeypatch.setattr(iteration8, "_solve_with_cache", solve)
    monkeypatch.setattr(
        iteration8,
        "_load_upstream_checkpoint",
        lambda path: SimpleNamespace(
            path=path,
            sha256="d" * 64,
            epoch=1424,
            state_dict=upstream_state,
        ),
    )
    monkeypatch.setattr(iteration8, "_new_pretrained_model", new_model)
    monkeypatch.setattr(iteration8, "_tensor_dataset", tensor_dataset)
    monkeypatch.setattr(iteration8, "_train_fixed_epochs", train)
    monkeypatch.setattr(iteration8, "save_final_checkpoint", save)
    monkeypatch.setattr(iteration8, "load_final_checkpoint", load)
    monkeypatch.setattr(
        iteration8,
        "_source_hashes",
        lambda _root: {"experiments/spike/iteration8.py": "e" * 64},
    )
    monkeypatch.setattr(iteration8, "_environment", lambda: {"device": "cpu"})
    monkeypatch.setattr(
        torch,
        "manual_seed",
        lambda seed: model_seeds.append(seed),
    )
    monkeypatch.setattr(torch, "set_num_threads", lambda _count: None)

    report = reproduce_iteration3_protocol(
        repo_root=tmp_path,
        upstream_checkpoint_path=upstream_path,
        artifact_root=artifact_root,
    )

    assert sample_calls == [
        {
            "master_seed": iteration8.CAMPAIGN_SEED,
            "count_per_family": 30,
            "nx": 32,
            "ny": 10,
        }
    ]
    assert solved_ids == expected_ids
    assert set(solved_ids).isdisjoint(evaluation_ids)
    assert model_seeds == [REPRODUCTION_MODEL_SEED]
    assert training_call == {
        "learning_rate": REPRODUCTION_LEARNING_RATE,
        "epochs": REPRODUCTION_EPOCHS,
        "random_seed": REPRODUCTION_SAMPLER_SEED,
        "label": "iteration3 protocol reproduction",
        "peak_supervised_p_weight": 0.0,
        "served_stress_physics": iteration8.NO_SERVED_STRESS_PHYSICS,
        "stress_supervision": "nodal_patch",
        "max_seconds": iteration8.REPRODUCTION_MAX_SECONDS,
    }
    assert events == ["model", "dataset", "train", "save", "reload"]
    provenance = checkpoint_call["provenance"]
    assert provenance["configuration_alias"] == "iteration3_joint_adam"
    assert provenance["architecture"] == "conditioned_mlp"
    assert provenance["optimizer"] == "Adam"
    assert provenance["learning_rate"] == REPRODUCTION_LEARNING_RATE
    assert provenance["epochs"] == REPRODUCTION_EPOCHS
    assert provenance["model_seed"] == REPRODUCTION_MODEL_SEED
    assert provenance["sampler_seed"] == REPRODUCTION_SAMPLER_SEED
    assert provenance["training_geometry_ids"] == expected_ids
    assert provenance["model_parameter_count"] == 614_939
    assert provenance["stress_head_parameter_count"] == 4_484
    assert provenance["final_hybrid_loss"] == 0.0125

    checkpoint_path = artifact_root / "checkpoints" / REPRODUCTION_CHECKPOINT_NAME
    report_path = artifact_root / REPRODUCTION_REPORT_NAME
    assert checkpoint_call["path"] == checkpoint_path
    assert report["checkpoint"]["sha256"] == checkpoint_sha
    assert (
        report["checkpoint"]["reload_probe_output_sha256"]
        == (provenance["probe_output_sha256"])
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def _reproduction_report(path: Path) -> tuple[Path, dict[str, object]]:
    checkpoint_path = path.parent / "iteration3_protocol_reproduction.pt"
    provenance = {
        "schema_version": "iteration3-protocol-reproduction-v1",
        "architecture": "conditioned_mlp",
        "epochs": 1500,
    }
    path.write_text(
        json.dumps(
            {
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": "a" * 64,
                    "provenance": provenance,
                }
            }
        ),
        encoding="utf-8",
    )
    return checkpoint_path, provenance


def _regularity_result(*, well_posed: bool) -> dict[str, object]:
    return {
        "selected_geometry_ids": ["regularity-case"],
        "relative_change_threshold": 0.10,
        "all_converged": well_posed,
        "all_last_changes_within_limit": well_posed,
        "well_posed": well_posed,
        "samples": [
            {
                "geometry_id": "regularity-case",
                "family": "dogbone",
                "family_index": 0,
                "levels": [
                    {
                        "nx": 32,
                        "ny": 10,
                        "mesh_id": "mesh-regularity",
                        "converged": well_posed,
                        "failure_reason": None if well_posed else "failed",
                        "peak_stress_MPa": 10.0 if well_posed else None,
                    }
                ],
                "relative_change_32_to_64": 0.05 if well_posed else None,
            }
        ],
    }


def _raw_confirmation_gates(
    *,
    peak_pass: bool,
    displacement_pass: bool,
    bootstrap_pass: bool = True,
) -> dict[str, object]:
    family_gates = {
        family: {
            "relative_l2_P": 0.10,
            "bootstrap_95_ci_upper": 0.13,
            "P_within_15_fraction": 0.90,
            "P_within_15_wilson_lower": 0.70,
            "relative_l2_u": 0.05 if displacement_pass else 0.08,
            "relative_l2_P_p90": 0.18,
            "worst_relative_peak_stress_error": 0.90,
            "mean_inference_seconds": 0.01,
            "point_estimate_pass": True,
            "bootstrap_upper_pass": bootstrap_pass,
            "pass_fraction_gate": True,
            "wilson_lower_gate": True,
            "displacement_gate": displacement_pass,
            "stress_p90_gate": True,
            "peak_stress_gate": peak_pass,
            "inference_time_gate": True,
            "all_metrics_finite": True,
            "all_non_bootstrap_gates_pass": peak_pass and displacement_pass,
        }
        for family in ("dogbone", "centre_notched")
    }
    return {
        "families": family_gates,
        "all_point_estimates_pass": True,
        "all_bootstrap_upper_bounds_pass": bootstrap_pass,
        "all_other_gates_pass": peak_pass and displacement_pass,
        "all_metrics_finite": True,
    }


@pytest.mark.parametrize(
    (
        "regularity_well_posed",
        "displacement_pass",
        "raw_status",
        "expected_status",
        "bootstrap_pass",
    ),
    (
        pytest.param(True, True, "CONFIRMED", "CONFIRMED", True, id="well_posed"),
        pytest.param(
            False,
            True,
            "NOT_CONFIRMED",
            "INCONCLUSIVE",
            True,
            id="peak_not_run",
        ),
        pytest.param(
            False,
            True,
            "INCONCLUSIVE",
            "INCONCLUSIVE",
            False,
            id="peak_not_run_and_bootstrap_uncertain",
        ),
        pytest.param(
            False,
            False,
            "NOT_CONFIRMED",
            "NOT_CONFIRMED",
            True,
            id="other_gate_failure",
        ),
    ),
)
def test_run_iteration8_confirmation_freezes_then_merges_complete_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    regularity_well_posed: bool,
    displacement_pass: bool,
    raw_status: str,
    expected_status: str,
    bootstrap_pass: bool,
) -> None:
    artifact_root = tmp_path / f"iteration8-{regularity_well_posed}"
    reproduction_path = tmp_path / f"reproduction-{regularity_well_posed}.json"
    checkpoint_path, provenance = _reproduction_report(reproduction_path)
    historical = sample_geometry_cases(
        master_seed=20260809,
        count_per_family=30,
        nx=6,
        ny=3,
    )
    fresh = build_fresh_campaign_manifest(
        historical_cases=historical,
        master_seed=2026081301,
        nx=6,
        ny=3,
    )
    frozen_model = object()
    fem_hash = "b" * 64
    events: list[str] = []
    load_call: dict[str, object] = {}
    fresh_call: dict[str, object] = {}
    regularity_call: dict[str, object] = {}
    original_write_json = iteration8.write_json_strict

    def load(
        source: Path,
        *,
        expected_sha256: str,
        expected_provenance: object,
    ) -> object:
        load_call.update(
            source=source,
            sha256=expected_sha256,
            provenance=expected_provenance,
        )
        events.append("load_frozen")
        return frozen_model

    def sample(**kwargs: int) -> tuple[object, ...]:
        assert kwargs == {
            "master_seed": iteration8.CAMPAIGN_SEED,
            "count_per_family": 30,
            "nx": 32,
            "ny": 10,
        }
        events.append("historical")
        return historical

    def build_fresh(**kwargs: object) -> object:
        fresh_call.update(kwargs)
        events.append("fresh")
        return fresh

    def write_json(path: Path, document: object) -> None:
        if path.name == PREREGISTRATION_NAME:
            events.append("write_preregistration")
        elif path.name == CONFIRMATION_REPORT_NAME:
            events.append("write_final_report")
        original_write_json(path, document)

    def sanity(**kwargs: object) -> SanityResult:
        assert kwargs == {"axial_stretch": 1.20, "nx": 6, "ny": 6}
        assert (artifact_root / PREREGISTRATION_NAME).exists()
        events.append("sanity")
        return SanityResult(True, None, 0.95, 1e-9, 1e-9, 1e-9)

    def solve_with_cache(mesh: object, cache_dir: Path, received_hash: str) -> object:
        assert (artifact_root / PREREGISTRATION_NAME).exists()
        assert f"confirm-v1-{iteration8.CONFIRMATION_GEOMETRY_SEED}" in cache_dir.name
        assert received_hash == fem_hash
        events.append("solve")
        return _converged_result(mesh)

    def run_frozen_confirmation(**kwargs: object) -> dict[str, object]:
        assert (artifact_root / PREREGISTRATION_NAME).exists()
        assert kwargs["model"] is frozen_model
        assert kwargs["checkpoint_sha256"] == "a" * 64
        assert kwargs["checkpoint_provenance"] == provenance
        assert kwargs["cases"] == fresh.cases
        assert kwargs["historical_cases"] == historical
        assert kwargs["campaign_manifest"] == fresh.manifest
        assert kwargs["evaluator"] is iteration8._evaluate_confirmation_pairs
        assert kwargs["artifact_writer"] is iteration8._write_confirmation_fields
        assert kwargs["bootstrap_seed"] == iteration8.CONFIRMATION_STATISTICS_SEED
        assert kwargs["bootstrap_resamples"] == 10_000
        assert kwargs["report_path"] == artifact_root / CORE_CONFIRMATION_REPORT_NAME
        kwargs["fem_solver"](fresh.cases[0])
        events.append("confirmation")
        core_report = {
            "schema_version": "iteration8-confirmation-v1",
            "role": "LEGACY",
            "no_release_evidence": True,
            "no_tuning": True,
            "status": raw_status,
            "decision": {"status": raw_status},
            "gates": _raw_confirmation_gates(
                peak_pass=regularity_well_posed,
                displacement_pass=displacement_pass,
                bootstrap_pass=bootstrap_pass,
            ),
            "per_case_metrics": [{"geometry_id": "case"}],
            "statistics": {"dogbone": {}, "centre_notched": {}},
            "artifacts": {"comparison_plot_svg_count": 50},
        }
        original_write_json(kwargs["report_path"], core_report)
        events.append("write_core_report")
        return core_report

    def regularity(**kwargs: object) -> dict[str, object]:
        regularity_call.update(kwargs)
        events.append("regularity")
        return _regularity_result(well_posed=regularity_well_posed)

    def plots(
        records: object,
        statistics: object,
        received_regularity: object,
        output_dir: Path,
    ) -> dict[str, object]:
        assert records == [{"geometry_id": "case"}]
        assert statistics == {"dogbone": {}, "centre_notched": {}}
        assert received_regularity == _regularity_result(
            well_posed=regularity_well_posed
        )
        assert output_dir == artifact_root / "summary_plots"
        events.append("plots")
        return {
            "directory": str(output_dir.resolve()),
            "count": 6,
            "sha256": {"summary.svg": "c" * 64},
        }

    monkeypatch.setattr(iteration8, "load_final_checkpoint", load)
    monkeypatch.setattr(iteration8, "sample_geometry_cases", sample)
    monkeypatch.setattr(iteration8, "build_fresh_campaign_manifest", build_fresh)
    monkeypatch.setattr(iteration8, "_fem_content_hash", lambda _root: fem_hash)
    monkeypatch.setattr(iteration8, "write_json_strict", write_json)
    monkeypatch.setattr(iteration8, "_file_sha256", lambda _path: "d" * 64)
    monkeypatch.setattr(iteration8, "_solve_with_cache", solve_with_cache)
    monkeypatch.setattr(iteration8, "run_confirmation", run_frozen_confirmation)
    monkeypatch.setattr(iteration8, "_confirmation_regularity", regularity)
    monkeypatch.setattr(
        "experiments.spike.fem.solve_uniaxial_square",
        sanity,
    )
    monkeypatch.setattr(
        "experiments.spike.iteration8_plots.write_summary_plots",
        plots,
    )

    report = run_iteration8_confirmation(
        repo_root=tmp_path,
        reproduction_report_path=reproduction_path,
        artifact_root=artifact_root,
    )

    assert load_call == {
        "source": checkpoint_path,
        "sha256": "a" * 64,
        "provenance": provenance,
    }
    assert fresh_call == {
        "historical_cases": historical,
        "master_seed": iteration8.CONFIRMATION_GEOMETRY_SEED,
        "nx": 32,
        "ny": 10,
    }
    assert events[:5] == [
        "load_frozen",
        "historical",
        "fresh",
        "write_preregistration",
        "sanity",
    ]
    assert events.index("write_preregistration") < events.index("solve")
    assert events[-4:] == [
        "write_core_report",
        "regularity",
        "plots",
        "write_final_report",
    ]
    assert regularity_call["cases"] == fresh.cases
    assert regularity_call["fem_content_sha256"] == fem_hash

    preregistration = json.loads(
        (artifact_root / PREREGISTRATION_NAME).read_text(encoding="utf-8")
    )
    expected_regularity_ids = [
        family_cases[index].geometry_id
        for family in ("dogbone", "centre_notched")
        for family_cases in [
            tuple(case for case in fresh.cases if case.family == family)
        ]
        for index in (0, 12, 24)
    ]
    assert preregistration["created_before_fem_solves"] is True
    assert preregistration["regularity_geometry_ids"] == expected_regularity_ids
    assert report["status"] == expected_status
    assert report["decision"]["status"] == expected_status
    if regularity_well_posed:
        assert "regularity_reason" not in report["decision"]
    else:
        assert "10% gate" in report["decision"]["regularity_reason"]
        family_gates = report["gates"]["families"]
        assert all(
            gates["peak_stress_gate"] == "NOT_RUN" for gates in family_gates.values()
        )
        assert all(
            gates["all_non_bootstrap_gates_pass"] is displacement_pass
            for gates in family_gates.values()
        )
        assert report["gates"]["all_other_gates_pass"] is displacement_pass
    assert report["preregistration"]["sha256"] == "d" * 64
    assert report["preregistration"]["created_before_fem_solves"] is True
    assert report["core_evaluation"]["path"] == str(
        (artifact_root / CORE_CONFIRMATION_REPORT_NAME).resolve()
    )
    assert report["core_evaluation"]["sha256"] == "d" * 64
    assert report["fem_content_sha256"] == fem_hash
    assert report["fem_sanity"]["max_error"] == 1e-9
    assert report["peak_regularity"]["well_posed"] is regularity_well_posed
    assert report["summary_plots"]["count"] == 6
    assert (
        json.loads(
            (artifact_root / CONFIRMATION_REPORT_NAME).read_text(encoding="utf-8")
        )
        == report
    )


@pytest.mark.parametrize(
    ("mode", "expected_well_posed", "expected_change"),
    (
        ("stable", True, 0.05),
        ("mesh_sensitive", False, 0.20),
        ("nonconverged", False, None),
    ),
)
def test_confirmation_regularity_status_requires_convergence_and_ten_percent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_well_posed: bool,
    expected_change: float | None,
) -> None:
    cases = sample_geometry_cases(
        master_seed=2026081301,
        count_per_family=25,
        nx=6,
        ny=3,
    )

    def solve(mesh: object, _cache: Path, _hash: str) -> object:
        nx = {160: 16, 640: 32, 2560: 64}[len(mesh.cells)]
        if mode == "nonconverged" and nx == 64:
            return SimpleNamespace(
                converged=False,
                failure_reason="deliberate refinement failure",
                displacement=None,
                first_piola=None,
                solve_seconds=0.01,
                newton_iterations=3,
            )
        scale_by_nx = {
            "stable": {16: 0.90, 32: 0.95, 64: 1.00},
            "mesh_sensitive": {16: 0.70, 32: 0.80, 64: 1.00},
            "nonconverged": {16: 0.90, 32: 0.95},
        }
        return _converged_result(mesh, stress_scale=scale_by_nx[mode][nx])

    monkeypatch.setattr(iteration8, "_solve_with_cache", solve)

    regularity = _confirmation_regularity(
        cases=cases,
        cache_dir=tmp_path / "cache",
        fem_content_sha256="a" * 64,
    )

    assert regularity["well_posed"] is expected_well_posed
    assert regularity["all_converged"] is (mode != "nonconverged")
    assert regularity["all_last_changes_within_limit"] is (mode != "mesh_sensitive")
    assert len(regularity["samples"]) == 6
    assert len(regularity["selected_geometry_ids"]) == 6
    assert all(
        sample["family_index"] in {0, 12, 24} for sample in regularity["samples"]
    )
    for sample in regularity["samples"]:
        change = sample["relative_change_32_to_64"]
        if expected_change is None:
            assert change is None
        else:
            assert change == pytest.approx(expected_change)


@pytest.mark.parametrize(
    "existing_name",
    (REPRODUCTION_CHECKPOINT_NAME, REPRODUCTION_REPORT_NAME),
)
def test_reproduction_refuses_to_overwrite_before_starting_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_name: str,
) -> None:
    artifact_root = tmp_path / "iteration8"
    existing = (
        artifact_root / "checkpoints" / existing_name
        if existing_name == REPRODUCTION_CHECKPOINT_NAME
        else artifact_root / existing_name
    )
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("preserve me", encoding="utf-8")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preflight refusal must happen before FEM work")

    monkeypatch.setattr(iteration8, "_fem_content_hash", forbidden)

    with pytest.raises(FileExistsError, match="already exists"):
        reproduce_iteration3_protocol(
            repo_root=tmp_path,
            upstream_checkpoint_path=tmp_path / "upstream.pt",
            artifact_root=artifact_root,
        )

    assert existing.read_text(encoding="utf-8") == "preserve me"


@pytest.mark.parametrize(
    "existing_name",
    (PREREGISTRATION_NAME, CORE_CONFIRMATION_REPORT_NAME, CONFIRMATION_REPORT_NAME),
)
def test_confirmation_refuses_to_overwrite_before_loading_the_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_name: str,
) -> None:
    artifact_root = tmp_path / "iteration8"
    existing = artifact_root / existing_name
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text("preserve me", encoding="utf-8")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preflight refusal must happen before checkpoint loading")

    monkeypatch.setattr(iteration8, "_load_reproduction", forbidden)

    with pytest.raises(FileExistsError, match="already been started"):
        run_iteration8_confirmation(
            repo_root=tmp_path,
            reproduction_report_path=tmp_path / "reproduction.json",
            artifact_root=artifact_root,
        )

    assert existing.read_text(encoding="utf-8") == "preserve me"
