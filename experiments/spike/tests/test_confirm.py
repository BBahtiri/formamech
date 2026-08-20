from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from experiments.spike.confirm import (
    CONFIRMATION_CASES_PER_FAMILY,
    CONFIRMATION_GEOMETRY_SEED,
    CONFIRMATION_STATISTICS_SEED,
    MIN_CONVERGED_PER_FAMILY,
    MIN_CONVERGED_TOTAL,
    build_fresh_campaign_manifest,
    load_final_checkpoint,
    run_confirmation,
    save_final_checkpoint,
    summarize_confirmation_statistics,
    validate_fresh_campaign,
)
from experiments.spike.geometry import SpikeMesh, sample_geometry_cases
from experiments.spike.model import MixedPhysicsOperator, ModelConfig

HISTORICAL_GEOMETRY_SEED = 20260809


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        boundary_points=20,
        latent_tokens=4,
        embed_dim=16,
        num_heads=4,
        encoder_self_layers=1,
        decoder_cross_layers=2,
        positional_encoding_max_degree=2,
        neighbor_count=4,
        stress_head_variant="conditioned_mlp",
    )


def _provenance() -> dict[str, object]:
    return {
        "schema_version": "iteration3-protocol-reproduction-v1",
        "role": "LEGACY",
        "no_release_evidence": True,
        "architecture": "conditioned_mlp",
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "epochs": 1500,
        "model_seed": 314159,
        "sampler_seed": 314179,
        "training_geometry_ids": [f"training-{index:02d}" for index in range(40)],
        "training_geometry_digest": "1" * 64,
        "fem_content_sha256": "2" * 64,
        "upstream_checkpoint_sha256": "3" * 64,
        "upstream_checkpoint_epoch": 1424,
        "model_source_sha256": "4" * 64,
        "run_source_sha256": "5" * 64,
        "objective": "joint_u_P_equilibrium_constitutive",
    }


def _historical_cases() -> tuple[SpikeMesh, ...]:
    return sample_geometry_cases(
        master_seed=HISTORICAL_GEOMETRY_SEED,
        count_per_family=30,
        nx=6,
        ny=3,
    )


def _fresh_campaign() -> tuple[tuple[SpikeMesh, ...], Mapping[str, Any]]:
    campaign = build_fresh_campaign_manifest(
        historical_cases=_historical_cases(),
        master_seed=CONFIRMATION_GEOMETRY_SEED,
        nx=6,
        ny=3,
    )
    return campaign.cases, campaign.manifest


def _recursive_floats(value: object) -> list[float]:
    if isinstance(value, Mapping):
        output: list[float] = []
        for item in value.values():
            output.extend(_recursive_floats(item))
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        output = []
        for item in value:
            output.extend(_recursive_floats(item))
        return output
    if isinstance(value, float):
        return [value]
    return []


def _metric(mesh: SpikeMesh, index: int) -> dict[str, object]:
    relative_p = 0.08 + 0.002 * (index % 10)
    relative_u = 0.02 + 0.001 * (index % 5)
    return {
        "geometry_id": mesh.geometry_id,
        "family": mesh.family,
        "P_squared_error": relative_p**2,
        "P_squared_reference": 1.0,
        "u_squared_error": relative_u**2,
        "u_squared_reference": 1.0,
        "relative_l2_P": relative_p,
        "relative_l2_u": relative_u,
        "relative_peak_stress_error": 0.25 + 0.01 * (index % 7),
        "inference_seconds": 0.001 + 0.00001 * index,
    }


def test_confirmation_protocol_constants_are_preregistered() -> None:
    assert CONFIRMATION_CASES_PER_FAMILY == 25
    assert CONFIRMATION_GEOMETRY_SEED == 2026081301
    assert CONFIRMATION_STATISTICS_SEED == 2026081302
    assert MIN_CONVERGED_PER_FAMILY == 24
    assert MIN_CONVERGED_TOTAL == 48


def test_final_checkpoint_roundtrip_restores_the_complete_conditioned_model(
    tmp_path: Path,
) -> None:
    model = MixedPhysicsOperator(_tiny_config())
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_(0.001 * (index + 1))
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    assert any(name.startswith("p_head.") for name in original_state)

    target = tmp_path / "iteration3_joint_adam.pt"
    provenance = _provenance()
    digest = save_final_checkpoint(target, model, provenance=provenance)

    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    restored = load_final_checkpoint(
        target,
        expected_sha256=digest,
        expected_provenance=provenance,
    )

    assert restored.spike_config == _tiny_config()
    assert restored.spike_config.stress_head_variant == "conditioned_mlp"
    assert set(restored.state_dict()) == set(original_state)
    for name, expected in original_state.items():
        torch.testing.assert_close(
            restored.state_dict()[name],
            expected,
            atol=0.0,
            rtol=0.0,
        )


def test_final_checkpoint_normalises_torch_version_for_weights_only_load(
    tmp_path: Path,
) -> None:
    model = MixedPhysicsOperator(_tiny_config())
    provenance = {**_provenance(), "torch_version": torch.__version__}
    target = tmp_path / "torch-version.pt"

    digest = save_final_checkpoint(target, model, provenance=provenance)

    restored = load_final_checkpoint(
        target,
        expected_sha256=digest,
        expected_provenance=provenance,
    )
    assert isinstance(restored, MixedPhysicsOperator)


def test_final_checkpoint_loader_rejects_sha_and_provenance_mismatch(
    tmp_path: Path,
) -> None:
    target = tmp_path / "iteration3_joint_adam.pt"
    provenance = _provenance()
    digest = save_final_checkpoint(
        target,
        MixedPhysicsOperator(_tiny_config()),
        provenance=provenance,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        load_final_checkpoint(
            target,
            expected_sha256="0" * 64,
            expected_provenance=provenance,
        )

    incompatible = {**provenance, "epochs": 1499}
    with pytest.raises(ValueError, match="provenance"):
        load_final_checkpoint(
            target,
            expected_sha256=digest,
            expected_provenance=incompatible,
        )


def test_fresh_manifest_is_exactly_25_per_family_and_disjoint_from_history() -> None:
    historical = _historical_cases()
    campaign = build_fresh_campaign_manifest(
        historical_cases=historical,
        master_seed=CONFIRMATION_GEOMETRY_SEED,
        nx=6,
        ny=3,
    )
    fresh = campaign.cases
    manifest = campaign.manifest
    proof = validate_fresh_campaign(fresh, historical_cases=historical)

    assert len(historical) == 60
    assert len(fresh) == 50
    assert {
        family: sum(case.family == family for case in fresh)
        for family in ("dogbone", "centre_notched")
    } == {"dogbone": 25, "centre_notched": 25}

    historical_geometry = {case.geometry_id for case in historical}
    historical_meshes = {case.mesh_id for case in historical}
    historical_seeds = {case.seed for case in historical}
    historical_nodes = {node for case in historical for node in case.node_ids}
    assert historical_geometry.isdisjoint(case.geometry_id for case in fresh)
    assert historical_meshes.isdisjoint(case.mesh_id for case in fresh)
    assert historical_seeds.isdisjoint(case.seed for case in fresh)
    assert historical_nodes.isdisjoint(node for case in fresh for node in case.node_ids)

    assert manifest["schema_version"] == "confirm-v1"
    assert manifest["role"] == "LEGACY"
    assert manifest["no_release_evidence"] is True
    assert manifest["master_seed"] == CONFIRMATION_GEOMETRY_SEED
    assert manifest["case_count"] == 50
    assert manifest["family_counts"] == {"dogbone": 25, "centre_notched": 25}
    assert len(manifest["cases"]) == 50
    assert len(campaign.sha256) == 64
    assert (
        campaign.sha256
        == hashlib.sha256(
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )
    assert proof == {
        "fresh_geometry_ids_unique": True,
        "fresh_mesh_ids_unique": True,
        "fresh_rng_seeds_unique": True,
        "fresh_node_ids_unique": True,
        "geometry_ids_disjoint": True,
        "mesh_ids_disjoint": True,
        "rng_seeds_disjoint": True,
        "mesh_qualified_node_ids_disjoint": True,
    }


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("geometry_id", "geometry"),
        ("mesh_id", "mesh|node"),
        ("seed", "seed"),
    ),
)
def test_fresh_campaign_rejects_every_historical_identity_collision(
    field: str,
    message: str,
) -> None:
    historical = _historical_cases()
    fresh, _ = _fresh_campaign()
    historical_dogbone = next(case for case in historical if case.family == "dogbone")
    fresh_dogbone_index = next(
        index for index, case in enumerate(fresh) if case.family == "dogbone"
    )
    collision = replace(
        fresh[fresh_dogbone_index],
        **{field: getattr(historical_dogbone, field)},
    )
    invalid = (
        fresh[:fresh_dogbone_index] + (collision,) + fresh[fresh_dogbone_index + 1 :]
    )

    with pytest.raises(ValueError, match=message):
        validate_fresh_campaign(invalid, historical_cases=historical)


def test_fresh_campaign_rejects_internal_duplicates_without_resampling() -> None:
    historical = _historical_cases()
    fresh, _ = _fresh_campaign()
    dogbone_indices = [
        index for index, case in enumerate(fresh) if case.family == "dogbone"
    ]
    invalid = list(fresh)
    invalid[dogbone_indices[-1]] = invalid[dogbone_indices[0]]

    with pytest.raises(ValueError, match="duplicate|unique"):
        validate_fresh_campaign(tuple(invalid), historical_cases=historical)


def test_whole_geometry_bootstrap_and_wilson_statistics_are_deterministic() -> None:
    cases, _ = _fresh_campaign()
    records = tuple(_metric(case, index) for index, case in enumerate(cases))

    first = summarize_confirmation_statistics(
        records,
        seed=CONFIRMATION_STATISTICS_SEED,
        bootstrap_resamples=256,
        pass_threshold=0.15,
    )
    repeated = summarize_confirmation_statistics(
        records,
        seed=CONFIRMATION_STATISTICS_SEED,
        bootstrap_resamples=256,
        pass_threshold=0.15,
    )
    reordered = summarize_confirmation_statistics(
        tuple(reversed(records)),
        seed=CONFIRMATION_STATISTICS_SEED,
        bootstrap_resamples=256,
        pass_threshold=0.15,
    )

    assert first == repeated == reordered
    assert all(math.isfinite(value) for value in _recursive_floats(first))
    for family in ("dogbone", "centre_notched"):
        family_records = [record for record in records if record["family"] == family]
        expected_p = math.sqrt(
            sum(float(record["P_squared_error"]) for record in family_records)
            / sum(float(record["P_squared_reference"]) for record in family_records)
        )
        stress = first[family]["relative_l2_P"]
        assert stress["estimate"] == pytest.approx(expected_p)
        assert stress["bootstrap_95_ci"][0] <= stress["estimate"]
        assert stress["estimate"] <= stress["bootstrap_95_ci"][1]

        passing = first[family]["P_within_15_percent"]
        assert 0.0 <= passing["wilson_95_ci"][0] <= passing["fraction"]
        assert passing["fraction"] <= passing["wilson_95_ci"][1] <= 1.0
        assert passing["count"] == 25
        assert passing["passed"] == 25


@dataclass(frozen=True)
class _FakeFemResult:
    converged: bool
    failure_reason: str | None
    displacement: np.ndarray | None = None
    first_piola: np.ndarray | None = None
    solve_seconds: float = 0.01
    newton_iterations: int = 3


def _complete_artifact_writer(
    _model: object,
    pairs: Sequence[tuple[SpikeMesh, object]],
    artifact_dir: Path,
) -> Mapping[str, object]:
    prediction_dir = artifact_dir / "predictions"
    plot_dir = artifact_dir / "plots"
    prediction_dir.mkdir(parents=True)
    plot_dir.mkdir(parents=True)
    geometry_ids = sorted(mesh.geometry_id for mesh, _result in pairs)
    for geometry_id in geometry_ids:
        (prediction_dir / f"{geometry_id}.npz").write_bytes(b"LEGACY")
        (plot_dir / f"{geometry_id}_fields.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"></svg>\n',
            encoding="utf-8",
        )
    return {
        "directory": str(artifact_dir.resolve()),
        "prediction_npz_count": len(geometry_ids),
        "comparison_plot_svg_count": len(geometry_ids),
        "prediction_geometry_ids": geometry_ids,
        "plot_geometry_ids": geometry_ids,
    }


def _evaluate_pairs(
    model: object,
    pairs: Sequence[tuple[SpikeMesh, object]],
) -> tuple[Mapping[str, object], ...]:
    assert isinstance(model, torch.nn.Module)
    assert not model.training
    assert not torch.is_grad_enabled()
    return tuple(_metric(mesh, index) for index, (mesh, _result) in enumerate(pairs))


def _evaluate_bad_stress_pairs(
    model: object,
    pairs: Sequence[tuple[SpikeMesh, object]],
) -> tuple[Mapping[str, object], ...]:
    records = []
    for index, (mesh, _result) in enumerate(pairs):
        metric = dict(_metric(mesh, index))
        metric.update(
            {
                "P_squared_error": 0.20**2,
                "relative_l2_P": 0.20,
            }
        )
        records.append(metric)
    return tuple(records)


def _successful_solver(mesh: SpikeMesh) -> _FakeFemResult:
    point_count = len(mesh.points)
    return _FakeFemResult(
        converged=True,
        failure_reason=None,
        displacement=np.zeros((point_count, 2)),
        first_piola=np.zeros((point_count, 2, 2)),
    )


def test_confirmation_runner_is_evaluation_only_and_writes_complete_legacy_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = _historical_cases()
    cases, manifest = _fresh_campaign()
    model = MixedPhysicsOperator(_tiny_config())
    initial_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    report_path = tmp_path / "iteration8_confirmation.json"
    artifact_dir = tmp_path / "iteration8_artifacts"

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("confirmation must not train or backpropagate")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)

    report = run_confirmation(
        model=model,
        checkpoint_sha256="a" * 64,
        checkpoint_provenance=_provenance(),
        cases=cases,
        historical_cases=historical,
        campaign_manifest=manifest,
        fem_solver=_successful_solver,
        evaluator=_evaluate_pairs,
        artifact_writer=_complete_artifact_writer,
        artifact_dir=artifact_dir,
        report_path=report_path,
        bootstrap_seed=CONFIRMATION_STATISTICS_SEED,
        bootstrap_resamples=64,
    )

    for name, expected in initial_state.items():
        torch.testing.assert_close(
            model.state_dict()[name],
            expected,
            atol=0.0,
            rtol=0.0,
        )
    assert report["schema_version"] == "iteration8-confirmation-v1"
    assert report["role"] == "LEGACY"
    assert report["no_release_evidence"] is True
    assert report["no_tuning"] is True
    assert report["execution_mode"] == "evaluation_only"
    assert report["protocol"]["training_performed"] is False
    assert report["protocol"]["optimizer_steps"] == 0
    assert report["protocol"]["backward_calls"] == 0
    assert report["protocol"]["minimum_converged_per_family"] == 24
    assert report["protocol"]["minimum_converged_total"] == 48
    assert report["campaign"]["attempted"] == 50
    assert report["campaign"]["converged"] == 50
    assert report["campaign"]["failed"] == 0
    assert report["campaign"]["families"] == {
        "dogbone": {"attempted": 25, "converged": 25, "failed": 0},
        "centre_notched": {"attempted": 25, "converged": 25, "failed": 0},
    }

    expected_geometry_ids = sorted(case.geometry_id for case in cases)
    assert report["artifacts"]["prediction_npz_count"] == 50
    assert report["artifacts"]["comparison_plot_svg_count"] == 50
    assert report["artifacts"]["prediction_geometry_ids"] == expected_geometry_ids
    assert report["artifacts"]["plot_geometry_ids"] == expected_geometry_ids
    assert len(tuple((artifact_dir / "predictions").glob("*.npz"))) == 50
    assert len(tuple((artifact_dir / "plots").glob("*.svg"))) == 50

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted == report
    json.dumps(persisted, allow_nan=False)
    assert all(math.isfinite(value) for value in _recursive_floats(persisted))


def test_confirmation_runner_requires_stress_point_estimates_to_confirm(
    tmp_path: Path,
) -> None:
    historical = _historical_cases()
    cases, manifest = _fresh_campaign()

    report = run_confirmation(
        model=MixedPhysicsOperator(_tiny_config()),
        checkpoint_sha256="a" * 64,
        checkpoint_provenance=_provenance(),
        cases=cases,
        historical_cases=historical,
        campaign_manifest=manifest,
        fem_solver=_successful_solver,
        evaluator=_evaluate_bad_stress_pairs,
        artifact_writer=_complete_artifact_writer,
        artifact_dir=tmp_path / "artifacts",
        report_path=tmp_path / "iteration8_confirmation.json",
        bootstrap_seed=CONFIRMATION_STATISTICS_SEED,
        bootstrap_resamples=32,
    )

    assert report["status"] == "NOT_CONFIRMED"
    for family in ("dogbone", "centre_notched"):
        gates = report["gates"]["families"][family]
        assert gates["point_estimate_pass"] is False
        assert gates["relative_l2_P"] > 0.15


def test_confirmation_runner_refuses_a_campaign_below_convergence_threshold(
    tmp_path: Path,
) -> None:
    historical = _historical_cases()
    cases, manifest = _fresh_campaign()
    failed_dogbones = {case.geometry_id for case in cases if case.family == "dogbone"}
    failed_dogbones = set(sorted(failed_dogbones)[:2])
    calls = {"evaluate": 0, "artifacts": 0}

    def solver(mesh: SpikeMesh) -> _FakeFemResult:
        if mesh.geometry_id in failed_dogbones:
            return _FakeFemResult(False, "deliberate nonconvergence")
        return _successful_solver(mesh)

    def evaluator(*_args: object, **_kwargs: object) -> object:
        calls["evaluate"] += 1
        raise AssertionError("invalid campaigns must not be evaluated")

    def artifact_writer(*_args: object, **_kwargs: object) -> object:
        calls["artifacts"] += 1
        raise AssertionError("invalid campaigns must not write artifacts")

    report_path = tmp_path / "must_not_exist.json"
    with pytest.raises(ValueError, match="converg|24"):
        run_confirmation(
            model=MixedPhysicsOperator(_tiny_config()),
            checkpoint_sha256="a" * 64,
            checkpoint_provenance=_provenance(),
            cases=cases,
            historical_cases=historical,
            campaign_manifest=manifest,
            fem_solver=solver,
            evaluator=evaluator,
            artifact_writer=artifact_writer,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
            bootstrap_seed=CONFIRMATION_STATISTICS_SEED,
            bootstrap_resamples=32,
        )

    assert calls == {"evaluate": 0, "artifacts": 0}
    assert not report_path.exists()


def test_confirmation_runner_fails_closed_on_partial_case_artifacts(
    tmp_path: Path,
) -> None:
    historical = _historical_cases()
    cases, manifest = _fresh_campaign()

    def incomplete_writer(
        _model: object,
        pairs: Sequence[tuple[SpikeMesh, object]],
        artifact_dir: Path,
    ) -> Mapping[str, object]:
        complete = dict(_complete_artifact_writer(_model, pairs[:-1], artifact_dir))
        return complete

    report_path = tmp_path / "must_not_exist.json"
    with pytest.raises(ValueError, match="artifact|complete|50"):
        run_confirmation(
            model=MixedPhysicsOperator(_tiny_config()),
            checkpoint_sha256="a" * 64,
            checkpoint_provenance=_provenance(),
            cases=cases,
            historical_cases=historical,
            campaign_manifest=manifest,
            fem_solver=_successful_solver,
            evaluator=_evaluate_pairs,
            artifact_writer=incomplete_writer,
            artifact_dir=tmp_path / "artifacts",
            report_path=report_path,
            bootstrap_seed=CONFIRMATION_STATISTICS_SEED,
            bootstrap_resamples=32,
        )

    assert not report_path.exists()
