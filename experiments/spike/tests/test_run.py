from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spike.geometry import sample_geometry_cases
from experiments.spike.run import (
    FINAL_TRAINING_EPOCHS,
    ITERATION3_SCREEN_SHA256,
    ITERATION5_PHYSICS_WEIGHTS,
    ITERATION6_PHYSICS_WEIGHTS,
    LEARNING_RATES,
    LR_SWEEP_EPOCHS,
    S1_FEM_CONTENT_SHA256,
    UPSTREAM_APPLIED_DISPLACEMENT,
    UPSTREAM_CHECKPOINT_SHA256,
    UPSTREAM_CHECKPOINT_URL,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    FineTuneStage,
    ServedStressPhysicsWeights,
    _apply_fine_tune_stage,
    _architecture_iteration_evidence,
    _element_centroid_fields,
    _element_tensor_dataset,
    _equilibrium_residual_prefix,
    _evaluate_model,
    _evaluate_model_element_quadrature,
    _hybrid_batch_loss,
    _lumped_nodal_areas,
    _new_pretrained_model,
    _partial_traction_boundary_quadrature,
    _partial_traction_loss,
    _regularity_check,
    _screen_guardrail_checks,
    _section_resultant_loss,
    _section_resultant_quadrature,
    _select_learning_rate,
    _staged_optimizer_groups,
    _summarize_screen_folds,
    _supervised_validation_score,
    _tensor_dataset,
    _top_free_boundary_quadrature,
    _traction_loss,
    _train_staged_epochs,
    _weighted_squared_norm,
    _write_evaluation_artifacts,
    build_gate_document,
    run_iteration3_screen,
    run_iteration4_screen,
    run_iteration5_screen,
    run_iteration6_screen,
    run_iteration7_screen,
    solve_and_maybe_train,
    split_by_geometry,
    validate_results,
    write_json_strict,
)


def test_split_is_deterministic_grouped_and_disjoint() -> None:
    cases = sample_geometry_cases(master_seed=20260809, count_per_family=30, nx=6, ny=3)
    first = split_by_geometry(cases, master_seed=991)
    reordered = split_by_geometry(tuple(reversed(cases)), master_seed=991)

    assert [case.geometry_id for case in first.train] == [
        case.geometry_id for case in reordered.train
    ]
    assert [case.geometry_id for case in first.evaluate] == [
        case.geometry_id for case in reordered.evaluate
    ]
    for family in ("dogbone", "centre_notched"):
        train = tuple(case for case in first.train if case.family == family)
        evaluate = tuple(case for case in first.evaluate if case.family == family)
        assert len(train) == 20
        assert len(evaluate) == 10
        assert {case.geometry_id for case in train}.isdisjoint(
            case.geometry_id for case in evaluate
        )
        assert {case.mesh_id for case in train}.isdisjoint(
            case.mesh_id for case in evaluate
        )
        assert {case.seed for case in train}.isdisjoint(case.seed for case in evaluate)
        train_nodes = {node_id for case in train for node_id in case.node_ids}
        evaluate_nodes = {node_id for case in evaluate for node_id in case.node_ids}
        assert train_nodes.isdisjoint(evaluate_nodes)


@dataclass(frozen=True)
class _FakeSolve:
    converged: bool
    failure_reason: str | None


def test_protocol_stops_before_training_below_fifty_converged() -> None:
    cases = tuple(range(60))
    trainer_calls: list[int] = []

    def solver(case: int) -> _FakeSolve:
        return _FakeSolve(case < 49, None if case < 49 else "deliberate failure")

    def trainer(_: object) -> object:
        trainer_calls.append(1)
        return object()

    outcome = solve_and_maybe_train(cases, fem_solver=solver, trainer=trainer)

    assert outcome.attempted == 60
    assert outcome.converged == 49
    assert outcome.failed == 11
    assert outcome.stopped_before_training
    assert outcome.training_result is None
    assert outcome.failure_reasons == {"deliberate failure": 11}
    assert trainer_calls == []


def _valid_results() -> dict[str, object]:
    family = {
        "evaluation_count": 10,
        "metrics": {
            "relative_l2_u": 0.1,
            "relative_l2_P": 0.2,
            "worst_relative_peak_stress_error": 0.3,
        },
        "fem": {
            "attempted": 30,
            "converged": 30,
            "failed": 0,
            "failure_reasons": {},
        },
        "timing": {
            "mean_fem_seconds": 1.2,
            "mean_inference_seconds": 0.01,
        },
        "peak_qoi": {
            "status": "well_posed",
            "reason": "masked smooth-root refinement check stabilized",
        },
    }
    return {
        "schema_version": "1.0",
        "role": "LEGACY",
        "no_release_evidence": True,
        "execution_profile": {
            "applied_displacement_mm": UPSTREAM_APPLIED_DISPLACEMENT,
        },
        "model": {
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_checkpoint": {
                "url": UPSTREAM_CHECKPOINT_URL,
                "sha256": UPSTREAM_CHECKPOINT_SHA256,
                "epoch": 1424,
            },
            "architecture_adaptation": (
                "exact vendored upstream PI_GINOT encoder and displacement decoder; "
                "a direct four-component first-Piola conditioned MLP head is added "
                "with y_max/x_max input and E*u_delta/x_max output scaling"
            ),
            "stress_head": {
                "variant": "conditioned_mlp",
                "conditioning": ["y_max/x_max"],
                "output_scale": "E*u_delta/x_max",
                "youngs_modulus_MPa": 760.0,
                "output_layer_initialization": "zeros",
            },
            "served_stress": "P_served = P_predicted (direct second head)",
            "material_stress_role": "in-plane consistency oracle only",
        },
        "artifacts": {
            "checkpoint": {"sha256": "a" * 64},
            "prediction_npz_count": 20,
            "comparison_plot_svg_count": 20,
        },
        "lr_sweep": {
            "learning_rates": list(LEARNING_RATES),
            "epochs_per_trial": LR_SWEEP_EPOCHS,
            "final_training_epochs": FINAL_TRAINING_EPOCHS,
            "gradient_clip_norm": 1.0,
            "supervised_sampling": (
                "reference-area-proportional nodal sampling with replacement"
            ),
            "fixed_loss_weights": {"equilibrium_on_P_served": 1.0},
        },
        "architecture_iteration": {
            "iteration": 3,
            "selected_variant": "conditioned_mlp",
            "selection": {
                "training_geometry_count": 40,
                "held_out_geometry_accessed": False,
                "fold_count": 2,
                "development_geometries_per_family_per_fold": 4,
                "epochs_per_candidate_fold": 300,
                "learning_rate": 1e-3,
                "peak_supervised_P_weight": 0.0,
                "variants": {
                    "conditioned_mlp": {
                        "pooled_worst_family_relative_l2_P": 0.1186927794,
                        "guardrails_passed": None,
                        "peak_supervised_P_weight": 0.0,
                    },
                    "fully_conditioned_mlp": {
                        "pooled_worst_family_relative_l2_P": 0.12,
                        "guardrails_passed": False,
                        "peak_supervised_P_weight": 0.0,
                    },
                },
            },
            "held_out_evaluation_status": "confirmatory_not_pristine",
        },
        "families": {"dogbone": family, "centre_notched": family},
    }


def _valid_screen() -> dict[str, object]:
    return {
        "role": "LEGACY",
        "no_release_evidence": True,
        "upstream_checkpoint_sha256": UPSTREAM_CHECKPOINT_SHA256,
        "fem_content_sha256": S1_FEM_CONTENT_SHA256,
        "architecture_iteration": _valid_results()["architecture_iteration"],
    }


def test_results_schema_requires_complete_legacy_finite_family_metrics(
    tmp_path: Path,
) -> None:
    document = _valid_results()
    validate_results(document)
    target = tmp_path / "results.json"
    write_json_strict(target, document)
    assert json.loads(target.read_text(encoding="utf-8")) == document

    bad_role = {**document, "role": "RELEASE"}
    with pytest.raises(ValueError, match="LEGACY"):
        validate_results(bad_role)

    bad_metric = _valid_results()
    families = bad_metric["families"]
    assert isinstance(families, dict)
    dogbone = families["dogbone"]
    assert isinstance(dogbone, dict)
    metrics = dogbone["metrics"]
    assert isinstance(metrics, dict)
    metrics["relative_l2_P"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_results(bad_metric)


def test_results_schema_rejects_stale_training_horizon() -> None:
    document = copy.deepcopy(_valid_results())
    document["lr_sweep"]["final_training_epochs"] = 120

    with pytest.raises(ValueError, match="final_training_epochs"):
        validate_results(document)


def test_results_schema_rejects_legacy_ten_percent_strain_profile() -> None:
    document = copy.deepcopy(_valid_results())
    profile = document["execution_profile"]
    profile.pop("applied_displacement_mm")
    profile["axial_engineering_strain"] = 0.1

    with pytest.raises(ValueError, match="applied_displacement_mm"):
        validate_results(document)


def test_results_schema_rejects_reimplemented_model_report() -> None:
    document = copy.deepcopy(_valid_results())
    document["model"]["architecture_adaptation"] = "clean reconstruction"

    with pytest.raises(ValueError, match="architecture_adaptation"):
        validate_results(document)


def test_results_schema_requires_conditioned_direct_stress_head() -> None:
    document = copy.deepcopy(_valid_results())
    document["model"].pop("stress_head")

    with pytest.raises(ValueError, match="model.stress_head"):
        validate_results(document)


def test_results_schema_requires_quarantined_architecture_selection() -> None:
    missing = copy.deepcopy(_valid_results())
    missing.pop("architecture_iteration")
    with pytest.raises(ValueError, match="architecture_iteration"):
        validate_results(missing)

    leaked = copy.deepcopy(_valid_results())
    leaked["architecture_iteration"]["selection"]["held_out_geometry_accessed"] = True
    with pytest.raises(ValueError, match="held_out_geometry_accessed"):
        validate_results(leaked)


def test_architecture_evidence_fails_closed_without_a_valid_screen(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-screen.json"
    with pytest.raises(FileNotFoundError, match="architecture screen"):
        _architecture_iteration_evidence(missing)

    screen = _valid_screen()
    screen_path = tmp_path / "screen.json"
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    loaded = _architecture_iteration_evidence(screen_path)
    assert loaded["selected_variant"] == "conditioned_mlp"
    assert loaded["decision"] == (
        "candidate rejected by the training-only guardrails; retain conditioned_mlp"
    )
    assert "root_cause" not in loaded
    assert "correction" not in loaded

    screen["architecture_iteration"]["selection"]["held_out_geometry_accessed"] = True
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    with pytest.raises(ValueError, match="held-out"):
        _architecture_iteration_evidence(screen_path)

    screen["architecture_iteration"]["selection"]["held_out_geometry_accessed"] = False
    screen["architecture_iteration"]["selection"]["peak_supervised_P_weight"] = 0.5
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    with pytest.raises(ValueError, match="peak-supervised"):
        _architecture_iteration_evidence(screen_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("role", "RELEASE", "LEGACY"),
        ("no_release_evidence", False, "LEGACY"),
        ("upstream_checkpoint_sha256", "0" * 64, "upstream checkpoint"),
        ("fem_content_sha256", "0" * 64, "FEM content"),
        ("architecture_iteration", None, "architecture_iteration"),
    ),
)
def test_architecture_evidence_rejects_incompatible_screen_metadata(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    screen = _valid_screen()
    screen[field] = value
    screen_path = tmp_path / f"{field}.json"
    screen_path.write_text(json.dumps(screen), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _architecture_iteration_evidence(screen_path)


def test_architecture_evidence_requires_a_json_object(tmp_path: Path) -> None:
    screen_path = tmp_path / "list.json"
    screen_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        _architecture_iteration_evidence(screen_path)


def test_architecture_evidence_rejects_incomplete_or_unselected_candidates(
    tmp_path: Path,
) -> None:
    def reject(screen: dict[str, object], message: str) -> None:
        screen_path = tmp_path / "invalid-screen.json"
        screen_path.write_text(json.dumps(screen), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            _architecture_iteration_evidence(screen_path)

    screen = _valid_screen()
    screen["architecture_iteration"]["selection"] = None
    reject(screen, "selection")

    screen = _valid_screen()
    screen["architecture_iteration"]["selection"]["variants"] = None
    reject(screen, "candidate results")

    screen = _valid_screen()
    del screen["architecture_iteration"]["selection"]["variants"][
        "fully_conditioned_mlp"
    ]
    reject(screen, "predeclared candidate")

    screen = _valid_screen()
    screen["architecture_iteration"]["selection"]["variants"][
        "fully_conditioned_mlp"
    ] = "invalid"
    reject(screen, "candidate result")

    screen = _valid_screen()
    screen["architecture_iteration"]["selection"]["variants"]["fully_conditioned_mlp"][
        "peak_supervised_P_weight"
    ] = 0.5
    reject(screen, "peak-supervised")

    screen = _valid_screen()
    screen["architecture_iteration"]["selected_variant"] = "unknown"
    reject(screen, "incompatible variant")

    screen = _valid_screen()
    screen["architecture_iteration"]["selected_variant"] = "fully_conditioned_mlp"
    reject(screen, "did not promote")

    screen = _valid_screen()
    screen["architecture_iteration"]["selected_variant"] = "fully_conditioned_mlp"
    screen["architecture_iteration"]["selection"]["variants"]["fully_conditioned_mlp"][
        "guardrails_passed"
    ] = True
    reject(screen, "incompatible variant")

    screen = _valid_screen()
    screen["architecture_iteration"]["iteration"] = 2
    reject(screen, "incompatible variant")


def test_default_architecture_evidence_is_digest_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.spike.run as spike_run

    assert len(ITERATION3_SCREEN_SHA256) == 64
    monkeypatch.setattr(spike_run, "ITERATION3_SCREEN_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        spike_run._architecture_iteration_evidence()


def test_results_schema_rejects_peak_confounded_architecture_screen() -> None:
    document = copy.deepcopy(_valid_results())
    selection = document["architecture_iteration"]["selection"]
    selection["peak_supervised_P_weight"] = 0.5

    with pytest.raises(ValueError, match="peak_supervised_P_weight"):
        validate_results(document)


def test_results_schema_requires_the_pinned_upstream_checkpoint() -> None:
    document = copy.deepcopy(_valid_results())
    document["model"]["upstream_checkpoint"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="upstream_checkpoint.sha256"):
        validate_results(document)


def test_results_validation_does_not_import_the_torch_backed_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def import_without_model(name: str, *args: object, **kwargs: object) -> object:
        if name == "experiments.spike.model":
            raise AssertionError("report validation imported the model")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_model)

    validate_results(_valid_results())


def test_spatial_l2_helpers_use_lumped_reference_area() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=int)
    weights = _lumped_nodal_areas(points, cells)

    np.testing.assert_allclose(weights, [1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 6.0])
    assert weights.sum() == pytest.approx(1.0)
    vector_field = np.ones((4, 2))
    assert _weighted_squared_norm(vector_field, weights) == pytest.approx(2.0)


def test_element_centroid_fields_are_exact_for_an_affine_p1_solution() -> None:
    from experiments.spike.fem import first_piola_plane_stress

    points = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]], dtype=float)
    cells = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=int)
    displacement_gradient = np.asarray([[0.10, 0.02], [-0.03, 0.05]])
    translation = np.asarray([0.4, -0.2])
    displacement = points @ displacement_gradient.T + translation

    centroids, centroid_u, stress, areas = _element_centroid_fields(
        points,
        cells,
        displacement,
    )

    expected_stress = first_piola_plane_stress(np.eye(2) + displacement_gradient)
    np.testing.assert_allclose(centroids, points[cells].mean(axis=1))
    np.testing.assert_allclose(
        centroid_u, centroids @ displacement_gradient.T + translation
    )
    np.testing.assert_allclose(stress, np.broadcast_to(expected_stress, stress.shape))
    np.testing.assert_allclose(areas, [1.0, 1.0])


def test_element_centroid_fields_reject_degenerate_cells() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    cells = np.asarray([[0, 1, 2]], dtype=int)

    with pytest.raises(ValueError, match="positive-area triangles"):
        _element_centroid_fields(points, cells, np.zeros((3, 2)))


def test_gate_document_covers_every_required_g1_check() -> None:
    results = {
        **_valid_results(),
        "commit": "abc123",
        "worktree_content_sha256": "f" * 64,
        "execution_profile_hash": "e" * 64,
        "protocol": {
            "attempted": 60,
            "converged": 60,
            "failed": 0,
            "split_proof": {
                "geometry_ids_disjoint": True,
                "mesh_ids_disjoint": True,
                "rng_seeds_disjoint": True,
                "mesh_qualified_node_ids_disjoint": True,
                "outcome_independent_preassignment": True,
            },
        },
        "fem_sanity": {"max_error": 1e-12, "threshold": 1e-6},
    }
    gate = build_gate_document(
        results,
        isolation_passed=True,
        verification={"spike_tests": "14 passed"},
        timestamp_utc="2026-08-09T00:00:00Z",
    )

    assert gate["role"] == "LEGACY"
    assert gate["no_release_evidence"] is True
    assert gate["verdict"] == "PASS"
    assert {check["id"] for check in gate["checks"]} == {
        "G1.solves",
        "G1.sanity",
        "G1.split",
        "G1.report",
        "G1.isolation",
    }


def test_failed_regularity_refinement_emits_strict_diagnostic_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.spike.fem as fem
    from experiments.spike.fem import FEMResult

    cases = sample_geometry_cases(master_seed=4, count_per_family=1, nx=6, ny=3)
    monkeypatch.setattr(
        fem,
        "solve_case",
        lambda _: FEMResult(False, "deliberate failure", None, None, 0.0, 0),
    )

    report = _regularity_check(cases)

    assert report["dogbone"]["status"] == "singular_or_diagnostic"
    assert report["dogbone"]["worst_last_relative_change"] is None
    json.dumps(report, allow_nan=False)


def test_learning_rate_selection_ignores_failed_trials_without_infinity() -> None:
    trials = (
        {"learning_rate": 1e-3, "status": "failed", "failure_reason": "inverted F"},
        {
            "learning_rate": 3e-4,
            "status": "completed",
            "training_only_development_score": 0.4,
        },
        {
            "learning_rate": 1e-4,
            "status": "completed",
            "training_only_development_score": 0.5,
        },
    )

    assert _select_learning_rate(trials) == pytest.approx(3e-4)
    with pytest.raises(RuntimeError, match="every learning-rate trial failed"):
        _select_learning_rate(trials[:1])


def test_development_score_targets_the_worst_family_stress_error() -> None:
    import torch

    class _StressErrorModel:
        def eval(self) -> None:
            return None

        def __call__(
            self,
            query: torch.Tensor,
            _boundary: torch.Tensor,
            _u_delta: torch.Tensor,
            _x_max: torch.Tensor,
            _y_max: torch.Tensor,
        ) -> SimpleNamespace:
            relative_error = query[:, :1, :1]
            stress = torch.ones((*query.shape[:2], 2, 2)) * (
                1.0 + relative_error[..., None]
            )
            return SimpleNamespace(
                u=torch.full((*query.shape[:2], 2), 100.0),
                P_served=stress,
            )

    errors = torch.tensor([0.1, 0.1, 0.3, 0.3])
    data = {
        "meshes": tuple(
            SimpleNamespace(family=family)
            for family in ("dogbone", "dogbone", "centre_notched", "centre_notched")
        ),
        "points": errors[:, None, None].expand(4, 2, 2),
        "boundary": torch.zeros(4, 3, 2),
        "u_delta": torch.ones(4),
        "x_max": torch.ones(4),
        "y_max": torch.ones(4),
        "displacement": torch.zeros(4, 2, 2),
        "stress": torch.ones(4, 2, 2, 2),
        "nodal_areas": torch.ones(4, 2),
    }

    score = _supervised_validation_score(_StressErrorModel(), data, range(4))

    assert score == pytest.approx(0.3)


def test_training_uses_the_upstream_loading_and_full_training_horizon() -> None:
    assert UPSTREAM_APPLIED_DISPLACEMENT == pytest.approx(1.0)
    assert LR_SWEEP_EPOCHS == 150
    assert FINAL_TRAINING_EPOCHS == 1500


def test_tensor_dataset_uses_the_upstream_fixed_displacement() -> None:
    from experiments.spike.fem import solve_case

    cases = sample_geometry_cases(master_seed=12, count_per_family=1, nx=6, ny=3)
    pairs = tuple(
        (
            case,
            solve_case(case, applied_displacement=UPSTREAM_APPLIED_DISPLACEMENT),
        )
        for case in cases
    )

    data = _tensor_dataset(pairs)

    np.testing.assert_allclose(
        data["u_delta"].numpy(),
        np.full(len(cases), UPSTREAM_APPLIED_DISPLACEMENT),
    )
    qoi_weights = data["qoi_weights"].numpy()
    assert qoi_weights.shape == data["nodal_areas"].shape
    for row, case in enumerate(cases):
        assert np.count_nonzero(qoi_weights[row]) == len(case.qoi_node_ids)
        assert np.all(qoi_weights[row, case.qoi_node_ids] > 0.0)
    assert data["top_free_points"].shape == (2, 6, 2)
    assert data["top_free_normals"].shape == (2, 6, 2)
    assert data["partial_traction_points"].shape == (2, 12, 2)
    assert data["section_points"].shape == (2, 8, 16, 2)
    assert data["section_weights"].shape == (2, 8, 16)


def test_element_tensor_dataset_derives_stress_from_displacement_not_patch_labels() -> (
    None
):
    from experiments.spike.fem import solve_case

    cases = sample_geometry_cases(master_seed=13, count_per_family=1, nx=6, ny=3)
    pairs = []
    for case in cases:
        result = solve_case(case, applied_displacement=UPSTREAM_APPLIED_DISPLACEMENT)
        corrupted = SimpleNamespace(
            converged=result.converged,
            displacement=result.displacement,
            first_piola=np.full_like(result.first_piola, 1.0e6),
        )
        pairs.append((case, corrupted))

    data = _element_tensor_dataset(tuple(pairs), validate_patch_recovery=False)

    element_count = len(cases[0].cells)
    assert data["stress_quadrature_points"].shape == (2, 3 * element_count, 2)
    assert data["stress_quadrature_targets"].shape == (
        2,
        3 * element_count,
        2,
        2,
    )
    assert data["stress_quadrature_displacement"].shape == (
        2,
        3 * element_count,
        2,
    )
    assert data["stress_quadrature_weights"].shape == (2, 3 * element_count)
    assert data["element_stress"].shape == (2, element_count, 2, 2)
    assert data["element_areas"].shape == (2, element_count)
    assert np.max(np.abs(data["element_stress"].numpy())) < 1.0e4
    assert np.all(data["element_areas"].numpy() > 0.0)
    np.testing.assert_allclose(
        data["stress_quadrature_weights"].numpy().sum(axis=1),
        data["element_areas"].numpy().sum(axis=1),
        rtol=5e-7,
    )
    with pytest.raises(ValueError, match="cached nodal patch recovery"):
        _element_tensor_dataset(tuple(pairs))


def test_element_quadrature_evaluation_reports_legacy_and_raw_stress_metrics() -> None:
    import torch

    from experiments.spike.fem import first_piola_plane_stress

    cases = sample_geometry_cases(master_seed=14, count_per_family=1, nx=6, ny=3)
    gradient = np.asarray([[0.08, 0.01], [-0.02, 0.03]])
    translation = np.asarray([0.2, -0.1])
    stress = first_piola_plane_stress(np.eye(2) + gradient)
    pairs = tuple(
        (
            case,
            SimpleNamespace(
                converged=True,
                displacement=case.points @ gradient.T + translation,
                first_piola=np.broadcast_to(stress, (len(case.points), 2, 2)).copy(),
            ),
        )
        for case in cases
    )

    class _AffineModel:
        def eval(self) -> None:
            return None

        def __call__(
            self,
            query: torch.Tensor,
            _boundary: torch.Tensor,
            _u_delta: torch.Tensor,
            _x_max: torch.Tensor,
            _y_max: torch.Tensor,
        ) -> SimpleNamespace:
            tensor_gradient = torch.as_tensor(gradient, dtype=query.dtype)
            tensor_translation = torch.as_tensor(translation, dtype=query.dtype)
            tensor_stress = torch.as_tensor(stress, dtype=query.dtype)
            return SimpleNamespace(
                u=query @ tensor_gradient.T + tensor_translation,
                P_served=tensor_stress.expand(*query.shape[:2], 2, 2),
            )

    evaluation = _evaluate_model_element_quadrature(_AffineModel(), pairs)

    for metrics in evaluation.family_metrics.values():
        assert metrics["relative_l2_P"] < 1e-6
        assert metrics["legacy_nodal_relative_l2_P"] < 1e-6
        assert metrics["P_l2_quadrature"] == "positive three-point element area"
    for metrics in evaluation.per_case_metrics.values():
        assert metrics["relative_l2_P"] < 1e-6
        assert "legacy_nodal_relative_l2_P" in metrics


def test_supervised_sampling_uses_reference_area_probabilities() -> None:
    import torch

    from experiments.spike.run import _sample_supervised_indices

    nodal_areas = torch.tensor([[0.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    sampled = _sample_supervised_indices(
        nodal_areas,
        count=16,
        generator=torch.Generator().manual_seed(9),
    )

    assert sampled.shape == (2, 16)
    assert torch.equal(sampled[0], torch.full((16,), 2))
    assert torch.equal(sampled[1], torch.full((16,), 1))


def test_top_free_boundary_quadrature_has_outward_unit_normals() -> None:
    mesh = sample_geometry_cases(master_seed=31, count_per_family=1, nx=6, ny=3)[0]

    points, normals, weights = _top_free_boundary_quadrature(mesh)

    assert points.shape == (6, 2)
    assert normals.shape == (6, 2)
    assert weights.shape == (6,)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.all(normals[:, 1] > 0.0)
    assert np.all(weights > 0.0)
    assert float(np.sum(weights)) == pytest.approx(1.0)


def test_partial_traction_quadrature_matches_symmetry_and_loaded_edges() -> None:
    mesh = sample_geometry_cases(master_seed=33, count_per_family=1, nx=6, ny=3)[0]

    points, normals, directions, weights = _partial_traction_boundary_quadrature(mesh)

    assert points.shape == (12, 2)
    assert normals.shape == (12, 2)
    assert directions.shape == (12,)
    assert weights.shape == (12,)
    assert np.count_nonzero(directions == 0) == 6
    assert np.count_nonzero(directions == 1) == 6
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert float(np.sum(weights)) == pytest.approx(1.0)
    bottom = np.isclose(points[:, 1], 0.0)
    left = np.isclose(points[:, 0], 0.0)
    right = np.isclose(points[:, 0], np.max(mesh.points[:, 0]))
    assert np.all(directions[bottom] == 0)
    assert np.all(directions[left | right] == 1)
    np.testing.assert_allclose(
        normals[bottom], np.broadcast_to([0.0, -1.0], normals[bottom].shape)
    )
    np.testing.assert_allclose(
        normals[left], np.broadcast_to([-1.0, 0.0], normals[left].shape)
    )
    np.testing.assert_allclose(
        normals[right], np.broadcast_to([1.0, 0.0], normals[right].shape)
    )


def test_partial_traction_loss_selects_only_the_unconstrained_component() -> None:
    import torch

    stress = torch.zeros((1, 2, 2, 2))
    stress[:, 0, 1, 0] = 7.0
    stress[:, 1, 0, 0] = 5.0
    normals = torch.tensor([[[-1.0, 0.0], [1.0, 0.0]]])
    directions = torch.tensor([[1, 1]])
    weights = torch.tensor([[0.5, 0.5]])

    loss = _partial_traction_loss(
        stress,
        normals,
        directions,
        weights,
        torch.ones(1),
    )

    assert float(loss) == pytest.approx(24.5)


def test_traction_loss_is_zero_for_stress_tangent_to_sloped_boundary() -> None:
    import torch

    normal = torch.tensor([[[-1.0, 1.0]]]) / np.sqrt(2.0)
    tangent = torch.tensor([[[1.0, 1.0]]]) / np.sqrt(2.0)
    row_scale = torch.tensor([[[2.0, -3.0]]])
    stress = torch.einsum("bqi,bqj->bqij", row_scale, tangent)

    loss = _traction_loss(
        stress,
        normal,
        torch.ones((1, 1)),
        torch.ones(1),
    )

    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_traction_loss_penalizes_nonzero_reference_traction() -> None:
    import torch

    stress = torch.zeros((1, 1, 2, 2))
    stress[..., 0, 1] = 2.0

    loss = _traction_loss(
        stress,
        torch.tensor([[[0.0, 1.0]]]),
        torch.ones((1, 1)),
        torch.ones(1),
    )

    assert float(loss) == pytest.approx(4.0)


def test_section_resultant_loss_detects_axial_force_drift_without_fem_labels() -> None:
    import torch

    stress = torch.zeros((1, 3, 2, 2, 2))
    stress[:, 0, :, 0, 0] = 1.0
    stress[:, 1, :, 0, 0] = 2.0
    stress[:, 2, :, 0, 0] = 3.0
    weights = torch.full((1, 3, 2), 0.5)

    loss = _section_resultant_loss(
        stress,
        weights,
        torch.ones(1),
        torch.ones(1),
    )

    assert float(loss) > 0.0
    constant = stress.clone()
    constant[..., 0, 0] = 2.0
    assert float(
        _section_resultant_loss(
            constant,
            weights,
            torch.ones(1),
            torch.ones(1),
        )
    ) == pytest.approx(0.0, abs=1e-12)


def test_section_resultant_quadrature_uses_only_geometry() -> None:
    mesh = sample_geometry_cases(master_seed=32, count_per_family=1, nx=6, ny=3)[0]

    points, weights = _section_resultant_quadrature(
        mesh,
        slice_count=4,
        y_point_count=5,
    )

    assert points.shape == (4, 5, 2)
    assert weights.shape == (4, 5)
    assert np.all(points[..., 0] > 0.0)
    assert np.all(points[..., 0] < np.max(mesh.points[:, 0]))
    assert np.all(points[..., 1] > 0.0)
    assert np.all(weights > 0.0)
    top = np.asarray(
        [
            np.max(mesh.points[np.isclose(mesh.points[:, 0], x), 1])
            for x in np.unique(mesh.points[:, 0])
        ]
    )
    expected_height = np.interp(
        points[:, 0, 0],
        np.unique(mesh.points[:, 0]),
        top,
    )
    np.testing.assert_allclose(np.sum(weights, axis=1), expected_height)


def test_variable_height_force_balanced_stress_has_zero_section_loss() -> None:
    import torch

    heights = torch.tensor([[1.0, 2.0, 4.0]])
    weights = heights[..., None] * torch.tensor([[[0.5, 0.5]]])
    stress = torch.zeros((1, 3, 2, 2, 2))
    stress[..., 0, 0] = (1.0 / heights)[..., None]

    loss = _section_resultant_loss(
        stress,
        weights,
        torch.ones(1),
        torch.tensor([4.0]),
    )

    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_equilibrium_residual_prefix_differentiates_only_interior_outputs() -> None:
    import torch

    coordinates = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        requires_grad=True,
    )
    x = coordinates[..., 0]
    y = coordinates[..., 1]
    stress = torch.stack(
        (
            torch.stack((x.square(), y), dim=-1),
            torch.stack((x * y, 3.0 * y), dim=-1),
        ),
        dim=-2,
    )

    residual = _equilibrium_residual_prefix(stress, coordinates, count=2)

    expected = torch.tensor([[[3.0, 5.0], [7.0, 7.0]]])
    torch.testing.assert_close(residual, expected)


def test_held_out_evaluation_uses_the_upstream_fixed_displacement() -> None:
    import torch

    cases = sample_geometry_cases(master_seed=21, count_per_family=1, nx=6, ny=3)
    pairs = tuple(
        (
            case,
            SimpleNamespace(
                displacement=np.zeros((len(case.points), 2), dtype=np.float64),
                first_piola=np.zeros((len(case.points), 2, 2), dtype=np.float64),
            ),
        )
        for case in cases
    )
    observed_loads: list[float] = []

    class RecordingModel:
        def eval(self) -> RecordingModel:
            return self

        def __call__(
            self,
            points: torch.Tensor,
            _boundary: torch.Tensor,
            u_delta: torch.Tensor,
            _x_max: torch.Tensor,
            _y_max: torch.Tensor,
        ) -> SimpleNamespace:
            observed_loads.extend(float(value) for value in u_delta)
            return SimpleNamespace(
                u=torch.zeros_like(points),
                P_served=torch.zeros((*points.shape[:2], 2, 2)),
            )

    _evaluate_model(RecordingModel(), pairs)

    assert observed_loads
    assert observed_loads == pytest.approx(
        [UPSTREAM_APPLIED_DISPLACEMENT] * len(observed_loads)
    )


def test_new_training_model_loads_upstream_weights_before_fine_tuning() -> None:
    import torch

    from experiments.spike.model import MixedPhysicsOperator, ModelConfig

    config = ModelConfig(
        boundary_points=20,
        latent_tokens=4,
        embed_dim=16,
        num_heads=4,
        encoder_self_layers=1,
        decoder_cross_layers=2,
        positional_encoding_max_degree=2,
        neighbor_count=4,
    )
    upstream = MixedPhysicsOperator(config)
    with torch.no_grad():
        upstream.u_head.bias.fill_(0.314)

    model = _new_pretrained_model(upstream.upstream_state_dict(), config=config)

    torch.testing.assert_close(model.u_head.bias, upstream.u_head.bias)
    torch.testing.assert_close(
        model.p_head.output.weight,
        torch.zeros_like(model.p_head.output.weight),
    )
    torch.testing.assert_close(
        model.p_head.output.bias,
        torch.zeros_like(model.p_head.output.bias),
    )


def test_checkpoint_loader_verifies_hash_and_model_payload(tmp_path: Path) -> None:
    import hashlib

    import torch

    from experiments.spike.model import MixedPhysicsOperator, ModelConfig
    from experiments.spike.run import _load_upstream_checkpoint

    config = ModelConfig(
        boundary_points=20,
        latent_tokens=4,
        embed_dim=16,
        num_heads=4,
        encoder_self_layers=1,
        decoder_cross_layers=2,
        positional_encoding_max_degree=2,
        neighbor_count=4,
    )
    model = MixedPhysicsOperator(config)
    target = tmp_path / "checkpoint.pt"
    torch.save(
        {"epoch": 7, "model_state_dict": model.upstream_state_dict()},
        target,
    )
    expected_hash = hashlib.sha256(target.read_bytes()).hexdigest()

    checkpoint = _load_upstream_checkpoint(target, expected_sha256=expected_hash)

    assert checkpoint.epoch == 7
    assert checkpoint.sha256 == expected_hash
    assert set(checkpoint.state_dict) == set(model.upstream_state_dict())

    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        _load_upstream_checkpoint(target, expected_sha256=expected_hash)


def test_run_spike_builds_a_complete_corrected_report_without_partial_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.spike.fem as fem
    import experiments.spike.geometry as geometry
    import experiments.spike.run as spike_run

    cases = sample_geometry_cases(master_seed=30, count_per_family=30, nx=6, ny=3)
    monkeypatch.setattr(
        geometry,
        "sample_geometry_cases",
        lambda **_kwargs: cases,
    )

    def fake_solve(mesh: object, _cache: Path, _hash: str) -> object:
        point_count = len(mesh.points)
        return SimpleNamespace(
            converged=True,
            failure_reason=None,
            displacement=np.zeros((point_count, 2)),
            first_piola=np.zeros((point_count, 2, 2)),
            solve_seconds=0.1,
            newton_iterations=2,
        )

    family_metrics = {
        family: {
            "l2_quadrature": "P1 lumped reference-area weights",
            "relative_l2_u": 0.02,
            "relative_l2_P": 0.12,
            "worst_relative_peak_stress_error": 0.3,
            "mean_inference_seconds": 0.01,
        }
        for family in ("dogbone", "centre_notched")
    }
    regularity = {
        family: {
            "status": "well_posed",
            "reason": "stubbed smooth-root refinement converged",
        }
        for family in ("dogbone", "centre_notched")
    }
    written: list[tuple[Path, object]] = []

    monkeypatch.setattr(
        spike_run,
        "_resolve_upstream_checkpoint",
        lambda _path=None: spike_run.UpstreamCheckpoint(
            path=Path("best.pt"),
            sha256=UPSTREAM_CHECKPOINT_SHA256,
            epoch=1424,
            state_dict={},
        ),
    )
    monkeypatch.setattr(spike_run, "_fem_content_hash", lambda _root: "e" * 64)
    monkeypatch.setattr(spike_run, "_solve_with_cache", fake_solve)
    monkeypatch.setattr(
        spike_run,
        "_fit_and_select_model",
        lambda _pairs, _state: spike_run.TrainingOutcome(
            model=object(),
            selected_learning_rate=1e-3,
            trials=(),
            final_mean_hybrid_loss=0.01,
            training_seconds=1.0,
        ),
    )
    monkeypatch.setattr(
        spike_run,
        "_evaluate_model",
        lambda _model, _pairs: spike_run.EvaluationOutcome(family_metrics, {}),
    )
    monkeypatch.setattr(
        spike_run,
        "_write_evaluation_artifacts",
        lambda _model, _pairs, _path: {
            "checkpoint": {"sha256": "a" * 64},
            "prediction_npz_count": 20,
            "comparison_plot_svg_count": 20,
        },
    )
    monkeypatch.setattr(
        spike_run,
        "_architecture_iteration_evidence",
        lambda: _valid_results()["architecture_iteration"],
    )
    monkeypatch.setattr(spike_run, "_regularity_check", lambda _meshes: regularity)
    monkeypatch.setattr(
        fem,
        "solve_uniaxial_square",
        lambda **_kwargs: SimpleNamespace(
            converged=True,
            failure_reason=None,
            lateral_stretch=0.98,
            relative_l2_u=1e-12,
            relative_l2_p=1e-12,
            max_error=1e-12,
        ),
    )
    monkeypatch.setattr(spike_run, "_git_head", lambda _root: "abc123")
    monkeypatch.setattr(spike_run, "_worktree_content_hash", lambda _root: "f" * 64)
    monkeypatch.setattr(
        spike_run,
        "write_json_strict",
        lambda path, document: written.append((path, document)),
    )

    document = spike_run.run_spike(Path("best.pt"))

    assert document["protocol"]["converged"] == 60
    assert document["execution_profile"]["applied_displacement_mm"] == 1.0
    assert document["lr_sweep"]["final_training_epochs"] == 1500
    assert document["model"]["upstream_checkpoint"]["sha256"] == (
        UPSTREAM_CHECKPOINT_SHA256
    )
    assert document["model"]["stress_head"] == {
        "variant": "conditioned_mlp",
        "conditioning": ["y_max/x_max"],
        "output_scale": "E*u_delta/x_max",
        "youngs_modulus_MPa": 760.0,
        "output_layer_initialization": "zeros",
    }
    assert all(
        "no explicit aspect-ratio conditioning" not in limitation
        for limitation in document["model"]["known_legacy_limitations"]
    )
    assert document["lr_sweep"]["gradient_clip_norm"] == 1.0
    assert document["lr_sweep"]["supervised_sampling"] == (
        "reference-area-proportional nodal sampling with replacement"
    )
    assert "output layer initialized to zero" in document["lr_sweep"]["initialization"]
    assert document["architecture_iteration"]["selected_variant"] == ("conditioned_mlp")
    assert (
        document["architecture_iteration"]["selection"]["held_out_geometry_accessed"]
        is False
    )
    assert document["architecture_iteration"]["held_out_evaluation_status"] == (
        "confirmatory_not_pristine"
    )
    assert len(document["cases"]) == 60
    assert len(written) == 1


def test_fixed_epoch_trainer_updates_parameters_and_returns_finite_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import experiments.spike.run as spike_run

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    def differentiable_loss(
        candidate: torch.nn.Module,
        _data: object,
        _indices: object,
        _generator: object,
        *,
        peak_supervised_p_weight: float,
        served_stress_physics: ServedStressPhysicsWeights,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert peak_supervised_p_weight == pytest.approx(0.25)
        assert not served_stress_physics.enabled
        loss = candidate.weight.square().mean()
        return loss, {"total": float(loss.detach())}

    monkeypatch.setattr(spike_run, "_hybrid_batch_loss", differentiable_loss)
    clip_calls: list[float] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def record_clip(parameters: object, max_norm: float) -> object:
        clip_calls.append(max_norm)
        return original_clip(parameters, max_norm)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)

    final_loss = spike_run._train_fixed_epochs(
        model,
        {},
        range(4),
        learning_rate=0.1,
        epochs=2,
        random_seed=7,
        label="unit",
        peak_supervised_p_weight=0.25,
    )

    assert np.isfinite(final_loss)
    assert float(model.weight.detach()) < 1.0
    assert clip_calls == [1.0, 1.0]


def test_fixed_epoch_trainer_uses_a_separate_element_stress_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import experiments.spike.run as spike_run

    model = torch.nn.Linear(1, 1, bias=False)
    sampled: list[tuple[int, int]] = []

    def differentiable_loss(
        candidate: torch.nn.Module,
        _data: object,
        _indices: object,
        generator: torch.Generator,
        **kwargs: object,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        stress_generator = kwargs["stress_generator"]
        assert isinstance(stress_generator, torch.Generator)
        sampled.append((generator.initial_seed(), stress_generator.initial_seed()))
        loss = candidate.weight.square().mean()
        return loss, {"total": float(loss.detach())}

    monkeypatch.setattr(spike_run, "_hybrid_batch_loss", differentiable_loss)

    spike_run._train_fixed_epochs(
        model,
        {},
        range(4),
        learning_rate=0.1,
        epochs=1,
        random_seed=7,
        label="element unit",
        stress_supervision="element_quadrature",
        stress_random_seed=17,
    )

    assert sampled == [(7, 17)]


def test_fine_tune_stages_freeze_parameters_and_apply_discriminative_lrs() -> None:
    import torch

    class _StagedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2)
            )
            self.decoder = torch.nn.Linear(2, 2)
            self.p_head = torch.nn.Linear(2, 4)
            self.geom_aux_head = torch.nn.Linear(2, 1)

    model = _StagedModel()
    groups = _staged_optimizer_groups(model)
    optimizer = torch.optim.Adam(groups)
    head_only = FineTuneStage("head_only", 1, None, None, 1e-3)

    trainable = _apply_fine_tune_stage(model, optimizer, head_only)

    assert {group["group_name"]: group["lr"] for group in optimizer.param_groups} == {
        "encoder": 0.0,
        "decoder": 0.0,
        "p_head": 1e-3,
    }
    assert set(trainable) == set(model.p_head.parameters())
    assert model.encoder.training is True
    assert model.decoder.training is True
    assert model.p_head.training is True
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.decoder.parameters())
    assert all(parameter.requires_grad for parameter in model.p_head.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.geom_aux_head.parameters()
    )

    full = FineTuneStage("full", 1, 3e-5, 1e-4, 3e-4)
    trainable = _apply_fine_tune_stage(model, optimizer, full)

    assert {group["group_name"]: group["lr"] for group in optimizer.param_groups} == {
        "encoder": 3e-5,
        "decoder": 1e-4,
        "p_head": 3e-4,
    }
    assert set(trainable) == set(model.encoder.parameters()) | set(
        model.decoder.parameters()
    ) | set(model.p_head.parameters())
    assert model.encoder.training is True
    assert model.decoder.training is True
    assert model.p_head.training is True


def test_staged_trainer_preserves_one_optimizer_and_declared_epoch_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    import experiments.spike.run as spike_run

    class _StagedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1, bias=False)
            self.decoder = torch.nn.Linear(1, 1, bias=False)
            self.p_head = torch.nn.Linear(1, 1, bias=False)

    model = _StagedModel()
    calls: list[tuple[bool, bool, bool]] = []

    def differentiable_loss(
        candidate: _StagedModel,
        _data: object,
        _indices: object,
        _generator: object,
        *,
        peak_supervised_p_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert peak_supervised_p_weight == 0.0
        calls.append(
            (
                candidate.encoder.weight.requires_grad,
                candidate.decoder.weight.requires_grad,
                candidate.p_head.weight.requires_grad,
            )
        )
        loss = sum(parameter.square().mean() for parameter in candidate.parameters())
        return loss, {"total": float(loss.detach())}

    monkeypatch.setattr(spike_run, "_hybrid_batch_loss", differentiable_loss)
    stages = (
        FineTuneStage("head", 1, None, None, 1e-3),
        FineTuneStage("decoder", 1, None, 1e-4, 1e-3),
        FineTuneStage("full", 1, 3e-5, 1e-4, 3e-4),
    )

    outcome = _train_staged_epochs(
        model,
        {},
        range(4),
        stages=stages,
        random_seed=7,
        label="unit staged",
    )

    assert calls == [
        (False, False, True),
        (False, True, True),
        (True, True, True),
    ]
    assert outcome["total_epochs"] == 3
    assert [phase["name"] for phase in outcome["phases"]] == [
        "head",
        "decoder",
        "full",
    ]
    assert np.isfinite(outcome["final_hybrid_loss"])


def test_screen_summary_and_guardrails_are_family_aware() -> None:
    def fold(dog_p: float, centre_p: float, dog_u: float, centre_u: float) -> object:
        return {
            "family_metrics": {
                "dogbone": {
                    "relative_l2_P": dog_p,
                    "relative_l2_u": dog_u,
                    "worst_relative_peak_stress_error": 0.30,
                    "mean_inference_seconds": 0.010,
                },
                "centre_notched": {
                    "relative_l2_P": centre_p,
                    "relative_l2_u": centre_u,
                    "worst_relative_peak_stress_error": 0.35,
                    "mean_inference_seconds": 0.011,
                },
            },
            "per_case_metrics": {
                "dog": {
                    "relative_l2_u": dog_u,
                    "relative_peak_stress_error": 0.30,
                },
                "centre": {
                    "relative_l2_u": centre_u,
                    "relative_peak_stress_error": 0.35,
                },
            },
        }

    baseline = _summarize_screen_folds(
        (fold(0.12, 0.11, 0.04, 0.05), fold(0.10, 0.13, 0.03, 0.04))
    )
    candidate = _summarize_screen_folds(
        (fold(0.09, 0.09, 0.04, 0.05), fold(0.08, 0.10, 0.03, 0.04))
    )
    checks = _screen_guardrail_checks(
        baseline,
        candidate,
        stress_head_parameter_count=4_700,
        total_parameter_growth=0.005,
    )

    assert baseline["pooled_worst_family_relative_l2_P"] == pytest.approx(0.12)
    assert candidate["pooled_worst_family_relative_l2_P"] == pytest.approx(0.095)
    assert checks["primary_score_improved_by_at_least_10_percent"] is True
    assert checks["all_passed"] is True

    regressed = {
        **candidate,
        "mean_relative_l2_u": {"dogbone": 0.06, "centre_notched": 0.04},
    }
    failed = _screen_guardrail_checks(
        baseline,
        regressed,
        stress_head_parameter_count=4_700,
        total_parameter_growth=0.005,
    )
    assert failed["family_u_within_absolute_0_005"] is False
    assert failed["all_passed"] is False


def test_iteration3_screen_uses_only_the_training_split_and_writes_legacy_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    import experiments.spike.geometry as geometry
    import experiments.spike.run as spike_run

    cases = sample_geometry_cases(master_seed=20260809, count_per_family=30, nx=6, ny=3)
    split = split_by_geometry(cases, master_seed=991)
    train_ids = {mesh.geometry_id for mesh in split.train}
    evaluate_ids = {mesh.geometry_id for mesh in split.evaluate}
    solved_ids: list[str] = []
    train_calls: list[tuple[str, float, int]] = []

    monkeypatch.setattr(geometry, "sample_geometry_cases", lambda **_kwargs: cases)
    monkeypatch.setattr(
        spike_run,
        "_load_upstream_checkpoint",
        lambda _path: spike_run.UpstreamCheckpoint(
            path=Path("best.pt"),
            sha256=UPSTREAM_CHECKPOINT_SHA256,
            epoch=1424,
            state_dict={},
        ),
    )
    monkeypatch.setattr(spike_run, "_fem_content_hash", lambda _root: "f" * 64)

    def fake_solve(mesh: object, _cache: Path, _hash: str) -> object:
        solved_ids.append(mesh.geometry_id)
        return SimpleNamespace(
            converged=True,
            displacement=np.zeros((len(mesh.points), 2)),
            first_piola=np.ones((len(mesh.points), 2, 2)),
        )

    monkeypatch.setattr(spike_run, "_solve_with_cache", fake_solve)

    class _Candidate(torch.nn.Module):
        def __init__(self, variant: str) -> None:
            super().__init__()
            self.variant = variant
            self.backbone = torch.nn.Parameter(torch.zeros(100_000))
            self.p_head = torch.nn.Linear(4, 4)

    monkeypatch.setattr(
        spike_run,
        "_new_pretrained_model",
        lambda _state, *, config: _Candidate(config.stress_head_variant),
    )

    def fake_train(
        model: _Candidate,
        _data: object,
        _indices: object,
        *,
        learning_rate: float,
        epochs: int,
        random_seed: int,
        label: str,
        peak_supervised_p_weight: float,
    ) -> float:
        del learning_rate, random_seed
        train_calls.append((model.variant, peak_supervised_p_weight, epochs))
        assert label
        return 0.1

    monkeypatch.setattr(spike_run, "_train_fixed_epochs", fake_train)

    def fake_evaluate(model: _Candidate, pairs: object) -> object:
        p_error = 0.20 if model.variant == "conditioned_mlp" else 0.15
        family_metrics = {
            family: {
                "relative_l2_P": p_error,
                "relative_l2_u": 0.04,
                "worst_relative_peak_stress_error": 0.30,
                "mean_inference_seconds": 0.01,
            }
            for family in ("dogbone", "centre_notched")
        }
        per_case = {
            mesh.geometry_id: {
                "relative_l2_u": 0.04,
                "relative_peak_stress_error": 0.30,
            }
            for mesh, _result in pairs
        }
        return spike_run.EvaluationOutcome(family_metrics, per_case)

    monkeypatch.setattr(spike_run, "_evaluate_model", fake_evaluate)
    output_path = tmp_path / "iteration3_screen.json"

    screen = run_iteration3_screen(
        Path("best.pt"),
        output_path,
        epochs=1,
        peak_supervised_p_weight=0.0,
    )

    assert set(solved_ids) == train_ids
    assert set(solved_ids).isdisjoint(evaluate_ids)
    assert len(train_calls) == 4
    assert {call[:2] for call in train_calls} == {
        ("conditioned_mlp", 0.0),
        ("fully_conditioned_mlp", 0.0),
    }
    assert screen["role"] == "LEGACY"
    assert screen["architecture_iteration"]["selected_variant"] == (
        "fully_conditioned_mlp"
    )
    assert (
        screen["architecture_iteration"]["selection"]["held_out_geometry_accessed"]
        is False
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == screen


def test_iteration4_screen_pairs_joint_and_staged_training_without_holdout_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    import experiments.spike.geometry as geometry
    import experiments.spike.run as spike_run

    cases = sample_geometry_cases(master_seed=20260809, count_per_family=30, nx=6, ny=3)
    split = split_by_geometry(cases, master_seed=991)
    train_ids = {mesh.geometry_id for mesh in split.train}
    evaluate_ids = {mesh.geometry_id for mesh in split.evaluate}
    solved_ids: list[str] = []
    train_calls: list[tuple[str, int]] = []

    monkeypatch.setattr(geometry, "sample_geometry_cases", lambda **_kwargs: cases)
    monkeypatch.setattr(
        spike_run,
        "_load_upstream_checkpoint",
        lambda _path: spike_run.UpstreamCheckpoint(
            path=Path("best.pt"),
            sha256=UPSTREAM_CHECKPOINT_SHA256,
            epoch=1424,
            state_dict={},
        ),
    )
    monkeypatch.setattr(spike_run, "_fem_content_hash", lambda _root: "f" * 64)

    def fake_solve(mesh: object, _cache: Path, _hash: str) -> object:
        solved_ids.append(mesh.geometry_id)
        return SimpleNamespace(
            converged=True,
            displacement=np.zeros((len(mesh.points), 2)),
            first_piola=np.ones((len(mesh.points), 2, 2)),
        )

    monkeypatch.setattr(spike_run, "_solve_with_cache", fake_solve)

    class _Candidate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)
            self.p_head = torch.nn.Linear(2, 4)
            self.protocol = "untrained"

    monkeypatch.setattr(
        spike_run,
        "_new_pretrained_model",
        lambda _state, *, config: _Candidate(),
    )

    expected_development_ids = [
        [
            mesh.geometry_id
            for family in ("dogbone", "centre_notched")
            for mesh in tuple(item for item in split.train if item.family == family)[
                development_slice
            ]
        ]
        for development_slice in (slice(16, 20), slice(0, 4))
    ]
    sealed_baseline = {
        "folds": [
            {
                "fold": fold,
                "development_geometry_ids": expected_development_ids[fold_index],
                "family_metrics": {
                    family: {
                        "relative_l2_P": 0.20,
                        "relative_l2_u": 0.04,
                        "worst_relative_peak_stress_error": 0.30,
                        "mean_inference_seconds": 0.01,
                    }
                    for family in ("dogbone", "centre_notched")
                },
                "per_case_metrics": {
                    f"{fold}-{index}": {
                        "relative_l2_u": 0.04,
                        "relative_peak_stress_error": 0.30,
                    }
                    for index in range(8)
                },
            }
            for fold_index, fold in enumerate(("tail", "head"))
        ],
        "mean_relative_l2_P": {"dogbone": 0.20, "centre_notched": 0.20},
        "mean_relative_l2_u": {"dogbone": 0.04, "centre_notched": 0.04},
        "pooled_worst_family_relative_l2_P": 0.20,
        "median_peak_stress_error": 0.30,
        "overall_worst_peak_stress_error": 0.30,
        "worst_single_case_relative_l2_u": 0.04,
        "mean_inference_seconds": 0.01,
        "total_parameter_count": 18,
        "stress_head_parameter_count": 12,
        "peak_supervised_P_weight": 0.0,
        "guardrails_passed": None,
    }
    monkeypatch.setattr(
        spike_run,
        "_architecture_iteration_evidence",
        lambda: {
            "selection": {
                "variants": {"conditioned_mlp": sealed_baseline},
                "held_out_geometry_accessed": False,
            }
        },
    )

    def reject_joint_retraining(*_args: object, **_kwargs: object) -> float:
        raise AssertionError("sealed iteration-3 baseline must not be retrained")

    def fake_staged(
        model: _Candidate,
        _data: object,
        _indices: object,
        *,
        stages: object,
        random_seed: int,
        label: str,
        peak_supervised_p_weight: float,
    ) -> object:
        del random_seed
        assert peak_supervised_p_weight == 0.0
        assert label
        model.protocol = "staged_adam"
        total_epochs = sum(stage.epochs for stage in stages)
        train_calls.append((model.protocol, total_epochs))
        return {
            "final_hybrid_loss": 0.08,
            "total_epochs": total_epochs,
            "phases": [],
        }

    monkeypatch.setattr(spike_run, "_train_fixed_epochs", reject_joint_retraining)
    monkeypatch.setattr(spike_run, "_train_staged_epochs", fake_staged)

    def fake_evaluate(model: _Candidate, pairs: object) -> object:
        p_error = 0.20 if model.protocol == "joint_adam" else 0.15
        family_metrics = {
            family: {
                "relative_l2_P": p_error,
                "relative_l2_u": 0.04,
                "worst_relative_peak_stress_error": 0.30,
                "mean_inference_seconds": 0.01,
            }
            for family in ("dogbone", "centre_notched")
        }
        per_case = {
            mesh.geometry_id: {
                "relative_l2_u": 0.04,
                "relative_peak_stress_error": 0.30,
            }
            for mesh, _result in pairs
        }
        return spike_run.EvaluationOutcome(family_metrics, per_case)

    monkeypatch.setattr(spike_run, "_evaluate_model", fake_evaluate)
    output_path = tmp_path / "iteration4_screen.json"
    stages = (
        FineTuneStage("head_only", 1, None, None, 1e-3),
        FineTuneStage("decoder_and_head", 1, None, 1e-4, 1e-3),
        FineTuneStage("full_low_lr", 1, 3e-5, 1e-4, 3e-4),
    )

    screen = run_iteration4_screen(
        Path("best.pt"),
        output_path,
        stages=stages,
    )

    assert set(solved_ids) == train_ids
    assert set(solved_ids).isdisjoint(evaluate_ids)
    assert sorted(train_calls) == [
        ("staged_adam", 3),
        ("staged_adam", 3),
    ]
    iteration = screen["training_protocol_iteration"]
    assert iteration["selected_protocol"] == "staged_adam"
    assert iteration["selection"]["held_out_geometry_accessed"] is False
    assert iteration["selection"]["same_epoch_budget"] is False
    assert json.loads(output_path.read_text(encoding="utf-8")) == screen


def test_iteration4_official_screen_rejects_a_non_preregistered_schedule() -> None:
    with pytest.raises(ValueError, match="preregistered stages"):
        run_iteration4_screen(
            Path("best.pt"),
            stages=(FineTuneStage("short", 1, None, None, 1e-3),),
        )


def test_iteration5_screen_changes_only_training_physics_and_quarantines_holdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    import experiments.spike.geometry as geometry
    import experiments.spike.run as spike_run

    cases = sample_geometry_cases(master_seed=20260809, count_per_family=30, nx=6, ny=3)
    split = split_by_geometry(cases, master_seed=991)
    train_ids = {mesh.geometry_id for mesh in split.train}
    evaluate_ids = {mesh.geometry_id for mesh in split.evaluate}
    solved_ids: list[str] = []
    train_calls: list[ServedStressPhysicsWeights] = []

    monkeypatch.setattr(geometry, "sample_geometry_cases", lambda **_kwargs: cases)
    monkeypatch.setattr(
        spike_run,
        "_load_upstream_checkpoint",
        lambda _path: spike_run.UpstreamCheckpoint(
            path=Path("best.pt"),
            sha256=UPSTREAM_CHECKPOINT_SHA256,
            epoch=1424,
            state_dict={},
        ),
    )
    monkeypatch.setattr(spike_run, "_fem_content_hash", lambda _root: "f" * 64)

    def fake_solve(mesh: object, _cache: Path, _hash: str) -> object:
        solved_ids.append(mesh.geometry_id)
        return SimpleNamespace(
            converged=True,
            displacement=np.zeros((len(mesh.points), 2)),
            first_piola=np.ones((len(mesh.points), 2, 2)),
        )

    monkeypatch.setattr(spike_run, "_solve_with_cache", fake_solve)

    class _Candidate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)
            self.p_head = torch.nn.Linear(2, 4)

    monkeypatch.setattr(
        spike_run,
        "_new_pretrained_model",
        lambda _state, *, config: _Candidate(),
    )
    family_indices = {
        family: tuple(
            index for index, mesh in enumerate(split.train) if mesh.family == family
        )
        for family in ("dogbone", "centre_notched")
    }
    fold_slices = (("tail", slice(16, 20)), ("head", slice(0, 4)))
    baseline_folds = []
    for fold_name, development_slice in fold_slices:
        development = tuple(
            index
            for family in ("dogbone", "centre_notched")
            for index in family_indices[family][development_slice]
        )
        development_set = set(development)
        fit = tuple(
            index for index in range(len(split.train)) if index not in development_set
        )
        baseline_folds.append(
            {
                "fold": fold_name,
                "fit_geometry_ids": [split.train[index].geometry_id for index in fit],
                "development_geometry_ids": [
                    split.train[index].geometry_id for index in development
                ],
                "family_metrics": {
                    family: {
                        "relative_l2_P": 0.20,
                        "relative_l2_u": 0.04,
                        "worst_relative_peak_stress_error": 0.30,
                        "mean_inference_seconds": 0.01,
                    }
                    for family in ("dogbone", "centre_notched")
                },
                "per_case_metrics": {
                    split.train[index].geometry_id: {
                        "relative_l2_u": 0.04,
                        "relative_peak_stress_error": 0.30,
                    }
                    for index in development
                },
            }
        )
    baseline = {
        **spike_run._summarize_screen_folds(baseline_folds),
        "folds": baseline_folds,
        "total_parameter_count": 24,
        "stress_head_parameter_count": 12,
        "peak_supervised_P_weight": 0.0,
        "guardrails_passed": None,
    }
    monkeypatch.setattr(
        spike_run,
        "_architecture_iteration_evidence",
        lambda: {
            "selection": {
                "variants": {"conditioned_mlp": baseline},
                "held_out_geometry_accessed": False,
            }
        },
    )

    def fake_train(
        _model: object,
        _data: object,
        _indices: object,
        **kwargs: object,
    ) -> float:
        weights = kwargs["served_stress_physics"]
        assert isinstance(weights, ServedStressPhysicsWeights)
        train_calls.append(weights)
        return 0.01

    monkeypatch.setattr(spike_run, "_train_fixed_epochs", fake_train)

    def fake_evaluate(_model: object, pairs: object) -> object:
        family_metrics = {
            family: {
                "relative_l2_P": 0.15,
                "relative_l2_u": 0.04,
                "worst_relative_peak_stress_error": 0.30,
                "mean_inference_seconds": 0.01,
            }
            for family in ("dogbone", "centre_notched")
        }
        per_case = {
            mesh.geometry_id: {
                "relative_l2_u": 0.04,
                "relative_peak_stress_error": 0.30,
            }
            for mesh, _result in pairs
        }
        return spike_run.EvaluationOutcome(family_metrics, per_case)

    monkeypatch.setattr(spike_run, "_evaluate_model", fake_evaluate)
    monkeypatch.setattr(
        spike_run,
        "_served_stress_physics_diagnostics",
        lambda *_args, **_kwargs: {
            family: {
                "traction_free_normalized_rms": 0.05,
                "partial_traction_normalized_rms": 0.01,
                "section_resultant_normalized_rms": 0.01,
                "section_resultant_coefficient_of_variation": 0.01,
            }
            for family in ("dogbone", "centre_notched")
        },
    )
    output_path = tmp_path / "iteration5_screen.json"

    screen = run_iteration5_screen(Path("best.pt"), output_path, epochs=1)

    assert set(solved_ids) == train_ids
    assert set(solved_ids).isdisjoint(evaluate_ids)
    assert train_calls == [ITERATION5_PHYSICS_WEIGHTS, ITERATION5_PHYSICS_WEIGHTS]
    iteration = screen["physics_objective_iteration"]
    assert iteration["selected_objective"] == "natural_bc_and_section_balance"
    assert iteration["selection"]["held_out_geometry_accessed"] is False
    assert json.loads(output_path.read_text(encoding="utf-8")) == screen

    solved_ids.clear()
    train_calls.clear()
    iteration6_path = tmp_path / "iteration6_screen.json"
    iteration6_screen = run_iteration6_screen(
        Path("best.pt"), iteration6_path, epochs=1
    )

    assert set(solved_ids) == train_ids
    assert set(solved_ids).isdisjoint(evaluate_ids)
    assert train_calls == [ITERATION6_PHYSICS_WEIGHTS, ITERATION6_PHYSICS_WEIGHTS]
    iteration6 = iteration6_screen["physics_objective_iteration"]
    assert iteration6["iteration"] == 6
    assert iteration6["selected_objective"] == "weak_upstream_physics_0p1x"
    assert iteration6["selection"]["held_out_geometry_accessed"] is False
    assert iteration6["selection"]["prior_iteration_evidence_sealed"] is False
    assert "sealed_rejected_iteration5_sha256" not in iteration6["selection"]
    assert json.loads(iteration6_path.read_text(encoding="utf-8")) == iteration6_screen


def test_iteration5_official_screen_rejects_a_non_preregistered_epoch_budget() -> None:
    with pytest.raises(ValueError, match="preregistered 300 epochs"):
        run_iteration5_screen(Path("best.pt"), epochs=1)


def test_iteration6_weakens_only_the_upstream_physics_weights() -> None:
    assert ITERATION6_PHYSICS_WEIGHTS == ServedStressPhysicsWeights(
        traction_free=0.2,
        partial_traction=0.2,
        section_resultant=1.0,
    )
    assert ITERATION6_PHYSICS_WEIGHTS.traction_free == pytest.approx(
        0.1 * ITERATION5_PHYSICS_WEIGHTS.traction_free
    )
    assert ITERATION6_PHYSICS_WEIGHTS.partial_traction == pytest.approx(
        0.1 * ITERATION5_PHYSICS_WEIGHTS.partial_traction
    )
    assert ITERATION6_PHYSICS_WEIGHTS.section_resultant == pytest.approx(
        0.1 * ITERATION5_PHYSICS_WEIGHTS.section_resultant
    )


def test_iteration6_official_screen_rejects_a_non_preregistered_epoch_budget() -> None:
    with pytest.raises(ValueError, match="preregistered 300 epochs"):
        run_iteration6_screen(Path("best.pt"), epochs=1)


def test_iteration6_evidence_grade_screen_validates_the_sealed_prior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import experiments.spike.run as spike_run

    def reject_prior() -> object:
        raise ValueError("sealed prior rejected")

    monkeypatch.setattr(spike_run, "_iteration5_rejection_evidence", reject_prior)

    with pytest.raises(ValueError, match="sealed prior rejected"):
        run_iteration6_screen(
            Path("best.pt"),
            tmp_path / "iteration6_screen.json",
            epochs=300,
        )


def test_iteration7_screen_pairs_nodal_control_with_element_supervision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    import experiments.spike.geometry as geometry
    import experiments.spike.run as spike_run

    cases = sample_geometry_cases(master_seed=20260809, count_per_family=30, nx=6, ny=3)
    split = split_by_geometry(cases, master_seed=991)
    train_ids = {mesh.geometry_id for mesh in split.train}
    evaluate_ids = {mesh.geometry_id for mesh in split.evaluate}
    solved_ids: list[str] = []
    training_modes: list[str] = []

    monkeypatch.setattr(geometry, "sample_geometry_cases", lambda **_kwargs: cases)
    monkeypatch.setattr(
        spike_run,
        "_load_upstream_checkpoint",
        lambda _path: spike_run.UpstreamCheckpoint(
            path=Path("best.pt"),
            sha256=UPSTREAM_CHECKPOINT_SHA256,
            epoch=1424,
            state_dict={},
        ),
    )
    monkeypatch.setattr(spike_run, "_fem_content_hash", lambda _root: "f" * 64)

    gradient = np.asarray([[0.05, 0.01], [-0.01, 0.02]])

    def fake_solve(mesh: object, _cache: Path, _hash: str) -> object:
        from experiments.spike.fem import first_piola_plane_stress

        solved_ids.append(mesh.geometry_id)
        stress = first_piola_plane_stress(np.eye(2) + gradient)
        return SimpleNamespace(
            converged=True,
            displacement=mesh.points @ gradient.T,
            first_piola=np.broadcast_to(stress, (len(mesh.points), 2, 2)).copy(),
        )

    monkeypatch.setattr(spike_run, "_solve_with_cache", fake_solve)

    class _Candidate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)
            self.p_head = torch.nn.Linear(2, 4)
            self.training_mode = "nodal_patch"

    monkeypatch.setattr(
        spike_run,
        "_new_pretrained_model",
        lambda _state, *, config: _Candidate(),
    )

    family_indices = {
        family: tuple(
            index for index, mesh in enumerate(split.train) if mesh.family == family
        )
        for family in ("dogbone", "centre_notched")
    }
    baseline_folds = []
    for fold_name, development_slice in (
        ("tail", slice(16, 20)),
        ("head", slice(0, 4)),
    ):
        development = tuple(
            index
            for family in ("dogbone", "centre_notched")
            for index in family_indices[family][development_slice]
        )
        development_set = set(development)
        fit = tuple(
            index for index in range(len(split.train)) if index not in development_set
        )
        baseline_folds.append(
            {
                "fold": fold_name,
                "fit_geometry_ids": [split.train[index].geometry_id for index in fit],
                "development_geometry_ids": [
                    split.train[index].geometry_id for index in development
                ],
                "family_metrics": {
                    family: {
                        "relative_l2_P": 0.10,
                        "relative_l2_u": 0.02,
                        "worst_relative_peak_stress_error": 0.25,
                        "mean_inference_seconds": 0.01,
                    }
                    for family in ("dogbone", "centre_notched")
                },
            }
        )
    baseline = {
        "folds": baseline_folds,
        "total_parameter_count": 24,
        "stress_head_parameter_count": 12,
        "pooled_worst_family_relative_l2_P": 0.20,
        "mean_relative_l2_P": {"dogbone": 0.20, "centre_notched": 0.20},
        "mean_relative_l2_u": {"dogbone": 0.04, "centre_notched": 0.04},
        "worst_single_case_relative_l2_u": 0.01,
        "overall_worst_peak_stress_error": 0.10,
        "median_peak_stress_error": 0.10,
        "mean_inference_seconds": 0.01,
    }
    monkeypatch.setattr(
        spike_run,
        "_architecture_iteration_evidence",
        lambda: {
            "selection": {
                "variants": {"conditioned_mlp": baseline},
                "held_out_geometry_accessed": False,
            }
        },
    )
    monkeypatch.setattr(spike_run, "_iteration6_rejection_evidence", lambda: {})

    def fake_train(
        model: _Candidate, _data: object, _indices: object, **kwargs: object
    ) -> float:
        model.training_mode = str(kwargs.get("stress_supervision", "nodal_patch"))
        training_modes.append(model.training_mode)
        return 0.01

    monkeypatch.setattr(spike_run, "_train_fixed_epochs", fake_train)

    def fake_evaluate(model: _Candidate, pairs: object) -> object:
        p_error = 0.15 if model.training_mode == "element_quadrature" else 0.20
        legacy_p = 0.195 if model.training_mode == "element_quadrature" else 0.20
        family_metrics = {
            family: {
                "relative_l2_P": p_error,
                "legacy_nodal_relative_l2_P": legacy_p,
                "relative_l2_u": 0.04,
                "worst_relative_peak_stress_error": 0.30,
                "mean_inference_seconds": 0.01,
            }
            for family in ("dogbone", "centre_notched")
        }
        per_case = {
            mesh.geometry_id: {
                "relative_l2_P": p_error,
                "legacy_nodal_relative_l2_P": legacy_p,
                "relative_l2_u": 0.04,
                "relative_peak_stress_error": 0.30,
                "inference_seconds": 0.01,
            }
            for mesh, _result in pairs
        }
        return spike_run.EvaluationOutcome(family_metrics, per_case)

    monkeypatch.setattr(spike_run, "_evaluate_model_element_quadrature", fake_evaluate)
    monkeypatch.setattr(
        spike_run,
        "_served_stress_physics_diagnostics",
        lambda *_args, **_kwargs: {},
    )

    output = tmp_path / "iteration7_screen.json"
    screen = run_iteration7_screen(Path("best.pt"), output, epochs=1)

    assert set(solved_ids) == train_ids
    assert set(solved_ids).isdisjoint(evaluate_ids)
    assert training_modes == [
        "nodal_patch",
        "element_quadrature",
        "nodal_patch",
        "element_quadrature",
    ]
    iteration = screen["stress_representation_iteration"]
    assert iteration["selected_objective"] == "element_quadrature_P_supervision"
    assert iteration["selection"]["held_out_geometry_accessed"] is False
    assert (
        iteration["selection"]["historical_control_metric_drift_diagnostic_only"] > 0.05
    )
    assert iteration["selection"]["guardrails"]["paired_current_code_control_completed"]
    assert json.loads(output.read_text(encoding="utf-8")) == screen


def test_iteration7_official_screen_rejects_a_non_preregistered_epoch_budget() -> None:
    with pytest.raises(ValueError, match="preregistered 300 epochs"):
        run_iteration7_screen(Path("best.pt"), epochs=1)


def test_iteration7_evidence_grade_screen_validates_the_control_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import experiments.spike.run as spike_run

    monkeypatch.setattr(spike_run, "_iteration6_rejection_evidence", lambda: {})
    monkeypatch.setattr(spike_run, "_file_sha256", lambda _path: "0" * 64)

    with pytest.raises(ValueError, match="control audit SHA-256"):
        run_iteration7_screen(
            Path("best.pt"),
            tmp_path / "iteration7_screen.json",
            epochs=300,
        )


def test_real_hybrid_loss_supports_end_to_end_backward() -> None:
    import torch

    from experiments.spike.fem import solve_case
    from experiments.spike.model import MixedPhysicsOperator, ModelConfig

    cases = sample_geometry_cases(master_seed=5, count_per_family=1, nx=6, ny=3)
    pairs = tuple((case, solve_case(case)) for case in cases)
    data = _tensor_dataset(pairs)
    data = {key: value for key, value in data.items() if key != "qoi_weights"}
    model = MixedPhysicsOperator(
        ModelConfig(
            boundary_points=320,
            latent_tokens=4,
            embed_dim=16,
            num_heads=4,
            encoder_self_layers=1,
            decoder_cross_layers=2,
            positional_encoding_max_degree=2,
            neighbor_count=4,
        )
    )
    loss, components = _hybrid_batch_loss(
        model,
        data,
        torch.arange(2),
        torch.Generator().manual_seed(8),
        peak_supervised_p_weight=0.0,
        served_stress_physics=ServedStressPhysicsWeights(
            traction_free=2.0,
            partial_traction=2.0,
            section_resultant=10.0,
        ),
    )

    loss.backward()

    assert torch.isfinite(loss)


def test_hybrid_loss_can_supervise_p_on_independent_element_quadrature() -> None:
    import torch

    from experiments.spike.fem import solve_case
    from experiments.spike.model import MixedPhysicsOperator, ModelConfig

    cases = sample_geometry_cases(master_seed=7, count_per_family=1, nx=6, ny=3)
    pairs = tuple((case, solve_case(case)) for case in cases)
    data = _element_tensor_dataset(pairs)
    base_model = MixedPhysicsOperator(
        ModelConfig(
            boundary_points=320,
            latent_tokens=4,
            embed_dim=16,
            num_heads=4,
            encoder_self_layers=1,
            decoder_cross_layers=2,
            positional_encoding_max_degree=2,
            neighbor_count=4,
        )
    )

    class _RecordingModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = base_model
            self.query_counts: list[int] = []

        def forward(self, *args: torch.Tensor) -> object:
            self.query_counts.append(int(args[0].shape[1]))
            return self.model(*args)

    model = _RecordingModel()
    model.eval()
    data = dict(data)
    data["stress_quadrature_targets"] = data["stress_quadrature_targets"] * 100.0
    loss, components = _hybrid_batch_loss(
        model,
        data,
        torch.arange(2),
        torch.Generator().manual_seed(8),
        stress_supervision="element_quadrature",
        stress_generator=torch.Generator().manual_seed(18),
    )
    _, nodal_components = _hybrid_batch_loss(
        model,
        data,
        torch.arange(2),
        torch.Generator().manual_seed(8),
    )

    loss.backward()

    assert model.query_counts[0] == 2 * 96
    assert components["equilibrium"] == pytest.approx(nodal_components["equilibrium"])
    assert components["constitutive_in_plane"] == pytest.approx(
        nodal_components["constitutive_in_plane"]
    )
    assert torch.isfinite(loss)
    assert {
        "traction_free_on_P_served",
        "partial_traction_on_P_served",
        "section_resultant_on_P_served",
    }.issubset(components)
    assert all(np.isfinite(value) for value in components.values())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_evaluation_artifacts_are_legacy_tagged_and_use_a_common_stress_scale(
    tmp_path: Path,
) -> None:
    import re

    import torch

    from experiments.spike.fem import solve_case

    mesh = sample_geometry_cases(master_seed=44, count_per_family=1, nx=6, ny=3)[0]
    result = solve_case(mesh)
    assert result.converged

    class _ZeroModel(torch.nn.Module):
        def forward(
            self,
            points: torch.Tensor,
            _boundary: torch.Tensor,
            _u_delta: torch.Tensor,
            _x_max: torch.Tensor,
            _y_max: torch.Tensor,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                u=torch.zeros_like(points),
                P_served=torch.zeros((*points.shape[:2], 2, 2)),
            )

    summary = _write_evaluation_artifacts(
        _ZeroModel(),
        ((mesh, result),),
        tmp_path / "artifacts",
    )

    checkpoint = torch.load(
        summary["checkpoint"]["path"], map_location="cpu", weights_only=True
    )
    assert checkpoint["role"] == "LEGACY"
    prediction_path = next((tmp_path / "artifacts" / "predictions").glob("*.npz"))
    with np.load(prediction_path) as prediction:
        assert prediction["role"].item() == "LEGACY"

    plot_path = next((tmp_path / "artifacts" / "plots").glob("*.svg"))
    svg = plot_path.read_text(encoding="utf-8")
    fem_range = re.search(r"FEM \|P\|: ([^<]+)", svg)
    operator_range = re.search(r"PI-GINOT \|P\|: ([^<]+)", svg)
    assert fem_range is not None
    assert operator_range is not None
    assert fem_range.group(1) == operator_range.group(1)
