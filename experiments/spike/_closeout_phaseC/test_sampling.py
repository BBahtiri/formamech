"""RED tests for the Phase C supervised-sampling fix."""

from __future__ import annotations

import torch

from experiments.spike.geometry import generate_geometry
from experiments.spike.run import (
    QOI_SUPERVISED_FRACTION,
    _lumped_nodal_areas,
    _sample_supervised_indices,
    supervised_sampling_weights,
)

_PARAMS = {"width": 16.57, "notch_radius": 6.63, "notch_depth": 4.31}


def _mesh_tensors():
    mesh = generate_geometry("centre_notched", _PARAMS, 1, nx=32, ny=10)
    areas = torch.from_numpy(_lumped_nodal_areas(mesh.points, mesh.cells))[None]
    qoi = torch.zeros_like(areas)
    qoi[0, mesh.qoi_node_ids] = 1.0
    return mesh, areas, qoi


def test_area_strategy_starves_the_qoi_band() -> None:
    """Regression: documents the behaviour the incumbent trained under."""
    mesh, areas, qoi = _mesh_tensors()
    weights = supervised_sampling_weights(areas, qoi, strategy="area")
    band_share = float(weights[0][qoi[0] > 0].sum())
    uniform_share = len(mesh.qoi_node_ids) / len(mesh.points)
    assert band_share < uniform_share / 2.0, (band_share, uniform_share)


def test_stratified_qoi_gives_the_band_its_declared_share() -> None:
    _mesh, areas, qoi = _mesh_tensors()
    weights = supervised_sampling_weights(areas, qoi, strategy="stratified_qoi")
    band_share = float(weights[0][qoi[0] > 0].sum())
    assert abs(band_share - QOI_SUPERVISED_FRACTION) < 1e-6


def test_stratified_qoi_never_samples_below_the_uniform_share() -> None:
    mesh, areas, qoi = _mesh_tensors()
    weights = supervised_sampling_weights(areas, qoi, strategy="stratified_qoi")
    band_share = float(weights[0][qoi[0] > 0].sum())
    assert band_share > len(mesh.qoi_node_ids) / len(mesh.points)


def test_weights_are_a_normalised_distribution() -> None:
    _mesh, areas, qoi = _mesh_tensors()
    for strategy in ("area", "stratified_qoi"):
        weights = supervised_sampling_weights(areas, qoi, strategy=strategy)
        assert torch.allclose(weights.sum(dim=1), torch.ones(1, dtype=weights.dtype))
        assert bool((weights >= 0).all())


def test_draws_land_in_the_band_at_the_declared_rate() -> None:
    _mesh, areas, qoi = _mesh_tensors()
    generator = torch.Generator().manual_seed(11)
    indices = _sample_supervised_indices(
        areas, count=96, generator=generator, qoi_weights=qoi,
        strategy="stratified_qoi",
    )
    hits = qoi[0][indices[0]].sum().item()
    assert hits / 96 > 0.15, hits


def test_sampling_is_deterministic_for_a_fixed_generator() -> None:
    _mesh, areas, qoi = _mesh_tensors()
    draws = [
        _sample_supervised_indices(
            areas, count=96, generator=torch.Generator().manual_seed(5),
            qoi_weights=qoi, strategy="stratified_qoi",
        )
        for _ in range(2)
    ]
    assert torch.equal(draws[0], draws[1])


def test_unknown_strategy_raises() -> None:
    _mesh, areas, qoi = _mesh_tensors()
    try:
        supervised_sampling_weights(areas, qoi, strategy="nonsense")
    except ValueError as error:
        assert "strategy" in str(error)
    else:
        raise AssertionError("unknown strategy must raise")
