"""Tests for Feature B: GeometryTokenizer and geometry-level OOD scoring."""

from __future__ import annotations

import pytest
import torch

from piginot.contracts import Unit
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from piginot.lifting import AffineDirichletLifting
from piginot.operators import (
    GEOMETRY_FEATURE_NAMES,
    BoundaryEnergyConfig,
    BoundaryEnergyOperator,
    BoundaryRoleMap,
    EnergyOperatorConfig,
    GeometryOODDetector,
    GeometryOODStatus,
    GeometryTokenizer,
    compute_geometry_descriptor,
)


def _rectangle_mesh(width: float = 2.0, height: float = 1.0):
    return canonicalize_mesh(
        MeshPayload(
            ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height)),
            ((0, 1, 2), (0, 2, 3)),
            CellType.TRIANGLE,
            ((0, 1), (1, 2), (2, 3), (3, 0)),
            {"bottom": (0,), "right": (1,), "top": (2,), "left": (3,)},
            (1, 1),
            Unit.MM,
            {"kind": f"rect-{width}x{height}"},
        )
    )


def _plate_with_hole_mesh():
    """Annular square [0,4]^2 minus central hole [1,3]^2, triangulated with 8 cells."""
    nodes = (
        (0.0, 0.0),  # 0
        (4.0, 0.0),  # 1
        (4.0, 4.0),  # 2
        (0.0, 4.0),  # 3
        (1.0, 1.0),  # 4
        (3.0, 1.0),  # 5
        (3.0, 3.0),  # 6
        (1.0, 3.0),  # 7
    )
    cells = (
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    )
    facets = (
        (0, 1),  # 0 bottom
        (1, 2),  # 1 right
        (2, 3),  # 2 top
        (3, 0),  # 3 left
        (4, 5),  # 4 hole
        (5, 6),  # 5 hole
        (6, 7),  # 6 hole
        (7, 4),  # 7 hole
    )
    return canonicalize_mesh(
        MeshPayload(
            nodes,
            cells,
            CellType.TRIANGLE,
            facets,
            {
                "bottom": (0,),
                "right": (1,),
                "top": (2,),
                "left": (3,),
                "hole": (4, 5, 6, 7),
            },
            (1,) * len(cells),
            Unit.MM,
            {"kind": "plate-with-hole"},
        )
    )


def _rect_roles() -> BoundaryRoleMap:
    return BoundaryRoleMap(
        {
            "left": "dirichlet_xy",
            "right": "dirichlet_x",
            "bottom": "symmetry",
            "top": "traction_free",
        }
    )


def _hole_roles() -> BoundaryRoleMap:
    return BoundaryRoleMap(
        {
            "left": "dirichlet_xy",
            "right": "dirichlet_x",
            "bottom": "traction_free",
            "top": "traction_free",
            "hole": "traction_free",
        }
    )


def test_tokenizer_detects_hole_loop_and_outward_normals() -> None:
    mesh = _plate_with_hole_mesh()
    tok = GeometryTokenizer(subdivisions_per_facet=2)
    xy, normals, features, weights, frame, topo = tok.tokenize_one(mesh, _hole_roles())

    assert topo.loop_count == 2
    assert topo.hole_count == 1
    assert topo.domain_area_mm2 == pytest.approx(12.0, abs=1e-12)
    assert topo.outer_loop_area_mm2 == pytest.approx(16.0, abs=1e-12)
    assert float(weights.sum()) == pytest.approx(24.0, abs=1e-12)
    assert xy.shape == (16, 2)
    assert features.shape == (16, len(GEOMETRY_FEATURE_NAMES))

    # All normals are unit vectors; hole normals point inward toward center (2, 2),
    # i.e. outward from the solid material.
    norms = torch.linalg.vector_norm(normals, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12)
    hole_mask = features[:, 1] > 0.5
    assert int(hole_mask.sum()) == 8
    center = torch.tensor([2.0, 2.0], dtype=torch.float64)
    to_center = center - xy[hole_mask]
    dots = torch.sum(to_center * normals[hole_mask], dim=-1)
    assert bool((dots > 0.0).all().item())
    assert frame.extent[0, 0].item() == pytest.approx(4.0)


def test_boundary_operator_is_permutation_and_padding_invariant() -> None:
    torch.manual_seed(0)
    energy_cfg = EnergyOperatorConfig(
        parameter_names=("strain",),
        hidden_dim=16,
        latent_token_count=8,
        parameter_hidden_dim=16,
        num_attention_heads=4,
        decoder_layers=2,
        positional_frequencies=3,
    )
    op = BoundaryEnergyOperator(
        BoundaryEnergyConfig(energy_cfg, GEOMETRY_FEATURE_NAMES)
    ).to(torch.float64)

    mesh = _plate_with_hole_mesh()
    tok = GeometryTokenizer(subdivisions_per_facet=2)
    batch_tight = tok.tokenize_batch([(mesh, _hole_roles())], dtype=torch.float64)
    batch_padded = tok.tokenize_batch(
        [(mesh, _hole_roles())], dtype=torch.float64, pad_to=24
    )

    enc_tight = op.encode_boundary(
        boundary_xy=batch_tight.boundary_xy,
        boundary_normals=batch_tight.boundary_normals,
        boundary_features=batch_tight.boundary_features,
        weights=batch_tight.weights,
        mask=batch_tight.mask,
        frame=batch_tight.frame,
    )
    enc_padded = op.encode_boundary(
        boundary_xy=batch_padded.boundary_xy,
        boundary_normals=batch_padded.boundary_normals,
        boundary_features=batch_padded.boundary_features,
        weights=batch_padded.weights,
        mask=batch_padded.mask,
        frame=batch_padded.frame,
    )
    assert torch.allclose(enc_tight.tokens, enc_padded.tokens, atol=1e-12, rtol=1e-12)

    # Permute point order inside batch_tight.
    perm = torch.randperm(batch_tight.boundary_xy.shape[1])
    enc_perm = op.encode_boundary(
        boundary_xy=batch_tight.boundary_xy[:, perm],
        boundary_normals=batch_tight.boundary_normals[:, perm],
        boundary_features=batch_tight.boundary_features[:, perm],
        weights=batch_tight.weights[:, perm],
        mask=batch_tight.mask[:, perm],
        frame=batch_tight.frame,
    )
    assert torch.allclose(enc_tight.tokens, enc_perm.tokens, atol=1e-12, rtol=1e-12)

    # Confirm displacement evaluation matches under permutation.
    queries = torch.tensor([[[0.5, 0.5], [3.5, 2.0]]], dtype=torch.float64)
    params = torch.tensor([[0.15]], dtype=torch.float64)

    def lift(q: torch.Tensor) -> AffineDirichletLifting:
        return AffineDirichletLifting(
            offset=torch.zeros_like(q), scale=torch.ones_like(q)
        )

    u1 = op.displacement(
        encoding=enc_tight, query_xy=queries, parameters=params, lifting=lift
    )
    u2 = op.displacement(
        encoding=enc_perm, query_xy=queries, parameters=params, lifting=lift
    )
    assert torch.allclose(u1, u2, atol=1e-12, rtol=1e-12)


def test_geometry_ood_detector_flags_ood_and_unsupported_hole_topology() -> None:
    roles = _rect_roles()
    train_meshes = [
        _rectangle_mesh(2.0, 1.0),
        _rectangle_mesh(2.2, 1.0),
        _rectangle_mesh(1.8, 1.0),
        _rectangle_mesh(2.1, 1.1),
    ]
    descriptors = [compute_geometry_descriptor(m, roles) for m in train_meshes]
    detector = GeometryOODDetector.fit(descriptors, descriptor_threshold=3.5)

    # In-distribution rectangle passes.
    rep_in = detector.score(_rectangle_mesh(2.05, 1.0), roles)
    assert rep_in.status is GeometryOODStatus.IN_DISTRIBUTION
    assert len(rep_in.report_hash) == 64

    # Extreme aspect-ratio rectangle flagged OUT_OF_DISTRIBUTION.
    rep_ood = detector.score(_rectangle_mesh(25.0, 1.0), roles)
    assert rep_ood.status is GeometryOODStatus.OUT_OF_DISTRIBUTION
    assert rep_ood.descriptor_distance is not None
    assert rep_ood.descriptor_distance > rep_ood.descriptor_threshold

    # Unseen hole topology explicitly refused as UNSUPPORTED_TOPOLOGY.
    rep_hole = detector.score(_plate_with_hole_mesh(), _hole_roles())
    assert rep_hole.status is GeometryOODStatus.UNSUPPORTED_TOPOLOGY
    assert rep_hole.observed_hole_count == 1
    assert rep_hole.supported_hole_counts == (0,)

    # Latent-token OOD scoring flags distant tokens even for an in-distribution shape.
    train_tokens = [
        torch.tensor([0.1 * i, -0.05 * i, 0.2, 0.0], dtype=torch.float64)
        for i in range(len(train_meshes))
    ]
    tok_detector = GeometryOODDetector.fit(
        descriptors,
        descriptor_threshold=5.0,
        train_token_means=train_tokens,
        token_threshold=2.0,
    )
    rep_tok_ood = tok_detector.score(
        _rectangle_mesh(2.05, 1.0),
        roles,
        pooled_tokens=torch.tensor([10.0, 10.0, 10.0, 10.0], dtype=torch.float64),
    )
    assert rep_tok_ood.status is GeometryOODStatus.OUT_OF_DISTRIBUTION
    assert rep_tok_ood.token_distance is not None
    assert rep_tok_ood.token_distance > 2.0

    # Missing boundary role triggers fail-closed UNSUPPORTED_TOPOLOGY in score().
    incomplete = BoundaryRoleMap({"left": "dirichlet_xy"})
    rep_incomplete = detector.score(_rectangle_mesh(), incomplete)
    assert rep_incomplete.status is GeometryOODStatus.UNSUPPORTED_TOPOLOGY

    with pytest.raises(ValueError, match="unsupported boundary role"):
        BoundaryRoleMap({"left": "robin"})
    with pytest.raises(ValueError, match="subdivisions_per_facet"):
        GeometryTokenizer(subdivisions_per_facet=0)
