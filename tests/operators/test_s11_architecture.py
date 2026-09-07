from __future__ import annotations

from dataclasses import replace
from typing import NotRequired, TypedDict

import pytest
import torch

from piginot.operators import (
    ARCHITECTURE_ID,
    BoundaryConditionMode,
    DirichletData,
    LiftingVerification,
    OperatorConfig,
    PiGinotOperator,
    module_digest,
)

PARAMETER_NAMES = (
    "L_total",
    "W_grip",
    "W_gauge",
    "R_fillet",
    "E",
    "nu",
    "nominal_axial_strain",
)
TAG_NAMES = ("symmetry_x", "symmetry_y", "grip", "traction_free")


class OperatorInputs(TypedDict):
    boundary_xy: torch.Tensor
    boundary_tags: torch.Tensor
    boundary_normals: torch.Tensor
    boundary_weights: torch.Tensor
    boundary_mask: torch.Tensor
    query_xy: torch.Tensor
    parameters: torch.Tensor
    dirichlet: NotRequired[DirichletData]


def _config(**overrides: object) -> OperatorConfig:
    values: dict[str, object] = {
        "parameter_names": PARAMETER_NAMES,
        "boundary_tag_names": TAG_NAMES,
        "hidden_dim": 32,
        "latent_token_count": 8,
        "num_attention_heads": 4,
        "num_fourier_frequencies": 4,
        "encoder_self_attention_layers": 1,
        "decoder_cross_attention_layers": 2,
    }
    values.update(overrides)
    return OperatorConfig(**values)  # type: ignore[arg-type]


def _model(
    config: OperatorConfig | None = None,
    *,
    lifting_verification: LiftingVerification | None = None,
) -> PiGinotOperator:
    torch.manual_seed(11012)
    return PiGinotOperator(
        config or _config(), lifting_verification=lifting_verification
    ).to(dtype=torch.float64)


def _boundary_cloud(
    count: int,
    *,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    angle = torch.arange(count, dtype=torch.float64) * (2.0 * torch.pi / count)
    base_xy = torch.stack((1.4 * torch.cos(angle), 0.7 * torch.sin(angle)), dim=-1)
    base_normals = torch.nn.functional.normalize(
        torch.stack((torch.cos(angle) / 1.4, torch.sin(angle) / 0.7), dim=-1),
        dim=-1,
    )
    quadrant = torch.remainder(torch.floor(2.0 * angle / torch.pi), 4).long()
    base_tags = torch.nn.functional.one_hot(quadrant, num_classes=4).to(torch.float64)
    xy = base_xy.unsqueeze(0).repeat(batch_size, 1, 1)
    tags = base_tags.unsqueeze(0).repeat(batch_size, 1, 1)
    normals = base_normals.unsqueeze(0).repeat(batch_size, 1, 1)
    weights = torch.full((batch_size, count), 1.0 / count, dtype=torch.float64)
    return xy, tags, normals, weights


def _inputs(
    *,
    boundary_points: int = 64,
    query_points: int = 11,
    batch_size: int = 1,
) -> OperatorInputs:
    boundary_xy, tags, normals, weights = _boundary_cloud(
        boundary_points, batch_size=batch_size
    )
    generator = torch.Generator(device="cpu").manual_seed(11011)
    query_xy = torch.rand(
        batch_size, query_points, 2, generator=generator, dtype=torch.float64
    )
    query_xy[..., 0] = 2.4 * query_xy[..., 0] - 1.2
    query_xy[..., 1] = 1.2 * query_xy[..., 1] - 0.6
    parameters = torch.rand(
        batch_size, len(PARAMETER_NAMES), generator=generator, dtype=torch.float64
    )
    return {
        "boundary_xy": boundary_xy,
        "boundary_tags": tags,
        "boundary_normals": normals,
        "boundary_weights": weights,
        "boundary_mask": torch.ones(batch_size, boundary_points, dtype=torch.bool),
        "query_xy": query_xy,
        "parameters": parameters,
    }


def _perturb_boundary(inputs: OperatorInputs, name: str) -> OperatorInputs:
    changed = inputs.copy()
    if name == "boundary_xy":
        changed["boundary_xy"] = inputs["boundary_xy"].clone()
        changed["boundary_xy"][:, 0] += 0.2
    elif name == "boundary_tags":
        changed["boundary_tags"] = inputs["boundary_tags"].clone()
        changed["boundary_tags"][:, 0] += 0.2
    elif name == "boundary_normals":
        changed["boundary_normals"] = inputs["boundary_normals"].clone()
        changed["boundary_normals"][:, 0] += 0.2
    else:
        raise ValueError(f"unknown boundary input {name!r}")
    return changed


def _permute_boundary(inputs: OperatorInputs, order: torch.Tensor) -> OperatorInputs:
    changed = inputs.copy()
    changed["boundary_xy"] = inputs["boundary_xy"][:, order]
    changed["boundary_tags"] = inputs["boundary_tags"][:, order]
    changed["boundary_normals"] = inputs["boundary_normals"][:, order]
    changed["boundary_weights"] = inputs["boundary_weights"][:, order]
    changed["boundary_mask"] = inputs["boundary_mask"][:, order]
    return changed


def _relative_change(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(left - right)
    denominator = torch.clamp(torch.linalg.vector_norm(left), min=1.0e-12)
    return float((numerator / denominator).item())


def test_architecture_identity_is_fixed_and_weight_independent() -> None:
    model = _model()
    digest = module_digest(model)

    with torch.no_grad():
        next(model.parameters()).add_(7.0)

    assert ARCHITECTURE_ID == "pi-ginot-s11-boundary-cross-attention-v1"
    assert model.architecture_id == ARCHITECTURE_ID
    assert digest.startswith("blake2b256:")
    assert module_digest(model) == digest
    assert module_digest(_model(replace(_config(), hidden_dim=64))) != digest


def test_encode_once_matches_full_forward_and_uses_all_boundary_features() -> None:
    model = _model()
    inputs = _inputs()

    encoded = model.encode_boundary(
        boundary_xy=inputs["boundary_xy"],
        boundary_tags=inputs["boundary_tags"],
        boundary_normals=inputs["boundary_normals"],
        boundary_weights=inputs["boundary_weights"],
        boundary_mask=inputs["boundary_mask"],
    )
    decoded = model.decode_queries(
        encoding=encoded,
        query_xy=inputs["query_xy"],
        parameters=inputs["parameters"],
    )
    direct = model(**inputs)

    torch.testing.assert_close(decoded.u, direct.u)
    torch.testing.assert_close(decoded.P, direct.P)
    for name in ("boundary_xy", "boundary_tags", "boundary_normals"):
        assert not torch.allclose(
            model(**_perturb_boundary(inputs, name)).flat(), direct.flat()
        )


def test_heads_have_mechanics_shapes_and_served_stress_is_predicted_head() -> None:
    model = _model()
    inputs = _inputs(batch_size=2)

    before = model(**inputs)
    with torch.no_grad():
        model.u_head.weight.zero_()
        model.u_head.bias.zero_()
    after = model(**inputs)

    assert before.u.shape == (2, 11, 2)
    assert before.P.shape == (2, 11, 2, 2)
    assert before.served_stress is before.P
    assert torch.count_nonzero(after.u) == 0
    torch.testing.assert_close(before.P, after.P)


def test_boundary_permutation_gate_runs_one_hundred_random_permutations() -> None:
    model = _model()
    inputs = _inputs(boundary_points=320)
    baseline = model(**inputs).flat()
    generator = torch.Generator(device="cpu").manual_seed(11013)
    maximum = 0.0

    for _ in range(100):
        order = torch.randperm(320, generator=generator)
        permuted = _permute_boundary(inputs, order)
        maximum = max(maximum, _relative_change(baseline, model(**permuted).flat()))

    assert maximum < 1.0e-5


def test_boundary_resampling_gate_uses_exact_320_and_480_populations() -> None:
    model = _model()
    low = _inputs(boundary_points=320)
    high = _inputs(boundary_points=480)
    high["query_xy"] = low["query_xy"]
    high["parameters"] = low["parameters"]

    change = _relative_change(model(**low).flat(), model(**high).flat())

    assert low["boundary_xy"].shape[1] == 320
    assert high["boundary_xy"].shape[1] == 480
    assert change < 0.02


def test_boundary_mask_ignores_padding_and_ignore_mask_mutant_fires() -> None:
    model = _model()
    inputs = _inputs(boundary_points=64)
    baseline = model(**inputs).flat()
    padded = inputs.copy()
    padded["boundary_xy"] = torch.cat(
        (
            inputs["boundary_xy"],
            torch.full_like(inputs["boundary_xy"][:, :8], 1000.0),
        ),
        dim=1,
    )
    padded["boundary_tags"] = torch.cat(
        (
            inputs["boundary_tags"],
            torch.full_like(inputs["boundary_tags"][:, :8], 1000.0),
        ),
        dim=1,
    )
    padded["boundary_normals"] = torch.cat(
        (
            inputs["boundary_normals"],
            torch.full_like(inputs["boundary_normals"][:, :8], 1000.0),
        ),
        dim=1,
    )
    padded["boundary_weights"] = torch.cat(
        (
            inputs["boundary_weights"],
            torch.ones_like(inputs["boundary_weights"][:, :8]),
        ),
        dim=1,
    )
    padded["boundary_mask"] = torch.cat(
        (inputs["boundary_mask"], torch.zeros(1, 8, dtype=torch.bool)), dim=1
    )

    control = model(**padded).flat()
    mutant = padded.copy()
    mutant["boundary_mask"] = torch.ones_like(padded["boundary_mask"])

    torch.testing.assert_close(control, baseline, rtol=0.0, atol=1.0e-10)
    assert _relative_change(baseline, model(**mutant).flat()) > 1.0e-3


def test_node_order_and_query_batching_are_invariant() -> None:
    model = _model()
    inputs = _inputs(query_points=17)
    together = model(**inputs)
    order = torch.tensor([3, 0, 16, 1, 5, 2, 9, 7, 8, 4, 6, 10, 15, 14, 13, 12, 11])
    inverse = torch.argsort(order)
    permuted_inputs = inputs.copy()
    permuted_inputs["query_xy"] = inputs["query_xy"][:, order]
    permuted = model(**permuted_inputs)

    assert _relative_change(together.u, permuted.u[:, inverse]) < 1.0e-6
    assert _relative_change(together.P, permuted.P[:, inverse]) < 1.0e-6

    pieces = []
    for query_chunk in torch.split(inputs["query_xy"], [5, 6, 6], dim=1):
        chunk = inputs.copy()
        chunk["query_xy"] = query_chunk
        pieces.append(model(**chunk))
    joined_u = torch.cat([part.u for part in pieces], dim=1)
    joined_p = torch.cat([part.P for part in pieces], dim=1)
    assert _relative_change(together.u, joined_u) < 1.0e-6
    assert _relative_change(together.P, joined_p) < 1.0e-6


def test_every_named_parameter_reaches_both_output_heads() -> None:
    model = _model()
    inputs = _inputs()
    parameters = inputs["parameters"].clone().requires_grad_(True)
    inputs["parameters"] = parameters
    output = model(**inputs)

    u_gradient = torch.autograd.grad(
        output.u.square().sum(), parameters, retain_graph=True
    )[0]
    p_gradient = torch.autograd.grad(output.P.square().sum(), parameters)[0]

    assert tuple(model.config.parameter_names) == PARAMETER_NAMES
    assert torch.all(torch.abs(u_gradient) > 1.0e-12)
    assert torch.all(torch.abs(p_gradient) > 1.0e-12)


def _lifting_verification() -> LiftingVerification:
    return LiftingVerification(
        verification_id="dogbone-distance-lifting-fixture-v1",
        parameter_range_hash="1" * 64,
        verification_hash="2" * 64,
        boundary_refinement_levels=(1, 2, 3),
        full_parameter_range_verified=True,
        max_boundary_residual=1.0e-12,
    )


def test_soft_dirichlet_is_default_and_does_not_silently_lift() -> None:
    model = _model()
    inputs = _inputs(query_points=5)
    raw = model(**inputs)
    data = DirichletData(
        component_mask=torch.ones(1, 5, 2, dtype=torch.bool),
        prescribed_u=torch.full((1, 5, 2), 42.0, dtype=torch.float64),
        approximate_distance=torch.zeros(1, 5, 2, dtype=torch.float64),
    )
    inputs["dirichlet"] = data
    with_data = model(**inputs)

    assert model.config.boundary_condition_mode is BoundaryConditionMode.SOFT
    assert model.config.soft_dirichlet_residual_tolerance == 1.0e-6
    torch.testing.assert_close(with_data.u, raw.u)
    assert float(model.dirichlet_residual(with_data, data).detach().item()) > 1.0


def test_hard_lifting_requires_verification_and_is_component_wise() -> None:
    hard_config = _config(boundary_condition_mode=BoundaryConditionMode.HARD_LIFTING)
    with pytest.raises(ValueError, match="verified distance-lifting"):
        PiGinotOperator(hard_config)

    model = _model(hard_config, lifting_verification=_lifting_verification())
    inputs = _inputs(query_points=4)
    component_mask = torch.tensor(
        [[[True, False], [False, True], [True, True], [False, False]]]
    )
    prescribed = torch.tensor(
        [[[1.0, 9.0], [8.0, 2.0], [3.0, 4.0], [7.0, 6.0]]],
        dtype=torch.float64,
    )
    distance = torch.ones(1, 4, 2, dtype=torch.float64)
    distance[component_mask] = 0.0
    data = DirichletData(component_mask, prescribed, distance)
    inputs["dirichlet"] = data

    output = model(**inputs)

    torch.testing.assert_close(output.u[component_mask], prescribed[component_mask])
    assert float(model.dirichlet_residual(output, data).detach().item()) == 0.0


@pytest.mark.parametrize(
    "change",
    (
        {"parameter_names": ("L_total", "L_total")},
        {"boundary_tag_names": ()},
        {"hidden_dim": 30, "num_attention_heads": 4},
        {"soft_dirichlet_residual_tolerance": 0.0},
    ),
)
def test_config_rejects_ambiguous_or_invalid_contracts(
    change: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _config(**change)


def test_input_contract_rejects_invalid_masks_weights_and_parameter_width() -> None:
    model = _model()
    inputs = _inputs()
    bad_mask = inputs.copy()
    bad_mask["boundary_mask"] = torch.zeros_like(inputs["boundary_mask"])
    with pytest.raises(ValueError, match="at least one"):
        model(**bad_mask)

    bad_weights = inputs.copy()
    bad_weights["boundary_weights"] = -torch.ones_like(inputs["boundary_weights"])
    with pytest.raises(ValueError, match="non-negative"):
        model(**bad_weights)

    bad_parameters = inputs.copy()
    bad_parameters["parameters"] = inputs["parameters"][:, :-1]
    with pytest.raises(ValueError, match="parameters"):
        model(**bad_parameters)
