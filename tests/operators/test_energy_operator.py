from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from piginot.contracts import CanonicalQuantity, MaterialSpec, Unit
from piginot.operators import (
    ENERGY_ARCHITECTURE_ID,
    AffineDirichletLifting,
    AxisAlignedTensionLifting,
    EnergyFormOperator,
    EnergyOperatorConfig,
    OperatorOutput,
    QueryFrame,
    constitutive_in_plane_stress,
    embed_plane_stress_deformation,
    energy_module_digest,
)
from piginot.physics import first_piola, plane_stress_closure

_DTYPE = torch.float64
_EXTENT = (27.0, 10.0)
_DISPLACEMENT = 4.05


def _material() -> MaterialSpec:
    return MaterialSpec(
        "energy-operator-test",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(760.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.23, Unit.ONE),
        },
    )


def _small_config(parameter_count: int = 3) -> EnergyOperatorConfig:
    return EnergyOperatorConfig(
        parameter_names=tuple(f"p{index}" for index in range(parameter_count)),
        hidden_dim=8,
        latent_token_count=3,
        parameter_hidden_dim=6,
        num_attention_heads=2,
        decoder_layers=2,
        positional_frequencies=3,
        output_init_std=0.3,
    )


def _model(
    seed: int = 7, config: EnergyOperatorConfig | None = None
) -> EnergyFormOperator:
    torch.manual_seed(seed)
    return EnergyFormOperator(config or _small_config()).to(_DTYPE)


def _frame(batch: int = 1) -> QueryFrame:
    return QueryFrame(
        torch.zeros((batch, 2), dtype=_DTYPE),
        torch.tensor([_EXTENT] * batch, dtype=_DTYPE),
    )


def _lifting(batch: int = 1) -> AxisAlignedTensionLifting:
    return AxisAlignedTensionLifting(
        torch.tensor([_EXTENT] * batch, dtype=_DTYPE),
        torch.full((batch,), _DISPLACEMENT, dtype=_DTYPE),
    )


def _queries(count: int = 17, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    points = torch.rand((1, count, 2), generator=generator, dtype=_DTYPE)
    return points * torch.tensor(_EXTENT, dtype=_DTYPE)


def _parameters(batch: int = 1, width: int = 3) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, batch * width, dtype=_DTYPE).reshape(batch, width)


def _displacement(model: EnergyFormOperator, query: torch.Tensor) -> torch.Tensor:
    return model.displacement(
        query_xy=query,
        parameters=_parameters(query.shape[0]),
        frame=_frame(query.shape[0]),
        lifting=_lifting(query.shape[0]),
    )


def test_default_configuration_reproduces_the_enhanced_parameter_count() -> None:
    four = EnergyFormOperator(
        EnergyOperatorConfig(parameter_names=("L", "W", "w", "R"))
    )
    five = EnergyFormOperator(
        EnergyOperatorConfig(parameter_names=("L", "W", "w", "R", "strain"))
    )

    assert sum(p.numel() for p in four.parameters()) == 573_378
    assert sum(p.numel() for p in five.parameters()) == 573_474
    assert four.architecture_id == ENERGY_ARCHITECTURE_ID


@pytest.mark.parametrize(
    "changes",
    [
        {"parameter_names": ()},
        {"parameter_names": ("a", "a")},
        {"parameter_names": (" a",)},
        {"hidden_dim": 0},
        {"hidden_dim": 9},
        {"decoder_layers": True},
        {"output_init_std": 0.0},
        {"output_init_std": math.inf},
    ],
)
def test_config_rejects_invalid_contracts(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(_small_config(), **changes)


def test_operator_requires_exact_config() -> None:
    with pytest.raises(TypeError):
        EnergyFormOperator(object())  # type: ignore[arg-type]


def test_dirichlet_lifting_is_exact_on_every_constrained_edge() -> None:
    model = _model()
    heights = torch.linspace(0.0, _EXTENT[1], 9, dtype=_DTYPE)
    lengths = torch.linspace(0.0, _EXTENT[0], 9, dtype=_DTYPE)
    left = torch.stack((torch.zeros_like(heights), heights), dim=-1)
    right = torch.stack((torch.full_like(heights, _EXTENT[0]), heights), dim=-1)
    bottom = torch.stack((lengths, torch.zeros_like(lengths)), dim=-1)
    query = torch.cat((left, right, bottom)).unsqueeze(0)

    u = _displacement(model, query)[0]

    assert torch.equal(u[:9, 0], torch.zeros(9, dtype=_DTYPE))
    assert torch.allclose(
        u[9:18, 0], torch.full((9,), _DISPLACEMENT, dtype=_DTYPE), atol=1e-13
    )
    assert torch.equal(u[18:, 1], torch.zeros(9, dtype=_DTYPE))
    assert bool((u[18:, 0] != 0.0).any())


def test_initial_field_is_close_to_the_uniform_stretch_baseline() -> None:
    model = EnergyFormOperator(replace(_small_config(), output_init_std=1.0e-5)).to(
        _DTYPE
    )
    query = _queries()
    u = _displacement(model, query)[0]
    baseline = _DISPLACEMENT * query[0, :, 0] / _EXTENT[0]

    assert torch.allclose(u[:, 0], baseline, atol=1e-3)
    assert float(u[:, 1].detach().abs().max()) < 1e-3


def test_decoder_is_point_local_and_order_invariant() -> None:
    model = _model()
    query = _queries(20)
    permutation = torch.randperm(20, generator=torch.Generator().manual_seed(5))
    full = _displacement(model, query)
    permuted = _displacement(model, query[:, permutation])
    halves = torch.cat(
        (_displacement(model, query[:, :7]), _displacement(model, query[:, 7:])),
        dim=1,
    )

    assert torch.allclose(permuted, full[:, permutation], atol=1e-14, rtol=0.0)
    assert torch.allclose(halves, full, atol=1e-14, rtol=0.0)


def test_identical_seeds_give_identical_operators() -> None:
    first = _model(seed=11)
    second = _model(seed=11)
    third = _model(seed=12)
    query = _queries()

    assert torch.equal(_displacement(first, query), _displacement(second, query))
    assert not torch.equal(_displacement(first, query), _displacement(third, query))


def test_displacement_gradient_matches_finite_differences() -> None:
    model = _model()
    query = _queries(5)
    _, gradient = model.displacement_and_gradient(
        query_xy=query,
        parameters=_parameters(),
        frame=_frame(),
        lifting=_lifting(),
        create_graph=False,
    )
    step = 1.0e-6
    for axis in range(2):
        shift = torch.zeros_like(query)
        shift[..., axis] = step
        difference = (
            _displacement(model, query + shift) - _displacement(model, query - shift)
        ) / (2.0 * step)
        assert torch.allclose(gradient[0, :, :, axis], difference[0], atol=1e-7)


def _unreached_parameters(model: EnergyFormOperator) -> tuple[torch.Tensor, list[str]]:
    parameters = _parameters().clone().requires_grad_(True)
    u, gradient = model.displacement_and_gradient(
        query_xy=_queries(),
        parameters=parameters,
        frame=_frame(),
        lifting=_lifting(),
        create_graph=True,
    )
    (u.square().sum() + gradient.square().sum()).backward()
    assert parameters.grad is not None
    return parameters.grad, [
        name
        for name, value in model.named_parameters()
        if value.grad is None or not bool((value.grad != 0.0).any())
    ]


def test_every_parameter_and_conditioning_input_reaches_the_output() -> None:
    initial = _model()
    conditioning_gradient, unreached = _unreached_parameters(initial)
    # The pooled-latent bias starts as an exact zero map (enhanced design), so
    # only its first layer is unreached until its output layer has moved.
    assert unreached == ["latent_bias.0.weight", "latent_bias.0.bias"]
    assert bool((conditioning_gradient != 0.0).all())

    trained = _model()
    with torch.no_grad():
        trained.latent_bias[-1].weight.normal_(0.0, 0.1)
    _, unreached_after_update = _unreached_parameters(trained)
    assert unreached_after_update == []


def test_forward_serves_the_constitutive_stress_of_the_predicted_gradient() -> None:
    model = _model()
    query = _queries(6)
    material = _material()

    output = model(
        query_xy=query,
        parameters=_parameters(),
        frame=_frame(),
        lifting=_lifting(),
        materials=(material,),
    )
    _, gradient = model.displacement_and_gradient(
        query_xy=query,
        parameters=_parameters(),
        frame=_frame(),
        lifting=_lifting(),
        create_graph=False,
    )

    assert type(output) is OperatorOutput
    assert output.u.shape == (1, 6, 2)
    assert output.P.shape == (1, 6, 2, 2)
    assert torch.allclose(
        output.P[0], constitutive_in_plane_stress(gradient[0], material), atol=1e-12
    )
    with pytest.raises(ValueError, match="one MaterialSpec per sample"):
        model(
            query_xy=query,
            parameters=_parameters(),
            frame=_frame(),
            lifting=_lifting(),
            materials=(),
        )


def test_constitutive_stress_is_the_plane_stress_neo_hookean_stress() -> None:
    material = _material()
    gradient = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0]], [[0.15, 0.02], [-0.01, -0.03]]], dtype=_DTYPE
    )
    stress = constitutive_in_plane_stress(gradient, material)
    deformation_2d = torch.eye(2, dtype=_DTYPE) + gradient
    f33 = plane_stress_closure(deformation_2d, material)
    full = first_piola(embed_plane_stress_deformation(deformation_2d, f33), material)

    assert torch.allclose(stress[0], torch.zeros((2, 2), dtype=_DTYPE), atol=1e-12)
    assert torch.allclose(stress, full[..., :2, :2])
    assert float(full[1, 2, 2].abs()) < 1e-9
    with pytest.raises(ValueError):
        constitutive_in_plane_stress(torch.zeros((2, 2), dtype=_DTYPE), material)


def test_embedding_rejects_misaligned_closure() -> None:
    with pytest.raises(ValueError):
        embed_plane_stress_deformation(
            torch.eye(2, dtype=_DTYPE).expand(3, 2, 2), torch.ones(2, dtype=_DTYPE)
        )


def test_nerf_encoding_matches_the_enhanced_layout() -> None:
    model = _model()
    coordinates = torch.tensor([[[0.25, -0.5]]], dtype=_DTYPE)
    encoded = model.query_encoding(coordinates)
    scales = 2.0 ** torch.arange(3, dtype=_DTYPE)
    scaled = (coordinates.reshape(-1, 1, 2) * scales.view(1, -1, 1)).reshape(1, 1, -1)
    expected = torch.cat(
        (coordinates, torch.cat((scaled, scaled + math.pi / 2.0), dim=-1).sin()),
        dim=-1,
    )

    assert encoded.shape[-1] == model.config.positional_dim
    assert torch.allclose(encoded, expected)


def test_module_digest_binds_architecture_not_learned_values() -> None:
    first = _model(seed=1)
    second = _model(seed=2)
    wider = EnergyFormOperator(replace(_small_config(), hidden_dim=10)).to(_DTYPE)

    assert energy_module_digest(first) == energy_module_digest(second)
    assert energy_module_digest(first) != energy_module_digest(wider)
    assert energy_module_digest(first).startswith("blake2b256:")
    with pytest.raises(TypeError):
        energy_module_digest(torch.nn.Linear(1, 1))  # type: ignore[arg-type]


def test_frame_and_lifting_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        QueryFrame(torch.zeros((1, 2)), torch.zeros((1, 2)))
    with pytest.raises(ValueError, match="same shape"):
        QueryFrame(torch.zeros((1, 2)), torch.ones((2, 2)))
    with pytest.raises(ValueError, match="share a batch"):
        _frame().normalize(torch.zeros((2, 3, 2), dtype=_DTYPE))
    with pytest.raises(ValueError, match="strictly positive"):
        AxisAlignedTensionLifting(torch.zeros((1, 2)), torch.ones(1))
    with pytest.raises(ValueError, match="shape"):
        AxisAlignedTensionLifting(torch.ones((1, 2)), torch.ones(2))
    with pytest.raises(TypeError):
        AxisAlignedTensionLifting(torch.ones((1, 2)), 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        AxisAlignedTensionLifting(torch.ones((1, 2)), torch.tensor([math.nan]))
    with pytest.raises(ValueError, match="share a batch"):
        _lifting()(torch.zeros((2, 3, 2), dtype=_DTYPE))
    with pytest.raises(ValueError, match="same shape"):
        AffineDirichletLifting(torch.zeros((1, 2, 2)), torch.zeros((1, 3, 2)))


def test_displacement_rejects_inconsistent_inputs() -> None:
    model = _model()
    query = _queries()
    with pytest.raises(TypeError):
        model.displacement(
            query_xy=query,
            parameters=_parameters(),
            frame=object(),  # type: ignore[arg-type]
            lifting=_lifting(),
        )
    with pytest.raises(ValueError, match="batch"):
        model.displacement(
            query_xy=query,
            parameters=_parameters(2),
            frame=_frame(),
            lifting=_lifting(),
        )
    with pytest.raises(ValueError, match="shape"):
        model.encode_parameters(torch.zeros((1, 4), dtype=_DTYPE))
    with pytest.raises(TypeError, match="AffineDirichletLifting"):
        model.displacement(
            query_xy=query,
            parameters=_parameters(),
            frame=_frame(),
            lifting=lambda points: points,  # type: ignore[arg-type,return-value]
        )
    with pytest.raises(ValueError, match="align"):
        model.displacement(
            query_xy=query,
            parameters=_parameters(),
            frame=_frame(),
            lifting=lambda points: AffineDirichletLifting(
                torch.zeros((1, 2, 2), dtype=_DTYPE),
                torch.zeros((1, 2, 2), dtype=_DTYPE),
            ),
        )
    with pytest.raises(ValueError, match="finite"):
        model.displacement(
            query_xy=torch.full((1, 2, 2), math.nan, dtype=_DTYPE),
            parameters=_parameters(),
            frame=_frame(),
            lifting=_lifting(),
        )
