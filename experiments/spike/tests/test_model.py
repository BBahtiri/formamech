from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

import experiments.spike.model as spike_model
from experiments.spike.model import (
    MixedPhysicsOperator,
    ModelConfig,
    equilibrium_residual,
)

EXPECTED_UPSTREAM_SOURCE_SHA256 = {
    "LICENSE": "64fd2f3673386742e9374f2fceb125fbc1a4b80cb687e7a8206e2ebfd1338eeb",
    "config.py": "bd610289505b574ef4f9addf5ebd437178519686c4c2c60ea501b387ce2294e1",
    "models/__init__.py": "bbcf1c958d0c8c10ff8e49bcbb70d531d86dfcec6170d1037ca032bf02ea0019",
    "models/geometry_encoder.py": "fe7466e1eb10f2adce81b78da8a4ccec6b4aae72d3adb8976d3bd900716c824b",
    "models/physics_decoder.py": "305c784733c6aea6a28d65f3ddf31ba46e39454b10cc09cb705bece16df0e557",
    "models/pi_ginot.py": "b31033fa294c061ef16df421ab4e007b6e04f8edeeded933a8dfa8873cc660c5",
    "models/modules/__init__.py": "c684199acfc7d30c6e49e75e4cf30c9e0cde8b24c9f36d4b1136418b9e918584",
    "models/modules/point_encoding.py": "7a2823e8641869279d89b33f7e213893524a7bdebb931195f883925b6600db28",
    "models/modules/point_position_embedding.py": "e253bb32ceeab2c37358a75f773785955ffb13b8e1596d77d4ebf7f9ad1d7df0",
    "models/modules/pointnet2_utils.py": "ff191b5f3046792d73675bb75fe4a604e31c07756f6bf44c1e9abd36d5f98a22",
    "models/modules/transformer.py": "a329673ffeb93229b8005ef46862b101ba5ad512163c6026956437996428368f",
}


def _tiny_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(17)
    query = torch.tensor([[[0.3, 0.2], [0.7, 0.4], [0.5, 0.8]]], dtype=torch.float32)
    boundary = torch.rand((1, 20, 2), dtype=torch.float32)
    u_delta = torch.tensor([0.1])
    x_max = torch.tensor([1.0])
    y_max = torch.tensor([1.0])
    return query, boundary, u_delta, x_max, y_max


def test_default_configuration_recovers_the_authoritative_decoder_dimensions() -> None:
    config = ModelConfig()

    assert config.boundary_points == 320
    assert config.latent_tokens == 32
    assert config.embed_dim == 64
    assert config.num_heads == 4
    assert config.encoder_self_layers == 3
    assert config.decoder_cross_layers == 6
    assert config.positional_encoding_max_degree == 6
    assert config.stress_head_variant == "conditioned_mlp"


def test_model_vendors_the_exact_pinned_upstream_sources() -> None:
    assert spike_model.UPSTREAM_COMMIT == ("9f6794141beca9d3e66433465c3b07039a7e41d2")
    assert spike_model.UPSTREAM_SOURCE_SHA256 == EXPECTED_UPSTREAM_SOURCE_SHA256

    vendor_root = Path(spike_model.__file__).parent / "upstream_pi_ginot"
    measured = {
        relative: hashlib.sha256(
            (vendor_root / relative)
            .read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .encode()
        ).hexdigest()
        for relative in EXPECTED_UPSTREAM_SOURCE_SHA256
    }
    assert measured == EXPECTED_UPSTREAM_SOURCE_SHA256


def test_mixed_operator_preserves_the_exact_upstream_displacement_path() -> None:
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
    model = MixedPhysicsOperator(config).eval()
    query, boundary, u_delta, x_max, y_max = _tiny_inputs()

    assert isinstance(model, spike_model.UpstreamPIGINOT)
    assert model.encoder.__class__.__module__.startswith(
        "experiments.spike.upstream_pi_ginot.models"
    )
    assert model.decoder.__class__.__module__.startswith(
        "experiments.spike.upstream_pi_ginot.models"
    )
    upstream = spike_model.UpstreamPIGINOT(
        config.encoder_config(), config.decoder_config()
    ).eval()
    upstream.load_state_dict(model.upstream_state_dict())
    with torch.no_grad():
        upstream_u = upstream(query, boundary, u_delta, x_max, y_max)
        mixed_u = model(query, boundary, u_delta, x_max, y_max).u

    torch.testing.assert_close(mixed_u, upstream_u, atol=0.0, rtol=0.0)


def test_upstream_checkpoint_state_loads_into_the_unmodified_backbone() -> None:
    model = MixedPhysicsOperator(
        ModelConfig(
            boundary_points=20,
            latent_tokens=4,
            embed_dim=16,
            num_heads=4,
            encoder_self_layers=1,
            decoder_cross_layers=2,
            positional_encoding_max_degree=2,
            neighbor_count=4,
        )
    )

    outcome = model.load_upstream_state_dict(model.upstream_state_dict())

    assert outcome.missing_keys
    assert all(key.startswith("p_head.") for key in outcome.missing_keys)
    assert outcome.unexpected_keys == []


def test_spike_model_does_not_reimplement_upstream_network_blocks() -> None:
    forbidden = {
        "_FiLM",
        "_GeometryEncoder",
        "_ResidualCrossAttention",
        "_ResidualSelfAttention",
        "_farthest_point_indices",
        "_gather_points",
        "_positional_encoding",
    }

    assert forbidden.isdisjoint(vars(spike_model))


def test_mixed_state_is_upstream_state_plus_the_direct_p_head_only() -> None:
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
    upstream = spike_model.UpstreamPIGINOT(
        config.encoder_config(), config.decoder_config()
    )
    mixed = MixedPhysicsOperator(config)

    upstream_keys = set(upstream.state_dict())
    mixed_keys = set(mixed.state_dict())
    assert upstream_keys - mixed_keys == set()
    assert mixed_keys - upstream_keys
    assert all(key.startswith("p_head.") for key in mixed_keys - upstream_keys)
    assert mixed.p_head.output.out_features == 4


def test_conditioned_stress_head_restores_fixed_displacement_scale_response() -> None:
    config = ModelConfig(
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
    model = MixedPhysicsOperator(config).eval()
    query, boundary, u_delta, x_max, y_max = _tiny_inputs()
    with torch.no_grad():
        model.p_head.output.weight.fill_(0.01)
        model.p_head.output.bias.fill_(1.0)
        small = model(query, boundary, u_delta, x_max, y_max).P_served
        large = model(
            2.0 * query,
            2.0 * boundary,
            u_delta,
            2.0 * x_max,
            2.0 * y_max,
        ).P_served

    torch.testing.assert_close(large, 0.5 * small, atol=1e-6, rtol=1e-6)


def test_fully_conditioned_stress_head_uses_strain_and_query_location() -> None:
    head = spike_model.DirectStressHead(4, "fully_conditioned_mlp").eval()
    features = torch.zeros(1, 2, 4)
    query = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]])
    x_max = torch.tensor([1.0])
    y_max = torch.tensor([1.0])
    with torch.no_grad():
        hidden_linear = head.hidden[0]
        hidden_linear.weight.zero_()
        hidden_linear.bias.zero_()
        head.output.weight.zero_()
        head.output.bias.zero_()
        head.output.weight[:, 0] = 1.0

        hidden_linear.weight[0, 5] = 1.0
        low_load = head(
            features[:, :1],
            query[:, :1],
            torch.tensor([0.1]),
            x_max,
            y_max,
        )
        high_load = head(
            features[:, :1],
            query[:, :1],
            torch.tensor([0.2]),
            x_max,
            y_max,
        )
        low_dimensionless = low_load / (spike_model.MATERIAL_YOUNGS_MODULUS * 0.1)
        high_dimensionless = high_load / (spike_model.MATERIAL_YOUNGS_MODULUS * 0.2)

        hidden_linear.weight.zero_()
        hidden_linear.weight[0, 6] = 1.0
        stress_by_location = head(
            features,
            query,
            torch.tensor([0.1]),
            x_max,
            y_max,
        )

    assert not torch.allclose(low_dimensionless, high_dimensionless)
    assert not torch.allclose(stress_by_location[:, 0], stress_by_location[:, 1])


def test_mixed_heads_have_correct_shapes_and_predicted_p_is_authoritative() -> None:
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
    model = MixedPhysicsOperator(config).eval()
    inputs = _tiny_inputs()

    with patch(
        "experiments.spike.model.material_first_piola_from_displacement",
        side_effect=AssertionError("the material oracle must not serve stress"),
    ):
        baseline = model(*inputs)

    assert baseline.u.shape == (1, 3, 2)
    assert baseline.P_served.shape == (1, 3, 2, 2)
    assert baseline.P_predicted.data_ptr() == baseline.P_served.data_ptr()

    with torch.no_grad():
        model.u_head.bias.add_(0.5)
    changed_u = model(*inputs)
    assert not torch.allclose(changed_u.u, baseline.u)
    torch.testing.assert_close(changed_u.P_served, baseline.P_served)

    with torch.no_grad():
        model.p_head.output.bias.add_(0.5)
    changed_p = model(*inputs)
    torch.testing.assert_close(changed_p.u, changed_u.u)
    assert not torch.allclose(changed_p.P_served, changed_u.P_served)


def test_equilibrium_uses_the_direct_first_piola_divergence_and_backpropagates() -> (
    None
):
    coordinates = torch.tensor(
        [[[0.2, 0.3], [0.6, 0.7]]], dtype=torch.float64, requires_grad=True
    )
    scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    x = coordinates[..., 0]
    y = coordinates[..., 1]
    stress = scale * torch.stack(
        (
            torch.stack((x**2 + 3.0 * y, 4.0 * x + y**2), dim=-1),
            torch.stack((x * y, x - 5.0 * y), dim=-1),
        ),
        dim=-2,
    )
    exact_divergence = torch.stack((2.0 * x + 2.0 * y, y - 5.0), dim=-1)

    zero = equilibrium_residual(stress, coordinates, body_force=-exact_divergence)
    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-12, rtol=0.0)

    nonzero = equilibrium_residual(stress, coordinates)
    nonzero.square().mean().backward()
    assert scale.grad is not None
    assert torch.isfinite(scale.grad)
    assert scale.grad.abs() > 0.0


def test_operator_equilibrium_supports_cpu_higher_order_autograd() -> None:
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
    query, boundary, u_delta, x_max, y_max = _tiny_inputs()
    query = query.requires_grad_(True)
    output = model(query, boundary, u_delta, x_max, y_max)
    residual = equilibrium_residual(output.P_served, query)

    (residual.square().mean() + output.P_served.square().mean()).backward()

    assert model.p_head.output.weight.grad is not None
    assert torch.isfinite(model.p_head.output.weight.grad).all()
