from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from piginot.contracts import CanonicalQuantity, MaterialSpec, Unit
from piginot.operators import (
    BOUNDARY_ENERGY_ARCHITECTURE_ID,
    AffineDirichletLifting,
    AxisAlignedTensionLifting,
    BoundaryEnergyConfig,
    BoundaryEnergyEncoding,
    BoundaryEnergyOperator,
    EnergyOperatorConfig,
    QueryFrame,
    boundary_energy_module_digest,
    constitutive_in_plane_stress,
    embed_plane_stress_deformation,
)
from piginot.physics import energy_density, plane_stress_closure
from tests.materials.helpers import BUILTIN_VALUES, material

DTYPE = torch.float64


def _config() -> BoundaryEnergyConfig:
    return BoundaryEnergyConfig(
        energy=EnergyOperatorConfig(
            parameter_names=("load", "size"),
            hidden_dim=8,
            latent_token_count=3,
            parameter_hidden_dim=6,
            num_attention_heads=2,
            decoder_layers=2,
            positional_frequencies=2,
            output_init_std=0.01,
        ),
        boundary_feature_names=("fixed_x", "fixed_y", "traction"),
    )


def _model(seed: int = 3) -> BoundaryEnergyOperator:
    torch.manual_seed(seed)
    return BoundaryEnergyOperator(_config()).to(DTYPE)


def _source(batch: int = 1, dtype: torch.dtype = DTYPE) -> dict[str, Any]:
    xy = torch.tensor(
        [[[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]]], dtype=dtype
    ).repeat(batch, 1, 1)
    normals = xy - 1.0
    features = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]]],
        dtype=dtype,
    ).repeat(batch, 1, 1)
    return {
        "boundary_xy": xy,
        "boundary_normals": normals,
        "boundary_features": features,
        "weights": torch.tensor([[0.3, 0.2, 0.4, 0.1]], dtype=dtype).repeat(batch, 1),
        "mask": torch.ones((batch, 4), dtype=torch.bool),
        "frame": QueryFrame(
            torch.zeros((batch, 2), dtype=dtype),
            torch.full((batch, 2), 2.0, dtype=dtype),
        ),
    }


def _query(batch: int = 1, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.tensor(
        [[[0.3, 0.6], [0.7, 1.2], [1.3, 0.9], [1.5, 1.6]]], dtype=dtype
    ).repeat(batch, 1, 1)


def _arguments(model: BoundaryEnergyOperator, batch: int = 1) -> dict[str, Any]:
    dtype = model.geometry_queries.dtype
    return {
        "encoding": model.encode_boundary(**_source(batch, dtype)),
        "query_xy": _query(batch, dtype),
        "parameters": torch.tensor([[0.2, -0.1]], dtype=dtype).repeat(batch, 1),
        "lifting": AxisAlignedTensionLifting(
            torch.full((batch, 2), 2.0, dtype=dtype),
            torch.full((batch,), 0.1, dtype=dtype),
        ),
    }


def _material() -> MaterialSpec:
    return MaterialSpec(
        "boundary-operator-test",
        "compressible_neo_hookean",
        {
            "elastic_modulus": CanonicalQuantity(760.0, Unit.MPA),
            "poisson_ratio": CanonicalQuantity(0.23, Unit.ONE),
        },
    )


@pytest.mark.parametrize("names", [(), ("a", "a"), (" a",), ("",), (3,), ["a"]])
def test_config_rejects_invalid_boundary_names(names: Any) -> None:
    with pytest.raises(ValueError, match="boundary_feature_names"):
        replace(_config(), boundary_feature_names=names)


def test_exact_config_and_architecture_contract() -> None:
    with pytest.raises(TypeError, match="EnergyOperatorConfig"):
        replace(_config(), energy=object())
    with pytest.raises(TypeError, match="BoundaryEnergyConfig"):
        BoundaryEnergyOperator(object())  # type: ignore[arg-type]
    model = _model()
    assert model.architecture_id == BOUNDARY_ENERGY_ARCHITECTURE_ID
    assert model.config.boundary_feature_dim == 3
    assert not any(
        isinstance(layer, torch.nn.MultiheadAttention) for layer in model.modules()
    )


def test_encoding_is_invariant_to_permutation_split_measure_and_padding() -> None:
    model = _model()
    source = _source(batch=2)
    original = model.encode_boundary(**source)
    order = torch.tensor([2, 0, 3, 1])
    permuted = {
        key: value[:, order] if isinstance(value, torch.Tensor) else value
        for key, value in source.items()
    }
    duplicated = {
        key: value.repeat_interleave(2, dim=1)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in source.items()
    }
    duplicated["weights"] = duplicated["weights"] / 2.0
    padded = {
        key: torch.cat((value, value[:, :1]), dim=1)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in source.items()
    }
    padded["mask"] = torch.cat(
        (source["mask"], torch.zeros((2, 1), dtype=torch.bool)), dim=1
    )
    padded["boundary_xy"][:, -1] = 100.0
    padded["boundary_normals"][:, -1] = 0.0
    padded["boundary_features"][:, -1] = -999.0
    for variation in (permuted, duplicated, padded):
        encoded = model.encode_boundary(**variation)
        assert encoded.tokens.shape == (2, 3, 8)
        assert torch.allclose(encoded.tokens, original.tokens, atol=2e-15, rtol=0.0)
        args = _arguments(model, batch=2)
        assert torch.allclose(
            model.displacement(**{**args, "encoding": encoded}),
            model.displacement(**{**args, "encoding": original}),
            atol=2e-15,
            rtol=0.0,
        )


def test_zero_measure_points_are_excluded_and_scaling_measure_is_invariant() -> None:
    model = _model()
    source = _source()
    masked = {**source, "mask": torch.tensor([[True, True, True, False]])}
    zeroed = {**source, "weights": torch.tensor([[0.3, 0.2, 0.4, 0.0]], dtype=DTYPE)}
    a = model.encode_boundary(**masked)
    b = model.encode_boundary(**zeroed)
    scaled = model.encode_boundary(**{**zeroed, "weights": zeroed["weights"] * 31})
    assert torch.allclose(a.tokens, b.tokens, atol=1e-15, rtol=0.0)
    assert torch.allclose(b.tokens, scaled.tokens, atol=1e-15, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_subnormal_measures_preserve_relative_scale_and_split_invariance(
    dtype: torch.dtype,
) -> None:
    model = _model().to(dtype)
    source = _source(dtype=dtype)
    source["weights"] = torch.tensor([[0.125, 0.25, 0.5, 1.0]], dtype=dtype)
    expected = model.encode_boundary(**source)
    tiny = {**source, "weights": source["weights"] * torch.finfo(dtype).tiny / 4.0}
    actual = model.encode_boundary(**tiny)
    duplicated = {
        key: value.repeat_interleave(2, dim=1)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in tiny.items()
    }
    duplicated["weights"] = duplicated["weights"] / 2.0
    split = model.encode_boundary(**duplicated)
    tolerance = 2e-7 if dtype is torch.float32 else 2e-15
    assert torch.allclose(actual.tokens, expected.tokens, atol=tolerance, rtol=0.0)
    assert torch.allclose(split.tokens, expected.tokens, atol=tolerance, rtol=0.0)
    # An excluded huge weight must not determine the normalization scale.
    masked = {**tiny, "mask": torch.tensor([[True, True, True, False]])}
    reference = model.encode_boundary(**masked)
    large_padding = masked["weights"].clone()
    large_padding[:, -1] = torch.finfo(dtype).max
    padded = model.encode_boundary(**{**masked, "weights": large_padding})
    assert torch.equal(padded.tokens, reference.tokens)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("excluded_by", ["mask", "weights"])
def test_unusable_extreme_finite_padding_cannot_contaminate_tokens(
    dtype: torch.dtype, excluded_by: str
) -> None:
    model = _model().to(dtype)
    source = _source(dtype=dtype)
    source[excluded_by] = source[excluded_by].clone()
    source[excluded_by][:, -1] = False if excluded_by == "mask" else 0.0
    expected = model.encode_boundary(**source)
    extreme = dict(source)
    for name in ("boundary_xy", "boundary_normals", "boundary_features"):
        extreme[name] = source[name].clone()
        extreme[name][:, -1] = torch.finfo(dtype).max
    actual = model.encode_boundary(**extreme)
    assert torch.equal(actual.tokens, expected.tokens)


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("boundary_xy", None, TypeError),
        ("boundary_xy", torch.ones((1, 4, 3), dtype=DTYPE), ValueError),
        ("boundary_xy", torch.full((1, 4, 2), float("nan"), dtype=DTYPE), ValueError),
        ("boundary_normals", torch.zeros((1, 4, 2), dtype=DTYPE), ValueError),
        ("boundary_normals", torch.ones((2, 4, 2), dtype=DTYPE), ValueError),
        ("boundary_normals", torch.ones((1, 4, 2)), ValueError),
        ("boundary_features", torch.zeros((1, 4, 2), dtype=DTYPE), ValueError),
        ("boundary_features", torch.zeros((1, 3, 3), dtype=DTYPE), ValueError),
        ("boundary_features", torch.zeros((1, 4, 3)), ValueError),
        (
            "boundary_features",
            torch.full((1, 4, 3), float("inf"), dtype=DTYPE),
            ValueError,
        ),
        ("weights", None, TypeError),
        ("weights", torch.ones((1, 4, 1), dtype=DTYPE), ValueError),
        ("weights", torch.ones((2, 4), dtype=DTYPE), ValueError),
        ("weights", torch.ones((1, 4)), ValueError),
        ("weights", torch.full((1, 4), -1.0, dtype=DTYPE), ValueError),
        ("weights", torch.full((1, 4), float("nan"), dtype=DTYPE), ValueError),
        ("weights", torch.zeros((1, 4), dtype=DTYPE), ValueError),
        ("mask", None, TypeError),
        ("mask", torch.ones((1, 4), dtype=DTYPE), ValueError),
        ("mask", torch.ones((1, 4, 1), dtype=torch.bool), ValueError),
        ("mask", torch.ones((1, 4), dtype=torch.bool, device="meta"), ValueError),
        ("mask", torch.zeros((1, 4), dtype=torch.bool), ValueError),
        ("frame", None, TypeError),
        (
            "frame",
            QueryFrame(
                torch.zeros((2, 2), dtype=DTYPE), torch.ones((2, 2), dtype=DTYPE)
            ),
            ValueError,
        ),
        ("frame", QueryFrame(torch.zeros((1, 2)), torch.ones((1, 2))), ValueError),
    ],
)
def test_source_contract_fails_closed(
    key: str, value: Any, error: type[Exception]
) -> None:
    with pytest.raises(error):
        _model().encode_boundary(**{**_source(), key: value})


def test_every_batch_needs_positive_unmasked_measure() -> None:
    source = _source(2)
    source["weights"] = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=DTYPE
    )
    source["mask"] = torch.tensor(
        [[True, True, True, False], [True, True, True, False]]
    )
    with pytest.raises(ValueError, match="positive"):
        _model().encode_boundary(**source)


def test_authoritative_frame_is_preserved_instead_of_inferred_from_points() -> None:
    model = _model()
    source = _source()
    first = model.encode_boundary(**source)
    shifted = {
        **source,
        "boundary_xy": source["boundary_xy"] + 7.0,
        "frame": QueryFrame(source["frame"].origin + 7.0, source["frame"].extent),
    }
    second = model.encode_boundary(**shifted)
    assert first.frame is source["frame"]
    assert torch.allclose(first.tokens, second.tokens, atol=1e-15, rtol=0.0)
    changed_frame = model.encode_boundary(
        **{
            **source,
            "frame": QueryFrame(source["frame"].origin, source["frame"].extent * 2.0),
        }
    )
    assert not torch.allclose(first.tokens, changed_frame.tokens, atol=1e-10)


@pytest.mark.parametrize(
    "key", ["boundary_xy", "boundary_normals", "boundary_features", "weights"]
)
def test_each_boundary_input_can_change_the_prediction(key: str) -> None:
    model = _model()
    source = _source()
    changed = source[key].clone()
    if key == "boundary_normals":
        changed = -changed
    else:
        changed[:, 0] += 0.3
    args = _arguments(model)
    first = model.displacement(**args)
    second = model.displacement(
        **{**args, "encoding": model.encode_boundary(**{**source, key: changed})}
    )
    assert float((first - second).detach().abs().max()) > 1e-8


def test_parameter_conditioning_changes_the_field() -> None:
    model = _model()
    args = _arguments(model)
    first = model.displacement(**args)
    second = model.displacement(**{**args, "parameters": args["parameters"] + 0.5})
    assert float((first - second).detach().abs().max()) > 1e-8


def test_query_local_gradients_match_chunks_permutations_and_finite_differences() -> (
    None
):
    model = _model()
    args = _arguments(model)
    full, gradient = model.displacement_and_gradient(**args, create_graph=False)
    parts = [
        model.displacement_and_gradient(
            **{**args, "query_xy": args["query_xy"][:, part]}, create_graph=False
        )
        for part in (slice(None, 2), slice(2, None))
    ]
    assert torch.allclose(
        torch.cat([p[0] for p in parts], dim=1), full, atol=1e-15, rtol=0.0
    )
    assert torch.allclose(
        torch.cat([p[1] for p in parts], dim=1), gradient, atol=1e-15, rtol=0.0
    )
    order = torch.tensor([2, 0, 3, 1])
    reordered = model.displacement_and_gradient(
        **{**args, "query_xy": args["query_xy"][:, order]}, create_graph=False
    )
    assert torch.allclose(reordered[0], full[:, order], atol=1e-15, rtol=0.0)
    assert torch.allclose(reordered[1], gradient[:, order], atol=1e-15, rtol=0.0)
    for axis in range(2):
        shift = torch.zeros_like(args["query_xy"])
        shift[..., axis] = 1e-6
        plus = model.displacement(**{**args, "query_xy": args["query_xy"] + shift})
        minus = model.displacement(**{**args, "query_xy": args["query_xy"] - shift})
        assert torch.allclose(
            gradient[..., axis], (plus - minus) / 2e-6, atol=1e-9, rtol=1e-5
        )
    assert not full.requires_grad and not gradient.requires_grad


def test_lifting_preserves_exact_essential_conditions() -> None:
    model = _model()
    args = _arguments(model)
    query = torch.tensor([[[0.0, 0.7], [2.0, 1.4], [0.8, 0.0]]], dtype=DTYPE)
    actual = model.displacement(**{**args, "query_xy": query})
    assert actual[0, 0, 0] == 0.0
    assert actual[0, 1, 0] == 0.1
    assert actual[0, 2, 1] == 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_physical_energy_backpropagates_through_encoder_decoder_and_conditioning(
    dtype: torch.dtype,
) -> None:
    model = _model().to(dtype)
    source = _source(dtype=dtype)
    source["boundary_xy"] = source["boundary_xy"].requires_grad_(True)
    source["boundary_features"] = source["boundary_features"].requires_grad_(True)
    args = _arguments(model)
    args["encoding"] = model.encode_boundary(**source)
    args["parameters"] = args["parameters"].requires_grad_(True)
    args["query_xy"] = args["query_xy"].requires_grad_(True)
    _, gradient = model.displacement_and_gradient(**args, create_graph=True)
    deformation = torch.eye(2, dtype=dtype) + gradient
    material = _material()
    stretch = plane_stress_closure(deformation, material)
    energy = energy_density(
        embed_plane_stress_deformation(deformation, stretch), material
    ).mean()
    energy.backward()
    for input_tensor in (
        source["boundary_xy"],
        source["boundary_features"],
        args["parameters"],
    ):
        assert input_tensor.grad is not None
        assert bool(torch.isfinite(input_tensor.grad).all())
        assert bool((input_tensor.grad != 0.0).any())
    for name, parameter in model.named_parameters():
        if name in ("decoder.latent_bias.0.weight", "decoder.latent_bias.0.bias"):
            continue  # The existing decoder initializes its pooled-bias output to zero.
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        assert bool((parameter.grad != 0.0).any()), name


def test_served_stress_matches_registered_material_constitutive_gradient() -> None:
    model = _model()
    args = _arguments(model, batch=2)
    material = _material()
    output = model(**args, materials=(material, material))
    _, gradient = model.displacement_and_gradient(**args, create_graph=False)
    assert output.u.shape == (2, 4, 2)
    assert output.P.shape == (2, 4, 2, 2)
    assert torch.allclose(
        output.P, constitutive_in_plane_stress(gradient, material), atol=1e-12
    )
    with pytest.raises(ValueError, match="one MaterialSpec per sample"):
        model(**args, materials=())


@pytest.mark.parametrize("model_name", ["mooney_rivlin", "hgo", "goh"])
def test_forward_dispatches_registered_formula_materials(model_name: str) -> None:
    model = _model()
    args = _arguments(model)
    chosen = material(model_name, BUILTIN_VALUES[model_name])
    output = model(**args, materials=(chosen,))
    _, gradient = model.displacement_and_gradient(**args, create_graph=False)
    expected = constitutive_in_plane_stress(gradient[0], chosen)
    assert torch.isfinite(output.P).all()
    assert torch.allclose(output.P[0], expected, atol=1e-12, rtol=0.0)


def test_safetensors_roundtrip_and_semantic_digest(tmp_path: Path) -> None:
    model = _model()
    restored = _model(seed=19)
    digest = boundary_energy_module_digest(model)
    assert digest == boundary_energy_module_digest(restored)
    assert digest.startswith("blake2b256:")
    renamed = BoundaryEnergyOperator(
        replace(_config(), boundary_feature_names=("x", "y", "t"))
    ).to(DTYPE)
    assert boundary_energy_module_digest(renamed) != digest
    assert boundary_energy_module_digest(_model().float()) != digest
    path = tmp_path / "boundary.safetensors"
    save_file(
        model.state_dict(),
        str(path),
        metadata={"architecture_id": model.architecture_id, "module_digest": digest},
    )
    restored.load_state_dict(load_file(str(path)), strict=True)
    args = _arguments(model)
    assert torch.equal(restored.displacement(**args), model.displacement(**args))
    assert torch.equal(_arguments(restored)["encoding"].tokens, args["encoding"].tokens)
    with pytest.raises(TypeError):
        boundary_energy_module_digest(torch.nn.Linear(1, 1))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "tokens,frame,error",
    [
        (None, _source()["frame"], TypeError),
        (torch.tensor(0.0, dtype=DTYPE), _source()["frame"], ValueError),
        (torch.zeros((1, 8), dtype=DTYPE), _source()["frame"], ValueError),
        (
            torch.full((1, 3, 8), float("nan"), dtype=DTYPE),
            _source()["frame"],
            ValueError,
        ),
        (torch.zeros((1, 3, 8), dtype=DTYPE), object(), TypeError),
        (torch.zeros((2, 3, 8), dtype=DTYPE), _source()["frame"], ValueError),
        (torch.zeros((1, 3, 8)), _source()["frame"], ValueError),
    ],
)
def test_encoding_contract_fails_closed(
    tokens: Any, frame: Any, error: type[Exception]
) -> None:
    with pytest.raises(error):
        BoundaryEnergyEncoding(tokens, frame)


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("encoding", object(), TypeError),
        (
            "encoding",
            BoundaryEnergyEncoding(
                torch.zeros((1, 2, 8), dtype=DTYPE), _source()["frame"]
            ),
            ValueError,
        ),
        (
            "encoding",
            BoundaryEnergyEncoding(
                torch.zeros((1, 3, 6), dtype=DTYPE), _source()["frame"]
            ),
            ValueError,
        ),
        ("query_xy", torch.zeros((2, 4, 2), dtype=DTYPE), ValueError),
        ("query_xy", torch.zeros((1, 4, 2)), ValueError),
        ("parameters", torch.zeros((1, 3), dtype=DTYPE), ValueError),
        ("parameters", torch.zeros((2, 2), dtype=DTYPE), ValueError),
        ("parameters", torch.zeros((1, 2)), ValueError),
        ("parameters", torch.full((1, 2), float("nan"), dtype=DTYPE), ValueError),
        ("lifting", lambda points: points, TypeError),
        (
            "lifting",
            lambda points: AffineDirichletLifting(
                torch.zeros((1, 1, 2), dtype=DTYPE), torch.ones((1, 1, 2), dtype=DTYPE)
            ),
            ValueError,
        ),
        (
            "lifting",
            lambda points: AffineDirichletLifting(
                torch.zeros_like(points).float(), torch.ones_like(points).float()
            ),
            ValueError,
        ),
    ],
)
def test_prediction_contract_fails_closed(
    key: str, value: Any, error: type[Exception]
) -> None:
    model = _model()
    with pytest.raises(error):
        model.displacement(**{**_arguments(model), key: value})
