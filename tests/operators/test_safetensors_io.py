from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

from piginot.operators import (
    ARCHITECTURE_ID,
    OperatorConfig,
    PiGinotOperator,
    load_operator_state_dict,
    module_digest,
    state_dict_schema,
)


class _Payload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return self.marker.write_text, ("DESERIALIZED",)


def _model() -> PiGinotOperator:
    torch.manual_seed(11020)
    config = OperatorConfig(
        parameter_names=("geometry", "material", "load"),
        boundary_tag_names=("fixed", "loaded", "free"),
        hidden_dim=16,
        latent_token_count=4,
        num_attention_heads=4,
        num_fourier_frequencies=2,
        encoder_self_attention_layers=1,
        decoder_cross_attention_layers=1,
    )
    return PiGinotOperator(config)


def _save(path: Path, model: PiGinotOperator, **changes: torch.Tensor) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    tensors = {
        name: value.detach().contiguous() for name, value in model.state_dict().items()
    }
    tensors.update(changes)
    safetensors_torch.save_file(
        tensors,
        str(path),
        metadata={
            "architecture_id": ARCHITECTURE_ID,
            "module_digest": module_digest(model),
            "format": "pt",
        },
    )


def test_valid_safetensors_round_trip_matches_exact_schema(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "fixture.safetensors"
    _save(path, model)

    loaded = load_operator_state_dict(path, model=model)

    assert state_dict_schema(model) == state_dict_schema(loaded)
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(loaded[name], expected)


@pytest.mark.parametrize("suffix", (".pkl", ".pt", ".pth", ".bin"))
def test_pickle_formats_are_refused_before_deserialization(
    tmp_path: Path, suffix: str
) -> None:
    marker = tmp_path / "executed.txt"
    path = tmp_path / f"model{suffix}"
    path.write_bytes(pickle.dumps(_Payload(marker)))

    with pytest.raises(ValueError, match="\\.safetensors"):
        load_operator_state_dict(path, model=_model())

    assert not marker.exists()


def test_pickle_renamed_to_safetensors_is_never_deserialized(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    path = tmp_path / "malicious.safetensors"
    path.write_bytes(pickle.dumps(_Payload(marker)))

    with pytest.raises(ValueError, match="header"):
        load_operator_state_dict(path, model=_model())

    assert not marker.exists()


def test_wrong_shape_is_rejected_from_header_before_tensor_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model()
    path = tmp_path / "wrong-shape.safetensors"
    name, value = next(iter(model.state_dict().items()))
    _save(path, model, **{name: torch.zeros(value.numel() + 1, dtype=value.dtype)})
    import safetensors.torch

    monkeypatch.setattr(
        safetensors.torch,
        "load_file",
        lambda *_args, **_kwargs: pytest.fail("tensor data was materialized"),
    )

    with pytest.raises(ValueError, match="shape"):
        load_operator_state_dict(path, model=model)


def test_wrong_dtype_is_rejected_from_header_before_tensor_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model()
    path = tmp_path / "wrong-dtype.safetensors"
    name, value = next(iter(model.state_dict().items()))
    _save(path, model, **{name: value.to(torch.float64)})
    import safetensors.torch

    monkeypatch.setattr(
        safetensors.torch,
        "load_file",
        lambda *_args, **_kwargs: pytest.fail("tensor data was materialized"),
    )

    with pytest.raises(ValueError, match="dtype"):
        load_operator_state_dict(path, model=model)


def test_missing_unexpected_and_wrong_identity_are_rejected(tmp_path: Path) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    model = _model()
    state = dict(model.state_dict())
    state.pop(next(iter(state)))
    state["unexpected"] = torch.zeros(1)
    path = tmp_path / "wrong-identity.safetensors"
    safetensors_torch.save_file(
        state,
        str(path),
        metadata={
            "architecture_id": "other",
            "module_digest": "blake2b256:" + "0" * 64,
        },
    )

    with pytest.raises(ValueError, match="architecture_id"):
        load_operator_state_dict(path, model=model)


def test_state_schema_accepts_a_loaded_mapping_and_rejects_other_values() -> None:
    model = _model()
    schema = state_dict_schema(model)

    assert schema == state_dict_schema(dict(model.state_dict()))
    with pytest.raises(TypeError):
        state_dict_schema({"bad": object()})  # type: ignore[dict-item]
