from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from piginot.operators import (
    ARCHITECTURE_ID,
    BoundaryConditionMode,
    DirichletData,
    GeometryEncoding,
    LiftingVerification,
    OperatorConfig,
    OperatorOutput,
    PiGinotOperator,
    load_operator_state_dict,
    module_digest,
    state_dict_schema,
)


def _config(**overrides: object) -> OperatorConfig:
    values: dict[str, object] = {
        "parameter_names": ("geometry", "material", "load"),
        "boundary_tag_names": ("fixed", "free"),
        "hidden_dim": 8,
        "latent_token_count": 2,
        "num_attention_heads": 2,
        "num_fourier_frequencies": 1,
        "encoder_self_attention_layers": 1,
        "decoder_cross_attention_layers": 1,
    }
    values.update(overrides)
    return OperatorConfig(**values)  # type: ignore[arg-type]


def _model() -> PiGinotOperator:
    torch.manual_seed(11030)
    return PiGinotOperator(_config())


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "boundary_xy": torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]),
        "boundary_tags": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]]
        ),
        "boundary_normals": torch.tensor(
            [[[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]]]
        ),
        "query_xy": torch.tensor([[[0.25, 0.25], [0.75, 0.75]]]),
        "parameters": torch.tensor([[0.2, 0.4, 0.6]]),
    }


@pytest.mark.parametrize(
    "changes",
    (
        {"parameter_names": ["geometry"]},
        {"parameter_names": ("",)},
        {"parameter_names": (" geometry",)},
        {"boundary_tag_names": ("fixed", "fixed")},
        {"latent_token_count": 0},
        {"num_fourier_frequencies": True},
        {"boundary_condition_mode": "soft"},
        {"soft_dirichlet_residual_tolerance": float("nan")},
    ),
)
def test_operator_config_fails_closed(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _config(**changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"verification_id": ""},
        {"parameter_range_hash": "xyz"},
        {"verification_hash": "A" * 64},
        {"boundary_refinement_levels": (1,)},
        {"boundary_refinement_levels": (2, 1)},
        {"boundary_refinement_levels": (1, True)},
        {"full_parameter_range_verified": False},
        {"max_boundary_residual": -1.0},
    ),
)
def test_lifting_verification_fails_closed(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "verification_id": "fixture",
        "parameter_range_hash": "1" * 64,
        "verification_hash": "2" * 64,
        "boundary_refinement_levels": (1, 2),
        "full_parameter_range_verified": True,
        "max_boundary_residual": 0.0,
    }
    values.update(changes)
    with pytest.raises(ValueError):
        LiftingVerification(**values)  # type: ignore[arg-type]


def test_soft_mode_rejects_misaligned_dirichlet_data() -> None:
    model = _model()
    data = DirichletData(
        torch.ones(1, 1, 2, dtype=torch.bool),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
    )

    with pytest.raises(ValueError, match="align"):
        model(**_inputs(), dirichlet=data)


def test_dirichlet_and_output_contracts_reject_invalid_tensors() -> None:
    with pytest.raises(ValueError, match="boolean"):
        DirichletData(torch.ones(1, 2, 2), torch.zeros(1, 2, 2), torch.zeros(1, 2, 2))
    with pytest.raises(ValueError, match="same shape"):
        DirichletData(
            torch.ones(1, 1, 2, dtype=torch.bool),
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2),
        )
    with pytest.raises(ValueError, match="non-negative"):
        DirichletData(
            torch.ones(1, 2, 2, dtype=torch.bool),
            torch.zeros(1, 2, 2),
            -torch.ones(1, 2, 2),
        )
    with pytest.raises(ValueError, match="P must have shape"):
        OperatorOutput(torch.zeros(1, 2, 2), torch.zeros(1, 2, 4))
    with pytest.raises(ValueError, match="share batch"):
        OperatorOutput(torch.zeros(1, 2, 2), torch.zeros(1, 3, 2, 2))


def test_geometry_encoding_and_model_constructor_reject_ambiguous_values() -> None:
    with pytest.raises(ValueError, match="batch dimension"):
        GeometryEncoding(torch.zeros(2, 2, 8), torch.zeros(1, 2), torch.ones(1, 2))
    with pytest.raises(ValueError, match="positive"):
        GeometryEncoding(
            torch.zeros(1, 2, 8), torch.zeros(1, 2), torch.tensor([[1.0, 0.0]])
        )
    with pytest.raises(TypeError, match="OperatorConfig"):
        PiGinotOperator(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LiftingVerification"):
        PiGinotOperator(_config(), lifting_verification=object())  # type: ignore[arg-type]


def test_soft_mode_refuses_irrelevant_lifting_certificate() -> None:
    verification = LiftingVerification("fixture", "1" * 64, "2" * 64, (1, 2), True, 0.0)
    with pytest.raises(ValueError, match="only valid"):
        PiGinotOperator(_config(), lifting_verification=verification)


def test_input_validation_checks_shape_dtype_device_and_coordinate_span() -> None:
    model = _model()
    inputs = _inputs()
    with pytest.raises(ValueError, match="align"):
        model(**{**inputs, "boundary_tags": torch.zeros(1, 3, 2)})
    with pytest.raises(ValueError, match="float32 or float64"):
        model(**{**inputs, "boundary_normals": torch.ones(1, 4, 2, dtype=torch.int64)})
    with pytest.raises(ValueError, match="span both"):
        model(**{**inputs, "boundary_xy": torch.zeros(1, 4, 2)})
    with pytest.raises(ValueError, match="batch dimension"):
        model(**{**inputs, "parameters": torch.zeros(2, 3)})
    with pytest.raises(ValueError, match="boolean"):
        model(**inputs, boundary_mask=torch.ones(1, 4))
    with pytest.raises(ValueError, match="align"):
        model(**inputs, boundary_weights=torch.ones(1, 3))


def test_dirichlet_residual_refuses_empty_component_selection() -> None:
    model = _model()
    output = model(**_inputs())
    data = DirichletData(
        torch.zeros_like(output.u, dtype=torch.bool),
        torch.zeros_like(output.u),
        torch.zeros_like(output.u),
    )
    with pytest.raises(ValueError, match="at least one"):
        model.dirichlet_residual(output, data)


def _write_header(path: Path, header: object) -> None:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded)


def _valid_header(model: PiGinotOperator) -> dict[str, Any]:
    dtype_codes = {torch.float32: "F32", torch.float64: "F64"}
    header: dict[str, Any] = {
        "__metadata__": {
            "architecture_id": ARCHITECTURE_ID,
            "module_digest": module_digest(model),
        }
    }
    for name, spec in state_dict_schema(model).items():
        header[name] = {
            "dtype": dtype_codes[spec.dtype],
            "shape": list(spec.shape),
            "data_offsets": [0, 0],
        }
    return header


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (b"short", "prefix"),
        ((2**32).to_bytes(8, "little"), "length"),
        ((3).to_bytes(8, "little") + b"xxx", "JSON"),
    ),
)
def test_loader_rejects_malformed_headers(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        load_operator_state_dict(path, model=_model())


def test_loader_rejects_missing_file_and_empty_schema(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_operator_state_dict(tmp_path / "missing.safetensors", model=_model())
    with pytest.raises(ValueError, match="must not be empty"):
        state_dict_schema({})
    with pytest.raises(ValueError, match="unsupported dtype"):
        state_dict_schema({"complex": torch.zeros(1, dtype=torch.complex64)})


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda header: header.pop("__metadata__"), "metadata"),
        (
            lambda header: header["__metadata__"].update({"module_digest": "wrong"}),
            "module_digest",
        ),
        (lambda header: header.update({"unexpected": {}}), "keys"),
    ),
)
def test_loader_rejects_header_contract_mutations(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    model = _model()
    header = _valid_header(model)
    mutation(header)
    path = tmp_path / "mutant.safetensors"
    _write_header(path, header)
    with pytest.raises(ValueError, match=message):
        load_operator_state_dict(path, model=model)


def test_loader_rechecks_materialized_tensor_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safetensors_torch = pytest.importorskip("safetensors.torch")
    model = _model()
    path = tmp_path / "valid.safetensors"
    safetensors_torch.save_file(
        dict(model.state_dict()),
        str(path),
        metadata={
            "architecture_id": ARCHITECTURE_ID,
            "module_digest": module_digest(model),
        },
    )
    monkeypatch.setattr(safetensors_torch, "load_file", lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="changed after preflight"):
        load_operator_state_dict(path, model=model)


def test_module_digest_requires_the_exact_operator_type() -> None:
    with pytest.raises(TypeError):
        module_digest(torch.nn.Linear(1, 1))  # type: ignore[arg-type]
    hard = replace(
        _config(), boundary_condition_mode=BoundaryConditionMode.HARD_LIFTING
    )
    assert hard.boundary_condition_mode is BoundaryConditionMode.HARD_LIFTING
