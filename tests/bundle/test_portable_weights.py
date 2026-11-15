"""Portable weight precision is preserved independently of legacy loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save

from piginot.bundle import BundleIntegrityError, load_bundle
from piginot.bundle.serving import (
    CONSTITUTIVE_PROBLEM_PLANE_STRESS,
    EnergyFormServing,
    problem_energy_module_digest,
)
from piginot.operators import EnergyFormOperator
from tests.bundle import helpers
from tests.bundle.test_portable_problem import build, portable_problem


def fixed_model(dtype: torch.dtype) -> EnergyFormOperator:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = EnergyFormOperator(helpers.tiny_config()).to(dtype)
    if dtype == torch.float64:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.23456789e-9)
    return model.requires_grad_(False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_portable_weights_preserve_fields_at_their_stored_precision(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    problem = portable_problem()
    model = fixed_model(dtype)
    payload = save(
        {name: value.contiguous() for name, value in model.state_dict().items()},
        metadata={
            "architecture_id": model.architecture_id,
            "architecture_module_digest": problem_energy_module_digest(
                model, problem.problem_hash
            ),
            "problem_hash": problem.problem_hash,
        },
    )
    bundle = load_bundle(build(tmp_path / str(dtype), problem=problem, weights=payload))
    values = problem.spec.values(helpers.request())
    frame, lifting, material, _ = problem.request_context(values)
    parameters = torch.zeros((1, len(helpers.NAMES)), dtype=torch.float64)
    points = torch.tensor([[2.0, 1.0], [20.0, 4.0]], dtype=torch.float64)
    arguments = {
        "parameters": parameters,
        "frame": frame,
        "lifting": lifting,
        "material": material,
        "chunk": 512,
    }
    expected = EnergyFormServing(
        model.to(torch.float64).eval(),
        stress_authority=CONSTITUTIVE_PROBLEM_PLANE_STRESS,
    ).serve(points, **arguments)
    actual = bundle.serving.serve(points, **arguments)
    assert torch.equal(actual.u, expected.u)
    assert torch.equal(actual.gradient, expected.gradient)
    assert torch.equal(actual.stress, expected.stress)


@pytest.mark.parametrize("change", ["mixed", "integer", "schema", "nonfinite"])
def test_portable_weights_refuse_inconsistent_precision(
    tmp_path: Path, change: str
) -> None:
    problem = portable_problem()
    model = fixed_model(torch.float64)
    tensors = {name: value.contiguous() for name, value in model.state_dict().items()}
    if change == "mixed":
        name = next(iter(tensors))
        tensors[name] = tensors[name].to(torch.float32)
    elif change == "integer":
        tensors = {name: value.to(torch.int64) for name, value in tensors.items()}
    elif change == "nonfinite":
        name = next(iter(tensors))
        tensors[name] = torch.full_like(tensors[name], torch.nan)
    else:
        model = model.to(torch.float32)
    payload = save(
        tensors,
        metadata={
            "architecture_id": model.architecture_id,
            "architecture_module_digest": problem_energy_module_digest(
                model, problem.problem_hash
            ),
            "problem_hash": problem.problem_hash,
        },
    )
    with pytest.raises(
        BundleIntegrityError, match="finite" if change == "nonfinite" else "weights"
    ):
        build(tmp_path / change, problem=problem, weights=payload)
