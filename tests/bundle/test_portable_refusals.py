"""Finite weights can predict a deformation outside a material's domain."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save

from piginot.bundle import inference, load_bundle, predict
from piginot.bundle.serving import problem_energy_module_digest
from piginot.contracts import CanonicalQuantity, ProblemSpec, Unit
from piginot.contracts.energy_formula import BUILTIN_FORMULAS
from piginot.morphing.membership import PrimitiveBound
from piginot.physics.errors import (
    InadmissibleDeformationError,
    NonFiniteTensorError,
    PlaneStressClosureError,
)
from tests.bundle import helpers
from tests.bundle.test_portable_problem import (
    build,
    material_envelope,
    material_inputs,
    portable_problem,
    problem_spec,
)
from tests.bundle.test_portable_weights import fixed_model


@pytest.mark.parametrize("formula", [False, True])
def test_material_domain_failure_returns_an_explicit_portable_refusal(
    tmp_path: Path, formula: bool
) -> None:
    document = problem_spec(formula=False).document()
    document["parameters"] = [
        item for item in document["parameters"] if item["role"] != "material"
    ]
    for name, value, unit in (
        ("mu", 300.0, "MPa"),
        ("bulk", 1000.0, "MPa"),
        ("jm", 2.0, "1"),
    ):
        document["parameters"].append(
            {
                "name": name,
                "role": "material",
                "lo": value,
                "hi": value,
                "unit": unit,
                "distribution": "fixed",
            }
        )
    for item in document["parameters"]:
        if item["name"] == "nominal_axial_strain":
            item["hi"] = 11.0
    document["material"] = {
        **(
            {"energy": BUILTIN_FORMULAS["gent"].document()}
            if formula
            else {"model": "gent"}
        ),
        "parameters": {"shear_modulus": "mu", "bulk_modulus": "bulk", "jm": "jm"},
    }
    spec = ProblemSpec.from_document(document)
    problem = portable_problem(spec)
    model = fixed_model(torch.float32)
    payload = save(
        {name: torch.zeros_like(value) for name, value in model.state_dict().items()},
        metadata={
            "architecture_id": model.architecture_id,
            "architecture_module_digest": problem_energy_module_digest(
                model, problem.problem_hash
            ),
            "problem_hash": problem.problem_hash,
        },
    )
    envelope = replace(
        material_envelope(spec),
        load_bounds=(
            PrimitiveBound(
                "nominal_axial_strain",
                CanonicalQuantity(0.1, Unit("1")),
                CanonicalQuantity(11.0, Unit("1")),
            ),
        ),
    )
    inputs = {
        **material_inputs(spec),
        "nominal_axial_strain": CanonicalQuantity(10.0, Unit("1")),
    }
    bundle = load_bundle(
        build(
            tmp_path / "domain-failure",
            problem=problem,
            envelope=envelope,
            weights=payload,
        )
    )
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = predict(bundle, inputs, confidence=False)
    finally:
        torch.set_num_threads(previous_threads)
    assert not result.in_domain
    assert result.reason_codes == ("PREDICTION_MATERIAL_DOMAIN_FAILURE",)
    assert result.claim_level.value == "NOT_VERIFIED"
    assert result.confidence_status == "NOT_CALIBRATED"
    assert result.vertices is None and result.queries is None


def test_portable_prediction_does_not_hide_programming_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = load_bundle(build(tmp_path / "portable"))

    def broken(*args: object) -> object:
        raise RuntimeError("implementation error")

    monkeypatch.setattr(inference, "_fields", broken)
    with pytest.raises(RuntimeError, match="implementation error"):
        predict(bundle, helpers.request(), confidence=False)


@pytest.mark.parametrize("error", [InadmissibleDeformationError, NonFiniteTensorError])
def test_portable_computed_domain_failures_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    bundle = load_bundle(build(tmp_path / "portable"))

    def failed(*args: object) -> object:
        raise error("predicted field outside material domain")

    monkeypatch.setattr(inference, "_fields", failed)
    result = predict(bundle, helpers.request(), confidence=False)
    assert result.reason_codes == ("PREDICTION_MATERIAL_DOMAIN_FAILURE",)


def test_legacy_prediction_retains_its_physics_exception_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helpers.build(tmp_path / "legacy")
    bundle = load_bundle(tmp_path / "legacy")

    def failed(*args: object) -> object:
        raise PlaneStressClosureError("no root")

    monkeypatch.setattr(inference, "_fields", failed)
    with pytest.raises(PlaneStressClosureError, match="no root"):
        predict(bundle, helpers.request(), confidence=False)
