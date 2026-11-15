"""Portable definitions refuse unbound, malformed or unsupported evidence."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import rfc8785

from piginot.bundle import BundleIntegrityError, predict
from piginot.bundle.serving import ARCHITECTURES, problem_energy_module_digest
from piginot.bundle.specs import ModelSpec, SpecDecodeError, decode_model_spec
from piginot.contracts import CanonicalQuantity, ProblemSpec, Unit
from piginot.morphing.membership import PrimitiveBound
from tests.bundle import helpers
from tests.bundle.test_portable_problem import (
    build,
    material_envelope,
    model_spec,
    portable_problem,
    problem_spec,
)
from tests.bundle.test_portable_weights import fixed_model


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("lifting_evidence",), []),
        (("lifting_evidence", "ignored"), True),
        (("lifting_evidence", "problem_hash"), "bad"),
        (("lifting_evidence", "kind"), "unknown"),
        (("lifting_evidence", "passed"), False),
        (("lifting_evidence", "thresholds"), {}),
        (("lifting_evidence", "parameter_vectors"), True),
        (("lifting_evidence", "refused_geometries"), -1),
        (("lifting_evidence", "refinements"), None),
        (("lifting_evidence", "boundary_residual"), []),
        (("lifting_evidence", "node_residual"), True),
        (("lifting_evidence", "mutant_miss"), 0),
        (("lifting_evidence", "conformity"), 1e-5),
        (("lifting_evidence", "kind"), "adf"),
        (("lifting_verification",), []),
        (("lifting_verification", "ignored"), True),
        (("lifting_verification", "boundary_refinement_levels"), "320,4800"),
    ],
)
def test_malformed_lifting_evidence_never_decodes(
    path: tuple[str, ...], value: object
) -> None:
    document = model_spec().document()
    target = document["problem"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(SpecDecodeError):
        decode_model_spec(rfc8785.dumps(document))


@pytest.mark.parametrize("field", ["spec", "lifting_verification", "lifting_evidence"])
def test_portable_python_api_requires_exact_contracts(field: str) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(portable_problem(), **{field: object()})


def test_portable_lifting_cannot_claim_full_range_with_refused_geometries() -> None:
    problem = portable_problem()
    evidence = dataclasses.replace(problem.lifting_evidence, refused_geometries=1)
    with pytest.raises(ValueError, match="full parameter range"):
        dataclasses.replace(
            problem,
            lifting_evidence=evidence,
            lifting_verification=evidence.verification(problem.spec),
        )


@pytest.mark.parametrize(
    "change",
    ["problem", "source", "units", "schema", "legacy", "ordering", "duplicate"],
)
def test_model_schema_rejects_ambiguous_or_incomplete_values(change: str) -> None:
    spec = model_spec()
    with pytest.raises(SpecDecodeError):
        if change == "problem":
            dataclasses.replace(spec, problem=object())
        elif change == "source":
            first, *rest = spec.conditioning
            dataclasses.replace(
                spec, conditioning=(dataclasses.replace(first, source="unknown"), *rest)
            )
        elif change == "units":
            dataclasses.replace(spec, units={"u": "m", "P_served": "MPa"})
        elif change == "schema":
            decode_model_spec(rfc8785.dumps({**spec.document(), "schema": "unknown"}))
        elif change == "ordering":
            dataclasses.replace(spec, conditioning=tuple(reversed(spec.conditioning)))
        elif change == "duplicate":
            decode_model_spec(rfc8785.dumps({**spec.document(), "frame": {}}))
        else:
            dataclasses.replace(helpers.tiny_model_spec(), confidence=None)


def test_problem_factory_requires_the_portable_contract() -> None:
    legacy = helpers.tiny_model_spec()
    with pytest.raises(SpecDecodeError):
        ModelSpec.from_problem(
            problem=object(),
            architecture_id=legacy.architecture_id,
            operator_config=legacy.operator_config,
            conditioning=legacy.conditioning,
            material_name=legacy.material_name,
            dtype=legacy.dtype,
            query_chunk=512,
            units=legacy.units,
        )


def test_unsupported_neumann_data_cannot_be_bundled() -> None:
    document = problem_spec().document()
    document["boundary_conditions"][-1] = {
        "name": "pressure",
        "region": "right_arc",
        "kind": "neumann",
        "components": ["x"],
        "values": [{"expression": "1", "unit": "MPa"}],
    }
    with pytest.raises(ValueError, match="Neumann"):
        portable_problem(ProblemSpec.from_document(document))


def test_envelope_cannot_add_inputs_absent_from_the_portable_problem(
    tmp_path: Path,
) -> None:
    envelope = material_envelope(problem_spec())
    extra = PrimitiveBound(
        "unbound", CanonicalQuantity(0.0, Unit.ONE), CanonicalQuantity(1.0, Unit.ONE)
    )
    envelope = dataclasses.replace(
        envelope, material_bounds=(*envelope.material_bounds, extra)
    )
    with pytest.raises(BundleIntegrityError, match="envelope inputs"):
        build(tmp_path / "extra-input", envelope=envelope)


def test_undefined_constraint_refuses_before_numerical_inference(
    tmp_path: Path,
) -> None:
    from piginot.bundle import load_bundle

    document = problem_spec().document()
    document["constraints"].append(
        {
            "name": "undefined_ratio",
            "expression": "L_total / (W_grip - W_grip)",
            "lo": 0,
            "hi": 1,
            "unit": "1",
        }
    )
    bundle = load_bundle(
        build(
            tmp_path / "undefined",
            problem=portable_problem(ProblemSpec.from_document(document)),
        )
    )
    result = predict(bundle, helpers.request())
    assert not result.in_domain
    assert result.reason_codes == ("PREDICTION_PROBLEM_OUT_OF_DOMAIN",)


def test_unregistered_portable_architecture_is_explicitly_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(ARCHITECTURES, "unsupported-shared", lambda *args: None)
    with pytest.raises(BundleIntegrityError, match="energy architecture"):
        build(
            tmp_path / "shared",
            model=dataclasses.replace(
                model_spec(), architecture_id="unsupported-shared"
            ),
        )


@pytest.mark.parametrize("change", ["model", "hash"])
def test_portable_digest_refuses_wrong_contracts(change: str) -> None:
    import torch

    with pytest.raises((TypeError, ValueError)):
        problem_energy_module_digest(
            object() if change == "model" else fixed_model(torch.float32), "invalid"
        )
