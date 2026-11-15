"""Portable ProblemSpec bundle candidates; fixed weights, no FEM or training."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import rfc8785
import torch
from safetensors.torch import save

from piginot.bundle import BundleIntegrityError, inspect, load_bundle, predict
from piginot.bundle.serving import (
    CONSTITUTIVE_PROBLEM_PLANE_STRESS,
    problem_energy_module_digest,
)
from piginot.bundle.specs import (
    ModelSpec,
    PortableProblem,
    SpecDecodeError,
    decode_model_spec,
)
from piginot.contracts import CanonicalQuantity, ProblemSpec
from piginot.morphing.membership import PrimitiveBound
from piginot.operators import EnergyFormOperator
from piginot.problem.verification import LiftingEvidence
from tests.bundle import helpers

ROOT = Path(__file__).resolve().parents[2]


def problem_spec(*, formula: bool = True) -> ProblemSpec:
    document = json.loads(
        (ROOT / "campaigns/dogbone_v1/problem_spec.json").read_bytes()
    )
    if formula:
        document["material"] = {
            "energy": {
                "energy": "E/(4*(1+nu))*(I1-3)-E/(2*(1+nu))*log(J)+E*nu/(2*(1+nu)*(1-2*nu))*log(J)^2",
                "parameters": {"E": "MPa", "nu": "1"},
                "positive": ["E"],
                "small_strain": {
                    "shear_modulus": "E/(2*(1+nu))",
                    "bulk_modulus": "E/(3*(1-2*nu))",
                },
            },
            "parameters": {"E": "elastic_modulus", "nu": "poisson_ratio"},
        }
    return ProblemSpec.from_document(document)


def portable_problem(
    spec: ProblemSpec | None = None, **changes: Any
) -> PortableProblem:
    spec = spec or problem_spec()
    evidence = LiftingEvidence(
        spec.problem_hash,
        spec.lifting.kind,
        1,
        0,
        (320, 4800),
        1e-14,
        None if spec.lifting.kind == "quarter" else 1e-14,
        None if spec.lifting.kind == "quarter" else 1e-14,
        {} if spec.lifting.kind == "quarter" else {320: 1e-14, 4800: 1e-14},
        1.0,
    )
    values = {
        "spec": spec,
        "material_admission_sha256": "a" * 64,
        "lifting_verification": evidence.verification(spec),
        "lifting_evidence": evidence,
    }
    return PortableProblem(**{**values, **changes})


def model_spec(problem: PortableProblem | None = None) -> ModelSpec:
    legacy = helpers.tiny_model_spec()
    return ModelSpec.from_problem(
        problem=problem or portable_problem(),
        architecture_id=legacy.architecture_id,
        operator_config=legacy.operator_config,
        conditioning=legacy.conditioning,
        material_name="portable-test-material",
        dtype=legacy.dtype,
        query_chunk=legacy.query_chunk,
        units=legacy.units,
    )


def weights(problem: PortableProblem, **changes: str) -> bytes:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = EnergyFormOperator(helpers.tiny_config()).to(torch.float32)
    metadata = {
        "architecture_id": model.architecture_id,
        "architecture_module_digest": problem_energy_module_digest(
            model, problem.problem_hash
        ),
        "problem_hash": problem.problem_hash,
    }
    return save(
        {name: value.contiguous() for name, value in model.state_dict().items()},
        metadata={**metadata, **changes},
    )


def build(
    directory: Path, *, problem: PortableProblem | None = None, **changes: Any
) -> Path:
    problem = problem or portable_problem()
    arguments = {
        "model": model_spec(problem),
        "weights": weights(problem),
        "claim_level": "NOT_VERIFIED",
        "verification": helpers.verification(
            claim_level="NOT_VERIFIED", verdict="NOT_VERIFIED"
        ),
    }
    helpers.build(directory, **{**arguments, **changes})
    return directory


def material_envelope(spec: ProblemSpec) -> Any:
    from campaigns.dogbone_v1.s14_envelope import dogbone_envelope

    return dataclasses.replace(
        dogbone_envelope(),
        material_bounds=tuple(
            PrimitiveBound(
                parameter.name,
                CanonicalQuantity(parameter.lo, parameter.unit),
                CanonicalQuantity(parameter.hi, parameter.unit),
            )
            for parameter in spec.parameters
            if parameter.role == "material"
        ),
    )


def material_inputs(spec: ProblemSpec) -> dict[str, object]:
    inputs = {
        name: quantity
        for name, quantity in helpers.request().items()
        if name in helpers.NAMES
    }
    return {
        **inputs,
        **{
            parameter.name: CanonicalQuantity(parameter.lo, parameter.unit)
            for parameter in spec.parameters
            if parameter.role == "material"
        },
    }


def fibre_problem() -> ProblemSpec:
    from piginot.contracts.energy_formula import BUILTIN_FORMULAS

    document = json.loads(
        (ROOT / "campaigns/dogbone_v1/problem_spec_hgo.json").read_bytes()
    )
    document["material"] = {
        "energy": BUILTIN_FORMULAS["hgo"].document(),
        "parameters": document["material"]["parameters"],
    }
    return ProblemSpec.from_document(document)


def test_legacy_model_bytes_are_unchanged() -> None:
    legacy = helpers.tiny_model_spec()
    assert (
        hashlib.sha256(legacy.to_bytes()).hexdigest()
        == "b367f5ca583154ff373b778c5afaae60ec073221c7aef2530c45218500a27c31"
    )
    assert "problem" not in legacy.document()


def test_portable_codec_uses_one_authoritative_problem() -> None:
    problem = portable_problem()
    spec = model_spec(problem)
    document = spec.document()
    assert document["schema"] == "piginot.bundle-model-v2"
    assert (
        "frame" not in document
        and "lifting" not in document
        and "material" not in document
    )
    assert document["problem"]["problem_hash"] == problem.spec.problem_hash
    assert spec.frame_origin is None and spec.lifting_displacement is None
    assert spec.material_model == problem.spec.material.model
    assert dict(spec.material_parameters) == dict(problem.spec.material.parameters)
    assert spec.confidence is None and "confidence" not in document
    assert set(spec.inputs) == {parameter.name for parameter in problem.spec.parameters}
    assert decode_model_spec(spec.to_bytes()) == spec


@pytest.mark.parametrize(
    "change", ["problem_hash", "admission", "range", "residual", "unknown", "unsafe"]
)
def test_portable_problem_refuses_unbound_or_unsafe_data(change: str) -> None:
    document = model_spec().document()
    problem = document["problem"]
    if change == "problem_hash":
        problem["problem_hash"] = "f" * 64
    elif change == "admission":
        problem["material_admission_sha256"] = "not-a-hash"
    elif change == "range":
        problem["lifting_verification"]["parameter_range_hash"] = "f" * 64
    elif change == "residual":
        problem["lifting_verification"]["max_boundary_residual"] = 1e-5
    elif change == "unknown":
        problem["ignored"] = True
    else:
        problem["spec"]["material"]["energy"]["energy"] = "__import__('os')"
    with pytest.raises(SpecDecodeError):
        decode_model_spec(rfc8785.dumps(document))


@pytest.mark.parametrize(
    "change", ["frame", "material", "confidence", "qoi", "boundary"]
)
def test_portable_model_refuses_contradictory_duplicate_definitions(
    change: str,
) -> None:
    spec = model_spec()
    values = {
        "frame": {"frame_origin": helpers.tiny_model_spec().frame_origin},
        "material": {"material_model": "compressible_neo_hookean"},
        "confidence": {"confidence": helpers.tiny_model_spec().confidence},
        "qoi": {"qois": helpers.tiny_model_spec().qois},
        "boundary": {"boundary_conditions": {}},
    }
    with pytest.raises(SpecDecodeError):
        dataclasses.replace(spec, **values[change])


def test_portable_candidate_cannot_inherit_a_verified_claim(tmp_path: Path) -> None:
    with pytest.raises(BundleIntegrityError, match="NOT_VERIFIED"):
        build(
            tmp_path / "claimed",
            claim_level=helpers.CLAIM,
            verification=helpers.verification(),
        )


def test_lifting_proof_cannot_transfer_between_problem_definitions() -> None:
    document = model_spec().document()
    document["problem"]["spec"]["name"] = "different-problem"
    changed = ProblemSpec.from_document(document["problem"]["spec"])
    document["problem"]["problem_hash"] = changed.problem_hash
    with pytest.raises(SpecDecodeError, match="lifting"):
        decode_model_spec(rfc8785.dumps(document))


@pytest.mark.parametrize("change", ["legacy", "problem", "digest", "authority"])
def test_portable_weights_bind_material_lifting_and_authority(
    tmp_path: Path, change: str
) -> None:
    problem = portable_problem()
    if change == "legacy":
        payload = helpers.tiny_weights()
    elif change == "problem":
        payload = weights(problem, problem_hash="f" * 64)
    elif change == "digest":
        payload = weights(problem, architecture_module_digest="blake2b256:" + "f" * 64)
    else:
        with pytest.raises(SpecDecodeError):
            dataclasses.replace(
                model_spec(problem),
                served_stress="constitutive-plane-stress-neo-hookean",
            )
        return
    with pytest.raises(BundleIntegrityError):
        build(tmp_path / change, problem=problem, weights=payload)


def test_user_material_works_after_registry_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = build(tmp_path / "portable")
    import piginot.materials.registry as registry

    monkeypatch.setattr(registry, "_USER", {})
    registry._bound.cache_clear()
    bundle = load_bundle(directory)
    prediction = predict(bundle, helpers.request(), [[5.0, 2.0]], confidence=False)
    assert prediction.in_domain
    assert prediction.stress_authority == CONSTITUTIVE_PROBLEM_PLANE_STRESS
    assert prediction.vertices is not None and prediction.queries is not None
    assert (
        prediction.constitutive_gap is not None and prediction.constitutive_gap < 1e-12
    )
    assert prediction.confidence_quantile is None
    assert prediction.vertices.confidence_bound is None
    assert prediction.document()["confidence_status"] == "NOT_CALIBRATED"


@pytest.mark.parametrize("fibres", [False, True])
def test_fresh_process_needs_no_fem_stack(tmp_path: Path, fibres: bool) -> None:
    spec = fibre_problem() if fibres else problem_spec()
    directory = build(
        tmp_path / "fresh",
        problem=portable_problem(spec),
        envelope=material_envelope(spec),
    )
    script = """
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'scipy','skfem','sympy'} or fullname.startswith('piginot.fem'):
            raise AssertionError('FEM import: '+fullname)
sys.meta_path.insert(0, Block())
from piginot.bundle import load_bundle, predict
from piginot.contracts import CanonicalQuantity, Unit
values=json.loads(sys.argv[2])
inputs={k:CanonicalQuantity(v['value'],Unit(v['unit'])) for k,v in values.items()}
result=predict(load_bundle(sys.argv[1]),inputs,confidence=False)
assert result.in_domain
assert result.document()['confidence_status']=='NOT_CALIBRATED'
print(result.constitutive_gap)
"""
    request = {
        name: {"value": value.value, "unit": value.unit.value}
        for name, value in material_inputs(spec).items()
    }
    result = subprocess.run(
        [sys.executable, "-c", script, str(directory), json.dumps(request)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_embedded_constraint_is_enforced_even_when_envelope_is_wider(
    tmp_path: Path,
) -> None:
    document = problem_spec().document()
    document["constraints"][0]["hi"] = 4.0
    problem = portable_problem(ProblemSpec.from_document(document))
    bundle = load_bundle(build(tmp_path / "narrow-domain", problem=problem))
    prediction = predict(
        bundle, helpers.request(R_fillet=18.0, W_grip=18.0, W_gauge=10.0)
    )
    assert not prediction.in_domain
    assert prediction.reason_codes == ("PREDICTION_PROBLEM_OUT_OF_DOMAIN",)
    assert prediction.confidence_status == "NOT_CALIBRATED"


def test_portable_inspection_and_default_prediction_have_no_transferred_confidence(
    tmp_path: Path,
) -> None:
    bundle = load_bundle(build(tmp_path / "no-curve"))
    prediction = predict(bundle, helpers.request())
    assert prediction.in_domain and prediction.vertices is not None
    assert prediction.confidence_quantile is None
    assert prediction.vertices.confidence_bound is None
    assert inspect(bundle)["model"]["confidence"] is None
    assert bundle.identities["problem_hash"] == bundle.model.problem.problem_hash


@pytest.mark.parametrize("change", ["mesh", "morph", "domain", "inputs", "units"])
def test_portable_bundle_rejects_inconsistent_geometry_and_domain(
    tmp_path: Path, change: str
) -> None:
    document = problem_spec().document()
    if change == "mesh":
        document["geometry"]["reference_mesh_id"] = "f" * 64
    elif change == "morph":
        document["geometry"]["morph_map_sha256"] = "f" * 64
    elif change == "domain":
        document["parameters"][0]["lo"] = 45.0
    elif change == "inputs":
        document["parameters"].append(
            {
                "name": "unused",
                "role": "material",
                "lo": 1,
                "hi": 1,
                "unit": "1",
                "distribution": "fixed",
            }
        )
    else:
        document["parameters"][4]["unit"] = "mm"
        document["boundary_conditions"][2]["values"] = [
            {"expression": "nominal_axial_strain", "unit": "mm"}
        ]
        document["lifting"]["displacement_scale"] = "nominal_axial_strain"
    with pytest.raises(BundleIntegrityError):
        build(
            tmp_path / change,
            problem=portable_problem(ProblemSpec.from_document(document)),
        )


@pytest.mark.parametrize("material", ["closed-form", "hgo"])
def test_portable_constitutive_dispatch_uses_its_declared_material(
    tmp_path: Path, material: str
) -> None:
    spec = (
        problem_spec(formula=False)
        if material == "closed-form"
        else ProblemSpec.from_document(
            json.loads(
                (ROOT / "campaigns/dogbone_v1/problem_spec_hgo.json").read_bytes()
            )
        )
    )
    envelope = material_envelope(spec)
    problem = portable_problem(spec)
    bundle = load_bundle(build(tmp_path / material, problem=problem, envelope=envelope))
    prediction = predict(bundle, material_inputs(spec))
    assert prediction.in_domain and prediction.vertices is not None
    assert (
        prediction.constitutive_gap is not None and prediction.constitutive_gap < 1e-12
    )
    assert torch.isfinite(prediction.vertices.sigma).all()


def test_adf_lifting_pieces_are_rebuilt_from_the_bundled_problem(
    tmp_path: Path,
) -> None:
    document = problem_spec().document()
    document["lifting"] = {
        "kind": "adf",
        "displacement_scale": "0.5 * nominal_axial_strain * L_total",
        "pieces": {
            "left_x_symmetry": [
                {"line": {"start": ["0", "0.5 * W_gauge"], "end": ["0", "0"]}}
            ],
            "bottom_y_symmetry": [
                {"line": {"start": ["0", "0"], "end": ["0.5 * L_total", "0"]}}
            ],
            "right_grip_displacement": [
                {
                    "line": {
                        "start": ["0.5 * L_total", "0"],
                        "end": ["0.5 * L_total", "0.5 * W_grip"],
                    }
                }
            ],
        },
    }
    problem = portable_problem(ProblemSpec.from_document(document))
    bundle = load_bundle(build(tmp_path / "adf", problem=problem))
    prediction = predict(bundle, helpers.request())
    assert prediction.in_domain and prediction.vertices is not None
    mesh = prediction.mesh
    assert mesh is not None
    for tag, component, expected in (
        ("left_symmetry", 0, 0.0),
        ("bottom", 1, 0.0),
        ("right_grip", 0, 0.5 * 54.0 * 0.15),
    ):
        indices = sorted(
            {
                vertex
                for facet in mesh.semantic_tags[tag]
                for vertex in mesh.boundary_facets[facet]
            }
        )
        assert torch.allclose(
            prediction.vertices.u[indices, component],
            torch.full((len(indices),), expected, dtype=torch.float64),
            atol=1e-12,
            rtol=0.0,
        )
