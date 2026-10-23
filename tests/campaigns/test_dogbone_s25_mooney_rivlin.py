"""The S25 Mooney–Rivlin viability glue (ADR 0107 decision 8); no training here."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from campaigns.dogbone_v1 import s25_mooney_rivlin as mr
from campaigns.dogbone_v1.family import compiled_family
from campaigns.dogbone_v1.s10_production import ProductionFemRunError
from campaigns.dogbone_v1.s10_requests import _load_public_sources
from campaigns.dogbone_v1.s11_contingency_source import (
    VerifiedS10Request,
    _rebuild_request,
)
from campaigns.dogbone_v1.s13c_data import case_energy, geometry_case
from campaigns.dogbone_v1.s13c_evaluate import _verify_binding
from campaigns.dogbone_v1.s13c_protocol import (
    build_dogbone_s13c_protocol,
    build_initial_operator,
    energy_config,
    load_s13c_normalization,
)
from piginot.contracts import (
    CanonicalQuantity,
    ExecutionProfile,
    ProblemSpec,
)
from piginot.contracts.units import Dimension, Unit
from piginot.fem import FemRequest
from piginot.physics import reference_modulus
from piginot.problem import admit_material, fem_request
from tests.platform_support import requires_pinned_initial_weights
from tests.problem.test_problem import PROFILE
from tests.problem_helpers import plate_document, plate_mesh

_PROFILE = ExecutionProfile(
    python_version="3.11.0",
    library_versions={"torch": "test"},
    blas_threads=1,
    device="cpu",
    dtype="float64",
    rng_algorithm="none",
    seed=0,
    container_digest=None,
)
_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _same(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if hasattr(left, "__dataclass_fields__"):
        return type(left) is type(right) and all(
            _same(getattr(left, f.name), getattr(right, f.name)) for f in fields(left)
        )
    return left == right


@pytest.fixture(scope="module")
def requests() -> tuple[VerifiedS10Request, ...]:
    protocol = build_dogbone_s13c_protocol()
    _, manifest = _load_public_sources()
    cases = {case.case_id: case for case in manifest.cases}
    family = compiled_family()
    chosen = (protocol.fit_case_ids[0], protocol.screen_case_ids[0])
    return tuple(
        VerifiedS10Request(
            cases[case_id], _rebuild_request(cases[case_id], _PROFILE, family)
        )
        for case_id in chosen
    )


def test_the_material_has_the_shipped_small_strain_moduli() -> None:
    material = mr.physics_material()
    assert material.model == "mooney_rivlin"
    assert set(material.properties) == {"c10", "c01", "bulk_modulus"}
    c10 = material.properties["c10"].value
    c01 = material.properties["c01"].value
    assert c10 == pytest.approx(4.0 * c01, rel=1e-15)
    shear = 2.0 * (c10 + c01)
    bulk = material.properties["bulk_modulus"].value
    assert shear == pytest.approx(760.0 / (2.0 * 1.23), rel=1e-14)
    assert bulk == pytest.approx(760.0 / (3.0 * (1.0 - 2.0 * 0.23)), rel=1e-14)
    assert reference_modulus(material) == pytest.approx(760.0, rel=1e-13)
    assert mr.fem_material().name == "dogbone-mooney-rivlin-reference-material"
    assert mr.fem_material().properties == material.properties


def test_the_viability_spec_is_admitted_and_only_the_material_differs() -> None:
    spec = mr.load_spec()
    assert admit_material(spec).passed
    base = json.loads(
        (mr.REPOSITORY / "campaigns/dogbone_v1/problem_spec.json").read_text()
    )
    document = json.loads((mr.REPOSITORY / mr.SPEC_PATH).read_text())
    differing = {key for key in document if document[key] != base[key]}
    assert differing == {"name", "parameters", "material", "fem"}
    geometry = [p for p in document["parameters"] if p["role"] != "material"]
    assert geometry == [p for p in base["parameters"] if p["role"] != "material"]
    fem = {k: v for k, v in document["fem"].items() if k != "material_name"}
    assert fem == {k: v for k, v in base["fem"].items() if k != "material_name"}


def test_the_spec_must_use_mooney_rivlin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mr, "SPEC_PATH", Path("campaigns/dogbone_v1/problem_spec.json"))
    with pytest.raises(ValueError, match="Mooney-Rivlin"):
        mr.load_spec.__wrapped__()


def test_the_request_and_case_change_only_the_material(
    requests: tuple[VerifiedS10Request, ...],
) -> None:
    normalization = load_s13c_normalization()
    for verified in requests:
        request = mr.mr_request(verified.request)
        assert request.material == mr.fem_material()
        assert replace(request, material=verified.request.material) == (
            verified.request
        )
        case = mr.mr_geometry_case(verified, normalization, dtype=torch.float64)
        base = geometry_case(verified, normalization, dtype=torch.float64)
        assert case.material == mr.physics_material()
        assert base.material.model == "compressible_neo_hookean"
        for item in fields(base):
            if item.name != "material":
                assert _same(getattr(case, item.name), getattr(base, item.name))


def test_the_case_trains_in_float32_with_finite_gradients(
    requests: tuple[VerifiedS10Request, ...],
) -> None:
    case = mr.mr_geometry_case(
        requests[0], load_s13c_normalization(), dtype=torch.float32
    )
    model = build_initial_operator()
    terms = case_energy(
        model, case, energy_config(), generator=torch.Generator().manual_seed(0)
    )
    assert bool(torch.isfinite(terms.total))
    terms.total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)


def test_materialization_requires_a_verified_source() -> None:
    with pytest.raises(TypeError, match="SelectiveS10Source"):
        mr.materialize_mr_training_data(
            object(),  # type: ignore[arg-type]
            build_dogbone_s13c_protocol(),
            load_s13c_normalization(),
        )


def test_the_committed_protocol_is_the_rebuild_and_accepted() -> None:
    committed = json.loads((mr.REPOSITORY / mr.PROTOCOL_PATH).read_text())
    assert committed == mr.build_protocol()
    assert committed["status"] == "accepted"
    viability = committed["viability"]
    assert viability["problem_spec_hash"] == mr.load_spec().problem_hash
    assert viability["optimizer_steps"] == 2400
    s13c = build_dogbone_s13c_protocol()
    assert viability["references"]["cases"] == list(s13c.screen_case_ids)
    assert len(viability["references"]["cases"]) == 64
    assert viability["closed_evidence"] == ["DEVELOPMENT", "HOLDOUT", "H1", "H2"]
    assert mr.accepted_protocol(require_committed=False) == committed


def test_a_changed_or_missing_protocol_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / mr.PROTOCOL_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "accepted"}), encoding="utf-8")
    monkeypatch.setattr(mr, "REPOSITORY", tmp_path)
    monkeypatch.setattr(mr, "build_protocol", lambda: {"rebuilt": True})
    with pytest.raises(ValueError, match="differs from its rebuild"):
        mr.accepted_protocol(require_committed=False)
    monkeypatch.setattr(mr, "build_protocol", lambda: {"status": "proposed"})
    path.write_text(json.dumps({"status": "proposed"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not accepted"):
        mr.accepted_protocol(require_committed=False)
    monkeypatch.setattr(mr, "_committed", lambda path: False)
    with pytest.raises(RuntimeError, match="committed before the run"):
        mr.accepted_protocol()
    path.unlink()
    with pytest.raises(RuntimeError, match="not written"):
        mr.accepted_protocol(require_committed=False)


class _Dirty:
    head = _REVISION
    dirty_worktree = True


def test_runs_require_the_protocol_and_a_clean_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(**_: object) -> None:
        raise RuntimeError("must be committed before the run")

    monkeypatch.setattr(mr, "accepted_protocol", refuse)
    with pytest.raises(RuntimeError, match="committed"):
        mr.run_mr_trial(s10_run_root=tmp_path, run_root=tmp_path / "run")
    with pytest.raises(RuntimeError, match="committed"):
        mr.write_references(s10_run_root=tmp_path, output=tmp_path / "references")
    monkeypatch.setattr(mr, "accepted_protocol", lambda **_: {})
    monkeypatch.setattr(mr, "_git_provenance", lambda: _Dirty())
    with pytest.raises(RuntimeError, match="clean revision"):
        mr.run_mr_trial(s10_run_root=tmp_path, run_root=tmp_path / "run")
    with pytest.raises(RuntimeError, match="clean revision"):
        mr.write_references(s10_run_root=tmp_path, output=tmp_path / "references")


def test_references_are_written_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mr, "accepted_protocol", lambda **_: {})
    for output in (Path("relative"), mr.REPOSITORY / "build" / "references"):
        with pytest.raises(ValueError, match="outside the repository"):
            mr.write_references(
                s10_run_root=tmp_path, output=output, require_committed=False
            )


@requires_pinned_initial_weights
def test_the_binding_binds_the_spec_and_the_evaluator_checks_it() -> None:
    protocol = build_dogbone_s13c_protocol()
    binding = mr.build_mr_binding(protocol, _REVISION)
    assert binding.problem_spec_hash == mr.load_spec().problem_hash
    _verify_binding(binding, protocol, binding.problem_spec_hash)
    with pytest.raises(ValueError, match="problem_spec_hash"):
        _verify_binding(binding, protocol)


# ---------------------------------------------------------------- references


def _plate_request(case_id: str) -> FemRequest:
    document = plate_document()
    document["parameters"] = [
        p for p in document["parameters"] if p["role"] != "material"
    ] + [
        {
            "name": name,
            "role": "material",
            "lo": value,
            "hi": value,
            "unit": "MPa",
            "distribution": "fixed",
        }
        for name, value in (("c10", 120.0), ("c01", 30.0), ("kappa", 470.0))
    ]
    document["material"] = {
        "model": "mooney_rivlin",
        "parameters": {"c10": "c10", "c01": "c01", "bulk_modulus": "kappa"},
    }
    spec = ProblemSpec.from_document(document)
    inputs = {
        "width": CanonicalQuantity(16.0, Unit.MM, Dimension.LENGTH),
        "height": CanonicalQuantity(6.0, Unit.MM, Dimension.LENGTH),
        "strain": CanonicalQuantity(0.05, Unit.ONE, Dimension.DIMENSIONLESS),
        "c10": CanonicalQuantity(120.0, Unit.MPA, Dimension.STRESS),
        "c01": CanonicalQuantity(30.0, Unit.MPA, Dimension.STRESS),
        "kappa": CanonicalQuantity(470.0, Unit.MPA, Dimension.STRESS),
    }
    return fem_request(spec, inputs, plate_mesh(), case_id, PROFILE)


@pytest.fixture(scope="module")
def plate() -> tuple[FemRequest, bytes]:
    request = _plate_request("plate-a")
    return request, mr.solve_reference(request)


def test_a_reference_solve_is_stored_like_s10(plate: tuple[FemRequest, bytes]) -> None:
    request, payload = plate
    entry = mr.reference_entry(request, payload)
    assert set(entry) == {"request_content_hash", "result_hash", "file_sha256"}
    document = json.loads(payload)
    assert document["domain"] == "content"
    assert document["payload"]["request_id"] == "plate-a"
    other = replace(request, request_id="plate-b")
    with pytest.raises(ProductionFemRunError):
        mr.reference_entry(other, payload)


def test_a_failed_reference_solve_is_an_error(
    plate: tuple[FemRequest, bytes],
) -> None:
    request, _ = plate
    unloadable = replace(
        request, config=replace(request.config, max_newton_iterations=1)
    )
    with pytest.raises(RuntimeError, match="plate-a"):
        mr.solve_reference(unloadable)


def test_solving_references_resumes_and_indexes(
    tmp_path: Path, plate: tuple[FemRequest, bytes]
) -> None:
    request, payload = plate
    second = replace(request, request_id="plate-b")
    requests = {"plate-a": request, "plate-b": second}
    solved: list[str] = []

    def solve(item: FemRequest) -> bytes:
        solved.append(item.request_id)
        return payload.replace(b'"plate-a"', f'"{item.request_id}"'.encode())

    messages: list[str] = []
    (tmp_path / "plate-a.json").write_bytes(payload)
    index = mr.solve_references(
        requests, tmp_path, solve=solve, progress=messages.append
    )
    assert solved == ["plate-b"]
    assert messages == ["phase=REFERENCE case=plate-b"]
    assert sorted(index) == ["plate-a", "plate-b"]
    assert not list(tmp_path.glob("*.partial"))
    again = mr.solve_references(requests, tmp_path, solve=solve)
    assert again == index and solved == ["plate-b"]

    document = mr.references_document(index, _REVISION)
    assert document["implementation_revision"] == _REVISION
    assert document["problem_spec_hash"] == mr.load_spec().problem_hash
    assert document["cases"] == index


def test_the_reference_loader_checks_every_file(
    tmp_path: Path,
    plate: tuple[FemRequest, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, payload = plate
    monkeypatch.setattr(mr, "mr_request", lambda item: item)
    (tmp_path / "plate-a.json").write_bytes(payload)
    index = {"plate-a": mr.reference_entry(request, payload)}
    document = mr.references_document(index, _REVISION)
    loader = mr.reference_loader(tmp_path, document)
    verified = SimpleNamespace(case=SimpleNamespace(case_id="plate-a"), request=request)
    loaded_request, result = loader(verified)
    assert loaded_request == request and result.request_id == "plate-a"
    with pytest.raises(ValueError, match="not this protocol's"):
        mr.reference_loader(tmp_path, {**document, "material_hash": "0" * 64})
    tampered = replace(result, total_strain_energy=result.total_strain_energy + 1.0)
    (tmp_path / "plate-a.json").write_bytes(mr.encode_result(tampered))
    with pytest.raises(ValueError, match="differs from its index"):
        loader(verified)


def test_parallel_solves_keep_the_case_order(
    tmp_path: Path, plate: tuple[FemRequest, bytes]
) -> None:
    request, _ = plate
    requests = {
        "plate-a": request,
        "plate-c": replace(request, request_id="plate-c"),
    }
    index = mr.solve_references(requests, tmp_path, workers=2)
    assert sorted(index) == ["plate-a", "plate-c"]
    for case_id in requests:
        stored = json.loads((tmp_path / f"{case_id}.json").read_bytes())
        assert stored["payload"]["request_id"] == case_id


# ------------------------------------------------------------------ evaluate


def _evaluation(**changes: Any) -> SimpleNamespace:
    metrics = SimpleNamespace(
        median_absolute_peak_error=0.01,
        p90_absolute_peak_error=0.02,
        median_signed_peak_error=-0.005,
        p_relative_l2=0.012,
        u_relative_l2=0.001,
        invalid_collocation_points=0,
    )
    fields = {
        "result": SimpleNamespace(metrics=metrics),
        "run_binding": SimpleNamespace(
            problem_spec_hash="f" * 64, implementation_revision=_REVISION
        ),
        "manifest": SimpleNamespace(
            run_hash="1" * 64,
            step=2400,
            manifest_hash="2" * 64,
            model_state_hash="3" * 64,
        ),
        "diagnostics": [
            SimpleNamespace(relative_energy_gap=g) for g in (3e-4, 1e-4, 2e-4)
        ],
        "minimum_collocation_determinant": 0.98,
        "failed_guardrails": (),
        "development_opened": False,
    }
    fields.update(changes)
    return SimpleNamespace(**fields)


def test_the_screen_record_applies_the_guardrails() -> None:
    protocol = mr.build_protocol()
    references = {"cases": {}}
    record = mr.screen_record(_evaluation(), protocol, references)
    assert record["passed"] and record["median_relative_energy_gap"] == 2e-4
    assert record["step"] == "S25" and record["evidence_role"] == "TRAIN"
    assert len(record["references_sha256"]) == 64
    failing = _evaluation(failed_guardrails=("u_relative_l2",))
    assert not mr.screen_record(failing, protocol, references)["passed"]
    opened = _evaluation(development_opened=True)
    assert not mr.screen_record(opened, protocol, references)["passed"]
    empty = mr.screen_record(_evaluation(diagnostics=[]), protocol, references)
    assert empty["median_relative_energy_gap"] is None


def test_evaluation_requires_references_for_every_screen_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = mr.build_protocol()
    monkeypatch.setattr(mr, "accepted_protocol", lambda **_: protocol)
    (tmp_path / mr.INDEX_NAME).write_text(json.dumps({"cases": {"x": {}}}))
    with pytest.raises(ValueError, match="do not cover"):
        mr.evaluate_mr_checkpoint(
            s10_run_root=tmp_path, checkpoint_path=tmp_path, references=tmp_path
        )
    cases = {case_id: {} for case_id in protocol["viability"]["references"]["cases"]}
    (tmp_path / mr.INDEX_NAME).write_text(json.dumps({"cases": cases}))
    committed = tmp_path / "repo" / mr.REFERENCES_PATH
    committed.parent.mkdir(parents=True)
    committed.write_text(json.dumps({"cases": {}}))
    monkeypatch.setattr(mr, "REPOSITORY", tmp_path / "repo")
    with pytest.raises(ValueError, match="committed index"):
        mr.evaluate_mr_checkpoint(
            s10_run_root=tmp_path, checkpoint_path=tmp_path, references=tmp_path
        )


def test_the_protocol_command_prints_the_rebuild(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert mr.main(["protocol"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == mr.build_protocol()
