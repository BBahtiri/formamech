"""S26's amended reference solver and its resumable provenance."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from campaigns.dogbone_v1 import material_run
from campaigns.dogbone_v1 import s26_references as refs
from campaigns.dogbone_v1.s10_production import _decode_stored_result as decode_result
from campaigns.dogbone_v1.s26_hgo import RUN
from piginot.fem import FemRequest, ScikitFemBackend
from piginot.fem.linearization import FemLinearization
from tests.campaigns.test_dogbone_s25_mooney_rivlin import _plate_request

REVISION = "a" * 40


def _request(order: int = 2) -> FemRequest:
    request = RUN.request(_plate_request("plate-a"))
    return replace(request, config=replace(request.config, element_order=order))


@pytest.fixture(scope="module")
def solved() -> tuple[FemRequest, bytes, bytes]:
    request = _request()
    initial, final = refs.solve_reference(request)
    return request, initial, final


@pytest.mark.parametrize("order", [1, 2])
def test_initializer_preserves_every_field_except_material(order: int) -> None:
    request = _request(order)
    initial = refs.initializer_request(request)
    assert replace(initial, material=request.material) == request
    assert initial.material.model == "compressible_neo_hookean"
    assert initial.material.properties["elastic_modulus"].value == pytest.approx(760)
    assert initial.material.properties["poisson_ratio"].value == pytest.approx(0.23)
    with pytest.raises(ValueError, match="HGO"):
        refs.initializer_request(_plate_request("not-hgo"))


@pytest.mark.parametrize("order", [1, 2])
def test_warm_and_cold_hgo_solve_the_same_equilibrium(order: int) -> None:
    request = _request(order)
    _, payload = refs.solve_reference(request)
    warm = decode_result(request, payload)
    cold = ScikitFemBackend().solve_many((request,))[0]
    assert cold.success and cold.result is not None
    assert np.allclose(warm.displacement, cold.result.displacement, atol=1e-9, rtol=0)
    assert np.allclose(
        warm.first_piola_stress, cold.result.first_piola_stress, atol=1e-8, rtol=0
    )
    assert {entry.load_step for entry in warm.convergence_history} == {1}
    assert warm.convergence_history[-1].residual_norm <= request.config.solver_tolerance
    assert len(warm.convergence_history) <= request.config.max_newton_iterations


def test_solver_uses_nh_displacement_then_full_load_hgo(
    solved: tuple[FemRequest, bytes, bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    request, initial_payload, final_payload = solved
    initial_request = refs.initializer_request(request)
    initial = decode_result(initial_request, initial_payload)
    final = decode_result(request, final_payload)
    calls: list[str] = []

    def cold(items: tuple[FemRequest, ...]) -> tuple[SimpleNamespace, ...]:
        assert items == (initial_request,)
        calls.append("NH")
        return (SimpleNamespace(success=True, result=initial),)

    def warm(
        item: FemRequest, vector: NDArray[np.float64], factors: tuple[float, ...]
    ) -> SimpleNamespace:
        assert item is request and factors == (1.0,)
        expected = FemLinearization(request).vector(np.asarray(initial.displacement))
        assert np.array_equal(vector, expected)
        calls.append("HGO")
        return SimpleNamespace(success=True, result=final)

    backend = SimpleNamespace(solve_many=cold, solve_from=warm)
    monkeypatch.setattr(refs, "ScikitFemBackend", lambda: backend)
    assert refs.solve_reference(request) == (initial_payload, final_payload)
    assert calls == ["NH", "HGO"]


@pytest.mark.parametrize("stage", ["initializer", "HGO"])
def test_either_failed_stage_is_visible(
    solved: tuple[FemRequest, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    request, initial_payload, _ = solved
    initial = decode_result(refs.initializer_request(request), initial_payload)
    failed = SimpleNamespace(success=False, result=None, message="did not converge")

    def warm(*_: object) -> SimpleNamespace:
        assert stage == "HGO", "HGO must not run after an initializer failure"
        return failed

    backend = SimpleNamespace(
        solve_many=lambda _: (
            failed
            if stage == "initializer"
            else SimpleNamespace(success=True, result=initial),
        ),
        solve_from=warm,
    )
    monkeypatch.setattr(refs, "ScikitFemBackend", lambda: backend)
    with pytest.raises(RuntimeError, match=f"plate-a: {stage}.*did not converge"):
        refs.solve_reference(request)


def test_amendment_is_accepted_exact_and_requires_a_commit(tmp_path: Path) -> None:
    root = material_run.REPOSITORY
    assert refs.accepted_amendment(root, require_committed=False) == (
        refs.build_amendment(root)
    )
    path = tmp_path / refs.AMENDMENT_PATH
    path.parent.mkdir(parents=True)
    protocol = tmp_path / RUN.protocol_path
    protocol.write_bytes((root / RUN.protocol_path).read_bytes())
    with pytest.raises(RuntimeError, match="not written"):
        refs.accepted_amendment(tmp_path, require_committed=False)
    document = refs.build_amendment(tmp_path)
    path.write_text(json.dumps({**document, "status": "proposed"}))
    with pytest.raises(ValueError, match="differs"):
        refs.accepted_amendment(tmp_path, require_committed=False)
    path.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="committed"):
        refs.accepted_amendment(tmp_path)


def test_resume_binds_revision_amendment_and_both_results(
    tmp_path: Path,
    solved: tuple[FemRequest, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, initial, final = solved
    calls: list[str] = []

    def solve(item: FemRequest) -> tuple[bytes, bytes]:
        calls.append(item.request_id)
        return initial, final

    monkeypatch.setattr(refs, "solve_reference", solve)

    def run(revision: str = REVISION) -> dict[str, dict[str, str]]:
        return refs.solve_references(
            {"plate-a": request},
            tmp_path,
            root=material_run.REPOSITORY,
            implementation_revision=revision,
            require_committed=False,
        )

    index = run()
    assert calls == ["plate-a"]
    assert "initializer_result_hash" in index["plate-a"]
    assert run() == index
    assert calls == ["plate-a"]
    with pytest.raises(ValueError, match="binding"):
        run("b" * 40)
    monkeypatch.setattr(refs, "amendment_digest", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="binding"):
        run()


def test_old_unbound_results_are_refused_before_solving(tmp_path: Path) -> None:
    (tmp_path / "plate-a.json").write_text("{}")
    with pytest.raises(ValueError, match="fresh.*directory"):
        refs.solve_references(
            {"plate-a": _request()},
            tmp_path,
            root=material_run.REPOSITORY,
            implementation_revision=REVISION,
            require_committed=False,
        )


def test_an_interrupted_amended_run_resumes_only_pending_cases(
    tmp_path: Path,
    solved: tuple[FemRequest, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, initial, final = solved
    requests = {"plate-a": request, "plate-b": replace(request, request_id="plate-b")}
    calls: list[str] = []

    def fail_second(item: FemRequest) -> tuple[bytes, bytes]:
        calls.append(item.request_id)
        if item.request_id == "plate-b":
            raise RuntimeError("interrupted")
        return initial, final

    monkeypatch.setattr(refs, "solve_reference", fail_second)

    def run() -> dict[str, dict[str, str]]:
        return RUN.solve_reference_cases(
            requests,
            tmp_path,
            workers=1,
            progress=None,
            implementation_revision=REVISION,
            require_committed=False,
        )

    with pytest.raises(RuntimeError, match="interrupted"):
        run()
    assert (tmp_path / "plate-a.json").is_file()
    assert not (tmp_path / "plate-b.json").exists()

    def resume(item: FemRequest) -> tuple[bytes, bytes]:
        calls.append(item.request_id)
        assert item.request_id == "plate-b"
        return (
            initial.replace(b'"plate-a"', b'"plate-b"'),
            final.replace(b'"plate-a"', b'"plate-b"'),
        )

    monkeypatch.setattr(refs, "solve_reference", resume)
    assert sorted(run()) == ["plate-a", "plate-b"]
    assert calls == ["plate-a", "plate-b", "plate-b"]


def test_parallel_workers_store_both_solutions_and_report_progress(
    tmp_path: Path,
) -> None:
    request = _request()
    requests = {"plate-a": request, "plate-b": replace(request, request_id="plate-b")}
    messages: list[str] = []
    index = RUN.solve_reference_cases(
        requests,
        tmp_path,
        workers=2,
        progress=messages.append,
        implementation_revision=REVISION,
        require_committed=False,
    )
    assert sorted(index) == ["plate-a", "plate-b"]
    assert messages == ["phase=REFERENCE case=plate-a", "phase=REFERENCE case=plate-b"]
    for case_id in requests:
        assert (tmp_path / "initializers" / f"{case_id}.json").is_file()
    assert not list(tmp_path.rglob("*.partial"))


def test_loader_binds_initializer_and_final_hgo(
    tmp_path: Path,
    solved: tuple[FemRequest, bytes, bytes],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, initial, final = solved
    monkeypatch.setattr(refs, "solve_reference", lambda _: (initial, final))
    index = refs.solve_references(
        {"plate-a": request},
        tmp_path,
        root=material_run.REPOSITORY,
        implementation_revision=REVISION,
        require_committed=False,
    )
    monkeypatch.setattr(
        material_run.MaterialViability, "request", lambda self, _: request
    )
    document = RUN.references_document(index, REVISION)
    assert document["reference_amendment_sha256"] == refs.amendment_digest(
        material_run.REPOSITORY
    )
    loader = RUN.reference_loader(tmp_path, document)
    verified = SimpleNamespace(case=SimpleNamespace(case_id="plate-a"), request=request)
    assert loader(verified)[0] == request
    initializer_path = tmp_path / "initializers" / "plate-a.json"
    initializer_path.write_bytes(final)
    with pytest.raises(ValueError, match="differs from its index"):
        loader(verified)
