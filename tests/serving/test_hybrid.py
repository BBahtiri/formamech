"""S22 verified serving: modes, escalation, the solve ladder, the light install."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from campaigns import s21_certificate as s21
from campaigns import s22_recipes
from piginot.bundle import predict
from piginot.cli import parse_request
from piginot.serving import (
    ServingMode,
    ServingPath,
    default_request_id,
    operator_displacement,
    serve,
    solve_ladder,
)
from piginot.serving.recipe import FemRecipe
from tests.platform_support import windows_safe_environment

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN, CASE = "notched-v1", "candidate-000020"


@pytest.fixture(scope="module")
def setup() -> tuple[Any, FemRecipe, dict[str, Any]]:
    name = s21.runtime(CAMPAIGN).bundle
    bundle = s21._bundle(name)
    recipe = FemRecipe.load(
        ROOT / s22_recipes.recipe_path(name),
        bundle=name,
        manifest_sha256=bundle.manifest.manifest_sha256,
    )
    _, inputs = s22_recipes.case_inputs(CAMPAIGN, CASE)
    return bundle, recipe, inputs


def test_fast_is_the_unchanged_v010_prediction(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    bundle, recipe, inputs = setup
    served = serve(bundle, inputs, mode="fast", recipe=recipe)
    assert served.path is ServingPath.OPERATOR
    assert json.dumps(served.document(), sort_keys=True) == json.dumps(
        predict(bundle, inputs).document(), sort_keys=True
    )


def test_certified_and_exact_agree(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    bundle, recipe, inputs = setup
    certified = serve(bundle, inputs, mode="certified", recipe=recipe, corrected=True)
    assert certified.path is ServingPath.OPERATOR_CERTIFICATE
    assert (
        certified.certificate is not None and certified.certificate["verdict"] == "OK"
    )
    assert certified.exact is None and not certified.escalated
    plain = serve(bundle, inputs, mode="certified", recipe=recipe)
    assert plain.corrected_displacement is None
    exact = serve(bundle, inputs, mode=ServingMode.EXACT, recipe=recipe)
    assert exact.path is ServingPath.WARM_NEWTON
    assert exact.exact is not None and exact.exact_result_hash
    assert [a.path for a in exact.attempts] == [ServingPath.WARM_NEWTON]
    gap = np.abs(
        certified.corrected_displacement - np.asarray(exact.exact.displacement)
    )
    assert float(gap.max()) < 1e-3
    document = exact.document()
    assert document["path"] == "warm_newton" and document["exact"]["result_hash"]
    assert json.loads(json.dumps(document)) == document


def test_escalation_promotes_a_loose_certificate(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    bundle, recipe, inputs = setup
    tight = serve(
        bundle, inputs, mode="certified", recipe=recipe, escalate=True, tolerance=1e-12
    )
    assert tight.escalated and tight.path is ServingPath.WARM_NEWTON
    loose = serve(
        bundle, inputs, mode="certified", recipe=recipe, escalate=True, tolerance=1.0
    )
    assert not loose.escalated and loose.path is ServingPath.OPERATOR_CERTIFICATE


def test_refused_requests_skip_the_fem(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    bundle, recipe, _ = setup
    document = json.loads(
        (ROOT / "bundles/requests/notched-out-of-envelope.json").read_text()
    )
    inputs, _ = parse_request(document)
    served = serve(bundle, inputs, mode="exact", recipe=recipe)
    assert served.path is ServingPath.REFUSED
    assert served.exact is None and "refused" in served.reason


def test_a_recipe_is_required_beyond_fast(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    bundle, _, inputs = setup
    with pytest.raises(ValueError, match="recipe"):
        serve(bundle, inputs, mode="certified")


def test_operator_displacement_is_bit_identical(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    bundle, _, inputs = setup
    mesh = predict(bundle, inputs, confidence=False).mesh
    points = np.asarray(mesh.nodes, dtype=np.float64)[:200] * 0.999
    full = predict(bundle, inputs, points.tolist(), confidence=False).queries.u.numpy()
    assert np.array_equal(operator_displacement(bundle, inputs, mesh, points), full)


@dataclass
class _Attempt:
    success: bool
    result: Any = None
    failure_code: str | None = None
    message: str = ""
    result_hash: str | None = None


class _Result:
    convergence_history = (1, 2)


class _Backend:
    def __init__(self, outcomes: list[bool]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    def _next(self, label: str) -> _Attempt:
        self.calls.append(label)
        ok = self.outcomes.pop(0)
        return _Attempt(
            ok, _Result() if ok else None, None if ok else "NEWTON_NON_CONVERGENCE"
        )

    def solve_from(self, request: Any, start: Any, factors: Any) -> _Attempt:
        return self._next(f"from{tuple(factors)}")

    def solve_many(self, requests: Any) -> tuple[_Attempt]:
        return (self._next("cold"),)


@pytest.mark.parametrize(
    ("outcomes", "path", "calls"),
    [
        ([True], ServingPath.WARM_NEWTON, 1),
        ([False, True], ServingPath.CONTINUATION, 2),
        ([False, False, True], ServingPath.COLD_FEM, 3),
        ([False, False, False], None, 3),
    ],
)
def test_the_solve_ladder(
    outcomes: list[bool], path: ServingPath | None, calls: int
) -> None:
    backend = _Backend(list(outcomes))
    attempt, attempts = solve_ladder(backend, None, np.ones(4))  # type: ignore[arg-type]
    assert len(backend.calls) == calls
    assert (
        backend.calls[:2]
        == ["from(1.0,)", "from(0.5, 0.75, 1.0)"][: len(backend.calls[:2])]
    )
    if path is None:
        assert attempt is None and all(not a.success for a in attempts)
    else:
        assert attempts[-1].path is path and attempts[-1].success


def test_request_ids_follow_the_inputs(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    _, _, inputs = setup
    assert default_request_id(inputs) == default_request_id(dict(inputs))
    assert default_request_id(inputs).startswith("served-")


def test_the_light_install_serves_fast_and_explains_the_rest() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in ("scipy", "skfem", "meshio", "sympy"):
                    raise ImportError("blocked " + name)
        sys.meta_path.insert(0, Block())
        import json
        from pathlib import Path
        from piginot.bundle import load_bundle
        from piginot.cli import parse_request
        from piginot.serving import MissingExtraError, serve
        root = Path(sys.argv[1])
        bundle = load_bundle(root / "bundles/notched-v1-cp-freeze")
        inputs, _ = parse_request(json.loads((root / "bundles/requests/notched-in-domain.json").read_text()))
        assert serve(bundle, inputs, mode="fast").document()["verdict"] == "IN_DOMAIN"
        try:
            serve(bundle, inputs, mode="certified", recipe=object())
        except MissingExtraError as exc:
            print("explained:", "piginot[verify]" in str(exc))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
        env=windows_safe_environment(
            {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
        ),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "explained: True"
