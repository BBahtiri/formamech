"""S23 bounds in serving: opt-in, unchanged outputs without them (ADR 0104)."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from campaigns import s21_certificate as s21
from campaigns import s22_recipes
from piginot.bounds import (
    IN_CALIBRATION,
    OUT_OF_CALIBRATION,
    CalibrationError,
    CalibrationRecord,
    MahalanobisGate,
    case_ids_sha256,
)
from piginot.bundle import predict
from piginot.cli import main
from piginot.contracts import canonical_json_bytes
from piginot.serving import ServingPath, serve
from piginot.serving.recipe import FemRecipe
from tests.platform_support import windows_safe_environment

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN, CASE = "notched-v1", "candidate-000020"
BUNDLE = ROOT / "bundles" / "notched-v1-cp-freeze"
AXES = ("width", "notch_radius", "notch_depth", "nominal_axial_strain")


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


def synthetic(
    bundle: Any, inputs: dict[str, Any], *, max_distance: float, shift: float = 0.0
) -> CalibrationRecord:
    """A calibration around the test case; not the S23 calibration."""

    mean = tuple(float(inputs[axis].value) + shift for axis in AXES)
    precision = tuple(
        tuple(1.0 if i == j else 0.0 for j in range(len(AXES)))
        for i in range(len(AXES))
    )
    return CalibrationRecord(
        bundle=BUNDLE.name,
        bundle_manifest_sha256=bundle.manifest.manifest_sha256,
        campaign_id=CAMPAIGN,
        protocol_sha256="0" * 64,
        alpha=0.1,
        n=64,
        k=59,
        calibration_case_ids_sha256=case_ids_sha256(["synthetic"]),
        bounds={
            "budget_total": {"p_relative_l2": 0.05, "absolute_peak_error": 0.06},
            "reference_terms": {"p_relative_l2": 0.02, "absolute_peak_error": 0.03},
        },
        gate=MahalanobisGate(AXES, mean, precision, max_distance),
    )


def test_fast_without_bounds_is_unchanged(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    bundle, _, inputs = setup
    served = serve(bundle, inputs)
    assert served.bounds is None
    assert served.document() == predict(bundle, inputs).document()


def test_fast_with_bounds_wraps_the_unchanged_prediction(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    bundle, _, inputs = setup
    served = serve(
        bundle, inputs, calibration=synthetic(bundle, inputs, max_distance=1)
    )
    document = served.document()
    assert document["mode"] == "fast" and document["path"] == "operator"
    assert document["prediction"] == predict(bundle, inputs).document()
    assert document["bounds"]["status"] == IN_CALIBRATION
    assert document["bounds"]["kind"] == "budget_total"
    assert document["bounds"]["upper_bounds"]["absolute_peak_error"] == 0.06
    assert document["bounds"]["distance_to_train"] == 0.0


def test_out_of_calibration_requests_are_served_without_bounds(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    bundle, _, inputs = setup
    calibration = synthetic(bundle, inputs, max_distance=0.5, shift=1.0)
    served = serve(bundle, inputs, calibration=calibration)
    assert served.prediction.in_domain
    assert served.bounds is not None
    assert served.bounds["status"] == OUT_OF_CALIBRATION
    assert served.bounds["upper_bounds"] is None


def test_refused_requests_carry_no_bounds(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    bundle, _, inputs = setup
    outside = dict(inputs)
    outside["nominal_axial_strain"] = type(inputs["nominal_axial_strain"])(
        0.5,
        inputs["nominal_axial_strain"].unit,
        inputs["nominal_axial_strain"].dimension,
    )
    served = serve(
        bundle, outside, calibration=synthetic(bundle, inputs, max_distance=9)
    )
    assert served.bounds is None
    assert served.document() == predict(bundle, outside).document()


def test_certified_bounds_the_reference_terms_only_when_asked(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    bundle, recipe, inputs = setup
    plain = serve(bundle, inputs, mode="certified", recipe=recipe)
    assert "bounds" not in plain.document()
    bounded = serve(
        bundle,
        inputs,
        mode="certified",
        recipe=recipe,
        calibration=synthetic(bundle, inputs, max_distance=1.0),
    )
    assert bounded.path is ServingPath.OPERATOR_CERTIFICATE
    document = bounded.document()
    assert document["bounds"]["kind"] == "reference_terms"
    assert document["bounds"]["upper_bounds"]["p_relative_l2"] == 0.02
    assert {k: v for k, v in document.items() if k != "bounds"} == plain.document()


def test_a_calibration_of_another_bundle_is_refused(
    setup: tuple[Any, FemRecipe, dict[str, Any]],
) -> None:
    bundle, _, inputs = setup
    calibration = synthetic(bundle, inputs, max_distance=1.0)
    other = CalibrationRecord.from_document(
        {**calibration.document(), "bundle_manifest_sha256": "f" * 64}
    )
    with pytest.raises(CalibrationError, match="another bundle"):
        serve(bundle, inputs, calibration=other)


def test_the_cli_attaches_bounds_from_a_sidecar(
    setup: tuple[Any, FemRecipe, dict[str, Any]], tmp_path: Path
) -> None:
    bundle, _, _ = setup
    request = ROOT / "bundles" / "requests" / "notched-in-domain.json"
    from piginot.cli import parse_request

    inputs, _ = parse_request(json.loads(request.read_text()))
    sidecar = tmp_path / "calibration.json"
    calibration = synthetic(bundle, inputs, max_distance=1.0)
    sidecar.write_bytes(canonical_json_bytes(calibration.document()))
    out = tmp_path / "out.json"
    common = ["predict", str(BUNDLE), "--request", str(request), "--output", str(out)]
    assert main([*common, "--bounds", "--calibration", str(sidecar)]) == 0
    document = json.loads(out.read_text())
    assert document["bounds"]["status"] == IN_CALIBRATION
    out.unlink()
    missing = tmp_path / "none.json"
    assert main([*common, "--bounds", "--calibration", str(missing)]) == 1
    assert not out.exists()


def test_bounds_need_no_fem_packages(
    setup: tuple[Any, FemRecipe, dict[str, Any]], tmp_path: Path
) -> None:
    from piginot.cli import parse_request

    bundle, _, _ = setup
    request = ROOT / "bundles" / "requests" / "notched-in-domain.json"
    inputs, _ = parse_request(json.loads(request.read_text()))
    sidecar = tmp_path / "calibration.json"
    sidecar.write_bytes(
        canonical_json_bytes(synthetic(bundle, inputs, max_distance=1.0).document())
    )
    script = textwrap.dedent(
        """
        import importlib.abc, json, sys
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in ("scipy", "skfem", "meshio", "sympy"):
                    raise ImportError("blocked " + name)
        sys.meta_path.insert(0, Block())
        from pathlib import Path
        from piginot.bounds import CalibrationRecord
        from piginot.bundle import load_bundle
        from piginot.cli import parse_request
        from piginot.serving import serve
        root, sidecar = Path(sys.argv[1]), Path(sys.argv[2])
        bundle = load_bundle(root / "bundles/notched-v1-cp-freeze")
        calibration = CalibrationRecord.load(
            sidecar,
            bundle="notched-v1-cp-freeze",
            manifest_sha256=bundle.manifest.manifest_sha256,
        )
        request = root / "bundles/requests/notched-in-domain.json"
        inputs, _ = parse_request(json.loads(request.read_text()))
        served = serve(bundle, inputs, calibration=calibration)
        print(served.document()["bounds"]["status"])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT), str(sidecar)],
        capture_output=True,
        text=True,
        check=False,
        env=windows_safe_environment(
            {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
        ),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == IN_CALIBRATION
