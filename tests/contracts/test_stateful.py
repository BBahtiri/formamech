"""Tests for Feature F0: pure-stdlib stateful contracts and state bindings."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from piginot.contracts.expressions import DerivedExpression
from piginot.contracts.problem_spec import ProblemSpec
from piginot.contracts.stateful import (
    BoundTrajectoryState,
    IntegratorSpec,
    LoadSegment,
    StatefulProblemSpec,
    StatefulVerificationPolicy,
    StateSchema,
    StateVariableSpec,
    TimeGrid,
    evaluate_load_path,
)
from tests.platform_support import windows_safe_environment

ROOT = Path(__file__).resolve().parents[2]


def _sample_v1_spec() -> ProblemSpec:
    return ProblemSpec.from_bytes(
        (ROOT / "campaigns" / "dogbone_v1" / "problem_spec.json").read_bytes()
    )


def test_stateful_contracts_import_without_numerical_packages() -> None:
    script = textwrap.dedent(
        """
        import sys
        import formamech.contracts.stateful
        assert not {"torch", "numpy", "scipy", "skfem", "gmsh"} & set(sys.modules)
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=windows_safe_environment({"PYTHONPATH": str(ROOT / "src")}),
    )
    assert res.returncode == 0, res.stderr


def test_time_grid_and_piecewise_load_path_evaluation() -> None:
    grid = TimeGrid(t0=0.0, t1=3.0, dt_initial=0.5, dt_min=0.05, dt_max=0.5)
    assert grid.fixed_times() == (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)

    segments = (
        LoadSegment(
            start=0.0,
            end=1.0,
            kind="ramp",
            boundary_values={"ux": DerivedExpression.parse("u_max * s")},
        ),
        LoadSegment(
            start=1.0,
            end=2.0,
            kind="hold",
            boundary_values={"ux": DerivedExpression.parse("u_max")},
        ),
        LoadSegment(
            start=2.0,
            end=3.0,
            kind="ramp",
            boundary_values={"ux": DerivedExpression.parse("u_max * (1 - s)")},
        ),
    )
    params = {"u_max": 0.2}
    assert evaluate_load_path(segments, 0.0, params)["ux"] == pytest.approx(0.0)
    assert evaluate_load_path(segments, 0.5, params)["ux"] == pytest.approx(0.1)
    assert evaluate_load_path(segments, 1.5, params)["ux"] == pytest.approx(0.2)
    assert evaluate_load_path(segments, 2.5, params)["ux"] == pytest.approx(0.1)
    assert evaluate_load_path(segments, 3.0, params)["ux"] == pytest.approx(0.0)


def test_stateful_problem_spec_and_bound_state_refuse_mismatch() -> None:
    base = _sample_v1_spec()
    grid = TimeGrid(t0=0.0, t1=2.0, dt_initial=0.25, dt_min=0.05, dt_max=0.5)
    segments = (
        LoadSegment(0.0, 1.0, "ramp", {"ux": DerivedExpression.parse("0.1 * s")}),
        LoadSegment(1.0, 2.0, "hold", {"ux": DerivedExpression.parse("0.1")}),
    )
    schema = StateSchema(
        model_id="generalized-maxwell-finite-strain-v1",
        variables=(StateVariableSpec("viscous_strain", (2, 3, 3), "1", "zeros"),),
    )
    spec = StatefulProblemSpec(
        base=base,
        time_grid=grid,
        load_path=segments,
        material_state_schema=schema,
        integrator=IntegratorSpec(),
        verification=StatefulVerificationPolicy(),
    )
    assert len(spec.stateful_problem_hash) == 64
    assert len(spec.load_path_hash) == 64
    assert len(spec.material_schema_hash) == 64

    # Gap in load_path segments is rejected.
    with pytest.raises(ValueError, match="contiguous"):
        StatefulProblemSpec(
            base=base,
            time_grid=grid,
            load_path=(
                LoadSegment(
                    0.0, 0.8, "ramp", {"ux": DerivedExpression.parse("0.1 * s")}
                ),
                LoadSegment(1.0, 2.0, "hold", {"ux": DerivedExpression.parse("0.1")}),
            ),
            material_state_schema=schema,
        )

    bound = BoundTrajectoryState(
        problem_spec_hash=spec.stateful_problem_hash,
        mesh_hash="m" * 64,
        material_schema_hash=spec.material_schema_hash,
        load_path_hash=spec.load_path_hash,
        time_index=2,
        time_value=0.5,
        step_controller_version="stateful-guarded-stepper-v1",
        state_hash="blake2b256:" + "0" * 64,
    )
    ok, _ = bound.verify_compatible(
        problem_spec_hash=spec.stateful_problem_hash,
        mesh_hash="m" * 64,
        material_schema_hash=spec.material_schema_hash,
        load_path_hash=spec.load_path_hash,
        step_controller_version="stateful-guarded-stepper-v1",
    )
    assert ok is True

    # Changed mesh or load-path prefix is strictly refused.
    ok_mesh, msg_mesh = bound.verify_compatible(
        problem_spec_hash=spec.stateful_problem_hash,
        mesh_hash="x" * 64,
        material_schema_hash=spec.material_schema_hash,
        load_path_hash=spec.load_path_hash,
        step_controller_version="stateful-guarded-stepper-v1",
    )
    assert ok_mesh is False
    assert "mesh_hash" in msg_mesh
    assert len(bound.binding_hash) == 64

    # Exercise unary minus and division in safe load expression AST + validation guards.
    seg_div = LoadSegment(
        0.0, 2.0, "function", {"ux": DerivedExpression.parse("-u_max / 2 + t")}
    )
    assert seg_div.load_at(1.0, {"u_max": 0.4})["ux"] == pytest.approx(0.8)
    with pytest.raises(ValueError, match="t1 must strictly exceed t0"):
        TimeGrid(t0=1.0, t1=1.0, dt_initial=0.1, dt_min=0.05, dt_max=0.2)
    with pytest.raises(ValueError, match="dt_min <= dt_initial <= dt_max"):
        TimeGrid(t0=0.0, t1=1.0, dt_initial=0.5, dt_min=0.1, dt_max=0.2)
    with pytest.raises(ValueError, match="unique"):
        StateSchema(
            model_id="dup",
            variables=(
                StateVariableSpec("ev", (3, 3), "1"),
                StateVariableSpec("ev", (3, 3), "1"),
            ),
        )
    with pytest.raises(ValueError, match="unsupported integrator"):
        IntegratorSpec(method="explicit")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_det_f"):
        StatefulVerificationPolicy(min_det_f=0.0)
