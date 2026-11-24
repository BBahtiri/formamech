"""Verification tests for Phase 1 (V4-1) external FEA solver oracles & CAD connectors."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from piginot.cad.connectors import (
    CadConnectorRegistry,
    CadConnectorUnavailableError,
    OnshapeRestConnector,
)
from piginot.contracts import MaterialSpec, Quantity, Unit
from piginot.fem.backend_3d import DirichletBC3D, FemRequest3D
from piginot.fem.external_solvers import (
    CalculixOracleBackend,
    DolfinxSubprocessOracleBackend,
    ExternalSolverConfig,
    OracleUnavailableError,
    PyAnsysOracleBackend,
    list_external_oracles,
    write_calculix_inp_3d,
)
from piginot.geometry.mesh3d import make_brick_tet4_mesh


def _write_mock_ccx(script_path: Path, *, mode: str = "valid") -> Path:
    """Create a deterministic Python mock ``ccx`` binary emitting genuine ``.frd`` output."""
    if mode == "sleep":
        code = "import time; time.sleep(5)\n"
    elif mode == "fail":
        code = "import sys; sys.stderr.write('solver failed'); sys.exit(2)\n"
    else:
        code = (
            "import sys\n"
            "from pathlib import Path\n"
            "job = sys.argv[sys.argv.index('-i') + 1]\n"
            "inp_lines = Path(f'{job}.inp').read_text(encoding='utf-8').splitlines()\n"
            "nodes = []\n"
            "in_nodes = False\n"
            "for ln in inp_lines:\n"
            "    if ln.startswith('*NODE,'): in_nodes = True; continue\n"
            "    if ln.startswith('*'): in_nodes = False; continue\n"
            "    if in_nodes and ln.strip():\n"
            "        parts = [p.strip() for p in ln.split(',')]\n"
            "        nodes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))\n"
            "out = ['    1C', ' -4  DISP        4    1']\n"
            "for nid, x, y, z in nodes:\n"
            "    ux, uy, uz = 0.01 * x, -0.003 * y, -0.003 * z\n"
            "    out.append(f' -1{nid:10d}{ux:12.5E}{uy:12.5E}{uz:12.5E}')\n"
            "out.append(' -3')\n"
            "out.append(' -4  STRESS      6    1')\n"
            "for nid, x, y, z in nodes:\n"
            "    out.append(f' -1{nid:10d}{10.0:12.5E}{0.0:12.5E}{0.0:12.5E}{0.0:12.5E}{0.0:12.5E}{0.0:12.5E}')\n"
            "out.append(' -3')\n"
            "Path(f'{job}.frd').write_text('\\n'.join(out) + '\\n', encoding='utf-8')\n"
        )
    script_path.write_text(code, encoding="utf-8")
    return script_path


def test_calculix_oracle_deck_generation_execution_timeout_and_refusals(
    tmp_path: Path,
) -> None:
    mesh = make_brick_tet4_mesh(10.0, 4.0, 4.0, nx=2, ny=1, nz=1)
    mat = MaterialSpec(
        name="oracle-nh",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": Quantity(1000.0, Unit.MPA).to_canonical(),
            "poisson_ratio": Quantity(0.30, Unit.ONE).to_canonical(),
        },
    )
    req = FemRequest3D(
        request_id="oracle-patch-1",
        mesh=mesh,
        material=mat,
        dirichlet_bcs=(
            DirichletBC3D("fix", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
            DirichletBC3D("pull", ("x_max",), ("x",), (0.10,)),
        ),
    )

    # 1) Direct deck serialization verification
    inp_file = write_calculix_inp_3d(req, tmp_path / "check.inp")
    deck_txt = inp_file.read_text(encoding="utf-8")
    assert "*ELEMENT, TYPE=C3D4" in deck_txt
    assert "*BOUNDARY" in deck_txt
    assert "*NODE FILE" in deck_txt

    # 2) Live subprocess execution against mock ccx emitting exact uniaxial field
    mock_ok = _write_mock_ccx(tmp_path / "mock_ccx.py", mode="valid")
    oracle = CalculixOracleBackend(
        ExternalSolverConfig(
            executable=str(mock_ok),
            command_prefix=(sys.executable,),
            timeout_s=30.0,
        )
    )
    res = oracle.solve(req)
    assert res.converged is True
    assert res.backend_name == "calculix-ccx"
    assert res.min_det_f > 0.99
    assert np.isclose(
        res.max_displacement_mm, np.linalg.norm([0.1, -0.012, -0.012]), rtol=1e-4
    )
    assert np.isclose(res.peak_von_mises_mpa, 10.0, atol=1e-4)

    # 3) Timeout refusal
    mock_slow = _write_mock_ccx(tmp_path / "mock_slow.py", mode="sleep")
    slow_oracle = CalculixOracleBackend(
        ExternalSolverConfig(
            executable=str(mock_slow),
            command_prefix=(sys.executable,),
            timeout_s=0.3,
        )
    )
    res_to = slow_oracle.solve(req)
    assert res_to.converged is False
    assert "timed out" in res_to.reason

    # 4) Non-zero exit refusal
    mock_fail = _write_mock_ccx(tmp_path / "mock_fail.py", mode="fail")
    fail_oracle = CalculixOracleBackend(
        ExternalSolverConfig(
            executable=str(mock_fail),
            command_prefix=(sys.executable,),
            timeout_s=10.0,
        )
    )
    res_fail = fail_oracle.solve(req)
    assert res_fail.converged is False
    assert "code 2" in res_fail.reason

    # 5) Missing binary raises OracleUnavailableError cleanly
    missing = CalculixOracleBackend(
        ExternalSolverConfig(executable="__no_such_ccx_bin__")
    )
    with pytest.raises(OracleUnavailableError):
        missing.solve(req)
    with pytest.raises(OracleUnavailableError):
        PyAnsysOracleBackend(
            ExternalSolverConfig(executable="__no_such_ansys__")
        ).solve(req)
    with pytest.raises(OracleUnavailableError):
        DolfinxSubprocessOracleBackend(
            ExternalSolverConfig(executable="__no_such_dolfinx__")
        ).solve(req)
    assert len(list_external_oracles()) == 4


def test_live_cad_connectors_opencascade_and_onshape_guards(tmp_path: Path) -> None:
    reg = CadConnectorRegistry()
    names = {c["name"] for c in reg.list_connectors()}
    assert {"opencascade-builtin", "cadquery", "freecad-cli", "onshape-rest"} <= names

    # OpenCASCADE live STEP generation
    occ = reg.get("opencascade-builtin")
    assert occ.is_available() is True
    step_out = tmp_path / "live_part.step"
    res = occ.generate_step("box", {"lx": 18.0, "ly": 7.0, "lz": 6.0}, step_out)
    assert step_out.is_file()
    assert len(res.step_sha256) == 64

    # Onshape fails closed when credentials are absent
    onshape_no_cred = OnshapeRestConnector(access_key=None, secret_key=None)
    with pytest.raises(CadConnectorUnavailableError, match="credentials"):
        onshape_no_cred.generate_step("doc/w/e", {"w": 10.0}, tmp_path / "fail.step")

    # Onshape succeeds with injected transport
    onshape_mock = OnshapeRestConnector(
        access_key="k",
        secret_key="s",
        http_fetch=lambda _url, _hdr, _p: b"ISO-10303-21;END-ISO-10303-21;",
    )
    o_res = onshape_mock.generate_step("part-1", {"r": 3.0}, tmp_path / "onshape.step")
    assert (tmp_path / "onshape.step").is_file()
    assert o_res.connector_name == "onshape-rest"
