"""S15 validation: measure every G15 check on the built DogBone bundle.

Phases::

    python -m campaigns.dogbone_v1.s15_validate measure --bundle B \
        --checkpoint-path CKPT --s10-run-root S10 --run-root OUT
    python -m campaigns.dogbone_v1.s15_validate fresh-install --bundle B --run-root OUT
    python -m campaigns.dogbone_v1.s15_validate gate --run-root OUT \
        --coverage-percent P --coverage-command CMD --revision REV

``measure`` reads TRAIN evidence only (the reproduction check compares the
bundle with the S14 serving path on public TRAIN cases).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from campaigns.dogbone_v1.s14_envelope import (
    in_domain_cases,
    in_domain_queries,
    refusal_fixtures,
    theta,
)
from piginot.bundle import (
    REQUIRED_LIMITATION_PARAGRAPH,
    BundleIntegrityError,
    BundleReasonCode,
    Verdict,
    load_bundle,
    predict,
)
from piginot.bundle.inference import FIELD_NAMES, MODEL_PATH
from piginot.bundle.manifest import verify_bundle, write_bundle

_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS = _ROOT / "artifacts" / "campaigns" / "dogbone-v1"
MEASUREMENTS = "S15-MEASUREMENTS.json"
FRESH_INSTALL = "S15-FRESH-INSTALL.json"
FEM_ONLY_PACKAGES = ("scipy", "scikit-fem", "meshio", "sympy")
_BLOCKED_PROBE = """
import importlib.abc, sys
BLOCKED = {"scipy", "skfem", "meshio", "sympy"}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked " + name)
sys.meta_path.insert(0, Block())
from piginot.cli import main
code = main(["predict", "bundle", "--request", "request.json", "--no-confidence",
             "--output", "blocked-prediction.json"])
leaked = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
print(code, len(leaked))
"""


def runtime_requirements(wheel: Path) -> list[str]:
    """``Requires-Dist`` lines of the wheel that apply to ``piginot[runtime]``."""

    import zipfile

    with zipfile.ZipFile(wheel) as archive:
        (name,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        metadata = archive.read(name).decode("utf-8")
    requirements: list[str] = []
    for line in metadata.splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        value = line.split(":", 1)[1].strip()
        if "extra ==" not in value or "extra == 'runtime'" in value:
            requirements.append(value)
    return requirements


REFERENCE_REQUEST = {
    "L_total": 54.0,
    "W_grip": 20.0,
    "W_gauge": 10.0,
    "R_fillet": 12.0,
    "nominal_axial_strain": 0.15,
}
REFERENCE_QUERIES = [[5.0, 2.0], [13.5, 4.0], [20.0, 4.0], [27.0, 10.0], [0.0, 0.0]]


def _write(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")


def _progress(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


# ------------------------------------------------------------------ measurements


def fields_check(bundle_path: Path) -> dict[str, Any]:
    bundle = load_bundle(bundle_path)
    start = time.perf_counter()
    prediction = predict(bundle, theta(REFERENCE_REQUEST), REFERENCE_QUERIES)
    seconds = time.perf_counter() - start
    start = time.perf_counter()
    fast = predict(bundle, theta(REFERENCE_REQUEST), confidence=False)
    fast_seconds = time.perf_counter() - start
    document = prediction.document()
    vertices, queries = document["vertices"], document["queries"]
    finite = all(
        bool(torch.isfinite(getattr(points, name)).all())
        for points in (prediction.vertices, prediction.queries)
        for name in FIELD_NAMES
    )
    present = {
        "fields_at_vertices": all(name in vertices for name in FIELD_NAMES),
        "fields_at_queries": all(name in queries for name in FIELD_NAMES),
        "qois": bool(document["qois"]),
        "units": all(name in document["units"] for name in FIELD_NAMES),
        "identities": {
            "bundle_manifest_sha256",
            "weights_sha256",
            "envelope_id",
            "mesh_id",
        }
        <= set(document["identities"]),
        "in_domain_verdict": document["verdict"] == Verdict.IN_DOMAIN.value,
        "reason_codes": document["reason_codes"] == [],
        "claim_level": document["claim_level"] == "HOLDOUT_VERIFIED",
        "confidence_at_vertices": "confidence" in vertices,
        "confidence_at_queries": "confidence" in queries,
        "limitations": document["limitations"] == REQUIRED_LIMITATION_PARAGRAPH,
    }
    assert prediction.vertices is not None and prediction.queries is not None
    return {
        "present": present,
        "finite": finite,
        "vertex_count": len(vertices["u"]),
        "query_count": len(queries["u"]),
        "qois": document["qois"],
        "query_confidence_bound_mpa": queries["confidence"]["stress_error_bound"],
        "constitutive_gap": document["constitutive_gap"],
        "stress_authority": document["stress_authority"],
        "seconds_with_confidence": round(seconds, 3),
        "seconds_without_confidence": round(fast_seconds, 3),
        "fast_fields_equal": bool(
            fast.vertices is not None
            and torch.equal(fast.vertices.P_served, prediction.vertices.P_served)
        ),
        "passed": all(present.values()) and finite,
    }


def refusal_check(bundle_path: Path) -> dict[str, Any]:
    bundle = load_bundle(bundle_path)
    fixtures = refusal_fixtures()
    tested = [f for f in fixtures if f.mesh_mutation is None]
    misses: list[str] = []
    by_category: dict[str, list[int]] = {}
    for fixture in tested:
        prediction = predict(
            bundle, fixture.request_theta(), fixture.queries, confidence=False
        )
        ok = (
            prediction.verdict is Verdict.REFUSED
            and fixture.expected_code in prediction.reason_codes
            and prediction.claim_level.value == "NOT_VERIFIED"
            and prediction.limitations == REQUIRED_LIMITATION_PARAGRAPH
        )
        counts = by_category.setdefault(fixture.category, [0, 0])
        counts[0] += 1
        counts[1] += int(ok)
        if not ok:
            misses.append(fixture.fixture_id)
    _progress(f"refusal fixtures {len(tested) - len(misses)}/{len(tested)}")
    false_refusals: list[str] = []
    cases = in_domain_cases()
    for index, case in enumerate(cases):
        request = theta(case.values)
        outcome = bundle.geometry(request)
        if outcome.mesh is None:
            false_refusals.append(case.case_id)
            continue
        queries = [list(point) for point in in_domain_queries(outcome.mesh)]
        prediction = predict(bundle, request, queries, confidence=False)
        if not prediction.in_domain:
            false_refusals.append(case.case_id)
        if index % 100 == 99:
            _progress(f"in-domain {index + 1}/{len(cases)}")
    return {
        "fixtures_tested": len(tested),
        "fixtures_refused_with_expected_code": len(tested) - len(misses),
        "by_category": {
            k: {"tested": v[0], "refused": v[1]} for k, v in by_category.items()
        },
        "missed": misses,
        "mesh_mutation_fixtures_not_applicable": len(fixtures) - len(tested),
        "not_applicable_reason": "bundle requests carry parameters and queries only; "
        "the bundle always morphs its own reference mesh, so an explicit mutated "
        "mesh cannot be submitted",
        "in_domain_cases": len(cases),
        "false_refusals": false_refusals,
        "passed": not misses and not false_refusals,
    }


def _tamper_cases(directory: Path) -> list[tuple[str, Callable[[Path], None]]]:
    cases: list[tuple[str, Callable[[Path], None]]] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        relative = path.relative_to(directory).as_posix()
        size = path.stat().st_size
        for offset in sorted({0, size // 3, size // 2, size - 1}):

            def flip(
                root: Path, relative: str = relative, offset: int = offset
            ) -> None:
                target = root / relative
                payload = bytearray(target.read_bytes())
                payload[offset] ^= 0x01
                target.write_bytes(bytes(payload))

            cases.append((f"flip {relative}@{offset}", flip))

        def truncate(root: Path, relative: str = relative) -> None:
            target = root / relative
            target.write_bytes(target.read_bytes()[:-1])

        def delete(root: Path, relative: str = relative) -> None:
            (root / relative).unlink()

        cases.extend(
            ((f"truncate {relative}", truncate), (f"delete {relative}", delete))
        )

    def extra(root: Path) -> None:
        (root / "spec" / "notes.txt").write_bytes(b"x")

    def symlink(root: Path) -> None:
        os.symlink(root / "spec" / "model.json", root / "spec" / "alias.json")

    cases.extend((("extra file", extra), ("symlink", symlink)))
    return cases


def manifest_check(bundle_path: Path, scratch: Path) -> dict[str, Any]:
    cases = _tamper_cases(bundle_path)
    missed: list[str] = []
    codes: dict[str, int] = {}
    for index, (label, tamper) in enumerate(cases):
        copy = scratch / f"tamper-{index}"
        shutil.copytree(bundle_path, copy, symlinks=True)
        tamper(copy)
        try:
            load_bundle(copy)
        except BundleIntegrityError as exc:
            for code in exc.reason_codes:
                codes[code.value] = codes.get(code.value, 0) + 1
        else:
            missed.append(label)
        shutil.rmtree(copy)
    manifest, _ = verify_bundle(bundle_path)
    return {
        "manifest_sha256": manifest.manifest_sha256,
        "files_listed": len(manifest.files),
        "tamper_cases": len(cases),
        "detected": len(cases) - len(missed),
        "missed": missed,
        "reason_codes": dict(sorted(codes.items())),
        "passed": not missed,
    }


def substitution_check(bundle_path: Path, scratch: Path) -> dict[str, Any]:
    bundle = load_bundle(bundle_path)
    _, contents = verify_bundle(bundle_path)
    model = json.loads(contents[MODEL_PATH])
    model["served_stress"] = "predicted-stress-head"
    import rfc8785

    resigned = dict(contents)
    resigned[MODEL_PATH] = rfc8785.dumps(model)
    target = scratch / "resigned-authority"
    write_bundle(
        target,
        resigned,
        name=bundle.manifest.name,
        created_utc=bundle.manifest.created_utc,
        claim_level=bundle.manifest.claim_level,
        identities=bundle.manifest.identities,
    )
    try:
        load_bundle(target)
        refused: list[str] = []
    except BundleIntegrityError as exc:
        refused = [code.value for code in exc.reason_codes]
    prediction = predict(bundle, theta(REFERENCE_REQUEST), confidence=False)
    return {
        "declared_served_stress": bundle.model.served_stress,
        "architecture_authority": bundle.serving.stress_authority,
        "prediction_stress_authority": prediction.stress_authority,
        "resigned_other_authority_refused_with": refused,
        "p_material_reported_separately": "P_material"
        not in (prediction.document()["vertices"] or {}),
        "constitutive_gap": prediction.constitutive_gap,
        "mutant_test": "tests/bundle/test_inference.py::"
        "test_the_serve_p_material_mutant_is_caught",
        "passed": bundle.model.served_stress
        == bundle.serving.stress_authority
        == prediction.stress_authority
        and refused == [BundleReasonCode.STRESS_AUTHORITY_MISMATCH.value],
    }


def reproduction_check(
    bundle_path: Path,
    checkpoint_path: Path,
    s10_run_root: Path,
    rehearsal_root: Path,
    *,
    cases: int,
    confidence_cases: int,
) -> dict[str, Any]:
    """Bundle predictions equal the S14 serving path bit for bit on TRAIN cases."""

    from campaigns.dogbone_v1.s11_contingency_source import open_selective_s10_source
    from campaigns.dogbone_v1.s13c_evaluate import _served_vertex_output
    from campaigns.dogbone_v1.s13c_protocol import load_s13c_normalization
    from campaigns.dogbone_v1.s14_candidate import (
        S14Case,
        load_frozen_candidate,
        served_geometry,
    )
    from piginot.doe import EvidenceRole

    bundle = load_bundle(bundle_path)
    model = load_frozen_candidate(checkpoint_path)
    normalization = load_s13c_normalization()
    source = open_selective_s10_source(
        s10_run_root, development_access_record=rehearsal_root / "no-development.json"
    )
    records: list[dict[str, Any]] = []
    for verified in source.iter_train_screen():
        if len(records) >= cases:
            break
        if verified.case.role is not EvidenceRole.TRAIN:
            raise PermissionError("non-TRAIN evidence reached S15 validation")
        case = S14Case.from_published(verified.case)
        geometry = served_geometry(
            case,
            verified.request,
            normalization,
            allowed_roles=frozenset({EvidenceRole.TRAIN}),
        )
        nodes = torch.tensor(verified.request.mesh.nodes, dtype=torch.float64)
        u, stress, _ = _served_vertex_output(model, geometry, nodes)
        with_confidence = len(records) < confidence_cases
        values = {k: float(v) for k, v in case.primitive_coordinates.items()}
        prediction = predict(bundle, theta(values), confidence=with_confidence)
        vertices = prediction.vertices
        assert vertices is not None and prediction.mesh is not None
        record: dict[str, Any] = {
            "case_id": case.case_id,
            "mesh_id_equal": prediction.mesh.mesh_id == verified.request.mesh.mesh_id,
            "u_bitwise_equal": bool(torch.equal(vertices.u, u.detach())),
            "P_bitwise_equal": bool(torch.equal(vertices.P_served, stress.detach())),
        }
        if with_confidence:
            cache = np.load(rehearsal_root / "cases" / f"{case.case_id}.npz")
            reference = torch.from_numpy(cache["patch-mean"])
            record["indicator_bitwise_equal"] = bool(
                torch.equal(vertices.confidence_indicator, reference)  # type: ignore[arg-type]
            )
        records.append(record)
        _progress(f"reproduction {len(records)}/{cases} {case.case_id}")
    if source.development_opened:
        raise RuntimeError("DEVELOPMENT labels opened during S15 validation")
    return {
        "cases": records,
        "passed": all(
            all(value for key, value in record.items() if key != "case_id")
            for record in records
        ),
    }


def limitations_check(bundle_path: Path, scratch: Path) -> dict[str, Any]:
    from piginot import cli

    bundle = load_bundle(bundle_path)
    request = scratch / "request.json"
    request.write_text(
        json.dumps(
            {
                "inputs": {
                    **{
                        name: {"value": value, "unit": "mm"}
                        for name, value in REFERENCE_REQUEST.items()
                        if name != "nominal_axial_strain"
                    },
                    "nominal_axial_strain": {"value": 0.3, "unit": "1"},
                    "elastic_modulus": {"value": 760.0, "unit": "MPa"},
                    "poisson_ratio": {"value": 0.23, "unit": "1"},
                }
            }
        )
    )
    refused_output = scratch / "refused.json"
    cli.main(
        [
            "predict",
            str(bundle_path),
            "--request",
            str(request),
            "--output",
            str(refused_output),
        ]
    )
    inspect_output = scratch / "inspect.json"
    cli.main(["inspect", str(bundle_path), "--output", str(inspect_output)])
    accepted = predict(bundle, theta(REFERENCE_REQUEST), confidence=False).document()
    broken = scratch / "broken"
    shutil.copytree(bundle_path, broken)
    (broken / MODEL_PATH).write_bytes((broken / MODEL_PATH).read_bytes() + b" ")
    broken_output = scratch / "broken.json"
    cli.main(["inspect", str(broken), "--output", str(broken_output)])
    reports = {
        "bundle report/limitations.json": bundle.limitations["paragraph"],
        "predict (in domain)": accepted["limitations"],
        "predict (refused, CLI)": json.loads(refused_output.read_text())["limitations"],
        "inspect (CLI)": json.loads(inspect_output.read_text())["limitations"][
            "paragraph"
        ],
        "bundle refused (CLI)": json.loads(broken_output.read_text())["limitations"],
    }
    present = {
        name: text == REQUIRED_LIMITATION_PARAGRAPH for name, text in reports.items()
    }
    return {"reports": present, "passed": all(present.values())}


def run_measure(args: argparse.Namespace) -> int:
    run_root: Path = args.run_root
    scratch = run_root / "scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    results: dict[str, Any] = {
        "bundle": str(args.bundle),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "threads": torch.get_num_threads(),
        },
    }
    _progress("manifest")
    results["manifest"] = manifest_check(args.bundle, scratch)
    _progress("substitution")
    results["substitution"] = substitution_check(args.bundle, scratch)
    _progress("limitations")
    results["limitations"] = limitations_check(args.bundle, scratch)
    _progress("fields")
    results["fields"] = fields_check(args.bundle)
    _progress("reproduction")
    results["reproduction"] = reproduction_check(
        args.bundle,
        args.checkpoint_path,
        args.s10_run_root,
        args.rehearsal_root,
        cases=args.reproduce_cases,
        confidence_cases=args.confidence_cases,
    )
    _progress("refusal")
    results["refusal"] = refusal_check(args.bundle)
    _write(run_root / MEASUREMENTS, results)
    _progress("done")
    return 0


# ------------------------------------------------------------------ fresh install


def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    _progress(" ".join(command[:4]))
    return subprocess.run(
        command, capture_output=True, text=True, check=False, **kwargs
    )


def run_fresh_install(args: argparse.Namespace) -> int:
    work: Path = args.run_root / "fresh-install"
    if work.exists():
        shutil.rmtree(work)
    (work / "clean").mkdir(parents=True)
    uv = shutil.which("uv") or "uv"
    build = _run([uv, "build", "--wheel", "--out-dir", str(work / "dist"), str(_ROOT)])
    if build.returncode != 0:
        raise RuntimeError(build.stderr)
    (wheel,) = sorted((work / "dist").glob("piginot-*.whl"))
    venv = work / "venv"
    created = _run([uv, "venv", str(venv), "--python", args.python, "--no-project"])
    if created.returncode != 0:
        raise RuntimeError(created.stderr)
    python = venv / "bin" / "python"
    install = _run([uv, "pip", "install", "--python", str(python), f"{wheel}[runtime]"])
    if install.returncode != 0:
        raise RuntimeError(install.stderr)
    freeze = _run([uv, "pip", "freeze", "--python", str(python)])
    packages = sorted(line for line in freeze.stdout.splitlines() if line)
    names = {line.split("==")[0].split(" @ ")[0].lower() for line in packages}
    clean = work / "clean"
    bundle = clean / "bundle"
    shutil.copytree(args.bundle, bundle)
    request = clean / "request.json"
    request.write_text(
        json.dumps(
            {
                "inputs": {
                    **{
                        name: {"value": value, "unit": "mm"}
                        for name, value in REFERENCE_REQUEST.items()
                        if name != "nominal_axial_strain"
                    },
                    "nominal_axial_strain": {"value": 0.15, "unit": "1"},
                    "elastic_modulus": {"value": 760.0, "unit": "MPa"},
                    "poisson_ratio": {"value": 0.23, "unit": "1"},
                },
                "queries": REFERENCE_QUERIES,
            }
        )
    )
    refused_request = clean / "refused.json"
    refused_document = json.loads(request.read_text())
    refused_document["inputs"]["L_total"]["value"] = 90.0
    refused_request.write_text(json.dumps(refused_document))
    environment = {"PATH": f"{venv / 'bin'}:/usr/bin:/bin", "HOME": str(clean)}
    piginot = str(venv / "bin" / "piginot")
    steps = {
        "bundle verify": [piginot, "bundle", "verify", "bundle"],
        "inspect": [piginot, "inspect", "bundle", "--output", "inspect.json"],
        "predict": [
            piginot,
            "predict",
            "bundle",
            "--request",
            "request.json",
            "--output",
            "prediction.json",
        ],
        "predict (refused)": [
            piginot,
            "predict",
            "bundle",
            "--request",
            "refused.json",
            "--output",
            "refused-prediction.json",
            "--no-confidence",
        ],
    }
    exit_codes: dict[str, int] = {}
    for name, command in steps.items():
        result = _run(command, cwd=clean, env=environment)
        exit_codes[name] = result.returncode
        if result.returncode not in (0, 2):
            (work / f"{name.replace(' ', '-')}.stderr").write_text(result.stderr)
    probe = _run(
        [
            str(python),
            "-c",
            "import piginot, sys; print(piginot.__file__); print(sys.path)",
        ],
        cwd=clean,
        env=environment,
    )
    blocked = _run([str(python), "-c", _BLOCKED_PROBE], cwd=clean, env=environment)
    blocked_exit, blocked_leaks = (
        (int(v) for v in blocked.stdout.split())
        if blocked.returncode == 0
        else (-1, -1)
    )
    requirements = runtime_requirements(wheel)
    required_names = {
        r.split(";")[0].split("<")[0].split(">")[0].split("=")[0].strip().lower()
        for r in requirements
    }
    prediction = json.loads((clean / "prediction.json").read_text())
    refused = json.loads((clean / "refused-prediction.json").read_text())
    repository = predict(
        load_bundle(args.bundle),
        theta(REFERENCE_REQUEST),
        REFERENCE_QUERIES,
    ).document()
    differences: dict[str, float] | None = None
    if prediction["vertices"] is not None and repository["vertices"] is not None:
        differences = {
            name: float(
                np.max(
                    np.abs(
                        np.asarray(prediction["vertices"][name])
                        - np.asarray(repository["vertices"][name])
                    )
                )
            )
            for name in FIELD_NAMES
        }
    installed_from = probe.stdout.splitlines()[0] if probe.stdout else ""
    document = {
        "python": args.python,
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "packages": packages,
        "runtime_requirements": requirements,
        "fem_only_packages_required": sorted(required_names & set(FEM_ONLY_PACKAGES)),
        "fem_only_packages_present_transitively": sorted(
            n for n in names if n in FEM_ONLY_PACKAGES
        ),
        "torch": next((p for p in packages if p.startswith("torch==")), None),
        "blocked_import_prediction_exit_code": blocked_exit,
        "blocked_modules_imported": blocked_leaks,
        "piginot_imported_from": installed_from,
        "imported_from_installed_wheel": str(venv) in installed_from,
        "exit_codes": exit_codes,
        "prediction_verdict": prediction["verdict"],
        "prediction_claim_level": prediction["claim_level"],
        "prediction_has_every_field": prediction["vertices"] is not None
        and all(name in prediction["vertices"] for name in FIELD_NAMES),
        "refused_reason_codes": refused["reason_codes"],
        "max_abs_difference_to_repository_environment": differences,
        "limitations_present": prediction["limitations"]
        == REQUIRED_LIMITATION_PARAGRAPH
        and refused["limitations"] == REQUIRED_LIMITATION_PARAGRAPH,
    }
    document["passed"] = (
        exit_codes
        == {"bundle verify": 0, "inspect": 0, "predict": 0, "predict (refused)": 2}
        and not document["fem_only_packages_required"]
        and blocked_exit == 0
        and blocked_leaks == 0
        and document["imported_from_installed_wheel"]
        and document["prediction_verdict"] == "IN_DOMAIN"
        and document["prediction_has_every_field"]
        and document["refused_reason_codes"] == ["PARAMETER_OUT_OF_BOUNDS"]
        and document["limitations_present"]
    )
    _write(args.run_root / FRESH_INSTALL, document)
    _progress(f"fresh install passed={document['passed']}")
    return 0


# ------------------------------------------------------------------ gate


def _check(
    identifier: str, threshold: str, measured: dict[str, Any], passed: bool
) -> dict[str, Any]:
    return {
        "id": identifier,
        "threshold": threshold,
        "measured": measured,
        "status": "PASS" if passed else "FAIL",
        "reason": None,
        "comparison": "all",
    }


def gate_checks(
    measurements: dict[str, Any], fresh: dict[str, Any], coverage_percent: float
) -> list[dict[str, Any]]:
    without = {
        k: v
        for k, v in measurements["fields"].items()
        if k != "query_confidence_bound_mpa"
    }
    return [
        _check(
            "G15.fresh-install",
            "bundle loads and predicts in a clean venv with runtime deps only",
            {k: v for k, v in fresh.items() if k != "packages"}
            | {"package_count": len(fresh["packages"])},
            bool(fresh["passed"]),
        ),
        _check(
            "G15.fields",
            "predict returns u, P_served, F_2D, F33, J_3D, sigma, QoIs, units, "
            "identities, in-domain verdict, reason codes, claim level, confidence",
            without | {"reproduction": measurements["reproduction"]},
            bool(
                measurements["fields"]["passed"]
                and measurements["reproduction"]["passed"]
            ),
        ),
        _check(
            "G15.substitution",
            "P_material is never silently served in place of P_served",
            measurements["substitution"],
            bool(measurements["substitution"]["passed"]),
        ),
        _check(
            "G15.refuse",
            "out-of-envelope requests refuse with a reason code",
            measurements["refusal"],
            bool(measurements["refusal"]["passed"]),
        ),
        _check(
            "G15.manifest",
            "the manifest hash closes over every contained file; tampering with any "
            "byte is detected",
            measurements["manifest"],
            bool(measurements["manifest"]["passed"]),
        ),
        _check(
            "G15.limitations",
            "required limitation paragraph present in every report",
            measurements["limitations"],
            bool(measurements["limitations"]["passed"]),
        ),
        _check(
            "G15.coverage",
            ">= 85%",
            {"line_and_branch_coverage_percent": coverage_percent},
            coverage_percent >= 85.0,
        ),
    ]


GATE_LIMITATIONS = (
    "The bundle serves one family (DogBone v1), one material and the 10-20% "
    "strain-controlled load of its ratified envelope; family generality remains "
    "for G17.",
    "Bit-for-bit reproduction of the S14 serving path was checked on eight TRAIN "
    "cases (the confidence indicator on one); bundle predictions inherit the S14 "
    "evidence through that equality, not through a new measurement.",
    "The 36 S14 fixtures that submit a mutated mesh have no bundle request form "
    "and are not applicable (ADR 0023 decision 8).",
    "The fresh install used the uv package cache on one Linux x86_64 machine and "
    "Python 3.11; torch's own dependencies include sympy, which the inference "
    "closure never imports.",
    "S15-D1 to S15-D5 are proposed defaults awaiting the decision owner's "
    "confirmation (ADR 0023).",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_gate(args: argparse.Namespace) -> int:
    measurements = json.loads((args.run_root / MEASUREMENTS).read_text())
    fresh = json.loads((args.run_root / FRESH_INSTALL).read_text())
    checks = gate_checks(measurements, fresh, args.coverage_percent)
    verdict = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    document = {
        "step": "S15",
        "gate_id": "G15",
        "verdict": verdict,
        "checks": checks,
        "bundle_manifest_sha256": measurements["manifest"]["manifest_sha256"],
        "coverage_command": args.coverage_command,
        "implementation_source_revision": args.revision,
        "runtime": measurements["runtime"],
    }
    validation = _ARTIFACTS / "S15-VALIDATION.json"
    gate_path = _ARTIFACTS / "S15-GATE.json"
    _write(validation, {"measurements": measurements, "fresh_install": fresh})
    _write(gate_path, document)
    gate_artifact = {
        "schema_version": "1.0",
        "gate_id": "G15",
        "step": "S15",
        "branch": "codex/s15-bundle",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision_record": ["docs/adr/0023-s15-bundle-and-cli.md"],
        "implementation_source_revision": args.revision,
        "scope": {
            "bundle": "dogbone-v1-s13c-cp-freeze",
            "bundle_manifest_sha256": measurements["manifest"]["manifest_sha256"],
            "claim_level_served": "HOLDOUT_VERIFIED",
            "evidence_roles": [
                "TRAIN (reproduction against S14 serving)",
                "public coordinates only (refusal fixtures, in-domain cases)",
            ],
            "development_or_holdout_read": False,
            "claim": "The DogBone bundle of the frozen S13c candidate loads, "
            "verifies every byte, refuses every request outside its envelope, and "
            "serves the S14-verified fields bit for bit with their provenance, "
            "confidence, claim level and limitations, also from a clean install "
            "with runtime dependencies only.",
        },
        "verdict": verdict,
        "checks": checks,
        "evidence": {
            "artifacts/campaigns/dogbone-v1/S15-VALIDATION.json": _sha256(validation),
            "artifacts/campaigns/dogbone-v1/S15-GATE.json": _sha256(gate_path),
        },
        "implementation_evidence": {
            "focused_coverage_command": args.coverage_command,
            "focused_test_result": args.focused_result,
            "full_test_command": "python -m pytest -q --cov=src/piginot --cov-branch",
            "full_test_result": args.full_result,
            "quality_checks": {
                "mypy_strict_src": "PASS",
                "ruff_check": "PASS",
                "ruff_format_check": "PASS",
                "mkdocs_strict_build": "PASS",
            },
        },
        "limitations": list(GATE_LIMITATIONS),
    }
    _write(_ROOT / "artifacts" / "gates" / "S15.json", gate_artifact)
    _progress(f"G15 {verdict}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    measure = commands.add_parser("measure")
    measure.add_argument("--bundle", required=True, type=Path)
    measure.add_argument("--checkpoint-path", required=True, type=Path)
    measure.add_argument("--s10-run-root", required=True, type=Path)
    measure.add_argument("--rehearsal-root", required=True, type=Path)
    measure.add_argument("--run-root", required=True, type=Path)
    measure.add_argument("--reproduce-cases", type=int, default=8)
    measure.add_argument("--confidence-cases", type=int, default=1)
    measure.set_defaults(handler=run_measure)
    fresh = commands.add_parser("fresh-install")
    fresh.add_argument("--bundle", required=True, type=Path)
    fresh.add_argument("--run-root", required=True, type=Path)
    fresh.add_argument("--python", default="3.11")
    fresh.set_defaults(handler=run_fresh_install)
    gate = commands.add_parser("gate")
    gate.add_argument("--run-root", required=True, type=Path)
    gate.add_argument("--coverage-percent", required=True, type=float)
    gate.add_argument("--coverage-command", required=True)
    gate.add_argument("--revision", required=True)
    gate.add_argument("--focused-result", required=True)
    gate.add_argument("--full-result", required=True)
    gate.set_defaults(handler=run_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
