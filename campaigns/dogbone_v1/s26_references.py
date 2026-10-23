"""S26 amendment 1: uniform neo-Hookean initialization of HGO references."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from campaigns.dogbone_v1.s13c_trial import DEVELOPMENT_ACCESS_NAME, Progress
from campaigns.dogbone_v1.s25_mooney_rivlin import encode_result
from campaigns.dogbone_v1.s25_mooney_rivlin import reference_entry as final_entry
from piginot.contracts import (
    CanonicalQuantity,
    MaterialSpec,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.fem import FemRequest, ScikitFemBackend
from piginot.fem.linearization import FemLinearization

AMENDMENT_PATH = Path("artifacts/materials/S26-REFERENCE-AMENDMENT.json")
PROTOCOL_PATH = Path("artifacts/materials/S26-PROTOCOL.json")
BINDING_NAME = "reference-solver.json"


def build_amendment(root: Path) -> dict[str, Any]:
    """The owner-accepted reference procedure; the original protocol is unchanged."""

    return {
        "schema_version": "1.0",
        "step": "S26",
        "status": "accepted",
        "amends": PROTOCOL_PATH.as_posix() + ", viability.references",
        "protocol_sha256": hashlib.sha256(
            (root / PROTOCOL_PATH).read_bytes()
        ).hexdigest(),
        "decision_record": "docs/adr/0108-s26-fibre-families.md, amendment 1",
        "accepted": {
            "by": "decision owner",
            "on": "2026-10-23",
            "answer": "Accept the amendment",
            "before": "any reference re-solve under this amended procedure",
        },
        "reason": "candidate-001561 failed the cold HGO reference solve at load "
        "step 1 within the registered 20 Newton iterations; no viability "
        "guardrail result was used to select this procedure",
        "rule": "all 64 TRAIN-screen cases: solve a compressible neo-Hookean "
        "initializer with matching small-strain E and nu, then solve the full "
        "HGO request at load factor 1 from its converged displacement",
        "initializer": {
            "model": "compressible_neo_hookean",
            "elastic_modulus": "9*K*mu/(3*K+mu)",
            "poisson_ratio": "(3*K-2*mu)/(2*(3*K+mu))",
            "other_request_fields": "unchanged, including mesh, loads and solver config",
        },
        "hgo_load_factors": [1.0],
        "unchanged": [
            "HGO material, mesh, loads, boundary conditions and execution profile",
            "solver tolerance, Newton iteration limit and plane-stress closure",
            "training recipe, populations, evidence boundaries and guardrails",
        ],
        "cache_rule": "fresh output directory; resume only with an identical "
        "amendment hash, implementation revision and request hashes; retain "
        "both initializer and final HGO results",
    }


def accepted_amendment(root: Path, *, require_committed: bool = True) -> dict[str, Any]:
    path = root / AMENDMENT_PATH
    if not path.is_file():
        raise RuntimeError(f"{AMENDMENT_PATH} is not written")
    document: dict[str, Any] = json.loads(path.read_text("utf-8"))
    if document != build_amendment(root):
        raise ValueError(
            "the S26 reference amendment differs from its accepted rebuild"
        )
    if require_committed:
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{AMENDMENT_PATH.as_posix()}"],
            capture_output=True,
            check=False,
        )
        if committed.returncode != 0 or committed.stdout != path.read_bytes():
            raise RuntimeError(
                "the S26 reference amendment must be committed before re-solves"
            )
    return document


def amendment_digest(root: Path) -> str:
    return hashlib.sha256((root / AMENDMENT_PATH).read_bytes()).hexdigest()


def initializer_request(request: FemRequest) -> FemRequest:
    """Same request, using the existing NH law with matching small-strain moduli."""

    if request.material.model != "hgo":
        raise ValueError("the S26 reference initializer requires an HGO request")
    mu = request.material.properties["shear_modulus"].value
    bulk = request.material.properties["bulk_modulus"].value
    material = MaterialSpec(
        name="dogbone-hgo-reference-initializer",
        model="compressible_neo_hookean",
        properties={
            "elastic_modulus": CanonicalQuantity(
                9 * bulk * mu / (3 * bulk + mu), Unit.MPA
            ),
            "poisson_ratio": CanonicalQuantity(
                (3 * bulk - 2 * mu) / (2 * (3 * bulk + mu)), Unit.ONE
            ),
        },
    )
    return replace(request, material=material)


def solve_reference(request: FemRequest) -> tuple[bytes, bytes]:
    """Keep both successful solves; neither failed stage produces a reference."""

    torch.set_num_threads(1)
    backend = ScikitFemBackend()
    initial = backend.solve_many((initializer_request(request),))[0]
    if not initial.success or initial.result is None:
        raise RuntimeError(f"{request.request_id}: initializer: {initial.message}")
    vector = FemLinearization(request).vector(np.asarray(initial.result.displacement))
    final = backend.solve_from(request, vector, (1.0,))
    if not final.success or final.result is None:
        raise RuntimeError(f"{request.request_id}: HGO: {final.message}")
    return encode_result(initial.result), encode_result(final.result)


def reference_entry(
    request: FemRequest, payload: bytes, output: Path
) -> dict[str, str]:
    initializer = (output / "initializers" / f"{request.request_id}.json").read_bytes()
    return {
        **final_entry(request, payload),
        **{
            "initializer_" + name: value
            for name, value in final_entry(
                initializer_request(request), initializer
            ).items()
        },
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload)
    os.replace(partial, path)


def _bind_output(
    requests: Mapping[str, FemRequest], output: Path, root: Path, revision: str
) -> None:
    binding = {
        "reference_amendment_sha256": amendment_digest(root),
        "implementation_revision": revision,
        "requests": {
            case_id: {
                "hgo": hash_spec(request, "content"),
                "initializer": hash_spec(initializer_request(request), "content"),
            }
            for case_id, request in requests.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / BINDING_NAME
    if path.is_file():
        if json.loads(path.read_text("utf-8")) != binding:
            raise ValueError(
                "S26 reference directory binding differs; use a fresh output directory"
            )
    else:
        old_results = any(
            p.name != DEVELOPMENT_ACCESS_NAME for p in output.glob("*.json")
        )
        if old_results or any((output / "initializers").glob("*.json")):
            raise ValueError("old unbound references require a fresh output directory")
        _atomic_write(path, canonical_json_bytes(binding) + b"\n")
    (output / "initializers").mkdir(exist_ok=True)


def _solved(requests: list[FemRequest], workers: int) -> Iterator[tuple[bytes, bytes]]:
    if workers <= 1:
        yield from map(solve_reference, requests)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(solve_reference, requests)


def solve_references(
    requests: Mapping[str, FemRequest],
    output: Path,
    *,
    root: Path,
    implementation_revision: str,
    workers: int = 1,
    require_committed: bool = True,
    progress: Progress | None = None,
) -> dict[str, dict[str, str]]:
    accepted_amendment(root, require_committed=require_committed)
    _bind_output(requests, output, root, implementation_revision)
    pending = [case for case in requests if not (output / f"{case}.json").is_file()]
    results = _solved([requests[case] for case in pending], workers)
    for case_id, (initial, final) in zip(pending, results, strict=True):
        _atomic_write(output / "initializers" / f"{case_id}.json", initial)
        _atomic_write(output / f"{case_id}.json", final)
        if progress is not None:
            progress(f"phase=REFERENCE case={case_id}")
    return {
        case_id: reference_entry(
            request, (output / f"{case_id}.json").read_bytes(), output
        )
        for case_id, request in requests.items()
    }
