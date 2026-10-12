"""Verified serving: ``fast``, ``certified`` and ``exact`` (S22, ADR 0103).

- ``fast``: the bundle's prediction, unchanged from v0.1.0.
- ``certified``: the prediction plus the S21 certificate, evaluated on the FEM
  problem the bundle's recipe rebuilds for the request; with ``corrected`` also
  the one-step corrected displacement at the FEM nodes. With ``escalate``, a
  certificate that is not ``OK`` (or whose stress change exceeds
  ``tolerance``) promotes the request to ``exact``.
- ``exact``: the converged FEM solution, started from the prediction. The
  solve ladder is warm Newton at full load, then a short continuation from the
  half-scaled prediction, then the cold load-stepped solve, then an explicit
  failure. Every response names the path that produced it.

``fast`` needs only the runtime extra. The other modes import the FEM stack
lazily and raise ``MissingExtraError`` without it.
"""

from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from piginot.bundle import predict

if TYPE_CHECKING:
    from piginot.fem.models import FemRequest, FemResult
    from piginot.serving.recipe import FemRecipe


class ServingMode(enum.StrEnum):
    FAST = "fast"
    CERTIFIED = "certified"
    EXACT = "exact"


class ServingPath(enum.StrEnum):
    OPERATOR = "operator"
    OPERATOR_CERTIFICATE = "operator+certificate"
    WARM_NEWTON = "warm_newton"
    CONTINUATION = "continuation"
    COLD_FEM = "cold_fem"
    SOLVE_FAILED = "solve_failed"
    REFUSED = "refused"


class MissingExtraError(ImportError):
    """A certified or exact request without the FEM dependencies installed."""

    def __init__(self, mode: str) -> None:
        super().__init__(
            f"mode {mode!r} needs the FEM extra: pip install 'piginot[verify]'"
        )


@dataclass(frozen=True, slots=True)
class SolveAttempt:
    """One rung of the solve ladder."""

    path: ServingPath
    success: bool
    newton_iterations: int
    failure_code: str | None
    message: str

    def document(self) -> dict[str, Any]:
        return {
            "path": self.path.value,
            "success": self.success,
            "newton_iterations": self.newton_iterations,
            "failure_code": self.failure_code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class VerifiedPrediction:
    """A served prediction and how far it was verified."""

    mode: ServingMode
    path: ServingPath
    prediction: Any
    certificate: Mapping[str, Any] | None = None
    corrected_displacement: np.ndarray[Any, Any] | None = None
    exact: FemResult | None = None
    exact_result_hash: str | None = None
    attempts: tuple[SolveAttempt, ...] = ()
    escalated: bool = False
    request_hash: str | None = None
    reason: str = ""

    def document(self) -> dict[str, Any]:
        """JSON form; ``fast`` returns exactly the v0.1.0 prediction document."""

        if self.mode is ServingMode.FAST:
            document: dict[str, Any] = self.prediction.document()
            return document
        return {
            "mode": self.mode.value,
            "path": self.path.value,
            "escalated": self.escalated,
            "reason": self.reason,
            "request_hash": self.request_hash,
            "prediction": self.prediction.document(),
            "certificate": None if self.certificate is None else dict(self.certificate),
            "corrected_displacement": None
            if self.corrected_displacement is None
            else self.corrected_displacement.tolist(),
            "exact": None
            if self.exact is None
            else {
                "result_hash": self.exact_result_hash,
                "total_potential_energy": self.exact.total_potential_energy,
                "displacement": [list(v) for v in self.exact.displacement],
                "displacement_dof_coordinates": [
                    list(v) for v in self.exact.displacement_dof_coordinates
                ],
            },
            "attempts": [attempt.document() for attempt in self.attempts],
        }


def default_request_id(inputs: Mapping[str, Any]) -> str:
    """A request ID derived from the inputs (canonical values and units)."""

    payload = {
        name: [float(q.value), str(getattr(q.unit, "value", q.unit))]
        for name, q in sorted(inputs.items())
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return "served-" + digest.hexdigest()[:16]


def operator_displacement(
    bundle: Any, inputs: Mapping[str, Any], mesh: Any, points: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """The operator's displacement at ``points``, without stresses (S22).

    Bit-identical to the ``u`` that :func:`piginot.bundle.predict` serves at the
    same query points, at a fraction of the cost.
    """

    import torch

    from piginot.bundle.inference import _Request

    request = _Request(bundle, inputs, mesh)
    u = bundle.serving.displacement(
        torch.tensor(points, dtype=torch.float64),
        parameters=request.parameters,
        frame=request.frame,
        lifting=request.lifting,
        chunk=bundle.model.query_chunk,
    )
    result: np.ndarray[Any, Any] = u.numpy().astype(np.float64)
    return result


def _fem(mode: ServingMode) -> tuple[Any, Any, Any]:
    try:
        from piginot.certify import certify_with_correction
        from piginot.fem import ScikitFemBackend
        from piginot.fem.linearization import FemLinearization
    except ImportError as exc:
        raise MissingExtraError(mode.value) from exc
    return certify_with_correction, ScikitFemBackend, FemLinearization


def solve_ladder(
    backend: Any, request: FemRequest, initial: np.ndarray[Any, Any]
) -> tuple[Any | None, tuple[SolveAttempt, ...]]:
    """Warm Newton, continuation, cold solve; the first success wins."""

    ladder: Sequence[
        tuple[ServingPath, np.ndarray[Any, Any] | None, tuple[float, ...]]
    ] = (
        (ServingPath.WARM_NEWTON, initial, (1.0,)),
        (ServingPath.CONTINUATION, 0.5 * initial, (0.5, 0.75, 1.0)),
        (ServingPath.COLD_FEM, None, ()),
    )
    attempts: list[SolveAttempt] = []
    for path, start, factors in ladder:
        attempt = (
            backend.solve_many((request,))[0]
            if start is None
            else backend.solve_from(request, start, factors)
        )
        iterations = (
            len(attempt.result.convergence_history) if attempt.result is not None else 0
        )
        attempts.append(
            SolveAttempt(
                path,
                bool(attempt.success),
                iterations,
                None if attempt.failure_code is None else str(attempt.failure_code),
                str(attempt.message),
            )
        )
        if attempt.success:
            return attempt, tuple(attempts)
    return None, tuple(attempts)


def serve(
    bundle: Any,
    inputs: Mapping[str, Any],
    queries: Sequence[Sequence[float]] | None = None,
    *,
    mode: ServingMode | str = ServingMode.FAST,
    recipe: FemRecipe | None = None,
    corrected: bool = False,
    escalate: bool = False,
    tolerance: float | None = None,
    request_id: str | None = None,
    backend: Any = None,
    confidence: bool | None = None,
) -> VerifiedPrediction:
    """Serve one request in the chosen mode (see the module docstring).

    ``confidence`` controls the S14 pointwise bound in the returned prediction:
    on by default in ``fast`` (as in v0.1.0), off by default otherwise, where
    the certificate supersedes it.
    """

    mode = ServingMode(mode)
    if mode is ServingMode.FAST:
        prediction = (
            predict(bundle, inputs, queries)
            if confidence is None
            else predict(bundle, inputs, queries, confidence=confidence)
        )
        return VerifiedPrediction(mode, ServingPath.OPERATOR, prediction)
    prediction = predict(bundle, inputs, queries, confidence=bool(confidence))
    if prediction.mesh is None or prediction.verdict.value == "REFUSED":
        return VerifiedPrediction(
            mode,
            ServingPath.REFUSED,
            prediction,
            reason="the bundle refused the request: "
            + ", ".join(str(code) for code in prediction.reason_codes),
        )
    if recipe is None:
        raise ValueError("certified and exact modes need the bundle's FEM recipe")
    certify_with_correction, backend_type, linearization_type = _fem(mode)
    request = recipe.build_request(
        inputs, prediction.mesh, request_id or default_request_id(inputs)
    )
    linearization = linearization_type(request)
    surrogate = operator_displacement(
        bundle, inputs, prediction.mesh, linearization.dof_coordinates
    )
    certificate, corrected_vector = certify_with_correction(linearization, surrogate)
    corrected_nodal = (
        linearization.nodal(corrected_vector)
        if corrected and corrected_vector is not None
        else None
    )
    p_change = certificate.indicators.get("p_l2_change", float("inf"))
    needs_exact = mode is ServingMode.EXACT or (
        escalate
        and (
            certificate.verdict.value != "OK"
            or (tolerance is not None and p_change > tolerance)
        )
    )
    request_hash = str(request.recipe_hash)
    if not needs_exact:
        return VerifiedPrediction(
            mode,
            ServingPath.OPERATOR_CERTIFICATE,
            prediction,
            certificate.document(),
            corrected_nodal,
            request_hash=request_hash,
        )
    start = linearization.with_exact_dirichlet(linearization.vector(surrogate))
    attempt, attempts = solve_ladder(backend or backend_type(), request, start)
    return VerifiedPrediction(
        mode,
        attempts[-1].path if attempt is not None else ServingPath.SOLVE_FAILED,
        prediction,
        certificate.document(),
        corrected_nodal,
        None if attempt is None else attempt.result,
        None if attempt is None else attempt.result_hash,
        attempts,
        escalated=mode is ServingMode.CERTIFIED,
        request_hash=request_hash,
        reason="" if attempt is not None else "every rung of the solve ladder failed",
    )
