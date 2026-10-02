"""Request-level envelope refusal (``G14.refusal``, ``G14.reason-codes``).

A request is a parameter point, optionally an explicit mesh, and optionally
query points. It is accepted only when

1. the S9 executable envelope accepts every primitive, material, load and
   coupled derived bound (exact comparisons: a value just outside a bound is
   refused, which is the adversarial guard band);
2. the geometry provider accepts the point (S6 morph validity) or an explicit
   mesh is supplied;
3. the mesh keeps the family's topology signature and the reference boundary
   tag identity (tag names and facet counts);
4. every query point is finite and lies in the mesh closure.

Reason codes are the stable values of the existing S6, S9 and S5 enums, sorted,
so repeated runs emit identical codes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from piginot.contracts import CanonicalMesh
from piginot.doe.envelope import ExecutableEnvelope
from piginot.geometry.errors import GeometryReasonCode
from piginot.geometry.models import PointLocation
from piginot.geometry.queries import locate_points
from piginot.geometry.topology import topology_summary
from piginot.morphing.models import MorphReasonCode


@dataclass(frozen=True, slots=True)
class GeometryOutcome:
    """What a geometry provider returns for one parameter point."""

    mesh: CanonicalMesh | None
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RefusalDecision:
    """Accept, or refuse with sorted stable reason codes."""

    accepted: bool
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.accepted and self.reason_codes:
            raise ValueError("an accepted request carries no reason code")
        if not self.accepted and not self.reason_codes:
            raise ValueError("a refusal needs at least one reason code")


def _codes(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(sorted({str(getattr(value, "value", value)) for value in values}))


def _tag_identity(mesh: CanonicalMesh) -> dict[str, int]:
    return {name: len(facets) for name, facets in mesh.semantic_tags.items()}


def _finite_pair(point: object) -> bool:
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        return False
    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in point
    )


class EnvelopeChecker:
    """Fail-closed request checker for one compiled family."""

    def __init__(
        self,
        envelope: ExecutableEnvelope,
        *,
        reference_mesh: CanonicalMesh,
        geometry: Callable[[Mapping[str, object]], GeometryOutcome],
    ) -> None:
        if type(envelope) is not ExecutableEnvelope:
            raise TypeError("envelope must be an ExecutableEnvelope")
        if not isinstance(reference_mesh, CanonicalMesh):
            raise TypeError("reference_mesh must be a CanonicalMesh")
        if not callable(geometry):
            raise TypeError("geometry must be callable")
        self._envelope = envelope
        self._topology = topology_summary(reference_mesh).signature
        self._tags = _tag_identity(reference_mesh)
        self._geometry = geometry

    def check(
        self,
        theta: Mapping[str, object],
        *,
        mesh: CanonicalMesh | None = None,
        queries: Sequence[object] | None = None,
    ) -> RefusalDecision:
        decision = self._envelope.evaluate(theta)
        if not decision.accepted:
            return RefusalDecision(False, _codes(decision.reason_codes))
        if mesh is None:
            outcome = self._geometry(theta)
            if outcome.mesh is None or outcome.reason_codes:
                codes = outcome.reason_codes or (
                    MorphReasonCode.INVALID_MORPHED_MESH.value,
                )
                return RefusalDecision(False, _codes(codes))
            mesh = outcome.mesh
        elif not isinstance(mesh, CanonicalMesh):
            return RefusalDecision(False, (MorphReasonCode.INVALID_MORPHED_MESH.value,))
        reasons: list[str] = []
        if topology_summary(mesh).signature != self._topology:
            reasons.append(MorphReasonCode.TOPOLOGY_CHANGED.value)
        if _tag_identity(mesh) != self._tags:
            reasons.append(MorphReasonCode.TAG_IDENTITY_CHANGED.value)
        if reasons:
            return RefusalDecision(False, _codes(reasons))
        if queries is not None:
            points = tuple(queries)
            if not points or not all(_finite_pair(point) for point in points):
                return RefusalDecision(False, (GeometryReasonCode.INVALID_QUERY.value,))
            locations = locate_points(mesh, points)
            if any(location is PointLocation.EXTERIOR for location in locations):
                return RefusalDecision(
                    False, (GeometryReasonCode.QUERY_OUTSIDE_DOMAIN.value,)
                )
        return RefusalDecision(True)


__all__ = ("EnvelopeChecker", "GeometryOutcome", "RefusalDecision")
