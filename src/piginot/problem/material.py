"""The admission gate of a ProblemSpec's material (S25, ADR 0107).

The material is admitted at every corner of its parameters' box and at the
centre (one point when every material parameter is fixed), over the spec's
stretch range: checks 1-7 always, check 8 (FEM convergence on a
manufactured solution, ``verify`` extra) when ``fem`` is true.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

from piginot.contracts import CanonicalQuantity, MaterialSpec, ProblemSpec
from piginot.materials.admission import AdmissionReport, admit


def material_spec(
    spec: ProblemSpec, values: Mapping[str, float], name: str | None = None
) -> MaterialSpec:
    """The material contract of ``spec`` for parameter ``values``; registers a
    user formula."""

    material = spec.material
    if material.energy is not None:
        from piginot.materials import register

        register(material.energy)
    properties = {}
    for prop, parameter in material.parameters.items():
        unit = spec.parameter(parameter).unit
        properties[prop] = CanonicalQuantity(
            float(values[parameter]), unit, unit.dimension
        )
    return MaterialSpec(
        name=name or f"{spec.name}-material",
        model=material.model,
        properties=properties,
    )


def material_points(spec: ProblemSpec) -> list[dict[str, float]]:
    """The corners and the centre of the material parameters' box."""

    names = sorted(set(spec.material.parameters.values()))
    ranges = [(spec.parameter(n).lo, spec.parameter(n).hi) for n in names]
    points = []
    for corner in product(*({lo, hi} for lo, hi in ranges)):
        points.append(dict(zip(names, corner, strict=True)))
    centre = {n: 0.5 * (lo + hi) for n, (lo, hi) in zip(names, ranges, strict=True)}
    if centre not in points:
        points.append(centre)
    return points


@dataclass(frozen=True, slots=True)
class MaterialAdmission:
    """The admission reports of a spec's material at each parameter point."""

    problem_hash: str
    model: str
    points: tuple[Mapping[str, float], ...]
    reports: tuple[AdmissionReport, ...]
    fem: tuple[Any, ...]

    @property
    def passed(self) -> bool:
        return all(report.passed for report in self.reports) and all(
            check.passed for check in self.fem
        )

    def document(self) -> dict[str, Any]:
        return {
            "problem_hash": self.problem_hash,
            "model": self.model,
            "points": [dict(point) for point in self.points],
            "reports": [report.document() for report in self.reports],
            "fem": [check.document() for check in self.fem],
            "passed": self.passed,
        }


def admit_material(spec: ProblemSpec, *, fem: bool = False) -> MaterialAdmission:
    """Run the admission gate on ``spec``'s material."""

    points = material_points(spec)
    materials = [material_spec(spec, point) for point in points]
    reports = tuple(
        admit(material, stretch_range=spec.material.stretch_range)
        for material in materials
    )
    checks: tuple[Any, ...] = ()
    if fem:
        from piginot.materials.fem_admission import check_fem

        checks = tuple(check_fem(material) for material in materials)
    return MaterialAdmission(
        problem_hash=spec.problem_hash,
        model=spec.material.model,
        points=tuple(points),
        reports=reports,
        fem=checks,
    )


__all__ = ("MaterialAdmission", "admit_material", "material_points", "material_spec")
