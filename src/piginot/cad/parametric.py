"""Parametric CAD contracts, OpenCASCADE templates, and topology stability guards (M4 / F3).

Provides:
* :class:`ParametricCadSpec` (``schema = "formamech.parametric-cad-v1"``): immutable,
  hash-bound declaration of CAD design parameters, safe :class:`GeometricBound`
  constraints (evaluated via :class:`~piginot.contracts.expressions.DerivedExpression`),
  expected boundary topology, BC role map, and meshing policy.
* Pluggable CAD generator registry (built-in ``"gmsh-occ-template-v1"`` with
  ``"notched-mounting-bracket"`` and ``"holed-plate"`` OpenCASCADE templates, plus
  custom/entry-point plugins under ``formamech.cad.generators``).
* :class:`TopologyStabilityGuard`: fail-closed verifier enforcing parameter box,
  derived geometric bounds, exact Euler component/hole counts (genus invariance),
  exact semantic boundary group set, closed-manifold loop admissibility, and
  positive element quality on every generated CAD instance.
* Deterministic quasi-random DOE sampler (:func:`sample_parametric_cad_doe`) and
  parametric project persistence (:func:`init_parametric_project`,
  :func:`load_parametric_project`).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final, cast

from piginot.cad.mesh_policy import MeshPolicy
from piginot.contracts import CanonicalMesh, Unit, canonical_json_bytes
from piginot.contracts.expressions import DerivedExpression, ExpressionTree
from piginot.geometry import (
    CellType,
    MeshFamilyKind,
    MeshPayload,
    TopologySummary,
    canonicalize_mesh,
    classify_mesh_family,
    topology_summary,
)
from piginot.operators.geometry_tokenizer import (
    SUPPORTED_BC_ROLES,
    BoundaryRoleMap,
    GeometryDescriptor,
    GeometryTokenizer,
    UnsupportedTopologyError,
    compute_geometry_descriptor,
)
from piginot.optimization.problem import DesignParameter

PARAMETRIC_CAD_SCHEMA: Final = "formamech.parametric-cad-v1"
PARAMETRIC_DOE_SCHEMA: Final = "formamech.parametric-doe-v1"
GENERATOR_ENTRY_POINT_GROUP: Final = "formamech.cad.generators"
BUILTIN_TEMPLATE_GENERATOR_ID: Final = "gmsh-occ-template-v1"
BUILTIN_TEMPLATES: Final[tuple[str, ...]] = (
    "notched-mounting-bracket",
    "holed-plate",
)


class ParametricTopologyError(ValueError):
    """Fail-closed exception raised when a CAD parameter point breaks topology or bounds."""


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _finite(
    value: object, label: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{label} must be finite")
    if positive and num <= 0.0:
        raise ValueError(f"{label} must be strictly positive")
    if nonnegative and num < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return num


def _eval_ast(tree: ExpressionTree, env: Mapping[str, float]) -> float:
    kind = tree[0]
    if kind == "number":
        return float(cast(float, tree[1]))
    if kind == "name":
        name = cast(str, tree[1])
        if name not in env:
            raise ValueError(f"unknown geometric bound symbol {name!r}")
        return float(env[name])
    if kind == "unary":
        op = cast(str, tree[1])
        val = _eval_ast(cast(ExpressionTree, tree[2]), env)
        return -val if op == "-" else val
    if kind == "binary":
        op = cast(str, tree[1])
        lhs = _eval_ast(cast(ExpressionTree, tree[2]), env)
        rhs = _eval_ast(cast(ExpressionTree, tree[3]), env)
        if op == "+":
            return lhs + rhs
        if op == "-":
            return lhs - rhs
        if op == "*":
            return lhs * rhs
        if op == "/":
            if abs(rhs) <= 1.0e-15:
                raise ZeroDivisionError("division by zero in geometric bound")
            return lhs / rhs
    raise ValueError(f"unsupported expression AST node: {tree!r}")


@dataclass(frozen=True, slots=True)
class GeometricBound:
    """Declarative safe algebraic bound over CAD parameters (e.g. minimum ligament)."""

    name: str
    expression: str
    lower: float | None = None
    upper: float | None = None
    unit: str = "mm"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("bound name must be non-empty text")
        if not isinstance(self.expression, str) or not self.expression.strip():
            raise ValueError("bound expression must be non-empty text")
        DerivedExpression.parse(self.expression)
        lo = None if self.lower is None else _finite(self.lower, f"{self.name}.lower")
        hi = None if self.upper is None else _finite(self.upper, f"{self.name}.upper")
        if lo is None and hi is None:
            raise ValueError(f"{self.name}: at least one of lower or upper must be set")
        if lo is not None and hi is not None and hi < lo:
            raise ValueError(f"{self.name}: upper must not be less than lower")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError(f"{self.name}: unit must be non-empty text")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    def evaluate(self, env: Mapping[str, float]) -> float:
        expr = DerivedExpression.parse(self.expression)
        return _eval_ast(expr.canonical_ast, env)

    def check(self, env: Mapping[str, float]) -> tuple[bool, float, str]:
        val = self.evaluate(env)
        if self.lower is not None and val < self.lower - 1.0e-12:
            return (
                False,
                val,
                f"geometric bound {self.name!r} violated: {val:.6g} < {self.lower:.6g} {self.unit}",
            )
        if self.upper is not None and val > self.upper + 1.0e-12:
            return (
                False,
                val,
                f"geometric bound {self.name!r} violated: {val:.6g} > {self.upper:.6g} {self.unit}",
            )
        return True, val, "ok"

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "lower": self.lower,
            "upper": self.upper,
            "unit": self.unit,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> GeometricBound:
        return cls(
            name=str(doc["name"]),
            expression=str(doc["expression"]),
            lower=None if doc.get("lower") is None else float(doc["lower"]),
            upper=None if doc.get("upper") is None else float(doc["upper"]),
            unit=str(doc.get("unit", "mm")),
        )


@dataclass(frozen=True, slots=True)
class ExpectedTopology:
    """Topology invariance contract for every instance of a parametric CAD family."""

    component_count: int
    hole_count: int
    boundary_groups: tuple[str, ...]
    boundary_kinds: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.component_count) is not int or self.component_count < 1:
            raise ValueError("component_count must be an integer >= 1")
        if type(self.hole_count) is not int or self.hole_count < 0:
            raise ValueError("hole_count must be an integer >= 0")
        groups = tuple(str(g) for g in self.boundary_groups)
        if not groups or any(not g.strip() for g in groups):
            raise ValueError("boundary_groups must be non-empty names")
        if len(set(groups)) != len(groups):
            raise ValueError("boundary_groups must be unique")
        cleaned_kinds: dict[str, str] = {}
        for key, kind in sorted(self.boundary_kinds.items()):
            if key not in groups:
                raise ValueError(
                    f"boundary_kinds key {key!r} is not listed in boundary_groups"
                )
            if kind not in ("line", "arc", "other"):
                raise ValueError(
                    f"boundary kind for {key!r} must be 'line', 'arc', or 'other'"
                )
            cleaned_kinds[str(key)] = str(kind)
        object.__setattr__(self, "boundary_groups", groups)
        object.__setattr__(self, "boundary_kinds", cleaned_kinds)

    def document(self) -> dict[str, Any]:
        return {
            "component_count": self.component_count,
            "hole_count": self.hole_count,
            "boundary_groups": list(self.boundary_groups),
            "boundary_kinds": dict(sorted(self.boundary_kinds.items())),
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> ExpectedTopology:
        return cls(
            component_count=int(doc["component_count"]),
            hole_count=int(doc["hole_count"]),
            boundary_groups=tuple(str(g) for g in doc["boundary_groups"]),
            boundary_kinds={
                str(k): str(v) for k, v in dict(doc.get("boundary_kinds", {})).items()
            },
        )


@dataclass(frozen=True, slots=True)
class ParametricCadSpec:
    """Declarative, hash-bound parametric CAD family specification."""

    name: str
    generator_id: str
    template: str
    parameters: tuple[DesignParameter, ...]
    expected_topology: ExpectedTopology
    boundary_roles: Mapping[str, str]
    derived_bounds: tuple[GeometricBound, ...] = ()
    fixed_constants: Mapping[str, float] = ()  # type: ignore[assignment]
    unit: Unit = Unit.MM
    mesh_size_mm: float = 2.5
    min_angle_deg: float = 12.0
    mesh_policy: MeshPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty text")
        if not isinstance(self.generator_id, str) or not self.generator_id.strip():
            raise ValueError("generator_id must be non-empty text")
        if not isinstance(self.template, str) or not self.template.strip():
            raise ValueError("template must be non-empty text")
        if not self.parameters:
            raise ValueError("parameters must contain at least one DesignParameter")
        p_names = [p.name for p in self.parameters]
        if len(set(p_names)) != len(p_names):
            raise ValueError("parameter names must be unique")
        if self.unit not in (Unit.MM, Unit.CM, Unit.M):
            raise ValueError("unit must be mm, cm, or m")
        mesh_sz = _finite(self.mesh_size_mm, "mesh_size_mm", positive=True)
        min_ang = _finite(self.min_angle_deg, "min_angle_deg", positive=True)
        if min_ang >= 60.0:
            raise ValueError("min_angle_deg must be strictly less than 60")

        fixed_raw = dict(self.fixed_constants) if self.fixed_constants else {}
        fixed_clean = {
            str(k): _finite(v, f"fixed_constants.{k}")
            for k, v in sorted(fixed_raw.items())
        }
        overlap = set(p_names) & set(fixed_clean)
        if overlap:
            raise ValueError(
                f"fixed_constants overlap with parameters: {sorted(overlap)}"
            )

        roles_clean: dict[str, str] = {}
        for grp, role in sorted(self.boundary_roles.items()):
            if role not in SUPPORTED_BC_ROLES:
                raise ValueError(
                    f"unsupported boundary role {role!r} for group {grp!r}"
                )
            roles_clean[str(grp)] = str(role)
        if set(roles_clean) != set(self.expected_topology.boundary_groups):
            raise ValueError(
                "boundary_roles keys must match expected_topology.boundary_groups"
            )

        object.__setattr__(self, "mesh_size_mm", mesh_sz)
        object.__setattr__(self, "min_angle_deg", min_ang)
        object.__setattr__(self, "fixed_constants", fixed_clean)
        object.__setattr__(self, "boundary_roles", roles_clean)

    @property
    def role_map(self) -> BoundaryRoleMap:
        return BoundaryRoleMap(self.boundary_roles)

    def nominal_parameters(self) -> dict[str, float]:
        return {p.name: p.initial for p in self.parameters}

    def full_environment(self, params: Mapping[str, float]) -> dict[str, float]:
        env = dict(self.fixed_constants)
        for p in self.parameters:
            if p.name not in params:
                raise ParametricTopologyError(f"missing CAD parameter {p.name!r}")
            env[p.name] = _finite(params[p.name], f"param.{p.name}")
        return env

    def check_parameter_envelope(
        self, params: Mapping[str, float], *, enforce_box: bool = True
    ) -> tuple[bool, str]:
        """Validate parameter completeness, box bounds, and derived geometric bounds."""
        try:
            env = self.full_environment(params)
        except (ParametricTopologyError, ValueError) as exc:
            return False, str(exc)

        if enforce_box:
            for p in self.parameters:
                v = env[p.name]
                if v < p.lower - 1.0e-12 or v > p.upper + 1.0e-12:
                    return (
                        False,
                        f"parameter {p.name!r}={v:.6g} outside [{p.lower:.6g}, {p.upper:.6g}] {p.unit}",
                    )
        for bound in self.derived_bounds:
            ok, _, msg = bound.check(env)
            if not ok:
                return False, msg
        return True, "parameter envelope and geometric bounds satisfied"

    def document(self) -> dict[str, Any]:
        return {
            "schema": PARAMETRIC_CAD_SCHEMA,
            "name": self.name,
            "generator_id": self.generator_id,
            "template": self.template,
            "parameters": [p.document() for p in self.parameters],
            "derived_bounds": [b.document() for b in self.derived_bounds],
            "fixed_constants": dict(sorted(self.fixed_constants.items())),
            "expected_topology": self.expected_topology.document(),
            "boundary_roles": dict(sorted(self.boundary_roles.items())),
            "unit": self.unit.value,
            "mesh_size_mm": self.mesh_size_mm,
            "min_angle_deg": self.min_angle_deg,
            "mesh_policy": None
            if self.mesh_policy is None
            else self.mesh_policy.document(),
        }

    @property
    def spec_hash(self) -> str:
        return _sha256_doc(self.document())

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> ParametricCadSpec:
        if doc.get("schema") != PARAMETRIC_CAD_SCHEMA:
            raise ValueError(
                f"expected schema {PARAMETRIC_CAD_SCHEMA!r}, got {doc.get('schema')!r}"
            )
        params = tuple(
            DesignParameter(
                name=str(p["name"]),
                lower=float(p["lower"]),
                upper=float(p["upper"]),
                initial=float(p["initial"]),
                unit=str(p["unit"]),
            )
            for p in doc["parameters"]
        )
        bounds = tuple(
            GeometricBound.from_document(b) for b in doc.get("derived_bounds", ())
        )
        topo = ExpectedTopology.from_document(doc["expected_topology"])
        mp_raw = doc.get("mesh_policy")
        mesh_policy = None if mp_raw is None else MeshPolicy.from_document(mp_raw)
        return cls(
            name=str(doc["name"]),
            generator_id=str(doc["generator_id"]),
            template=str(doc["template"]),
            parameters=params,
            derived_bounds=bounds,
            fixed_constants={
                str(k): float(v)
                for k, v in dict(doc.get("fixed_constants", {})).items()
            },
            expected_topology=topo,
            boundary_roles={str(k): str(v) for k, v in doc["boundary_roles"].items()},
            unit=Unit(str(doc.get("unit", "mm"))),
            mesh_size_mm=float(doc.get("mesh_size_mm", 2.5)),
            min_angle_deg=float(doc.get("min_angle_deg", 12.0)),
            mesh_policy=mesh_policy,
        )


def default_notched_bracket_spec(
    *,
    plate_length_mm: float = 40.0,
    mesh_size_mm: float = 2.5,
) -> ParametricCadSpec:
    """Canonical M4 MVP parametric CAD spec: notched mounting bracket.

    Design variables: ``hole_radius``, ``notch_radius``, ``plate_width``.
    Derived bound guarantees >= 2.2 mm ligament on each side between central hole
    and top/bottom semicircular edge cutouts.
    """
    return ParametricCadSpec(
        name="notched-mounting-bracket",
        generator_id=BUILTIN_TEMPLATE_GENERATOR_ID,
        template="notched-mounting-bracket",
        parameters=(
            DesignParameter(
                "hole_radius", lower=2.0, upper=4.8, initial=3.4, unit="mm"
            ),
            DesignParameter(
                "notch_radius", lower=1.5, upper=4.2, initial=2.6, unit="mm"
            ),
            DesignParameter(
                "plate_width", lower=18.0, upper=24.0, initial=20.0, unit="mm"
            ),
        ),
        derived_bounds=(
            GeometricBound(
                name="min_ligament_mm",
                expression="plate_width / 2 - hole_radius - notch_radius",
                lower=2.2,
                unit="mm",
            ),
            GeometricBound(
                name="max_notch_span_mm",
                expression="plate_length / 2 - notch_radius",
                lower=6.0,
                unit="mm",
            ),
        ),
        fixed_constants={"plate_length": float(plate_length_mm)},
        expected_topology=ExpectedTopology(
            component_count=1,
            hole_count=1,
            boundary_groups=(
                "left",
                "right",
                "bottom",
                "top",
                "notch_bottom",
                "notch_top",
                "hole",
            ),
            boundary_kinds={
                "left": "line",
                "right": "line",
                "bottom": "line",
                "top": "line",
                "notch_bottom": "arc",
                "notch_top": "arc",
                "hole": "arc",
            },
        ),
        boundary_roles={
            "left": "dirichlet_xy",
            "right": "dirichlet_x",
            "bottom": "traction_free",
            "top": "traction_free",
            "notch_bottom": "traction_free",
            "notch_top": "traction_free",
            "hole": "traction_free",
        },
        unit=Unit.MM,
        mesh_size_mm=mesh_size_mm,
        min_angle_deg=12.0,
    )


def default_holed_plate_spec(
    *,
    plate_length_mm: float = 40.0,
    mesh_size_mm: float = 2.5,
) -> ParametricCadSpec:
    """Parametric holed plate spec (``hole_radius``, ``plate_width``)."""
    return ParametricCadSpec(
        name="parametric-holed-plate",
        generator_id=BUILTIN_TEMPLATE_GENERATOR_ID,
        template="holed-plate",
        parameters=(
            DesignParameter(
                "hole_radius", lower=2.0, upper=6.0, initial=4.0, unit="mm"
            ),
            DesignParameter(
                "plate_width", lower=16.0, upper=24.0, initial=20.0, unit="mm"
            ),
        ),
        derived_bounds=(
            GeometricBound(
                name="min_ligament_mm",
                expression="plate_width / 2 - hole_radius",
                lower=2.5,
                unit="mm",
            ),
        ),
        fixed_constants={"plate_length": float(plate_length_mm)},
        expected_topology=ExpectedTopology(
            component_count=1,
            hole_count=1,
            boundary_groups=("left", "right", "bottom", "top", "hole"),
            boundary_kinds={
                "left": "line",
                "right": "line",
                "bottom": "line",
                "top": "line",
                "hole": "arc",
            },
        ),
        boundary_roles={
            "left": "dirichlet_xy",
            "right": "dirichlet_x",
            "bottom": "traction_free",
            "top": "traction_free",
            "hole": "traction_free",
        },
        unit=Unit.MM,
        mesh_size_mm=mesh_size_mm,
        min_angle_deg=12.0,
    )


@dataclass(frozen=True, slots=True)
class GeneratedCadMesh:
    """Raw mesh + boundary kind metadata returned by a parametric CAD generator."""

    mesh: CanonicalMesh
    boundary_kinds: Mapping[str, str]
    step_path: Path | None = None


ParametricGeneratorFn = Callable[
    [ParametricCadSpec, Mapping[str, float], Path | None],
    GeneratedCadMesh,
]

_GENERATOR_REGISTRY: dict[str, ParametricGeneratorFn] = {}


def register_parametric_generator(generator_id: str, fn: ParametricGeneratorFn) -> None:
    """Register a custom parametric CAD generator callable under ``generator_id``."""
    if not isinstance(generator_id, str) or not generator_id.strip():
        raise ValueError("generator_id must be non-empty text")
    _GENERATOR_REGISTRY[generator_id] = fn


def get_parametric_generator(generator_id: str) -> ParametricGeneratorFn:
    """Resolve a built-in, registered, or entry-point parametric CAD generator."""
    if generator_id in _GENERATOR_REGISTRY:
        return _GENERATOR_REGISTRY[generator_id]
    for ep in metadata.entry_points().select(group=GENERATOR_ENTRY_POINT_GROUP):
        if ep.name == generator_id:
            loaded = ep.load()
            if not callable(loaded):
                raise TypeError(
                    f"entry point {generator_id!r} in {GENERATOR_ENTRY_POINT_GROUP!r} is not callable"
                )
            return cast(ParametricGeneratorFn, loaded)
    raise ValueError(
        f"unknown parametric CAD generator {generator_id!r}; registered: "
        f"{sorted(_GENERATOR_REGISTRY)}"
    )


def _triangle_min_angle_deg(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float
) -> float:
    a = math.hypot(bx - cx, by - cy)
    b = math.hypot(ax - cx, ay - cy)
    c = math.hypot(ax - bx, ay - by)
    if min(a, b, c) <= 1.0e-15:
        return 0.0
    cos_a = max(-1.0, min(1.0, (b * b + c * c - a * a) / (2.0 * b * c)))
    cos_b = max(-1.0, min(1.0, (a * a + c * c - b * b) / (2.0 * a * c)))
    cos_c = max(-1.0, min(1.0, (a * a + b * b - c * c) / (2.0 * a * b)))
    return math.degrees(min(math.acos(cos_a), math.acos(cos_b), math.acos(cos_c)))


def _mesh_area_and_min_angle(mesh: CanonicalMesh) -> tuple[float, float]:
    total_area = 0.0
    min_ang = 180.0
    for i, j, k in mesh.cells:
        ax, ay = mesh.nodes[i]
        bx, by = mesh.nodes[j]
        cx, cy = mesh.nodes[k]
        area = 0.5 * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        if area <= 0.0:
            raise ParametricTopologyError("mesh contains non-positive triangle area")
        total_area += area
        ang = _triangle_min_angle_deg(ax, ay, bx, by, cx, cy)
        if ang < min_ang:
            min_ang = ang
    return total_area, min_ang


def _classify_template_curve(
    template: str,
    curve_type: str,
    bbox: tuple[float, float, float, float, float, float],
    *,
    length: float,
    width: float,
) -> str:
    xmin, ymin, _, xmax, ymax, _ = bbox
    tol = 1.0e-3 * max(1.0, length, width)
    if curve_type == "Line":
        if abs(xmin) <= tol and abs(xmax) <= tol:
            return "left"
        if abs(xmin - length) <= tol and abs(xmax - length) <= tol:
            return "right"
        if abs(ymin) <= tol and abs(ymax) <= tol:
            return "bottom"
        if abs(ymin - width) <= tol and abs(ymax - width) <= tol:
            return "top"
    if curve_type in ("Circle", "Ellipse"):
        if template == "holed-plate":
            return "hole"
        if template == "notched-mounting-bracket":
            ymid = 0.5 * (ymin + ymax)
            if abs(ymin) <= tol and ymid < 0.45 * width:
                return "notch_bottom"
            if abs(ymax - width) <= tol and ymid > 0.55 * width:
                return "notch_top"
            return "hole"
    return "unclassified"


def _gmsh_occ_template_generator(
    spec: ParametricCadSpec,
    params: Mapping[str, float],
    step_out: Path | None = None,
) -> GeneratedCadMesh:
    """Built-in OpenCASCADE parametric 2D planar template generator via Gmsh."""
    if spec.template not in BUILTIN_TEMPLATES:
        raise ValueError(
            f"unknown built-in template {spec.template!r}; expected one of {BUILTIN_TEMPLATES}"
        )
    env = spec.full_environment(params)
    length = float(env.get("plate_length", 40.0))
    width = float(env["plate_width"])
    hole_r = float(env["hole_radius"])
    hole_x = float(env.get("hole_x", 0.5 * length))
    hole_y = float(env.get("hole_y", 0.5 * width))

    try:
        gmsh = __import__("gmsh")
    except (ImportError, OSError) as exc:
        raise ImportError(
            "parametric CAD generation requires formamech[cad] (gmsh)"
        ) from exc

    size_mm = (
        spec.mesh_policy.global_size_mm
        if spec.mesh_policy is not None
        else spec.mesh_size_mm
    )
    min_size_mm = (
        spec.mesh_policy.minimum_size_mm
        if spec.mesh_policy is not None
        else 0.45 * size_mm
    )

    gmsh.initialize([], readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size_mm)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", size_mm)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber(
            "Mesh.MinimumElementsPerTwoPi",
            float(
                spec.mesh_policy.curvature_points_per_circle
                if spec.mesh_policy is not None
                else 24
            ),
        )
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.add(f"param-{spec.template}")

        plate = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, length, width)
        tools: list[tuple[int, int]] = []
        if hole_r > 0.0:
            hole = gmsh.model.occ.addDisk(hole_x, hole_y, 0.0, hole_r, hole_r)
            tools.append((2, hole))
        if spec.template == "notched-mounting-bracket":
            notch_r = float(env["notch_radius"])
            if notch_r > 0.0:
                n_bot = gmsh.model.occ.addDisk(0.5 * length, 0.0, 0.0, notch_r, notch_r)
                n_top = gmsh.model.occ.addDisk(
                    0.5 * length, width, 0.0, notch_r, notch_r
                )
                tools.extend([(2, n_bot), (2, n_top)])

        if tools:
            out_dimtags, _ = gmsh.model.occ.cut([(2, plate)], tools)
        else:
            out_dimtags = [(2, plate)]
        gmsh.model.occ.synchronize()

        surfaces = [tag for dim, tag in out_dimtags if dim == 2]
        if len(surfaces) != 1:
            raise ParametricTopologyError(
                f"boolean cut produced {len(surfaces)} planar faces (expected 1)"
            )
        surface = int(surfaces[0])

        saved_step: Path | None = None
        if step_out is not None:
            saved_step = Path(step_out).resolve()
            saved_step.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(saved_step))

        # Classify every 1D boundary curve deterministically by geometry
        curves = sorted(
            abs(int(tag))
            for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False)
            if dim == 1
        )
        grouped_curves: dict[str, list[int]] = {}
        group_kinds: dict[str, str] = {}
        for c_tag in curves:
            ctype = str(gmsh.model.getType(1, c_tag))
            bbox = tuple(float(v) for v in gmsh.model.getBoundingBox(1, c_tag))
            g_name = _classify_template_curve(
                spec.template,
                ctype,
                cast(tuple[float, float, float, float, float, float], bbox),
                length=length,
                width=width,
            )
            grouped_curves.setdefault(g_name, []).append(c_tag)
            kind_str = {"Line": "line", "Circle": "arc", "Ellipse": "arc"}.get(
                ctype, "other"
            )
            group_kinds[g_name] = kind_str

        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        for g_name, c_tags in sorted(grouped_curves.items()):
            pg = gmsh.model.addPhysicalGroup(1, c_tags)
            gmsh.model.setPhysicalName(1, pg, g_name)

        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        tag_to_idx: dict[int, int] = {}
        raw_nodes: list[tuple[float, float]] = []
        for idx, ntag in enumerate(node_tags):
            tag_to_idx[int(ntag)] = idx
            raw_nodes.append((float(coords[3 * idx]), float(coords[3 * idx + 1])))

        elem_types, _, elem_nodes = gmsh.model.mesh.getElements(2, surface)
        tri_cells: list[tuple[int, int, int]] = []
        for etype, flat_nodes in zip(elem_types, elem_nodes, strict=True):
            if int(etype) != 2:
                continue
            for offset in range(0, len(flat_nodes), 3):
                tri_cells.append(
                    (
                        tag_to_idx[int(flat_nodes[offset])],
                        tag_to_idx[int(flat_nodes[offset + 1])],
                        tag_to_idx[int(flat_nodes[offset + 2])],
                    )
                )
        if not tri_cells:
            raise ParametricTopologyError("mesher produced zero linear triangles")

        boundary_facets: list[tuple[int, int]] = []
        boundary_tags: dict[str, list[int]] = {}
        for g_name, c_tags in sorted(grouped_curves.items()):
            f_ids: list[int] = []
            for c_tag in c_tags:
                b_types, _, b_nodes = gmsh.model.mesh.getElements(1, c_tag)
                for btype, bflat in zip(b_types, b_nodes, strict=True):
                    if int(btype) != 1:
                        continue
                    for offset in range(0, len(bflat), 2):
                        f_ids.append(len(boundary_facets))
                        boundary_facets.append(
                            (
                                tag_to_idx[int(bflat[offset])],
                                tag_to_idx[int(bflat[offset + 1])],
                            )
                        )
            boundary_tags[g_name] = f_ids

        # Prune unused nodes (e.g. OpenCASCADE disk centers emitted by Gmsh)
        used_indices = sorted(
            {nid for cell in tri_cells for nid in cell}
            | {nid for facet in boundary_facets for nid in facet}
        )
        remap = {old: new for new, old in enumerate(used_indices)}
        compact_nodes = tuple(raw_nodes[i] for i in used_indices)
        compact_cells = tuple((remap[a], remap[b], remap[c]) for a, b, c in tri_cells)
        compact_facets = tuple((remap[a], remap[b]) for a, b in boundary_facets)

        prov = {
            "generator_id": spec.generator_id,
            "template": spec.template,
            "spec_hash": spec.spec_hash,
            "parameters": canonical_json_bytes(
                {k: round(float(v), 12) for k, v in sorted(params.items())}
            ).decode("utf-8"),
        }
        mesh = canonicalize_mesh(
            MeshPayload(
                nodes=compact_nodes,
                cells=compact_cells,
                cell_type=CellType.TRIANGLE,
                boundary_facets=compact_facets,
                boundary_tags={k: tuple(v) for k, v in boundary_tags.items()},
                region_ids=tuple(1 for _ in compact_cells),
                unit=Unit.MM,
                source_provenance=prov,
            )
        )
        return GeneratedCadMesh(
            mesh=mesh,
            boundary_kinds=group_kinds,
            step_path=saved_step,
        )
    finally:
        gmsh.finalize()


register_parametric_generator(
    BUILTIN_TEMPLATE_GENERATOR_ID, _gmsh_occ_template_generator
)


@dataclass(frozen=True, slots=True)
class ParametricInstance:
    """Topology-verified parametric CAD mesh instance with canonical descriptor."""

    parameters: dict[str, float]
    mesh: CanonicalMesh
    topology: TopologySummary
    descriptor: GeometryDescriptor
    area_mm2: float
    min_angle_deg: float
    boundary_kinds: Mapping[str, str]
    step_path: Path | None = None

    def summary_document(self) -> dict[str, Any]:
        return {
            "parameters": dict(sorted(self.parameters.items())),
            "mesh_id": self.mesh.mesh_id,
            "topology_signature": self.topology.signature,
            "component_count": self.topology.component_count,
            "hole_count": self.topology.hole_count,
            "n_nodes": len(self.mesh.nodes),
            "n_cells": len(self.mesh.cells),
            "area_mm2": self.area_mm2,
            "min_angle_deg": self.min_angle_deg,
            "descriptor": self.descriptor.document(),
            "step_path": None if self.step_path is None else str(self.step_path),
        }


class TopologyStabilityGuard:
    """Fail-closed verifier ensuring topology & semantic-tag stability across CAD variants."""

    def __init__(self, spec: ParametricCadSpec) -> None:
        self.spec = spec
        self._tokenizer = GeometryTokenizer(subdivisions_per_facet=1)

    def validate_generated(
        self,
        generated: GeneratedCadMesh,
        params: Mapping[str, float],
    ) -> ParametricInstance:
        """Verify Euler topology, semantic tags, closed manifold loops, and mesh quality."""
        mesh = generated.mesh
        topo = topology_summary(mesh)
        expected = self.spec.expected_topology

        if topo.component_count != expected.component_count:
            raise ParametricTopologyError(
                f"topology component_count={topo.component_count} != expected "
                f"{expected.component_count} for params={dict(sorted(params.items()))}"
            )
        if topo.hole_count != expected.hole_count:
            raise ParametricTopologyError(
                f"topology hole_count={topo.hole_count} != expected "
                f"{expected.hole_count} (genus change) for params={dict(sorted(params.items()))}"
            )

        observed_tags = set(mesh.semantic_tags)
        expected_tags = set(expected.boundary_groups)
        if observed_tags != expected_tags:
            raise ParametricTopologyError(
                f"semantic boundary group mismatch: observed={sorted(observed_tags)}, "
                f"expected={sorted(expected_tags)}"
            )
        for grp, exp_kind in expected.boundary_kinds.items():
            obs_kind = generated.boundary_kinds.get(grp)
            if obs_kind is not None and obs_kind != exp_kind:
                raise ParametricTopologyError(
                    f"boundary {grp!r} kind changed to {obs_kind!r} (expected {exp_kind!r})"
                )

        area_mm2, min_ang = _mesh_area_and_min_angle(mesh)
        if min_ang < self.spec.min_angle_deg:
            raise ParametricTopologyError(
                f"mesh minimum angle {min_ang:.2f} deg below floor {self.spec.min_angle_deg:.2f} deg"
            )
        if (
            self.spec.mesh_policy is not None
            and len(mesh.cells) > self.spec.mesh_policy.maximum_triangles
        ):
            raise ParametricTopologyError(
                f"mesh triangle count {len(mesh.cells)} exceeds policy maximum "
                f"{self.spec.mesh_policy.maximum_triangles}"
            )

        try:
            descriptor = compute_geometry_descriptor(
                mesh, self.spec.role_map, tokenizer=self._tokenizer
            )
        except (UnsupportedTopologyError, ValueError) as exc:
            raise ParametricTopologyError(
                f"closed-manifold tokenization failed: {exc}"
            ) from exc

        return ParametricInstance(
            parameters={p.name: float(params[p.name]) for p in self.spec.parameters},
            mesh=mesh,
            topology=topo,
            descriptor=descriptor,
            area_mm2=area_mm2,
            min_angle_deg=min_ang,
            boundary_kinds=dict(generated.boundary_kinds),
            step_path=generated.step_path,
        )

    def generate(
        self,
        params: Mapping[str, float],
        *,
        enforce_box: bool = True,
        enforce_bounds: bool = True,
        step_out: Path | None = None,
    ) -> ParametricInstance:
        """Regenerate one CAD variant and enforce all topology stability checks."""
        if enforce_bounds:
            ok, msg = self.spec.check_parameter_envelope(
                params, enforce_box=enforce_box
            )
            if not ok:
                raise ParametricTopologyError(msg)
        generator = get_parametric_generator(self.spec.generator_id)
        generated = generator(self.spec, params, step_out)
        return self.validate_generated(generated, params)


def _halton_value(index: int, base: int) -> float:
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= float(base)
        r += f * float(i % base)
        i //= base
    return r


_PRIMES: Final[tuple[int, ...]] = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
)


@dataclass(frozen=True, slots=True)
class ParametricDoeReport:
    """Verified DOE sample set across a parametric CAD envelope."""

    spec_hash: str
    requested_count: int
    instances: tuple[ParametricInstance, ...]
    rejected: tuple[dict[str, Any], ...]
    family_kind: MeshFamilyKind

    def document(self) -> dict[str, Any]:
        return {
            "schema": PARAMETRIC_DOE_SCHEMA,
            "spec_hash": self.spec_hash,
            "requested_count": self.requested_count,
            "valid_count": len(self.instances),
            "rejected_count": len(self.rejected),
            "family_kind": self.family_kind.value,
            "instances": [inst.summary_document() for inst in self.instances],
            "rejected": list(self.rejected),
        }

    @property
    def report_hash(self) -> str:
        return _sha256_doc(self.document())


def sample_parametric_cad_doe(
    spec: ParametricCadSpec,
    *,
    count: int,
    seed_offset: int = 1,
    include_nominal: bool = True,
    max_attempts: int | None = None,
    step_dir: Path | None = None,
) -> ParametricDoeReport:
    """Deterministically sample ``count`` topology-verified CAD variants via Halton QMC."""
    if type(count) is not int or count < 2:
        raise ValueError("count must be an integer >= 2")
    if len(spec.parameters) > len(_PRIMES):
        raise ValueError(f"at most {len(_PRIMES)} parameters supported by Halton DOE")
    guard = TopologyStabilityGuard(spec)
    limit = max_attempts if max_attempts is not None else max(count * 12, 40)

    candidates: list[dict[str, float]] = []
    if include_nominal:
        candidates.append(spec.nominal_parameters())
    for k in range(limit):
        idx = seed_offset + k
        pt = {
            p.name: p.lower + _halton_value(idx, _PRIMES[dim]) * p.span
            for dim, p in enumerate(spec.parameters)
        }
        candidates.append(pt)

    instances: list[ParametricInstance] = []
    rejected: list[dict[str, Any]] = []
    for cand_idx, cand in enumerate(candidates):
        if len(instances) >= count:
            break
        step_path = (
            None
            if step_dir is None
            else Path(step_dir) / f"variant-{len(instances) + 1:03d}.step"
        )
        try:
            inst = guard.generate(cand, step_out=step_path)
            instances.append(inst)
        except ParametricTopologyError as exc:
            rejected.append(
                {
                    "candidate_index": cand_idx,
                    "parameters": dict(sorted(cand.items())),
                    "reason": str(exc),
                }
            )

    if len(instances) < count:
        raise ParametricTopologyError(
            f"only {len(instances)}/{count} valid parametric CAD variants found "
            f"within {limit} attempts"
        )

    family_kind = classify_mesh_family([inst.mesh for inst in instances])
    return ParametricDoeReport(
        spec_hash=spec.spec_hash,
        requested_count=count,
        instances=tuple(instances),
        rejected=tuple(rejected),
        family_kind=family_kind,
    )


def init_parametric_project(
    spec: ParametricCadSpec,
    output_dir: Path,
    *,
    verify_nominal: bool = True,
) -> dict[str, Any]:
    """Initialize a hash-bound parametric CAD project directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "parametric-spec.json"
    spec_path.write_bytes(canonical_json_bytes(spec.document()) + b"\n")

    nominal_summary: dict[str, Any] | None = None
    if verify_nominal:
        guard = TopologyStabilityGuard(spec)
        nominal = guard.generate(
            spec.nominal_parameters(),
            step_out=out / "nominal.step",
        )
        nominal_summary = nominal.summary_document()
        (out / "nominal-summary.json").write_bytes(
            canonical_json_bytes(nominal_summary) + b"\n"
        )

    manifest = {
        "schema": "formamech.parametric-project-v1",
        "name": spec.name,
        "spec_file": "parametric-spec.json",
        "spec_hash": spec.spec_hash,
        "nominal_verified": verify_nominal,
        "nominal": nominal_summary,
    }
    (out / "project.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_parametric_project(project_dir: Path) -> ParametricCadSpec:
    """Load and verify ``parametric-spec.json`` from a parametric CAD project directory."""
    import json

    proj = Path(project_dir)
    spec_path = proj / "parametric-spec.json"
    if not spec_path.is_file():
        raise FileNotFoundError(f"parametric-spec.json not found in {proj}")
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    spec = ParametricCadSpec.from_document(raw)
    manifest_path = proj / "project.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("spec_hash") != spec.spec_hash:
            raise ValueError("parametric project spec_hash mismatch (tampered spec)")
    return spec


__all__ = (
    "BUILTIN_TEMPLATES",
    "BUILTIN_TEMPLATE_GENERATOR_ID",
    "GENERATOR_ENTRY_POINT_GROUP",
    "PARAMETRIC_CAD_SCHEMA",
    "PARAMETRIC_DOE_SCHEMA",
    "ExpectedTopology",
    "GeneratedCadMesh",
    "GeometricBound",
    "ParametricCadSpec",
    "ParametricDoeReport",
    "ParametricGeneratorFn",
    "ParametricInstance",
    "ParametricTopologyError",
    "TopologyStabilityGuard",
    "default_holed_plate_spec",
    "default_notched_bracket_spec",
    "get_parametric_generator",
    "init_parametric_project",
    "load_parametric_project",
    "register_parametric_generator",
    "sample_parametric_cad_doe",
)
