"""Parametric 3D CAD contracts, OpenCASCADE templates, and topology guards (V3-2).

Extends FormaMech's fail-closed parametric CAD system to 3D solid B-Rep parts:

* :class:`ParametricCadSpec3D` (``schema = "formamech.parametric-cad-3d-v1"``):
  immutable specification of 3D design parameters, safe algebraic
  :class:`~piginot.cad.parametric.GeometricBound` constraints (minimum wall /
  ligament thicknesses), :class:`ExpectedTopology3D` (Euler characteristic,
  face-kind inventory, minimum tet quality), and face selectors.
* Built-in 3D OpenCASCADE templates (``parametric-holed-block``,
  ``parametric-l-bracket``, ``parametric-mounting-lug``) plus pluggable
  external generators under ``formamech.cad.generators_3d``.
* :class:`TopologyStabilityGuard3D`: fail-closed verifier refusing parameter-box
  violations, algebraic ligament violations, genus/Euler breaks, missing face
  kinds, or empty clamped/loaded face groups.
* Deterministic Halton QMC 3D DOE sampler (:func:`sample_parametric_cad_3d_doe`).
"""

from __future__ import annotations

import hashlib
import math
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Final

from piginot.cad.importing_3d import (
    ImportedCad3D,
    classify_faces_by_bbox,
    generate_3d_l_bracket_step,
    generate_3d_mounting_lug_step,
    import_cad_3d,
)
from piginot.cad.parametric import GeometricBound
from piginot.contracts import Unit, canonical_json_bytes
from piginot.geometry.mesh3d import CellType3D
from piginot.operators.geometry_3d import (
    GeometryDescriptor3D,
    compute_geometry_descriptor_3d,
)
from piginot.optimization.problem import DesignParameter

PARAMETRIC_CAD_3D_SCHEMA: Final = "formamech.parametric-cad-3d-v1"
PARAMETRIC_DOE_3D_SCHEMA: Final = "formamech.parametric-doe-3d-v1"
GENERATOR_3D_ENTRY_POINT_GROUP: Final = "formamech.cad.generators_3d"
BUILTIN_3D_GENERATOR_ID: Final = "gmsh-occ-3d-template-v1"
BUILTIN_3D_TEMPLATES: Final[tuple[str, ...]] = (
    "parametric-holed-block",
    "parametric-l-bracket",
    "parametric-mounting-lug",
)


class ParametricTopology3DError(ValueError):
    """Fail-closed exception when a 3D CAD parameter point breaks topology or bounds."""


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _halton(index: int, base: int) -> float:
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= float(base)
        r += f * float(i % base)
        i //= base
    return r


_PRIMES: Final[tuple[int, ...]] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def generate_3d_holed_block_step(
    destination: Path | str,
    *,
    lx: float = 20.0,
    ly: float = 12.0,
    lz: float = 8.0,
    hole_radius: float = 2.4,
) -> Path:
    """Generate a compact 3D rectangular block with a central vertical through-hole."""
    import gmsh  # type: ignore[import-untyped]

    dest = Path(destination).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize([], readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("holed-block-3d")
        box = gmsh.model.occ.addBox(0.0, 0.0, 0.0, float(lx), float(ly), float(lz))
        cyl = gmsh.model.occ.addCylinder(
            0.5 * float(lx),
            0.5 * float(ly),
            -1.0,
            0.0,
            0.0,
            float(lz) + 2.0,
            float(hole_radius),
        )
        gmsh.model.occ.cut([(3, box)], [(3, cyl)])
        gmsh.model.occ.synchronize()
        gmsh.write(str(dest))
    finally:
        gmsh.finalize()
    return dest


@dataclass(frozen=True, slots=True)
class ExpectedTopology3D:
    """Declarative topological invariant for every valid 3D CAD variant in a family."""

    expected_euler_characteristic: int
    min_face_count: int = 6
    max_face_count: int = 64
    required_face_kinds: tuple[str, ...] = ("planar",)
    min_volume_mm3: float = 1.0e-3

    def __post_init__(self) -> None:
        if type(self.expected_euler_characteristic) is not int:
            raise ValueError("expected_euler_characteristic must be an int")
        if type(self.min_face_count) is not int or self.min_face_count < 4:
            raise ValueError("min_face_count must be >= 4")
        if (
            type(self.max_face_count) is not int
            or self.max_face_count < self.min_face_count
        ):
            raise ValueError("max_face_count must be >= min_face_count")
        if not math.isfinite(self.min_volume_mm3) or self.min_volume_mm3 <= 0.0:
            raise ValueError("min_volume_mm3 must be finite and positive")

    def document(self) -> dict[str, Any]:
        return {
            "expected_euler_characteristic": self.expected_euler_characteristic,
            "min_face_count": self.min_face_count,
            "max_face_count": self.max_face_count,
            "required_face_kinds": list(self.required_face_kinds),
            "min_volume_mm3": float(self.min_volume_mm3),
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> ExpectedTopology3D:
        return cls(
            expected_euler_characteristic=int(doc["expected_euler_characteristic"]),
            min_face_count=int(doc.get("min_face_count", 6)),
            max_face_count=int(doc.get("max_face_count", 64)),
            required_face_kinds=tuple(
                str(k) for k in doc.get("required_face_kinds", ("planar",))
            ),
            min_volume_mm3=float(doc.get("min_volume_mm3", 1.0e-3)),
        )


@dataclass(frozen=True, slots=True)
class ParametricCadSpec3D:
    """Content-addressed specification of a 3D parametric CAD solid family."""

    name: str
    template_name: str
    parameters: tuple[DesignParameter, ...]
    geometric_bounds: tuple[GeometricBound, ...]
    expected_topology: ExpectedTopology3D
    clamped_bbox_face: str = "x_min"
    loaded_bbox_face: str = "x_max"
    mesh_size_mm: float = 5.5
    element_type: CellType3D = CellType3D.TET10
    unit: Unit = Unit.MM
    generator_id: str = BUILTIN_3D_GENERATOR_ID
    fixed_constants: Mapping[str, float] = ()  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty text")
        if not self.parameters:
            raise ValueError("parameters must be non-empty")
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        if not math.isfinite(self.mesh_size_mm) or self.mesh_size_mm <= 0.0:
            raise ValueError("mesh_size_mm must be finite and positive")
        object.__setattr__(self, "element_type", CellType3D(self.element_type))
        object.__setattr__(self, "unit", Unit(self.unit))
        consts = (
            {str(k): float(v) for k, v in self.fixed_constants.items()}
            if isinstance(self.fixed_constants, Mapping)
            else {}
        )
        object.__setattr__(self, "fixed_constants", consts)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def nominal_parameters(self) -> dict[str, float]:
        return {p.name: float(p.initial) for p in self.parameters}

    def full_env(self, params: Mapping[str, float]) -> dict[str, float]:
        env = dict(self.fixed_constants)
        for p in self.parameters:
            if p.name not in params:
                raise ParametricTopology3DError(f"missing 3D CAD parameter {p.name!r}")
            val = float(params[p.name])
            if not math.isfinite(val):
                raise ParametricTopology3DError(
                    f"non-finite 3D CAD parameter {p.name!r}={val}"
                )
            env[p.name] = val
        return env

    @property
    def spec_hash(self) -> str:
        return _sha256_doc(self.document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": PARAMETRIC_CAD_3D_SCHEMA,
            "name": self.name,
            "template_name": self.template_name,
            "generator_id": self.generator_id,
            "parameters": [p.document() for p in self.parameters],
            "geometric_bounds": [
                {
                    "name": b.name,
                    "expression": b.expression,
                    "lower": b.lower,
                    "upper": b.upper,
                    "unit": b.unit,
                }
                for b in self.geometric_bounds
            ],
            "expected_topology": self.expected_topology.document(),
            "clamped_bbox_face": self.clamped_bbox_face,
            "loaded_bbox_face": self.loaded_bbox_face,
            "mesh_size_mm": float(self.mesh_size_mm),
            "element_type": self.element_type.value,
            "unit": self.unit.value,
            "fixed_constants": dict(sorted(self.fixed_constants.items())),
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> ParametricCadSpec3D:
        return cls(
            name=str(doc["name"]),
            template_name=str(doc["template_name"]),
            generator_id=str(doc.get("generator_id", BUILTIN_3D_GENERATOR_ID)),
            parameters=tuple(
                DesignParameter(
                    name=str(p["name"]),
                    lower=float(p["lower"]),
                    upper=float(p["upper"]),
                    initial=float(p["initial"]),
                    unit=str(p.get("unit", "mm")),
                )
                for p in doc["parameters"]
            ),
            geometric_bounds=tuple(
                GeometricBound(
                    name=str(b["name"]),
                    expression=str(b["expression"]),
                    lower=None if b.get("lower") is None else float(b["lower"]),
                    upper=None if b.get("upper") is None else float(b["upper"]),
                    unit=str(b.get("unit", "mm")),
                )
                for b in doc.get("geometric_bounds", ())
            ),
            expected_topology=ExpectedTopology3D.from_document(
                doc["expected_topology"]
            ),
            clamped_bbox_face=str(doc.get("clamped_bbox_face", "x_min")),
            loaded_bbox_face=str(doc.get("loaded_bbox_face", "x_max")),
            mesh_size_mm=float(doc.get("mesh_size_mm", 5.5)),
            element_type=CellType3D(str(doc.get("element_type", "tet10"))),
            unit=Unit(str(doc.get("unit", "mm"))),
            fixed_constants={
                str(k): float(v) for k, v in doc.get("fixed_constants", {}).items()
            },
        )


def _builtin_3d_generator(
    template_name: str, env: Mapping[str, float], destination: Path
) -> Path:
    if template_name == "parametric-holed-block":
        return generate_3d_holed_block_step(
            destination,
            lx=env["lx"],
            ly=env["ly"],
            lz=env["lz"],
            hole_radius=env["hole_radius"],
        )
    if template_name == "parametric-l-bracket":
        return generate_3d_l_bracket_step(
            destination,
            arm_x=env["arm_x"],
            arm_y=env["arm_y"],
            thickness=env["thickness"],
            depth=env["depth"],
            hole_radius=env["hole_radius"],
        )
    if template_name == "parametric-mounting-lug":
        return generate_3d_mounting_lug_step(
            destination,
            base_lx=env["base_lx"],
            base_ly=env["base_ly"],
            base_lz=env["base_lz"],
            boss_height=env["boss_height"],
            boss_radius=env["boss_radius"],
            boss_thickness=env["boss_thickness"],
            bore_radius=env["bore_radius"],
            bolt_radius=env["bolt_radius"],
        )
    raise ParametricTopology3DError(
        f"unknown built-in 3D CAD template {template_name!r}"
    )


Generator3DFn = Callable[[str, Mapping[str, float], Path], Path]
_CUSTOM_3D_GENERATORS: dict[str, Generator3DFn] = {}


def register_cad_3d_generator(generator_id: str, fn: Generator3DFn) -> None:
    _CUSTOM_3D_GENERATORS[generator_id] = fn


def resolve_cad_3d_generator(generator_id: str) -> Generator3DFn:
    if generator_id == BUILTIN_3D_GENERATOR_ID:
        return _builtin_3d_generator
    if generator_id in _CUSTOM_3D_GENERATORS:
        return _CUSTOM_3D_GENERATORS[generator_id]
    for ep in metadata.entry_points(group=GENERATOR_3D_ENTRY_POINT_GROUP):
        if ep.name == generator_id:
            loaded: Generator3DFn = ep.load()
            return loaded
    raise ParametricTopology3DError(f"unknown 3D CAD generator {generator_id!r}")


@dataclass(frozen=True, slots=True)
class ParametricInstance3D:
    """One topology-verified 3D CAD solid variant and its mesh + descriptors."""

    instance_id: str
    parameters: Mapping[str, float]
    imported: ImportedCad3D
    clamped_face_tags: tuple[str, ...]
    loaded_face_tags: tuple[str, ...]
    descriptor: GeometryDescriptor3D

    def summary_document(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "parameters": {k: float(v) for k, v in sorted(self.parameters.items())},
            "mesh_id": self.imported.mesh.mesh_id,
            "volume_mm3": float(self.imported.volume_mm3),
            "clamped_face_tags": list(self.clamped_face_tags),
            "loaded_face_tags": list(self.loaded_face_tags),
            "descriptor": self.descriptor.document(),
        }


class TopologyStabilityGuard3D:
    """Fail-closed verifier for every 3D parametric CAD variant."""

    def __init__(self, spec: ParametricCadSpec3D) -> None:
        self.spec = spec

    def validate_parameters(self, params: Mapping[str, float]) -> dict[str, float]:
        env = self.spec.full_env(params)
        for p in self.spec.parameters:
            val = env[p.name]
            if val < p.lower - 1.0e-12 or val > p.upper + 1.0e-12:
                raise ParametricTopology3DError(
                    f"3D CAD parameter {p.name!r}={val:.6g} outside [{p.lower:.6g}, {p.upper:.6g}]"
                )
        for bound in self.spec.geometric_bounds:
            ok, _, msg = bound.check(env)
            if not ok:
                raise ParametricTopology3DError(msg)
        return env

    def build_and_verify(
        self,
        params: Mapping[str, float],
        *,
        instance_id: str = "variant",
        step_path: Path | None = None,
    ) -> ParametricInstance3D:
        env = self.validate_parameters(params)
        gen_fn = resolve_cad_3d_generator(self.spec.generator_id)
        if step_path is not None:
            out_step = gen_fn(self.spec.template_name, env, step_path)
            imported = import_cad_3d(
                out_step,
                unit=self.spec.unit,
                mesh_size_mm=self.spec.mesh_size_mm,
                element_type=self.spec.element_type,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="formamech3d_") as tmp:
                out_step = gen_fn(
                    self.spec.template_name, env, Path(tmp) / f"{instance_id}.step"
                )
                imported = import_cad_3d(
                    out_step,
                    unit=self.spec.unit,
                    mesh_size_mm=self.spec.mesh_size_mm,
                    element_type=self.spec.element_type,
                )

        topo = self.spec.expected_topology
        if imported.volume_mm3 < topo.min_volume_mm3:
            raise ParametricTopology3DError(
                f"3D solid volume {imported.volume_mm3:.6g} < min {topo.min_volume_mm3:.6g}"
            )
        if not (topo.min_face_count <= len(imported.faces) <= topo.max_face_count):
            raise ParametricTopology3DError(
                f"3D face count {len(imported.faces)} outside "
                f"[{topo.min_face_count}, {topo.max_face_count}]"
            )
        face_kinds = {f.kind for f in imported.faces}
        missing_kinds = set(topo.required_face_kinds) - face_kinds
        if missing_kinds:
            raise ParametricTopology3DError(
                f"3D solid missing required face kinds: {sorted(missing_kinds)}"
            )

        bbox = classify_faces_by_bbox(imported)
        clamped = tuple(bbox.get(self.spec.clamped_bbox_face, ()))
        loaded = tuple(bbox.get(self.spec.loaded_bbox_face, ()))
        if not clamped:
            raise ParametricTopology3DError(
                f"no 3D faces matched clamped selector {self.spec.clamped_bbox_face!r}"
            )
        if not loaded:
            raise ParametricTopology3DError(
                f"no 3D faces matched loaded selector {self.spec.loaded_bbox_face!r}"
            )

        descriptor = compute_geometry_descriptor_3d(
            imported.mesh,
            clamped_face_tags=clamped,
            loaded_face_tags=loaded,
        )
        if (
            descriptor.surface_euler_characteristic
            != topo.expected_euler_characteristic
        ):
            raise ParametricTopology3DError(
                f"3D Euler characteristic mismatch: expected "
                f"{topo.expected_euler_characteristic}, got "
                f"{descriptor.surface_euler_characteristic}"
            )

        return ParametricInstance3D(
            instance_id=instance_id,
            parameters={p.name: float(env[p.name]) for p in self.spec.parameters},
            imported=imported,
            clamped_face_tags=clamped,
            loaded_face_tags=loaded,
            descriptor=descriptor,
        )


@dataclass(frozen=True, slots=True)
class Parametric3DDoeReport:
    """Deterministic Halton QMC 3D CAD DOE result with rejection audit."""

    spec_hash: str
    instances: tuple[ParametricInstance3D, ...]
    rejected_count: int
    seed_offset: int

    @property
    def report_hash(self) -> str:
        return _sha256_doc(self.document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": PARAMETRIC_DOE_3D_SCHEMA,
            "spec_hash": self.spec_hash,
            "seed_offset": self.seed_offset,
            "accepted_count": len(self.instances),
            "rejected_count": self.rejected_count,
            "instances": [inst.summary_document() for inst in self.instances],
        }


def sample_parametric_cad_3d_doe(
    spec: ParametricCadSpec3D,
    *,
    count: int,
    seed_offset: int = 1,
    max_attempts: int | None = None,
    step_dir: Path | None = None,
) -> Parametric3DDoeReport:
    """Sample ``count`` topology-verified 3D CAD variants via Halton QMC."""
    if type(count) is not int or count < 1:
        raise ValueError("count must be >= 1")
    guard = TopologyStabilityGuard3D(spec)
    limit = max(count * 8, 32) if max_attempts is None else max_attempts
    accepted: list[ParametricInstance3D] = []
    rejected = 0
    if step_dir is not None:
        step_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(limit):
        if len(accepted) >= count:
            break
        idx = seed_offset + attempt
        cand = {
            p.name: p.lower
            + _halton(idx, _PRIMES[d % len(_PRIMES)]) * (p.upper - p.lower)
            for d, p in enumerate(spec.parameters)
        }
        inst_id = f"{spec.name}-v{len(accepted):03d}"
        s_path = None if step_dir is None else step_dir / f"{inst_id}.step"
        try:
            inst = guard.build_and_verify(cand, instance_id=inst_id, step_path=s_path)
            accepted.append(inst)
        except ParametricTopology3DError:
            rejected += 1

    if len(accepted) < count:
        raise ParametricTopology3DError(
            f"only sampled {len(accepted)}/{count} valid 3D CAD variants in {limit} attempts"
        )
    return Parametric3DDoeReport(
        spec_hash=spec.spec_hash,
        instances=tuple(accepted),
        rejected_count=rejected,
        seed_offset=seed_offset,
    )


def default_holed_block_3d_spec(
    *, mesh_size_mm: float = 5.5, element_type: CellType3D = CellType3D.TET10
) -> ParametricCadSpec3D:
    """Built-in genus-1 3D holed block parametric family (chi = 0)."""
    return ParametricCadSpec3D(
        name="holed-block-3d",
        template_name="parametric-holed-block",
        parameters=(
            DesignParameter("lx", lower=16.0, upper=24.0, initial=20.0, unit="mm"),
            DesignParameter("ly", lower=10.0, upper=15.0, initial=12.0, unit="mm"),
            DesignParameter(
                "hole_radius", lower=1.6, upper=3.4, initial=2.4, unit="mm"
            ),
        ),
        geometric_bounds=(
            GeometricBound(
                name="min_wall_y",
                expression="ly / 2 - hole_radius",
                lower=1.5,
                unit="mm",
            ),
            GeometricBound(
                name="min_wall_x",
                expression="lx / 2 - hole_radius",
                lower=3.0,
                unit="mm",
            ),
        ),
        expected_topology=ExpectedTopology3D(
            expected_euler_characteristic=0,  # Genus-1 closed surface: chi = 2 - 2g = 0
            min_face_count=6,
            max_face_count=16,
            required_face_kinds=("planar", "cylindrical"),
            min_volume_mm3=500.0,
        ),
        clamped_bbox_face="x_min",
        loaded_bbox_face="x_max",
        mesh_size_mm=mesh_size_mm,
        element_type=element_type,
        fixed_constants={"lz": 7.0},
    )


def default_l_bracket_3d_spec(
    *, mesh_size_mm: float = 6.5, element_type: CellType3D = CellType3D.TET10
) -> ParametricCadSpec3D:
    """Built-in genus-1 3D cantilevered L-bracket family with through-hole (chi = 0)."""
    return ParametricCadSpec3D(
        name="l-bracket-3d",
        template_name="parametric-l-bracket",
        parameters=(
            DesignParameter("arm_x", lower=30.0, upper=42.0, initial=36.0, unit="mm"),
            DesignParameter(
                "thickness", lower=8.0, upper=12.0, initial=10.0, unit="mm"
            ),
            DesignParameter(
                "hole_radius", lower=2.0, upper=3.8, initial=3.0, unit="mm"
            ),
        ),
        geometric_bounds=(
            GeometricBound(
                name="min_depth_ligament",
                expression="depth / 2 - hole_radius",
                lower=1.8,
                unit="mm",
            ),
        ),
        expected_topology=ExpectedTopology3D(
            expected_euler_characteristic=0,
            min_face_count=8,
            max_face_count=24,
            required_face_kinds=("planar", "cylindrical"),
            min_volume_mm3=1_000.0,
        ),
        clamped_bbox_face="y_max",
        loaded_bbox_face="x_max",
        mesh_size_mm=mesh_size_mm,
        element_type=element_type,
        fixed_constants={"arm_y": 26.0, "depth": 12.0},
    )


def default_mounting_lug_3d_spec(
    *, mesh_size_mm: float = 7.0, element_type: CellType3D = CellType3D.TET10
) -> ParametricCadSpec3D:
    """Built-in genus-3 3D industrial mounting lug family (chi = -4)."""
    return ParametricCadSpec3D(
        name="mounting-lug-3d",
        template_name="parametric-mounting-lug",
        parameters=(
            DesignParameter(
                "boss_height", lower=14.0, upper=20.0, initial=16.0, unit="mm"
            ),
            DesignParameter(
                "boss_thickness", lower=8.0, upper=13.0, initial=10.0, unit="mm"
            ),
            DesignParameter(
                "bore_radius", lower=3.2, upper=5.4, initial=4.5, unit="mm"
            ),
        ),
        geometric_bounds=(
            GeometricBound(
                name="min_crown_wall",
                expression="boss_radius - bore_radius",
                lower=2.5,
                unit="mm",
            ),
        ),
        expected_topology=ExpectedTopology3D(
            expected_euler_characteristic=-4,  # 3 holes => genus 3 => chi = 2 - 2*3 = -4
            min_face_count=10,
            max_face_count=40,
            required_face_kinds=("planar", "cylindrical"),
            min_volume_mm3=2_000.0,
        ),
        clamped_bbox_face="z_min",
        loaded_bbox_face="z_max",
        mesh_size_mm=mesh_size_mm,
        element_type=element_type,
        fixed_constants={
            "base_lx": 42.0,
            "base_ly": 22.0,
            "base_lz": 7.0,
            "boss_radius": 9.0,
            "bolt_radius": 2.4,
        },
    )


__all__ = (
    "BUILTIN_3D_GENERATOR_ID",
    "BUILTIN_3D_TEMPLATES",
    "GENERATOR_3D_ENTRY_POINT_GROUP",
    "PARAMETRIC_CAD_3D_SCHEMA",
    "PARAMETRIC_DOE_3D_SCHEMA",
    "ExpectedTopology3D",
    "Parametric3DDoeReport",
    "ParametricCadSpec3D",
    "ParametricInstance3D",
    "ParametricTopology3DError",
    "TopologyStabilityGuard3D",
    "default_holed_block_3d_spec",
    "default_l_bracket_3d_spec",
    "default_mounting_lug_3d_spec",
    "generate_3d_holed_block_step",
    "register_cad_3d_generator",
    "resolve_cad_3d_generator",
    "sample_parametric_cad_3d_doe",
)
