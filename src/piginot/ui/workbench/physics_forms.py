"""Guided physics, boundary, material, and campaign-plan builders for the Workbench.

Produces exact ``formamech.cad-settings-v2`` JSON documents and invokes the
standard ``prepare_campaign`` + ``build_plan`` backend contract so zero manual
JSON editing is needed while preserving 100% CLI and CI reproducibility.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from piginot.cad.configuration import prepare_campaign
from piginot.cad.settings import SCHEMA_V2, load_settings
from piginot.compiler.adapters import resolve_configured_adapter
from piginot.compiler.config import CampaignConfig
from piginot.compiler.planning import build_plan
from piginot.compiler.records import write_plan
from piginot.ui.workbench.state import WorkbenchWorkspace

BoundaryKind = Literal["dirichlet", "symmetry", "traction_free"]
ParamRole = Literal["material", "load"]

_IDENT_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class MaterialPreset:
    """Curated material definition aligned with ``energy_formula.py``."""

    model: str
    display_name: str
    description: str
    parameters: tuple[tuple[str, str, float, float, str], ...]
    # Tuple entries: (property_name, default_param_name, lo, hi, unit)


MATERIAL_PRESETS: Final[Mapping[str, MaterialPreset]] = {
    "compressible_neo_hookean": MaterialPreset(
        model="compressible_neo_hookean",
        display_name="Compressible Neo-Hookean",
        description="Closed-form finite-strain isotropic hyperelastic elastomer/polymer.",
        parameters=(
            ("elastic_modulus", "E", 10.0, 10.0, "MPa"),
            ("poisson_ratio", "nu", 0.35, 0.35, "1"),
        ),
    ),
    "mooney_rivlin": MaterialPreset(
        model="mooney_rivlin",
        display_name="Mooney-Rivlin",
        description="Two-term isochoric + volumetric elastomer model (c10, c01, K).",
        parameters=(
            ("c10", "c10", 1.2, 1.2, "MPa"),
            ("c01", "c01", 0.3, 0.3, "MPa"),
            ("bulk_modulus", "K", 15.0, 15.0, "MPa"),
        ),
    ),
    "yeoh": MaterialPreset(
        model="yeoh",
        display_name="Yeoh (3-term)",
        description="Cubic I1bar polynomial for large-stretch rubbers.",
        parameters=(
            ("c10", "c10", 1.5, 1.5, "MPa"),
            ("c20", "c20", -0.1, -0.1, "MPa"),
            ("c30", "c30", 0.02, 0.02, "MPa"),
            ("bulk_modulus", "K", 20.0, 20.0, "MPa"),
        ),
    ),
    "gent": MaterialPreset(
        model="gent",
        display_name="Gent (Limiting Chain Extensibility)",
        description="Finite chain extensibility elastomer with locking parameter jm.",
        parameters=(
            ("shear_modulus", "mu", 3.0, 3.0, "MPa"),
            ("jm", "jm", 30.0, 30.0, "1"),
            ("bulk_modulus", "K", 20.0, 20.0, "MPa"),
        ),
    ),
    "hgo": MaterialPreset(
        model="hgo",
        display_name="HGO (2-family Fibre Reinforced)",
        description="Holzapfel-Gasser-Ogden symmetric two-family fibre composite.",
        parameters=(
            ("shear_modulus", "mu", 2.0, 2.0, "MPa"),
            ("bulk_modulus", "K", 20.0, 20.0, "MPa"),
            ("k1", "k1", 4.0, 4.0, "MPa"),
            ("k2", "k2", 2.0, 2.0, "1"),
            ("fibre_angle", "theta_f", 30.0, 30.0, "deg"),
        ),
    ),
    "hgo_single": MaterialPreset(
        model="hgo_single",
        display_name="HGO (Single-Family Fibre)",
        description="Single-family directional fibre reinforcement.",
        parameters=(
            ("shear_modulus", "mu", 2.0, 2.0, "MPa"),
            ("bulk_modulus", "K", 20.0, 20.0, "MPa"),
            ("k1", "k1", 4.0, 4.0, "MPa"),
            ("k2", "k2", 2.0, 2.0, "1"),
            ("fibre_angle", "theta_f", 0.0, 0.0, "deg"),
        ),
    ),
    "goh": MaterialPreset(
        model="goh",
        display_name="GOH (Dispersed Fibre Families)",
        description="Gasser-Ogden-Holzapfel dispersed fibre composite (kappa_d).",
        parameters=(
            ("shear_modulus", "mu", 2.0, 2.0, "MPa"),
            ("bulk_modulus", "K", 20.0, 20.0, "MPa"),
            ("k1", "k1", 4.0, 4.0, "MPa"),
            ("k2", "k2", 2.0, 2.0, "1"),
            ("fibre_angle", "theta_f", 30.0, 30.0, "deg"),
            ("kappa_d", "kappa_d", 0.1, 0.1, "1"),
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class BoundaryAssignment:
    """One explicit boundary condition assigned to a CAD semantic boundary group."""

    region: str
    name: str
    kind: BoundaryKind
    components: tuple[str, ...] = ()
    values: tuple[tuple[str, str], ...] = ()  # ((expression, unit), ...)

    def document(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "name": self.name,
            "region": self.region,
            "kind": self.kind,
        }
        if self.kind != "traction_free":
            base["components"] = list(self.components)
            base["values"] = [
                {"expression": expr, "unit": unit} for expr, unit in self.values
            ]
        return base

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> BoundaryAssignment:
        kind_raw = str(doc["kind"])
        if kind_raw not in ("dirichlet", "symmetry", "traction_free"):
            raise ValueError(f"unsupported boundary kind {kind_raw!r}")
        kind: BoundaryKind = (
            "dirichlet"
            if kind_raw == "dirichlet"
            else ("symmetry" if kind_raw == "symmetry" else "traction_free")
        )
        comps = tuple(str(c) for c in doc.get("components", ()))
        vals = tuple(
            (str(v["expression"]), str(v["unit"])) for v in doc.get("values", ())
        )
        return cls(
            region=str(doc["region"]),
            name=str(doc["name"]),
            kind=kind,
            components=comps,
            values=vals,
        )


@dataclass(frozen=True, slots=True)
class ParameterSpecDraft:
    """One material or load parameter with explicit unit and bounds."""

    name: str
    role: ParamRole
    lo: float
    hi: float
    unit: str

    def document(self) -> dict[str, Any]:
        lo_f = float(self.lo)
        hi_f = float(self.hi)
        if not (math.isfinite(lo_f) and math.isfinite(hi_f)) or hi_f < lo_f:
            raise ValueError(f"parameter {self.name!r}: invalid range [{lo_f}, {hi_f}]")
        return {
            "name": self.name,
            "role": self.role,
            "lo": lo_f,
            "hi": hi_f,
            "unit": self.unit,
            "distribution": "fixed" if hi_f == lo_f else "uniform",
        }


@dataclass(frozen=True, slots=True)
class WorkbenchPhysicsDraft:
    """Complete zero-JSON authoring draft mapped directly to ``cad-settings-v2``."""

    project_name: str
    material_model: str
    material_param_map: Mapping[str, str]
    parameters: tuple[ParameterSpecDraft, ...]
    boundaries: tuple[BoundaryAssignment, ...]
    displacement_scale: str = "cad_extent_x"
    qoi_region: str | None = None
    optimizer_steps: int = 50
    case_batch_size: int = 2
    checkpoint_interval: int = 25
    learning_rate: float = 1.0e-3
    final_learning_rate: float = 1.0e-4
    constant_steps: int = 0
    seed: int = 42
    fit_cases: int = 4
    screen_cases: int = 2
    u_tol: float = 0.05
    p_tol: float = 0.10
    peak_tol: float = 0.10
    quantile: float = 0.90


def _boundary_bbox(
    boundary_entry: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for seg in boundary_entry.get("segments_mm", ()):
        for pt in seg:
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), max(xs), min(ys), max(ys))


def default_boundaries_from_preview(
    preview: Mapping[str, Any],
) -> tuple[BoundaryAssignment, ...]:
    """Build sensible, fully editable boundary defaults from a CAD preview."""
    raw_boundaries = list(preview.get("boundaries", ()))
    if not raw_boundaries:
        raise ValueError("CAD preview contains no boundary groups")

    bbox = preview.get("bounding_box_mm", {})
    x_min = float(bbox.get("min", [0.0, 0.0])[0])
    x_max = float(bbox.get("max", [1.0, 1.0])[0])
    span_x = max(x_max - x_min, 1.0e-6)
    tol = 1.0e-4 * span_x

    left_region: str | None = None
    right_region: str | None = None
    for entry in raw_boundaries:
        if entry.get("kind") != "line":
            continue
        bx0, bx1, by0, by1 = _boundary_bbox(entry)
        if abs(bx0 - x_min) <= tol and abs(bx1 - x_min) <= tol and (by1 - by0) > tol:
            if left_region is None:
                left_region = str(entry["name"])
        elif abs(bx0 - x_max) <= tol and abs(bx1 - x_max) <= tol and (by1 - by0) > tol:
            if right_region is None:
                right_region = str(entry["name"])

    assignments: list[BoundaryAssignment] = []
    for idx, entry in enumerate(raw_boundaries):
        reg = str(entry["name"])
        if reg == left_region:
            assignments.append(
                BoundaryAssignment(
                    region=reg,
                    name="left_clamp",
                    kind="dirichlet",
                    components=("x", "y"),
                    values=(("0", "mm"), ("0", "mm")),
                )
            )
        elif reg == right_region:
            assignments.append(
                BoundaryAssignment(
                    region=reg,
                    name="right_load",
                    kind="dirichlet",
                    components=("x",),
                    values=(("ux", "mm"),),
                )
            )
        else:
            assignments.append(
                BoundaryAssignment(
                    region=reg,
                    name=f"free_{idx + 1}",
                    kind="traction_free",
                )
            )
    return tuple(assignments)


def default_draft_from_preview(
    preview: Mapping[str, Any],
    *,
    project_name: str = "cad-workbench-model",
    material_model: str = "compressible_neo_hookean",
) -> WorkbenchPhysicsDraft:
    """Create a complete valid ``WorkbenchPhysicsDraft`` from imported CAD geometry."""
    if material_model not in MATERIAL_PRESETS:
        raise ValueError(
            f"unknown material_model {material_model!r}; "
            f"choose from {sorted(MATERIAL_PRESETS)}"
        )
    preset = MATERIAL_PRESETS[material_model]
    bbox = preview.get("bounding_box_mm", {})
    x_min = float(bbox.get("min", [0.0, 0.0])[0])
    x_max = float(bbox.get("max", [10.0, 10.0])[0])
    extent_x = max(x_max - x_min, 1.0)

    params: list[ParameterSpecDraft] = [
        ParameterSpecDraft(name=pname, role="material", lo=lo, hi=hi, unit=unit)
        for _prop, pname, lo, hi, unit in preset.parameters
    ]
    params.append(
        ParameterSpecDraft(
            name="ux",
            role="load",
            lo=round(0.02 * extent_x, 4),
            hi=round(0.08 * extent_x, 4),
            unit="mm",
        )
    )
    param_map = {prop: pname for prop, pname, *_ in preset.parameters}
    boundaries = default_boundaries_from_preview(preview)
    free_regions = [b.region for b in boundaries if b.kind == "traction_free"]
    qoi_region = free_regions[0] if free_regions else boundaries[0].region

    return WorkbenchPhysicsDraft(
        project_name=project_name,
        material_model=material_model,
        material_param_map=param_map,
        parameters=tuple(params),
        boundaries=boundaries,
        qoi_region=qoi_region,
    )


def validate_boundaries_against_preview(
    preview: Mapping[str, Any],
    boundaries: Sequence[BoundaryAssignment],
) -> tuple[bool, tuple[str, ...]]:
    """Pre-flight validation of UI boundary assignments against ``preview.json``."""
    errors: list[str] = []
    preview_kinds = {
        str(entry["name"]): str(entry.get("kind", "other"))
        for entry in preview.get("boundaries", ())
    }
    assigned_regions = [b.region for b in boundaries]
    if len(set(assigned_regions)) != len(assigned_regions):
        errors.append("duplicate boundary region assignment detected")
    missing = sorted(set(preview_kinds) - set(assigned_regions))
    extra = sorted(set(assigned_regions) - set(preview_kinds))
    if missing:
        errors.append(f"unassigned CAD boundary groups: {', '.join(missing)}")
    if extra:
        errors.append(f"unknown CAD boundary groups: {', '.join(extra)}")

    seen_names: set[str] = set()
    essential_count = 0
    for bc in boundaries:
        if not _IDENT_RE.match(bc.name):
            errors.append(
                f"{bc.region}: condition name {bc.name!r} must be a valid identifier"
            )
        if bc.name in seen_names:
            errors.append(f"duplicate boundary condition name {bc.name!r}")
        seen_names.add(bc.name)

        if bc.kind != "traction_free":
            essential_count += 1
            bkind = preview_kinds.get(bc.region, "other")
            if bkind not in ("line", "arc"):
                errors.append(
                    f"{bc.region} ({bc.name}): essential condition requires a "
                    f"'line' or 'arc' boundary, found {bkind!r}"
                )
            if not bc.components or len(bc.components) != len(bc.values):
                errors.append(
                    f"{bc.region} ({bc.name}): components and values must be non-empty "
                    "and have matching length"
                )
            for comp in bc.components:
                if comp not in ("x", "y"):
                    errors.append(
                        f"{bc.region} ({bc.name}): unsupported component {comp!r}"
                    )

    if essential_count == 0:
        errors.append(
            "at least one dirichlet or symmetry boundary condition is required"
        )
    return len(errors) == 0, tuple(errors)


def build_settings_document(draft: WorkbenchPhysicsDraft) -> dict[str, Any]:
    """Emit a validated canonical ``formamech.cad-settings-v2`` settings document."""
    if not draft.project_name.strip():
        raise ValueError("project_name must be non-empty")
    if not any(p.hi > p.lo for p in draft.parameters):
        raise ValueError(
            "at least one material or load parameter must vary strictly (lo < hi)"
        )

    essential_regions = sorted(
        b.region for b in draft.boundaries if b.kind != "traction_free"
    )
    free_regions = [b.region for b in draft.boundaries if b.kind == "traction_free"]
    qoi_region = draft.qoi_region
    if qoi_region is None or qoi_region in essential_regions:
        qoi_region = free_regions[0] if free_regions else draft.boundaries[0].region
    exclude_regions = [r for r in essential_regions if r != qoi_region]

    doc: dict[str, Any] = {
        "schema": SCHEMA_V2,
        "name": draft.project_name.strip(),
        "parameters": [p.document() for p in draft.parameters],
        "material": {
            "model": draft.material_model,
            "parameters": dict(draft.material_param_map),
        },
        "boundary_conditions": [b.document() for b in draft.boundaries],
        "qois": [
            {
                "name": "free_stress_peak",
                "field": "P_served",
                "reduction": "max-frobenius",
                "region": qoi_region,
                "exclude_regions": exclude_regions,
            }
        ],
        "verification": {
            "quantile": float(draft.quantile),
            "tolerances": {
                "u_relative_l2": float(draft.u_tol),
                "p_relative_l2": float(draft.p_tol),
                "absolute_peak_error": float(draft.peak_tol),
            },
        },
        "displacement_scale": draft.displacement_scale,
        "training": {
            "optimizer_steps": int(draft.optimizer_steps),
            "case_batch_size": int(draft.case_batch_size),
            "checkpoint_interval": int(draft.checkpoint_interval),
            "learning_rate": float(draft.learning_rate),
            "final_learning_rate": float(draft.final_learning_rate),
            "constant_steps": int(draft.constant_steps),
            "seed": int(draft.seed),
        },
        "sampling": {
            "fit_cases": int(draft.fit_cases),
            "screen_cases": int(draft.screen_cases),
        },
    }
    # Validate strictly against the canonical CAD settings loader
    load_settings(json.dumps(doc).encode("utf-8"))
    return doc


def prepare_and_plan_workspace(
    workspace: WorkbenchWorkspace,
    draft: WorkbenchPhysicsDraft,
    *,
    execution_profile: bool = True,
) -> dict[str, Any]:
    """Write ``physics.json``, run ``cad prepare``, and emit ``preview/plan.json``."""
    preview = workspace.load_preview()
    if preview is None:
        raise ValueError("import a CAD file before preparing physics inputs")

    ok, errors = validate_boundaries_against_preview(preview, draft.boundaries)
    if not ok:
        raise ValueError("; ".join(errors))

    settings_doc = build_settings_document(draft)
    workspace.reset_prepared_inputs()
    from piginot.ui.workbench.state import _atomic_write_text, _file_sha256

    _atomic_write_text(
        workspace.physics_path,
        json.dumps(settings_doc, indent=2, sort_keys=True) + "\n",
    )
    prep_record = prepare_campaign(
        workspace.geometry_dir,
        workspace.physics_path,
        workspace.inputs_dir,
        execution_profile=execution_profile,
    )

    config_path = workspace.inputs_dir / "config.json"
    problem_path = workspace.inputs_dir / "problem.json"
    campaign = CampaignConfig.load(config_path)
    adapter = resolve_configured_adapter(campaign.adapter, config_path)
    plan = build_plan(
        problem_path,
        config_path,
        workspace.preview_dir,
        descriptor=adapter.descriptor,
    )
    write_plan(plan)

    state = workspace.load_state()
    state["project_name"] = draft.project_name
    state["material_model"] = draft.material_model
    state["boundaries"] = [b.document() for b in draft.boundaries]
    state["physics_sha256"] = _file_sha256(workspace.physics_path)
    state["geometry_sha256"] = _file_sha256(workspace.geometry_dir / "geometry.json")
    workspace.save_state(state)

    return {
        "settings": settings_doc,
        "preparation": prep_record,
        "plan": plan.document(),
    }


__all__ = (
    "MATERIAL_PRESETS",
    "BoundaryAssignment",
    "BoundaryKind",
    "MaterialPreset",
    "ParamRole",
    "ParameterSpecDraft",
    "WorkbenchPhysicsDraft",
    "build_settings_document",
    "default_boundaries_from_preview",
    "default_draft_from_preview",
    "prepare_and_plan_workspace",
    "validate_boundaries_against_preview",
)
