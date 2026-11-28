"""Full built-in FormaMech capability catalog for AI agents (V4-0).

Exposes every v0.1–v0.3 FormaMech feature (and dynamically accepts V4-1..V4-5
extensions) through one strongly-typed, sandboxed, budget-governed registry.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from piginot.contracts.state_machine import registered_state_machines
from piginot.harness.contracts import CapabilityDomain, CapabilitySpec, CostClass
from piginot.harness.registry import CapabilityRegistry
from piginot.harness.session import HarnessContext


def _register_discover_capabilities(reg: CapabilityRegistry) -> None:
    def _list_caps(ctx: HarnessContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
        items = ctx.registry.list_capabilities(
            domain=args.get("domain"),
            cost_class=args.get("cost_class"),
            search=args.get("search"),
        )
        return {
            "count": len(items),
            "domains": ctx.registry.domains_summary(),
            "capabilities": items,
        }

    reg.register(
        CapabilitySpec(
            name="discover.capabilities",
            domain=CapabilityDomain.DISCOVER,
            summary="List registered FormaMech capabilities filtered by domain, cost class, or keyword.",
            description=(
                "Progressive discovery entrypoint returning compact capability descriptors "
                "across all engineering domains without bloating agent context."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "enum": [d.value for d in CapabilityDomain],
                    },
                    "cost_class": {
                        "type": "string",
                        "enum": [c.value for c in CostClass],
                    },
                    "search": {"type": "string"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("discovery", "schema"),
        ),
        _list_caps,
    )

    def _describe_cap(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        return ctx.registry.describe(str(args["name"]))

    reg.register(
        CapabilitySpec(
            name="discover.describe",
            domain=CapabilityDomain.DISCOVER,
            summary="Return full JSON Schema and epistemic metadata for one capability.",
            description="Inspect exact parameter schemas, defaults, cost class, and preconditions.",
            input_schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("discovery", "schema"),
        ),
        _describe_cap,
    )

    def _list_machines(
        _ctx: HarnessContext, _args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        all_specs = [spec.describe() for spec in registered_state_machines().values()]
        return {
            "count": len(all_specs),
            "machines": all_specs,
        }

    reg.register(
        CapabilitySpec(
            name="discover.state_machines",
            domain=CapabilityDomain.DISCOVER,
            summary="Inspect all registered FormaMech explicit finite state machines.",
            description="Returns formal transition tables and terminal states for every lifecycle.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("fsm", "governance"),
        ),
        _list_machines,
    )

    def _workspace_tree(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        sub = ctx.resolve_path(str(args.get("subdir", ".")))
        max_entries = int(args.get("max_entries", 120))
        files: list[dict[str, Any]] = []
        if sub.is_dir():
            for path in sorted(sub.rglob("*")):
                if len(files) >= max_entries:
                    break
                if ".git" in path.parts or "__pycache__" in path.parts:
                    continue
                if path.is_file():
                    files.append(
                        {
                            "path": ctx.relative_path(path),
                            "bytes": path.stat().st_size,
                        }
                    )
        return {
            "workspace_root": str(ctx.workspace_root),
            "file_count": len(files),
            "files": files,
        }

    reg.register(
        CapabilitySpec(
            name="discover.workspace_tree",
            domain=CapabilityDomain.DISCOVER,
            summary="List files inside the sandboxed agent workspace.",
            description="Returns relative paths and file sizes strictly inside workspace_root.",
            input_schema={
                "type": "object",
                "properties": {
                    "subdir": {"type": "string", "default": "."},
                    "max_entries": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("workspace", "files"),
        ),
        _workspace_tree,
    )

    def _list_backends(
        _ctx: HarnessContext, _args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.cad.connectors import CadConnectorRegistry
        from piginot.fem.external_solvers import list_external_oracles
        from piginot.fem.linear_solvers import LinearSolverMethod, is_pyamg_available
        from piginot.fem.registry import available_backends

        return {
            "fem_backends": list(available_backends()),
            "external_oracles": list(list_external_oracles()),
            "cad_connectors": CadConnectorRegistry().list_connectors(),
            "linear_solvers": [m.value for m in LinearSolverMethod],
            "pyamg_available": is_pyamg_available(),
        }

    reg.register(
        CapabilitySpec(
            name="discover.backends",
            domain=CapabilityDomain.DISCOVER,
            summary="List installed 2D/3D FEM backends and sparse iterative linear solvers.",
            description="Reports discoverable entry-point FEM solvers and optional accelerator availability.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("fem", "solvers"),
        ),
        _list_backends,
    )


def _register_cad_capabilities(reg: CapabilityRegistry) -> None:
    def _gen_sample_2d(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.contracts import Unit
        from piginot.ui.workbench.state import WorkbenchWorkspace

        ws_path = ctx.resolve_path(str(args.get("workspace_dir", "study_2d")))
        ws = WorkbenchWorkspace(ws_path)
        preview = ws.import_sample(
            mesh_size_mm=float(args.get("mesh_size_mm", 4.0)),
        )
        ctx.record_artifact(ws.geometry_dir / "preview.json")
        return {
            "workspace_dir": ctx.relative_path(ws_path),
            "node_count": preview["mesh"]["node_count"],
            "triangle_count": preview["mesh"]["triangle_count"],
            "boundaries": [b["name"] for b in preview["boundaries"]],
            "unit": Unit.MM.value,
        }

    reg.register(
        CapabilitySpec(
            name="cad.generate_sample_2d",
            domain=CapabilityDomain.CAD,
            summary="Generate and mesh a built-in 2D OpenCASCADE bracket STEP study.",
            description="Creates a canonical 2D CAD workspace with mesh and semantic boundary tags.",
            input_schema={
                "type": "object",
                "properties": {
                    "workspace_dir": {"type": "string", "default": "study_2d"},
                    "mesh_size_mm": {"type": "number", "minimum": 0.5, "maximum": 50.0},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("cad", "2d", "step"),
        ),
        _gen_sample_2d,
    )

    def _gen_sample_3d(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.cad.importing_3d import (
            generate_3d_box_step,
            generate_3d_l_bracket_step,
            generate_3d_mounting_lug_step,
        )
        from piginot.geometry.mesh3d import CellType3D
        from piginot.ui.workbench_3d.state_3d import Workbench3DWorkspace

        ws_path = ctx.resolve_path(str(args.get("workspace_dir", "study_3d")))
        template = str(args.get("template", "box"))
        mesh_size = float(args.get("mesh_size_mm", 7.0))
        ctype = CellType3D(str(args.get("element_type", "tet10")))
        ws = Workbench3DWorkspace(ws_path)
        ws._ensure_dirs()
        step_file = ws.geometry_dir / f"{template}.step"
        if template == "box":
            generate_3d_box_step(
                step_file,
                lx=float(args.get("lx", 20.0)),
                ly=float(args.get("ly", 8.0)),
                lz=float(args.get("lz", 8.0)),
            )
        elif template == "l-bracket":
            generate_3d_l_bracket_step(step_file)
        elif template == "mounting-lug":
            generate_3d_mounting_lug_step(step_file)
        else:
            raise ValueError(f"unknown 3D sample template {template!r}")
        preview = ws.import_step_path(
            step_file, mesh_size_mm=mesh_size, element_type=ctype
        )
        n_dofs = int(preview["node_count"]) * 3
        ctx.check_dofs(n_dofs)
        ctx.record_artifact(step_file)
        ctx.record_artifact(ws.preview_path)
        ctx.record_artifact(ws.mesh_path)
        ctx.state_kv["last_3d_workspace"] = ctx.relative_path(ws_path)
        return {
            "workspace_dir": ctx.relative_path(ws_path),
            "template": template,
            "geometry_id": preview["geometry_id"],
            "node_count": preview["node_count"],
            "cell_count": preview["cell_count"],
            "dof_count": n_dofs,
            "face_tags": [fg["face_tag"] for fg in preview["face_groups"]],
        }

    reg.register(
        CapabilitySpec(
            name="cad.generate_sample_3d",
            domain=CapabilityDomain.CAD,
            summary="Generate and mesh a 3D OpenCASCADE STEP solid (box, l-bracket, or mounting-lug).",
            description="Persists a canonical TET4/TET10 3D workspace with face tags and default BCs.",
            input_schema={
                "type": "object",
                "properties": {
                    "workspace_dir": {"type": "string", "default": "study_3d"},
                    "template": {
                        "type": "string",
                        "enum": ["box", "l-bracket", "mounting-lug"],
                        "default": "box",
                    },
                    "mesh_size_mm": {"type": "number", "minimum": 1.0, "maximum": 50.0},
                    "element_type": {
                        "type": "string",
                        "enum": ["tet4", "tet10"],
                        "default": "tet10",
                    },
                    "lx": {"type": "number", "minimum": 1.0, "maximum": 500.0},
                    "ly": {"type": "number", "minimum": 1.0, "maximum": 500.0},
                    "lz": {"type": "number", "minimum": 1.0, "maximum": 500.0},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("cad", "3d", "step"),
        ),
        _gen_sample_3d,
    )

    def _import_3d(ctx: HarnessContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
        from piginot.geometry.mesh3d import CellType3D
        from piginot.ui.workbench_3d.state_3d import Workbench3DWorkspace

        step_path = ctx.resolve_path(str(args["step_path"]))
        ws_path = ctx.resolve_path(str(args.get("workspace_dir", "study_3d")))
        mesh_size = float(args.get("mesh_size_mm", 6.0))
        ctype = CellType3D(str(args.get("element_type", "tet10")))
        ws = Workbench3DWorkspace(ws_path)
        preview = ws.import_step_path(
            step_path, mesh_size_mm=mesh_size, element_type=ctype
        )
        n_dofs = int(preview["node_count"]) * 3
        ctx.check_dofs(n_dofs)
        ctx.record_artifact(ws.preview_path)
        ctx.record_artifact(ws.mesh_path)
        ctx.state_kv["last_3d_workspace"] = ctx.relative_path(ws_path)
        return {
            "workspace_dir": ctx.relative_path(ws_path),
            "geometry_id": preview["geometry_id"],
            "node_count": preview["node_count"],
            "cell_count": preview["cell_count"],
            "dof_count": n_dofs,
            "face_tags": [fg["face_tag"] for fg in preview["face_groups"]],
        }

    reg.register(
        CapabilitySpec(
            name="cad.import_3d",
            domain=CapabilityDomain.CAD,
            summary="Import and mesh an arbitrary 3D STEP/STP solid inside the workspace.",
            description="Enforces DOF ceiling and persists canonical 3D mesh + B-Rep face tags.",
            input_schema={
                "type": "object",
                "required": ["step_path"],
                "properties": {
                    "step_path": {"type": "string"},
                    "workspace_dir": {"type": "string", "default": "study_3d"},
                    "mesh_size_mm": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 100.0,
                    },
                    "element_type": {
                        "type": "string",
                        "enum": ["tet4", "tet10"],
                        "default": "tet10",
                    },
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("cad", "3d", "step"),
        ),
        _import_3d,
    )

    def _parametric_3d_sample(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.cad.parametric_3d import (
            default_holed_block_3d_spec,
            default_l_bracket_3d_spec,
            default_mounting_lug_3d_spec,
            sample_parametric_cad_3d_doe,
        )
        from piginot.contracts import canonical_json_bytes

        tpl = str(args.get("template", "parametric-holed-block"))
        mesh_size = float(args.get("mesh_size_mm", 8.0))
        count = int(args.get("count", 3))
        out_dir = ctx.resolve_path(str(args.get("output_dir", "parametric_3d_doe")))
        out_dir.mkdir(parents=True, exist_ok=True)
        if tpl == "parametric-holed-block":
            spec = default_holed_block_3d_spec(mesh_size_mm=mesh_size)
        elif tpl == "parametric-l-bracket":
            spec = default_l_bracket_3d_spec(mesh_size_mm=mesh_size)
        elif tpl == "parametric-mounting-lug":
            spec = default_mounting_lug_3d_spec(mesh_size_mm=mesh_size)
        else:
            raise ValueError(f"unknown parametric 3D template {tpl!r}")
        doe = sample_parametric_cad_3d_doe(
            spec, count=count, seed_offset=int(args.get("seed_offset", 1))
        )
        manifest = out_dir / "doe-3d-manifest.json"
        doc = {**doe.document(), "report_hash": doe.report_hash}
        manifest.write_bytes(canonical_json_bytes(doc) + b"\n")
        ctx.record_artifact(manifest)
        ctx.set_receipt(doe.report_hash)
        return doc

    reg.register(
        CapabilitySpec(
            name="cad.parametric_3d_sample",
            domain=CapabilityDomain.CAD,
            summary="Generate a topology-guarded Halton QMC 3D parametric CAD DOE.",
            description="Verifies Euler characteristic / ligament bounds across all sampled 3D variants.",
            input_schema={
                "type": "object",
                "properties": {
                    "template": {
                        "type": "string",
                        "enum": [
                            "parametric-holed-block",
                            "parametric-l-bracket",
                            "parametric-mounting-lug",
                        ],
                        "default": "parametric-holed-block",
                    },
                    "count": {"type": "integer", "minimum": 2, "maximum": 64},
                    "mesh_size_mm": {"type": "number", "minimum": 2.0, "maximum": 30.0},
                    "seed_offset": {"type": "integer", "minimum": 0, "maximum": 10000},
                    "output_dir": {"type": "string", "default": "parametric_3d_doe"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("cad", "3d", "parametric", "doe"),
        ),
        _parametric_3d_sample,
    )

    def _connector_generate(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.cad.connectors import CadConnectorRegistry

        conn = CadConnectorRegistry().get(
            str(args.get("connector", "opencascade-builtin"))
        )
        out_step = ctx.resolve_path(str(args["output_step"]))
        params = {str(k): float(v) for k, v in dict(args.get("parameters", {})).items()}
        res = conn.generate_step(str(args["model_ref"]), params, out_step)
        ctx.record_artifact(out_step)
        ctx.set_receipt(res.step_sha256)
        return {**res.document(), "step_path": ctx.relative_path(out_step)}

    reg.register(
        CapabilitySpec(
            name="cad.connector_generate",
            domain=CapabilityDomain.CAD,
            summary="Generate a live parametric STEP part through a registered CAD connector.",
            description="Supports opencascade-builtin, cadquery, freecad-cli, and onshape-rest.",
            input_schema={
                "type": "object",
                "required": ["model_ref", "output_step"],
                "properties": {
                    "connector": {"type": "string", "default": "opencascade-builtin"},
                    "model_ref": {"type": "string"},
                    "parameters": {"type": "object"},
                    "output_step": {"type": "string"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("cad", "connector", "step"),
        ),
        _connector_generate,
    )


def _register_fem_and_multiphysics_capabilities(reg: CapabilityRegistry) -> None:
    def _solve_3d_solid(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.fem.linear_solvers import LinearSolverConfig, LinearSolverMethod
        from piginot.ui.workbench_3d.state_3d import Workbench3DWorkspace

        ws_path = ctx.resolve_path(
            str(
                args.get(
                    "workspace_dir", ctx.state_kv.get("last_3d_workspace", "study_3d")
                )
            )
        )
        ws = Workbench3DWorkspace(ws_path)
        mesh = ws.load_mesh()
        if mesh is None:
            raise FileNotFoundError(f"no 3D mesh in workspace {ws_path}")
        ctx.check_dofs(len(mesh.nodes) * 3)
        ctx.consume_fem_solves(1)
        ls_cfg = LinearSolverConfig(
            method=LinearSolverMethod(str(args.get("linear_solver", "auto")))
        )
        res = ws.solve_preview_fem(
            youngs_modulus_mpa=float(args.get("youngs_modulus_mpa", 100.0)),
            poisson_ratio=float(args.get("poisson_ratio", 0.30)),
            load_steps=int(args.get("load_steps", 2)),
            linear_solver=ls_cfg,
        )
        # Save nodal fields for downstream export capabilities
        res_dir = ws.results_dir
        res_dir.mkdir(parents=True, exist_ok=True)
        npz_path = res_dir / "latest_fem_3d.npz"
        np.savez(
            npz_path,
            nodal_displacement=res.nodal_displacement,
            nodal_von_mises_mpa=res.nodal_von_mises_mpa,
        )
        ctx.record_artifact(npz_path)
        ctx.set_receipt(res.result_hash)
        return {
            **res.summary_document(),
            "fields_npz": ctx.relative_path(npz_path),
        }

    reg.register(
        CapabilitySpec(
            name="fem.solve_3d_solid",
            domain=CapabilityDomain.FEM,
            summary="Solve 3D nonlinear finite-strain hyperelastic FEM on an imported 3D workspace.",
            description=(
                "Runs TorchFemBackend3D with exact consistent tangents, det(F)>0 checks, "
                "and saves nodal displacement and von Mises fields for VTU/GLB export."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "workspace_dir": {"type": "string"},
                    "youngs_modulus_mpa": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 10000.0,
                    },
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "load_steps": {"type": "integer", "minimum": 1, "maximum": 20},
                    "linear_solver": {
                        "type": "string",
                        "enum": [
                            "auto",
                            "direct",
                            "pcg-jacobi",
                            "pcg-ssor",
                            "pcg-ilu",
                            "pcg-amg",
                            "gmres-ilu",
                        ],
                        "default": "auto",
                    },
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("fem", "3d", "hyperelastic"),
        ),
        _solve_3d_solid,
    )

    def _solve_stateful_j2_3d(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.ui.workbench_3d.stateful_editor import (
            run_stateful_3d_cyclic_preview,
        )

        out_dir = ctx.resolve_path(str(args.get("output_dir", "stateful_3d")))
        ux_seq = [float(v) for v in args.get("ux_sequence_mm", [0.05, 0.25, 0.10])]
        ctx.consume_fem_solves(len(ux_seq))
        traj, summary = run_stateful_3d_cyclic_preview(
            out_dir,
            ux_sequence_mm=ux_seq,
            model_kind=str(args.get("model_kind", "j2_combined")),
            youngs_modulus_mpa=float(args.get("youngs_modulus_mpa", 20_000.0)),
            yield_stress_mpa=float(args.get("yield_stress_mpa", 250.0)),
        )
        doc = traj.document()
        ctx.set_receipt(traj.steps[-1].state_blake2b if traj.steps else "")
        return {
            **doc,
            "all_converged": summary.all_converged,
            "total_dissipation": summary.total_dissipation,
        }

    reg.register(
        CapabilitySpec(
            name="multiphysics.solve_stateful_j2_3d",
            domain=CapabilityDomain.MULTIPHYSICS,
            summary="Run incremental 3D cyclic J2 combined kinematic or finite-strain plasticity.",
            description="Verifies non-negative plastic dissipation and emits BLAKE2b state digests.",
            input_schema={
                "type": "object",
                "properties": {
                    "output_dir": {"type": "string", "default": "stateful_3d"},
                    "model_kind": {
                        "type": "string",
                        "enum": ["j2_combined", "finite_strain_j2"],
                        "default": "j2_combined",
                    },
                    "ux_sequence_mm": {"type": "array"},
                    "youngs_modulus_mpa": {"type": "number", "minimum": 10.0},
                    "yield_stress_mpa": {"type": "number", "minimum": 1.0},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=3,
            tags=("plasticity", "3d", "stateful"),
        ),
        _solve_stateful_j2_3d,
    )

    def _solve_thermo_3d(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.contracts import MaterialSpec, Quantity, Unit
        from piginot.fem.backend_3d import DirichletBC3D
        from piginot.fem.thermo_mechanical_3d import (
            TemperatureBC3D,
            ThermalMaterialSpec,
            solve_3d_thermo_mechanical,
        )
        from piginot.ui.workbench_3d.state_3d import Workbench3DWorkspace

        ws_path = ctx.resolve_path(
            str(
                args.get(
                    "workspace_dir", ctx.state_kv.get("last_3d_workspace", "study_3d")
                )
            )
        )
        ws = Workbench3DWorkspace(ws_path)
        mesh = ws.load_mesh()
        preview = ws.load_preview()
        if mesh is None or preview is None:
            raise FileNotFoundError(f"no 3D mesh in workspace {ws_path}")
        ctx.check_dofs(len(mesh.nodes) * 3)
        ctx.consume_fem_solves(1)

        bbox = preview["bbox_classification"]
        mat = MaterialSpec(
            name="thermo-3d-nh",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(
                    float(args.get("youngs_modulus_mpa", 1_000.0)), Unit.MPA
                ).to_canonical(),
                "poisson_ratio": Quantity(
                    float(args.get("poisson_ratio", 0.30)), Unit.ONE
                ).to_canonical(),
            },
        )
        thermal = ThermalMaterialSpec(
            conductivity_w_per_mm_k=float(args.get("conductivity_w_per_mm_k", 0.05)),
            density_t_per_mm3=float(args.get("density_t_per_mm3", 7.8e-9)),
            specific_heat_mj_per_t_k=float(args.get("specific_heat_mj_per_t_k", 5.0e8)),
            thermal_expansion_per_k=float(args.get("thermal_expansion_per_k", 1.2e-5)),
            reference_temp_k=float(args.get("reference_temp_k", 293.15)),
        )
        t_hot = float(args.get("hot_temp_k", 393.15))
        t_cold = float(args.get("cold_temp_k", 293.15))
        res = solve_3d_thermo_mechanical(
            mesh,
            mat,
            thermal,
            dirichlet_bcs=(
                DirichletBC3D(
                    "fix_x_min",
                    tuple(bbox["x_min"]),
                    ("x", "y", "z"),
                    (0.0, 0.0, 0.0),
                ),
            ),
            temperature_bcs=(
                TemperatureBC3D("cold", tuple(bbox["x_min"]), t_cold),
                TemperatureBC3D("hot", tuple(bbox["x_max"]), t_hot),
            ),
        )
        return {
            "converged": res.converged,
            "min_temp_k": float(np.min(res.nodal_temperature_k)),
            "max_temp_k": float(np.max(res.nodal_temperature_k)),
            "max_displacement_mm": float(
                np.max(np.linalg.norm(res.nodal_displacement, axis=1))
            ),
            "peak_von_mises_mpa": float(np.max(res.nodal_von_mises_mpa)),
            "min_det_f": float(res.min_det_f),
        }

    reg.register(
        CapabilitySpec(
            name="multiphysics.solve_thermo_mechanical_3d",
            domain=CapabilityDomain.MULTIPHYSICS,
            summary="Solve coupled 3D steady heat conduction + multiplicative F = F_m F_th mechanics.",
            description="Solves thermal field and coupled finite-strain thermal expansion stresses.",
            input_schema={
                "type": "object",
                "properties": {
                    "workspace_dir": {"type": "string"},
                    "youngs_modulus_mpa": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 10000.0,
                    },
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "cold_temp_k": {"type": "number", "minimum": 1.0},
                    "hot_temp_k": {"type": "number", "minimum": 1.0},
                    "thermal_expansion_per_k": {"type": "number", "minimum": 0.0},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("thermo-mechanical", "heat", "3d"),
        ),
        _solve_thermo_3d,
    )

    def _solve_modal_3d(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.contracts import MaterialSpec, Quantity, Unit
        from piginot.fem.backend_3d import DirichletBC3D
        from piginot.fem.modal_3d import solve_3d_modal_analysis
        from piginot.ui.workbench_3d.state_3d import Workbench3DWorkspace

        ws_path = ctx.resolve_path(
            str(
                args.get(
                    "workspace_dir", ctx.state_kv.get("last_3d_workspace", "study_3d")
                )
            )
        )
        ws = Workbench3DWorkspace(ws_path)
        mesh = ws.load_mesh()
        preview = ws.load_preview()
        if mesh is None or preview is None:
            raise FileNotFoundError(f"no 3D mesh in workspace {ws_path}")
        ctx.check_dofs(len(mesh.nodes) * 3)
        ctx.consume_fem_solves(1)
        bbox = preview["bbox_classification"]
        mat = MaterialSpec(
            name="modal-3d-nh",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(
                    float(args.get("youngs_modulus_mpa", 5_000.0)), Unit.MPA
                ).to_canonical(),
                "poisson_ratio": Quantity(
                    float(args.get("poisson_ratio", 0.30)), Unit.ONE
                ).to_canonical(),
            },
        )
        res = solve_3d_modal_analysis(
            mesh,
            mat,
            density_t_per_mm3=float(args.get("density_t_per_mm3", 2.7e-9)),
            dirichlet_bcs=(
                DirichletBC3D(
                    "fix_x_min",
                    tuple(bbox["x_min"]),
                    ("x", "y", "z"),
                    (0.0, 0.0, 0.0),
                ),
            ),
            num_modes=int(args.get("num_modes", 4)),
        )
        return res.document()

    reg.register(
        CapabilitySpec(
            name="multiphysics.solve_modal_3d",
            domain=CapabilityDomain.MULTIPHYSICS,
            summary="Compute 3D structural natural eigenfrequencies and mode shapes.",
            description="Assembles consistent/lumped mass and solves K_t phi = omega^2 M phi.",
            input_schema={
                "type": "object",
                "properties": {
                    "workspace_dir": {"type": "string"},
                    "youngs_modulus_mpa": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 10000.0,
                    },
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "density_t_per_mm3": {"type": "number", "minimum": 1e-12},
                    "num_modes": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("modal", "eigenfrequency", "3d"),
        ),
        _solve_modal_3d,
    )


def _register_surrogate_opt_ingest_export_governance(reg: CapabilityRegistry) -> None:
    def _inspect_bundle(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.bundle import inspect as inspect_bundle
        from piginot.bundle import load_bundle

        bpath = ctx.resolve_path(str(args["bundle_path"]))
        return dict(inspect_bundle(load_bundle(bpath)))

    reg.register(
        CapabilitySpec(
            name="surrogate.inspect_bundle",
            domain=CapabilityDomain.SURROGATE,
            summary="Inspect a verified bundle manifest, parameter envelope, claim level, and QoIs.",
            description="Verifies SHA-256 integrity of the bundle and reports envelope bounds.",
            input_schema={
                "type": "object",
                "required": ["bundle_path"],
                "properties": {"bundle_path": {"type": "string"}},
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("bundle", "inspect"),
        ),
        _inspect_bundle,
    )

    def _predict_verified(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.bundle import load_bundle
        from piginot.bundle.problem import PortableProblem
        from piginot.cli import parse_request
        from piginot.serving import ServingMode, load_recipe, serve

        bpath = ctx.resolve_path(str(args["bundle_path"]))
        bundle = load_bundle(bpath)
        ctx.check_dofs(len(bundle.reference_mesh.nodes) * 2)
        mode = ServingMode(str(args.get("mode", "certified")))
        if mode is ServingMode.EXACT:
            ctx.consume_fem_solves(1)
        parsed_inputs, parsed_queries = parse_request({"inputs": dict(args["inputs"])})
        is_portable = isinstance(
            getattr(bundle.model, "problem", None), PortableProblem
        )
        recipe = (
            load_recipe(bundle, bpath)
            if (mode is not ServingMode.FAST and not is_portable)
            else None
        )
        served = serve(
            bundle,
            parsed_inputs,
            cast("Sequence[Sequence[float]] | None", parsed_queries),
            mode=mode,
            recipe=recipe,
            corrected=mode is not ServingMode.FAST,
            emit_receipt=True,
        )
        doc = served.document()
        if isinstance(doc.get("trust_receipt_sha256"), str):
            ctx.set_receipt(doc["trust_receipt_sha256"])
        return {
            "mode": doc["mode"],
            "serving_path": doc["path"],
            "verdict": doc["prediction"]["verdict"],
            "qois": doc["prediction"].get("qois", []),
            "trust_receipt": doc.get("trust_receipt"),
            "trust_receipt_sha256": doc.get("trust_receipt_sha256"),
        }

    reg.register(
        CapabilitySpec(
            name="surrogate.predict_verified",
            domain=CapabilityDomain.SURROGATE,
            summary="Serve a 3-tier verified prediction (fast/certified/exact) with RFC 8785 Trust Receipt.",
            description="Enforces envelope bounds, Tier 0 det(F)>0 admissibility, and Trust Receipt signing.",
            input_schema={
                "type": "object",
                "required": ["bundle_path", "inputs"],
                "properties": {
                    "bundle_path": {"type": "string"},
                    "inputs": {"type": "object"},
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "certified", "exact"],
                        "default": "certified",
                    },
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SURROGATE,
            tags=("serve", "predict", "receipt"),
        ),
        _predict_verified,
    )

    def _train_p3d(ctx: HarnessContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
        import torch

        from piginot.cad.parametric_3d import (
            default_holed_block_3d_spec,
            sample_parametric_cad_3d_doe,
        )
        from piginot.operators.geometry_3d import SURFACE_3D_GALE_FEATURE_DIM
        from piginot.operators.surface_operator_3d import Surface3DConfig
        from piginot.training.parametric_3d_engine import (
            Parametric3DPhysicsConfig,
            Parametric3DTrainingConfig,
            save_parametric_3d_bundle,
            train_parametric_3d_operator,
        )

        epochs = int(args.get("epochs", 10))
        count = int(args.get("train_count", 3))
        use_fem = bool(args.get("use_fem_supervision", False))
        ctx.consume_training_epochs(epochs)
        if use_fem:
            ctx.consume_fem_solves(count)
        spec = default_holed_block_3d_spec(
            mesh_size_mm=float(args.get("mesh_size_mm", 8.5))
        )
        doe = sample_parametric_cad_3d_doe(spec, count=count, seed_offset=1)
        physics = Parametric3DPhysicsConfig(
            youngs_modulus_mpa=float(args.get("youngs_modulus_mpa", 15.0)),
            poisson_ratio=float(args.get("poisson_ratio", 0.35)),
            traction_mpa=(float(args.get("traction_x_mpa", 0.35)), 0.0, 0.0),
            load_steps=1,
        )
        bundle = train_parametric_3d_operator(
            spec,
            physics,
            doe.instances,
            operator_config=Surface3DConfig(
                parameter_dim=len(spec.parameters),
                surface_feature_dim=SURFACE_3D_GALE_FEATURE_DIM,
                hidden_dim=24,
                slice_count=4,
                num_blocks=1,
                num_heads=2,
            ),
            training_config=Parametric3DTrainingConfig(
                epochs=epochs,
                quadrature_subsample=32,
                use_fem_supervision=use_fem,
                seed=int(args.get("seed", 0)),
            ),
            dtype=torch.float32,
        )
        out_dir = ctx.resolve_path(str(args.get("output_dir", "parametric_3d_bundle")))
        save_parametric_3d_bundle(bundle, out_dir)
        ctx.record_artifact(out_dir / "bundle.json")
        ctx.record_artifact(out_dir / "weights.safetensors")
        ctx.set_receipt(bundle.bundle_hash)
        return {
            "bundle_dir": ctx.relative_path(out_dir),
            "bundle_hash": bundle.bundle_hash,
            "train_count": bundle.train_count,
            "final_loss": bundle.loss_history[-1] if bundle.loss_history else None,
            "claim_level": bundle.claim_level,
        }

    reg.register(
        CapabilitySpec(
            name="surrogate.train_parametric_3d",
            domain=CapabilityDomain.SURROGATE,
            summary="Train and persist a geometry-generalizing 3D parametric Surface3DEnergyOperator.",
            description="Trains a 16-channel GALE 3D operator + GeometryOODDetector3D across a 3D DOE.",
            input_schema={
                "type": "object",
                "properties": {
                    "output_dir": {"type": "string", "default": "parametric_3d_bundle"},
                    "train_count": {"type": "integer", "minimum": 2, "maximum": 32},
                    "epochs": {"type": "integer", "minimum": 1, "maximum": 200},
                    "mesh_size_mm": {"type": "number", "minimum": 2.0, "maximum": 30.0},
                    "use_fem_supervision": {"type": "boolean", "default": False},
                    "youngs_modulus_mpa": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 10000.0,
                    },
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "traction_x_mpa": {"type": "number"},
                    "seed": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.TRAINING_JOB,
            tags=("surrogate", "3d", "parametric", "train"),
        ),
        _train_p3d,
    )

    def _predict_p3d(ctx: HarnessContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
        from piginot.training.parametric_3d_engine import (
            load_parametric_3d_bundle,
            predict_parametric_3d,
        )

        bdir = ctx.resolve_path(str(args["bundle_dir"]))
        mode = str(args.get("mode", "fast"))
        if mode in ("certified", "exact"):
            ctx.consume_fem_solves(1)
        bundle = load_parametric_3d_bundle(bdir)
        params = {str(k): float(v) for k, v in dict(args["parameters"]).items()}
        rep = predict_parametric_3d(bundle, params, mode=cast(Any, mode))
        ctx.set_receipt(rep.report_hash)
        return {**rep.document(), "report_hash": rep.report_hash}

    reg.register(
        CapabilitySpec(
            name="surrogate.predict_parametric_3d",
            domain=CapabilityDomain.SURROGATE,
            summary="Predict (or OOD-refuse) an unseen 3D parametric CAD variant.",
            description="Supports fast, certified, and warm-start exact TET10 FEM verification.",
            input_schema={
                "type": "object",
                "required": ["bundle_dir", "parameters"],
                "properties": {
                    "bundle_dir": {"type": "string"},
                    "parameters": {"type": "object"},
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "certified", "exact"],
                        "default": "fast",
                    },
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SURROGATE,
            tags=("surrogate", "3d", "parametric", "ood"),
        ),
        _predict_p3d,
    )

    def _ingest_archive(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.ui.workbench_3d.vv10_panel import run_cae_ingest_summary

        arch = ctx.resolve_path(str(args["archive_dir"]))
        return run_cae_ingest_summary(arch)

    reg.register(
        CapabilitySpec(
            name="ingest.screen_cae_archive",
            domain=CapabilityDomain.INGEST_VALIDATE,
            summary="Screen external CAE archives (VTU/XDMF/CalculiX) for duplicates, det(F), & outliers.",
            description="Produces a deterministic screening report and leak-free TRAIN/DEV/HOLDOUT split.",
            input_schema={
                "type": "object",
                "required": ["archive_dir"],
                "properties": {"archive_dir": {"type": "string"}},
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("ingest", "cae", "screening"),
        ),
        _ingest_archive,
    )

    def _validate_vv10(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.ui.workbench_3d.vv10_panel import run_vv10_calibration_summary

        ds_path = ctx.resolve_path(str(args["dataset_path"]))
        return run_vv10_calibration_summary(
            ds_path, u_num_mpa=float(args.get("u_num_mpa", 0.05))
        )

    reg.register(
        CapabilitySpec(
            name="validate.experimental_vv10",
            domain=CapabilityDomain.INGEST_VALIDATE,
            summary="Run differentiable inverse material calibration & ASME V&V 10 validation budget.",
            description="Emits orthogonal ExperimentalValidationTier without mutating S14 ClaimLevel.",
            input_schema={
                "type": "object",
                "required": ["dataset_path"],
                "properties": {
                    "dataset_path": {"type": "string"},
                    "u_num_mpa": {"type": "number", "minimum": 0.0},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.TRAINING_JOB,
            tags=("validation", "vv10", "inverse"),
        ),
        _validate_vv10,
    )

    def _export_3d(ctx: HarnessContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
        from piginot.export.vtu_export import export_deformed_glb, export_fields_to_vtu
        from piginot.ui.workbench_3d.state_3d import Workbench3DWorkspace

        ws_path = ctx.resolve_path(
            str(
                args.get(
                    "workspace_dir", ctx.state_kv.get("last_3d_workspace", "study_3d")
                )
            )
        )
        out_path = ctx.resolve_path(str(args["output_path"]))
        fmt = str(args.get("format", "vtu"))
        ws = Workbench3DWorkspace(ws_path)
        mesh = ws.load_mesh()
        if mesh is None:
            raise FileNotFoundError(f"no 3D mesh in workspace {ws_path}")
        npz = ws.results_dir / "latest_fem_3d.npz"
        disp = None
        vm = None
        if npz.is_file():
            data = np.load(npz)
            disp = np.asarray(data["nodal_displacement"], dtype=np.float64)
            vm = np.asarray(data["nodal_von_mises_mpa"], dtype=np.float64)

        if fmt == "glb":
            rep = export_deformed_glb(
                mesh, out_path, displacement=disp, scalar_field=vm
            )
            sha = str(rep["file_sha256"])
        else:
            pfields: dict[str, Any] = {}
            if disp is not None:
                pfields["displacement_mm"] = disp
            if vm is not None:
                pfields["von_mises_mpa"] = vm
            vrep = export_fields_to_vtu(mesh, out_path, point_fields=pfields or None)
            rep = vrep.document()
            sha = vrep.file_sha256

        ctx.record_artifact(out_path)
        ctx.set_receipt(sha)
        return {**rep, "output_path": ctx.relative_path(out_path)}

    reg.register(
        CapabilitySpec(
            name="export.vtu_or_glb_3d",
            domain=CapabilityDomain.EXPORT,
            summary="Export a 3D workspace mesh and solved fields to ParaView .vtu or glTF 2.0 .glb.",
            description="Writes content-hashed 3D visualization files strictly inside the workspace.",
            input_schema={
                "type": "object",
                "required": ["output_path"],
                "properties": {
                    "workspace_dir": {"type": "string"},
                    "output_path": {"type": "string"},
                    "format": {
                        "type": "string",
                        "enum": ["vtu", "glb"],
                        "default": "vtu",
                    },
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("export", "vtu", "glb", "paraview"),
        ),
        _export_3d,
    )

    def _export_fmu(ctx: HarnessContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
        from piginot.export import export_bundle_to_fmu

        bpath = ctx.resolve_path(str(args["bundle_path"]))
        out_path = ctx.resolve_path(str(args["output_path"]))
        res = export_bundle_to_fmu(bpath, out_path)
        sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
        ctx.record_artifact(out_path)
        ctx.set_receipt(sha)
        return {
            "fmu_path": ctx.relative_path(out_path),
            "fmu_sha256": sha,
            "bundle_manifest_sha256": res.bundle_manifest_sha256,
        }

    reg.register(
        CapabilitySpec(
            name="export.fmu",
            domain=CapabilityDomain.EXPORT,
            summary="Package a verified bundle as an FMI 3.0 Co-Simulation .fmu archive.",
            description="Emits a standards-compliant FMI 3.0 FMU with embedded envelope guards.",
            input_schema={
                "type": "object",
                "required": ["bundle_path", "output_path"],
                "properties": {
                    "bundle_path": {"type": "string"},
                    "output_path": {"type": "string"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("export", "fmu", "fmi3"),
        ),
        _export_fmu,
    )

    def _verify_receipt(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.certify import verify_receipt_document

        obj = args.get("receipt")
        if obj is None:
            rpath = ctx.resolve_path(str(args["receipt_path"]))
            obj = json.loads(rpath.read_text(encoding="utf-8"))
        ok, errs = verify_receipt_document(obj)
        return {"verified": ok, "errors": list(errs)}

    reg.register(
        CapabilitySpec(
            name="governance.verify_receipt",
            domain=CapabilityDomain.GOVERNANCE,
            summary="Independently verify an RFC 8785 Trust Receipt SHA-256 and epistemic scope.",
            description="Catches any byte or claim-level tampering in Trust Receipts.",
            input_schema={
                "type": "object",
                "properties": {
                    "receipt": {"type": "object"},
                    "receipt_path": {"type": "string"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("governance", "receipt", "verify"),
        ),
        _verify_receipt,
    )

    def _verify_ledger(
        ctx: HarnessContext, _args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        ok, errs = ctx.harness.verify_ledger()
        return {
            "session_id": ctx.harness.session_id,
            "verified": ok,
            "entry_count": len(ctx.harness.read_ledger()),
            "errors": list(errs),
        }

    reg.register(
        CapabilitySpec(
            name="governance.verify_session_ledger",
            domain=CapabilityDomain.GOVERNANCE,
            summary="Verify the SHA-256 hash chain of the current AI-agent session ledger.",
            description="Confirms zero retroactive tampering across all recorded tool calls and receipts.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("governance", "ledger", "audit"),
        ),
        _verify_ledger,
    )

    def _run_bench(ctx: HarnessContext, args: Mapping[str, Any]) -> Mapping[str, Any]:
        from piginot.benchmarks import run_benchmark_suite_3d

        ctx.consume_fem_solves(4)
        wdir = ctx.resolve_path(str(args.get("workdir", "bench_3d")))
        rep = run_benchmark_suite_3d(
            wdir, profile=cast(Any, str(args.get("profile", "quick")))
        )
        md_path = wdir / "benchmark-report.md"
        md_path.write_text(rep.to_markdown(), encoding="utf-8")
        ctx.record_artifact(md_path)
        ctx.set_receipt(rep.report_hash)
        return {
            **rep.document(),
            "report_hash": rep.report_hash,
            "markdown_path": ctx.relative_path(md_path),
        }

    reg.register(
        CapabilitySpec(
            name="governance.run_benchmarks_3d",
            domain=CapabilityDomain.GOVERNANCE,
            summary="Run reproducible B1-B4 3D solver scaling, foundation LoRA, OOD & parity benchmarks.",
            description="Emits a content-hashed JSON + Markdown credibility report.",
            input_schema={
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "string",
                        "enum": ["quick", "full"],
                        "default": "quick",
                    },
                    "workdir": {"type": "string", "default": "bench_3d"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=4,
            tags=("benchmarks", "3d", "scaling", "foundation"),
        ),
        _run_bench,
    )

    def _adapt_foundation(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.contracts import MaterialSpec, Quantity, Unit
        from piginot.operators.low_rank_adapter_3d import LowRankAdapter3DConfig
        from piginot.training.foundation_pretrain import (
            adapt_foundation_to_3d_part,
            default_foundation_checkpoint_dir,
        )
        from piginot.ui.workbench_3d.state_3d import Workbench3DWorkspace

        ws_path = ctx.resolve_path(
            str(
                args.get(
                    "workspace_dir", ctx.state_kv.get("last_3d_workspace", "study_3d")
                )
            )
        )
        ws = Workbench3DWorkspace(ws_path)
        mesh = ws.load_mesh()
        preview = ws.load_preview()
        if mesh is None or preview is None:
            raise FileNotFoundError(f"no 3D mesh in workspace {ws_path}")
        steps = int(args.get("steps", 6))
        ctx.consume_training_epochs(steps)
        ckpt_dir = (
            ctx.resolve_path(str(args["checkpoint_dir"]))
            if args.get("checkpoint_dir")
            else default_foundation_checkpoint_dir()
        )
        bbox = preview["bbox_classification"]
        mat = MaterialSpec(
            name="lora-3d-nh",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(
                    float(args.get("youngs_modulus_mpa", 100.0)), Unit.MPA
                ).to_canonical(),
                "poisson_ratio": Quantity(
                    float(args.get("poisson_ratio", 0.30)), Unit.ONE
                ).to_canonical(),
            },
        )
        _, res = adapt_foundation_to_3d_part(
            ckpt_dir,
            mesh=mesh,
            material=mat,
            clamped_face_tags=tuple(bbox["x_min"]),
            loaded_face_tags=tuple(bbox["x_max"]),
            adapter_config=LowRankAdapter3DConfig(
                rank=int(args.get("rank", 4)),
                max_steps=steps,
                from_scratch_reference_steps=max(100, steps * 10),
                allow_ood_adaptation=True,
            ),
        )
        return res.document()

    reg.register(
        CapabilitySpec(
            name="surrogate.adapt_foundation_lora_3d",
            domain=CapabilityDomain.SURROGATE,
            summary="Adapt the shipped struct3d-base-v1 foundation checkpoint to a 3D part via 3D LoRA.",
            description="Freezes base weights and trains <=10% LoRA parameters label-free on 3D energy.",
            input_schema={
                "type": "object",
                "properties": {
                    "workspace_dir": {"type": "string"},
                    "checkpoint_dir": {"type": "string"},
                    "rank": {"type": "integer", "minimum": 1, "maximum": 16},
                    "steps": {"type": "integer", "minimum": 1, "maximum": 50},
                    "youngs_modulus_mpa": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 10000.0,
                    },
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.TRAINING_JOB,
            tags=("foundation", "lora", "3d", "transfer"),
        ),
        _adapt_foundation,
    )


def _register_v4_physics_capabilities(reg: CapabilityRegistry) -> None:
    """Register V4-3 shells/beams, buckling/Riks, dynamics, fatigue, and fracture."""

    def _solve_shell_plate(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.fem.shells_beams_3d import (
            ShellDirichletBC,
            make_rectangular_plate_shell_mesh,
            solve_3d_dkt_shell,
        )

        lx = float(args.get("lx_mm", 100.0))
        ly = float(args.get("ly_mm", 100.0))
        t_mm = float(args.get("thickness_mm", 2.0))
        nx = int(args.get("nx", 8))
        ny = int(args.get("ny", 8))
        bc_kind = str(args.get("boundary_condition", "clamped"))
        mesh = make_rectangular_plate_shell_mesh(
            lx, ly, nx=nx, ny=ny, thickness_mm=t_mm
        )
        ctx.check_dofs(len(mesh.nodes) * 6)
        ctx.consume_fem_solves(1)
        if bc_kind == "simply_supported":
            bcs = (
                ShellDirichletBC(
                    "ss",
                    "perimeter",
                    components=(0, 1, 2),
                    values=(0.0, 0.0, 0.0),
                ),
            )
        else:
            bcs = (ShellDirichletBC("clamped", "perimeter"),)
        res = solve_3d_dkt_shell(
            mesh,
            youngs_modulus_mpa=float(args.get("youngs_modulus_mpa", 210_000.0)),
            poisson_ratio=float(args.get("poisson_ratio", 0.3)),
            dirichlet_bcs=bcs,
            uniform_pressure_mpa=float(args.get("pressure_mpa", 0.01)),
        )
        return res.document()

    reg.register(
        CapabilitySpec(
            name="fem.solve_shell_plate_3d",
            domain=CapabilityDomain.FEM,
            summary="Solve a shear-locking-free 3D DKT-CST flat/curved triangular shell.",
            description="6-DOF/node membrane CST + Discrete Kirchhoff Triangle bending plate/shell.",
            input_schema={
                "type": "object",
                "properties": {
                    "lx_mm": {"type": "number", "minimum": 1.0},
                    "ly_mm": {"type": "number", "minimum": 1.0},
                    "thickness_mm": {"type": "number", "minimum": 0.01},
                    "nx": {"type": "integer", "minimum": 2, "maximum": 32},
                    "ny": {"type": "integer", "minimum": 2, "maximum": 32},
                    "youngs_modulus_mpa": {"type": "number", "minimum": 1.0},
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "pressure_mpa": {"type": "number"},
                    "boundary_condition": {
                        "type": "string",
                        "enum": ["clamped", "simply_supported"],
                    },
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("shell", "dkt", "plate", "3d"),
        ),
        _solve_shell_plate,
    )

    def _solve_beam_3d(ctx: HarnessContext, args: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        from piginot.fem.shells_beams_3d import (
            BeamSection3D,
            solve_3d_timoshenko_beam,
        )

        length = float(args.get("length_mm", 100.0))
        width = float(args.get("width_mm", 10.0))
        height = float(args.get("height_mm", 10.0))
        n_el = int(args.get("num_elements", 4))
        ctx.check_dofs((n_el + 1) * 6)
        ctx.consume_fem_solves(1)
        xs = np.linspace(0.0, length, n_el + 1)
        nodes = np.column_stack((xs, np.zeros_like(xs), np.zeros_like(xs)))
        elems = np.column_stack((np.arange(n_el), np.arange(1, n_el + 1)))
        loads = np.zeros((n_el + 1, 6), dtype=np.float64)
        loads[-1, 1] = float(args.get("tip_load_y_n", 100.0))
        loads[-1, 2] = float(args.get("tip_load_z_n", 0.0))
        res = solve_3d_timoshenko_beam(
            nodes,
            elems,
            section=BeamSection3D.rectangular(width, height),
            youngs_modulus_mpa=float(args.get("youngs_modulus_mpa", 210_000.0)),
            poisson_ratio=float(args.get("poisson_ratio", 0.3)),
            constrained_dofs={c: 0.0 for c in range(6)},
            nodal_loads_n=loads,
        )
        return res.document()

    reg.register(
        CapabilitySpec(
            name="fem.solve_timoshenko_beam_3d",
            domain=CapabilityDomain.FEM,
            summary="Solve a 3D Timoshenko shear-deformable cantilever beam.",
            description="Exact 12-DOF/element 3D Timoshenko beam solver with transverse shear.",
            input_schema={
                "type": "object",
                "properties": {
                    "length_mm": {"type": "number", "minimum": 1.0},
                    "width_mm": {"type": "number", "minimum": 0.1},
                    "height_mm": {"type": "number", "minimum": 0.1},
                    "num_elements": {"type": "integer", "minimum": 1, "maximum": 64},
                    "youngs_modulus_mpa": {"type": "number", "minimum": 1.0},
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "tip_load_y_n": {"type": "number"},
                    "tip_load_z_n": {"type": "number"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("beam", "timoshenko", "3d"),
        ),
        _solve_beam_3d,
    )

    def _solve_buckling_3d(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.contracts import MaterialSpec, Quantity, Unit
        from piginot.fem.backend_3d import DirichletBC3D, TractionBC3D
        from piginot.fem.buckling_3d import solve_3d_linear_buckling
        from piginot.geometry.mesh3d import make_brick_tet4_mesh

        mesh = make_brick_tet4_mesh(
            float(args.get("length_mm", 40.0)),
            float(args.get("width_mm", 6.0)),
            float(args.get("depth_mm", 6.0)),
            nx=int(args.get("nx", 6)),
            ny=int(args.get("ny", 2)),
            nz=int(args.get("nz", 2)),
        )
        ctx.check_dofs(len(mesh.nodes) * 3)
        ctx.consume_fem_solves(1)
        mat = MaterialSpec(
            name="buckling-nh",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(
                    float(args.get("youngs_modulus_mpa", 1000.0)), Unit.MPA
                ).to_canonical(),
                "poisson_ratio": Quantity(
                    float(args.get("poisson_ratio", 0.3)), Unit.ONE
                ).to_canonical(),
            },
        )
        res = solve_3d_linear_buckling(
            mesh,
            mat,
            dirichlet_bcs=(
                DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
            ),
            traction_bcs=(
                TractionBC3D(
                    "compress",
                    ("x_max",),
                    (-abs(float(args.get("compressive_stress_mpa", 1.0))), 0.0, 0.0),
                ),
            ),
            num_modes=int(args.get("num_modes", 2)),
        )
        return res.document()

    reg.register(
        CapabilitySpec(
            name="fem.solve_linear_buckling_3d",
            domain=CapabilityDomain.FEM,
            summary="Solve 3D linearized eigen-buckling (K_0 + lambda K_G) phi = 0.",
            description="Computes lowest positive critical buckling load factors on a 3D solid.",
            input_schema={
                "type": "object",
                "properties": {
                    "length_mm": {"type": "number", "minimum": 2.0},
                    "width_mm": {"type": "number", "minimum": 1.0},
                    "depth_mm": {"type": "number", "minimum": 1.0},
                    "nx": {"type": "integer", "minimum": 2, "maximum": 20},
                    "ny": {"type": "integer", "minimum": 1, "maximum": 8},
                    "nz": {"type": "integer", "minimum": 1, "maximum": 8},
                    "youngs_modulus_mpa": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 10000.0,
                    },
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "compressive_stress_mpa": {"type": "number", "minimum": 1.0e-6},
                    "num_modes": {"type": "integer", "minimum": 1, "maximum": 6},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("buckling", "stability", "eigen", "3d"),
        ),
        _solve_buckling_3d,
    )

    def _solve_transient_dynamics(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.contracts import MaterialSpec, Quantity, Unit
        from piginot.fem.backend_3d import DirichletBC3D, TractionBC3D
        from piginot.fem.dynamics_3d import (
            DynamicsScheme,
            TransientDynamicsConfig,
            solve_3d_transient_dynamics,
        )
        from piginot.geometry.mesh3d import make_brick_tet4_mesh

        mesh = make_brick_tet4_mesh(
            float(args.get("lx_mm", 20.0)),
            float(args.get("ly_mm", 6.0)),
            float(args.get("lz_mm", 6.0)),
            nx=int(args.get("nx", 3)),
            ny=int(args.get("ny", 1)),
            nz=int(args.get("nz", 1)),
        )
        ctx.check_dofs(len(mesh.nodes) * 3)
        ctx.consume_fem_solves(1)
        mat = MaterialSpec(
            name="dyn-nh",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(
                    float(args.get("youngs_modulus_mpa", 1000.0)), Unit.MPA
                ).to_canonical(),
                "poisson_ratio": Quantity(
                    float(args.get("poisson_ratio", 0.3)), Unit.ONE
                ).to_canonical(),
            },
        )
        cfg = TransientDynamicsConfig(
            scheme=DynamicsScheme(str(args.get("scheme", "newmark-beta"))),
            dt_s=float(args.get("dt_s", 1.0e-4)),
            num_steps=int(args.get("num_steps", 15)),
            density_t_per_mm3=float(args.get("density_t_per_mm3", 7.85e-9)),
            rayleigh_alpha=float(args.get("rayleigh_alpha", 0.0)),
            rayleigh_beta=float(args.get("rayleigh_beta", 0.0)),
        )
        res = solve_3d_transient_dynamics(
            mesh,
            mat,
            dirichlet_bcs=(
                DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
            ),
            traction_bcs=(
                TractionBC3D(
                    "pulse",
                    ("x_max",),
                    (float(args.get("traction_x_mpa", 0.5)), 0.0, 0.0),
                ),
            ),
            config=cfg,
        )
        return res.document()

    reg.register(
        CapabilitySpec(
            name="multiphysics.solve_transient_dynamics_3d",
            domain=CapabilityDomain.MULTIPHYSICS,
            summary="Solve 3D transient elastodynamics (Newmark-beta, HHT-alpha, or Explicit CFL).",
            description="Integrates M u'' + C u' + K u = f(t) with automatic CFL guard.",
            input_schema={
                "type": "object",
                "properties": {
                    "scheme": {
                        "type": "string",
                        "enum": ["newmark-beta", "hht-alpha", "explicit-central-diff"],
                    },
                    "dt_s": {"type": "number", "minimum": 1.0e-9},
                    "num_steps": {"type": "integer", "minimum": 1, "maximum": 500},
                    "density_t_per_mm3": {"type": "number", "minimum": 1.0e-12},
                    "rayleigh_alpha": {"type": "number", "minimum": 0.0},
                    "rayleigh_beta": {"type": "number", "minimum": 0.0},
                    "lx_mm": {"type": "number", "minimum": 1.0},
                    "ly_mm": {"type": "number", "minimum": 1.0},
                    "lz_mm": {"type": "number", "minimum": 1.0},
                    "nx": {"type": "integer", "minimum": 1, "maximum": 12},
                    "ny": {"type": "integer", "minimum": 1, "maximum": 6},
                    "nz": {"type": "integer", "minimum": 1, "maximum": 6},
                    "youngs_modulus_mpa": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 10000.0,
                    },
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "traction_x_mpa": {"type": "number"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("dynamics", "newmark", "hht", "explicit", "3d"),
        ),
        _solve_transient_dynamics,
    )

    def _evaluate_fatigue(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.stateful.fatigue_fracture import (
            FatigueMaterialSpec,
            FatigueModel,
            MeanStressModel,
            evaluate_fatigue_life,
        )

        hist = [float(v) for v in args["stress_history_mpa"]]
        mat = FatigueMaterialSpec(
            name=str(args.get("material_name", "steel-generic")),
            ultimate_tensile_mpa=float(args.get("ultimate_tensile_mpa", 600.0)),
            fatigue_strength_coeff_mpa=float(
                args.get("fatigue_strength_coeff_mpa", 950.0)
            ),
            fatigue_strength_exp=float(args.get("fatigue_strength_exp", -0.09)),
            youngs_modulus_mpa=float(args.get("youngs_modulus_mpa", 210_000.0)),
            endurance_limit_mpa=float(args.get("endurance_limit_mpa", 0.0)),
        )
        res = evaluate_fatigue_life(
            hist,
            mat,
            model=FatigueModel(str(args.get("model", "basquin-sn"))),
            mean_stress_model=MeanStressModel(
                str(args.get("mean_stress_model", "goodman"))
            ),
            block_repetitions=float(args.get("block_repetitions", 1.0)),
        )
        return res.document()

    reg.register(
        CapabilitySpec(
            name="multiphysics.evaluate_fatigue_rainflow",
            domain=CapabilityDomain.MULTIPHYSICS,
            summary="Evaluate cyclic fatigue life via ASTM E1049 rainflow & Palmgren-Miner damage.",
            description="Supports Basquin S-N, Coffin-Manson E-N, and Goodman/Gerber mean-stress.",
            input_schema={
                "type": "object",
                "properties": {
                    "stress_history_mpa": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                    },
                    "material_name": {"type": "string"},
                    "ultimate_tensile_mpa": {"type": "number", "minimum": 1.0},
                    "fatigue_strength_coeff_mpa": {"type": "number", "minimum": 1.0},
                    "fatigue_strength_exp": {"type": "number", "maximum": -1.0e-6},
                    "youngs_modulus_mpa": {"type": "number", "minimum": 1.0},
                    "endurance_limit_mpa": {"type": "number", "minimum": 0.0},
                    "model": {
                        "type": "string",
                        "enum": ["basquin-sn", "coffin-manson-en"],
                    },
                    "mean_stress_model": {
                        "type": "string",
                        "enum": ["none", "goodman", "gerber"],
                    },
                    "block_repetitions": {"type": "number", "minimum": 1.0e-6},
                },
                "required": ["stress_history_mpa"],
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("fatigue", "rainflow", "miner", "durability"),
        ),
        _evaluate_fatigue,
    )

    def _solve_phase_field(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.fem.backend_3d import DirichletBC3D
        from piginot.geometry.mesh3d import make_brick_tet4_mesh
        from piginot.stateful.fatigue_fracture import (
            PhaseFieldConfig,
            solve_phase_field_fracture,
        )

        mesh = make_brick_tet4_mesh(
            float(args.get("lx_mm", 10.0)),
            float(args.get("ly_mm", 4.0)),
            float(args.get("lz_mm", 4.0)),
            nx=int(args.get("nx", 3)),
            ny=int(args.get("ny", 1)),
            nz=int(args.get("nz", 1)),
        )
        ctx.check_dofs(len(mesh.nodes) * 3)
        ctx.consume_fem_solves(1)
        u_end = float(args.get("end_displacement_x_mm", 0.05))
        cfg = PhaseFieldConfig(
            youngs_modulus_mpa=float(args.get("youngs_modulus_mpa", 20_000.0)),
            poisson_ratio=float(args.get("poisson_ratio", 0.25)),
            fracture_toughness_gc_n_per_mm=float(
                args.get("fracture_toughness_gc_n_per_mm", 0.5)
            ),
            length_scale_lc_mm=float(args.get("length_scale_lc_mm", 2.0)),
            load_steps=int(args.get("load_steps", 6)),
        )
        res = solve_phase_field_fracture(
            mesh,
            dirichlet_bcs=(
                DirichletBC3D("x0", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
                DirichletBC3D("x1", ("x_max",), ("x",), (u_end,)),
            ),
            config=cfg,
        )
        return res.document()

    reg.register(
        CapabilitySpec(
            name="multiphysics.solve_phase_field_fracture_3d",
            domain=CapabilityDomain.MULTIPHYSICS,
            summary="Solve staggered AT2 brittle phase-field fracture with spectral tension split.",
            description="Guarantees d_dot >= 0 irreversibility and zero damage under compression.",
            input_schema={
                "type": "object",
                "properties": {
                    "lx_mm": {"type": "number", "minimum": 1.0},
                    "ly_mm": {"type": "number", "minimum": 1.0},
                    "lz_mm": {"type": "number", "minimum": 1.0},
                    "nx": {"type": "integer", "minimum": 1, "maximum": 10},
                    "ny": {"type": "integer", "minimum": 1, "maximum": 4},
                    "nz": {"type": "integer", "minimum": 1, "maximum": 4},
                    "end_displacement_x_mm": {"type": "number"},
                    "youngs_modulus_mpa": {"type": "number", "minimum": 1.0},
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "fracture_toughness_gc_n_per_mm": {
                        "type": "number",
                        "minimum": 1.0e-6,
                    },
                    "length_scale_lc_mm": {"type": "number", "minimum": 1.0e-3},
                    "load_steps": {"type": "integer", "minimum": 1, "maximum": 40},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("fracture", "phase-field", "at2", "damage", "3d"),
        ),
        _solve_phase_field,
    )


def _register_v4_generative_capabilities(reg: CapabilityRegistry) -> None:
    """Register V4-4 multi-topology OOD, 3D SIMP topology opt, and FFD shape opt."""

    def _run_topology_simp_3d(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.fem.backend_3d import DirichletBC3D, TractionBC3D
        from piginot.geometry.mesh3d import make_brick_tet4_mesh
        from piginot.optimization.topology_opt_3d import (
            TopologyOpt3DConfig,
            optimize_3d_topology_simp,
        )

        mesh = make_brick_tet4_mesh(
            float(args.get("lx_mm", 24.0)),
            float(args.get("ly_mm", 12.0)),
            float(args.get("lz_mm", 6.0)),
            nx=int(args.get("nx", 4)),
            ny=int(args.get("ny", 2)),
            nz=int(args.get("nz", 1)),
        )
        ctx.check_dofs(len(mesh.nodes) * 3)
        ctx.consume_fem_solves(1)
        cfg = TopologyOpt3DConfig(
            volume_fraction=float(args.get("volume_fraction", 0.55)),
            penalization=float(args.get("penalization", 3.0)),
            filter_radius_mm=float(args.get("filter_radius_mm", 5.0)),
            max_iterations=int(args.get("max_iterations", 12)),
            threshold_density=float(args.get("threshold_density", 0.35)),
            youngs_modulus_mpa=float(args.get("youngs_modulus_mpa", 1000.0)),
            poisson_ratio=float(args.get("poisson_ratio", 0.30)),
        )
        res = optimize_3d_topology_simp(
            mesh,
            dirichlet_bcs=(
                DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
            ),
            traction_bcs=(
                TractionBC3D(
                    "shear",
                    ("x_max",),
                    (0.0, float(args.get("traction_y_mpa", -1.0)), 0.0),
                ),
            ),
            config=cfg,
        )
        ctx.set_receipt(res.receipt_sha256)
        doc = res.document()
        if args.get("output_dir"):
            out_dir = ctx.resolve_path(str(args["output_dir"]))
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = out_dir / "simp_3d_result.npz"
            np.savez_compressed(
                npz_path,
                nodes=np.asarray(mesh.nodes, dtype=np.float64),
                cells=np.asarray(mesh.cells, dtype=np.int64),
                element_densities=res.element_densities,
                solid_element_mask=res.solid_element_mask,
                compliance_history=np.asarray(res.compliance_history, dtype=np.float64),
            )
            ctx.record_artifact(npz_path)
            doc["artifact_npz"] = ctx.relative_path(npz_path)
        return doc

    reg.register(
        CapabilitySpec(
            name="optimize.topology_simp_3d",
            domain=CapabilityDomain.OPTIMIZE,
            summary="Run verified 3D SIMP topology optimization with mandatory final FEM check.",
            description=(
                "Density-based 3D SIMP with cone filter, Heaviside projection, exact adjoint "
                "sensitivities, and fail-closed final exact 3D FEM certification."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "lx_mm": {"type": "number", "minimum": 2.0},
                    "ly_mm": {"type": "number", "minimum": 2.0},
                    "lz_mm": {"type": "number", "minimum": 1.0},
                    "nx": {"type": "integer", "minimum": 2, "maximum": 12},
                    "ny": {"type": "integer", "minimum": 1, "maximum": 8},
                    "nz": {"type": "integer", "minimum": 1, "maximum": 6},
                    "volume_fraction": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 0.95,
                    },
                    "penalization": {"type": "number", "minimum": 1.0, "maximum": 5.0},
                    "filter_radius_mm": {"type": "number", "minimum": 0.0},
                    "max_iterations": {"type": "integer", "minimum": 1, "maximum": 60},
                    "threshold_density": {
                        "type": "number",
                        "minimum": 0.05,
                        "maximum": 0.99,
                    },
                    "youngs_modulus_mpa": {"type": "number", "minimum": 1.0},
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "traction_y_mpa": {"type": "number"},
                    "output_dir": {"type": "string"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("topology-optimization", "simp", "generative", "3d"),
        ),
        _run_topology_simp_3d,
    )

    def _run_ffd_shape_3d(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.contracts import MaterialSpec, Quantity, Unit
        from piginot.fem.backend_3d import DirichletBC3D, TractionBC3D
        from piginot.geometry.mesh3d import make_brick_tet4_mesh
        from piginot.optimization.ffd_shape_opt_3d import optimize_3d_ffd_shape

        mesh = make_brick_tet4_mesh(
            float(args.get("lx_mm", 20.0)),
            float(args.get("ly_mm", 8.0)),
            float(args.get("lz_mm", 8.0)),
            nx=int(args.get("nx", 3)),
            ny=int(args.get("ny", 1)),
            nz=int(args.get("nz", 1)),
        )
        ctx.check_dofs(len(mesh.nodes) * 3)
        ctx.consume_fem_solves(1)
        mat = MaterialSpec(
            name="ffd-nh",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(
                    float(args.get("youngs_modulus_mpa", 100.0)), Unit.MPA
                ).to_canonical(),
                "poisson_ratio": Quantity(
                    float(args.get("poisson_ratio", 0.30)), Unit.ONE
                ).to_canonical(),
            },
        )
        res = optimize_3d_ffd_shape(
            mesh,
            mat,
            dirichlet_bcs=(
                DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
            ),
            traction_bcs=(
                TractionBC3D(
                    "load",
                    ("x_max",),
                    (float(args.get("traction_x_mpa", 0.4)), 0.0, 0.0),
                ),
            ),
            max_displacement_limit_mm=float(
                args.get("max_displacement_limit_mm", 0.25)
            ),
            iterations=int(args.get("iterations", 4)),
        )
        ctx.set_receipt(res.receipt_sha256)
        doc = res.document()
        if args.get("output_dir"):
            out_dir = ctx.resolve_path(str(args["output_dir"]))
            out_dir.mkdir(parents=True, exist_ok=True)
            npz_path = out_dir / "ffd_3d_result.npz"
            np.savez_compressed(
                npz_path,
                initial_nodes=np.asarray(mesh.nodes, dtype=np.float64),
                optimized_nodes=np.asarray(res.optimized_mesh.nodes, dtype=np.float64),
                cells=np.asarray(mesh.cells, dtype=np.int64),
                surface_triangles=np.asarray(mesh.surface_triangles, dtype=np.int64),
                delta_control_mm=res.delta_control_mm,
            )
            ctx.record_artifact(npz_path)
            doc["artifact_npz"] = ctx.relative_path(npz_path)
        return doc

    reg.register(
        CapabilitySpec(
            name="optimize.ffd_shape_3d",
            domain=CapabilityDomain.OPTIMIZE,
            summary="Optimize 3D solid shape via Bernstein FFD cage with det J > 0 & FEM guard.",
            description="Minimizes 3D mass via Free-Form Deformation subject to FEM displacement bound.",
            input_schema={
                "type": "object",
                "properties": {
                    "lx_mm": {"type": "number", "minimum": 2.0},
                    "ly_mm": {"type": "number", "minimum": 2.0},
                    "lz_mm": {"type": "number", "minimum": 2.0},
                    "nx": {"type": "integer", "minimum": 1, "maximum": 8},
                    "ny": {"type": "integer", "minimum": 1, "maximum": 4},
                    "nz": {"type": "integer", "minimum": 1, "maximum": 4},
                    "youngs_modulus_mpa": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 10000.0,
                    },
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "traction_x_mpa": {"type": "number"},
                    "max_displacement_limit_mm": {"type": "number", "minimum": 1.0e-5},
                    "iterations": {"type": "integer", "minimum": 1, "maximum": 20},
                    "output_dir": {"type": "string"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("ffd", "shape-optimization", "generative", "3d"),
        ),
        _run_ffd_shape_3d,
    )


def _register_v4_fleet_continual_capabilities(reg: CapabilityRegistry) -> None:
    """Register V4-5 distributed fleet execution and continual qualification status."""

    def _fleet_submit_3d(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.training.fleet import (
            FleetTaskSpec,
            KubernetesJobFleet,
            LocalMultiprocessFleet,
            SlurmScriptFleet,
            WorkerBackend,
        )

        backend_kind = str(args.get("backend", "local-threads"))
        job_id = str(args.get("job_id", "fleet-3d-batch"))
        raw_tasks = args.get("tasks", [{"lx_mm": 18.0}, {"lx_mm": 22.0}])
        tasks = [
            FleetTaskSpec(task_id=f"t{i:03d}", payload=dict(item))
            for i, item in enumerate(raw_tasks)
        ]
        if backend_kind in ("local-threads", "local-multiprocess"):
            ctx.consume_fem_solves(len(tasks))
        workdir = ctx.resolve_path(str(args.get("workdir", "fleet-jobs")))
        backend: WorkerBackend
        if backend_kind == "slurm":
            backend = SlurmScriptFleet()
        elif backend_kind == "kubernetes":
            backend = KubernetesJobFleet()
        elif backend_kind == "local-multiprocess":
            backend = LocalMultiprocessFleet(
                max_workers=int(args.get("max_workers", 2)), use_threads=False
            )
        else:
            backend = LocalMultiprocessFleet(
                max_workers=int(args.get("max_workers", 2)), use_threads=True
            )
        rep = backend.submit_batch(job_id, tasks, workdir=workdir)
        return rep.document()

    reg.register(
        CapabilitySpec(
            name="governance.fleet_submit_3d",
            domain=CapabilityDomain.GOVERNANCE,
            summary="Dispatch a batch of 3D FEM tasks to Local, Slurm, or Kubernetes fleet workers.",
            description="Orchestrates parallel local solves or emits deterministic Slurm/K8s manifests.",
            input_schema={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": [
                            "local-threads",
                            "local-multiprocess",
                            "slurm",
                            "kubernetes",
                        ],
                    },
                    "job_id": {"type": "string"},
                    "workdir": {"type": "string"},
                    "max_workers": {"type": "integer", "minimum": 1, "maximum": 16},
                    "tasks": {
                        "type": "array",
                        "items": {"type": "object"},
                        "minItems": 1,
                    },
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=2,
            tags=("fleet", "slurm", "kubernetes", "parallel", "3d"),
        ),
        _fleet_submit_3d,
    )

    def _continual_status(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.serving.audit_ledger import SignedAuditLedger
        from piginot.store.dataset_3d import ShardedMeshDataset3D

        ledger_p = ctx.resolve_path(str(args.get("ledger_path", "audit-ledger.jsonl")))
        ds_p = ctx.resolve_path(str(args.get("dataset_dir", "dataset-3d")))
        ledger = SignedAuditLedger(ledger_p)
        ds = ShardedMeshDataset3D(ds_p)
        chain_valid, chain_errors = ledger.verify_chain()
        return {
            "ledger_valid": chain_valid,
            "ledger_errors": list(chain_errors),
            "ledger_entry_count": len(ledger.read_entries()),
            "fallback_count": len(ledger.query_fallbacks()),
            "dataset": ds.summary(),
        }

    reg.register(
        CapabilitySpec(
            name="governance.continual_status",
            domain=CapabilityDomain.GOVERNANCE,
            summary="Inspect SignedAuditLedger integrity, fallback backlog & sharded 3D dataset status.",
            description="Reports continual qualification readiness and hash-chain verification.",
            input_schema={
                "type": "object",
                "properties": {
                    "ledger_path": {"type": "string"},
                    "dataset_dir": {"type": "string"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("continual", "clpm", "audit-ledger", "dataset-3d"),
        ),
        _continual_status,
    )


def _register_v5_capabilities(reg: CapabilityRegistry) -> None:
    """Register FormaMech v0.5 market-gap production capabilities (V5-1 through V5-5)."""

    def _check_regime_guards_3d(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.certify.regime_guards import (
            PhysicsRegimeGuard3D,
            PhysicsRegimeGuardConfig3D,
        )
        from piginot.geometry.mesh3d import make_brick_tet4_mesh

        lx = float(args.get("lx_mm", 20.0))
        ly = float(args.get("ly_mm", 8.0))
        lz = float(args.get("lz_mm", 8.0))
        mesh = make_brick_tet4_mesh(lx, ly, lz, nx=2, ny=2, nz=2)
        cfg = PhysicsRegimeGuardConfig3D(
            yield_stress_mpa=(
                None
                if args.get("yield_stress_mpa") is None
                else float(args["yield_stress_mpa"])
            ),
            min_buckling_margin=(
                None
                if args.get("min_buckling_margin") is None
                else float(args["min_buckling_margin"])
            ),
        )
        guard = PhysicsRegimeGuard3D(cfg)
        verdict = guard.evaluate(
            mesh,
            clamped_face_tags=tuple(
                str(t) for t in args.get("clamped_face_tags", ("x_min",))
            ),
            loaded_face_tags=tuple(
                str(t) for t in args.get("loaded_face_tags", ("x_max",))
            ),
            element_peak_von_mises_mpa=(
                None
                if args.get("peak_von_mises_mpa") is None
                else np.full(
                    len(mesh.cells), float(args["peak_von_mises_mpa"]), dtype=np.float64
                )
            ),
            buckling_load_factor_override=(
                None
                if args.get("buckling_load_factor") is None
                else float(args["buckling_load_factor"])
            ),
        )
        ctx.set_receipt(verdict.verdict_sha256)
        return {**verdict.document(), "verdict_sha256": verdict.verdict_sha256}

    reg.register(
        CapabilitySpec(
            name="surrogate.check_regime_guards_3d",
            domain=CapabilityDomain.SURROGATE,
            summary="Run 3D physics-regime transition guards (yield onset, buckling proximity, BC shift).",
            description="Fail-closed physical regime checks preventing silent surrogate extrapolations.",
            input_schema={
                "type": "object",
                "properties": {
                    "lx_mm": {"type": "number", "minimum": 1.0},
                    "ly_mm": {"type": "number", "minimum": 1.0},
                    "lz_mm": {"type": "number", "minimum": 1.0},
                    "yield_stress_mpa": {"type": "number", "minimum": 1.0},
                    "peak_von_mises_mpa": {"type": "number", "minimum": 0.0},
                    "min_buckling_margin": {"type": "number", "minimum": 0.1},
                    "buckling_load_factor": {"type": "number"},
                    "clamped_face_tags": {"type": "array", "items": {"type": "string"}},
                    "loaded_face_tags": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("regime-guard", "yield", "buckling", "ood", "3d", "v5"),
        ),
        _check_regime_guards_3d,
    )

    def _ground_surrogate_with_tests(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.ui.workbench_3d.vv10_panel import run_hybrid_grounding_summary

        ds_path = ctx.resolve_path(str(args["dataset_json"]))
        summary = run_hybrid_grounding_summary(
            ds_path,
            u_num=float(args.get("u_num", 0.01)),
            u_input=float(args.get("u_input", 0.0)),
            batch_size=int(args.get("batch_size", 3)),
        )
        ctx.set_receipt(str(summary["receipt_sha256"]))
        return summary

    reg.register(
        CapabilitySpec(
            name="validate.ground_surrogate_with_tests",
            domain=CapabilityDomain.INGEST_VALIDATE,
            summary="Fuse simulation priors with physical component tests under ASME V&V 10.",
            description="Fits Kennedy-O'Hagan discrepancy GP, checks blind physical holdouts, and ranks next tests.",
            input_schema={
                "type": "object",
                "required": ["dataset_json"],
                "properties": {
                    "dataset_json": {"type": "string"},
                    "u_num": {"type": "number", "minimum": 0.0},
                    "u_input": {"type": "number", "minimum": 0.0},
                    "batch_size": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("hybrid-grounding", "vv10", "dic", "experimental", "v5"),
        ),
        _ground_surrogate_with_tests,
    )

    def _recommend_physical_test(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.active.test_planner import (
            PhysicalTestPlanConfig,
            recommend_next_physical_tests,
        )
        from piginot.ui.workbench_3d.vv10_panel import load_component_tests_json
        from piginot.validation.contracts import CouponRole

        ds_path = ctx.resolve_path(str(args["dataset_json"]))
        records, candidates = load_component_tests_json(ds_path)
        cal_records = [r for r in records if r.role is CouponRole.CALIBRATION]
        plan = recommend_next_physical_tests(
            cal_records,
            candidates,
            config=PhysicalTestPlanConfig(
                batch_size=int(args.get("batch_size", 2)),
                backend_name=str(args.get("backend", "rbf-gp-mi")),
            ),
        )
        ctx.set_receipt(plan.plan_hash)
        return {**plan.document(), "plan_hash": plan.plan_hash}

    reg.register(
        CapabilitySpec(
            name="active.recommend_physical_test",
            domain=CapabilityDomain.INGEST_VALIDATE,
            summary="Recommend highest-information-per-cost next physical laboratory tests.",
            description="Maximizes GP mutual information + sim uncertainty while refusing holdout/unsafe conditions.",
            input_schema={
                "type": "object",
                "required": ["dataset_json"],
                "properties": {
                    "dataset_json": {"type": "string"},
                    "batch_size": {"type": "integer", "minimum": 1, "maximum": 20},
                    "backend": {
                        "type": "string",
                        "enum": ["rbf-gp-mi", "botorch", "baybe"],
                    },
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("active-learning", "test-plan", "bayesian-opt", "v5"),
        ),
        _recommend_physical_test,
    )

    def _estimate_breakeven(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.training.archive_trainer import estimate_archive_breakeven

        del ctx
        comp = estimate_archive_breakeven(
            expected_queries=int(args["expected_queries"]),
            fem_seconds_per_solve=float(args["fem_seconds_per_solve"]),
            inference_seconds=float(args.get("inference_seconds", 0.01)),
            from_scratch_solves=int(args.get("from_scratch_solves", 300)),
            from_scratch_train_s=float(args.get("from_scratch_train_s", 600.0)),
            lora_target_solves=int(args.get("lora_target_solves", 30)),
            lora_adapt_s=float(args.get("lora_adapt_s", 60.0)),
            archive_already_paid=bool(args.get("archive_already_paid", True)),
        )
        return comp.document()

    reg.register(
        CapabilitySpec(
            name="surrogate.estimate_archive_breakeven",
            domain=CapabilityDomain.SURROGATE,
            summary="Pre-flight N* economics gate comparing direct FEM, archive-LoRA, and from-scratch.",
            description="Prevents negative-ROI surrogate training campaigns before spending compute.",
            input_schema={
                "type": "object",
                "required": ["expected_queries", "fem_seconds_per_solve"],
                "properties": {
                    "expected_queries": {"type": "integer", "minimum": 1},
                    "fem_seconds_per_solve": {"type": "number", "minimum": 0.001},
                    "inference_seconds": {"type": "number", "minimum": 1e-6},
                    "from_scratch_solves": {"type": "integer", "minimum": 1},
                    "from_scratch_train_s": {"type": "number", "minimum": 0.0},
                    "lora_target_solves": {"type": "integer", "minimum": 1},
                    "lora_adapt_s": {"type": "number", "minimum": 0.0},
                    "archive_already_paid": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("amortization", "roi", "economics", "v5"),
        ),
        _estimate_breakeven,
    )

    def _run_industry_bench(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.benchmarks.industry_suite import (
            IndustryBenchmarkConfig,
            run_industry_benchmark_suite,
        )

        scales = tuple(int(s) for s in args.get("scales", (3, 5, 7)))
        work_dir = ctx.resolve_path(str(args.get("output_dir", "industry_bench")))
        cfg = IndustryBenchmarkConfig(
            dataset_name=str(args.get("dataset_name", "deepjeb")),
            scales=scales,
            epochs_per_scale=int(args.get("epochs_per_scale", 6)),
            hidden_dim=int(args.get("hidden_dim", 16)),
            allow_noncommercial=bool(args.get("allow_noncommercial", False)),
            allow_synthetic_fallback=bool(args.get("allow_synthetic_fallback", True)),
            seed=int(args.get("seed", 0)),
        )
        rep = run_industry_benchmark_suite(cfg, work_dir=work_dir)
        ctx.record_artifact(work_dir / "industry_benchmark.json")
        ctx.record_artifact(work_dir / "industry_benchmark.md")
        ctx.set_receipt(rep.report_hash)
        return rep.document()

    reg.register(
        CapabilitySpec(
            name="governance.run_industry_benchmarks",
            domain=CapabilityDomain.GOVERNANCE,
            summary="Run neutral multi-scale + BC-shift industry benchmark suite (JSON + Markdown).",
            description="Reports mean, p95, and worst-case physical-unit errors, OOD AUROC, and N*.",
            input_schema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string"},
                    "scales": {"type": "array", "items": {"type": "integer"}},
                    "epochs_per_scale": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "hidden_dim": {"type": "integer", "minimum": 8, "maximum": 128},
                    "allow_noncommercial": {"type": "boolean"},
                    "allow_synthetic_fallback": {"type": "boolean"},
                    "output_dir": {"type": "string"},
                    "seed": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.TRAINING_JOB,
            tags=("benchmark", "physicsbench", "bc-shift", "v5"),
        ),
        _run_industry_bench,
    )

    def _synthesize_brep_3d(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.cad.brep_synthesis import (
            KeepOutDatumFeature,
            synthesize_watertight_brep_step,
        )
        from piginot.geometry.mesh3d import make_brick_tet4_mesh

        lx = float(args.get("lx_mm", 24.0))
        ly = float(args.get("ly_mm", 12.0))
        lz = float(args.get("lz_mm", 8.0))
        mesh = make_brick_tet4_mesh(lx, ly, lz, nx=3, ny=2, nz=2)
        mask = np.ones(len(mesh.cells), dtype=np.bool_)
        feats: list[KeepOutDatumFeature] = []
        if args.get("bore_radius_mm") is not None:
            feats.append(
                KeepOutDatumFeature(
                    name="datum-bore",
                    kind="cylindrical_bore",
                    origin_mm=(0.5 * lx, 0.5 * ly, 0.0),
                    axis=(0.0, 0.0, 1.0),
                    radius_mm=float(args["bore_radius_mm"]),
                    length_mm=lz,
                )
            )
        out_step = ctx.resolve_path(str(args.get("output_step", "synthesized.step")))
        res = synthesize_watertight_brep_step(
            mesh,
            mask,
            out_step,
            keep_out_features=tuple(feats),
            verify_roundtrip=True,
            run_remeshed_fem_check=bool(args.get("run_remeshed_fem_check", True)),
        )
        ctx.record_artifact(res.step_path)
        ctx.set_receipt(res.receipt_sha256)
        return {**res.document(), "receipt_sha256": res.receipt_sha256}

    reg.register(
        CapabilitySpec(
            name="cad.synthesize_brep_3d",
            domain=CapabilityDomain.CAD,
            summary="Synthesize watertight OpenCASCADE B-Rep STEP with exact bores & round-trip FEM.",
            description="Verifies zero open edges, <=0.05 mm bore accuracy, and exact re-meshed 3D FEM.",
            input_schema={
                "type": "object",
                "properties": {
                    "lx_mm": {"type": "number", "minimum": 4.0},
                    "ly_mm": {"type": "number", "minimum": 4.0},
                    "lz_mm": {"type": "number", "minimum": 2.0},
                    "bore_radius_mm": {"type": "number", "minimum": 0.5},
                    "output_step": {"type": "string"},
                    "run_remeshed_fem_check": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("brep", "step", "cad", "dfm", "v5"),
        ),
        _synthesize_brep_3d,
    )

    def _record_hitl_signoff(
        ctx: HarnessContext, args: Mapping[str, Any]
    ) -> dict[str, Any]:
        from piginot.governance.hitl_signoff import (
            HandCalculationSanityCheck,
            ReviewerRole,
            create_hitl_signoff,
        )
        from piginot.serving.audit_ledger import SignedAuditLedger

        hand_calc = None
        if args.get("hand_calc_note") is not None:
            hand_calc = HandCalculationSanityCheck(
                hand_calc_note=str(args["hand_calc_note"]),
                hand_calc_value=float(args["hand_calc_value"]),
                model_predicted_value=float(args["model_predicted_value"]),
                unit=str(args.get("unit", "MPa")),
            )
        ledger = SignedAuditLedger(ctx.resolve_path("hitl-audit-ledger.jsonl"))
        rec = create_hitl_signoff(
            artifact_id=str(args["artifact_id"]),
            artifact_sha256=str(args["artifact_sha256"]),
            reviewer_id=str(args["reviewer_id"]),
            reviewer_role=ReviewerRole(
                str(args.get("reviewer_role", "LEAD_STRESS_ENGINEER"))
            ),
            approved=bool(args["approved"]),
            decision_note=str(args["decision_note"]),
            hand_calc=hand_calc,
            audit_ledger=ledger,
        )
        ctx.set_receipt(rec.signature_sha256)
        return rec.document()

    reg.register(
        CapabilitySpec(
            name="governance.record_hitl_signoff",
            domain=CapabilityDomain.GOVERNANCE,
            summary="Record cryptographically signed senior-engineer HITL release review.",
            description="Enforces DRAFT -> PENDING_HUMAN_REVIEW -> APPROVED | REJECTED_WITH_FEEDBACK.",
            input_schema={
                "type": "object",
                "required": [
                    "artifact_id",
                    "artifact_sha256",
                    "reviewer_id",
                    "approved",
                    "decision_note",
                ],
                "properties": {
                    "artifact_id": {"type": "string"},
                    "artifact_sha256": {"type": "string"},
                    "reviewer_id": {"type": "string"},
                    "reviewer_role": {
                        "type": "string",
                        "enum": [
                            "LEAD_STRESS_ENGINEER",
                            "VALIDITY_OWNER",
                            "CHIEF_ENGINEER",
                        ],
                    },
                    "approved": {"type": "boolean"},
                    "decision_note": {"type": "string"},
                    "hand_calc_note": {"type": "string"},
                    "hand_calc_value": {"type": "number"},
                    "model_predicted_value": {"type": "number"},
                    "unit": {"type": "string"},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.ZERO_SOLVE,
            tags=("hitl", "signoff", "compliance", "eu-ai-act", "v5"),
        ),
        _record_hitl_signoff,
    )

    def _solve_etm_3d(ctx: HarnessContext, args: Mapping[str, Any]) -> dict[str, Any]:
        from piginot.cad.electronics_templates_3d import (
            generate_busway_3d,
            generate_chiplet_package_3d,
        )
        from piginot.fem.backend_3d import DirichletBC3D
        from piginot.fem.electro_thermal_3d import (
            VoltageBC3D,
            solve_3d_electro_thermal_mechanical,
        )
        from piginot.fem.thermo_mechanical_3d import ConvectionBC3D, TemperatureBC3D

        template = str(args.get("template", "parametric-chiplet-package-3d"))
        ctx.consume_fem_solves(1)
        if template == "parametric-busway-3d":
            bundle = generate_busway_3d(
                length_mm=float(args.get("length_mm", 40.0)),
                width_mm=float(args.get("width_mm", 10.0)),
                thickness_mm=float(args.get("thickness_mm", 4.0)),
                nx=4,
                ny=2,
                nz=2,
            )
            res = solve_3d_electro_thermal_mechanical(
                bundle.assembly,
                bundle.etm_materials,
                dirichlet_bcs=(
                    DirichletBC3D("fix", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
                ),
                voltage_bcs=(
                    VoltageBC3D("v0", ("x_min",), 0.0),
                    VoltageBC3D(
                        "v1", ("x_max",), float(args.get("voltage_drop_v", 0.02))
                    ),
                ),
                temperature_bcs=(TemperatureBC3D("t0", ("x_min",), 293.15),),
                convection_bcs=(
                    ConvectionBC3D(
                        "cool", ("z_max",), h_w_per_mm2_k=0.001, t_ambient_k=293.15
                    ),
                ),
                load_steps=1,
            )
        else:
            bundle = generate_chiplet_package_3d(
                package_xy_mm=float(args.get("package_xy_mm", 20.0)),
                chiplet_power_w=float(args.get("chiplet_power_w", 15.0)),
                nx=3,
                ny=3,
            )
            res = solve_3d_electro_thermal_mechanical(
                bundle.assembly,
                bundle.etm_materials,
                dirichlet_bcs=(
                    DirichletBC3D(
                        "mount", ("z_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)
                    ),
                ),
                convection_bcs=(
                    ConvectionBC3D(
                        "lid_cool", ("z_max",), h_w_per_mm2_k=0.005, t_ambient_k=298.15
                    ),
                ),
                volumetric_sources=bundle.power_sources,
                load_steps=1,
            )
        return {**res.document(), "template": template}

    reg.register(
        CapabilitySpec(
            name="multiphysics.solve_electro_thermal_3d",
            domain=CapabilityDomain.MULTIPHYSICS,
            summary="Solve coupled multi-material 3D Electro-Thermal-Mechanical equilibrium.",
            description="Supports 3D-IC chiplet packages (orthotropic substrate + Si + Cu) and Joule-heated busways.",
            input_schema={
                "type": "object",
                "properties": {
                    "template": {
                        "type": "string",
                        "enum": [
                            "parametric-chiplet-package-3d",
                            "parametric-busway-3d",
                        ],
                    },
                    "package_xy_mm": {"type": "number", "minimum": 5.0},
                    "chiplet_power_w": {"type": "number", "minimum": 0.1},
                    "length_mm": {"type": "number", "minimum": 5.0},
                    "width_mm": {"type": "number", "minimum": 2.0},
                    "thickness_mm": {"type": "number", "minimum": 1.0},
                    "voltage_drop_v": {"type": "number", "minimum": 0.0001},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("electro-thermal", "chiplet", "joule", "warpage", "3d", "v5"),
        ),
        _solve_etm_3d,
    )


def _register_v6_robotics_capabilities(reg: CapabilityRegistry) -> None:
    """Register V6-R1..V6-R6 robotics continuum FEM, tactile, dynamics & control tools."""

    def _solve_follower_3d(
        _ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.contracts import MaterialSpec, Quantity, Unit
        from piginot.fem.backend_3d import DirichletBC3D
        from piginot.fem.follower_3d import (
            FollowerPressureBC3D,
            solve_3d_follower_pressure,
        )
        from piginot.geometry.mesh3d import make_brick_tet10_mesh

        mesh = make_brick_tet10_mesh(
            float(args.get("lx_mm", 20.0)),
            float(args.get("ly_mm", 8.0)),
            float(args.get("lz_mm", 4.0)),
            nx=int(args.get("nx", 3)),
            ny=int(args.get("ny", 2)),
            nz=int(args.get("nz", 1)),
        )
        mat = MaterialSpec(
            name="robotics-elastomer",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(
                    float(args.get("youngs_modulus_mpa", 2.0)), Unit.MPA
                ).to_canonical(),
                "poisson_ratio": Quantity(
                    float(args.get("poisson_ratio", 0.45)), Unit.ONE
                ).to_canonical(),
            },
        )
        res = solve_3d_follower_pressure(
            mesh,
            mat,
            dirichlet_bcs=(
                DirichletBC3D("clamp", ("x_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
            ),
            follower_bcs=(
                FollowerPressureBC3D(
                    "top_pressure",
                    ("z_max",),
                    pressure_mpa=float(args.get("pressure_mpa", 0.02)),
                ),
            ),
            load_steps=int(args.get("load_steps", 2)),
        )
        return res.summary_document()

    reg.register(
        CapabilitySpec(
            name="fem.solve_follower_pressure_3d",
            domain=CapabilityDomain.FEM,
            summary="Solve 3D finite-strain TET10 FEM under configuration-dependent follower pressure.",
            description="Computes exact Nanson's-law 3D follower pressure equilibrium and unsymmetric load stiffness.",
            input_schema={
                "type": "object",
                "properties": {
                    "lx_mm": {"type": "number", "minimum": 2.0},
                    "ly_mm": {"type": "number", "minimum": 2.0},
                    "lz_mm": {"type": "number", "minimum": 1.0},
                    "nx": {"type": "integer", "minimum": 1, "maximum": 8},
                    "ny": {"type": "integer", "minimum": 1, "maximum": 8},
                    "nz": {"type": "integer", "minimum": 1, "maximum": 4},
                    "youngs_modulus_mpa": {"type": "number", "minimum": 0.05},
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "pressure_mpa": {"type": "number"},
                    "load_steps": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("robotics", "follower-pressure", "soft-robotics", "3d", "v6"),
        ),
        _solve_follower_3d,
    )

    def _solve_tendon_continuum(
        _ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.fem.tendon_3d import (
            CosseratBackboneSpec,
            solve_3d_tendon_continuum_backbone,
        )

        spec = CosseratBackboneSpec(
            length_mm=float(args.get("length_mm", 540.0)),
            num_disks=int(args.get("num_disks", 9)),
            youngs_modulus_mpa=float(args.get("youngs_modulus_mpa", 55_000.0)),
            poisson_ratio=float(args.get("poisson_ratio", 0.33)),
            outer_radius_mm=float(args.get("outer_radius_mm", 1.1)),
            inner_radius_mm=float(args.get("inner_radius_mm", 0.6)),
        )
        tensions = tuple(
            float(v) for v in args.get("tendon_tensions_n", [5.0, 0.0, 0.0, 0.0])
        )
        res = solve_3d_tendon_continuum_backbone(spec, tensions)
        return res.summary_document()

    reg.register(
        CapabilitySpec(
            name="fem.solve_tendon_continuum_3d",
            domain=CapabilityDomain.FEM,
            summary="Solve exact finite-rotation 3D Cosserat tendon-driven continuum backbone.",
            description="Returns 6-DOF disk poses, FBG bending microstrains, tendon strokes, and base 6D wrench.",
            input_schema={
                "type": "object",
                "properties": {
                    "length_mm": {"type": "number", "minimum": 10.0},
                    "num_disks": {"type": "integer", "minimum": 2, "maximum": 30},
                    "youngs_modulus_mpa": {"type": "number", "minimum": 1.0},
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "outer_radius_mm": {"type": "number", "minimum": 0.1},
                    "inner_radius_mm": {"type": "number", "minimum": 0.0},
                    "tendon_tensions_n": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0.0},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("robotics", "continuum", "tendon", "cosserat", "fbg", "v6"),
        ),
        _solve_tendon_continuum,
    )

    def _solve_sdf_contact(
        _ctx: HarnessContext, args: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        from piginot.fem.diff_contact_3d import SdfPrimitiveKind
        from piginot.robotics.tactile import (
            TactileSensorSpec,
            simulate_tactile_probe_fem,
        )

        spec = TactileSensorSpec(
            youngs_modulus_mpa=float(args.get("youngs_modulus_mpa", 0.25)),
            poisson_ratio=float(args.get("poisson_ratio", 0.48)),
            mesh_div_xyz=(3, 4, 1),
        )
        sample = simulate_tactile_probe_fem(
            spec,
            case_id="harness-probe",
            probe_x_mm=float(args.get("probe_x_mm", 9.0)),
            probe_y_mm=float(args.get("probe_y_mm", 12.0)),
            depth_mm=float(args.get("depth_mm", 0.6)),
            shear_dx_mm=float(args.get("shear_dx_mm", 0.0)),
            shear_dy_mm=float(args.get("shear_dy_mm", 0.0)),
            indenter_radius_mm=float(args.get("indenter_radius_mm", 4.0)),
            indenter_kind=SdfPrimitiveKind(str(args.get("indenter_kind", "sphere"))),
            load_steps=2,
        )
        return {
            "case_id": sample.case_id,
            "wrench_6d": list(sample.wrench_6d),
            "max_slip_ratio": sample.max_slip_ratio,
            "peak_von_mises_mpa": sample.peak_von_mises_mpa,
            "fem_wall_time_s": sample.fem_wall_time_s,
        }

    reg.register(
        CapabilitySpec(
            name="fem.solve_sdf_contact_3d",
            domain=CapabilityDomain.FEM,
            summary="Solve 3D TET10 elastomer SDF contact and extract per-taxel wrench & slip ratio.",
            description="Supports sphere, flat_punch, cylinder_pin, box, and wedge_edge rigid indenters with Coulomb friction.",
            input_schema={
                "type": "object",
                "properties": {
                    "youngs_modulus_mpa": {"type": "number", "minimum": 0.01},
                    "poisson_ratio": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 0.49,
                    },
                    "probe_x_mm": {"type": "number"},
                    "probe_y_mm": {"type": "number"},
                    "depth_mm": {"type": "number", "minimum": 0.05, "maximum": 2.5},
                    "shear_dx_mm": {"type": "number"},
                    "shear_dy_mm": {"type": "number"},
                    "indenter_radius_mm": {"type": "number", "minimum": 0.5},
                    "indenter_kind": {
                        "type": "string",
                        "enum": [
                            "sphere",
                            "flat_punch",
                            "cylinder_pin",
                            "box",
                            "wedge_edge",
                        ],
                    },
                },
                "additionalProperties": False,
            },
            cost_class=CostClass.SOLVER_ORACLE,
            default_fem_solves=1,
            tags=("robotics", "tactile", "sdf-contact", "elastomer", "3d", "v6"),
        ),
        _solve_sdf_contact,
    )


def build_default_registry(*, include_plugins: bool = True) -> CapabilityRegistry:
    """Create a populated :class:`CapabilityRegistry` covering all FormaMech domains."""
    reg = CapabilityRegistry()
    _register_discover_capabilities(reg)
    _register_cad_capabilities(reg)
    _register_fem_and_multiphysics_capabilities(reg)
    _register_surrogate_opt_ingest_export_governance(reg)
    _register_v4_physics_capabilities(reg)
    _register_v4_generative_capabilities(reg)
    _register_v4_fleet_continual_capabilities(reg)
    _register_v5_capabilities(reg)
    _register_v6_robotics_capabilities(reg)
    if include_plugins:
        reg.load_entry_point_plugins()
    return reg


__all__ = ("build_default_registry",)
