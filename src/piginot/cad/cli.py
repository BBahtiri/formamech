"""Handlers for CAD geometry preparation, hierarchy, convergence, and adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from piginot.cad.configuration import prepare_campaign
from piginot.cad.mesh_policy import MeshPolicy
from piginot.cad.project import (
    build_project_hierarchy,
    evaluate_project_geometry_convergence,
    inspect_geometry,
    prepare_geometry,
)
from piginot.cad.resource import load_geometry_resource
from piginot.contracts import MaterialSpec, Quantity, Unit


def _run_cad_adapt(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Adapt a pretrained ``BoundaryEnergyOperator`` to a CAD project via LoRA."""
    import torch

    from piginot.lifting import AffineDirichletLifting
    from piginot.objectives.energy import PotentialEnergyConfig, TriangleDomain
    from piginot.operators import (
        AdaptationPlan,
        AdaptationStatus,
        BoundaryRoleMap,
        GeometryTokenizer,
        LowRankAdapterConfig,
        adapt_boundary_operator_energy,
        boundary_energy_module_digest,
        load_boundary_operator_checkpoint,
        load_geometry_ood_detector,
        save_geometry_adapter,
    )

    project_manifest = Path(args.project) / "geometry.json"
    raw = project_manifest.read_bytes()
    resource = load_geometry_resource(project_manifest, hashlib.sha256(raw).hexdigest())
    mesh = resource.mesh(1)

    cfg_doc = json.loads(Path(args.config).read_text(encoding="utf-8"))
    roles = BoundaryRoleMap(dict(cfg_doc["boundary_roles"]))
    base_op = load_boundary_operator_checkpoint(Path(args.checkpoint))
    tok = GeometryTokenizer(
        subdivisions_per_facet=int(cfg_doc.get("subdivisions_per_facet", 1))
    )
    ood_report = None
    if args.ood_detector is not None:
        detector = load_geometry_ood_detector(Path(args.ood_detector))
        ood_report = detector.score(mesh, roles, tokenizer=tok)

    dtype = next(base_op.parameters()).dtype
    batch = tok.tokenize_batch([(mesh, roles)], dtype=dtype)
    domain = TriangleDomain(
        vertices=torch.tensor(mesh.nodes, dtype=dtype),
        cells=torch.tensor(mesh.cells, dtype=torch.int64),
    )

    mat_doc = cfg_doc["material"]
    material = MaterialSpec(
        name=str(mat_doc.get("name", "cad-adapt-material")),
        model=str(mat_doc.get("model", "compressible_neo_hookean")),
        properties={
            str(k): Quantity(float(v["value"]), Unit(str(v["unit"]))).to_canonical()
            for k, v in mat_doc["properties"].items()
        },
    )
    strain = float(cfg_doc.get("nominal_strain", 0.10))
    origin_x = float(batch.frame.origin[0, 0].item())
    origin_y = float(batch.frame.origin[0, 1].item())
    extent_x = float(batch.frame.extent[0, 0].item())
    extent_y = float(batch.frame.extent[0, 1].item())

    def _lift(q: torch.Tensor) -> AffineDirichletLifting:
        sx = (q[..., 0] - origin_x) / extent_x
        sy = (q[..., 1] - origin_y) / extent_y
        offset = torch.stack(
            (strain * (q[..., 0] - origin_x), torch.zeros_like(sy)), dim=-1
        )
        scale = torch.stack((4.0 * sx * (1.0 - sx), sy), dim=-1)
        return AffineDirichletLifting(offset=offset, scale=scale)

    param_vals = [float(v) for v in cfg_doc.get("parameter_vector", [0.0])]
    params_t = torch.tensor([param_vals], dtype=dtype)
    ad_doc = dict(cfg_doc.get("adapter", {}))
    if "target_substrings" in ad_doc:
        ad_doc["target_substrings"] = tuple(ad_doc["target_substrings"])
    adapter_cfg = LowRankAdapterConfig(**ad_doc)
    en_doc = dict(cfg_doc.get("energy", {}))
    energy_cfg = PotentialEnergyConfig(**en_doc)
    plan = AdaptationPlan(
        base_model_digest=boundary_energy_module_digest(base_op),
        target_mesh_id=mesh.mesh_id,
        adapter_config=adapter_cfg,
        energy_config=energy_cfg,
        ood_report=ood_report,
    )
    adapted_op, adapt_res = adapt_boundary_operator_energy(
        base_op,
        plan=plan,
        domain=domain,
        boundary_xy=batch.boundary_xy,
        boundary_normals=batch.boundary_normals,
        boundary_features=batch.boundary_features,
        boundary_weights=batch.weights,
        boundary_mask=batch.mask,
        frame=batch.frame,
        parameters=params_t,
        lifting=_lift,
        material=material,
    )
    if adapted_op is None or adapt_res.status is not AdaptationStatus.COMPLETED:
        return (
            {
                "status": adapt_res.status.value,
                "reason": adapt_res.reason,
                "result": adapt_res.document(),
                "ood_report": None if ood_report is None else ood_report.document(),
                "claim_level": "NOT_VERIFIED",
            },
            2,
        )
    manifest = save_geometry_adapter(
        Path(args.output),
        adapted_operator=adapted_op,
        result=adapt_res,
        ood_report=ood_report,
    )
    return manifest, 0


def run(args: argparse.Namespace) -> int:
    try:
        if args.cad_command == "import":
            sizing = {"mesh_size_mm": args.mesh_size_mm}
            if args.mesh_policy:
                sizing = {"mesh_policy": MeshPolicy.load(args.mesh_policy)}
            result = prepare_geometry(
                args.source,
                args.output,
                unit=Unit(args.unit),
                **sizing,
            )
            if getattr(args, "hierarchy", False):
                hier = build_project_hierarchy(
                    args.output,
                    output=args.output / "hierarchy.json",
                )
                result = {
                    **result,
                    "hierarchy_path": str(args.output / "hierarchy.json"),
                    "hierarchy_hash": hier["hierarchy_hash"],
                }
        elif args.cad_command == "inspect":
            result = inspect_geometry(args.project)
        elif args.cad_command == "prepare":
            result = prepare_campaign(
                args.project,
                args.settings,
                args.output,
                execution_profile=args.execution_profile,
            )
        elif args.cad_command == "hierarchy":
            result = build_project_hierarchy(
                args.project,
                output=args.output,
                geometry_tolerance_mm=args.geometry_tolerance_mm,
            )
        elif args.cad_command == "convergence":
            result = evaluate_project_geometry_convergence(
                args.source,
                unit=Unit(args.unit),
                mesh_sizes_mm=tuple(float(v) for v in args.mesh_sizes),
                max_relative_area_change=args.max_area_change,
                max_relative_perimeter_change=args.max_perimeter_change,
                max_chord_error_mm=args.max_chord_error_mm,
                output=args.output,
            )
        elif args.cad_command == "adapt":
            result, code = _run_cad_adapt(args)
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return code
        else:
            raise ValueError(f"unknown CAD action: {args.cad_command}")
    except (OSError, ValueError, ImportError, KeyError, TypeError) as error:
        print(
            json.dumps({"status": "REFUSED", "reason": str(error)}, allow_nan=False),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0
