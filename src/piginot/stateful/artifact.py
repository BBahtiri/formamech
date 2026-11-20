"""Portable ``safetensors`` viscoelastic operator artifacts & CLI rollout runner (P6).

Provides tamper-checked save/load of a trained ``HistoryAwareViscoelasticOperator``
plus its material configuration, mesh, and query frame, and a self-contained
rollout driver supporting both pure-operator evaluation and
``GuardedStatefulStepper`` execution with checkpoint save/resume.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import torch

from piginot.contracts import (
    CanonicalMesh,
    Unit,
    canonical_json_bytes,
    strict_json_loads,
)
from piginot.contracts.stateful import (
    BoundTrajectoryState,
    LoadSegment,
    StatefulVerificationPolicy,
    TimeGrid,
    evaluate_load_path,
)
from piginot.geometry import CellType, MeshPayload, canonicalize_mesh
from piginot.lifting import AffineDirichletLifting
from piginot.objectives.energy import TriangleDomain
from piginot.operators.energy_operator import QueryFrame
from piginot.stateful.stepper import (
    STEP_CONTROLLER_VERSION,
    CandidateStepOutput,
    GuardedStatefulStepper,
    hash_state_tensors,
    load_trajectory_checkpoint,
    save_trajectory_checkpoint,
)
from piginot.stateful.viscoelastic import (
    GeneralizedMaxwellConfig,
    GeneralizedMaxwellViscoelastic,
    MaxwellBranch,
)
from piginot.stateful.viscoelastic_operator import (
    HistoryAwareViscoelasticOperator,
    ViscoelasticOperatorConfig,
    project_cell_state_to_nodes,
    summarize_viscous_state,
)

VISCOELASTIC_ARTIFACT_SCHEMA: Final = "piginot.viscoelastic-operator-artifact-v1"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedViscoelasticArtifact:
    """Verified viscoelastic neural operator + constitutive model + reference mesh."""

    operator: HistoryAwareViscoelasticOperator
    material: GeneralizedMaxwellViscoelastic
    mesh: CanonicalMesh
    frame: QueryFrame
    manifest: dict[str, Any]
    manifest_sha256: str


def save_viscoelastic_artifact(
    directory: Path,
    *,
    operator: HistoryAwareViscoelasticOperator,
    material: GeneralizedMaxwellViscoelastic,
    mesh: CanonicalMesh,
    frame: QueryFrame,
    problem_spec_hash: str = "0" * 64,
    claim_level: str = "NOT_VERIFIED",
) -> dict[str, Any]:
    """Save a trained viscoelastic operator artifact (``safetensors`` + canonical JSON)."""
    from safetensors.torch import save as save_tensors

    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in operator.state_dict().items()}
    raw_weights = save_tensors(state)
    (directory / "weights.safetensors").write_bytes(raw_weights)

    mesh_bytes = canonical_json_bytes(
        {
            "nodes": [list(p) for p in mesh.nodes],
            "cells": [list(c) for c in mesh.cells],
            "boundary_facets": [list(f) for f in mesh.boundary_facets],
            "semantic_tags": {
                k: list(v) for k, v in sorted(mesh.semantic_tags.items())
            },
            "cell_tags": list(mesh.region_ids),
            "unit": mesh.unit.value,
            "metadata": dict(mesh.source_provenance),
        }
    )
    (directory / "mesh.json").write_bytes(mesh_bytes + b"\n")

    cfg = operator.config
    mat_cfg = material.config
    body: dict[str, Any] = {
        "schema": VISCOELASTIC_ARTIFACT_SCHEMA,
        "problem_spec_hash": problem_spec_hash,
        "claim_level": claim_level,
        "mesh_id": mesh.mesh_id,
        "mesh_sha256": _sha256_bytes(mesh_bytes + b"\n"),
        "weights_file": "weights.safetensors",
        "weights_sha256": _sha256_bytes(raw_weights),
        "dtype": str(next(operator.parameters()).dtype),
        "operator_config": {
            "load_dim": cfg.load_dim,
            "n_branches": cfg.n_branches,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "positional_frequencies": cfg.positional_frequencies,
            "output_init_std": cfg.output_init_std,
            "include_local_state": cfg.include_local_state,
        },
        "material_config": {
            "equilibrium_modulus_mpa": mat_cfg.equilibrium_modulus_mpa,
            "poisson_ratio": mat_cfg.poisson_ratio,
            "branches": [
                {
                    "stiffness_ratio": b.stiffness_ratio,
                    "relaxation_time": b.relaxation_time,
                }
                for b in mat_cfg.branches
            ],
        },
        "frame": {
            "origin": [float(v) for v in frame.origin.reshape(-1).tolist()],
            "extent": [float(v) for v in frame.extent.reshape(-1).tolist()],
        },
    }
    manifest = {**body, "manifest_sha256": _sha256_bytes(canonical_json_bytes(body))}
    (directory / "artifact.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_viscoelastic_artifact(directory: Path) -> LoadedViscoelasticArtifact:
    """Verify every SHA-256 and load a ``LoadedViscoelasticArtifact``."""
    from safetensors.torch import load as load_tensors

    parsed = strict_json_loads((directory / "artifact.json").read_bytes())
    if (
        not isinstance(parsed, dict)
        or parsed.get("schema") != VISCOELASTIC_ARTIFACT_SCHEMA
    ):
        raise ValueError("invalid viscoelastic artifact manifest")
    raw = cast(dict[str, Any], parsed)
    expected_sha = str(raw.get("manifest_sha256", ""))
    body: dict[str, Any] = {k: v for k, v in raw.items() if k != "manifest_sha256"}
    if _sha256_bytes(canonical_json_bytes(body)) != expected_sha:
        raise ValueError("viscoelastic artifact manifest SHA-256 mismatch")

    raw_weights = (directory / str(body["weights_file"])).read_bytes()
    if _sha256_bytes(raw_weights) != body["weights_sha256"]:
        raise ValueError("viscoelastic artifact weights SHA-256 mismatch")

    raw_mesh_bytes = (directory / "mesh.json").read_bytes()
    if _sha256_bytes(raw_mesh_bytes) != body["mesh_sha256"]:
        raise ValueError("viscoelastic artifact mesh SHA-256 mismatch")
    m_doc = cast(dict[str, Any], strict_json_loads(raw_mesh_bytes))
    mesh = canonicalize_mesh(
        MeshPayload(
            tuple((float(p[0]), float(p[1])) for p in m_doc["nodes"]),
            tuple((int(c[0]), int(c[1]), int(c[2])) for c in m_doc["cells"]),
            CellType.TRIANGLE,
            tuple((int(f[0]), int(f[1])) for f in m_doc["boundary_facets"]),
            {
                str(k): tuple(int(i) for i in v)
                for k, v in m_doc["semantic_tags"].items()
            },
            tuple(int(t) for t in m_doc["cell_tags"]),
            Unit(str(m_doc["unit"])),
            {str(k): str(v) for k, v in m_doc["metadata"].items()},
        )
    )
    if mesh.mesh_id != body["mesh_id"]:
        raise ValueError("viscoelastic artifact mesh_id mismatch")

    dtype = torch.float32 if body.get("dtype") == "torch.float32" else torch.float64
    op_cfg = ViscoelasticOperatorConfig(**body["operator_config"])
    op = HistoryAwareViscoelasticOperator(op_cfg).to(dtype)
    op.load_state_dict(dict(load_tensors(raw_weights)), strict=True)
    op.eval()

    mat_doc = body["material_config"]
    material = GeneralizedMaxwellViscoelastic(
        GeneralizedMaxwellConfig(
            equilibrium_modulus_mpa=float(mat_doc["equilibrium_modulus_mpa"]),
            poisson_ratio=float(mat_doc["poisson_ratio"]),
            branches=tuple(
                MaxwellBranch(
                    stiffness_ratio=float(b["stiffness_ratio"]),
                    relaxation_time=float(b["relaxation_time"]),
                )
                for b in mat_doc["branches"]
            ),
        )
    )
    frame = QueryFrame(
        origin=torch.tensor([body["frame"]["origin"]], dtype=dtype),
        extent=torch.tensor([body["frame"]["extent"]], dtype=dtype),
    )
    return LoadedViscoelasticArtifact(
        operator=op,
        material=material,
        mesh=mesh,
        frame=frame,
        manifest=raw,
        manifest_sha256=expected_sha,
    )


def run_viscoelastic_rollout(
    artifact: LoadedViscoelasticArtifact,
    *,
    time_grid: TimeGrid,
    segments: Sequence[LoadSegment],
    parameters: Mapping[str, float],
    load_key: str = "ux",
    residual_tolerance: float = 5.0,
    allow_fem_fallback: bool = False,
    resume_checkpoint: Path | None = None,
    save_checkpoint_to: Path | None = None,
) -> dict[str, Any]:
    """Execute a guarded stateful viscoelastic rollout with optional checkpointing."""
    import numpy as np

    mesh = artifact.mesh
    visco = artifact.material
    op = artifact.operator
    frame = artifact.frame
    length = float(frame.extent[0, 0].item())
    height = float(frame.extent[0, 1].item())
    domain = TriangleDomain(
        torch.tensor(mesh.nodes, dtype=torch.float64),
        torch.tensor(mesh.cells, dtype=torch.int64),
    )
    n_cells = int(domain.cells.shape[0])
    load_path_hash = hashlib.sha256(
        canonical_json_bytes([s.document() for s in segments])
    ).hexdigest()
    problem_hash = str(artifact.manifest.get("problem_spec_hash", "0" * 64))

    def _lift(eps: float) -> Callable[[torch.Tensor], AffineDirichletLifting]:
        def fn(q: torch.Tensor) -> AffineDirichletLifting:
            x = q[..., 0]
            y = q[..., 1]
            return AffineDirichletLifting(
                offset=torch.stack((eps * x, torch.zeros_like(y)), dim=-1),
                scale=torch.stack(
                    ((x / length) * (1.0 - x / length) * 4.0, y / height), dim=-1
                ),
            )

        return fn

    centroids = domain.vertices[domain.cells].mean(dim=1)

    if resume_checkpoint is not None:
        bound, u_cur, state_cur = load_trajectory_checkpoint(resume_checkpoint)
        state_cur = dict(state_cur)
    else:
        state_cur = visco.initial_state((n_cells,), dtype=torch.float64)
        u_cur = torch.zeros((domain.vertices.shape[0], 2), dtype=torch.float64)
        bound = BoundTrajectoryState(
            problem_spec_hash=problem_hash,
            mesh_hash=mesh.mesh_id,
            material_schema_hash=visco.state_schema.schema_hash,
            load_path_hash=load_path_hash,
            time_index=0,
            time_value=time_grid.t0,
            step_controller_version=STEP_CONTROLLER_VERSION,
            state_hash=hash_state_tensors(state_cur),
        )

    def _predictor(
        _u_old: torch.Tensor,
        s_old: Mapping[str, torch.Tensor],
        load_next: Mapping[str, float],
        dt_s: float,
    ) -> CandidateStepOutput:
        ux = float(load_next[load_key])
        eps = ux / length
        load_vec = torch.tensor([eps], dtype=torch.float64)
        hist_feat = summarize_viscous_state(s_old, weights=domain.areas)
        local_verts = (
            project_cell_state_to_nodes(domain, s_old["viscous_strain"])
            if op.config.include_local_state
            else None
        )
        with torch.no_grad():
            u_nodes = op.displacement(
                query_xy=domain.vertices.unsqueeze(0),
                frame=frame,
                load_vector=load_vec,
                dt=dt_s,
                history_features=hist_feat,
                lifting=_lift(eps),
                local_state_features=(
                    None if local_verts is None else local_verts.unsqueeze(0)
                ),
            )[0]
        local_cells = (
            None
            if local_verts is None
            else local_verts[domain.cells].mean(dim=1).unsqueeze(0)
        )
        _, grad_c = op.displacement_and_gradient(
            query_xy=centroids.unsqueeze(0),
            frame=frame,
            load_vector=load_vec,
            dt=dt_s,
            history_features=hist_feat,
            lifting=_lift(eps),
            create_graph=False,
            local_state_features=local_cells,
        )
        eye2 = torch.eye(2, dtype=torch.float64).expand(n_cells, 2, 2)
        f2d = eye2 + grad_c[0]
        ps = visco.plane_stress_update(eye2, f2d, s_old, dt_s)
        s_new = {k: v.detach().clone() for k, v in ps.update_3d.state_new.items()}
        min_det = float(ps.update_3d.det_f.min().item())
        min_diss = float(ps.update_3d.dissipation.min().item())
        tot_diss = float(
            torch.sum(ps.update_3d.dissipation * domain.areas).detach().item()
        )
        # Divergence-theorem axial reaction estimate: (1/L) * integral(P_11 dOmega)
        p11 = ps.update_3d.stress[:, 0, 0]
        rxn_est = float(torch.sum(p11 * domain.areas).detach().item()) / length
        return CandidateStepOutput(
            displacement=u_nodes.detach(),
            state_new=s_new,
            min_det_f=min_det,
            min_dissipation=min_diss,
            total_dissipation=tot_diss,
            residual_norm=0.0,
            admissible=min_det > 0.0 and min_diss >= -1e-12,
            reason=f"rxn_est={rxn_est:.6f}",
        )

    stepper = GuardedStatefulStepper(
        problem_spec_hash=problem_hash,
        mesh_hash=mesh.mesh_id,
        material_schema_hash=visco.state_schema.schema_hash,
        load_path_hash=load_path_hash,
        predictor=_predictor,
        fem_fallback=None,
        verification=StatefulVerificationPolicy(allow_fem_fallback=allow_fem_fallback),
        residual_tolerance=residual_tolerance,
        max_substep_halvings=0,
    )

    times = time_grid.fixed_times()
    step_records: list[dict[str, Any]] = []
    for idx in range(bound.time_index + 1, len(times)):
        dt_s = times[idx] - times[idx - 1]
        loads = evaluate_load_path(segments, times[idx], parameters)
        res = stepper.advance(
            bound_state=bound,
            displacement_old=u_cur,
            state_old=state_cur,
            load_next=loads,
            dt=dt_s,
        )
        if (
            not res.accepted
            or res.displacement is None
            or res.state_new is None
            or res.bound_state is None
        ):
            return {
                "status": res.status.value,
                "accepted": False,
                "failed_step_index": idx,
                "reason": res.reason,
                "steps": step_records,
                "claim_level": "NOT_VERIFIED",
            }
        u_cur = res.displacement
        state_cur = dict(res.state_new)
        bound = res.bound_state
        step_records.append(
            {
                "time_index": bound.time_index,
                "time_value": bound.time_value,
                "status": res.status.value,
                "min_det_f": res.min_det_f,
                "total_dissipation": res.total_dissipation,
                "state_hash": bound.state_hash,
                "max_disp_mm": float(np.max(np.abs(u_cur.detach().cpu().numpy()))),
            }
        )

    if save_checkpoint_to is not None:
        save_trajectory_checkpoint(
            save_checkpoint_to,
            bound_state=bound,
            displacement=u_cur,
            state_tensors=state_cur,
        )

    return {
        "status": "COMPLETED",
        "accepted": True,
        "artifact_manifest_sha256": artifact.manifest_sha256,
        "final_bound_state": bound.document(),
        "steps": step_records,
        "claim_level": "NOT_VERIFIED",
    }


__all__ = (
    "VISCOELASTIC_ARTIFACT_SCHEMA",
    "LoadedViscoelasticArtifact",
    "load_viscoelastic_artifact",
    "run_viscoelastic_rollout",
    "save_viscoelastic_artifact",
)
