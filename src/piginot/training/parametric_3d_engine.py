"""Heterogeneous multi-mesh 3D parametric operator training, serving & qualification (V3-2).

Trains one reusable 16-channel GALE :class:`Surface3DEnergyOperator` across a
Halton QMC DOE of distinct 3D CAD solids, fits a ``TRAIN``-only
:class:`GeometryOODDetector3D`, persists tamper-checked
``formamech.parametric-3d-bundle-v1`` artifacts, serves unseen 3D CAD variants
in ``fast`` / ``certified`` / warm-started ``exact`` ``TET10`` FEM modes, and
qualifies up to ``HOLDOUT_VERIFIED`` with mandatory 100% OOD/topology refusal.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from piginot.cad.parametric_3d import (
    ParametricCadSpec3D,
    ParametricInstance3D,
    ParametricTopology3DError,
    TopologyStabilityGuard3D,
    sample_parametric_cad_3d_doe,
)
from piginot.contracts import MaterialSpec, Quantity, Unit, canonical_json_bytes
from piginot.fem.backend_3d import (
    DirichletBC3D,
    FemRequest3D,
    FemResult3D,
    TorchFemBackend3D,
    TractionBC3D,
    _cauchy_von_mises_3d,
    _project_cell_scalar_to_nodes_3d,
    assemble_surface_tractions_3d,
)
from piginot.fem.linear_solvers import LinearSolverConfig
from piginot.lifting.adf_3d import (
    GeneralFaceADFLifting3D,
    build_adf_lifting_from_mesh_faces,
)
from piginot.operators.geometry_3d import (
    SURFACE_3D_GALE_FEATURE_DIM,
    GeometryOODDetector3D,
    GeometryOODReport3D,
    build_gale_surface_3d_tensors,
    pad_and_stack_surface_3d_batch,
)
from piginot.operators.surface_operator_3d import (
    Surface3DConfig,
    Surface3DEncoding,
    Surface3DEnergyOperator,
)
from piginot.physics import energy_density, first_piola, reference_modulus
from piginot.training.compute import (
    ComputeRuntimeConfig,
    ComputeRuntimeSession,
    autocast_context,
    validate_loss_finite,
)
from piginot.training.energy_engine_3d import (
    _volumetric_quadrature_3d,
    lifted_displacement_and_gradient,
)

FloatArray = NDArray[np.float64]
PARAMETRIC_3D_BUNDLE_SCHEMA: Final = "formamech.parametric-3d-bundle-v1"
PARAMETRIC_3D_PREDICTION_SCHEMA: Final = "formamech.parametric-3d-prediction-v1"
PARAMETRIC_3D_QUALIFICATION_SCHEMA: Final = "formamech.parametric-3d-qualification-v1"
PredictMode3D = Literal["fast", "certified", "exact"]


def _sha256_doc(doc: object) -> str:
    return hashlib.sha256(canonical_json_bytes(doc)).hexdigest()


@dataclass(frozen=True, slots=True)
class Parametric3DPhysicsConfig:
    """Declarative material & traction specification for a 3D parametric CAD family."""

    youngs_modulus_mpa: float = 12.0
    poisson_ratio: float = 0.35
    traction_mpa: tuple[float, float, float] = (0.40, 0.0, 0.0)
    load_steps: int = 2

    def material_spec(self) -> MaterialSpec:
        return MaterialSpec(
            name="parametric-3d-nh",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(
                    self.youngs_modulus_mpa, Unit.MPA
                ).to_canonical(),
                "poisson_ratio": Quantity(self.poisson_ratio, Unit.ONE).to_canonical(),
            },
        )

    def document(self) -> dict[str, Any]:
        return {
            "youngs_modulus_mpa": float(self.youngs_modulus_mpa),
            "poisson_ratio": float(self.poisson_ratio),
            "traction_mpa": [float(v) for v in self.traction_mpa],
            "load_steps": int(self.load_steps),
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> Parametric3DPhysicsConfig:
        return cls(
            youngs_modulus_mpa=float(doc.get("youngs_modulus_mpa", 12.0)),
            poisson_ratio=float(doc.get("poisson_ratio", 0.35)),
            traction_mpa=tuple(
                float(v) for v in doc.get("traction_mpa", (0.4, 0.0, 0.0))
            ),  # type: ignore[arg-type]
            load_steps=int(doc.get("load_steps", 2)),
        )


@dataclass(frozen=True, slots=True)
class Parametric3DTrainingConfig:
    """Hyperparameters for multi-mesh heterogeneous 3D operator training."""

    epochs: int = 40
    learning_rate: float = 3.0e-3
    barrier_weight: float = 400.0
    barrier_threshold: float = 0.2
    supervision_weight: float = 30.0
    quadrature_subsample: int | None = 64
    use_fem_supervision: bool = False
    seed: int = 0
    compute_config: ComputeRuntimeConfig | None = None

    def __post_init__(self) -> None:
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True, slots=True)
class Parametric3DBundle:
    """In-memory handle for a geometry-generalizing 3D parametric surrogate bundle."""

    spec: ParametricCadSpec3D
    physics: Parametric3DPhysicsConfig
    operator: Surface3DEnergyOperator
    ood_detector: GeometryOODDetector3D
    training_mode: str
    loss_history: tuple[float, ...]
    train_count: int
    claim_level: str = "NOT_VERIFIED"

    @property
    def bundle_hash(self) -> str:
        return _sha256_doc(
            {
                "schema": PARAMETRIC_3D_BUNDLE_SCHEMA,
                "spec_hash": self.spec.spec_hash,
                "physics": self.physics.document(),
                "operator_config": asdict(self.operator.config),
                "ood_detector_hash": self.ood_detector.detector_hash,
                "training_mode": self.training_mode,
            }
        )


def _norm_param_vector(
    spec: ParametricCadSpec3D, params: Mapping[str, float]
) -> list[float]:
    out: list[float] = []
    for p in spec.parameters:
        span = max(p.upper - p.lower, 1.0e-12)
        out.append((float(params[p.name]) - p.lower) / span)
    return out


def _build_fem_request(
    inst: ParametricInstance3D,
    physics: Parametric3DPhysicsConfig,
    *,
    linear_solver: LinearSolverConfig | None = None,
) -> FemRequest3D:
    return FemRequest3D(
        request_id=inst.instance_id,
        mesh=inst.imported.mesh,
        material=physics.material_spec(),
        dirichlet_bcs=(
            DirichletBC3D(
                "clamped", inst.clamped_face_tags, ("x", "y", "z"), (0.0, 0.0, 0.0)
            ),
        ),
        traction_bcs=(
            TractionBC3D("loaded", inst.loaded_face_tags, physics.traction_mpa),
        ),
        load_steps=physics.load_steps,
        linear_solver=linear_solver,
    )


def train_parametric_3d_operator(
    spec: ParametricCadSpec3D,
    physics: Parametric3DPhysicsConfig,
    train_instances: Sequence[ParametricInstance3D],
    *,
    operator_config: Surface3DConfig | None = None,
    training_config: Parametric3DTrainingConfig | None = None,
    dtype: torch.dtype = torch.float64,
    backend: TorchFemBackend3D | None = None,
) -> Parametric3DBundle:
    """Train one GALE ``Surface3DEnergyOperator`` across heterogeneous 3D CAD meshes."""
    if len(train_instances) < 2:
        raise ValueError(
            "train_parametric_3d_operator requires >= 2 TRAIN 3D CAD variants"
        )
    t_cfg = Parametric3DTrainingConfig() if training_config is None else training_config
    c_cfg = (
        ComputeRuntimeConfig() if t_cfg.compute_config is None else t_cfg.compute_config
    )
    session = ComputeRuntimeSession(c_cfg)
    device = session.device
    session.activate()
    torch.manual_seed(t_cfg.seed)

    mat = physics.material_spec()
    p_dim = len(spec.parameters)
    op_cfg = (
        Surface3DConfig(
            parameter_dim=p_dim,
            surface_feature_dim=SURFACE_3D_GALE_FEATURE_DIM,
            hidden_dim=48,
            slice_count=8,
            num_blocks=2,
            num_heads=4,
        )
        if operator_config is None
        else operator_config
    )
    if op_cfg.surface_feature_dim != SURFACE_3D_GALE_FEATURE_DIM:
        session.abort()
        raise ValueError(
            f"parametric 3D operator requires surface_feature_dim={SURFACE_3D_GALE_FEATURE_DIM}"
        )
    if op_cfg.parameter_dim != p_dim:
        session.abort()
        raise ValueError(
            f"operator_config.parameter_dim={op_cfg.parameter_dim} != {p_dim}"
        )

    operator = Surface3DEnergyOperator(op_cfg).to(dtype=dtype, device=device)

    # Precompute surface GALE tensors, 3D ADF liftings, quadrature & optional FEM anchors
    surf_items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    liftings: list[GeneralFaceADFLifting3D] = []
    quad_data: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    nodes_list: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    fem_targets: list[torch.Tensor | None] = []
    solver = TorchFemBackend3D() if backend is None else backend

    for inst in train_instances:
        mesh = inst.imported.mesh
        surf_items.append(
            build_gale_surface_3d_tensors(
                mesh,
                clamped_face_tags=inst.clamped_face_tags,
                loaded_face_tags=inst.loaded_face_tags,
                dtype=dtype,
                device=device,
            )
        )
        lift = build_adf_lifting_from_mesh_faces(
            mesh, inst.clamped_face_tags, displacement_scale_mm=1.0
        )
        liftings.append(lift)
        q_pts, q_wts, vol = _volumetric_quadrature_3d(mesh, dtype=dtype, device=device)
        quad_data.append((q_pts, q_wts, vol))
        n_t = torch.tensor(mesh.nodes, dtype=dtype, device=device).unsqueeze(0)
        n_sc, n_off = lift.scale_and_offset(n_t)
        f_ext = torch.tensor(
            assemble_surface_tractions_3d(
                mesh,
                (TractionBC3D("loaded", inst.loaded_face_tags, physics.traction_mpa),),
            ),
            dtype=dtype,
            device=device,
        ).unsqueeze(0)
        nodes_list.append((n_t, n_sc, n_off, f_ext))

        if t_cfg.use_fem_supervision:
            fem_res = solver.solve(_build_fem_request(inst, physics))
            if not fem_res.converged:
                session.abort()
                raise RuntimeError(
                    f"3D FEM anchor failed on {inst.instance_id}: {fem_res.reason}"
                )
            fem_targets.append(
                torch.tensor(
                    fem_res.nodal_displacement, dtype=dtype, device=device
                ).unsqueeze(0)
            )
        else:
            fem_targets.append(None)

    s_cent, s_norm, s_area, s_feat, s_mask = pad_and_stack_surface_3d_batch(surf_items)
    p_batch = torch.tensor(
        [_norm_param_vector(spec, inst.parameters) for inst in train_instances],
        dtype=dtype,
        device=device,
    )  # (B, P)
    e_ref = reference_modulus(mat)
    eye3 = torch.eye(3, dtype=dtype, device=device)
    optimizer = torch.optim.Adam(operator.parameters(), lr=t_cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(t_cfg.epochs, 1), eta_min=t_cfg.learning_rate * 0.05
    )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(t_cfg.seed)
    n_b = len(train_instances)
    loss_history: list[float] = []

    for _ in range(t_cfg.epochs):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(c_cfg):
            enc_all = operator.encode_surface(
                s_cent, s_norm, s_area, s_feat, surface_mask=s_mask
            )
        total_loss = torch.zeros((), dtype=dtype, device=device)
        for i in range(n_b):
            enc_i = Surface3DEncoding(
                slice_tokens=enc_all.slice_tokens[i : i + 1],
                origin=enc_all.origin[i : i + 1],
                extent=enc_all.extent[i : i + 1],
            )
            p_i = p_batch[i : i + 1]
            q_pts_all, q_wts_all, vol_i = quad_data[i]
            n_q = int(q_pts_all.shape[0])
            if (
                t_cfg.quadrature_subsample is not None
                and t_cfg.quadrature_subsample < n_q
            ):
                idx = torch.randperm(n_q, generator=gen)[
                    : t_cfg.quadrature_subsample
                ].to(device=device)
                q_pts = q_pts_all[idx].unsqueeze(0)
                q_wts = q_wts_all[idx] * (
                    float(n_q) / float(t_cfg.quadrature_subsample)
                )
            else:
                q_pts = q_pts_all.unsqueeze(0)
                q_wts = q_wts_all

            with autocast_context(c_cfg):
                _, grad_u = lifted_displacement_and_gradient(
                    operator,
                    encoding=enc_i,
                    query_xyz=q_pts,
                    parameters=p_i,
                    lifting=liftings[i],  # type: ignore[arg-type]
                    create_graph=True,
                )
            f_def = eye3 + grad_u.to(dtype=dtype)
            det_f = torch.linalg.det(f_def)
            adm = det_f > 1.0e-5
            safe_f = torch.where(adm[..., None, None], f_def, eye3.expand_as(f_def))
            w_d = torch.where(adm, energy_density(safe_f, mat), torch.zeros_like(det_f))
            strain_e = torch.sum(w_d * q_wts.unsqueeze(0))

            n_t, n_sc, n_off, f_ext = nodes_list[i]
            with autocast_context(c_cfg):
                u_nodes = operator.displacement(
                    encoding=enc_i,
                    query_xyz=n_t,
                    parameters=p_i,
                    lifting_offset=n_off,
                    lifting_scale=n_sc,
                )
            u_eval = u_nodes.to(dtype=dtype)
            ext_w = torch.sum(f_ext * u_eval)
            e_scale = max(e_ref * vol_i, 1.0e-12)
            pi_i = (strain_e - ext_w) / e_scale
            bar_i = torch.mean(torch.relu(t_cfg.barrier_threshold - det_f).square())
            inst_loss = pi_i + t_cfg.barrier_weight * bar_i

            u_ref = fem_targets[i]
            if u_ref is not None:
                ref_n = torch.linalg.vector_norm(u_ref).clamp_min(1.0e-6)
                rel = torch.linalg.vector_norm(u_eval - u_ref) / ref_n
                inst_loss = inst_loss + t_cfg.supervision_weight * (
                    rel + 0.5 * rel.square()
                )
            total_loss = total_loss + inst_loss / float(n_b)

        loss_val = validate_loss_finite(float(total_loss.detach().item()))
        total_loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(operator.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(loss_val)

    session.complete()
    operator = operator.to(device="cpu")

    # Fit TRAIN-only GeometryOODDetector3D using geometric descriptors + pooled slice latents
    with torch.no_grad():
        latents: list[list[float]] = []
        for inst in train_instances:
            sc, sn, sa, sf = build_gale_surface_3d_tensors(
                inst.imported.mesh,
                clamped_face_tags=inst.clamped_face_tags,
                loaded_face_tags=inst.loaded_face_tags,
                dtype=dtype,
            )
            enc = operator.encode_surface(sc, sn, sa, sf)
            pooled = enc.slice_tokens.mean(dim=1)[0, :8].tolist()
            latents.append([float(v) for v in pooled])

    ood_detector = GeometryOODDetector3D.fit(
        [inst.descriptor for inst in train_instances],
        latent_vectors=latents,
    )
    training_mode = (
        "multi-fidelity-label-efficient"
        if t_cfg.use_fem_supervision
        else "variational-energy-label-free"
    )
    return Parametric3DBundle(
        spec=spec,
        physics=physics,
        operator=operator,
        ood_detector=ood_detector,
        training_mode=training_mode,
        loss_history=tuple(loss_history),
        train_count=len(train_instances),
        claim_level="NOT_VERIFIED",
    )


def save_parametric_3d_bundle(
    bundle: Parametric3DBundle, output_dir: Path | str
) -> dict[str, Any]:
    """Persist a tamper-checked ``formamech.parametric-3d-bundle-v1`` directory."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = {
        k: v.detach().contiguous().cpu()
        for k, v in bundle.operator.state_dict().items()
    }
    dtype_str = str(next(iter(state.values())).dtype).removeprefix("torch.")
    w_bytes = save_safetensors(state, metadata={"schema": PARAMETRIC_3D_BUNDLE_SCHEMA})
    (out / "weights.safetensors").write_bytes(w_bytes)
    w_sha = hashlib.sha256(w_bytes).hexdigest()

    manifest = {
        "schema": PARAMETRIC_3D_BUNDLE_SCHEMA,
        "bundle_hash": bundle.bundle_hash,
        "spec": bundle.spec.document(),
        "physics": bundle.physics.document(),
        "operator_config": asdict(bundle.operator.config),
        "ood_detector": bundle.ood_detector.document(),
        "training_mode": bundle.training_mode,
        "loss_history": list(bundle.loss_history),
        "train_count": bundle.train_count,
        "claim_level": bundle.claim_level,
        "dtype": dtype_str,
        "weights_sha256": w_sha,
    }
    manifest["manifest_sha256"] = _sha256_doc(manifest)
    (out / "bundle.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_parametric_3d_bundle(bundle_dir: Path | str) -> Parametric3DBundle:
    """Verify SHA-256 digests and load a ``formamech.parametric-3d-bundle-v1`` bundle."""
    root = Path(bundle_dir).resolve()
    raw = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if raw.get("schema") != PARAMETRIC_3D_BUNDLE_SCHEMA:
        raise ValueError(
            f"unexpected parametric 3D bundle schema {raw.get('schema')!r}"
        )
    recorded = raw.get("manifest_sha256")
    without_m = {k: v for k, v in raw.items() if k != "manifest_sha256"}
    if _sha256_doc(without_m) != recorded:
        raise ValueError("parametric 3D bundle manifest SHA-256 mismatch")
    w_bytes = (root / "weights.safetensors").read_bytes()
    if hashlib.sha256(w_bytes).hexdigest() != raw["weights_sha256"]:
        raise ValueError("parametric 3D bundle weights SHA-256 mismatch")

    spec = ParametricCadSpec3D.from_document(raw["spec"])
    physics = Parametric3DPhysicsConfig.from_document(raw["physics"])
    op_cfg = Surface3DConfig(**dict(raw["operator_config"]))
    dtype = torch.float32 if raw.get("dtype") == "float32" else torch.float64
    op = Surface3DEnergyOperator(op_cfg).to(dtype=dtype)
    tensors = load_safetensors(w_bytes)
    op.load_state_dict({k: v.to(dtype=dtype) for k, v in tensors.items()}, strict=True)
    ood = GeometryOODDetector3D.from_document(raw["ood_detector"])
    return Parametric3DBundle(
        spec=spec,
        physics=physics,
        operator=op,
        ood_detector=ood,
        training_mode=str(raw["training_mode"]),
        loss_history=tuple(float(v) for v in raw.get("loss_history", ())),
        train_count=int(raw["train_count"]),
        claim_level=str(raw.get("claim_level", "NOT_VERIFIED")),
    )


@dataclass(frozen=True, slots=True)
class Parametric3DPredictionReport:
    """Result of predicting (or refusing) an unseen 3D CAD parameter variant."""

    parameters: Mapping[str, float]
    mode: PredictMode3D
    accepted: bool
    ood_report: GeometryOODReport3D | None
    volume_mm3: float | None
    max_displacement_mm: float | None
    peak_von_mises_mpa: float | None
    compliance_n_mm: float | None
    min_det_f: float | None
    displacement_rel_l2_error: float | None
    fem_result: FemResult3D | None
    nodal_displacement: FloatArray | None
    nodal_von_mises_mpa: FloatArray | None
    wall_time_s: float
    claim_level: str
    reason: str

    @property
    def report_hash(self) -> str:
        return _sha256_doc(self.document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": PARAMETRIC_3D_PREDICTION_SCHEMA,
            "parameters": {k: float(v) for k, v in sorted(self.parameters.items())},
            "mode": self.mode,
            "accepted": self.accepted,
            "ood_report": None
            if self.ood_report is None
            else self.ood_report.document(),
            "volume_mm3": self.volume_mm3,
            "max_displacement_mm": self.max_displacement_mm,
            "peak_von_mises_mpa": self.peak_von_mises_mpa,
            "compliance_n_mm": self.compliance_n_mm,
            "min_det_f": self.min_det_f,
            "displacement_rel_l2_error": self.displacement_rel_l2_error,
            "wall_time_s": float(self.wall_time_s),
            "claim_level": self.claim_level,
            "reason": self.reason,
        }


def predict_parametric_3d(
    bundle: Parametric3DBundle,
    parameters: Mapping[str, float],
    *,
    mode: PredictMode3D = "fast",
    backend: TorchFemBackend3D | None = None,
    linear_solver: LinearSolverConfig | None = None,
) -> Parametric3DPredictionReport:
    """Evaluate (or fail-closed refuse) an unseen 3D CAD variant."""
    t0 = time.perf_counter()
    guard = TopologyStabilityGuard3D(bundle.spec)
    try:
        inst = guard.build_and_verify(parameters, instance_id="query-3d")
    except ParametricTopology3DError as exc:
        return Parametric3DPredictionReport(
            parameters=dict(parameters),
            mode=mode,
            accepted=False,
            ood_report=None,
            volume_mm3=None,
            max_displacement_mm=None,
            peak_von_mises_mpa=None,
            compliance_n_mm=None,
            min_det_f=None,
            displacement_rel_l2_error=None,
            fem_result=None,
            nodal_displacement=None,
            nodal_von_mises_mpa=None,
            wall_time_s=time.perf_counter() - t0,
            claim_level=bundle.claim_level,
            reason=f"topology refused: {exc}",
        )

    mesh = inst.imported.mesh
    dtype = next(bundle.operator.parameters()).dtype
    sc, sn, sa, sf = build_gale_surface_3d_tensors(
        mesh,
        clamped_face_tags=inst.clamped_face_tags,
        loaded_face_tags=inst.loaded_face_tags,
        dtype=dtype,
    )
    with torch.no_grad():
        enc = bundle.operator.encode_surface(sc, sn, sa, sf)
        pooled = [float(v) for v in enc.slice_tokens.mean(dim=1)[0, :8].tolist()]

    ood_rep = bundle.ood_detector.score(inst.descriptor, latent_vector=pooled)
    if not ood_rep.accepted:
        return Parametric3DPredictionReport(
            parameters=dict(inst.parameters),
            mode=mode,
            accepted=False,
            ood_report=ood_rep,
            volume_mm3=float(inst.imported.volume_mm3),
            max_displacement_mm=None,
            peak_von_mises_mpa=None,
            compliance_n_mm=None,
            min_det_f=None,
            displacement_rel_l2_error=None,
            fem_result=None,
            nodal_displacement=None,
            nodal_von_mises_mpa=None,
            wall_time_s=time.perf_counter() - t0,
            claim_level=bundle.claim_level,
            reason=f"OOD refused: {ood_rep.reason}",
        )

    lift = build_adf_lifting_from_mesh_faces(
        mesh, inst.clamped_face_tags, displacement_scale_mm=1.0
    )
    p_t = torch.tensor([_norm_param_vector(bundle.spec, inst.parameters)], dtype=dtype)
    nodes_t = torch.tensor(mesh.nodes, dtype=dtype).unsqueeze(0)
    n_sc, n_off = lift.scale_and_offset(nodes_t)
    with torch.no_grad():
        u_pred_t = bundle.operator.displacement(
            encoding=enc,
            query_xyz=nodes_t,
            parameters=p_t,
            lifting_offset=n_off,
            lifting_scale=n_sc,
        )
    u_pred = u_pred_t[0].to(torch.float64).numpy()

    # Cell centroid gradient for det(F) & Cauchy von Mises stress
    nodes_np = np.asarray(mesh.nodes, dtype=np.float64)
    cells_np = np.asarray(mesh.cells, dtype=np.int64)
    cell_centers = np.mean(nodes_np[cells_np[:, :4]], axis=1)
    p0, p1, p2, p3 = (
        nodes_np[cells_np[:, 0]],
        nodes_np[cells_np[:, 1]],
        nodes_np[cells_np[:, 2]],
        nodes_np[cells_np[:, 3]],
    )
    cell_vols = np.abs(np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0)
    c_t = torch.tensor(cell_centers, dtype=dtype).unsqueeze(0)
    _, grad_c = lifted_displacement_and_gradient(
        bundle.operator,
        encoding=enc,
        query_xyz=c_t,
        parameters=p_t,
        lifting=lift,  # type: ignore[arg-type]
        create_graph=False,
    )
    f_c = (
        grad_c[0].detach().to(torch.float64).numpy()
        + np.eye(3, dtype=np.float64)[None, :, :]
    )
    min_det = float(np.min(np.linalg.det(f_c)))
    if min_det <= 1.0e-6 or not np.all(np.isfinite(f_c)):
        return Parametric3DPredictionReport(
            parameters=dict(inst.parameters),
            mode=mode,
            accepted=False,
            ood_report=ood_rep,
            volume_mm3=float(inst.imported.volume_mm3),
            max_displacement_mm=float(np.max(np.linalg.norm(u_pred, axis=1))),
            peak_von_mises_mpa=None,
            compliance_n_mm=None,
            min_det_f=min_det,
            displacement_rel_l2_error=None,
            fem_result=None,
            nodal_displacement=u_pred,
            nodal_von_mises_mpa=None,
            wall_time_s=time.perf_counter() - t0,
            claim_level=bundle.claim_level,
            reason=f"inadmissible predicted 3D deformation (min det F = {min_det:.4g})",
        )

    with torch.no_grad():
        p_c = first_piola(
            torch.from_numpy(np.ascontiguousarray(f_c)), bundle.physics.material_spec()
        ).numpy()
    vm_cell = _cauchy_von_mises_3d(p_c[:, None, :, :], f_c[:, None, :, :])[:, 0]
    vm_nodes = _project_cell_scalar_to_nodes_3d(mesh, vm_cell, cell_vols)
    f_ext = assemble_surface_tractions_3d(
        mesh,
        (TractionBC3D("loaded", inst.loaded_face_tags, bundle.physics.traction_mpa),),
    )
    comp = float(np.sum(f_ext * u_pred))

    fem_res: FemResult3D | None = None
    disp_err: float | None = None
    final_u = u_pred
    final_vm = vm_nodes
    final_min_det = min_det
    final_comp = comp

    if mode in ("certified", "exact"):
        solver = TorchFemBackend3D() if backend is None else backend
        req = _build_fem_request(inst, bundle.physics, linear_solver=linear_solver)
        fem_res = solver.solve(
            req, initial_displacement=u_pred if mode == "exact" else None
        )
        if fem_res.converged:
            ref_n = max(float(np.linalg.norm(fem_res.nodal_displacement)), 1.0e-12)
            disp_err = float(
                np.linalg.norm(u_pred - fem_res.nodal_displacement) / ref_n
            )
            if mode == "exact":
                final_u = fem_res.nodal_displacement
                final_vm = fem_res.nodal_von_mises_mpa
                final_min_det = fem_res.min_det_f
                final_comp = fem_res.external_work_n_mm

    return Parametric3DPredictionReport(
        parameters=dict(inst.parameters),
        mode=mode,
        accepted=True,
        ood_report=ood_rep,
        volume_mm3=float(inst.imported.volume_mm3),
        max_displacement_mm=float(np.max(np.linalg.norm(final_u, axis=1))),
        peak_von_mises_mpa=float(np.max(final_vm)),
        compliance_n_mm=final_comp,
        min_det_f=final_min_det,
        displacement_rel_l2_error=disp_err,
        fem_result=fem_res,
        nodal_displacement=final_u,
        nodal_von_mises_mpa=final_vm,
        wall_time_s=time.perf_counter() - t0,
        claim_level=bundle.claim_level,
        reason="accepted",
    )


@dataclass(frozen=True, slots=True)
class Parametric3DQualificationReport:
    """Held-out unseen 3D CAD qualification & adversarial OOD refusal report."""

    bundle_hash: str
    passed: bool
    granted_claim_level: str
    holdout_count: int
    ood_probe_count: int
    ood_refusal_rate: float
    mean_displacement_rel_l2: float
    max_displacement_rel_l2: float
    reason: str

    @property
    def manifest_hash(self) -> str:
        return _sha256_doc(self.document())

    def document(self) -> dict[str, Any]:
        return {
            "schema": PARAMETRIC_3D_QUALIFICATION_SCHEMA,
            "bundle_hash": self.bundle_hash,
            "passed": self.passed,
            "granted_claim_level": self.granted_claim_level,
            "holdout_count": self.holdout_count,
            "ood_probe_count": self.ood_probe_count,
            "ood_refusal_rate": float(self.ood_refusal_rate),
            "mean_displacement_rel_l2": float(self.mean_displacement_rel_l2),
            "max_displacement_rel_l2": float(self.max_displacement_rel_l2),
            "reason": self.reason,
        }


def qualify_parametric_3d_bundle(
    bundle: Parametric3DBundle,
    *,
    holdout_count: int = 3,
    seed_offset: int = 101,
    max_mean_displacement_rel_l2: float = 0.25,
    ood_probes: Sequence[Mapping[str, float]] | None = None,
    backend: TorchFemBackend3D | None = None,
) -> Parametric3DQualificationReport:
    """Qualify a 3D parametric bundle on unseen in-envelope solids + OOD refusal probes."""
    doe = sample_parametric_cad_3d_doe(
        bundle.spec, count=holdout_count, seed_offset=seed_offset
    )
    errs: list[float] = []
    for inst in doe.instances:
        rep = predict_parametric_3d(
            bundle, inst.parameters, mode="certified", backend=backend
        )
        if not rep.accepted or rep.displacement_rel_l2_error is None:
            return Parametric3DQualificationReport(
                bundle_hash=bundle.bundle_hash,
                passed=False,
                granted_claim_level="NOT_VERIFIED",
                holdout_count=holdout_count,
                ood_probe_count=0,
                ood_refusal_rate=0.0,
                mean_displacement_rel_l2=float("inf"),
                max_displacement_rel_l2=float("inf"),
                reason=f"in-envelope holdout {inst.instance_id} failed: {rep.reason}",
            )
        errs.append(rep.displacement_rel_l2_error)

    # Default adversarial OOD probes outside parameter box
    if ood_probes is None:
        nom = bundle.spec.nominal_parameters()
        p0 = bundle.spec.parameters[0]
        probes: list[dict[str, float]] = [
            {**nom, p0.name: p0.upper + 0.5 * (p0.upper - p0.lower)},
            {**nom, p0.name: max(0.1, p0.lower - 0.5 * (p0.upper - p0.lower))},
        ]
    else:
        probes = [dict(p) for p in ood_probes]

    refused = 0
    for probe in probes:
        rep_p = predict_parametric_3d(bundle, probe, mode="fast", backend=backend)
        if not rep_p.accepted:
            refused += 1
    refusal_rate = float(refused) / float(max(len(probes), 1))
    mean_err = float(np.mean(errs))
    max_err = float(np.max(errs))
    passed = (mean_err <= max_mean_displacement_rel_l2) and (
        refusal_rate >= 1.0 - 1e-12
    )
    claim = "HOLDOUT_VERIFIED" if passed else "NOT_VERIFIED"
    reason = (
        "passed"
        if passed
        else f"mean_disp_err={mean_err:.4f} (limit {max_mean_displacement_rel_l2:.4f}), "
        f"ood_refusal={refusal_rate:.2f}"
    )
    return Parametric3DQualificationReport(
        bundle_hash=bundle.bundle_hash,
        passed=passed,
        granted_claim_level=claim,
        holdout_count=len(errs),
        ood_probe_count=len(probes),
        ood_refusal_rate=refusal_rate,
        mean_displacement_rel_l2=mean_err,
        max_displacement_rel_l2=max_err,
        reason=reason,
    )


__all__ = (
    "PARAMETRIC_3D_BUNDLE_SCHEMA",
    "PARAMETRIC_3D_PREDICTION_SCHEMA",
    "PARAMETRIC_3D_QUALIFICATION_SCHEMA",
    "Parametric3DBundle",
    "Parametric3DPhysicsConfig",
    "Parametric3DPredictionReport",
    "Parametric3DQualificationReport",
    "Parametric3DTrainingConfig",
    "load_parametric_3d_bundle",
    "predict_parametric_3d",
    "qualify_parametric_3d_bundle",
    "save_parametric_3d_bundle",
    "train_parametric_3d_operator",
)
