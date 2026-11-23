"""Parametric CAD compiler adapter, geometry-aware surrogate bundle, serving, and qualification (M4 / F3).

Connects :mod:`piginot.cad.parametric` to FormaMech's P2 geometry-aware operator stack:
* :class:`~piginot.operators.geometry_tokenizer.GeometryTokenizer` &
  :class:`~piginot.operators.geometry_tokenizer.GeometryOODDetector`,
* :class:`~piginot.operators.boundary_energy_operator.BoundaryEnergyOperator`,
* :func:`~piginot.objectives.energy.plane_stress_potential_energy` (exact Deep Ritz
  variational objective with determinant barrier),
* :class:`~piginot.fem.backend.ScikitFemBackend` (converged P2 hyperelastic FEM
  verification and qualification).

Registers the ``cad-parametric-v1`` entry point under ``piginot.compiler.adapters``.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import torch

from piginot import __version__
from piginot.cad.parametric import (
    ParametricCadSpec,
    ParametricInstance,
    ParametricTopologyError,
    TopologyStabilityGuard,
    sample_parametric_cad_doe,
)
from piginot.compiler.adapters import (
    ENERGY_ARCHITECTURE,
    AdapterDescriptor,
    GeometryIdentity,
)
from piginot.contracts import (
    BoundaryConditionSpec,
    CanonicalMesh,
    ExecutionProfile,
    MaterialSpec,
    Quantity,
    Unit,
    canonical_json_bytes,
)
from piginot.contracts.state_machine import (
    ClaimLevel,
    StateMachineSpec,
    TransitionSpec,
    register_state_machine,
)
from piginot.fem.backend import ScikitFemBackend
from piginot.fem.models import FemRequest, FemSolverConfig
from piginot.lifting.base import AffineDirichletLifting
from piginot.objectives.energy import (
    PotentialEnergyConfig,
    TriangleDomain,
    plane_stress_potential_energy,
    stratified_triangle_quadrature,
)
from piginot.operators.adapter_artifact import (
    load_boundary_operator_checkpoint,
    load_geometry_ood_detector,
    save_boundary_operator_checkpoint,
    save_geometry_ood_detector,
)
from piginot.operators.boundary_energy_operator import (
    BoundaryEnergyConfig,
    BoundaryEnergyOperator,
    boundary_energy_module_digest,
)
from piginot.operators.energy_operator import (
    EnergyOperatorConfig,
    constitutive_in_plane_stress,
)
from piginot.operators.geometry_tokenizer import (
    GEOMETRY_FEATURE_NAMES,
    GeometryOODDetector,
    GeometryOODReport,
    GeometryOODStatus,
    GeometryTokenizer,
)

PARAMETRIC_CAD_ADAPTER_ID: Final = "cad-parametric-v1"
PARAMETRIC_BUNDLE_SCHEMA: Final = "formamech.parametric-bundle-v1"
PARAMETRIC_PREDICTION_SCHEMA: Final = "formamech.parametric-prediction-v1"
PARAMETRIC_QUALIFICATION_SCHEMA: Final = "formamech.parametric-qualification-v1"
PredictMode = Literal["fast", "certified", "exact"]


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    """Bind parametric CAD contract and adapter source files."""
    package = Path(__file__).resolve().parents[1]
    files = ("cad/parametric.py", "cad/parametric_adapter.py")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "formamech.parametric-cad-adapter-implementation-v1",
                "files": {
                    name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                    for name in files
                },
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ParametricCadAdapter:
    """Registered compiler adapter descriptor for ``cad-parametric-v1``."""

    adapter_id: str = PARAMETRIC_CAD_ADAPTER_ID

    @property
    def descriptor(self) -> AdapterDescriptor:
        impl_sha = implementation_sha256()
        family_id = hashlib.sha256(b"formamech.parametric-cad-family-v1").hexdigest()
        return AdapterDescriptor(
            adapter_id=self.adapter_id,
            version=__version__,
            implementation_sha256=impl_sha,
            geometries=(GeometryIdentity(family_id, impl_sha),),
            materials=(
                "compressible_neo_hookean",
                "mooney_rivlin",
                "yeoh",
                "gent",
            ),
            liftings=("adf",),
            architectures=(ENERGY_ARCHITECTURE,),
            reference_initializations=("cold",),
        )


PARAMETRIC_CAD: Final = ParametricCadAdapter()


@dataclass(frozen=True, slots=True)
class ParametricPhysicsConfig:
    """Material, tensile load, and reference FEM settings for a parametric family."""

    elastic_modulus_mpa: float = 10.0
    poisson_ratio: float = 0.35
    material_model: str = "compressible_neo_hookean"
    nominal_strain: float = 0.08
    fem_element_order: int = 2
    fem_load_steps: int = 4

    def __post_init__(self) -> None:
        if (
            isinstance(self.elastic_modulus_mpa, bool)
            or not math.isfinite(float(self.elastic_modulus_mpa))
            or float(self.elastic_modulus_mpa) <= 0.0
        ):
            raise ValueError("elastic_modulus_mpa must be finite and positive")
        if (
            isinstance(self.poisson_ratio, bool)
            or not math.isfinite(float(self.poisson_ratio))
            or not (0.0 <= float(self.poisson_ratio) < 0.5)
        ):
            raise ValueError("poisson_ratio must lie in [0, 0.5)")
        if (
            isinstance(self.nominal_strain, bool)
            or not math.isfinite(float(self.nominal_strain))
            or abs(float(self.nominal_strain)) <= 0.0
        ):
            raise ValueError("nominal_strain must be finite and non-zero")
        if self.fem_element_order not in (1, 2):
            raise ValueError("fem_element_order must be 1 or 2")
        if type(self.fem_load_steps) is not int or self.fem_load_steps < 1:
            raise ValueError("fem_load_steps must be an integer >= 1")

    def material_spec(self) -> MaterialSpec:
        return MaterialSpec(
            name=f"parametric-{self.material_model}",
            model=self.material_model,
            properties={
                "elastic_modulus": Quantity(
                    float(self.elastic_modulus_mpa), Unit.MPA
                ).to_canonical(),
                "poisson_ratio": Quantity(
                    float(self.poisson_ratio), Unit.ONE
                ).to_canonical(),
            },
        )

    def document(self) -> dict[str, Any]:
        return {
            "elastic_modulus_mpa": float(self.elastic_modulus_mpa),
            "poisson_ratio": float(self.poisson_ratio),
            "material_model": self.material_model,
            "nominal_strain": float(self.nominal_strain),
            "fem_element_order": self.fem_element_order,
            "fem_load_steps": self.fem_load_steps,
        }

    @classmethod
    def from_document(cls, doc: Mapping[str, Any]) -> ParametricPhysicsConfig:
        return cls(
            elastic_modulus_mpa=float(doc.get("elastic_modulus_mpa", 10.0)),
            poisson_ratio=float(doc.get("poisson_ratio", 0.35)),
            material_model=str(doc.get("material_model", "compressible_neo_hookean")),
            nominal_strain=float(doc.get("nominal_strain", 0.08)),
            fem_element_order=int(doc.get("fem_element_order", 2)),
            fem_load_steps=int(doc.get("fem_load_steps", 4)),
        )


@dataclass(frozen=True, slots=True)
class ParametricTrainingConfig:
    """Hyperparameters for geometry-aware ``BoundaryEnergyOperator`` training."""

    hidden_dim: int = 48
    latent_token_count: int = 8
    parameter_hidden_dim: int = 32
    num_attention_heads: int = 4
    decoder_layers: int = 3
    positional_frequencies: int = 3
    epochs: int = 60
    learning_rate: float = 3.0e-3
    points_per_cell: int = 2
    seed: int = 0
    use_fem_supervision: bool = False
    supervision_weight: float = 10.0
    ood_descriptor_threshold: float = 4.5
    ood_token_threshold: float = 4.5
    ood_regularization: float = 1.0e-2

    def __post_init__(self) -> None:
        for label in (
            "hidden_dim",
            "latent_token_count",
            "parameter_hidden_dim",
            "num_attention_heads",
            "decoder_layers",
            "positional_frequencies",
            "epochs",
            "points_per_cell",
        ):
            val = getattr(self, label)
            if type(val) is not int or val <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        for flabel in (
            "learning_rate",
            "supervision_weight",
            "ood_descriptor_threshold",
            "ood_token_threshold",
            "ood_regularization",
        ):
            fval = float(getattr(self, flabel))
            if not math.isfinite(fval) or fval <= 0.0:
                raise ValueError(f"{flabel} must be finite and positive")


class _ParametricTensileLifting:
    """Exact affine Dirichlet lifting clamping x=xmin and pulling x=xmax."""

    def __init__(
        self,
        *,
        origin_x: float,
        extent_x: float,
        extent_y: float,
        nominal_strain: float,
    ) -> None:
        self.origin_x = float(origin_x)
        self.extent_x = float(extent_x)
        self.extent_y = float(extent_y)
        self.nominal_strain = float(nominal_strain)

    def __call__(self, points: torch.Tensor) -> AffineDirichletLifting:
        xi = (points[..., 0] - self.origin_x) / self.extent_x
        ux_target = self.nominal_strain * self.extent_x
        offset = torch.stack((ux_target * xi, torch.zeros_like(xi)), dim=-1)
        scale_amp = max(abs(ux_target), 1.0e-3)
        scale = scale_amp * torch.stack((4.0 * xi * (1.0 - xi), xi), dim=-1)
        return AffineDirichletLifting(offset=offset, scale=scale)


def _mesh_bbox(mesh: CanonicalMesh) -> tuple[float, float, float, float]:
    xs = [p[0] for p in mesh.nodes]
    ys = [p[1] for p in mesh.nodes]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, xmax - xmin, ymax - ymin


def _lifting_for_mesh(
    mesh: CanonicalMesh, physics: ParametricPhysicsConfig
) -> _ParametricTensileLifting:
    xmin, _, ex, ey = _mesh_bbox(mesh)
    return _ParametricTensileLifting(
        origin_x=xmin,
        extent_x=max(ex, 1.0e-9),
        extent_y=max(ey, 1.0e-9),
        nominal_strain=physics.nominal_strain,
    )


def _normalized_param_tensor(
    spec: ParametricCadSpec,
    params: Mapping[str, float] | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if params is None:
        vals = [0.0 for _ in spec.parameters]
    else:
        vals = [
            2.0 * (float(params[p.name]) - p.lower) / p.span - 1.0
            for p in spec.parameters
        ]
    return torch.tensor([vals], dtype=dtype, device=device)


def _default_execution_profile() -> ExecutionProfile:
    return ExecutionProfile(
        python_version=platform.python_version(),
        library_versions={"piginot": __version__},
        blas_threads=1,
        device="cpu",
        dtype="float64",
        rng_algorithm="philox",
        seed=0,
    )


def build_parametric_fem_request(
    mesh: CanonicalMesh,
    physics: ParametricPhysicsConfig,
    *,
    request_id: str,
    left_tag: str = "left",
    right_tag: str = "right",
) -> FemRequest:
    """Build a canonical S7 ``FemRequest`` for one parametric CAD mesh instance."""
    _, _, extent_x, _ = _mesh_bbox(mesh)
    ux_mm = physics.nominal_strain * extent_x
    bcs = (
        BoundaryConditionSpec(
            name="clamp_left",
            boundary_tag=left_tag,
            kind="dirichlet",
            components=("x", "y"),
            values=(
                Quantity(0.0, Unit.MM).to_canonical(),
                Quantity(0.0, Unit.MM).to_canonical(),
            ),
        ),
        BoundaryConditionSpec(
            name="pull_right",
            boundary_tag=right_tag,
            kind="dirichlet",
            components=("x",),
            values=(Quantity(ux_mm, Unit.MM).to_canonical(),),
        ),
    )
    return FemRequest(
        request_id=request_id,
        mesh=mesh,
        material=physics.material_spec(),
        boundary_conditions=bcs,
        loads=(),
        qois=(),
        config=FemSolverConfig(
            element_order=physics.fem_element_order,
            solver_tolerance=1.0e-10,
            max_newton_iterations=25,
            plane_stress_max_iterations=32,
            load_steps=physics.fem_load_steps,
        ),
        execution_profile=_default_execution_profile(),
    )


@dataclass(frozen=True, slots=True)
class ParametricFieldMetrics:
    """Scalar engineering metrics + nodal fields for one parametric CAD evaluation."""

    converged: bool
    area_mm2: float
    reaction_force_n: float
    peak_von_mises_mpa: float
    max_displacement_mm: float
    min_det_f: float
    nodal_displacement: np.ndarray[Any, Any]
    nodal_von_mises_mpa: np.ndarray[Any, Any]
    result_hash: str | None = None
    reason: str = "ok"

    def scalar_document(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "area_mm2": self.area_mm2,
            "reaction_force_n": self.reaction_force_n,
            "peak_von_mises_mpa": self.peak_von_mises_mpa,
            "max_displacement_mm": self.max_displacement_mm,
            "min_det_f": self.min_det_f,
            "result_hash": self.result_hash,
            "reason": self.reason,
        }


def _cauchy_von_mises_from_p_and_f(
    p_tensor: np.ndarray[Any, Any], f_tensor: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Compute plane-stress Cauchy von Mises stress from 3x3 ``P`` and ``F``."""
    j = np.linalg.det(f_tensor)
    j_safe = np.where(np.abs(j) > 1.0e-12, j, 1.0)
    sigma = (
        np.einsum("...ij,...kj->...ik", p_tensor, f_tensor) / j_safe[..., None, None]
    )
    s11 = sigma[..., 0, 0]
    s22 = sigma[..., 1, 1]
    s12 = 0.5 * (sigma[..., 0, 1] + sigma[..., 1, 0])
    vm_sq = np.maximum(s11 * s11 - s11 * s22 + s22 * s22 + 3.0 * s12 * s12, 0.0)
    return np.asarray(np.sqrt(vm_sq), dtype=np.float64)


def _mesh_area_mm2(mesh: CanonicalMesh) -> float:
    pts = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    v0, v1, v2 = pts[cells[:, 0]], pts[cells[:, 1]], pts[cells[:, 2]]
    areas = 0.5 * np.abs(
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )
    return float(np.sum(areas))


def _project_cell_scalar_to_nodes(
    mesh: CanonicalMesh, cell_values: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    pts = np.asarray(mesh.nodes, dtype=np.float64)
    cells = np.asarray(mesh.cells, dtype=np.int64)
    v0, v1, v2 = pts[cells[:, 0]], pts[cells[:, 1]], pts[cells[:, 2]]
    areas = 0.5 * np.abs(
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )
    num = np.zeros(len(mesh.nodes), dtype=np.float64)
    den = np.zeros(len(mesh.nodes), dtype=np.float64)
    for c_idx in range(cells.shape[0]):
        a = areas[c_idx]
        val = float(cell_values[c_idx])
        for nid in cells[c_idx]:
            num[nid] += a * val
            den[nid] += a
    return np.asarray(num / np.maximum(den, 1.0e-12), dtype=np.float64)


def evaluate_parametric_fem(
    mesh: CanonicalMesh,
    physics: ParametricPhysicsConfig,
    *,
    backend: ScikitFemBackend | None = None,
    request_id: str = "parametric-fem",
) -> ParametricFieldMetrics:
    """Solve one parametric CAD geometry with exact converged ``ScikitFemBackend``."""
    solver = ScikitFemBackend() if backend is None else backend
    req = build_parametric_fem_request(mesh, physics, request_id=request_id)
    attempt = solver.solve_many((req,))[0]
    area = _mesh_area_mm2(mesh)
    n_nodes = len(mesh.nodes)
    if not attempt.success or attempt.result is None:
        return ParametricFieldMetrics(
            converged=False,
            area_mm2=area,
            reaction_force_n=0.0,
            peak_von_mises_mpa=0.0,
            max_displacement_mm=0.0,
            min_det_f=0.0,
            nodal_displacement=np.zeros((n_nodes, 2), dtype=np.float64),
            nodal_von_mises_mpa=np.zeros(n_nodes, dtype=np.float64),
            result_hash=None,
            reason=attempt.message,
        )

    res = attempt.result
    # Vertex nodal displacements are the first len(mesh.nodes) DOFs in MeshTri P1/P2
    disp_all = np.asarray(res.displacement, dtype=np.float64)
    disp_nodes = disp_all[:n_nodes]
    p_q = np.asarray(res.first_piola_stress, dtype=np.float64)
    f_q = np.asarray(res.deformation_gradient, dtype=np.float64)
    vm_q = _cauchy_von_mises_from_p_and_f(p_q, f_q)
    # Quadrature axis is (cells, qpts) -> cell-mean von Mises then area-weighted nodal
    vm_cell = vm_q.mean(axis=1)
    vm_nodes = _project_cell_scalar_to_nodes(mesh, vm_cell)
    det_f_2d = f_q[..., 0, 0] * f_q[..., 1, 1] - f_q[..., 0, 1] * f_q[..., 1, 0]
    min_det = float(np.min(det_f_2d))
    rx = float(res.reactions["pull_right"][0])
    max_u = float(np.max(np.linalg.norm(disp_nodes, axis=1)))
    return ParametricFieldMetrics(
        converged=True,
        area_mm2=area,
        reaction_force_n=rx,
        peak_von_mises_mpa=float(np.max(vm_nodes)),
        max_displacement_mm=max_u,
        min_det_f=min_det,
        nodal_displacement=disp_nodes,
        nodal_von_mises_mpa=vm_nodes,
        result_hash=attempt.result_hash,
        reason="converged",
    )


def _evaluate_operator_on_mesh(
    operator: BoundaryEnergyOperator,
    spec: ParametricCadSpec,
    physics: ParametricPhysicsConfig,
    mesh: CanonicalMesh,
    params: Mapping[str, float] | None,
    *,
    tokenizer: GeometryTokenizer,
) -> tuple[ParametricFieldMetrics, torch.Tensor]:
    """Run ``BoundaryEnergyOperator`` and compute physical stress & reaction metrics."""
    dtype = next(operator.parameters()).dtype
    device = next(operator.parameters()).device
    batch = tokenizer.tokenize_batch([(mesh, spec.role_map)], dtype=dtype)
    encoding = operator.encode_boundary(
        boundary_xy=batch.boundary_xy.to(device=device),
        boundary_normals=batch.boundary_normals.to(device=device),
        boundary_features=batch.boundary_features.to(device=device),
        weights=batch.weights.to(device=device),
        mask=batch.mask.to(device=device),
        frame=batch.frame,
    )
    pooled_tokens = encoding.tokens.mean(dim=1)[0].detach().cpu()

    param_t = _normalized_param_tensor(spec, params, dtype=dtype, device=device)
    lifting = _lifting_for_mesh(mesh, physics)
    nodes_t = torch.tensor(mesh.nodes, dtype=dtype, device=device)
    cells_t = torch.tensor(mesh.cells, dtype=torch.int64, device=device)
    centroids_t = nodes_t[cells_t].mean(dim=1)
    material = physics.material_spec()

    with torch.no_grad():
        disp_nodes = operator.displacement(
            encoding=encoding,
            query_xy=nodes_t.unsqueeze(0),
            parameters=param_t,
            lifting=lifting,
        )[0]
    _, grad_cells = operator.displacement_and_gradient(
        encoding=encoding,
        query_xy=centroids_t.unsqueeze(0),
        parameters=param_t,
        lifting=lifting,
        create_graph=False,
    )
    g_c = grad_cells[0]
    p_2d = constitutive_in_plane_stress(g_c, material)
    eye2 = torch.eye(2, dtype=dtype, device=device)
    f_2d = eye2 + g_c
    det_2d = f_2d[:, 0, 0] * f_2d[:, 1, 1] - f_2d[:, 0, 1] * f_2d[:, 1, 0]
    min_det = float(det_2d.min().item())

    # Exact plane-stress Cauchy stress sigma_2d = (1 / det_3d) * P_2d @ F_2d^T
    # using in-plane closure f33 = det_3d / det_2d
    from piginot.physics import plane_stress_closure

    safe_f2d = torch.where((det_2d > 1.0e-6)[:, None, None], f_2d, eye2.expand_as(f_2d))
    f33 = plane_stress_closure(safe_f2d, material)
    det_3d = torch.clamp(det_2d * f33, min=1.0e-6)
    sigma_2d = torch.einsum("nij,nkj->nik", p_2d, f_2d) / det_3d[:, None, None]
    s11 = sigma_2d[:, 0, 0]
    s22 = sigma_2d[:, 1, 1]
    s12 = 0.5 * (sigma_2d[:, 0, 1] + sigma_2d[:, 1, 0])
    vm_cells = (
        torch.sqrt(
            torch.clamp(s11 * s11 - s11 * s22 + s22 * s22 + 3.0 * s12 * s12, min=0.0)
        )
        .detach()
        .cpu()
        .numpy()
    )
    vm_nodes = _project_cell_scalar_to_nodes(mesh, vm_cells)

    # Zero-solve nodal reaction force on "right" edge: F_x = sum_{cells} A_c * (P_11 * dN_right/dx + P_12 * dN_right/dy)
    pts_np = np.asarray(mesh.nodes, dtype=np.float64)
    cells_np = np.asarray(mesh.cells, dtype=np.int64)
    p_np = p_2d.detach().cpu().numpy()
    right_nodes: set[int] = set()
    for fi in mesh.semantic_tags.get("right", ()):
        a_n, b_n = mesh.boundary_facets[fi]
        right_nodes.add(a_n)
        right_nodes.add(b_n)

    rx_total = 0.0
    for c_idx in range(cells_np.shape[0]):
        n0, n1, n2 = cells_np[c_idx]
        if not (n0 in right_nodes or n1 in right_nodes or n2 in right_nodes):
            continue
        x0, y0 = pts_np[n0]
        x1, y1 = pts_np[n1]
        x2, y2 = pts_np[n2]
        two_a = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if two_a <= 0.0:
            continue
        grads = (
            ((y1 - y2) / two_a, (x2 - x1) / two_a),
            ((y2 - y0) / two_a, (x0 - x2) / two_a),
            ((y0 - y1) / two_a, (x1 - x0) / two_a),
        )
        area_c = 0.5 * two_a
        p11, p12 = float(p_np[c_idx, 0, 0]), float(p_np[c_idx, 0, 1])
        for nid, (gx, gy) in zip((n0, n1, n2), grads, strict=True):
            if nid in right_nodes:
                rx_total += area_c * (p11 * gx + p12 * gy)

    u_np = disp_nodes.detach().cpu().numpy()
    max_u = float(np.max(np.linalg.norm(u_np, axis=1)))
    metrics = ParametricFieldMetrics(
        converged=min_det > 1.0e-6 and math.isfinite(rx_total),
        area_mm2=_mesh_area_mm2(mesh),
        reaction_force_n=float(rx_total),
        peak_von_mises_mpa=float(np.max(vm_nodes)),
        max_displacement_mm=max_u,
        min_det_f=min_det,
        nodal_displacement=u_np,
        nodal_von_mises_mpa=vm_nodes,
        result_hash=None,
        reason="ok" if min_det > 1.0e-6 else "inadmissible det_f <= 1e-6",
    )
    return metrics, pooled_tokens


@dataclass(frozen=True, slots=True)
class ParametricTrainingSummary:
    """Recorded metrics from training a geometry-aware parametric CAD surrogate."""

    epochs: int
    n_train_geometries: int
    initial_loss: float
    final_loss: float
    loss_history: tuple[float, ...]
    min_det_f_seen: float
    training_mode: str
    claim_level: str = "NOT_VERIFIED"

    def document(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "n_train_geometries": self.n_train_geometries,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "loss_history": list(self.loss_history),
            "min_det_f_seen": self.min_det_f_seen,
            "training_mode": self.training_mode,
            "claim_level": self.claim_level,
        }


@dataclass(frozen=True, slots=True)
class ParametricSurrogateBundle:
    """In-memory handle to a parametric CAD surrogate bundle."""

    spec: ParametricCadSpec
    physics: ParametricPhysicsConfig
    operator: BoundaryEnergyOperator
    ood_detector: GeometryOODDetector
    training_summary: ParametricTrainingSummary
    claim_level: str = "NOT_VERIFIED"
    bundle_dir: Path | None = None

    @property
    def bundle_hash(self) -> str:
        doc = {
            "schema": PARAMETRIC_BUNDLE_SCHEMA,
            "spec_hash": self.spec.spec_hash,
            "physics": self.physics.document(),
            "operator_digest": boundary_energy_module_digest(self.operator),
            "training_mode": self.training_summary.training_mode,
        }
        return _sha256_doc(doc)


def train_parametric_cad_surrogate(
    spec: ParametricCadSpec,
    physics: ParametricPhysicsConfig,
    train_instances: Sequence[ParametricInstance],
    *,
    config: ParametricTrainingConfig | None = None,
    backend: ScikitFemBackend | None = None,
) -> ParametricSurrogateBundle:
    """Train ``BoundaryEnergyOperator`` + ``GeometryOODDetector`` across a CAD DOE."""
    if len(train_instances) < 2:
        raise ValueError("training a parametric CAD surrogate requires >= 2 geometries")
    cfg = ParametricTrainingConfig() if config is None else config
    torch.manual_seed(cfg.seed)

    e_cfg = EnergyOperatorConfig(
        parameter_names=tuple(p.name for p in spec.parameters),
        hidden_dim=cfg.hidden_dim,
        latent_token_count=cfg.latent_token_count,
        parameter_hidden_dim=cfg.parameter_hidden_dim,
        num_attention_heads=cfg.num_attention_heads,
        decoder_layers=cfg.decoder_layers,
        positional_frequencies=cfg.positional_frequencies,
        output_init_std=1.0e-3,
    )
    b_cfg = BoundaryEnergyConfig(
        energy=e_cfg,
        boundary_feature_names=GEOMETRY_FEATURE_NAMES,
    )
    operator = BoundaryEnergyOperator(b_cfg).to(dtype=torch.float64)
    tokenizer = GeometryTokenizer(subdivisions_per_facet=1)
    material = physics.material_spec()
    energy_cfg = PotentialEnergyConfig(points_per_cell=cfg.points_per_cell)

    # Pre-build per-instance domains, token batches, liftings, and optional FEM anchors
    prepared: list[
        tuple[
            ParametricInstance,
            TriangleDomain,
            Any,
            torch.Tensor,
            _ParametricTensileLifting,
            torch.Tensor | None,
        ]
    ] = []
    for idx, inst in enumerate(train_instances):
        domain = TriangleDomain(
            vertices=torch.tensor(inst.mesh.nodes, dtype=torch.float64),
            cells=torch.tensor(inst.mesh.cells, dtype=torch.int64),
        )
        batch = tokenizer.tokenize_batch(
            [(inst.mesh, spec.role_map)], dtype=torch.float64
        )
        p_vec = _normalized_param_tensor(
            spec, inst.parameters, dtype=torch.float64, device=torch.device("cpu")
        )
        lifting = _lifting_for_mesh(inst.mesh, physics)
        sup_u: torch.Tensor | None = None
        if cfg.use_fem_supervision:
            fem_res = evaluate_parametric_fem(
                inst.mesh,
                physics,
                backend=backend,
                request_id=f"train-sup-{idx}",
            )
            if not fem_res.converged:
                raise RuntimeError(
                    f"FEM supervision solve failed on train geometry {idx}: {fem_res.reason}"
                )
            sup_u = torch.tensor(fem_res.nodal_displacement, dtype=torch.float64)
        prepared.append((inst, domain, batch, p_vec, lifting, sup_u))

    optimizer = torch.optim.Adam(operator.parameters(), lr=cfg.learning_rate)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.seed)
    loss_history: list[float] = []
    min_det_seen = float("inf")
    n_geom = float(len(prepared))

    for _ in range(cfg.epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        for _, domain, batch, p_vec, lifting, sup_u in prepared:
            q_pts, q_w = stratified_triangle_quadrature(
                domain, cfg.points_per_cell, generator=gen
            )
            encoding = operator.encode_boundary(
                boundary_xy=batch.boundary_xy,
                boundary_normals=batch.boundary_normals,
                boundary_features=batch.boundary_features,
                weights=batch.weights,
                mask=batch.mask,
                frame=batch.frame,
            )
            _, grad_u = operator.displacement_and_gradient(
                encoding=encoding,
                query_xy=q_pts.unsqueeze(0),
                parameters=p_vec,
                lifting=lifting,
                create_graph=True,
            )
            terms = plane_stress_potential_energy(
                grad_u[0],
                q_w,
                material,
                config=energy_cfg,
                reference_area=domain.area,
            )
            loss = terms.total
            if sup_u is not None:
                pred_u = operator.displacement(
                    encoding=encoding,
                    query_xy=domain.vertices.unsqueeze(0),
                    parameters=p_vec,
                    lifting=lifting,
                )[0]
                u_scale = max(float(torch.linalg.vector_norm(sup_u).item()), 1.0e-6)
                sup_loss = (torch.linalg.vector_norm(pred_u - sup_u) / u_scale) ** 2
                loss = loss + cfg.supervision_weight * sup_loss

            (loss / n_geom).backward()  # type: ignore[no-untyped-call]
            epoch_loss_sum += float(loss.detach().item())
            min_det_seen = min(min_det_seen, terms.minimum_determinant)

        torch.nn.utils.clip_grad_norm_(operator.parameters(), max_norm=5.0)
        optimizer.step()
        loss_history.append(epoch_loss_sum / n_geom)

    # Fit GeometryOODDetector strictly on TRAIN descriptors + TRAIN pooled tokens
    train_descriptors = [inst.descriptor for inst in train_instances]
    train_token_means: list[torch.Tensor] = []
    with torch.no_grad():
        for _, _, batch, _, _, _ in prepared:
            enc = operator.encode_boundary(
                boundary_xy=batch.boundary_xy,
                boundary_normals=batch.boundary_normals,
                boundary_features=batch.boundary_features,
                weights=batch.weights,
                mask=batch.mask,
                frame=batch.frame,
            )
            train_token_means.append(enc.tokens.mean(dim=1)[0].detach().cpu())

    detector = GeometryOODDetector.fit(
        train_descriptors,
        descriptor_threshold=cfg.ood_descriptor_threshold,
        regularization=cfg.ood_regularization,
        train_token_means=train_token_means,
        token_threshold=cfg.ood_token_threshold,
    )

    mode_str = (
        "multi-fidelity-label-efficient"
        if cfg.use_fem_supervision
        else "variational-energy-label-free"
    )
    summary = ParametricTrainingSummary(
        epochs=cfg.epochs,
        n_train_geometries=len(train_instances),
        initial_loss=loss_history[0],
        final_loss=loss_history[-1],
        loss_history=tuple(loss_history),
        min_det_f_seen=min_det_seen,
        training_mode=mode_str,
        claim_level="NOT_VERIFIED",
    )
    return ParametricSurrogateBundle(
        spec=spec,
        physics=physics,
        operator=operator,
        ood_detector=detector,
        training_summary=summary,
        claim_level="NOT_VERIFIED",
    )


def save_parametric_bundle(
    bundle: ParametricSurrogateBundle,
    output_dir: Path,
) -> dict[str, Any]:
    """Persist a tamper-checked parametric CAD surrogate bundle."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "parametric-spec.json"
    spec_path.write_bytes(canonical_json_bytes(bundle.spec.document()) + b"\n")

    op_dir = out / "operator"
    op_manifest = save_boundary_operator_checkpoint(op_dir, bundle.operator)

    ood_path = out / "ood-detector.json"
    ood_manifest = save_geometry_ood_detector(ood_path, bundle.ood_detector)

    manifest: dict[str, Any] = {
        "schema": PARAMETRIC_BUNDLE_SCHEMA,
        "adapter_id": PARAMETRIC_CAD_ADAPTER_ID,
        "version": __version__,
        "bundle_hash": bundle.bundle_hash,
        "claim_level": bundle.claim_level,
        "spec_file": "parametric-spec.json",
        "spec_sha256": _sha256_file(spec_path),
        "spec_hash": bundle.spec.spec_hash,
        "physics": bundle.physics.document(),
        "operator_dir": "operator",
        "operator_base_digest": op_manifest["base_digest"],
        "operator_weights_sha256": op_manifest["weights_sha256"],
        "ood_detector_file": "ood-detector.json",
        "ood_detector_sha256": ood_manifest["detector_sha256"],
        "training_summary": bundle.training_summary.document(),
    }
    (out / "bundle.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def load_parametric_bundle(bundle_dir: Path) -> ParametricSurrogateBundle:
    """Verify SHA-256 integrity and reload a saved parametric CAD bundle."""
    root = Path(bundle_dir)
    manifest_path = root / "bundle.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"parametric bundle.json not found in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PARAMETRIC_BUNDLE_SCHEMA:
        raise ValueError(
            f"expected schema {PARAMETRIC_BUNDLE_SCHEMA!r}, got {manifest.get('schema')!r}"
        )

    spec_path = root / str(manifest["spec_file"])
    if _sha256_file(spec_path) != manifest["spec_sha256"]:
        raise ValueError("parametric-spec.json SHA-256 mismatch")
    spec = ParametricCadSpec.from_document(
        json.loads(spec_path.read_text(encoding="utf-8"))
    )
    if spec.spec_hash != manifest.get("spec_hash"):
        raise ValueError("parametric-spec.json spec_hash mismatch")
    physics = ParametricPhysicsConfig.from_document(manifest["physics"])
    op_dir = root / str(manifest["operator_dir"])
    if _sha256_file(op_dir / "weights.safetensors") != manifest.get(
        "operator_weights_sha256"
    ):
        raise ValueError("parametric bundle operator_weights_sha256 mismatch")
    operator = load_boundary_operator_checkpoint(op_dir)
    if boundary_energy_module_digest(operator) != manifest.get("operator_base_digest"):
        raise ValueError("parametric bundle operator_base_digest mismatch")
    ood_path = root / str(manifest["ood_detector_file"])
    ood_doc = json.loads(ood_path.read_text(encoding="utf-8"))
    if not isinstance(ood_doc, dict) or ood_doc.get("detector_sha256") != manifest.get(
        "ood_detector_sha256"
    ):
        raise ValueError("parametric bundle ood_detector_sha256 mismatch")
    detector = load_geometry_ood_detector(ood_path)

    ts_doc = manifest["training_summary"]
    try:
        claim_level = ClaimLevel(str(manifest.get("claim_level", "NOT_VERIFIED"))).value
        summary_claim = ClaimLevel(str(ts_doc.get("claim_level", "NOT_VERIFIED"))).value
    except ValueError as exc:
        raise ValueError(
            f"refused unsupported claim_level {manifest.get('claim_level')!r}"
        ) from exc
    summary = ParametricTrainingSummary(
        epochs=int(ts_doc["epochs"]),
        n_train_geometries=int(ts_doc["n_train_geometries"]),
        initial_loss=float(ts_doc["initial_loss"]),
        final_loss=float(ts_doc["final_loss"]),
        loss_history=tuple(float(v) for v in ts_doc["loss_history"]),
        min_det_f_seen=float(ts_doc["min_det_f_seen"]),
        training_mode=str(ts_doc["training_mode"]),
        claim_level=summary_claim,
    )
    loaded = ParametricSurrogateBundle(
        spec=spec,
        physics=physics,
        operator=operator,
        ood_detector=detector,
        training_summary=summary,
        claim_level=claim_level,
        bundle_dir=root,
    )
    if loaded.bundle_hash != manifest.get("bundle_hash"):
        raise ValueError("parametric bundle_hash mismatch")
    return loaded


def is_parametric_bundle_dir(path: Path) -> bool:
    """Return True if ``path/bundle.json`` carries ``PARAMETRIC_BUNDLE_SCHEMA``."""
    manifest_path = Path(path) / "bundle.json"
    if not manifest_path.is_file():
        return False
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(doc, dict) and doc.get("schema") == PARAMETRIC_BUNDLE_SCHEMA


class ParametricPredictionStatus(StrEnum):
    """Terminal outcomes of geometry-aware parametric CAD prediction."""

    SERVED_FAST = "SERVED_FAST"
    SERVED_CERTIFIED = "SERVED_CERTIFIED"
    SERVED_EXACT = "SERVED_EXACT"
    REFUSED_TOPOLOGY = "REFUSED_TOPOLOGY"
    REFUSED_OOD = "REFUSED_OOD"
    REFUSED_FEM_FAILED = "REFUSED_FEM_FAILED"


class ParametricPredictPhase(StrEnum):
    """Explicit stages of the parametric CAD prediction and OOD-gating pipeline."""

    RECEIVED = "RECEIVED"
    TOPOLOGY_VERIFIED = "TOPOLOGY_VERIFIED"
    DESCRIPTOR_OOD_PASSED = "DESCRIPTOR_OOD_PASSED"
    OPERATOR_EVALUATED = "OPERATOR_EVALUATED"
    SERVED_FAST = "SERVED_FAST"
    SERVED_CERTIFIED = "SERVED_CERTIFIED"
    SERVED_EXACT = "SERVED_EXACT"
    REFUSED_TOPOLOGY = "REFUSED_TOPOLOGY"
    REFUSED_OOD = "REFUSED_OOD"
    REFUSED_FEM_FAILED = "REFUSED_FEM_FAILED"


PARAMETRIC_PREDICT_MACHINE: Final[StateMachineSpec[ParametricPredictPhase]] = (
    register_state_machine(
        StateMachineSpec(
            machine_id="formamech.cad.parametric_predict",
            description=(
                "Parametric CAD serving pipeline: topology stability guard -> "
                "descriptor OOD gate -> boundary energy operator -> token OOD gate -> "
                "fast response or P2 FEM verification."
            ),
            states=tuple(ParametricPredictPhase),
            initial_states=(ParametricPredictPhase.RECEIVED,),
            terminal_states=(
                ParametricPredictPhase.SERVED_FAST,
                ParametricPredictPhase.SERVED_CERTIFIED,
                ParametricPredictPhase.SERVED_EXACT,
                ParametricPredictPhase.REFUSED_TOPOLOGY,
                ParametricPredictPhase.REFUSED_OOD,
                ParametricPredictPhase.REFUSED_FEM_FAILED,
            ),
            transitions=(
                TransitionSpec(
                    ParametricPredictPhase.RECEIVED,
                    ParametricPredictPhase.REFUSED_TOPOLOGY,
                    "topology_guard_failed",
                    "Parameter box, derived ligament bound, or topological genus check failed.",
                ),
                TransitionSpec(
                    ParametricPredictPhase.RECEIVED,
                    ParametricPredictPhase.TOPOLOGY_VERIFIED,
                    "verify_topology",
                    "TopologyStabilityGuard verified mesh genus and boundary groups.",
                ),
                TransitionSpec(
                    ParametricPredictPhase.TOPOLOGY_VERIFIED,
                    ParametricPredictPhase.REFUSED_OOD,
                    "refuse_descriptor_ood",
                    "GeometryOODDetector rejected descriptor Mahalanobis distance or hole count.",
                ),
                TransitionSpec(
                    ParametricPredictPhase.TOPOLOGY_VERIFIED,
                    ParametricPredictPhase.DESCRIPTOR_OOD_PASSED,
                    "pass_descriptor_ood",
                    "Boundary geometry descriptor lies within calibrated training envelope.",
                ),
                TransitionSpec(
                    ParametricPredictPhase.DESCRIPTOR_OOD_PASSED,
                    ParametricPredictPhase.OPERATOR_EVALUATED,
                    "evaluate_operator",
                    "BoundaryEnergyOperator evaluated displacement, stress, and latent tokens.",
                ),
                TransitionSpec(
                    ParametricPredictPhase.OPERATOR_EVALUATED,
                    ParametricPredictPhase.REFUSED_OOD,
                    "refuse_token_ood",
                    "Latent geometry token distance exceeded calibrated OOD threshold.",
                ),
                TransitionSpec(
                    ParametricPredictPhase.OPERATOR_EVALUATED,
                    ParametricPredictPhase.SERVED_FAST,
                    "serve_fast",
                    "Mode is fast and both descriptor and token OOD gates passed.",
                ),
                TransitionSpec(
                    ParametricPredictPhase.OPERATOR_EVALUATED,
                    ParametricPredictPhase.SERVED_CERTIFIED,
                    "serve_certified",
                    "Mode is certified and reference P2 FEM converged.",
                ),
                TransitionSpec(
                    ParametricPredictPhase.OPERATOR_EVALUATED,
                    ParametricPredictPhase.SERVED_EXACT,
                    "serve_exact",
                    "Mode is exact and reference P2 FEM converged.",
                ),
                TransitionSpec(
                    ParametricPredictPhase.OPERATOR_EVALUATED,
                    ParametricPredictPhase.REFUSED_FEM_FAILED,
                    "fem_solve_failed",
                    "Reference P2 nonlinear FEM solve failed to converge.",
                ),
            ),
        )
    )
)


@dataclass(frozen=True, slots=True)
class ParametricPredictionReport:
    """Auditable prediction & geometry-OOD trust report for one CAD variant."""

    status: (
        ParametricPredictionStatus
        | Literal[
            "SERVED_FAST",
            "SERVED_CERTIFIED",
            "SERVED_EXACT",
            "REFUSED_TOPOLOGY",
            "REFUSED_OOD",
            "REFUSED_FEM_FAILED",
        ]
    )
    mode: PredictMode
    bundle_hash: str
    parameters: dict[str, float] | None
    mesh_id: str | None
    ood_report: GeometryOODReport | None
    surrogate_metrics: ParametricFieldMetrics | None
    fem_metrics: ParametricFieldMetrics | None
    displacement_rel_l2_error: float | None
    reaction_rel_error: float | None
    peak_von_mises_rel_error: float | None
    claim_level: str
    reason: str

    def __post_init__(self) -> None:
        st = ParametricPredictionStatus(str(self.status))
        object.__setattr__(self, "status", st)
        claim = ClaimLevel(str(self.claim_level)).value
        object.__setattr__(self, "claim_level", claim)
        if self.mode not in ("fast", "certified", "exact"):
            raise ValueError(f"unsupported mode {self.mode!r}")
        if st is ParametricPredictionStatus.REFUSED_TOPOLOGY:
            if (
                self.ood_report is not None
                or self.surrogate_metrics is not None
                or self.fem_metrics is not None
            ):
                raise ValueError(
                    "REFUSED_TOPOLOGY report must not carry ood_report, surrogate_metrics, or fem_metrics"
                )
        elif st is ParametricPredictionStatus.REFUSED_OOD:
            if self.ood_report is None or self.fem_metrics is not None:
                raise ValueError(
                    "REFUSED_OOD report requires ood_report and forbids fem_metrics"
                )
        elif st is ParametricPredictionStatus.SERVED_FAST:
            if (
                self.mode != "fast"
                or self.surrogate_metrics is None
                or self.fem_metrics is not None
            ):
                raise ValueError(
                    "SERVED_FAST requires mode='fast', surrogate_metrics, and fem_metrics=None"
                )
        elif st in (
            ParametricPredictionStatus.SERVED_CERTIFIED,
            ParametricPredictionStatus.SERVED_EXACT,
        ):
            expected_mode = (
                "certified"
                if st is ParametricPredictionStatus.SERVED_CERTIFIED
                else "exact"
            )
            if (
                self.mode != expected_mode
                or self.surrogate_metrics is None
                or self.fem_metrics is None
                or not self.fem_metrics.converged
                or self.displacement_rel_l2_error is None
            ):
                raise ValueError(
                    f"{st.value} requires mode={expected_mode!r}, surrogate_metrics, "
                    "converged fem_metrics, and relative error metrics"
                )
        elif st is ParametricPredictionStatus.REFUSED_FEM_FAILED:
            if self.fem_metrics is None or self.fem_metrics.converged:
                raise ValueError(
                    "REFUSED_FEM_FAILED requires non-converged fem_metrics"
                )

    @property
    def accepted(self) -> bool:
        return self.status in (
            ParametricPredictionStatus.SERVED_FAST,
            ParametricPredictionStatus.SERVED_CERTIFIED,
            ParametricPredictionStatus.SERVED_EXACT,
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema": PARAMETRIC_PREDICTION_SCHEMA,
            "status": str(self.status),
            "mode": self.mode,
            "bundle_hash": self.bundle_hash,
            "parameters": None
            if self.parameters is None
            else dict(sorted(self.parameters.items())),
            "mesh_id": self.mesh_id,
            "ood_report": None
            if self.ood_report is None
            else self.ood_report.document(),
            "surrogate_metrics": None
            if self.surrogate_metrics is None
            else self.surrogate_metrics.scalar_document(),
            "fem_metrics": None
            if self.fem_metrics is None
            else self.fem_metrics.scalar_document(),
            "displacement_rel_l2_error": self.displacement_rel_l2_error,
            "reaction_rel_error": self.reaction_rel_error,
            "peak_von_mises_rel_error": self.peak_von_mises_rel_error,
            "claim_level": self.claim_level,
            "reason": self.reason,
        }

    @property
    def report_hash(self) -> str:
        return _sha256_doc(self.document())


def predict_parametric_cad(
    bundle: ParametricSurrogateBundle,
    *,
    parameters: Mapping[str, float] | None = None,
    mesh: CanonicalMesh | None = None,
    mode: PredictMode = "fast",
    enforce_parameter_box: bool = True,
    backend: ScikitFemBackend | None = None,
) -> ParametricPredictionReport:
    """Run topology guard -> GeometryOODDetector -> operator -> optional P2 FEM check."""
    if parameters is None and mesh is None:
        raise ValueError("either parameters or mesh must be provided")
    if mode not in ("fast", "certified", "exact"):
        raise ValueError(f"unsupported mode {mode!r}")

    guard = TopologyStabilityGuard(bundle.spec)
    tokenizer = GeometryTokenizer(subdivisions_per_facet=1)
    clean_params: dict[str, float] | None = None

    if mesh is None:
        assert parameters is not None
        clean_params = {k: float(v) for k, v in parameters.items()}
        try:
            inst = guard.generate(
                clean_params,
                enforce_box=enforce_parameter_box,
                enforce_bounds=True,
            )
            target_mesh = inst.mesh
        except ParametricTopologyError as exc:
            return ParametricPredictionReport(
                status="REFUSED_TOPOLOGY",
                mode=mode,
                bundle_hash=bundle.bundle_hash,
                parameters=clean_params,
                mesh_id=None,
                ood_report=None,
                surrogate_metrics=None,
                fem_metrics=None,
                displacement_rel_l2_error=None,
                reaction_rel_error=None,
                peak_von_mises_rel_error=None,
                claim_level=bundle.claim_level,
                reason=str(exc),
            )
    else:
        target_mesh = mesh
        if parameters is not None:
            clean_params = {k: float(v) for k, v in parameters.items()}

    # Evaluate OOD on descriptor first (handles unsupported topology / genus change cleanly)
    pre_ood = bundle.ood_detector.score(
        target_mesh, bundle.spec.role_map, tokenizer=tokenizer
    )
    if pre_ood.status is not GeometryOODStatus.IN_DISTRIBUTION:
        return ParametricPredictionReport(
            status="REFUSED_OOD",
            mode=mode,
            bundle_hash=bundle.bundle_hash,
            parameters=clean_params,
            mesh_id=target_mesh.mesh_id,
            ood_report=pre_ood,
            surrogate_metrics=None,
            fem_metrics=None,
            displacement_rel_l2_error=None,
            reaction_rel_error=None,
            peak_von_mises_rel_error=None,
            claim_level=bundle.claim_level,
            reason=f"GeometryOODDetector refused ({pre_ood.status.value}): {pre_ood.reason}",
        )

    surr_metrics, pooled_tokens = _evaluate_operator_on_mesh(
        bundle.operator,
        bundle.spec,
        bundle.physics,
        target_mesh,
        clean_params,
        tokenizer=tokenizer,
    )
    full_ood = bundle.ood_detector.score(
        target_mesh,
        bundle.spec.role_map,
        pooled_tokens=pooled_tokens,
        tokenizer=tokenizer,
    )
    if full_ood.status is not GeometryOODStatus.IN_DISTRIBUTION:
        return ParametricPredictionReport(
            status="REFUSED_OOD",
            mode=mode,
            bundle_hash=bundle.bundle_hash,
            parameters=clean_params,
            mesh_id=target_mesh.mesh_id,
            ood_report=full_ood,
            surrogate_metrics=surr_metrics,
            fem_metrics=None,
            displacement_rel_l2_error=None,
            reaction_rel_error=None,
            peak_von_mises_rel_error=None,
            claim_level=bundle.claim_level,
            reason=f"GeometryOODDetector refused ({full_ood.status.value}): {full_ood.reason}",
        )

    if mode == "fast":
        return ParametricPredictionReport(
            status="SERVED_FAST",
            mode=mode,
            bundle_hash=bundle.bundle_hash,
            parameters=clean_params,
            mesh_id=target_mesh.mesh_id,
            ood_report=full_ood,
            surrogate_metrics=surr_metrics,
            fem_metrics=None,
            displacement_rel_l2_error=None,
            reaction_rel_error=None,
            peak_von_mises_rel_error=None,
            claim_level=bundle.claim_level,
            reason="in-distribution parametric CAD prediction served",
        )

    fem_metrics = evaluate_parametric_fem(
        target_mesh,
        bundle.physics,
        backend=backend,
        request_id=f"predict-{target_mesh.mesh_id[:12]}",
    )
    if not fem_metrics.converged:
        return ParametricPredictionReport(
            status="REFUSED_FEM_FAILED",
            mode=mode,
            bundle_hash=bundle.bundle_hash,
            parameters=clean_params,
            mesh_id=target_mesh.mesh_id,
            ood_report=full_ood,
            surrogate_metrics=surr_metrics,
            fem_metrics=fem_metrics,
            displacement_rel_l2_error=None,
            reaction_rel_error=None,
            peak_von_mises_rel_error=None,
            claim_level=bundle.claim_level,
            reason=f"reference P2 FEM failed: {fem_metrics.reason}",
        )

    u_ref_norm = max(float(np.linalg.norm(fem_metrics.nodal_displacement)), 1.0e-12)
    u_err = (
        float(
            np.linalg.norm(
                surr_metrics.nodal_displacement - fem_metrics.nodal_displacement
            )
        )
        / u_ref_norm
    )
    rx_err = abs(surr_metrics.reaction_force_n - fem_metrics.reaction_force_n) / max(
        abs(fem_metrics.reaction_force_n), 1.0e-12
    )
    vm_err = abs(
        surr_metrics.peak_von_mises_mpa - fem_metrics.peak_von_mises_mpa
    ) / max(abs(fem_metrics.peak_von_mises_mpa), 1.0e-12)

    return ParametricPredictionReport(
        status="SERVED_EXACT" if mode == "exact" else "SERVED_CERTIFIED",
        mode=mode,
        bundle_hash=bundle.bundle_hash,
        parameters=clean_params,
        mesh_id=target_mesh.mesh_id,
        ood_report=full_ood,
        surrogate_metrics=surr_metrics,
        fem_metrics=fem_metrics,
        displacement_rel_l2_error=u_err,
        reaction_rel_error=rx_err,
        peak_von_mises_rel_error=vm_err,
        claim_level=bundle.claim_level,
        reason="parametric CAD prediction verified against converged P2 FEM",
    )


def _build_genus_changed_probe_mesh(spec: ParametricCadSpec) -> CanonicalMesh:
    """Build a solid (0-hole) planar rectangle matching nominal outer dimensions."""
    from piginot.geometry import CellType, MeshPayload, canonicalize_mesh

    env = spec.full_environment(spec.nominal_parameters())
    lx = float(env.get("plate_length", 40.0))
    ly = float(env.get("plate_width", 20.0))
    nx, ny = 4, 2
    nodes = [(lx * i / nx, ly * j / ny) for j in range(ny + 1) for i in range(nx + 1)]

    def _nid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    cells: list[tuple[int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            n00, n10 = _nid(i, j), _nid(i + 1, j)
            n01, n11 = _nid(i, j + 1), _nid(i + 1, j + 1)
            cells.append((n00, n10, n11))
            cells.append((n00, n11, n01))

    facets: list[tuple[int, int]] = []
    tags: dict[str, list[int]] = {
        "left": [],
        "right": [],
        "bottom": [],
        "top": [],
    }
    for i in range(nx):
        tags["bottom"].append(len(facets))
        facets.append((_nid(i, 0), _nid(i + 1, 0)))
    for j in range(ny):
        tags["right"].append(len(facets))
        facets.append((_nid(nx, j), _nid(nx, j + 1)))
    for i in range(nx, 0, -1):
        tags["top"].append(len(facets))
        facets.append((_nid(i, ny), _nid(i - 1, ny)))
    for j in range(ny, 0, -1):
        tags["left"].append(len(facets))
        facets.append((_nid(0, j), _nid(0, j - 1)))

    return canonicalize_mesh(
        MeshPayload(
            nodes=tuple(nodes),
            cells=tuple(cells),
            cell_type=CellType.TRIANGLE,
            boundary_facets=tuple(facets),
            boundary_tags={k: tuple(v) for k, v in tags.items()},
            region_ids=tuple(1 for _ in cells),
            unit=Unit.MM,
            source_provenance={"probe": "genus-0-solid-plate"},
        )
    )


@dataclass(frozen=True, slots=True)
class ParametricQualificationReport:
    """Held-out unseen CAD qualification manifest (caps at ``HOLDOUT_VERIFIED``)."""

    bundle_hash: str
    passed: bool
    claim_level: str
    holdout_count: int
    in_envelope_ood_pass_rate: float
    ood_probe_refusal_rate: float
    mean_displacement_rel_l2_error: float
    max_displacement_rel_l2_error: float
    mean_reaction_rel_error: float
    max_reaction_rel_error: float
    mean_peak_von_mises_rel_error: float
    min_det_f_observed: float
    holdout_cases: tuple[dict[str, Any], ...]
    ood_probes: tuple[dict[str, Any], ...]
    reason: str

    def document(self) -> dict[str, Any]:
        return {
            "schema": PARAMETRIC_QUALIFICATION_SCHEMA,
            "bundle_hash": self.bundle_hash,
            "passed": self.passed,
            "claim_level": self.claim_level,
            "holdout_count": self.holdout_count,
            "in_envelope_ood_pass_rate": self.in_envelope_ood_pass_rate,
            "ood_probe_refusal_rate": self.ood_probe_refusal_rate,
            "mean_displacement_rel_l2_error": self.mean_displacement_rel_l2_error,
            "max_displacement_rel_l2_error": self.max_displacement_rel_l2_error,
            "mean_reaction_rel_error": self.mean_reaction_rel_error,
            "max_reaction_rel_error": self.max_reaction_rel_error,
            "mean_peak_von_mises_rel_error": self.mean_peak_von_mises_rel_error,
            "min_det_f_observed": self.min_det_f_observed,
            "holdout_cases": list(self.holdout_cases),
            "ood_probes": list(self.ood_probes),
            "reason": self.reason,
        }

    @property
    def manifest_hash(self) -> str:
        return _sha256_doc(self.document())


def qualify_parametric_bundle(
    bundle: ParametricSurrogateBundle,
    *,
    holdout_count: int = 10,
    seed_offset: int = 101,
    max_mean_displacement_rel_l2: float = 0.25,
    max_mean_reaction_rel_error: float = 0.25,
    backend: ScikitFemBackend | None = None,
    output_path: Path | None = None,
) -> ParametricQualificationReport:
    """Qualify a parametric bundle across unseen in-envelope CAD variants & OOD probes."""
    if type(holdout_count) is not int or holdout_count < 2:
        raise ValueError("holdout_count must be an integer >= 2")

    doe = sample_parametric_cad_doe(
        bundle.spec,
        count=holdout_count,
        seed_offset=seed_offset,
        include_nominal=False,
    )
    solver = ScikitFemBackend() if backend is None else backend

    holdout_records: list[dict[str, Any]] = []
    u_errs: list[float] = []
    rx_errs: list[float] = []
    vm_errs: list[float] = []
    min_det = float("inf")
    in_env_accepted = 0

    for inst in doe.instances:
        pred = predict_parametric_cad(
            bundle,
            parameters=inst.parameters,
            mesh=inst.mesh,
            mode="certified",
            backend=solver,
        )
        if pred.accepted and pred.surrogate_metrics is not None:
            in_env_accepted += 1
            assert pred.displacement_rel_l2_error is not None
            assert pred.reaction_rel_error is not None
            assert pred.peak_von_mises_rel_error is not None
            u_errs.append(pred.displacement_rel_l2_error)
            rx_errs.append(pred.reaction_rel_error)
            vm_errs.append(pred.peak_von_mises_rel_error)
            min_det = min(min_det, pred.surrogate_metrics.min_det_f)
        holdout_records.append(
            {
                "parameters": inst.parameters,
                "mesh_id": inst.mesh.mesh_id,
                "status": pred.status,
                "ood_distance": None
                if pred.ood_report is None
                else pred.ood_report.descriptor_distance,
                "displacement_rel_l2_error": pred.displacement_rel_l2_error,
                "reaction_rel_error": pred.reaction_rel_error,
                "peak_von_mises_rel_error": pred.peak_von_mises_rel_error,
            }
        )

    # OOD & genus-change refusal probes: must achieve 100% refusal
    probe_records: list[dict[str, Any]] = []
    probes_refused = 0

    # Probe 1: out-of-box parameter violation
    extreme_params = bundle.spec.nominal_parameters()
    p0 = bundle.spec.parameters[0]
    extreme_params[p0.name] = p0.upper + 0.5 * p0.span
    p_box = predict_parametric_cad(bundle, parameters=extreme_params, mode="fast")
    if not p_box.accepted:
        probes_refused += 1
    probe_records.append(
        {
            "probe": "out_of_parameter_envelope",
            "status": p_box.status,
            "refused": not p_box.accepted,
            "reason": p_box.reason,
        }
    )

    # Probe 2: genus-changed (0-hole) solid plate mesh passed directly to OOD detector
    genus_mesh = _build_genus_changed_probe_mesh(bundle.spec)
    p_genus = predict_parametric_cad(bundle, mesh=genus_mesh, mode="fast")
    if not p_genus.accepted:
        probes_refused += 1
    probe_records.append(
        {
            "probe": "genus_changed_0_holes",
            "status": p_genus.status,
            "refused": not p_genus.accepted,
            "reason": p_genus.reason,
        }
    )

    in_rate = in_env_accepted / float(holdout_count)
    ood_refusal_rate = probes_refused / float(len(probe_records))
    mean_u = float(np.mean(u_errs)) if u_errs else float("inf")
    max_u = float(np.max(u_errs)) if u_errs else float("inf")
    mean_rx = float(np.mean(rx_errs)) if rx_errs else float("inf")
    max_rx = float(np.max(rx_errs)) if rx_errs else float("inf")
    mean_vm = float(np.mean(vm_errs)) if vm_errs else float("inf")
    if not u_errs:
        min_det = 0.0

    failures: list[str] = []
    if in_rate < 1.0:
        failures.append(f"in-envelope OOD pass rate {in_rate:.2%} < 100%")
    if ood_refusal_rate < 1.0:
        failures.append(f"OOD refusal rate {ood_refusal_rate:.2%} < 100%")
    if mean_u > max_mean_displacement_rel_l2:
        failures.append(
            f"mean displacement rel L2 {mean_u:.4f} > {max_mean_displacement_rel_l2:.4f}"
        )
    if mean_rx > max_mean_reaction_rel_error:
        failures.append(
            f"mean reaction rel err {mean_rx:.4f} > {max_mean_reaction_rel_error:.4f}"
        )
    if min_det <= 0.0:
        failures.append(f"inadmissible min_det_f={min_det:.4g}")

    passed = len(failures) == 0
    claim_level = "HOLDOUT_VERIFIED" if passed else "NOT_VERIFIED"
    report = ParametricQualificationReport(
        bundle_hash=bundle.bundle_hash,
        passed=passed,
        claim_level=claim_level,
        holdout_count=holdout_count,
        in_envelope_ood_pass_rate=in_rate,
        ood_probe_refusal_rate=ood_refusal_rate,
        mean_displacement_rel_l2_error=mean_u,
        max_displacement_rel_l2_error=max_u,
        mean_reaction_rel_error=mean_rx,
        max_reaction_rel_error=max_rx,
        mean_peak_von_mises_rel_error=mean_vm,
        min_det_f_observed=min_det,
        holdout_cases=tuple(holdout_records),
        ood_probes=tuple(probe_records),
        reason="all held-out unseen CAD and OOD gates passed"
        if passed
        else "; ".join(failures),
    )
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(canonical_json_bytes(report.document()) + b"\n")
    return report


__all__ = (
    "PARAMETRIC_BUNDLE_SCHEMA",
    "PARAMETRIC_CAD",
    "PARAMETRIC_CAD_ADAPTER_ID",
    "PARAMETRIC_PREDICTION_SCHEMA",
    "PARAMETRIC_PREDICT_MACHINE",
    "PARAMETRIC_QUALIFICATION_SCHEMA",
    "ParametricCadAdapter",
    "ParametricFieldMetrics",
    "ParametricPhysicsConfig",
    "ParametricPredictPhase",
    "ParametricPredictionReport",
    "ParametricPredictionStatus",
    "ParametricQualificationReport",
    "ParametricSurrogateBundle",
    "ParametricTrainingConfig",
    "ParametricTrainingSummary",
    "build_parametric_fem_request",
    "evaluate_parametric_fem",
    "implementation_sha256",
    "is_parametric_bundle_dir",
    "load_parametric_bundle",
    "predict_parametric_cad",
    "qualify_parametric_bundle",
    "save_parametric_bundle",
    "train_parametric_cad_surrogate",
)
