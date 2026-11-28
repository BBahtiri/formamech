"""Vision-based elastomer tactile sensor FEM & certified per-taxel surrogate (V6-R2).

Bridges 3D finite-strain elastomer contact FEM (:mod:`piginot.fem.diff_contact_3d`)
with real-time per-taxel 3D normal/shear force reconstruction and AS6983
conservative slip/overload bounds (:mod:`piginot.bounds.conformal_3d`).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from piginot.bounds.conformal_3d import (
    Conformal3DAssessment,
    Conformal3DCalibrationRecord,
    Conformal3DCalibrator,
    Conformal3DDevSample,
)
from piginot.contracts import MaterialSpec, Quantity, Unit, canonical_json_bytes
from piginot.fem.backend_3d import DirichletBC3D
from piginot.fem.diff_contact_3d import (
    RigidSdfIndenter3D,
    SdfContact3DResult,
    SdfPrimitiveKind,
    solve_3d_sdf_contact,
)
from piginot.geometry.mesh3d import CanonicalMesh3D, make_brick_tet10_mesh

FloatArray = NDArray[np.float64]
TACTILE_SPEC_SCHEMA: Final = "formamech.tactile-sensor-spec-v1"
TACTILE_PRED_SCHEMA: Final = "formamech.tactile-prediction-v1"


def _sha256_doc(document: object) -> str:
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


@dataclass(frozen=True, slots=True)
class TactileSensorSpec:
    """Physical geometry & elastomer material properties of a GelSight-class pad."""

    name: str = "gelsight-mini-pad"
    width_x_mm: float = 18.0
    length_y_mm: float = 24.0
    thickness_z_mm: float = 4.0
    youngs_modulus_mpa: float = 0.25
    poisson_ratio: float = 0.48
    friction_mu: float = 0.65
    taxel_grid_xy: tuple[int, int] = (8, 10)
    mesh_div_xyz: tuple[int, int, int] = (4, 5, 2)

    def __post_init__(self) -> None:
        if (
            self.width_x_mm <= 0.0
            or self.length_y_mm <= 0.0
            or self.thickness_z_mm <= 0.0
        ):
            raise ValueError("pad dimensions must be positive")
        if self.youngs_modulus_mpa <= 0.0 or not (0.0 <= self.poisson_ratio < 0.5):
            raise ValueError("invalid elastomer elastic constants")
        if self.taxel_grid_xy[0] < 2 or self.taxel_grid_xy[1] < 2:
            raise ValueError("taxel_grid_xy must be at least (2, 2)")

    def material_spec(self) -> MaterialSpec:
        return MaterialSpec(
            name=f"{self.name}-elastomer",
            model="compressible_neo_hookean",
            properties={
                "elastic_modulus": Quantity(
                    self.youngs_modulus_mpa, Unit.MPA
                ).to_canonical(),
                "poisson_ratio": Quantity(self.poisson_ratio, Unit.ONE).to_canonical(),
            },
        )

    def build_tet10_mesh(self) -> CanonicalMesh3D:
        """Build quadratic ``TET10`` elastomer gel pad (incompressibility-locking safe)."""
        nx, ny, nz = self.mesh_div_xyz
        return make_brick_tet10_mesh(
            self.width_x_mm,
            self.length_y_mm,
            self.thickness_z_mm,
            nx=nx,
            ny=ny,
            nz=nz,
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema": TACTILE_SPEC_SCHEMA,
            "name": self.name,
            "width_x_mm": float(self.width_x_mm),
            "length_y_mm": float(self.length_y_mm),
            "thickness_z_mm": float(self.thickness_z_mm),
            "youngs_modulus_mpa": float(self.youngs_modulus_mpa),
            "poisson_ratio": float(self.poisson_ratio),
            "friction_mu": float(self.friction_mu),
            "taxel_grid_xy": list(self.taxel_grid_xy),
            "mesh_div_xyz": list(self.mesh_div_xyz),
        }


@dataclass(frozen=True, slots=True)
class TactileProbeSample:
    """One verified 3D FEM elastomer indentation & shear sample."""

    case_id: str
    probe_x_mm: float
    probe_y_mm: float
    depth_mm: float
    shear_dx_mm: float
    shear_dy_mm: float
    indenter_radius_mm: float
    indenter_kind: SdfPrimitiveKind
    taxel_forces_n: FloatArray  # (Nx, Ny, 3) per-taxel (fx, fy, fz)
    surface_marker_disp_mm: FloatArray  # (Nx, Ny, 3) marker displacement field
    wrench_6d: tuple[float, float, float, float, float, float]
    max_slip_ratio: float
    peak_von_mises_mpa: float
    fem_wall_time_s: float


def project_contact_to_taxel_grid(
    spec: TactileSensorSpec,
    mesh: CanonicalMesh3D,
    res: SdfContact3DResult,
) -> tuple[FloatArray, FloatArray, float]:
    """Bin 3D FEM top-surface nodal forces and displacements onto the tactile taxel grid."""
    gx, gy = spec.taxel_grid_xy
    taxel_f = np.zeros((gx, gy, 3), dtype=np.float64)
    taxel_u = np.zeros((gx, gy, 3), dtype=np.float64)
    counts = np.zeros((gx, gy), dtype=np.float64)
    nodes = np.asarray(mesh.nodes, dtype=np.float64)

    for k, nid in enumerate(res.contact_node_indices):
        pt = nodes[nid]
        ix = min(gx - 1, max(0, int(pt[0] / spec.width_x_mm * gx)))
        iy = min(gy - 1, max(0, int(pt[1] / spec.length_y_mm * gy)))
        taxel_f[ix, iy] += res.contact_node_forces_n[k]
        taxel_u[ix, iy] += res.nodal_displacement[nid]
        counts[ix, iy] += 1.0

    taxel_u = taxel_u / np.maximum(counts[..., None], 1.0)
    # Slip ratio ||f_tangential|| / (mu * |f_normal|) across loaded taxels
    fn = np.abs(taxel_f[..., 2])
    ft = np.sqrt(taxel_f[..., 0] ** 2 + taxel_f[..., 1] ** 2)
    active = fn > 1.0e-5
    if np.any(active) and spec.friction_mu > 0.0:
        max_slip = float(np.max(ft[active] / (spec.friction_mu * fn[active])))
    else:
        max_slip = 0.0
    return taxel_f, taxel_u, max_slip


def simulate_tactile_probe_fem(
    spec: TactileSensorSpec,
    *,
    case_id: str,
    probe_x_mm: float,
    probe_y_mm: float,
    depth_mm: float,
    shear_dx_mm: float = 0.0,
    shear_dy_mm: float = 0.0,
    indenter_radius_mm: float = 4.0,
    indenter_kind: SdfPrimitiveKind = SdfPrimitiveKind.SPHERE,
    mesh: CanonicalMesh3D | None = None,
    load_steps: int = 2,
) -> TactileProbeSample:
    """Run full 3D finite-strain ``TET10`` SDF contact FEM for one tactile probe."""
    m = spec.build_tet10_mesh() if mesh is None else mesh
    z_top = spec.thickness_z_mm
    z_init = z_top + indenter_radius_mm + 0.02
    z_final = z_top + indenter_radius_mm - float(depth_mm)
    ind = RigidSdfIndenter3D(
        name=f"probe-{case_id}",
        kind=indenter_kind,
        contact_face_tags=("z_max",),
        initial_center_mm=(float(probe_x_mm), float(probe_y_mm), z_init),
        center_mm=(float(probe_x_mm), float(probe_y_mm), z_final),
        radius_mm=float(indenter_radius_mm),
        half_extents_mm=(
            float(indenter_radius_mm),
            float(indenter_radius_mm),
            float(indenter_radius_mm),
        ),
        tangential_shift_mm=(float(shear_dx_mm), float(shear_dy_mm), 0.0),
        penalty_mpa_per_mm=max(10.0, 60.0 * spec.youngs_modulus_mpa),
        friction_mu=spec.friction_mu
        if (shear_dx_mm != 0.0 or shear_dy_mm != 0.0)
        else 0.0,
    )
    res = solve_3d_sdf_contact(
        m,
        spec.material_spec(),
        dirichlet_bcs=(
            DirichletBC3D("base_clamp", ("z_min",), ("x", "y", "z"), (0.0, 0.0, 0.0)),
        ),
        indenter=ind,
        load_steps=load_steps,
        max_al_iterations=5,
        penetration_tol_mm=5.0e-3,
        solver_tolerance=1.0e-6,
    )
    if not res.converged:
        raise RuntimeError(f"tactile FEM probe {case_id!r} failed: {res.reason}")
    taxel_f, taxel_u, max_slip = project_contact_to_taxel_grid(spec, m, res)
    fx, fy, fz = res.total_contact_force_n
    tx, ty, tz = res.total_contact_moment_n_mm
    return TactileProbeSample(
        case_id=case_id,
        probe_x_mm=float(probe_x_mm),
        probe_y_mm=float(probe_y_mm),
        depth_mm=float(depth_mm),
        shear_dx_mm=float(shear_dx_mm),
        shear_dy_mm=float(shear_dy_mm),
        indenter_radius_mm=float(indenter_radius_mm),
        indenter_kind=indenter_kind,
        taxel_forces_n=taxel_f,
        surface_marker_disp_mm=taxel_u,
        wrench_6d=(fx, fy, fz, tx, ty, tz),
        max_slip_ratio=max_slip,
        peak_von_mises_mpa=float(res.quadrature_peak_von_mises_mpa),
        fem_wall_time_s=res.wall_time_s,
    )


class TactileElastomerNet(nn.Module):
    """Fast real-time (<1 ms) per-taxel force & 6D wrench operator."""

    def __init__(self, taxel_grid_xy: tuple[int, int], hidden_dim: int = 96) -> None:
        super().__init__()
        self.taxel_grid_xy = taxel_grid_xy
        n_taxels = taxel_grid_xy[0] * taxel_grid_xy[1]
        in_dim = 6 + 3 * n_taxels  # (x, y, depth, dx, dy, radius) + marker_disp
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.taxel_head = nn.Linear(hidden_dim, 3 * n_taxels)
        self.wrench_head = nn.Linear(hidden_dim, 7)  # 6D wrench + peak_vm

    def forward(
        self, x_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encoder(x_in)
        gx, gy = self.taxel_grid_xy
        taxels = self.taxel_head(latent).view(-1, gx, gy, 3)
        summary = self.wrench_head(latent)
        return taxels, summary, latent


@dataclass(frozen=True, slots=True)
class TactileCertifiedPrediction:
    """Real-time tactile force/wrench prediction with AS6983 conservative bounds."""

    taxel_forces_n: FloatArray  # (Gx, Gy, 3)
    wrench_6d: tuple[float, float, float, float, float, float]
    normal_force_n: float
    shear_force_n: float
    peak_von_mises_mpa: float
    conformal_assessment: Conformal3DAssessment | None
    inference_ms: float

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": TACTILE_PRED_SCHEMA,
            "wrench_6d": [float(v) for v in self.wrench_6d],
            "normal_force_n": float(self.normal_force_n),
            "shear_force_n": float(self.shear_force_n),
            "peak_von_mises_mpa": float(self.peak_von_mises_mpa),
            "conformal_assessment": (
                None
                if self.conformal_assessment is None
                else self.conformal_assessment.document()
            ),
            "inference_ms": float(self.inference_ms),
        }
        doc["receipt_sha256"] = _sha256_doc(doc)
        return doc


class TactileElastomerSurrogate:
    """End-to-end FEM-trained, AS6983-conformal tactile sensor digital twin."""

    def __init__(
        self,
        spec: TactileSensorSpec,
        *,
        hidden_dim: int = 96,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self.spec = spec
        self.model = TactileElastomerNet(spec.taxel_grid_xy, hidden_dim=hidden_dim)
        self.x_mean: FloatArray | None = None
        self.x_std: FloatArray | None = None
        self.conformal_record: Conformal3DCalibrationRecord | None = None

    def _featurize(self, s: TactileProbeSample) -> FloatArray:
        probe_vec = np.array(
            [
                s.probe_x_mm / self.spec.width_x_mm,
                s.probe_y_mm / self.spec.length_y_mm,
                s.depth_mm / self.spec.thickness_z_mm,
                s.shear_dx_mm / self.spec.thickness_z_mm,
                s.shear_dy_mm / self.spec.thickness_z_mm,
                s.indenter_radius_mm / self.spec.width_x_mm,
            ],
            dtype=np.float64,
        )
        return np.concatenate((probe_vec, s.surface_marker_disp_mm.reshape(-1)))

    def fit(
        self,
        train_samples: Sequence[TactileProbeSample],
        dev_samples: Sequence[TactileProbeSample] = (),
        *,
        epochs: int = 400,
        lr: float = 3.0e-3,
        alpha: float = 0.10,
    ) -> dict[str, float]:
        """Train surrogate on FEM tactile samples and calibrate AS6983 bounds on DEV."""
        if not train_samples:
            raise ValueError("train_samples must be non-empty")
        x_raw = np.stack([self._featurize(s) for s in train_samples], axis=0)
        self.x_mean = np.mean(x_raw, axis=0)
        self.x_std = np.maximum(np.std(x_raw, axis=0), 1.0e-6)
        x_norm = (x_raw - self.x_mean) / self.x_std

        y_tax = np.stack([s.taxel_forces_n for s in train_samples], axis=0)
        y_sum = np.stack(
            [np.array([*s.wrench_6d, s.peak_von_mises_mpa]) for s in train_samples],
            axis=0,
        )

        x_t = torch.from_numpy(x_norm.astype(np.float32))
        yt_t = torch.from_numpy(y_tax.astype(np.float32))
        ys_t = torch.from_numpy(y_sum.astype(np.float32))

        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1.0e-5)
        self.model.train()
        final_loss = 0.0
        for _ in range(epochs):
            opt.zero_grad()
            pred_t, pred_s, _ = self.model(x_t)
            # Enforce physics sum consistency: sum of taxel forces == net wrench force
            net_from_taxels = pred_t.sum(dim=(1, 2))
            loss_tax = (pred_t - yt_t).square().mean()
            loss_sum = (pred_s - ys_t).square().mean()
            loss_cons = (net_from_taxels - pred_s[:, :3]).square().mean()
            loss = loss_tax + loss_sum + 0.5 * loss_cons
            loss.backward()
            opt.step()
            final_loss = float(loss.item())

        self.model.eval()
        self.conformal_record = None
        if len(dev_samples) >= 4:
            dev_records: list[Conformal3DDevSample] = []
            for ds in dev_samples:
                pred_raw = self.predict_raw(ds)
                fn_pred = abs(pred_raw[1][2])
                fs_pred = math.hypot(pred_raw[1][0], pred_raw[1][1])
                fn_true = abs(ds.wrench_6d[2])
                fs_true = math.hypot(ds.wrench_6d[0], ds.wrench_6d[1])
                dev_records.append(
                    Conformal3DDevSample(
                        case_id=ds.case_id,
                        predicted_qois={
                            "normal_force_n": fn_pred,
                            "shear_force_n": fs_pred,
                            "peak_von_mises_mpa": float(pred_raw[2]),
                        },
                        true_qois={
                            "normal_force_n": fn_true,
                            "shear_force_n": fs_true,
                            "peak_von_mises_mpa": ds.peak_von_mises_mpa,
                        },
                        latent_vector=tuple(float(v) for v in pred_raw[3][:16]),
                    )
                )
            self.conformal_record = Conformal3DCalibrator.fit(
                _sha256_doc(self.spec.document()),
                dev_records,
                qoi_units={
                    "normal_force_n": "N",
                    "shear_force_n": "N",
                    "peak_von_mises_mpa": "MPa",
                },
                alpha=alpha,
            )
        n_cal = (
            0
            if self.conformal_record is None
            else len(self.conformal_record.qoi_calibrations)
        )
        return {"final_loss": final_loss, "calibrated_qois": float(n_cal)}

    def predict_raw(
        self, sample: TactileProbeSample
    ) -> tuple[FloatArray, tuple[float, ...], float, FloatArray]:
        if self.x_mean is None or self.x_std is None:
            raise RuntimeError(
                "TactileElastomerSurrogate must be fit before prediction"
            )
        x_norm = (self._featurize(sample) - self.x_mean) / self.x_std
        x_t = torch.from_numpy(x_norm[None, :].astype(np.float32))
        with torch.no_grad():
            pred_t, pred_s, latent = self.model(x_t)
        tax_np = pred_t[0].numpy().astype(np.float64)
        sum_np = pred_s[0].numpy().astype(np.float64)
        lat_np = latent[0].numpy().astype(np.float64)
        wrench = tuple(float(v) for v in sum_np[:6])
        return tax_np, wrench, float(max(0.0, sum_np[6])), lat_np

    def predict(self, sample: TactileProbeSample) -> TactileCertifiedPrediction:
        """Predict per-taxel forces, 6D wrench, and AS6983 conservative safety bounds."""
        t0 = time.perf_counter()
        tax_np, wrench, peak_vm, lat_np = self.predict_raw(sample)
        fn = abs(wrench[2])
        fs = math.hypot(wrench[0], wrench[1])
        lat_vec = tuple(float(v) for v in lat_np[:16])
        assessment: Conformal3DAssessment | None = None
        if self.conformal_record is not None:
            assessment = self.conformal_record.evaluate(
                {
                    "normal_force_n": fn,
                    "shear_force_n": fs,
                    "peak_von_mises_mpa": peak_vm,
                },
                latent_vector=lat_vec,
            )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        w6 = (wrench[0], wrench[1], wrench[2], wrench[3], wrench[4], wrench[5])
        return TactileCertifiedPrediction(
            taxel_forces_n=tax_np,
            wrench_6d=w6,
            normal_force_n=fn,
            shear_force_n=fs,
            peak_von_mises_mpa=peak_vm,
            conformal_assessment=assessment,
            inference_ms=dt_ms,
        )


__all__ = (
    "TACTILE_PRED_SCHEMA",
    "TACTILE_SPEC_SCHEMA",
    "TactileCertifiedPrediction",
    "TactileElastomerNet",
    "TactileElastomerSurrogate",
    "TactileProbeSample",
    "TactileSensorSpec",
    "project_contact_to_taxel_grid",
    "simulate_tactile_probe_fem",
)
