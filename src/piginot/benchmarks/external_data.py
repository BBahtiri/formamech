"""License-tiered real-world V&V dataset registry & zero-network fallback cache (V5-3).

Enforces pure-MIT GitHub repository hygiene:

* **Tier A (`commercial_clean`):** Public Domain, MIT, BSD-3, Apache-2.0, CC-BY-4.0,
  ODC-By v1.0. Safe for verification, validation, and commercial pipelines.
* **Tier B (`eval_only_sharealike`):** CC-BY-SA-4.0 / ODbL-1.0. Allowed for local
  evaluation and publishing aggregate metrics only; never bundle raw files or
  shipped weights trained on ShareAlike data.
* **Tier C (`noncommercial_opt_in`):** CC-BY-NC-4.0 / GrabCAD NC. Refused by default
  unless caller explicitly sets ``allow_noncommercial=True``.
* **GPL-3.0 Excluded:** Repositories such as ``BRBtesting`` and ``zenodo/7635861``
  are explicitly excluded to keep FormaMech strictly pure-MIT.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np

from piginot.contracts import canonical_json_bytes
from piginot.geometry.mesh3d import make_brick_tet4_mesh

EXTERNAL_DATASET_MANIFEST_SCHEMA: Final = "formamech.external-dataset-cache-v1"


class LicenseTier(StrEnum):
    """Explicit IP / license safety tier for external open-source V&V datasets."""

    TIER_A_COMMERCIAL_CLEAN = "TIER_A_COMMERCIAL_CLEAN"
    TIER_B_EVAL_ONLY_SHAREALIKE = "TIER_B_EVAL_ONLY_SHAREALIKE"
    TIER_C_NONCOMMERCIAL_OPT_IN = "TIER_C_NONCOMMERCIAL_OPT_IN"


@dataclass(frozen=True, slots=True)
class ExternalDatasetSpec:
    """Pinned metadata record for one real-world open-source V&V corpus."""

    name: str
    title: str
    domain: str
    url: str
    license_name: str
    license_tier: LicenseTier
    milestones: tuple[str, ...]
    description: str
    expected_sha256: str | None = None

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "domain": self.domain,
            "url": self.url,
            "license_name": self.license_name,
            "license_tier": self.license_tier.value,
            "milestones": list(self.milestones),
            "description": self.description,
            "expected_sha256": self.expected_sha256,
        }


DATASET_REGISTRY: Final[Mapping[str, ExternalDatasetSpec]] = {
    "nist-ambench": ExternalDatasetSpec(
        name="nist-ambench",
        title="NIST AM-Bench (2018 / 2022 / 2025)",
        domain="thermo-mechanical-warpage",
        url="https://www.nist.gov/ambench",
        license_name="US Public Domain",
        license_tier=LicenseTier.TIER_A_COMMERCIAL_CLEAN,
        milestones=("V5-2", "V5-5"),
        description="Physical 3D IN625/IN718 thermo-mechanical deflection, residual stress, and in-situ thermography.",
    ),
    "nasa-pcoe": ExternalDatasetSpec(
        name="nasa-pcoe",
        title="NASA PCoE Component Degradation (MOSFET, Capacitors, Battery)",
        domain="power-electronics-degradation",
        url="https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
        license_name="US Public Domain",
        license_tier=LicenseTier.TIER_A_COMMERCIAL_CLEAN,
        milestones=("V5-2", "V5-5"),
        description="Physical run-to-failure power MOSFET thermal overstress, capacitor electrical stress, and randomized battery usage.",
    ),
    "matr-attia-clo": ExternalDatasetSpec(
        name="matr-attia-clo",
        title="Attia et al. (2020) Closed-Loop Battery Optimization (MATR.io)",
        domain="active-test-planning",
        url="https://data.matr.io/1",
        license_name="CC-BY-4.0",
        license_tier=LicenseTier.TIER_A_COMMERCIAL_CLEAN,
        milestones=("V5-2",),
        description="224 six-step fast-charging experimental protocols for benchmarking retrospective physical test reduction.",
    ),
    "doe-msu-fatigue": ExternalDatasetSpec(
        name="doe-msu-fatigue",
        title="DOE/MSU Composite Fatigue & Sandia Fracture Challenge 3",
        domain="fatigue-fracture",
        url="https://www.osti.gov/servlets/purl/1145693",
        license_name="US Gov / Open Research",
        license_tier=LicenseTier.TIER_A_COMMERCIAL_CLEAN,
        milestones=("V5-2",),
        description="Physical S-N/E-N fatigue curves and notched stereo-DIC fracture holdouts.",
    ),
    "deepjeb": ExternalDatasetSpec(
        name="deepjeb",
        title="DeepJEB & DeepJEB++ (KAIST Smart Design Lab)",
        domain="3d-structural-brackets",
        url="https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB",
        license_name="ODC-By v1.0",
        license_tier=LicenseTier.TIER_A_COMMERCIAL_CLEAN,
        milestones=("V5-1", "V5-3"),
        description="2,138-15,360 3D Ti-6Al-4V structural brackets across 4 load cases for multi-scale & BC-shift benchmarks.",
    ),
    "abc-cad": ExternalDatasetSpec(
        name="abc-cad",
        title="ABC Dataset (1M OpenCASCADE B-Rep CAD Solids)",
        domain="brep-cad-geometry",
        url="https://github.com/deep-geometry/abc-dataset",
        license_name="MIT",
        license_tier=LicenseTier.TIER_A_COMMERCIAL_CLEAN,
        milestones=("V5-4",),
        description="Analytic OpenCASCADE STEP B-Rep solids for watertightness & round-trip stress testing.",
    ),
    "euclid-dic": ExternalDatasetSpec(
        name="euclid-dic",
        title="EUCLID Full-Field DIC Hyperelastic Datasets (ETH Zurich)",
        domain="full-field-dic",
        url="https://doi.org/10.3929/ethz-b-000505693",
        license_name="Open ETH Research Collection (Eval-Only)",
        license_tier=LicenseTier.TIER_B_EVAL_ONLY_SHAREALIKE,
        milestones=("V5-2",),
        description="Real full-field DIC displacement/strain fields and reaction forces on perforated hyperelastic coupons.",
    ),
    "ic-thermbench": ExternalDatasetSpec(
        name="ic-thermbench",
        title="IC-ThermBench (2.5D/3D-IC Chiplet Thermal)",
        domain="electronics-etm",
        url="https://github.com/Day333/ThermalBench",
        license_name="Open Research (Eval-Only)",
        license_tier=LicenseTier.TIER_B_EVAL_ONLY_SHAREALIKE,
        milestones=("V5-5",),
        description="50,000 2.5D/3D-IC steady and transient chiplet thermal layouts with multi-material power maps.",
    ),
    "mechanical-mnist-buckling": ExternalDatasetSpec(
        name="mechanical-mnist-buckling",
        title="Mechanical MNIST Buckling Instability Corpus (BIC/ABC)",
        domain="buckling-instability",
        url="https://open.bu.edu/handle/2144/40085",
        license_name="CC-BY-SA-4.0",
        license_tier=LicenseTier.TIER_B_EVAL_ONLY_SHAREALIKE,
        milestones=("V5-1",),
        description="Finite-strain Neo-Hookean sheets with labeled buckling bifurcations for regime-guard evaluation only.",
    ),
    "simjeb": ExternalDatasetSpec(
        name="simjeb",
        title="SimJEB (Simulated Jet Engine Bracket)",
        domain="3d-structural-brackets",
        url="https://doi.org/10.7910/DVN/XFUWJG",
        license_name="ODC-By v1.0 DB / GrabCAD Non-Commercial STEP",
        license_tier=LicenseTier.TIER_C_NONCOMMERCIAL_OPT_IN,
        milestones=("V5-1", "V5-3"),
        description="381 crowdsourced 3D brackets across 4 load cases (non-commercial opt-in due to GrabCAD STEP terms).",
    ),
    "shift-battery": ExternalDatasetSpec(
        name="shift-battery",
        title="Luminary SHIFT-Battery Cold Plate CHT",
        domain="cold-plate-cht",
        url="https://huggingface.co/datasets/luminary-shift/Battery",
        license_name="CC-BY-NC-4.0",
        license_tier=LicenseTier.TIER_C_NONCOMMERCIAL_OPT_IN,
        milestones=("V5-5",),
        description="1,300+ battery cold-plate CHT runs (strictly non-commercial opt-in evaluation only).",
    ),
}


def list_external_datasets(
    *, include_noncommercial: bool = False
) -> list[dict[str, Any]]:
    """Return canonical dataset specifications filtered by license safety tier."""
    out: list[dict[str, Any]] = []
    for key in sorted(DATASET_REGISTRY):
        spec = DATASET_REGISTRY[key]
        if (
            spec.license_tier is LicenseTier.TIER_C_NONCOMMERCIAL_OPT_IN
            and not include_noncommercial
        ):
            continue
        out.append(spec.document())
    return out


def resolve_dataset_spec(
    name: str, *, allow_noncommercial: bool = False
) -> ExternalDatasetSpec:
    """Resolve a dataset spec while blocking Tier C non-commercial corpora unless opted in."""
    if name not in DATASET_REGISTRY:
        raise LookupError(
            f"unknown external dataset {name!r}; available: {sorted(DATASET_REGISTRY)}"
        )
    spec = DATASET_REGISTRY[name]
    if (
        spec.license_tier is LicenseTier.TIER_C_NONCOMMERCIAL_OPT_IN
        and not allow_noncommercial
    ):
        raise PermissionError(
            f"Dataset {name!r} is Tier C ({spec.license_name}) and blocked from "
            "commercial-clean runs by default. Pass allow_noncommercial=True "
            "(or CLI --allow-noncommercial) for local non-commercial evaluation only."
        )
    return spec


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedDatasetCache:
    """Verified local dataset cache directory (real download or MIT synthetic fallback)."""

    spec: ExternalDatasetSpec
    cache_dir: Path
    source_mode: str  # "verified_local" | "mit_synthetic_fallback"
    sample_files: tuple[Path, ...]
    manifest_sha256: str

    def document(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_DATASET_MANIFEST_SCHEMA,
            "dataset": self.spec.document(),
            "cache_dir": str(self.cache_dir),
            "source_mode": self.source_mode,
            "sample_count": len(self.sample_files),
            "manifest_sha256": self.manifest_sha256,
        }


def prepare_dataset_cache(
    name: str,
    cache_root: Path | str,
    *,
    allow_noncommercial: bool = False,
    allow_synthetic_fallback: bool = True,
    synthetic_sample_count: int = 8,
) -> PreparedDatasetCache:
    """Verify an existing local external dataset cache or emit a pure-MIT synthetic CI fixture."""
    spec = resolve_dataset_spec(name, allow_noncommercial=allow_noncommercial)
    ds_dir = Path(cache_root).resolve() / spec.name
    ds_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ds_dir / "cache_manifest.json"

    existing = sorted(p for p in ds_dir.glob("*.npz") if p.is_file())
    if existing and manifest_path.is_file():
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        for fname, expected in raw.get("file_sha256", {}).items():
            fpath = ds_dir / fname
            if not fpath.is_file() or _sha256_file(fpath) != expected:
                raise ValueError(f"SHA-256 verification failed for cached file {fpath}")
        return PreparedDatasetCache(
            spec=spec,
            cache_dir=ds_dir,
            source_mode=str(raw.get("source_mode", "verified_local")),
            sample_files=tuple(existing),
            manifest_sha256=_sha256_file(manifest_path),
        )

    if not allow_synthetic_fallback:
        raise FileNotFoundError(
            f"External dataset {spec.name!r} not found in {ds_dir}. "
            f"Download from {spec.url} (license: {spec.license_name}) into {ds_dir} "
            "or enable allow_synthetic_fallback=True for zero-network CI."
        )

    # Deterministic pure-MIT synthetic fallback corpus matching SimJEB/DeepJEB schema
    sample_paths: list[Path] = []
    file_hashes: dict[str, str] = {}
    for idx in range(max(4, synthetic_sample_count)):
        lx = 16.0 + 1.5 * (idx % 5)
        ly = 8.0 + 0.8 * (idx % 3)
        lz = 6.0 + 0.5 * (idx % 2)
        mesh = make_brick_tet4_mesh(lx, ly, lz, nx=2, ny=2, nz=2)
        nodes = np.asarray(mesh.nodes, dtype=np.float64)
        cells = np.asarray(mesh.cells, dtype=np.int64)
        load_mag = 1.0 + 0.25 * (idx % 4)
        # Smooth physics-consistent cantilever-like synthetic displacement (positive det F)
        x_norm = nodes[:, 0] / lx
        u = np.zeros_like(nodes)
        u[:, 0] = 0.015 * load_mag * x_norm
        u[:, 1] = -0.04 * load_mag * (x_norm**2)
        u[:, 2] = 0.005 * load_mag * x_norm * (nodes[:, 2] / lz - 0.5)
        vm = 12.0 * load_mag * (1.0 - 0.5 * x_norm) + 1.0
        fpath = ds_dir / f"{spec.name}_sample_{idx:03d}.npz"
        np.savez_compressed(
            fpath,
            nodes=nodes,
            cells=cells,
            displacement=u,
            von_mises_mpa=vm,
            parameters=np.asarray([lx / 20.0, load_mag], dtype=np.float64),
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "synthetic_mit_fallback": True,
                        "dataset": spec.name,
                        "load_case_id": f"lc_{(idx % 4) + 1}",
                    }
                )
            ),
        )
        sample_paths.append(fpath)
        file_hashes[fpath.name] = _sha256_file(fpath)

    manifest_doc = {
        "schema": EXTERNAL_DATASET_MANIFEST_SCHEMA,
        "dataset": spec.document(),
        "source_mode": "mit_synthetic_fallback",
        "file_sha256": file_hashes,
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest_doc))
    return PreparedDatasetCache(
        spec=spec,
        cache_dir=ds_dir,
        source_mode="mit_synthetic_fallback",
        sample_files=tuple(sample_paths),
        manifest_sha256=_sha256_file(manifest_path),
    )


__all__ = (
    "DATASET_REGISTRY",
    "EXTERNAL_DATASET_MANIFEST_SCHEMA",
    "ExternalDatasetSpec",
    "LicenseTier",
    "PreparedDatasetCache",
    "list_external_datasets",
    "prepare_dataset_cache",
    "resolve_dataset_spec",
)
