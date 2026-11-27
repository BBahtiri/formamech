"""Download and SHA-256-verify open-source 3D datasets into ``.formamech-runs/datasets/``.

Pure-MIT repository hygiene: raw third-party datasets are NEVER committed to Git.
This script fetches upstream benchmark files directly from their official DOIs /
repositories strictly into the gitignored ``.formamech-runs/datasets/`` cache and
records SHA-256 integrity digests.

Upstream datasets and licenses (commercial-clean open licenses):
  1. UARK-NED3/ColdPlate-CFD-2000sims
     DOI: 10.5061/dryad.k0p2ngfp5 | License: CC0-1.0 (Public Domain)
  2. HaomingLuo/AgentFEM-Bracket-Elasticity
     Source: HuggingFace | License: CC-BY-4.0
  3. HaomingLuo/AgentFEM-Pipe-Components
     Source: HuggingFace | License: CC-BY-4.0
  4. SUM4Re: FOS steel beam dataset (Jolastokieta pilot site, Basque Country, Spain)
     Authors: Jose Carlos Jimenez Fernandez & Mikel Ezquerro (EU SUM4Re)
     DOI: 10.5281/zenodo.22686007 | License: CC-BY-4.0
  5. CARPIUC Mortar Fracture Benchmark (LMT, ENS Paris-Saclay / CNRS UMR 8535)
     Author: Martin Poncelet (with A. Carpiuc-Prisacari)
     DOI: 10.5281/zenodo.3233491 | License: CC-BY-4.0
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from piginot.contracts import canonical_json_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_ROOT = REPO_ROOT / ".formamech-runs" / "datasets"
COLDPLATE_DIR = DATASETS_ROOT / "coldplate-cht-3d"
BRACKET_DIR = DATASETS_ROOT / "agentfem-bracket-3d"
PIPE_DIR = DATASETS_ROOT / "agentfem-pipe-3d"
SUM4RE_DIR = DATASETS_ROOT / "sum4re-fos-steel-beam"
CARPIUC_DIR = DATASETS_ROOT / "carpiuc-3d-fracture"

SUM4RE_FILE = "SUM4Re_FOS_STRAIN TEST BEAM1 JOLASTOKIETA.xlsx"
SUM4RE_EXPECTED_SHA256 = (
    "31133a36999d34384dc686dbd2040ec3e95f0251297bbfc411236a710b40722b"
)

CARPIUC_COUPON_SHA256: dict[str, str] = {
    "Specimen_RawData_ep3.csv": (
        "1878b98169b9f25aef5f73108b67a3ee9f5d97f5dbf5133beeb00b669cb1a5e6"
    ),
    "Specimen_RawData_ep4.csv": (
        "63b09a15ff3502e2b90574d991b62d32c85c61db781ab0dd4ec4e5a156dd9d00"
    ),
    "Specimen_RawData_ep5.csv": (
        "9d410c7b042f32d15ff63c43e2daf7ab4a3974b982e46b7fc3d4cfc2f54a1b2b"
    ),
    "Specimen_RawData_ep6.csv": (
        "dcc728f6583e08c756146e4f9564c3b844a62260f21f4b1025522b0081df3eb9"
    ),
    "Specimen_RawData_ep7.csv": (
        "5323c5b96b0b24b2ba35b1c6ae873d6da9c6a35c8ce6cf9795f455a33b64e19b"
    ),
    "Specimen_RawData_ep20.csv": (
        "07a20af33868ef1df6d8b80bac6a86c4490cb22a11030816e39922d093b053a4"
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    retries: int = 3,
) -> tuple[str, int, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        digest = sha256_file(dest)
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch on cached {dest.name}: "
                f"got {digest}, expected {expected_sha256}"
            )
        return dest.name, dest.stat().st_size, digest

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "FormaMech-DatasetFetch/1.0"}
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
            digest = hashlib.sha256(data).hexdigest()
            if expected_sha256 is not None and digest != expected_sha256:
                raise RuntimeError(
                    f"SHA-256 mismatch downloading {dest.name}: "
                    f"got {digest}, expected {expected_sha256}"
                )
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest.name, len(data), digest
        except Exception as exc:
            last_err = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Failed downloading {url}: {last_err}")


def fetch_coldplate(n_sims: int = 120) -> dict[str, Any]:
    t0 = time.perf_counter()
    COLDPLATE_DIR.mkdir(parents=True, exist_ok=True)
    base = (
        "https://huggingface.co/datasets/UARK-NED3/"
        "ColdPlate-CFD-2000sims/resolve/main/1_Dataset"
    )
    tasks: list[tuple[str, Path]] = [
        (f"{base}/model_setup.json", COLDPLATE_DIR / "model_setup.json"),
    ]
    for i in range(1, n_sims + 1):
        fname = f"sim_{i:04d}.npz"
        tasks.append((f"{base}/{fname}", COLDPLATE_DIR / fname))

    hashes: dict[str, str] = {}
    total_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_download_one, url, dest) for url, dest in tasks]
        for fut in concurrent.futures.as_completed(futs):
            name, sz, sha = fut.result()
            hashes[name] = sha
            total_bytes += sz

    manifest = {
        "dataset": "UARK-NED3/ColdPlate-CFD-2000sims",
        "doi": "10.5061/dryad.k0p2ngfp5",
        "paper_doi": "10.1016/j.aitf.2026.100040",
        "license": "CC0-1.0",
        "n_simulations": n_sims,
        "total_bytes": total_bytes,
        "file_sha256": dict(sorted(hashes.items())),
    }
    (COLDPLATE_DIR / "cache_manifest.json").write_bytes(canonical_json_bytes(manifest))
    print(
        f"[Study 1] ColdPlate-CFD-2000sims ({n_sims} sims, "
        f"{total_bytes / (1024 * 1024):.1f} MB, CC0-1.0) ready in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return manifest


def fetch_brackets() -> dict[str, Any]:
    t0 = time.perf_counter()
    BRACKET_DIR.mkdir(parents=True, exist_ok=True)
    base = (
        "https://huggingface.co/datasets/HaomingLuo/"
        "AgentFEM-Bracket-Elasticity/resolve/main"
    )
    files = [
        "train.csv",
        "validation.csv",
        "test.csv",
        "protocol.json",
        "fields-train-00.zip",
        "fields-train-01.zip",
        "fields-train-02.zip",
        "fields-train-03.zip",
        "fields-validation-00.zip",
        "fields-test-00.zip",
    ]
    hashes: dict[str, str] = {}
    total_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futs = [
            pool.submit(_download_one, f"{base}/{f}", BRACKET_DIR / f) for f in files
        ]
        for fut in concurrent.futures.as_completed(futs):
            name, sz, sha = fut.result()
            hashes[name] = sha
            total_bytes += sz

    manifest = {
        "dataset": "HaomingLuo/AgentFEM-Bracket-Elasticity",
        "license": "CC-BY-4.0",
        "total_bytes": total_bytes,
        "file_sha256": dict(sorted(hashes.items())),
    }
    (BRACKET_DIR / "cache_manifest.json").write_bytes(canonical_json_bytes(manifest))
    print(
        f"[Study 2a] AgentFEM-Bracket-Elasticity ({len(files)} files, "
        f"{total_bytes / (1024 * 1024):.1f} MB, CC-BY-4.0) ready in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return manifest


def fetch_pipes() -> dict[str, Any]:
    t0 = time.perf_counter()
    PIPE_DIR.mkdir(parents=True, exist_ok=True)
    api_url = (
        "https://huggingface.co/api/datasets/HaomingLuo/"
        "AgentFEM-Pipe-Components/tree/main/components"
    )
    req = urllib.request.Request(
        api_url, headers={"User-Agent": "FormaMech-DatasetFetch/1.0"}
    )
    tree = json.loads(urllib.request.urlopen(req, timeout=20).read())
    all_comps = sorted(e["path"].split("/")[-1] for e in tree)

    elbows = [c for c in all_comps if c.startswith("elbow-")]
    straights = [c for c in all_comps if c.startswith("straight-")]
    power_elbows = [c for c in all_comps if c.startswith("power-") and "elbow" in c][:6]
    selected = elbows + straights + power_elbows

    base = (
        "https://huggingface.co/datasets/HaomingLuo/"
        "AgentFEM-Pipe-Components/resolve/main/components"
    )
    tasks: list[tuple[str, Path, str]] = []
    for comp in selected:
        for fname in ("metadata.json", "mesh.npz", "component.npz"):
            tasks.append(
                (f"{base}/{comp}/{fname}", PIPE_DIR / comp / fname, f"{comp}/{fname}")
            )

    hashes: dict[str, str] = {}
    total_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        fut_map = {
            pool.submit(_download_one, url, dest): rel for url, dest, rel in tasks
        }
        for fut in concurrent.futures.as_completed(fut_map):
            rel = fut_map[fut]
            _, sz, sha = fut.result()
            hashes[rel] = sha
            total_bytes += sz

    manifest = {
        "dataset": "HaomingLuo/AgentFEM-Pipe-Components",
        "license": "CC-BY-4.0",
        "elbows": elbows,
        "straights": straights,
        "power_elbows_ood": power_elbows,
        "total_bytes": total_bytes,
        "file_sha256": dict(sorted(hashes.items())),
    }
    (PIPE_DIR / "cache_manifest.json").write_bytes(canonical_json_bytes(manifest))
    print(
        f"[Study 2b] AgentFEM-Pipe-Components ({len(elbows)} elbows, "
        f"{len(straights)} straights, {len(power_elbows)} power OOD elbows, "
        f"{total_bytes / (1024 * 1024):.1f} MB, CC-BY-4.0) ready in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return manifest


def fetch_sum4re() -> dict[str, Any]:
    t0 = time.perf_counter()
    SUM4RE_DIR.mkdir(parents=True, exist_ok=True)
    url = (
        "https://zenodo.org/api/records/22686007/files/"
        "SUM4Re_FOS_STRAIN%20TEST%20BEAM1%20JOLASTOKIETA.xlsx/content"
    )
    dest = SUM4RE_DIR / SUM4RE_FILE
    name, sz, sha = _download_one(url, dest, expected_sha256=SUM4RE_EXPECTED_SHA256)
    manifest = {
        "dataset": "SUM4Re: FOS steel beam dataset",
        "doi": "10.5281/zenodo.22686007",
        "license": "CC-BY-4.0",
        "total_bytes": sz,
        "file_sha256": {name: sha},
    }
    (SUM4RE_DIR / "cache_manifest.json").write_bytes(canonical_json_bytes(manifest))
    print(
        f"[Study 3] SUM4Re FOS Steel Beam ({sz / (1024 * 1024):.2f} MB, "
        f"CC-BY-4.0, sha256={sha[:12]}...) ready in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return manifest


def fetch_carpiuc() -> dict[str, Any]:
    t0 = time.perf_counter()
    CARPIUC_DIR.mkdir(parents=True, exist_ok=True)
    base = "https://zenodo.org/api/records/3233491/files"
    hashes: dict[str, str] = {}
    total_bytes = 0
    for fname, expected_sha in CARPIUC_COUPON_SHA256.items():
        name, sz, sha = _download_one(
            f"{base}/{fname}/content",
            CARPIUC_DIR / fname,
            expected_sha256=expected_sha,
        )
        hashes[name] = sha
        total_bytes += sz

    manifest = {
        "dataset": "CARPIUC Concrete/Mortar Fracture Benchmark",
        "doi": "10.5281/zenodo.3233491",
        "license": "CC-BY-4.0",
        "total_bytes": total_bytes,
        "file_sha256": dict(sorted(hashes.items())),
    }
    (CARPIUC_DIR / "cache_manifest.json").write_bytes(canonical_json_bytes(manifest))
    print(
        f"[Study 4] CARPIUC 6 Mortar Fracture Coupons "
        f"({total_bytes / 1024:.1f} KB, CC-BY-4.0) ready in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and SHA-256-verify open-source 3D datasets for Studies 1-4."
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=("1", "2", "3", "4", "all"),
        default=["all"],
        help="Which studies to fetch datasets for (default: all).",
    )
    args = parser.parse_args()
    selected = {"1", "2", "3", "4"} if "all" in args.studies else set(args.studies)
    if "1" in selected:
        fetch_coldplate(120)
    if "2" in selected:
        fetch_brackets()
        fetch_pipes()
    if "3" in selected:
        fetch_sum4re()
    if "4" in selected:
        fetch_carpiuc()
    print(
        "Dataset verification complete (strictly cached in .formamech-runs/datasets/)."
    )


if __name__ == "__main__":
    main()
