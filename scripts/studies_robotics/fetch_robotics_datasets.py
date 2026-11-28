"""On-demand fetcher & SHA-256 verifier for real open-source robotics V&V datasets.

Writes strictly into gitignored ``.formamech-runs/datasets/robotics/`` — raw
external datasets are never committed to Git or packaged in any wheel/sdist.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / ".formamech-runs" / "datasets" / "robotics"

# Verified direct download URLs for real physical open-source robotics datasets
DATASET_FILES: dict[str, dict[str, Any]] = {
    "r1_tactile": {
        "license": "MIT (FEATS CAD) + CC-BY-4.0 (GlowTact CNC force logs)",
        "tier": "TIER_A_COMMERCIAL_CLEAN",
        "files": {
            "gelsight_mini.step": "https://raw.githubusercontent.com/feats-ai/feats/main/src/calculix/obj/gelsight_mini/model/gelsight_mini.step",
            "glowtact_readme.md": "https://huggingface.co/datasets/yxma/glowtact-gelsight-tactile/resolve/main/README.md",
            "force_mon_010204.csv": "https://huggingface.co/datasets/yxma/glowtact-gelsight-tactile/resolve/main/shared/force/mon_20260712_010204.csv",
            "force_mon_012006.csv": "https://huggingface.co/datasets/yxma/glowtact-gelsight-tactile/resolve/main/shared/force/mon_20260712_012006.csv",
            "force_mon_014515.csv": "https://huggingface.co/datasets/yxma/glowtact-gelsight-tactile/resolve/main/shared/force/mon_20260712_014515.csv",
            "force_mon_045817.csv": "https://huggingface.co/datasets/yxma/glowtact-gelsight-tactile/resolve/main/shared/force/mon_20260717_045817.csv",
            "force_mon_045842.csv": "https://huggingface.co/datasets/yxma/glowtact-gelsight-tactile/resolve/main/shared/force/mon_20260717_045842.csv",
            "force_mon_045948.csv": "https://huggingface.co/datasets/yxma/glowtact-gelsight-tactile/resolve/main/shared/force/mon_20260717_045948.csv",
        },
    },
    "r2_continuum": {
        "license": "MIT (CRL-Dataset-CTCR-Pose) + CC-BY-4.0 (Gotelli et al. Figshare)",
        "tier": "TIER_A_COMMERCIAL_CLEAN",
        "files": {
            "CRL-Dataset-CTCR-Pose.csv": "https://raw.githubusercontent.com/ContinuumRoboticsLab/CRL-Dataset-CTCR-Pose/main/CRL-Dataset-CTCR-Pose.csv",
            "gotelli_code.zip": "https://ndownloader.figshare.com/files/69262969",
            "gotelli_custom_components.zip": "https://ndownloader.figshare.com/files/65558259",
        },
    },
    "r3_soft_pneumatic": {
        "license": "CC-BY-4.0 (Joshi & Paik Zenodo 3627901 + Pneumatic Hysteresis Zenodo 15584329) + CC-BY-SA-4.0 eval-only (SPONGE RepoLUH 19854)",
        "tier": "TIER_A_AND_TIER_B_EVAL_ONLY",
        "files": {
            "joshi_pulling_force.zip": "https://zenodo.org/api/records/3627901/files/Raw%20Data%20-%20Pulling%20Force.zip/content",
            "joshi_tip_force.zip": "https://zenodo.org/api/records/3627901/files/Raw%20Data%20-%20Tip%20Force.zip/content",
            "hysteresis_zenodo_15584329.csv": "https://zenodo.org/api/records/15584329/files/data.csv/content",
            "sponge_5dof.zip": "https://repo.uni-hannover.de/server/api/core/bitstreams/0fe7fa2e-ad9d-494b-9b24-a000e215b81c/content",
        },
    },
    "r4_surgical_tissue": {
        "license": "CC-BY-4.0 (Zenodo 8095559 & 21793225 Hyperelastic Human Brain 1-7)",
        "tier": "TIER_A_COMMERCIAL_CLEAN",
        "files": {
            "HBE_Data.zip": "https://zenodo.org/api/records/8095559/files/HBE_Data.zip/content",
            "HBE_README.txt": "https://zenodo.org/api/records/8095559/files/README.txt/content",
            "HBE_1-7_avg_NeoHooke_params.csv": "https://zenodo.org/api/records/21793225/files/HBE_1-7_avg_NeoHooke_params.csv/content",
        },
    },
    "r5_nist_reassemble": {
        "license": "CC-BY-4.0 (REASSEMBLE / NIST Assembly Task Board #1)",
        "tier": "TIER_A_COMMERCIAL_CLEAN",
        "files": {
            "reassemble_readme.md": "https://huggingface.co/datasets/robot-lev/reassemble/resolve/main/README.md",
            "reassemble_info.json": "https://huggingface.co/datasets/robot-lev/reassemble/resolve/main/meta/info.json",
            "reassemble_episodes.parquet": "https://huggingface.co/datasets/robot-lev/reassemble/resolve/main/meta/episodes/chunk-000/file-000.parquet",
            "reassemble_chunk000_file000.parquet": "https://huggingface.co/datasets/robot-lev/reassemble/resolve/main/data/chunk-000/file-000.parquet",
        },
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return sha256_file(dest)
    req = urllib.request.Request(url, headers={"User-Agent": "FormaMech-V6-VV10/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        tmp = dest.with_suffix(dest.suffix + ".part")
        h = hashlib.sha256()
        with tmp.open("wb") as out:
            while True:
                buf = resp.read(1 << 18)
                if not buf:
                    break
                out.write(buf)
                h.update(buf)
        tmp.replace(dest)
        return h.hexdigest()


def fetch_group(group: str) -> dict[str, Any]:
    spec = DATASET_FILES[group]
    gdir = CACHE_ROOT / group
    gdir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "group": group,
        "license": spec["license"],
        "tier": spec["tier"],
        "files": {},
    }
    for fname, url in spec["files"].items():
        dest = gdir / fname
        try:
            digest = download_file(url, dest)
            size = dest.stat().st_size
            manifest["files"][fname] = {
                "url": url,
                "size_bytes": size,
                "sha256": digest,
                "status": "ok",
            }
            print(
                f"[OK] {group}/{fname}: {size:,} bytes sha256={digest[:12]}...",
                flush=True,
            )
        except Exception as exc:
            manifest["files"][fname] = {"url": url, "status": f"error: {exc}"}
            print(f"[ERR] {group}/{fname}: {exc}", flush=True)
    (gdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def fetch_glowtact_probe_index(gdir: Path) -> int:
    """Fetch lightweight HuggingFace tree JSON to index GlowTact CNC (x, y, z, F) probes."""
    out_path = gdir / "glowtact_probes.json"
    if out_path.is_file() and out_path.stat().st_size > 1000:
        return len(json.loads(out_path.read_text(encoding="utf-8")))
    pat = re.compile(
        r"([a-z_]+)\|(\d+)pos\|([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\s*f\|([-\d\.]+)\.jpg$"
    )
    probes: list[dict[str, Any]] = []
    for shape in ("round", "quad", "triangle", "five_star"):
        url = (
            "https://huggingface.co/api/datasets/yxma/glowtact-gelsight-tactile/"
            f"tree/main/gelsight_mini/mini_pad0/Mini/{shape}"
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": "FormaMech-V6-VV10/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            items = json.loads(resp.read().decode("utf-8"))
        for it in items:
            name = str(it.get("path", "")).split("/")[-1]
            m = pat.match(name)
            if m:
                probes.append(
                    {
                        "shape": m.group(1),
                        "sample_id": int(m.group(2)),
                        "x_mm": float(m.group(3)),
                        "y_mm": float(m.group(4)),
                        "z_mm": float(m.group(5)),
                        "force_n": float(m.group(6)),
                    }
                )
    out_path.write_text(json.dumps(probes, indent=2), encoding="utf-8")
    return len(probes)


def main(groups: list[str] | None = None) -> int:
    selected = groups if groups else list(DATASET_FILES)
    summary: dict[str, Any] = {}
    for g in selected:
        summary[g] = fetch_group(g)
        if g == "r1_tactile":
            n_probes = fetch_glowtact_probe_index(CACHE_ROOT / g)
            summary[g]["glowtact_probe_index_count"] = n_probes
            print(
                f"[OK] r1_tactile/glowtact_probes.json: {n_probes} CNC probes",
                flush=True,
            )
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "fetch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
