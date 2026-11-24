"""Reproducible build script for the shipped ``bundles/foundation/struct3d-base-v1`` checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from piginot.training.foundation_pretrain import (
    FoundationPretrainConfig,
    pretrain_struct3d_foundation,
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    dest = repo_root / "bundles" / "foundation" / "struct3d-base-v1"
    cfg = FoundationPretrainConfig(
        checkpoint_id="struct3d-base-v1",
        samples_per_family=2,
        mesh_size_mm=9.5,
        epochs=15,
        hidden_dim=32,
        slice_count=6,
        num_blocks=2,
        num_heads=4,
        seed=7,
    )
    spec, summary = pretrain_struct3d_foundation(dest, config=cfg)
    readme = (
        "# `struct3d-base-v1` — Shipped 3D Structural Foundation Checkpoint\n\n"
        f"- **Checkpoint ID:** `{spec.checkpoint_id}`\n"
        f"- **Families:** {', '.join(summary['families'])}\n"
        f"- **Parameters:** {summary['parameter_count']:,}\n"
        f"- **Training Mode:** `{spec.training_mode}`\n"
        f"- **Claim Level:** `{spec.claim_level}` (must be qualified on target part before production deployment)\n"
        f"- **Weights SHA-256:** `{spec.weights_sha256}`\n\n"
        "Reproduce deterministically with:\n\n"
        "```bash\n"
        "python scripts/build_foundation_3d_checkpoint.py\n"
        "```\n"
    )
    (dest / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
