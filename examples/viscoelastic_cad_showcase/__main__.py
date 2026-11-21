"""CLI entrypoint for ``python -m examples.viscoelastic_cad_showcase``."""

from __future__ import annotations

import argparse
from pathlib import Path

from examples.viscoelastic_cad_showcase.run_showcase import (
    DEFAULT_OUTPUT_DIR,
    run_showcase,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end FormaMech viscoelastic CAD mount-bracket showcase."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated CAD, artifacts, JSON summary, and plots.",
    )
    args = parser.parse_args()
    run_showcase(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
