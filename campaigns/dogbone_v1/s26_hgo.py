"""S26: the registered HGO viability run on DogBone (ADR 0108 decision 8).

The S25 recipe (``campaigns.dogbone_v1.material_run``) with the material of
``campaigns/dogbone_v1/problem_spec_hgo.json``: a neo-Hookean matrix with the
shipped material's small-strain moduli and two HGO fibre families at ±30°
(``k1 = 50`` MPa, ``k2 = 2``), orthotropic about the loading axes so the
quarter model stays valid. Usage (repository root, clean committed revision)::

    python -m campaigns.dogbone_v1.s26_hgo protocol --write
    python -m campaigns.dogbone_v1.s26_hgo references \
        --s10-run-root <S10 root> --output <references dir> --workers 6 --write
    python -m campaigns.dogbone_v1.s26_hgo train \
        --s10-run-root <S10 root> --run-root <new run root>
    python -m campaigns.dogbone_v1.s26_hgo evaluate \
        --s10-run-root <S10 root> --references <references dir> \
        --checkpoint-path <run root>/checkpoint-step-2400 --write
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from campaigns.dogbone_v1.material_run import MaterialViability

RUN = MaterialViability(
    step="S26",
    model="hgo",
    label="HGO",
    spec_path=Path("campaigns/dogbone_v1/problem_spec_hgo.json"),
    protocol_path=Path("artifacts/materials/S26-PROTOCOL.json"),
    references_path=Path("artifacts/materials/S26-HGO-REFERENCES.json"),
    screen_path=Path("artifacts/materials/S26-HGO-TRAIN-SCREEN.json"),
    decision_record="docs/adr/0108-s26-fibre-families.md",
    accepted_on="2026-10-14",
    answers={
        "ADR 0108": "Accept as written",
        "angle unit": "Degrees, new unit deg",
        "viability material": "±30°, k1=50 MPa, k2=2",
        "kill rule": "Registry, isotropic training only",
    },
    material_note={
        "matrix_matches": "small-strain E = 760 MPa, nu = 0.23 (fibres carry no "
        "load at the reference)",
        "fibres": "two in-plane families at +30 and -30 degrees from the loading "
        "axis; quarter-model symmetric",
    },
    kill_rule="if a guardrail fails, 0.2.0 keeps fibre formulas in the registry "
    "and the admission gate, but the S28 compiler trains isotropic models only",
)


def main(arguments: Sequence[str] | None = None) -> int:
    return RUN.main(arguments, prog="python -m campaigns.dogbone_v1.s26_hgo")


if __name__ == "__main__":
    sys.exit(main())
