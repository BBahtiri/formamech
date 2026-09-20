from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_training_import_does_not_load_fem_verification_modules() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(
                None,
                (str(root / "src"), os.environ.get("PYTHONPATH", "")),
            )
        ),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import piginot.training; "
                "assert 'piginot.fem' not in sys.modules; "
                "assert 'piginot.objectives.verification' not in sys.modules; "
                "assert 'skfem' not in sys.modules"
            ),
        ],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
