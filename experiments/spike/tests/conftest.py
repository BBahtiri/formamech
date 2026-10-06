"""Skip the spike tests that need the third-party GINOT modules when absent.

The modules are not redistributed (ADR 0032); ``python -m
experiments.spike.fetch_upstream`` fetches them from the pinned upstream
commit. The geometry, FEM and plotting tests run without them.
"""

from __future__ import annotations

from experiments.spike.fetch_upstream import missing_files

NEEDS_UPSTREAM = (
    "test_confirm.py",
    "test_iteration8.py",
    "test_model.py",
    "test_run.py",
)

collect_ignore = list(NEEDS_UPSTREAM) if missing_files() else []
