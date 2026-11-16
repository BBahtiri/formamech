"""Share legacy implementation modules without duplicating contract classes.

Stored type tags and model identities retain their historical namespace. Imports
through FormaMech therefore resolve to the same modules, not re-executed copies.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any

_OWN_MODULES = frozenset({"formamech._compat", "formamech.cli", "formamech.__main__"})


class _LegacyLoader(importlib.abc.Loader):
    def __init__(self, legacy_name: str) -> None:
        self.legacy_name = legacy_name
        self.original_spec: ModuleSpec | None = None

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        module = importlib.import_module(self.legacy_name)
        self.original_spec = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        # module_from_spec sets __spec__ even on an existing module. Restore the
        # implementation's spec so reload and legacy imports keep their identity.
        module.__spec__ = self.original_spec


class _LegacyFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Any = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if not fullname.startswith("formamech.") or fullname in _OWN_MODULES:
            return None
        legacy = "piginot" + fullname[len("formamech") :]
        original = importlib.util.find_spec(legacy)
        if original is None:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return importlib.util.spec_from_loader(
            fullname,
            _LegacyLoader(legacy),
            is_package=original.submodule_search_locations is not None,
        )


def install() -> None:
    if not any(isinstance(finder, _LegacyFinder) for finder in sys.meta_path):
        # Run before PathFinder: aliased parents expose the implementation's
        # directory, which must never execute under a second module name.
        sys.meta_path.insert(0, _LegacyFinder())
