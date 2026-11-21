"""Entry-point plugin registry for pluggable :class:`SolverBackend` providers.

Default 2D reference oracle ``"scikit-fem"`` is always available when the
``verify``/``fem`` extra is installed. Additional backends (such as PyTorch GPU
FEM or optional LGPL-isolated FEniCSx/DOLFINx plugins) register via Python
entry points under ``formamech.fem.backends`` or programmatically via
:func:`register_backend`.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, Final

from piginot.fem.protocol import SolverBackend

FEM_BACKEND_GROUP: Final = "formamech.fem.backends"
BackendFactory = Callable[..., SolverBackend]

_REGISTERED_FACTORIES: dict[str, BackendFactory] = {}


def _default_scikit_fem(**kwargs: Any) -> SolverBackend:
    from piginot.fem.backend import ScikitFemBackend

    return ScikitFemBackend(**kwargs)


def _default_torch_fem(**kwargs: Any) -> SolverBackend:
    from piginot.fem.torch_backend import TorchFemBackend

    return TorchFemBackend(**kwargs)


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register an in-process :class:`SolverBackend` factory under ``name``."""
    if not isinstance(name, str) or not name.strip() or name.strip() != name:
        raise ValueError("backend name must be non-empty canonical text")
    if not callable(factory):
        raise TypeError("factory must be callable")
    _REGISTERED_FACTORIES[name] = factory


def unregister_backend(name: str) -> None:
    """Remove an in-process registered backend factory if present."""
    _REGISTERED_FACTORIES.pop(name, None)


def _entry_point_factories() -> dict[str, Any]:
    discovered: dict[str, Any] = {}
    for ep in entry_points(group=FEM_BACKEND_GROUP):
        discovered[ep.name] = ep
    return discovered


def available_backends() -> tuple[str, ...]:
    """Return sorted names of all discoverable :class:`SolverBackend` providers."""
    names = {"scikit-fem", "torch-fem"}
    names.update(_entry_point_factories())
    names.update(_REGISTERED_FACTORIES)
    return tuple(sorted(names))


def resolve_backend(name: str = "scikit-fem", **kwargs: Any) -> SolverBackend:
    """Instantiate a registered or entry-point :class:`SolverBackend` by ``name``."""
    if name in _REGISTERED_FACTORIES:
        backend = _REGISTERED_FACTORIES[name](**kwargs)
    elif name == "scikit-fem":
        backend = _default_scikit_fem(**kwargs)
    elif name == "torch-fem":
        backend = _default_torch_fem(**kwargs)
    else:
        eps = _entry_point_factories()
        if name not in eps:
            raise LookupError(
                f"unknown FEM solver backend {name!r}; "
                f"available: {list(available_backends())}"
            )
        loaded = eps[name].load()
        backend = loaded(**kwargs) if callable(loaded) else loaded
    if not isinstance(backend, SolverBackend):
        raise TypeError(
            f"backend {name!r} ({type(backend).__name__}) does not satisfy "
            "the SolverBackend protocol"
        )
    return backend


__all__ = (
    "FEM_BACKEND_GROUP",
    "BackendFactory",
    "available_backends",
    "register_backend",
    "resolve_backend",
    "unregister_backend",
)
