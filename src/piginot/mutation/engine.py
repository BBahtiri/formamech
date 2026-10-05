"""Source-level mutants applied to the running code (S16).

A mutant is a set of exact source edits to production functions. Applying it
reads the function's own source, replaces one exact snippet, compiles the
edited definition and swaps the resulting code object into the original
function object. Every reference to that function therefore runs the defect:
imports by name, default arguments, bound methods, properties and decorator
wrappers alike. Leaving the context restores the original code object.

No defect is simulated next to production code: the code that runs is the
production function with one edit.
"""

from __future__ import annotations
import __future__

import importlib
import inspect
import textwrap
import types
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

_UNWRAP_ATTRIBUTES = ("fget", "__func__", "__wrapped__", "form")


class MutationError(RuntimeError):
    """A mutant cannot be applied exactly as declared."""


@dataclass(frozen=True, slots=True)
class SourceEdit:
    """Replace ``old`` with ``new`` once in the source of one function.

    ``target`` is ``"package.module:Qualified.name"``; properties, static and
    class methods, ``functools`` wrappers and objects that hold their function
    as ``form`` are unwrapped to the function whose code runs.
    """

    target: str
    old: str
    new: str

    def __post_init__(self) -> None:
        module, separator, qualname = self.target.partition(":")
        if not module or not separator or not qualname:
            raise ValueError(f"target must be 'module:qualname', got {self.target!r}")
        if not self.old or self.old == self.new:
            raise ValueError("an edit must change a non-empty snippet")


@dataclass(frozen=True, slots=True)
class Mutant:
    """A deliberate defect and the gate check that must detect it."""

    name: str
    primary: str
    description: str
    edits: tuple[SourceEdit, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.primary or not self.edits:
            raise ValueError("a mutant needs a name, a primary gate and edits")

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "primary": self.primary,
            "description": self.description,
            "edits": [
                {"target": edit.target, "old": edit.old, "new": edit.new}
                for edit in self.edits
            ],
        }


def resolve_function(target: str) -> types.FunctionType:
    """The function object whose code a ``module:qualname`` target runs."""

    module_name, _, qualname = target.partition(":")
    try:
        value: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise MutationError(f"cannot import {module_name!r}") from exc
    for part in qualname.split("."):
        owner = value
        value = inspect.getattr_static(owner, part, None)
        if value is None:
            raise MutationError(f"{target!r} has no attribute {part!r}")
    for _ in range(4):
        if isinstance(value, types.FunctionType):
            return value
        for attribute in _UNWRAP_ATTRIBUTES:
            inner = getattr(value, attribute, None)
            if inner is not None:
                value = inner
                break
        else:
            break
    raise MutationError(f"{target!r} does not resolve to a Python function")


def _definition_only(source: str, qualname: str) -> str:
    lines = textwrap.dedent(source).splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.lstrip().startswith(("def ", "async def ")):
            return "".join(lines[index:])
    raise MutationError(f"no definition found for {qualname}")


def mutated_code(
    function: types.FunctionType, edits: Sequence[SourceEdit]
) -> types.CodeType:
    """Compile ``function`` with ``edits`` applied, in its own module namespace.

    Snippets are matched against the source exactly as written in the file,
    indentation included.
    """

    source = inspect.getsource(function)
    for edit in edits:
        count = source.count(edit.old)
        if count != 1:
            raise MutationError(
                f"{edit.target}: snippet occurs {count} times, expected exactly once"
            )
        source = source.replace(edit.old, edit.new)
    source = _definition_only(source, function.__qualname__)
    filename = f"<mutant {function.__module__}:{function.__qualname__}>"
    flags = __future__.annotations.compiler_flag
    try:
        compiled = compile(source, filename, "exec", flags=flags, dont_inherit=True)
    except SyntaxError as exc:
        raise MutationError(f"{function.__qualname__}: edit is not valid") from exc
    namespace = dict(function.__globals__)
    namespace.pop(function.__name__, None)
    exec(compiled, namespace)  # noqa: S102 - the module's own source, edited
    replacement = namespace.get(function.__name__)
    if not isinstance(replacement, types.FunctionType):
        raise MutationError(f"{function.__qualname__}: edit removed the definition")
    if replacement.__code__.co_freevars != function.__code__.co_freevars:
        raise MutationError(f"{function.__qualname__}: closures differ after the edit")
    return replacement.__code__


@contextmanager
def _swapped(function: types.FunctionType, code: types.CodeType) -> Iterator[None]:
    original = function.__code__
    function.__code__ = code
    try:
        yield
    finally:
        function.__code__ = original


@contextmanager
def applied(mutant: Mutant) -> Iterator[None]:
    """Run the enclosed block with ``mutant`` active; always restore."""

    grouped: dict[str, list[SourceEdit]] = {}
    for edit in mutant.edits:
        grouped.setdefault(edit.target, []).append(edit)
    compiled = [(resolve_function(target), edits) for target, edits in grouped.items()]
    codes = [(function, mutated_code(function, edits)) for function, edits in compiled]
    with ExitStack() as stack:
        for function, code in codes:
            stack.enter_context(_swapped(function, code))
        yield


def clear_caches(functions: Sequence[Callable[..., Any]]) -> None:
    """Clear ``functools`` caches so no result computed before a swap is reused."""

    for function in functions:
        clear = getattr(function, "cache_clear", None)
        if clear is None:
            raise MutationError(f"{function!r} has no cache to clear")
        clear()


__all__ = (
    "Mutant",
    "MutationError",
    "SourceEdit",
    "applied",
    "clear_caches",
    "mutated_code",
    "resolve_function",
)
