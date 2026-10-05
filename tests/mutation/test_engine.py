from __future__ import annotations

import pytest

from piginot.mutation import (
    Mutant,
    MutationError,
    SourceEdit,
    applied,
    clear_caches,
    mutated_code,
    resolve_function,
)
from tests.mutation import toy_targets as toy

MODULE = "tests.mutation.toy_targets"


def _mutant(*edits: SourceEdit) -> Mutant:
    return Mutant("TOY", "G0.toy", "toy", edits)


def test_targets_resolve_through_wrappers() -> None:
    assert resolve_function(f"{MODULE}:add") is toy.add
    assert resolve_function(f"{MODULE}:Counter.next") is toy.Counter.next
    assert resolve_function(f"{MODULE}:Counter.doubled").__name__ == "doubled"
    assert resolve_function(f"{MODULE}:Counter.limit").__name__ == "limit"
    assert resolve_function(f"{MODULE}:cached_square") is toy.cached_square.__wrapped__
    assert resolve_function(f"{MODULE}:KERNEL") is toy._kernel
    for target in (f"{MODULE}:LIMIT", f"{MODULE}:missing", "no.such.module:add"):
        with pytest.raises(MutationError):
            resolve_function(target)


def test_edits_reach_every_reference_and_are_restored() -> None:
    alias = toy.add
    counter = toy.Counter(4)
    mutant = _mutant(
        SourceEdit(f"{MODULE}:add", "return left + right", "return left - right"),
        SourceEdit(f"{MODULE}:Counter.next", "self.start + 1", "self.start + 2"),
        SourceEdit(f"{MODULE}:Counter.doubled", "self.start * 2", "self.start * 3"),
        SourceEdit(f"{MODULE}:Counter.limit", "return LIMIT", "return LIMIT + 1"),
        SourceEdit(f"{MODULE}:KERNEL", "value - 1", "value - 2"),
    )
    with applied(mutant):
        assert alias(5) == 4  # default argument kept, imported name reached
        assert counter.next() == 6
        assert counter.doubled == 12
        assert toy.Counter.limit() == 11
        assert toy.KERNEL.form(5) == 3
    assert alias(5) == 6
    assert (counter.next(), counter.doubled, toy.Counter.limit()) == (5, 8, 10)
    assert toy.KERNEL.form(5) == 4


def test_a_failure_inside_the_block_still_restores() -> None:
    mutant = _mutant(SourceEdit(f"{MODULE}:add", "left + right", "left * right"))
    with pytest.raises(RuntimeError), applied(mutant):
        assert toy.add(3, 4) == 12
        raise RuntimeError("boom")
    assert toy.add(3, 4) == 7


def test_inexact_or_invalid_edits_are_refused() -> None:
    add = resolve_function(f"{MODULE}:add")
    for edit in (
        SourceEdit(f"{MODULE}:add", "missing snippet", "x"),
        SourceEdit(f"{MODULE}:add", "e", "E"),  # occurs more than once
        SourceEdit(f"{MODULE}:add", "return left + right", "return left +"),
        SourceEdit(f"{MODULE}:add", "def add(", "def other("),
    ):
        with pytest.raises(MutationError):
            mutated_code(add, (edit,))
    inner = resolve_function(f"{MODULE}:INNER")
    with pytest.raises(MutationError, match="closures"):
        mutated_code(inner, (SourceEdit(f"{MODULE}:INNER", "+ offset", "- offset"),))


def test_declarations_are_validated() -> None:
    with pytest.raises(ValueError):
        SourceEdit("no-colon", "a", "b")
    with pytest.raises(ValueError):
        SourceEdit(f"{MODULE}:add", "a", "a")
    with pytest.raises(ValueError):
        Mutant("", "G0", "x", ())
    document = _mutant(SourceEdit(f"{MODULE}:add", "+", "-")).document()
    assert document["edits"] == [{"target": f"{MODULE}:add", "old": "+", "new": "-"}]


def test_caches_are_cleared_and_must_exist() -> None:
    toy.cached_square(3)
    assert toy.cached_square.cache_info().currsize == 1
    clear_caches((toy.cached_square,))
    assert toy.cached_square.cache_info().currsize == 0
    with pytest.raises(MutationError):
        clear_caches((toy.add,))
