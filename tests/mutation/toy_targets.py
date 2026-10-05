"""Small targets for the mutation engine tests."""

from __future__ import annotations

import functools
from dataclasses import dataclass

LIMIT = 10


def add(left: int, right: int = 1) -> int:
    return left + right


@functools.cache
def cached_square(value: int) -> int:
    return value * value


@dataclass(frozen=True, slots=True)
class Counter:
    start: int

    def next(self) -> int:
        return self.start + 1

    @property
    def doubled(self) -> int:
        return self.start * 2

    @staticmethod
    def limit() -> int:
        return LIMIT


class Form:
    """An object that keeps its function as ``form``, like a weak form."""

    def __init__(self, form: object) -> None:
        self.form = form


def _kernel(value: int) -> int:
    return value - 1


KERNEL = Form(_kernel)


def _outer() -> object:
    offset = 3

    def inner(value: int) -> int:
        return value + offset

    return inner


INNER = _outer()
