"""The FormaMech command line and request parser."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from piginot import cli as _legacy
from piginot.cli import parse_request as parse_request


def build_parser() -> argparse.ArgumentParser:
    return _legacy.build_parser(prog="formamech")


def main(argv: Sequence[str] | None = None) -> int:
    return _legacy.main(argv, prog="formamech")


if __name__ == "__main__":
    raise SystemExit(main())
