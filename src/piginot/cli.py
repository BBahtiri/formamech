import argparse
from collections.abc import Sequence
from pathlib import Path

from piginot import __version__
from piginot.store import ObjectStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="piginot")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    store_parser = subparsers.add_parser("store")
    store_subparsers = store_parser.add_subparsers(
        dest="store_command",
        required=True,
    )
    prune_parser = store_subparsers.add_parser("prune")
    prune_parser.add_argument("--root", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "store" and args.store_command == "prune":
        ObjectStore(args.root).prune()
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
