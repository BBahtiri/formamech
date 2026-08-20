import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from piginot import __version__
from piginot.contracts.canonical import canonical_json_bytes
from piginot.store import ObjectStore, StoreError


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
    prune_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "store" and args.store_command == "prune":
        try:
            result = ObjectStore.open(args.root).prune(dry_run=args.dry_run)
        except (OSError, StoreError) as exc:
            print(f"piginot: {exc}", file=sys.stderr)
            return 1
        report = {
            "dry_run": result.dry_run,
            "object_count": result.object_count,
            "pruned_bytes": result.pruned_bytes,
            "pruned_hashes": list(result.pruned_hashes),
        }
        print(canonical_json_bytes(report).decode("utf-8"))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
