"""Hash manifests that prove a fork carries the same bytes as its source (G19).

Two commands:

``manifest``
    Walk one or more directories under a root and write a canonical JSON
    manifest: for every regular file its POSIX path relative to the root, its
    size and its BLAKE2b-256 digest, and optionally its modification time in
    nanoseconds. ``__pycache__`` directories and ``*.pyc`` files are skipped.
    The manifest carries its own digest over the canonical file table.

``compare``
    Compare two manifests. Exit code 0 when the file tables are identical,
    1 otherwise, with every missing, extra and changed path printed. Times are
    compared only when both manifests recorded them (``--mtime``), because Git
    does not preserve them; ignored run roots copied with ``robocopy /COPY:DAT
    /DCOPY:T`` do (ADR 0009 live-mtime evidence).

Run the same ``manifest`` command on the source checkout and on the fork, then
``compare`` the two files. Paths are relative, so the two roots may differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

FORMAT = "piginot-fork-manifest-v1"
SKIPPED_DIRECTORIES = frozenset({"__pycache__", ".git", ".venv"})
SKIPPED_SUFFIXES = (".pyc",)
CHUNK = 1 << 20


def file_digest(path: Path) -> str:
    """BLAKE2b-256 hex digest of the file's bytes."""

    digest = hashlib.blake2b(digest_size=32)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(root: Path, include: Sequence[str]) -> Iterable[Path]:
    for name in include:
        start = root / name
        if start.is_file():
            yield start
            continue
        if not start.is_dir():
            raise FileNotFoundError(f"included path does not exist: {start}")
        for path in sorted(start.rglob("*")):
            relative = path.relative_to(root)
            if any(part in SKIPPED_DIRECTORIES for part in relative.parts):
                continue
            if path.is_file() and not path.name.endswith(SKIPPED_SUFFIXES):
                yield path


def canonical(document: Mapping[str, Any]) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def build_manifest(
    root: Path, include: Sequence[str], *, mtime: bool = False
) -> dict[str, Any]:
    """Manifest of every file under ``root / include[i]``, keyed by POSIX path."""

    table: dict[str, dict[str, Any]] = {}
    for path in _files(root, include):
        entry: dict[str, Any] = {
            "bytes": path.stat().st_size,
            "blake2b": file_digest(path),
        }
        if mtime:
            entry["mtime_ns"] = path.stat().st_mtime_ns
        table[path.relative_to(root).as_posix()] = entry
    digests = {path: entry["blake2b"] for path, entry in table.items()}
    return {
        "format": FORMAT,
        "hash": "blake2b-256",
        "include": sorted(include),
        "mtime": mtime,
        "file_count": len(table),
        "files": table,
        "table_blake2b": hashlib.blake2b(
            canonical(digests), digest_size=32
        ).hexdigest(),
    }


def compare_manifests(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, list[str]]:
    """Paths missing from ``right``, extra in ``right``, and changed."""

    for document in (left, right):
        if document.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} document")
    a: Mapping[str, Mapping[str, Any]] = left["files"]
    b: Mapping[str, Mapping[str, Any]] = right["files"]
    times = bool(left.get("mtime")) and bool(right.get("mtime"))
    changed = []
    for path in sorted(set(a) & set(b)):
        keys = ("bytes", "blake2b", "mtime_ns") if times else ("bytes", "blake2b")
        if any(a[path].get(key) != b[path].get(key) for key in keys):
            changed.append(path)
    return {
        "missing": sorted(set(a) - set(b)),
        "extra": sorted(set(b) - set(a)),
        "changed": changed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("manifest", help="write a manifest")
    make.add_argument("--root", type=Path, required=True)
    make.add_argument("--include", nargs="+", required=True)
    make.add_argument("--mtime", action="store_true")
    make.add_argument("--out", type=Path, required=True)
    diff = commands.add_parser("compare", help="compare two manifests")
    diff.add_argument("left", type=Path)
    diff.add_argument("right", type=Path)
    arguments = parser.parse_args(argv)

    if arguments.command == "manifest":
        document = build_manifest(
            arguments.root, arguments.include, mtime=arguments.mtime
        )
        arguments.out.write_bytes(canonical(document) + b"\n")
        print(f"{document['file_count']} files, table {document['table_blake2b']}")
        return 0
    left = json.loads(arguments.left.read_text("utf-8"))
    right = json.loads(arguments.right.read_text("utf-8"))
    difference = compare_manifests(left, right)
    for kind, paths in difference.items():
        for path in paths:
            print(f"{kind}: {path}")
    identical = not any(difference.values())
    print("identical" if identical else "DIFFERENT")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
