"""Fetch the third-party GINOT modules the quarantined S1 spike needs.

The S1 spike (``experiments/spike``) runs the exact PI-GINOT implementation of
``BBahtiri/PI-GINOT-DogBone`` at commit ``9f67941``. Its ``models/modules``
package was copied there from GINOT (Qibang Liu), whose files carry "All
Rights Reserved" headers and whose repository has no licence. This repository
does not redistribute them (ADR 0032, ``docs/adr/upstream-sources.md``).
Anyone reproducing the spike fetches them from the pinned commit; every file
must match its SHA-256 (LF-normalised) in ``THIRD_PARTY.json``.

Usage::

    python -m experiments.spike.fetch_upstream                       # from GitHub
    python -m experiments.spike.fetch_upstream --from-checkout PATH  # local clone

The fetched files are ignored by Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parent / "upstream_pi_ginot"
MANIFEST_PATH = VENDOR_ROOT / "THIRD_PARTY.json"
RAW_URL = "https://raw.githubusercontent.com/{repository}/{commit}/{path}"


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, object]:
    document: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return document


def normalised_sha256(content: bytes) -> str:
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def missing_files(manifest: dict[str, object] | None = None) -> list[str]:
    """Paths of the manifest that are not present under the vendor root."""

    manifest = manifest or load_manifest()
    return [
        str(item["path"])
        for item in manifest["files"]  # type: ignore[attr-defined]
        if not (VENDOR_ROOT / str(item["path"])).is_file()
    ]


def _from_checkout(checkout: Path, commit: str) -> Callable[[str], bytes]:
    def read(path: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(checkout), "show", f"{commit}:{path}"],
            capture_output=True,
            check=True,
        ).stdout

    return read


def _from_github(repository: str, commit: str) -> Callable[[str], bytes]:
    def read(path: str) -> bytes:
        url = RAW_URL.format(repository=repository, commit=commit, path=path)
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            content: bytes = response.read()
        return content

    return read


def fetch(
    read: Callable[[str], bytes],
    *,
    manifest: dict[str, object] | None = None,
    root: Path = VENDOR_ROOT,
) -> list[str]:
    """Write every manifest file under ``root`` after checking its digest.

    Nothing is written unless every file matches; the content is written with
    LF line endings, the form the pins and the S1 source digest were taken in.
    """

    manifest = manifest or load_manifest()
    contents = {}
    for item in manifest["files"]:  # type: ignore[attr-defined]
        path = str(item["path"])
        content = read(str(item["upstream_path"]))
        digest = normalised_sha256(content)
        if digest != item["sha256"]:
            raise ValueError(f"{path}: SHA-256 {digest} differs from the pin")
        contents[path] = content.replace(b"\r\n", b"\n")
    for path, content in contents.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return sorted(contents)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.spike.fetch_upstream")
    parser.add_argument(
        "--from-checkout",
        type=Path,
        help="a local clone of the upstream repository that contains the commit",
    )
    arguments = parser.parse_args(argv)
    manifest = load_manifest()
    repository = str(manifest["upstream_repository"])
    commit = str(manifest["upstream_commit"])
    read = (
        _from_checkout(arguments.from_checkout, commit)
        if arguments.from_checkout
        else _from_github(repository, commit)
    )
    written = fetch(read, manifest=manifest)
    print(f"fetched {len(written)} files from {repository}@{commit[:7]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
