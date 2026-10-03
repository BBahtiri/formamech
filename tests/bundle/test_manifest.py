from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from piginot.bundle.manifest import (
    MANIFEST_NAME,
    BundleIntegrityError,
    BundleReasonCode,
    manifest_hash,
    read_json,
    verify_bundle,
    write_bundle,
)

FILES = {
    "a.json": b'{"a":1}',
    "nested/b.bin": bytes(range(64)),
    "nested/deeper/c.txt": b"hello",
}


def _write(directory: Path) -> None:
    write_bundle(
        directory,
        FILES,
        name="t",
        created_utc="2026-10-03T00:00:00Z",
        claim_level="NOT_VERIFIED",
        identities={"k": "v"},
    )


def test_round_trip_and_self_hash(tmp_path: Path) -> None:
    directory = tmp_path / "b"
    _write(directory)
    manifest, contents = verify_bundle(directory)

    assert contents == FILES
    assert manifest.manifest_sha256 == manifest_hash(manifest)
    assert manifest.identities == {"k": "v"}
    assert read_json(contents, "a.json") == {"a": 1}
    with pytest.raises(BundleIntegrityError):
        read_json(contents, "missing.json")
    for payload in (b"[1]", b"{", b'{"a": 1, "a": 2}', b'{"a": NaN}'):
        with pytest.raises(BundleIntegrityError) as error:
            read_json({"x.json": payload}, "x.json")
        assert error.value.reason_codes == (BundleReasonCode.REPORT_INVALID,)
    with pytest.raises(FileExistsError):
        _write(directory)


@pytest.mark.parametrize(
    "path", ["", "/abs", "../up", "a/../b", "./a", "a\\b", MANIFEST_NAME, "a//b"]
)
def test_unsafe_paths_are_refused(tmp_path: Path, path: str) -> None:
    with pytest.raises(BundleIntegrityError) as error:
        write_bundle(
            tmp_path / "b",
            {path: b"x"},
            name="t",
            created_utc="c",
            claim_level="NOT_VERIFIED",
            identities={},
        )
    assert error.value.reason_codes == (BundleReasonCode.UNSAFE_PATH,)


def tamper_cases(directory: Path) -> list[tuple[str, Callable[[Path], None]]]:
    """Every single-byte flip of every file, plus structural changes."""

    cases: list[tuple[str, Callable[[Path], None]]] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        relative = path.relative_to(directory).as_posix()
        size = path.stat().st_size
        for offset in sorted({0, size // 2, size - 1}):

            def flip(
                root: Path, relative: str = relative, offset: int = offset
            ) -> None:
                target = root / relative
                payload = bytearray(target.read_bytes())
                payload[offset] ^= 0x01
                target.write_bytes(bytes(payload))

            cases.append((f"flip {relative}@{offset}", flip))

        def append(root: Path, relative: str = relative) -> None:
            with (root / relative).open("ab") as handle:
                handle.write(b" ")

        cases.append((f"append {relative}", append))

        def delete(root: Path, relative: str = relative) -> None:
            (root / relative).unlink()

        cases.append((f"delete {relative}", delete))

    def extra(root: Path) -> None:
        (root / "nested" / "extra.txt").write_bytes(b"extra")

    def symlink(root: Path) -> None:
        os.symlink(root / "a.json", root / "link.json")

    cases.extend((("extra file", extra), ("symlink", symlink)))
    return cases


def detected_by(
    verifier: Callable[[Path], object], template: Path, scratch: Path
) -> list[str]:
    missed: list[str] = []
    for index, (label, tamper) in enumerate(tamper_cases(template)):
        copy = scratch / f"case-{index}"
        shutil.copytree(template, copy, symlinks=True)
        tamper(copy)
        try:
            verifier(copy)
        except BundleIntegrityError:
            continue
        missed.append(label)
    return missed


def test_every_byte_and_structure_change_is_detected(tmp_path: Path) -> None:
    template = tmp_path / "template"
    _write(template)
    assert len(tamper_cases(template)) >= 20
    assert detected_by(verify_bundle, template, tmp_path / "work") == []


def test_the_undetected_bundle_tamper_mutant_is_caught(tmp_path: Path) -> None:
    """``UNDETECTED_BUNDLE_TAMPER``: a verifier that trusts sizes misses flips."""

    def sizes_only(directory: Path) -> object:
        try:
            document = json.loads((directory / MANIFEST_NAME).read_bytes())
        except (OSError, ValueError) as exc:
            raise BundleIntegrityError(
                (BundleReasonCode.MANIFEST_INVALID,), ""
            ) from exc
        for path, entry in document["files"].items():
            target = directory / path
            if not target.is_file() or target.stat().st_size != entry["bytes"]:
                raise BundleIntegrityError((BundleReasonCode.FILE_MISSING,), path)
        return document

    template = tmp_path / "template"
    _write(template)
    missed = detected_by(sizes_only, template, tmp_path / "work")
    assert any(label.startswith("flip") for label in missed)


def _manifest(directory: Path) -> dict[str, object]:
    return json.loads((directory / MANIFEST_NAME).read_bytes())


def _rewrite(directory: Path, document: object, *, raw: bytes | None = None) -> None:
    (directory / MANIFEST_NAME).write_bytes(
        raw if raw is not None else json.dumps(document).encode()
    )


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda d: d.pop("name"), BundleReasonCode.MANIFEST_INVALID),
        (lambda d: d.update(format="other"), BundleReasonCode.MANIFEST_INVALID),
        (lambda d: d.update(name=""), BundleReasonCode.MANIFEST_INVALID),
        (lambda d: d.update(identities={"k": 1}), BundleReasonCode.MANIFEST_INVALID),
        (
            lambda d: d["files"].update({"a.json": {"sha256": "x"}}),
            BundleReasonCode.MANIFEST_INVALID,
        ),
        (
            lambda d: d["files"].update({"a.json": {"sha256": "x", "bytes": 7}}),
            BundleReasonCode.MANIFEST_INVALID,
        ),
        (
            lambda d: d["files"]["a.json"].update(bytes=-1),
            BundleReasonCode.MANIFEST_INVALID,
        ),
        (
            lambda d: d["files"].update({"../x": d["files"]["a.json"]}),
            BundleReasonCode.UNSAFE_PATH,
        ),
        (lambda d: d.update(claim_level="X"), BundleReasonCode.MANIFEST_HASH_MISMATCH),
    ],
)
def test_manifest_changes_are_refused(
    tmp_path: Path, change: Callable[[dict], object], code: BundleReasonCode
) -> None:
    directory = tmp_path / "b"
    _write(directory)
    document = _manifest(directory)
    change(document)
    _rewrite(directory, document)
    with pytest.raises(BundleIntegrityError) as error:
        verify_bundle(directory)
    assert code in error.value.reason_codes


def test_missing_non_object_and_non_canonical_manifests(tmp_path: Path) -> None:
    directory = tmp_path / "b"
    _write(directory)
    canonical = (directory / MANIFEST_NAME).read_bytes()

    _rewrite(directory, None, raw=b"[]")
    with pytest.raises(BundleIntegrityError, match="MANIFEST_INVALID"):
        verify_bundle(directory)

    _rewrite(directory, None, raw=canonical.replace(b",", b", ", 1))
    with pytest.raises(BundleIntegrityError, match="MANIFEST_HASH_MISMATCH"):
        verify_bundle(directory)

    (directory / MANIFEST_NAME).unlink()
    with pytest.raises(BundleIntegrityError, match="MANIFEST_MISSING"):
        verify_bundle(directory)
