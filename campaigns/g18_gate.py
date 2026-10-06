"""The G18 release gate from recorded evidence (STEPS-v1.0 §S18).

Usage::

    python -m campaigns.g18_gate --bundle DOGBONE_BUNDLE --bundle NOTCHED_BUNDLE

Every check reads its evidence and fails (never passes by default) when the
evidence is missing:

- ``G18.all-gates``: G0–G17 are ``PASS`` in ``artifacts/gates/``; any
  ``NOT_RUN`` would have to be named in the release ADR (there is none);
- ``G18.claim-ceiling``: claim levels stop at ``HOLDOUT_VERIFIED``, no source
  file names a sealed-release claim, and every bundle loads with a claim at
  or below ``HOLDOUT_VERIFIED``;
- ``G18.licenses``: ``docs/adr/upstream-sources.md`` records the served
  operator's source, commit, digest and licence; the third-party GINOT
  modules are not tracked; the rebuilt wheel and sdist hold only the allowed
  files and no "All Rights Reserved" text;
- ``G18.naming``: the README disclaims faithful reproduction of GNOT and GINO
  and gives the exact served ``architecture_id``, which every bundle serves;
- ``G18.citation``: ``cffconvert`` validates ``CITATION.cff`` and its version
  is the package version;
- ``G18.install``: the TestPyPI run installed and smoke-tested the version on
  Linux, macOS and Windows (``artifacts/release/TESTPYPI-INSTALL.json``);
- ``G18.lock``: ``uv lock --check`` passes, so ``uv.lock`` is exactly what the
  tool generates;
- ``G18.changelog``: ``CHANGELOG.md`` has the version and the tag ``v<version>``
  exists at or before ``HEAD``;
- ``G18.signoff`` (⚑): the decision owner's signed confirmation that every
  sentence of ``docs/limitations.md`` is true of the code as built, naming the
  current file digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from campaigns.notched_v1.gate_io import (
    check_document,
    gate_document,
    write_gate_document,
)
from piginot.git_environment import isolated_git_environment

REPOSITORY = Path(__file__).resolve().parents[1]
GATE_PATH = Path("artifacts/gates/S18.json")
INSTALL_PATH = Path("artifacts/release/TESTPYPI-INSTALL.json")
SIGNOFF_PATH = Path("artifacts/release/LIMITATIONS-SIGNOFF.json")
LIMITATIONS_PATH = Path("docs/limitations.md")
UPSTREAM_PATH = Path("docs/adr/upstream-sources.md")
RELEASE_ADR = Path("docs/adr/0032-release-0.1.0.md")
SIGNATORY = "Betim Bahtiri"
GATE_FILES = {
    **{f"G{n}": (f"artifacts/gates/S{n}.json",) for n in range(18) if n != 11},
    "G11": ("artifacts/gates/S11.json", "artifacts/gates/S11-ENERGY.json"),
}
CEILING = ("NOT_VERIFIED", "DEVELOPMENT_VERIFIED", "HOLDOUT_VERIFIED")
NAMING_SENTENCE = "does not claim a faithful reproduction of GNOT or GINO"
SDIST_ALLOWED = (
    "src/piginot/",
    "LICENSE",
    "README.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    "PKG-INFO",
)
SERVED_SOURCE = (
    "https://github.com/BBahtiri/PI-GINOT",
    "12c9c1c7a13eaabcc2a289a886c34858d0330bad",
    "7401ad96d45d94b2bd8f8a4522878f317c09d75d8b326bebc630e85b3c0b7c7b",
)
OPERATING_SYSTEMS = ("Linux", "Darwin", "Windows")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return document


def _run(*command: str, cwd: Path = REPOSITORY) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=isolated_git_environment() if command[0] == "git" else None,
        capture_output=True,
        text=True,
    )


def _pytest(*tests: str) -> tuple[bool, str]:
    completed = _run(
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return completed.returncode == 0, lines[-1] if lines else ""


def version() -> str:
    data = tomllib.loads((REPOSITORY / "pyproject.toml").read_text("utf-8"))
    return str(data["project"]["version"])


def all_gates() -> dict[str, Any]:
    verdicts: dict[str, Any] = {}
    for gate, paths in GATE_FILES.items():
        documents = [_load(REPOSITORY / path) for path in paths]
        verdicts[gate] = [
            None if document is None else document.get("verdict")
            for document in documents
        ]
    not_run = sorted(g for g, v in verdicts.items() if "NOT_RUN" in v)
    adr = (
        (REPOSITORY / RELEASE_ADR).read_text("utf-8")
        if (REPOSITORY / RELEASE_ADR).is_file()
        else ""
    )
    unexplained = [g for g in not_run if g not in adr]
    passed = (
        all(
            all(verdict in ("PASS", "NOT_RUN") for verdict in values)
            for values in verdicts.values()
        )
        and not unexplained
    )
    measured = {
        "verdicts": verdicts,
        "not_run": not_run,
        "accepted_by_decision_owner": [
            "G14 DogBone DEVELOPMENT budget (ADR 0021)",
            "G14 notched DEVELOPMENT budget (ADR 0030)",
            "G17.non-dogbone at DEVELOPMENT_VERIFIED (ADR 0031)",
        ],
    }
    return check_document(
        "G18.all-gates",
        passed,
        measured,
        "G0–G17 all PASS, or NOT_RUN with a written reason accepted in the release ADR",
        "all",
    )


def claim_ceiling(bundles: Sequence[Path]) -> dict[str, Any]:
    from piginot.bundle import load_bundle
    from piginot.verification import ClaimLevel

    levels = [level.value for level in ClaimLevel]
    tests, summary = _pytest("tests/test_release_claim_ceiling.py")
    served = {}
    for path in bundles:
        bundle = load_bundle(path)
        served[bundle.manifest.name] = bundle.claim_level.value
    passed = (
        tuple(levels) == CEILING
        and tests
        and len(served) >= 2
        and all(level in CEILING for level in served.values())
    )
    return check_document(
        "G18.claim-ceiling",
        passed,
        {"claim_levels": levels, "bundles": served, "tests": summary},
        "no shipped bundle claims above HOLDOUT_VERIFIED; SEALED_RELEASE_VERIFIED "
        "is not emittable",
        "all",
    )


def _distribution_files() -> tuple[list[str], list[str], list[str]]:
    with tempfile.TemporaryDirectory() as directory:
        build = _run("uv", "build", "--out-dir", directory)
        if build.returncode:
            raise RuntimeError(build.stderr[-2000:])
        wheel = next(Path(directory).glob("*.whl"))
        sdist = next(Path(directory).glob("*.tar.gz"))
        with zipfile.ZipFile(wheel) as archive:
            wheel_names = archive.namelist()
            texts = [archive.read(name) for name in wheel_names]
        with tarfile.open(sdist) as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            sdist_names = [m.name.split("/", 1)[1] for m in members]
            for member in members:
                handle = archive.extractfile(member)
                texts.append(handle.read() if handle else b"")
    flagged = sorted(
        {
            name
            for name, text in zip([*wheel_names, *sdist_names], texts, strict=True)
            if b"All Rights Reserved" in text
        }
    )
    return wheel_names, sdist_names, flagged


def licenses() -> dict[str, Any]:
    record = (
        (REPOSITORY / UPSTREAM_PATH).read_text("utf-8")
        if (REPOSITORY / UPSTREAM_PATH).is_file()
        else ""
    )
    tracked = _run(
        "git", "ls-files", "--", "experiments/spike/upstream_pi_ginot/models/modules"
    ).stdout.split()
    wheel, sdist, flagged = _distribution_files()
    wheel_outside = sorted(
        {name.split("/")[0] for name in wheel}
        - {"piginot", f"piginot-{version()}.dist-info"}
    )
    sdist_outside = sorted(name for name in sdist if not name.startswith(SDIST_ALLOWED))
    tests, summary = _pytest("tests/test_upstream_sources.py")
    measured = {
        "record": UPSTREAM_PATH.as_posix(),
        "served_source_recorded": all(token in record for token in SERVED_SOURCE)
        and "MIT" in record,
        "gnot_gino_pointnet_addressed": all(
            name in record for name in ("GNOT", "GINO", "PointNet++")
        ),
        "third_party_modules_tracked": tracked,
        "manifest_present": (
            REPOSITORY / "experiments/spike/upstream_pi_ginot/THIRD_PARTY.json"
        ).is_file(),
        "wheel_files": len(wheel),
        "wheel_outside_package": wheel_outside,
        "sdist_files": len(sdist),
        "sdist_outside_allowed": sdist_outside,
        "all_rights_reserved_in_distributions": flagged,
        "tests": summary,
    }
    passed = (
        measured["served_source_recorded"] is True
        and measured["gnot_gino_pointnet_addressed"] is True
        and not tracked
        and measured["manifest_present"] is True
        and not wheel_outside
        and not sdist_outside
        and not flagged
        and tests
    )
    return check_document(
        "G18.licenses",
        passed,
        measured,
        "every adapted component recorded with source, commit, digest and licence "
        "compatibility in docs/adr/upstream-sources.md",
        "all",
    )


def naming(bundles: Sequence[Path]) -> dict[str, Any]:
    from piginot.bundle import load_bundle
    from piginot.bundle.serving import ARCHITECTURES
    from piginot.operators import ENERGY_ARCHITECTURE_ID

    readme = (REPOSITORY / "README.md").read_text("utf-8")
    served = {
        load_bundle(path).manifest.name: load_bundle(path).model.architecture_id
        for path in bundles
    }
    measured = {
        "disclaimer_present": NAMING_SENTENCE in " ".join(readme.split()),
        "architecture_id_in_readme": f"architecture_id: {ENERGY_ARCHITECTURE_ID}"
        in readme,
        "architecture_id": ENERGY_ARCHITECTURE_ID,
        "registered_architectures": sorted(ARCHITECTURES),
        "bundles": served,
    }
    passed = (
        measured["disclaimer_present"] is True
        and measured["architecture_id_in_readme"] is True
        and sorted(ARCHITECTURES) == [ENERGY_ARCHITECTURE_ID]
        and bool(served)
        and all(value == ENERGY_ARCHITECTURE_ID for value in served.values())
    )
    return check_document(
        "G18.naming",
        passed,
        measured,
        "README states the name does not claim faithful reproduction of GNOT or "
        "GINO and gives the exact architecture_id",
        "all",
    )


def citation() -> dict[str, Any]:
    command = shutil.which("cffconvert")
    completed = (
        _run(command, "--validate", "-i", "CITATION.cff")
        if command
        else _run(
            "uvx",
            "--from",
            "cffconvert==2.0.0",
            "cffconvert",
            "--validate",
            "-i",
            "CITATION.cff",
        )
    )
    text = (REPOSITORY / "CITATION.cff").read_text("utf-8")
    output = (completed.stdout + completed.stderr).strip().splitlines()
    measured = {
        "validator": "cffconvert 2.0.0",
        "result": output[-1] if output else "",
        "version_matches": f"version: {version()}" in text,
    }
    return check_document(
        "G18.citation",
        completed.returncode == 0 and measured["version_matches"] is True,
        measured,
        "CITATION.cff validates",
        "all",
    )


def install() -> dict[str, Any]:
    evidence = _load(REPOSITORY / INSTALL_PATH)
    if evidence is None:
        return {
            "id": "G18.install",
            "status": "NOT_RUN",
            "reason": f"no TestPyPI install evidence at {INSTALL_PATH.as_posix()}; "
            "run .github/workflows/release.yml with TEST_PYPI_API_TOKEN",
            "threshold": "install from TestPyPI succeeds on Linux, macOS and Windows",
            "comparison": "all",
        }
    results = evidence.get("results", [])
    by_os = {
        result.get("os"): result
        for result in results
        if result.get("passed") is True
        and result.get("installed_version") == version()
        and all(result.get("checks", {}).values())
    }
    measured = {
        "run_url": evidence.get("run_url"),
        "index": evidence.get("index"),
        "version": evidence.get("version"),
        "passed_on": sorted(by_os),
    }
    passed = (
        evidence.get("version") == version()
        and str(evidence.get("index", "")).startswith("https://test.pypi.org/")
        and bool(evidence.get("run_url"))
        and all(name in by_os for name in OPERATING_SYSTEMS)
    )
    return check_document(
        "G18.install",
        passed,
        measured,
        "install from TestPyPI succeeds on Linux, macOS and Windows",
        "all",
        evidence=INSTALL_PATH.as_posix(),
    )


def lock() -> dict[str, Any]:
    completed = _run("uv", "lock", "--check")
    uv = _run("uv", "--version").stdout.strip()
    header = (REPOSITORY / "uv.lock").read_text("utf-8").splitlines()[:2]
    measured = {
        "tool": uv,
        "uv_lock_check": completed.returncode == 0,
        "header": header,
        "committed": bool(_run("git", "ls-files", "uv.lock").stdout.strip()),
    }
    return check_document(
        "G18.lock",
        completed.returncode == 0 and measured["committed"] is True,
        measured,
        "lockfile committed and tool-generated (uv lock --check reproduces it)",
        "all",
    )


def changelog() -> dict[str, Any]:
    tag = f"v{version()}"
    text = (
        (REPOSITORY / "CHANGELOG.md").read_text("utf-8")
        if (REPOSITORY / "CHANGELOG.md").is_file()
        else ""
    )
    tagged = _run(
        "git", "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"
    )
    ancestor = (
        tagged.returncode == 0
        and _run(
            "git", "merge-base", "--is-ancestor", tagged.stdout.strip(), "HEAD"
        ).returncode
        == 0
    )
    measured = {
        "version": version(),
        "changelog_entry": f"## [{version()}]" in text,
        "tag": tag,
        "tag_commit": tagged.stdout.strip() or None,
        "tag_at_or_before_head": ancestor,
    }
    return check_document(
        "G18.changelog",
        measured["changelog_entry"] is True and ancestor,
        measured,
        "version tagged, changelog written",
        "all",
    )


def signoff() -> dict[str, Any]:
    record = _load(REPOSITORY / SIGNOFF_PATH)
    digest = _sha256(REPOSITORY / LIMITATIONS_PATH)
    if record is None:
        return {
            "id": "G18.signoff",
            "status": "NOT_RUN",
            "reason": "the decision owner has not yet signed off docs/limitations.md",
            "threshold": "⚑ the decision owner confirms every sentence of the shipped "
            "limitations.md is true of the code as built",
            "comparison": "all",
        }
    measured = {
        "signed_off_by": record.get("signed_off_by"),
        "signed_off_utc": record.get("signed_off_utc"),
        "limitations_sha256_signed": record.get("limitations_sha256"),
        "limitations_sha256_current": digest,
    }
    return check_document(
        "G18.signoff",
        record.get("signed_off_by") == SIGNATORY
        and record.get("limitations_sha256") == digest
        and record.get("confirmed") is True,
        measured,
        "⚑ the decision owner confirms every sentence of the shipped limitations.md "
        "is true of the code as built",
        "all",
        evidence=SIGNOFF_PATH.as_posix(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.g18_gate")
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    arguments = parser.parse_args(argv)
    checks = [
        all_gates(),
        claim_ceiling(arguments.bundle),
        licenses(),
        naming(arguments.bundle),
        citation(),
        install(),
        lock(),
        changelog(),
        signoff(),
    ]
    document = gate_document(
        "G18",
        "S18",
        REPOSITORY,
        checks,
        version=version(),
        release_adr=RELEASE_ADR.as_posix(),
    )
    document.pop("campaign_id", None)
    write_gate_document(REPOSITORY / GATE_PATH, document)
    print(document["verdict"], document["verdict_reason"])
    for check in checks:
        print(f"  {check['id']}: {check['status']}")
    return 0 if document["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
