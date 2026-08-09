from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
FORBIDDEN_TERMS = (b"dogbone", b"gauge", b"fillet", b"grip")


def test_source_contains_no_fixture_specific_names() -> None:
    assert SOURCE_ROOT.is_dir(), "src/ must exist"

    violations: list[str] = []
    for path in sorted(item for item in SOURCE_ROOT.rglob("*") if item.is_file()):
        content = path.read_bytes().lower()
        matches = [term.decode("ascii") for term in FORBIDDEN_TERMS if term in content]
        if matches:
            relative_path = path.relative_to(REPOSITORY_ROOT)
            violations.append(f"{relative_path}: {', '.join(matches)}")

    assert not violations, "Fixture-specific names found in src/:\n" + "\n".join(
        violations
    )
