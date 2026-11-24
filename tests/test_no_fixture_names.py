from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
FORBIDDEN_TERMS = (b"dogbone", b"gauge", b"fillet", b"grip")
# Compiled third-party frontend bundles (Three.js / Plotly / Streamlit) contain
# unrelated upstream identifiers such as `filmGauge`, `getGripSpace`, `gripBgColor`.
BUILT_FRONTEND_ASSET_DIRS = frozenset(
    {
        SOURCE_ROOT / "formamech" / "_boundary_picker_assets",
        SOURCE_ROOT / "formamech" / "_viewer_3d_assets",
    }
)


def test_source_contains_no_fixture_specific_names() -> None:
    assert SOURCE_ROOT.is_dir(), "src/ must exist"

    violations: list[str] = []
    for path in sorted(item for item in SOURCE_ROOT.rglob("*") if item.is_file()):
        if any(parent in BUILT_FRONTEND_ASSET_DIRS for parent in path.parents):
            continue
        content = path.read_bytes().lower()
        matches = [term.decode("ascii") for term in FORBIDDEN_TERMS if term in content]
        if matches:
            relative_path = path.relative_to(REPOSITORY_ROOT)
            violations.append(f"{relative_path}: {', '.join(matches)}")

    assert not violations, "Fixture-specific names found in src/:\n" + "\n".join(
        violations
    )
