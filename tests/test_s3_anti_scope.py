import ast
from pathlib import Path

import pytest

from piginot import cli
from piginot.contracts import CampaignPlan, CanonicalMesh

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = REPOSITORY_ROOT / "src" / "piginot"

FORBIDDEN_S4_PLUS_MODULES = {
    "bundle",
    "datasets",
    "doe",
    "fem",
    "geometry",
    "objectives",
    "operator",
    "physics",
    "training",
    "verification",
}

FORBIDDEN_S2_OPERATIONS = {
    "assign_role",
    "check_membership",
    "construct_mesh",
    "evaluate",
    "evaluate_expression",
    "generate_mesh",
    "morph",
    "remesh",
    "repair_mesh",
    "sample",
}

FORBIDDEN_S3_IMPORTS = {"fcntl", "filelock", "msvcrt", "threading"}
FORBIDDEN_S3_NAME_FRAGMENTS = {
    "arbitration",
    "compatibility_policy",
    "cross_profile",
    "lock_manager",
    "narrative_cache",
    "quarantine",
}


def test_no_s4_or_later_module_exists() -> None:
    discovered = {
        path.stem for path in SOURCE_PACKAGE.rglob("*.py") if path.name != "__init__.py"
    }

    assert discovered.isdisjoint(FORBIDDEN_S4_PLUS_MODULES)


def test_contract_modules_define_no_evaluation_or_workflow_operation() -> None:
    contracts_root = SOURCE_PACKAGE / "contracts"
    definitions: set[str] = set()
    for path in contracts_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        )

    assert definitions.isdisjoint(FORBIDDEN_S2_OPERATIONS)


def test_campaign_plan_contains_no_doe_or_evidence_role_contract() -> None:
    campaign_fields = {
        field.name for field in CampaignPlan.__dataclass_fields__.values()
    }

    assert "roles" not in campaign_fields
    assert "evidence_roles" not in campaign_fields
    assert "samples" not in campaign_fields


@pytest.mark.parametrize(
    "algorithm_name",
    sorted(FORBIDDEN_S2_OPERATIONS),
)
def test_canonical_mesh_exposes_no_geometry_algorithm(algorithm_name: str) -> None:
    assert not hasattr(CanonicalMesh, algorithm_name)


def test_cli_rejects_s4_plus_subcommands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main([]) == 0

    for command in ("simulate", "train", "bundle"):
        with pytest.raises(SystemExit) as exit_info:
            cli.main([command])
        assert exit_info.value.code == 2

    assert "invalid choice" in capsys.readouterr().err


def test_store_contains_no_s3_anti_scope_subsystem() -> None:
    store_root = SOURCE_PACKAGE / "store"
    imports: set[str] = set()
    definitions: set[str] = set()
    for path in store_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(
                    alias.name.split(".", maxsplit=1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", maxsplit=1)[0])
            elif isinstance(
                node,
                ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
            ):
                definitions.add(node.name.lower())

    assert imports.isdisjoint(FORBIDDEN_S3_IMPORTS)
    assert all(
        fragment not in definition
        for definition in definitions
        for fragment in FORBIDDEN_S3_NAME_FRAGMENTS
    )
