import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
EXPERIMENTS_PREFIX = "experiments"
IGNORED_DIRECTORY_NAMES = {
    ".formamech-runs",
    ".git",
    ".mypy_cache",
    ".piginot-private",
    ".piginot-runs",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "site",
}


@dataclass(frozen=True)
class Module:
    name: str
    package: str
    path: Path
    is_source: bool


def _module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _discover_modules() -> dict[str, Module]:
    modules: dict[str, Module] = {}
    for path in REPOSITORY_ROOT.rglob("*.py"):
        if any(part in IGNORED_DIRECTORY_NAMES for part in path.parts):
            continue

        is_source = path.is_relative_to(SOURCE_ROOT)
        root = SOURCE_ROOT if is_source else REPOSITORY_ROOT
        name = _module_name(path, root)
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        modules[name] = Module(name, package, path, is_source)
    return modules


def _resolve_from_import(module: Module, imported: ast.ImportFrom) -> str:
    if imported.level == 0:
        return imported.module or ""

    package_parts = module.package.split(".") if module.package else []
    parents_to_remove = imported.level - 1
    if parents_to_remove > len(package_parts):
        return ""
    if parents_to_remove:
        package_parts = package_parts[:-parents_to_remove]
    if imported.module:
        package_parts.extend(imported.module.split("."))
    return ".".join(package_parts)


def _importlib_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    module_aliases: set[str] = set()
    function_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    module_aliases.add(alias.asname or alias.name)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module == "importlib"
        ):
            for alias in node.names:
                if alias.name == "import_module":
                    function_aliases.add(alias.asname or alias.name)
    return module_aliases, function_aliases


def _imports(module: Module) -> Iterator[str]:
    tree = ast.parse(module.path.read_text(encoding="utf-8"), filename=str(module.path))
    module_aliases, function_aliases = _importlib_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_import(module, node)
            if base:
                yield base
            for alias in node.names:
                if alias.name != "*":
                    yield ".".join(part for part in (base, alias.name) if part)
        elif isinstance(node, ast.Call) and node.args:
            is_builtin_import = (
                isinstance(node.func, ast.Name) and node.func.id == "__import__"
            )
            is_importlib_call = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in module_aliases
                and node.func.attr == "import_module"
            )
            is_importlib_function = (
                isinstance(node.func, ast.Name) and node.func.id in function_aliases
            )
            first_argument = node.args[0]
            if (
                (is_builtin_import or is_importlib_call or is_importlib_function)
                and isinstance(first_argument, ast.Constant)
                and isinstance(first_argument.value, str)
            ):
                yield first_argument.value


def _local_targets(imported: str, module_names: set[str]) -> set[str]:
    parts = imported.split(".")
    return {
        ".".join(parts[:index])
        for index in range(1, len(parts) + 1)
        if ".".join(parts[:index]) in module_names
    }


def test_module_discovery_ignores_generated_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "src" / "piginot" / "__init__.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("", encoding="utf-8")

    for directory_name in (".venv", "build", "dist", "site"):
        generated_path = tmp_path / directory_name / "package" / "module.py"
        generated_path.parent.mkdir(parents=True)
        generated_path.write_text("", encoding="utf-8")

    monkeypatch.setitem(_discover_modules.__globals__, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setitem(
        _discover_modules.__globals__,
        "SOURCE_ROOT",
        tmp_path / "src",
    )

    discovered_paths = {module.path for module in _discover_modules().values()}
    assert discovered_paths == {source_path}


@pytest.mark.parametrize(
    "source",
    [
        "from importlib import import_module as load\nload('experiments.spike')\n",
        "import importlib as imports\nimports.import_module('experiments.spike')\n",
    ],
)
def test_dynamic_import_aliases_are_discovered(
    source: str,
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "module.py"
    module_path.write_text(source, encoding="utf-8")
    module = Module(
        name="piginot.example",
        package="piginot",
        path=module_path,
        is_source=True,
    )

    assert "experiments.spike" in set(_imports(module))


def test_source_has_no_import_path_into_experiments() -> None:
    modules = _discover_modules()
    source_modules = sorted(
        module.name for module in modules.values() if module.is_source
    )
    assert source_modules, "src/ must contain at least one Python module"

    module_names = set(modules)
    graph = {
        name: {
            target
            for imported in _imports(module)
            for target in _local_targets(imported, module_names) | {imported}
        }
        for name, module in modules.items()
    }

    violations: list[str] = []
    for source in source_modules:
        pending = [(source, (source,))]
        visited: set[str] = set()
        while pending:
            current, path = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for target in graph.get(current, set()):
                import_path = (*path, target)
                if target == EXPERIMENTS_PREFIX or target.startswith(
                    f"{EXPERIMENTS_PREFIX}."
                ):
                    violations.append(" -> ".join(import_path))
                elif target in graph:
                    pending.append((target, import_path))

    assert not violations, "Source import paths reach experiments/:\n" + "\n".join(
        sorted(set(violations))
    )
