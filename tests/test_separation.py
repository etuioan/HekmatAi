"""Architecture test: the dependency direction of the three parts.

agent imports no benchmark. own_benchmark and arc_reference import the agent
but never each other.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "hekmatai"


def _imports(package: str) -> set[str]:
    found: set[str] = set()
    for path in (SRC / package).rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                level = node.level
                mod = node.module or ""
                if level == 0:
                    found.add(mod)
                elif level >= 2:  # relative import leaving the package
                    found.add("hekmatai." + mod)
    return found


def _mentions(imports: set[str], name: str) -> bool:
    return any(i == f"hekmatai.{name}" or i.startswith(f"hekmatai.{name}.") for i in imports)


def test_agent_knows_no_benchmark():
    imports = _imports("agent")
    assert not _mentions(imports, "own_benchmark")
    assert not _mentions(imports, "arc_reference")


def test_own_benchmark_does_not_import_arc_reference():
    assert not _mentions(_imports("own_benchmark"), "arc_reference")


def test_arc_reference_does_not_import_own_benchmark():
    assert not _mentions(_imports("arc_reference"), "own_benchmark")
