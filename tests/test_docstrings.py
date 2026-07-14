"""Enforce repository-wide Python documentation and package typing artifacts."""

from __future__ import annotations

import ast
from pathlib import Path


def test_every_python_definition_has_a_docstring() -> None:
    """Require docstrings for modules, classes, functions, and nested functions."""
    missing: list[str] = []
    paths = sorted(path for root in ("src", "tests") for path in Path(root).rglob("*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if ast.get_docstring(tree) is None:
            missing.append(f"{path}: module")
        for node in ast.walk(tree):
            if (
                isinstance(
                    node,
                    (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                )
                and ast.get_docstring(node) is None
            ):
                missing.append(f"{path}:{node.lineno} {node.name}")
    assert not missing, "Missing docstrings:\n" + "\n".join(missing)


def test_package_includes_typing_marker() -> None:
    """Ship the PEP 561 marker inside the import package."""
    assert Path("src/trimwise/py.typed").is_file()
