"""Guard: no log call may include the user's question.

User questions can contain personal data, and logs are kept longer and
shared more widely than responses. This scans every logging call in src/ for
an argument that refers to a variable or attribute named ``query``.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "log"}


def references_query(node: ast.AST) -> bool:
    """True if the expression uses `query` other than inside len(...)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len":
        return False
    if isinstance(node, ast.Name) and node.id == "query":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "query":
        return True
    return any(references_query(child) for child in ast.iter_child_nodes(node))


def test_no_logging_call_passes_the_query() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr not in LOG_METHODS:
                continue
            owner = node.func.value
            if not (isinstance(owner, ast.Name) and "log" in owner.id.lower()):
                continue
            arguments = list(node.args[1:]) + [kw.value for kw in node.keywords]
            if any(references_query(arg) for arg in arguments):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, f"log calls that include the query: {offenders}"
