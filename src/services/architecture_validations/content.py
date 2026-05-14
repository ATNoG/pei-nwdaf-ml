import ast

ALLOWED_IMPORTS: set[str] = {"torch", "numpy", "logging", "math", "typing", "abc"}
BANNED_CALLS: set[str] = {"eval", "exec", "compile", "__import__", "open"}
BLOCKED_DUNDERS = {
    "__class__",
    "__bases__",
    "__subclasses__",
    "__globals__",
    "__builtins__",
    "__import__",
}


def run(tree: ast.Module) -> None:
    for node in ast.walk(tree):

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ValueError(f"Forbidden import: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] not in ALLOWED_IMPORTS:
                raise ValueError(f"Forbidden import: {node.module}")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
                raise ValueError(f"Forbidden call: {node.func.id}")

        if isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_DUNDERS:
                raise ValueError(f"Dunder access blocked: {node.attr}")
