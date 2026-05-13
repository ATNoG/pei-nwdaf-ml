import ast


def run(tree: ast.Module) -> None:
    """Validate that the model implements the ModelInterface."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            if "ModelInterface" in bases:
                methods = {
                    n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)
                }
                if "train" not in methods:
                    raise ValueError("ModelInterface subclass missing 'train' method")
                if "predict" not in methods:
                    raise ValueError("ModelInterface subclass missing 'predict' method")
                return
    raise ValueError("No ModelInterface subclass found")
