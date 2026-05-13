import re


def run(architecture_id: str):
    if not re.match(r"^[a-zA-Z0-9_-]+$", architecture_id):
        raise ValueError(
            "Invalid architecture name: only alphanumeric, underscore, hyphen allowed"
        )
