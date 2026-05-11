_MAX_SIZE = 100 * 1024  # 100 KB


def run(file: bytes):
    if len(file) > _MAX_SIZE:
        raise ValueError(f"File too large: max {_MAX_SIZE} bytes")
