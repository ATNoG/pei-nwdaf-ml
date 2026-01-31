FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .
COPY .python-version .

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/
COPY main.py .

CMD ["uv", "run", "python", "main.py"]
