FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    build-essential \
    python3-dev \
    librdkafka-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and extract the repository files
RUN mkdir utils && cd utils \
&& git init \
&& git remote add origin https://github.com/ATNoG/pei-nwdaf-comms.git \
&& git fetch \
&& git checkout origin/main kafka/src/kmw.py \
&& mv kafka/src/kmw.py . \
&& rm -rf .git \
&& rmdir -p kafka/src

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

COPY src/ ./src/
COPY main.py .

CMD ["python3", "main.py"]
