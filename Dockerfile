# ─── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for curl-cffi (needs libcurl + compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libcurl4-openssl-dev \
        libssl-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only what's needed for dependency resolution first (layer cache)
COPY pyproject.toml .
COPY deepcode_cli/__init__.py deepcode_cli/__init__.py

# Install all dependencies into a prefix we can copy to the final stage
RUN pip install --upgrade pip \
 && pip install --prefix=/install \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.29.0" \
        "pydantic>=2.0.0" \
        "rich" \
        "curl-cffi>=0.8.1" \
        "wasmtime" \
        "numpy"

# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="DeepCode Server" \
      org.opencontainers.image.description="OpenAI-compatible API backed by DeepSeek (deepseek4free)" \
      org.opencontainers.image.source="https://github.com/your-org/deepcode"

# Runtime system libraries needed by curl-cffi
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy the full project
COPY . .

# Install the deepcode package itself (editable-like, no extra deps)
RUN pip install --no-deps -e .

# ─── Runtime config ──────────────────────────────────────────────────────────

# Port the server listens on (override via Coolify env PORT)
ENV PORT=8000
ENV HOST=0.0.0.0

# Optional: pre-set the DeepSeek token via Coolify environment variable.
# If not set here, clients must send:  Authorization: Bearer <TOKEN>
# and it will be saved automatically on first use.
ENV DEEPSEEK_AUTH_TOKEN=""

# Config + session persistence — map this to a volume in Coolify
# so the saved token survives container restarts.
ENV HOME=/app/data
RUN mkdir -p /app/data

EXPOSE ${PORT}

# Healthcheck for Coolify / container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" \
    || exit 1

ENTRYPOINT ["sh", "-c", "exec deepcode-server --host $HOST --port $PORT"]
