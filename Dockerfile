FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    UPLOAD_DIR=/app/data/uploads \
    PATH="/app/.venv/bin:${PATH}"

# Pinned uv binary from the official image.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

# Run as an unprivileged user. Directories that become volumes are created
# here so Docker gives new volumes this user's ownership.
RUN useradd --create-home --uid 10001 ragforge \
    && mkdir -p /app/data/uploads /app/.cache \
    && chown -R ragforge:ragforge /app
WORKDIR /app
USER ragforge

# Dependencies first, in their own layer, so code changes do not reinstall them.
COPY --chown=ragforge:ragforge pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/ragforge/.cache/uv,uid=10001 \
    uv sync --locked --no-dev --no-install-project

# Then the project itself (uv_build needs README.md and the source tree).
COPY --chown=ragforge:ragforge README.md ./
COPY --chown=ragforge:ragforge src ./src
RUN --mount=type=cache,target=/home/ragforge/.cache/uv,uid=10001 \
    uv sync --locked --no-dev

EXPOSE 8000

# Structured request logs come from the app, so uvicorn's access log is off.
CMD ["uvicorn", "ragforge.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
