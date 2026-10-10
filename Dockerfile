FROM python:3.13-slim

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python output appears immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project metadata and source before installing the project package.
# uv_build needs the source tree to build the local package.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Install runtime dependencies and the local project package (no test tools).
RUN uv sync --frozen --no-dev

# Run the application
CMD ["uv", "run", "uvicorn", "ragforge.app.main:app", "--host", "0.0.0.0", "--port", "8000"]