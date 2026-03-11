# Stage 1: Builder
FROM python:3.12-slim as builder

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files (leverage Docker cache)
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv venv .venv && \
    .venv/bin/python -m uv sync --frozen

# Stage 2: Runtime
FROM python:3.12-slim

# Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]