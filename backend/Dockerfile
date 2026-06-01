FROM python:3.12-slim-bullseye

# Install git (needed to build the diffusers git dependency from source).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pin the version for reproducible builds).
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/uv

# Copy the application into the container.
COPY . /app

# Install the application dependencies (exclude dev tooling like ruff/ty from the image).
WORKDIR /app
RUN uv sync --locked --no-cache --no-dev

# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "app/main.py", "--port", "80"]