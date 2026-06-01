FROM python:3.12-slim-bullseye

# Install uv (pin the version for reproducible builds).
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/uv

# Copy the application into the container.
COPY . /app

# Install the application dependencies (exclude dev tooling like ruff/ty from the image).
WORKDIR /app
RUN uv sync --locked --no-cache --no-dev

# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "app/main.py", "--port", "80"]