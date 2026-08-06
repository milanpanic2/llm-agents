FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock main.py ./
# COPY alembic.ini ./
RUN uv sync --no-dev --no-install-project
# COPY db_migrations/ db_migrations/
COPY src/ src/

CMD ["uv", "run", "python", "main.py"]
