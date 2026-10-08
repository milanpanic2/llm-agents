FROM python:3.13-slim

# glibc malloc tuning for a memory-capped container. glibc sizes itself to the HOST
# (up to 8*cores arenas) and hoards freed memory in per-arena free lists instead of
# returning it to the OS, which ratchets RSS until the pod is OOM-killed.
#   MALLOC_ARENA_MAX=2       -> cap arenas (vs host-derived default) to limit hoarding
#   MALLOC_TRIM_THRESHOLD_   -> actually release freed regions >100KB back to the kernel
ENV MALLOC_ARENA_MAX=2 \
    MALLOC_TRIM_THRESHOLD_=100000

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock main.py ./
# COPY alembic.ini ./
RUN uv sync --no-dev --no-install-project
# COPY db_migrations/ db_migrations/
COPY src/ src/

CMD ["/app/.venv/bin/python", "main.py"]
