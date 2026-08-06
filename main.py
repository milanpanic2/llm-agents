import uvicorn
from sqlalchemy import create_engine, pool, text

from src.config.settings import settings


def ensure_database():
    engine = create_engine(
        f"postgresql://{settings.postgresql_user}:{settings.postgresql_password}@{settings.postgresql_host}:{settings.postgresql_port}/postgres",
        poolclass=pool.NullPool,
    )
    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db"), {"db": settings.postgresql_db_name}
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{settings.postgresql_db_name}"'))
    engine.dispose()


def main():
    ensure_database()
    uvicorn.run(app="src.app:app", host=settings.app_host, port=settings.app_port, reload=settings.debug)


if __name__ == "__main__":
    main()
