import uvicorn
from sqlalchemy import create_engine, pool, text

from src.config.settings import settings


def ensure_databases(db_names: list[str]):
    engine = create_engine(
        f"postgresql://{settings.postgresql_user}:{settings.postgresql_password}@{settings.postgresql_host}:{settings.postgresql_port}/postgres",
        poolclass=pool.NullPool,
    )
    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        for db_name in db_names:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db"), {"db": db_name}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    engine.dispose()


def main():
    ensure_databases(settings.postgresql_db_name)
    uvicorn.run(app="src.app:app", host=settings.app_host, port=settings.app_port, reload=settings.debug)


if __name__ == "__main__":
    main()
