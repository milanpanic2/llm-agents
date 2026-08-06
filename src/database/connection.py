from sqlalchemy import create_engine, pool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config.settings import settings
from src.database.models import Base

async_engine = create_async_engine(
    url=f"postgresql+asyncpg://{settings.database_url}",
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    echo=False,  # Set to True for SQL query logging
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Sync engine for init_db only (table creation)
sync_engine = create_engine(
    url=f"postgresql://{settings.database_url}",
    pool_pre_ping=True,
    poolclass=pool.NullPool
)


def init_db():
    Base.metadata.create_all(bind=sync_engine)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
