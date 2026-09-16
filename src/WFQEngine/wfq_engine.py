# Weighted Fair Queueing
# A way to queue long tasks, on limited gpu resources like agent calls.
# Spread tasks on multiple workers, with criteria for which gets executed next, for fair execution.
# - Supports multiple replicas
# - Needs postgresql or postgresql compatible database
import asyncio
from asyncio import Semaphore

# Claim the single fairest pending task and mark it running, atomically.
from sqlalchemy import text

from src.config import settings
from src.database.connection import AsyncSessionLocal



async def claim_one(run_task_command):
    async with AsyncSessionLocal as postgresql:
        row = (await postgresql.execute(text(run_task_command), "1",)).mappings().first()
        await postgresql.commit()
        return row



