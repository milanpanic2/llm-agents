import asyncio
import uuid
from asyncio import Semaphore
from enum import StrEnum

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.WFQEngine.wfq_engine import claim_one
from src.config import settings
from src.database.connection import AsyncSessionLocal


_TRANSCRIPTION_TASKS_TABLE_NAME = "transcription_tasks"


RUN_TRANSCRIPTIONS_TASK_COMMAND = """
WITH inflight AS (
    SELECT context_id, count(*) AS n
    FROM transcription_tasks
    WHERE status = 'running'
    GROUP BY context_id
),
done AS (
    SELECT context_id, count(*) AS n
    FROM transcription_tasks
    WHERE status = 'done'
    GROUP BY context_id
)
UPDATE transcription_tasks t
SET status = 'running', started_at = now(), worker_id = $1
WHERE t.id = (
    SELECT t2.id
    FROM transcription_tasks t2
    JOIN jobs j ON j.id = t2.context_id
    LEFT JOIN inflight i ON i.context_id = t2.context_id
    LEFT JOIN done d ON d.context_id = t2.context_id
    WHERE t2.status = 'pending'
    ORDER BY
        COALESCE(i.n, 0) ASC,        -- tenant with fewest tasks in flight
        COALESCE(d.n, 0) ASC,        -- then fewest completed (progress fairness)
        j.priority DESC,             -- optional: paid tiers cut ahead
        j.created_at ASC             -- then oldest job (aging / anti-starvation)
    FOR UPDATE OF t2 SKIP LOCKED
    LIMIT 1
)
RETURNING t.id, t.context_id, t.file_path;"""


ADD_TRANSCRIPTION_TASKS="""
INSERT INTO transcription_tasks
VALUES(id=$1, context_id=$2, $status=$3, file_path=$4, file_index=$5)
"""

class TranscriptionWFQTaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


class TranscriptionWFQTask(BaseModel):
    id: str
    context_id: str
    status: TranscriptionWFQTaskStatus
    file_path: str
    file_index: str


sem = Semaphore(settings.max_concurrent_transcriptions)


async def run(handler_func):
    async with AsyncSessionLocal() as postgresql:
        await postgresql.add_listener("new_transcription_task", lambda *_: queue.put_nowait(True))
        await sem.acquire()
        row = (await postgresql.execute(text(RUN_TRANSCRIPTIONS_TASK_COMMAND), "1", )).mappings().first()
        await postgresql.commit()
        if row is None:
            return

        handler_func()


async def add_transcription_task(postgresql: AsyncSession, tt: TranscriptionWFQTask):
    return await postgresql.execute(
        ADD_TRANSCRIPTION_TASKS,str(uuid.uuid4()), tt.context_id, tt.status, tt.file_path, tt.file_index)
