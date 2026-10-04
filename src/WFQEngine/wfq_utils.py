import json

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.WFQEngine.wfq_engine import TaskContextProgress, WFQTaskData
from src.agents.transcriptions_service import FailureReasons


async def add_tasks_bulk(session: AsyncSession, table_name: str, wfq_tasks: list[WFQTaskData]):
    # payload is a dict in memory; the asyncpg jsonb codec encodes a JSON *string*, so dump it here.
    sql = f"""INSERT INTO {table_name} (context_id, status, payload)
              VALUES (:context_id, 'pending', :payload)"""
    await session.execute(text(sql),
                          [{"context_id": task.context_id, "payload": json.dumps(task.payload)} for task in wfq_tasks])
    await session.commit()


async def get_progress(session: AsyncSession, table_name: str, context_id: str) -> TaskContextProgress:
    count_sql = f"""SELECT count(*)                                AS total,
                    count(*) FILTER (WHERE status = 'done')        AS done,
                    count(*) FILTER (WHERE status = 'failed')      AS failed
                    FROM {table_name}
                    WHERE context_id = :context_id"""

    counts = (await session.execute(text(count_sql), {"context_id": context_id})).mappings().one()
    await session.commit()
    return TaskContextProgress(**counts)


async def get_failure_reasons(session: AsyncSession, table_name: str, context_id: str) -> list[FailureReasons]:
    sql = f"""SELECT id, failure_reason
              FROM {table_name}
              WHERE context_id = :context_id AND status = 'failed'"""
    rows = (await session.execute(text(sql), {"context_id": context_id})).mappings().all()
    await session.commit()
    return [FailureReasons(**row) for row in rows]

