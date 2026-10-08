import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.WFQEngine.wfq_engine import FailureReasons, TaskContextProgress, WFQTaskData


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


async def check_if_context_finished(progress: TaskContextProgress) -> bool:
    if progress.done + progress.failed == progress.total:
        return True
    else:
        return False


async def get_failure_reasons(session: AsyncSession, table_name: str, context_id: str) -> list[FailureReasons]:
    sql = f"""SELECT id, failure_reason
              FROM {table_name}
              WHERE context_id = :context_id AND status = 'failed'
              ORDER BY finished_at DESC"""
    rows = (await session.execute(text(sql), {"context_id": context_id})).mappings().all()
    await session.commit()
    return [FailureReasons(**row) for row in rows]


async def retry_all_failed_tasks(session: AsyncSession, table_name: str, context_id: str):
    sql = f"""UPDATE {table_name}
              SET status = 'pending', failure_reason = '', started_at = NULL, finished_at = NULL
              WHERE context_id = :context_id AND status = 'failed'"""
    await session.execute(text(sql), {"context_id": context_id})
    await session.commit()
