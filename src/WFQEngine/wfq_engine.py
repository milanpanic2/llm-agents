import asyncio
import logging
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from sqlalchemy import RowMapping, text

from src.database.connection import AsyncSessionLocal

logger = logging.getLogger(__name__)


class _TaskLog(logging.LoggerAdapter):
    """Bakes task_id + context_id into every message, so per-task lines are grep-able
    without hand-writing the ids each time."""

    def process(self, msg, kwargs):
        extra = self.extra or {}
        return f"[task={extra.get('task_id')} ctx={extra.get('context_id')}] {msg}", kwargs


class WFQTaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class WFQTaskData(BaseModel):
    id: str = "" # DB assigns on insert
    context_id: str
    payload: dict[str, Any]
    failure_reason: str = ""


# TODO - make with DeclarativeBase from sql-alchemy
CREATE_WFQ_TABLE = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id             TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    context_id     TEXT NOT NULL,
    status         VARCHAR(20) NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'running', 'done', 'failed')),
    payload        JSONB NOT NULL,
    failure_reason TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at     TIMESTAMPTZ,
    finished_at    TIMESTAMPTZ,
    priority       VARCHAR(50) NOT NULL DEFAULT 'standard'
);"""


CREATE_WFQ_CONTEXT_INDEX = "CREATE INDEX IF NOT EXISTS {table_name}_context_id_idx ON {table_name} (context_id);"


class TaskContextProgress(BaseModel):
    total: int
    done: int
    failed: int


class WFQEngine:
    def __init__(self, table_name: str, sql_claim: str,
                 handler_func: Callable[[WFQTaskData], Awaitable[None]],
                 completion_func: Callable[[WFQTaskData], Awaitable[None]] | None = None,
                 retry_interval: float = 5):
        logger.info("initializing WFQEngine table=%s", table_name)
        self.table_name = table_name
        self.sql_claim = sql_claim
        self.handler_func = handler_func
        self.completion_func = completion_func
        self.retry_interval = retry_interval # after how much minutes should retry happen


    async def create_table(self):
        """Create the needed postgresql table for task tracking"""

        logger.info("creating table=%s", self.table_name)
        async with AsyncSessionLocal() as session:
            await session.execute(text(CREATE_WFQ_TABLE.format(table_name=self.table_name)))
            await session.execute(text(CREATE_WFQ_CONTEXT_INDEX.format(table_name=self.table_name)))
            await session.commit()


    async def start_concurrent_workers(self, count: int):
        """Run `count` workers until cancelled. Entry point for a dedicated worker process."""

        await asyncio.gather(*(self.worker(f"w{i}") for i in range(count)))


    async def worker(self, worker_name):
        """Worker function that runs until cancelled. Pings every 5 seconds if there is work submitted.
        First run long overdue 'running' taskiq, then run pending taskiq depending on chosen self.sql_claim
        Note: long overdue 'running' taskiq should be a rare occurrence. ex. interrupted worker mid execution"""

        logger.info("wfq worker started worker=%s table=%s", worker_name, self.table_name)
        while True:
            try:
                while True:
                    task: RowMapping | None = await self._claim_for_retry()
                    if task is not None:
                        logger.info("reclaimed stale task task_id=%s", task["id"])
                        await self._do_work(task)
                        continue
                    task = await self._claim_one()  # sql_claim
                    if task is None:
                        logger.debug("no taskiq; sleeping worker=%s table=%s", worker_name, self.table_name)
                        break
                    logger.info("claimed task task_id=%s", task["id"])
                    await self._do_work(task)
            except asyncio.CancelledError:
                logger.info("wfq worker cancelled worker=%s", worker_name)
                raise
            except Exception as exc:
                logger.exception(f"wfq worker loop error worker={worker_name}; backing off. reason: {exc}")
            await asyncio.sleep(5)


    async def _claim_one(self) -> RowMapping | None:
        """Calls the given sql_claim query on WFQEngine object creation - async safe"""
        sql = text(self.sql_claim.format(table_name=self.table_name))
        async with AsyncSessionLocal() as session:
            row = (await session.execute(sql)).mappings().first()
            await session.commit()
            return row


    async def _do_work(self, task: RowMapping):
        """Main thing a task does. Calls the given handler_func and completion_func on WFQEngine object creation.
        Any exception from either is caught and recorded as the task's failure_reason, so a task never gets
        stuck in 'running' because of an error we didn't anticipate."""

        log = _TaskLog(logger, {"task_id": task["id"], "context_id": task["context_id"]})
        try:
            await self.handler_func(WFQTaskData(**task))
        except Exception as exc:
            reason = repr(exc)
            log.warning("task failed: %s", reason, exc_info=exc)
            counts = await self._update_task_state_and_get_count(
                task["id"], task["context_id"], WFQTaskStatus.FAILED, reason)
        else:
            counts = await self._update_task_state_and_get_count(
                task["id"], task["context_id"], WFQTaskStatus.DONE)
            log.info("task done (%s/%s in context)", counts.done + counts.failed, counts.total)

        if counts.done + counts.failed == counts.total and self.completion_func is not None:
            try:
                await self.completion_func(WFQTaskData(**task))
                log.info("context complete")
            except Exception as exc:
                reason = repr(exc)
                log.exception("completion failed, failing whole context: %s", reason)
                await self._fail_whole_context(reason, task["context_id"])


    async def _claim_for_retry(self) -> RowMapping | None:
        """Outputs an element that is with 'running' status older than the given retry_interval - async safe
        Note: retry_interval should be longer than expected longest time for a task to finish.
        Otherwise, a running task is retried"""

        async with AsyncSessionLocal() as session:
            task = (await session.execute(text(f"""
                UPDATE {self.table_name} t
                SET started_at = now()
                WHERE t.id = (
                    SELECT t2.id
                    FROM {self.table_name} t2
                    WHERE t2.status = 'running'
                      AND t2.started_at < now() - ({self.retry_interval} * interval '1 minute')
                    ORDER BY t2.started_at ASC
                    FOR UPDATE OF t2 SKIP LOCKED
                    LIMIT 1
                )
                RETURNING t.id, t.context_id, t.payload;"""))).mappings().first()
            await session.commit()
        return task


    async def _update_task_state_and_get_count(self, task_id: str, context_id: str,
                                               status: WFQTaskStatus, failure_reason: str = "") -> TaskContextProgress:
        """Helper function that acquires lock on context_id,
        so no 2 or more concurrent taskiq can read same counts after updating. - async safe
        task_id - task to update status of, context_id - task is part of this context, so lock on it,
        status - to which status to update, failure_reason = is set only if task failed, for easier debug"""

        logger.debug("update task_id=%s -> status=%s %s", task_id, status, failure_reason)
        lock_sql = text("SELECT pg_advisory_xact_lock(hashtext(:context_id))")
        update_sql = text(f"""UPDATE {self.table_name}
                              SET status = :status,
                                  failure_reason = :failure_reason,
                                  finished_at = now()
                              WHERE context_id = :context_id AND id = :task_id""")
        count_sql = text(f"""SELECT count(*)                                 AS total,
                                    count(*) FILTER (WHERE status = 'done')   AS done,
                                    count(*) FILTER (WHERE status = 'failed') AS failed
                             FROM {self.table_name} WHERE context_id = :context_id""")

        async with AsyncSessionLocal() as session:
            await session.execute(lock_sql, {"context_id": context_id})
            await session.execute(update_sql,
                                  {"context_id": context_id,
                                   "task_id": task_id,
                                   "status": status,
                                   "failure_reason": failure_reason})
            counts = (await session.execute(count_sql, {"context_id": context_id})).mappings().one()
            await session.commit()
            logger.debug("update task_id=%s done, counts=%s", task_id, dict(counts))
            return TaskContextProgress(**counts)


    async def _fail_whole_context(self, failure_reason: str, context_id: str):
        """Helper function used to set all taskiq' statuses to 'failed' for a given context_id"""

        fail_whole_context_sql = f"""UPDATE {self.table_name} t
                                     SET status = 'failed', failure_reason = :failure_reason
                                     WHERE t.context_id = :context_id"""
        async with AsyncSessionLocal() as session:
            await session.execute(text(fail_whole_context_sql),
                                  {"context_id": context_id, "failure_reason": failure_reason})
            await session.commit()
