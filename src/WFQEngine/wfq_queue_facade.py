from sqlalchemy.ext.asyncio import AsyncSession

from src.WFQEngine import wfq_utils
from src.WFQEngine.wfq_engine import FailureReasons, TaskContextProgress
from src.WFQEngine.wfq_models import WFQTask


class WFQTaskQueue:
    """Table-bound facade over the stateless wfq_utils producer/reader helpers.

    Holds only the table name, so callers enqueue and query a queue without
    threading `table_name` through every call. The DB session stays an explicit
    argument on purpose: request-path work must run on the request's session."""

    def __init__(self, table_name: str):
        self.table_name = table_name

    async def add_tasks_bulk(self, session: AsyncSession, wfq_tasks: list[WFQTask], user_id: str):
        await wfq_utils.add_tasks_bulk(session, self.table_name, wfq_tasks, user_id)

    async def get_progress(self, session: AsyncSession, context_id: str) -> TaskContextProgress:
        return await wfq_utils.get_progress(session, self.table_name, context_id)

    async def get_failure_reasons(self, session: AsyncSession, context_id: str) -> list[FailureReasons]:
        return await wfq_utils.get_failure_reasons(session, self.table_name, context_id)

    async def retry_all_failed_tasks(self, session: AsyncSession, context_id: str):
        await wfq_utils.retry_all_failed_tasks(session, self.table_name, context_id)

    async def get_context_created_by(self, session: AsyncSession, context_id: str):
        return await wfq_utils.get_context_created_by(session, self.table_name, context_id)

    @staticmethod
    async def check_if_context_finished(progress: TaskContextProgress) -> bool:
        return progress.done + progress.failed == progress.total
