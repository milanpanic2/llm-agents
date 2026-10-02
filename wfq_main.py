import asyncio
from functools import partial

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.agents import transcriptions_service
from src.agents.transcriptions_service import TRANSCRIPTION_TASKS_TABLE_NAME
from src.config.settings import settings
from src.config.telemetry import init_worker_telemetry
from src.database.connection import async_engine, init_db
from src.garage.garage_client import init_garage_client
from src.WFQEngine.wfq_engine import WFQEngine
from src.WFQEngine.wfq_queries import WFQ_LONGEST_IDLE_CLAIM


# Optional: workers can be run as separate process from fastapi app with making a deployment of this
async def main():
    init_worker_telemetry(async_engine.sync_engine)  # same stdout logging + traces/metrics as the API
    init_db()

    model = OpenAIChatModel(
        settings.llm_model,
        provider=OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key),
    )
    garage = init_garage_client()

    transcription_licq = WFQEngine(
        table_name=TRANSCRIPTION_TASKS_TABLE_NAME,
        sql_claim=WFQ_LONGEST_IDLE_CLAIM,
        handler_func=partial(transcriptions_service.transcription_worker_handler_func,
                             model=model, garage_client=garage),
        completion_func=partial(transcriptions_service.transcription_worker_completion_func,
                                garage_client=garage),
    )
    await transcription_licq.create_table()

    try:
        await transcription_licq.start_concurrent_workers(3)
    finally:
        await async_engine.dispose()  # this process owns its pool -> dispose on exit


if __name__ == "__main__":
    asyncio.run(main())
