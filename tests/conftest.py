import pytest
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from src.config.settings import settings


def pytest_configure(config):
    config.addinivalue_line("markers", "eval: hits the local model; skipped in CI")


@pytest.fixture(scope="session")
def build_transcription_model() -> OpenAIChatModel:
    _openai_client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    _transcription_model = OpenAIChatModel(
        settings.llm_model,
        provider=OpenAIProvider(openai_client=_openai_client.with_options(max_retries=1, timeout=120.0)),
        settings=OpenAIChatModelSettings(
            max_tokens=600,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    )
    return _transcription_model
