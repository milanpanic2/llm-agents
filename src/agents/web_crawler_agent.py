from dataclasses import dataclass

import httpx
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.config import settings


@dataclass
class CrawlerDeps:
    """Per-request state handed to the tools via RunContext.deps."""

    http: httpx.AsyncClient
    crw_base: str


# Tools are module-level functions taking RunContext first. They read the shared
# http client from ctx.deps instead of creating one or reaching for a global.
async def search(ctx: RunContext[CrawlerDeps], query: str, limit: int = 5) -> list[dict]:
    """Search the web and return candidate results.

    Returns a list of {title, url, snippet}. Use this to discover URLs worth scraping.
    """
    query = query.strip()
    if not query:
        raise ModelRetry("The search query was empty. Provide a specific query string.")
    r = await ctx.deps.http.post(
        f"{ctx.deps.crw_base}/v1/search", json={"query": query, "limit": limit}, timeout=30
    )
    r.raise_for_status()
    results = r.json()["data"]["results"]
    if not results:
        raise ModelRetry(f"No results for {query!r}. Try broader or different keywords.")
    return [{"title": x["title"], "url": x["url"], "snippet": x.get("snippet", "")} for x in results]


async def scrape(ctx: RunContext[CrawlerDeps], url: str) -> str:
    """Fetch a single web page and return its plain-text content. Pass a URL from `search`."""
    if not url.startswith(("http://", "https://")):
        raise ModelRetry(f"{url!r} is not a valid URL. Use a full http(s) URL from search results.")
    r = await ctx.deps.http.post(
        f"{ctx.deps.crw_base}/v1/scrape",
        json={"url": url, "formats": ["plainText"]},
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    text = r.json().get("data", {}).get("plainText") or ""
    if not text.strip():
        raise ModelRetry(f"Scraping {url} returned no text. Try a different URL.")
    return text[:20000]  # trim so one page can't blow the context window


INSTRUCTIONS = (
    "You are a web research assistant. You do NOT answer factual questions from "
    "memory. Your workflow is always: (1) call `search` to find relevant pages, "
    "(2) call `scrape` on the 1-3 most promising URLs to read their full text, "
    "(3) answer using only what you scraped, and cite the source URLs. "
    "If the first search is unhelpful, refine the query and search again."
)


class WebCrawlerAgent:
    """Long-lived singleton: builds the agent once and reuses it for every request."""

    def __init__(self) -> None:
        model = OpenAIChatModel(
            settings.llm_model,
            provider=OpenAIProvider(base_url=settings.llm_base_url, api_key=settings.llm_api_key),
        )
        self._http = httpx.AsyncClient()
        self._agent = Agent(
            model,
            deps_type=CrawlerDeps,
            instructions=INSTRUCTIONS,
            tools=[search, scrape],
        )

    async def run(self, prompt: str) -> str:
        result = await self._agent.run(
            prompt, deps=CrawlerDeps(http=self._http, crw_base=settings.crw_base)
        )
        return result.output

    async def aclose(self) -> None:
        await self._http.aclose()
