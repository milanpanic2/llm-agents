import asyncio
import logging
import httpx

from pydantic import BaseModel, ValidationError

from src.config import settings
from src.errors import AppError, BadRequestError


logger = logging.getLogger(__name__)


class SearchHit(BaseModel):
    model_config = {"extra": "ignore"}
    url: str
    title: str
    description: str
    snippet: str
    position: int
    score: float
    category: str

class SearchResponseData(BaseModel):
    results: list[SearchHit]

class SearchResponse(BaseModel):
    success: bool
    data: SearchResponseData

class RenderDecision(BaseModel):
    kind: str
    chosen: str

class ScrapedData(BaseModel):
    plainText: str
    renderDecision: RenderDecision
    creditCost: int
    metadata: dict
    contentType: str
    url: str


class CrwClient:
    def __init__(self):
        self._http = httpx.AsyncClient(base_url=settings.crw_base, timeout=30)

    async def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        if not query:
            raise BadRequestError("CRWC-01", "Missing search query")

        try:
            r = await self._http.post("/v1/search", json={"query": query, "limit": limit})
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # request error or status 500
            raise AppError("CRWC-02", f"Crawler search returned status {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            # unreachable
            raise AppError("CRWC-03", "Crawler search backend unreachable") from exc

        try:
            search_response = SearchResponse(**r.json())
        except (ValidationError, ValueError) as exc:
            raise AppError("CRWC-04", "Crawler search returned malformed data") from exc

        return search_response.data.results


    async def scrape(self, urls: list[str], formats: list[str] | None = None) -> list[ScrapedData] | None:
        if len(urls) == 0:
            raise BadRequestError("CRWC-09", "Missing request urls")
        if formats is None:
            formats = ['plainText']

        results = await asyncio.gather(*(
            self._http.post("/v1/scrape",
                            json={"url": url, "formats": formats},
                            headers={"Content-Type": "application/json"})
            for url in urls), return_exceptions=True)

        scraped_data_list: list[ScrapedData] = []
        for url, item in zip(urls, results):
            if isinstance(item, Exception):
                logger.warning("CRWC-10 Partial success: failed scrape for %s: %s", url, item)
                continue
            else:
                try:
                    parsed = ScrapedData(**{**item.json(), "url": url})
                except (ValidationError, ValueError, TypeError) as exc:
                    logger.warning("CRWC-11 Partial success: CRW scraper returned malformed data for success %s: %s",
                                   url, exc)
                    continue

                scraped_data_list.append(parsed)

        return scraped_data_list if len(scraped_data_list) != 0 else None


    async def aclose(self) -> None:
        await self._http.aclose()
