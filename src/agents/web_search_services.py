import datetime as dt
import logging

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from src.clients.crw_client import CrwClient, ScrapedData, SearchHit
from src.errors import AppError, BadRequestError

logger = logging.getLogger(__name__)


WEB_QUERY_GENERATOR_AGENT_INSTRUCTIONS = (
    "You are a web search engine query generator."
    "Based on what you know and the given prompt, write 3 web search engine queries."
    "1. Generate 2 queries for getting more general information that you don't know, if there are such."
    "2. Generate 1 query for getting up to date information regarding the prompt."
)

web_query_generator_agent = Agent(
    instructions=WEB_QUERY_GENERATOR_AGENT_INSTRUCTIONS,
    output_type=list[str],
)


@web_query_generator_agent.tool_plain
def get_current_date() -> str:
    """Return today's date as an ISO string (YYYY-MM-DD), for building up-to-date queries."""
    return dt.date.today().isoformat()


CONTENT_RELEVANCY_AGENT_INSTRUCTIONS = (
    "You are content relevancy analyzer agent."
    "You will be given content in following format:\n"
            "[URL: ...\n"
            "Title: ...\n"
            "Description: ...\n"
            "Snippet: ...]\n\n"
    "Look through urls, titles, descriptions, and snippets for all the entries."
    "Output only the urls relevant to the prompt"
)

content_relevancy_agent = Agent(
    instructions=CONTENT_RELEVANCY_AGENT_INSTRUCTIONS,
    output_type=list[str]
)


WEBSITE_DATA_SUMMARY_AGENT_INSTRUCTIONS = (
    "You are a website data summary agent."
    "You will get information gathered from websites about the prompt"
    "You need to make an index of the information that best explains the prompt"
)

website_data_summary_agent = Agent(
    instructions=WEBSITE_DATA_SUMMARY_AGENT_INSTRUCTIONS,
    output_type=str
)


async def enhance_prompt_with_search(prompt: str, model: OpenAIChatModel, crw_client: CrwClient) -> str:
    queries = await query_expansion(model, prompt)

    search_results: list[SearchHit] = await execute_search_for_queries(crw_client, queries, 8)

    complete_prompt = f"{prompt}" + " ".join(queries)
    relevant_urls = await extract_relevant_urls_from_search_results(model, complete_prompt, search_results)

    scraped_data_list: list[ScrapedData] | None = await crw_client.scrape(urls=relevant_urls)
    if not scraped_data_list:
        raise AppError("WSP-03", "CRW scrape results are empty. Exiting.")

    website_data: str = "\n".join(item.plainText for item in scraped_data_list)
    answer = await website_data_summary_agent.run(user_prompt=
                                                  f"Prompt: {prompt}\n"
                                                  f"Website data:[\n{website_data}\n]",
                                                  model=model)

    return answer.output


async def query_expansion(model: OpenAIChatModel, query: str) -> list[str]:
    generated_queries_answer = await web_query_generator_agent.run(query, model=model)
    queries: list[str] = [query, generated_queries_answer.output]
    return queries


async def extract_relevant_urls_from_search_results(model: OpenAIChatModel, prompt: str, search_results: list[
    SearchHit]) -> list[str]:
    content_for_review: list[str] = []
    for search_result in search_results:
        item_for_review = (
            f"[URL: {search_result.url}\n"
            f"Title: {search_result.title}\n"
            f"Description: {search_result.description}\n"
            f"Snippet: {search_result.snippet}]\n\n"
        )
        content_for_review.append(item_for_review)

    content = "".join(content_for_review)
    message = (
        f"Prompt: {prompt}\n\n"
        f"Content to review: (\n{content}\n)"
    )

    answer = await content_relevancy_agent.run(user_prompt=message, model=model)
    relevant_urls: list[str] = answer.output

    allowed = {r.url for r in search_results}
    return [r for r in relevant_urls if r in allowed]


async def execute_search_for_queries(crw_client: CrwClient, queries: list[str], limit: int) -> list[SearchHit]:
    if limit > 10:
        raise BadRequestError("WSP-01", "Maximum search limit for a query is 10")
    search_results = []
    for query in queries:
        result: list[SearchHit] = await crw_client.search(query, limit)
        search_results.extend(result)

    if len(search_results) == 0:
        raise AppError("WSP-02", "CRW Search results are empty, CRW probably unreachable")

    return search_results
