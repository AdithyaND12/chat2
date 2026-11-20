"""Web search tool: Tavily or SerpAPI when configured, DuckDuckGo HTML otherwise."""

from __future__ import annotations

import html
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from langchain_core.tools import tool

from assistant.config import settings
from assistant.logging import get_logger

log = get_logger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text)).strip()


def _tavily(query: str, max_results: int) -> list[dict]:
    if not settings.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    response = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        },
        timeout=20.0,
    )
    response.raise_for_status()
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("content", ""),
        }
        for item in response.json().get("results", [])
    ]


def _serpapi(query: str, max_results: int) -> list[dict]:
    if not settings.serpapi_api_key:
        raise RuntimeError("SERPAPI_API_KEY not set")
    response = httpx.get(
        "https://serpapi.com/search",
        params={"engine": "google", "q": query, "api_key": settings.serpapi_api_key},
        timeout=20.0,
    )
    response.raise_for_status()
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "description": item.get("snippet", ""),
        }
        for item in response.json().get("organic_results", [])[:max_results]
    ]


def _duckduckgo(query: str, max_results: int) -> list[dict]:
    response = httpx.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": _USER_AGENT},
        timeout=20.0,
    )
    response.raise_for_status()
    page = response.text
    titles = re.findall(
        r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        page,
        re.DOTALL,
    )
    snippets = re.findall(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        page,
        re.DOTALL,
    )
    results: list[dict] = []
    for index, (href, title) in enumerate(titles[:max_results]):
        url = html.unescape(href)
        if "duckduckgo.com/l/" in url:
            query_params = parse_qs(urlparse(url).query)
            url = unquote(query_params.get("uddg", [url])[0])
        results.append(
            {
                "title": _clean(title),
                "url": url,
                "description": _clean(snippets[index]) if index < len(snippets) else "",
            }
        )
    if not results:
        raise RuntimeError("DuckDuckGo returned no results")
    return results


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for live information: facts, prices, or current events."""
    try:
        try:
            results = _tavily(query, max_results)
            provider = "Tavily"
        except RuntimeError:
            try:
                results = _serpapi(query, max_results)
                provider = "Google"
            except RuntimeError:
                results = _duckduckgo(query, max_results)
                provider = "DuckDuckGo"
        if not results:
            return f"I couldn't find any results for {query}."
        lines = [f"Here is what I found for {query} (via {provider}):"]
        for index, item in enumerate(results, start=1):
            lines.append(f"{index}. {item['title']}")
            if item.get("description"):
                lines.append(f"   {item['description']}")
            if item.get("url"):
                lines.append(f"   {item['url']}")
        return "\n".join(lines)
    except httpx.HTTPError as exc:
        log.warning("Search request failed: %s", exc)
        return "I couldn't reach the web search service right now."
    except Exception as exc:
        log.exception("Search tool error: %s", exc)
        return "Something went wrong while searching the web."
