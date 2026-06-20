"""HTTP and scraping tools implemented as LangChain tools."""

from __future__ import annotations

from langchain_core.tools import tool


@tool("http_get")
def http_get_tool(url: str) -> str:
    """Fetch a URL with HTTP GET and return the raw response body."""

    import requests

    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.text


@tool("page_text")
def page_text_tool(url: str) -> str:
    """Fetch a web page and return its visible text content."""

    from bs4 import BeautifulSoup

    html = http_get_tool.invoke(url)
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
