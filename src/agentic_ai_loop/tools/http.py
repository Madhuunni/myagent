"""HTTP and scraping tool helpers."""

from __future__ import annotations

import requests
from bs4 import BeautifulSoup


def http_get_tool(url: str) -> str:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.text


def page_text_tool(url: str) -> str:
    html = http_get_tool(url)
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
