"""Browser automation tools implemented as LangChain tools."""

from __future__ import annotations

from langchain_core.tools import tool


@tool("browser_title")
def selenium_title_tool(url: str) -> str:
    """Open a URL in a headless Selenium browser and return the page title."""

    from selenium import webdriver

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    with webdriver.Chrome(options=options) as driver:
        driver.get(url)
        return driver.title
