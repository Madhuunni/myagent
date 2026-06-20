"""Browser automation tool entry points."""

from __future__ import annotations


def selenium_title_tool(url: str) -> str:
    from selenium import webdriver

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    with webdriver.Chrome(options=options) as driver:
        driver.get(url)
        return driver.title
