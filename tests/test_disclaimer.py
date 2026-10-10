"""Both pages: the research-use notice shows on the first visit only."""
import pytest
from playwright.sync_api import Page, expect


@pytest.mark.parametrize("name", ["slide-viewer.html", "slide-annotator.html"])
def test_disclaimer_shows_once(page: Page, server: str, name: str):
    page.goto(f"{server}/{name}")
    expect(page.locator("#disclaimer")).to_be_visible()
    page.click("#disclaimer button")
    expect(page.locator("#disclaimer")).to_be_hidden()
    page.reload()
    page.wait_for_timeout(500)
    expect(page.locator("#disclaimer")).to_be_hidden()
