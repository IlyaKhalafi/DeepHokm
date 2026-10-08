"""Playwright smoke test: the web UI loads, starts a game, and renders state.

Run with the web UI already listening (``make webui``) — the target URL comes
from ``DEEPHOKM_BASE_URL`` (see .env.example for the documented default).
"""

from __future__ import annotations

import os
import sys

from playwright.sync_api import Browser, Page, sync_playwright

from deephokm.rules.state import HAKEM_FIRST_BATCH

BASE_URL = os.environ.get("DEEPHOKM_BASE_URL", "")
CARDS_PER_HAND = 13


def _new_page(browser: Browser, *, width: int, height: int) -> Page:
    """Open a configured page at the web UI."""
    page = browser.new_page(viewport={"width": width, "height": height})
    page.set_default_timeout(120_000)
    page.goto(BASE_URL)
    return page


def _check_human_game(browser: Browser) -> None:
    """Exercise the difficulty picker and a human game."""
    page = _new_page(browser, width=1440, height=900)
    page.wait_for_selector("#start", timeout=10_000)

    # Start a human-vs-model game on the low-latency path.
    page.select_option("#mode", "human")
    page.click("#difficulty")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    assert page.locator("#difficulty").input_value() == "hard"
    page.select_option("#difficulty", "fast")
    page.click("#start")
    page.wait_for_selector("#game:not(.hidden)", timeout=10_000)

    # The hand, scores, and turn indicator must render.
    page.wait_for_selector("#hand .card", timeout=10_000)
    hand_cards = page.locator("#hand .card").count()
    if hand_cards == HAKEM_FIRST_BATCH:
        # A randomly selected human hakem sees only the opening five
        # until declaring trump. Complete that valid initial phase before
        # asserting the normal card-play hand size.
        page.locator(".trump-btn:not([disabled])").first.click()
        page.wait_for_function("""document.querySelectorAll("#hand .card").length >= 12""")
        hand_cards = page.locator("#hand .card").count()
    assert CARDS_PER_HAND - 1 <= hand_cards <= CARDS_PER_HAND, (
        f"expected ~{CARDS_PER_HAND} hand cards, saw {hand_cards}"
    )

    assert page.locator("#points-us").inner_text() == "0"
    assert page.locator("#trump").inner_text() != ""
    assert page.locator("#turn").inner_text() != ""
    assert page.locator("#difficulty-badge").inner_text() == "FAST"

    os.makedirs("logs/visual_qa", exist_ok=True)
    page.screenshot(path="logs/visual_qa/smoke_desktop.png", full_page=True)


def _check_spectate_game(browser: Browser) -> None:
    """Verify spectate mode exposes no private hand."""
    page = _new_page(browser, width=1440, height=900)
    page.select_option("#mode", "spectate")
    page.select_option("#difficulty", "hard")
    page.click("#start")
    page.wait_for_selector("#game:not(.hidden)", timeout=120_000)
    assert page.locator("#hand .card").count() == 0, "spectator saw a hand"
    assert page.locator("#difficulty-badge").inner_text() == "HARD"
    page.click("#step")
    page.wait_for_timeout(300)
    page.screenshot(path="logs/visual_qa/smoke_spectate.png", full_page=True)


def _check_mobile_game(browser: Browser) -> None:
    """Exercise the core game flow at a mobile viewport."""
    page = _new_page(browser, width=390, height=844)
    page.wait_for_selector("#start", timeout=10_000)
    page.click("#start")
    page.wait_for_selector("#hand .card", timeout=10_000)
    page.screenshot(path="logs/visual_qa/smoke_mobile.png", full_page=True)


def main() -> int:
    """Load the page, start games, and assert the key UI elements appear."""
    if not BASE_URL:
        print(
            "DEEPHOKM_BASE_URL is not set; copy .env.example to .env first",
            file=sys.stderr,
        )
        return 2
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        _check_human_game(browser)
        _check_spectate_game(browser)
        _check_mobile_game(browser)
        browser.close()
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
