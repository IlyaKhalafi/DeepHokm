"""Playwright smoke test: the web UI loads, starts a game, and renders state.

Run with the web UI already listening (``make webui``) — the target URL comes
from ``DEEPHOKM_BASE_URL`` (see .env.example for the documented default).
"""

from __future__ import annotations

import os
import sys

from playwright.sync_api import sync_playwright

BASE_URL = os.environ.get("DEEPHOKM_BASE_URL", "")
CARDS_PER_HAND = 13


def main() -> int:
    """Load the page, start a human game, assert the key UI elements appear."""
    if not BASE_URL:
        print(
            "DEEPHOKM_BASE_URL is not set; copy .env.example to .env first",
            file=sys.stderr,
        )
        return 2
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(BASE_URL)
        page.wait_for_selector("#start", timeout=10_000)

        # Start a human-vs-model game with a fixed seed.
        page.select_option("#mode", "human")
        page.fill("#seed", "3")
        page.click("#start")
        page.wait_for_selector("#game:not(.hidden)", timeout=10_000)

        # The hand, scores, and turn indicator must render.
        page.wait_for_selector("#hand .card", timeout=10_000)
        hand_cards = page.locator("#hand .card").count()
        assert CARDS_PER_HAND - 1 <= hand_cards <= CARDS_PER_HAND, (
            f"expected ~{CARDS_PER_HAND} hand cards, saw {hand_cards}"
        )

        assert page.locator("#points-us").inner_text() == "0"
        assert page.locator("#trump").inner_text() != ""
        assert page.locator("#turn").inner_text() != ""

        # Screenshot for the record.
        os.makedirs("logs/visual_qa", exist_ok=True)
        page.screenshot(path="logs/visual_qa/smoke_desktop.png", full_page=True)

        # Spectate mode renders with no private hand.
        spectate = browser.new_page(viewport={"width": 1440, "height": 900})
        spectate.goto(BASE_URL)
        spectate.select_option("#mode", "spectate")
        spectate.fill("#seed", "5")
        spectate.click("#start")
        spectate.wait_for_selector("#game:not(.hidden)", timeout=10_000)
        assert spectate.locator("#hand .card").count() == 0, "spectator saw a hand"
        spectate.click("#step")
        spectate.wait_for_timeout(300)
        spectate.screenshot(path="logs/visual_qa/smoke_spectate.png", full_page=True)

        # Mobile viewport sanity.
        mobile = browser.new_page(viewport={"width": 390, "height": 844})
        mobile.goto(BASE_URL)
        mobile.wait_for_selector("#start", timeout=10_000)
        mobile.click("#start")
        mobile.wait_for_selector("#hand .card", timeout=10_000)
        mobile.screenshot(path="logs/visual_qa/smoke_mobile.png", full_page=True)

        browser.close()
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
