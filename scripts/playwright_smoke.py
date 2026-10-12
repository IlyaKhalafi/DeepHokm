"""Playwright smoke test: the web UI loads, starts a game, and renders state.

Run with the web UI already listening (``make webui``) — the target URL comes
from ``DEEPHOKM_BASE_URL`` (see .env.example for the documented default).
"""

from __future__ import annotations

import os
import sys
from typing import Literal

from playwright.sync_api import Browser, Page, Route, sync_playwright

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


def _check_superseded_game(
    browser: Browser,
    source: Literal["resume", "start"],
    outcome: Literal["200", "404", "500", "network"],
) -> None:
    """Given a held A reply, starting B must make A success/errors harmless."""
    with browser.new_context() as context:
        page = context.new_page()
        page.set_default_timeout(10_000)
        context.add_init_script("""document.addEventListener('DOMContentLoaded', () => {
            const originalResume = resumeGame;
            resumeGame = async () => {
                try { await originalResume(); }
                finally { window.resumeDone = true; }
            };
            const originalStart = startGame;
            let starts = 0;
            startGame = async () => {
                const number = ++starts;
                try { await originalStart(); }
                finally { window['startDone' + number] = true; }
            };
        });""")
        # Given: real API states with distinct IDs and visible hand numbers.
        payload = {"mode": "spectate", "difficulty": "fast", "seed": 1}
        a = context.request.post(f"{BASE_URL}/api/games", data=payload).json()
        b = context.request.post(f"{BASE_URL}/api/games", data=payload).json()
        a["hand_number"] = 41
        b["hand_number"] = 42
        held: list[Route] = []
        if source == "resume":
            context.add_init_script(f"localStorage.setItem('deephokm.gameId', '{a['game_id']}');")

            def hold_resume(route: Route) -> None:
                held.append(route)

            page.route(f"**/api/games/{a['game_id']}", hold_resume)
            page.route("**/api/games", lambda route: route.fulfill(json=b))
        else:

            def hold_first_start(route: Route) -> None:
                if held:
                    route.fulfill(json=b)
                else:
                    held.append(route)

            page.route("**/api/games", hold_first_start)
        page.goto(BASE_URL)
        if source == "start":
            page.click("#start")
        page.wait_for_function("document.readyState === 'complete'")
        assert len(held) == 1, "A request did not reach the barrier"
        # When: B's POST completes before A's held reply is released.
        page.click("#start")
        done = "startDone1" if source == "resume" else "startDone2"
        page.wait_for_function(f"window.{done} === true")
        before = page.evaluate("""() => ({gameId, state,
            saved: localStorage.getItem(SAVED_GAME_KEY),
            status: $('status').textContent, error: $('status').className,
            visible: !$('game').classList.contains('hidden'),
            phase: $('phase').textContent})""")
        assert before["gameId"] == b["game_id"]
        assert before["visible"] and "#42" in before["phase"]
        match outcome:
            case "200":
                held[0].fulfill(json=a)
            case "404" | "500":
                held[0].fulfill(status=int(outcome), json={"detail": "stale A"})
            case "network":
                held[0].abort("failed")
            case unreachable:
                raise AssertionError(f"Unexpected response outcome: {unreachable}")
        done = "resumeDone" if source == "resume" else "startDone1"
        page.wait_for_function(f"window.{done} === true")
        # Then: state, storage, visibility and status still belong to B.
        after = page.evaluate("""() => ({gameId, state,
            saved: localStorage.getItem(SAVED_GAME_KEY),
            status: $('status').textContent, error: $('status').className,
            visible: !$('game').classList.contains('hidden'),
            phase: $('phase').textContent})""")
        assert after == before, f"{source}/{outcome}: stale A changed B: {after}"
        print(f"race {source}/{outcome}: B preserved")


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
        for source in ("resume", "start"):
            for outcome in ("200", "404", "500", "network"):
                _check_superseded_game(browser, source, outcome)
        _check_human_game(browser)
        _check_spectate_game(browser)
        _check_mobile_game(browser)
        browser.close()
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
