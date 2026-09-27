"""Record an animated GIF of real play in the web UI.

Drives a human-mode game through the same browser the UI ships to,
screenshots the interesting moments (setup, trump call, each of the first
few tricks with the AI replies visible, the trick-result banner, and the
match-over state), and assembles the frames with ffmpeg.

Output: docs/media/demo.gif (overwrites). Target URL comes from
DEEPHOKM_BASE_URL; the recording is intentionally slower than real play so
a reader can follow each card.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BASE_URL = os.environ.get("DEEPHOKM_BASE_URL", "")
OUT = Path(__file__).resolve().parents[1] / "docs" / "media" / "demo.gif"
TRICKS_TO_SHOW = 4
FPS = 2


def _wait_for_our_turn(page: Page, attempts: int, pause_ms: int) -> None:
    """Step AI seats until it is our turn (trump call or first play).

    The step button is hidden while it is our turn, so every click is
    guarded on visibility.
    """
    for _ in range(attempts):
        if page.locator(".trump-btn:visible").count() > 0:
            return
        if page.locator(".card.legal:visible").count() > 0:
            return
        if page.locator("#step:visible").count() > 0:
            page.click("#step")
        page.wait_for_timeout(pause_ms)


def _play_one_trick(page: Page, shot: Callable[[int], None]) -> bool:
    """Play the viewer's card and hold the resolved trick on screen.

    Returns whether a card was actually available to play.
    """
    for _ in range(60):
        if page.locator(".card.legal").count() > 0:
            page.locator(".card.legal").last.click()
            shot(1800)  # replies landing one by one
            shot(1200)  # completed trick with banner
            _wait_for_our_turn(page, attempts=60, pause_ms=600)
            return True
        page.wait_for_timeout(500)
    return False


def _fast_forward_to_match_over(page: Page) -> bool:
    """Play out the rest of the match; return whether the banner appeared.

    A match this long always reaches at least one later hand, and if the
    viewer becomes hakem again the trump picker -- not a card or the step
    button -- is what's on screen; without handling it here the loop stalls
    instead of finishing. The iteration cap is generous because a real match
    against a real opponent can run to many hands before either team
    reaches 7 points.
    """
    for _ in range(4000):
        if page.locator(".banner:visible").count() > 0:
            return True
        if page.locator(".trump-btn:visible").count() > 0:
            page.locator(".trump-btn").first.click()
            page.wait_for_timeout(300)
        elif page.locator(".card.legal:visible").count() > 0:
            page.locator(".card.legal").first.click()
            page.wait_for_timeout(400)
        elif page.locator("#step:visible").count() > 0:
            page.click("#step")
            page.wait_for_timeout(150)
        else:
            page.wait_for_timeout(300)
    return False


def main() -> int:
    if not BASE_URL:
        print(
            "DEEPHOKM_BASE_URL is not set; copy .env.example to .env first",
            file=sys.stderr,
        )
        return 2
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(BASE_URL)
        page.wait_for_timeout(1200)

        with tempfile.TemporaryDirectory() as tmp:
            frame = 0

            def shot(hold_ms: int = 1400) -> None:
                nonlocal frame
                page.wait_for_timeout(hold_ms)
                page.screenshot(path=f"{tmp}/f{frame:03d}.png")
                frame += 1

            shot(600)  # 1. Setup screen

            # 2. Start a game
            page.select_option("#mode", "human")
            page.click("#start")
            page.wait_for_selector("#game:not(.hidden)", timeout=10000)
            _wait_for_our_turn(page, attempts=10, pause_ms=400)
            shot()

            # 3. Trump call, if it is ours
            if page.locator(".trump-btn:visible").count() > 0:
                page.locator(".trump-btn").first.click()
                shot(1000)

            # 4. A few tricks with the AI replies visible.
            for _ in range(TRICKS_TO_SHOW):
                if not _play_one_trick(page, shot):
                    break

            # 5. Fast-forward to the end so the result banner closes the demo.
            if _fast_forward_to_match_over(page):
                shot(800)

            browser.close()
            _assemble_gif(tmp)
    print(f"wrote {OUT}")
    return 0


def _assemble_gif(frames_dir: str) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", f"{frames_dir}/f%03d.png",
            "-vf", (
                "split[s0][s1];[s0]palettegen=max_colors=128[p];"
                "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
            ),
            "-loop", "0", str(OUT),
        ],
        check=True,
        capture_output=True,
    )


if __name__ == "__main__":
    sys.exit(main())
