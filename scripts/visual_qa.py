"""Automated visual QA for the DeepHokm web UI.

Drives the web UI through its API into representative game states, captures
screenshots with Playwright at two viewports, and sends each screenshot to
the configured vision-language reviewer (an OpenAI-compatible chat endpoint)
with an adversarial prompt. Findings are printed and written to the log
directory for triage; the exit code is 0 only when every capture returns
CLEAN on both dimensions (bug-free and well-designed).

Endpoint and model come from DEEPHOKM_VLM_BASE_URL / DEEPHOKM_VLM_MODEL.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

DEFAULT_BASE_URL = "http://127.0.0.1:8025"
DEFAULT_LOG_DIR = Path("logs/visual_qa")

VIEWPORTS = {
    "desktop": {"width": 1440, "height": 900},
    "mobile": {"width": 390, "height": 844},
}

REVIEW_PROMPT = """You are a meticulous and adversarial UI QA reviewer. \
Inspect this screenshot of a card-game web UI on two dimensions.
(1) BUGS: overlapping or clipped elements, misaligned cards, unreadable text, \
missing expected elements (hand, table, trump indicator, scores, turn \
indicator), impossible or inconsistent game-state displays, broken layout, \
unresponsive-looking controls.
(2) DESIGN QUALITY: visual hierarchy, spacing and alignment consistency, \
color contrast and readability, clarity of affordances (is it obvious what \
is playable and whose turn it is?), overall professional polish.
Output a numbered list with dimension, severity (critical/major/minor), and \
the concrete defect or improvement; output exactly `CLEAN` only if the UI is \
bug-free AND well-designed."""


def api(base_url: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call the web UI REST API."""
    data = json.dumps(body or {}).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


TARGET_STATES = [
    ("human", 101, "trump_selection"),
    ("human", 102, "empty_table"),
    ("human", 103, "mid_trick"),
    ("human", 104, "late_hand"),
    ("human", 105, "hand_won"),
    ("human", 106, "game_over"),
    ("spectate", 107, "mid_trick"),
    ("spectate", 108, "empty_table"),
]


def capture(base_url: str, out_dir: Path, states: list[tuple[str, int, str]]) -> list[Path]:
    """Screenshot every target state at both viewports.

    Each capture drives a fresh game to the target state through the page's
    own fetch calls (the same API the UI uses) and renders it through the
    frontend's real renderer, so the screenshot shows exactly what a user
    would see.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for mode, seed, target in states:
            for viewport_name, viewport in VIEWPORTS.items():
                context = browser.new_context(viewport=viewport)
                # Bypass browser caching so every pass screenshots the current
                # frontend build.
                context.set_extra_http_headers({"Cache-Control": "no-cache"})
                page = context.new_page()
                page.goto(base_url)
                state = window_drive(page, mode, seed, target)
                # The second evaluate argument is bound to the JS `arg`.
                page.evaluate("(s) => window.__deephokmRender(s)", state)
                page.wait_for_timeout(250)
                name = f"{target}_{mode}_{viewport_name}.png"
                path = out_dir / name
                page.screenshot(path=str(path), full_page=True)
                paths.append(path)
                page.close()
                context.close()
        browser.close()
    return paths


def window_drive(page: Any, mode: str, seed: int, target: str) -> dict[str, Any]:
    """Drive a fresh game to the target state through the page's fetch."""
    return page.evaluate(
        # fmt: off
        """async ([mode, seed, target]) => {
            const post = async (path, body) => {
                const response = await fetch(path, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(body || {}),
                });
                return response.json();
            };
            let state = await post('/api/games', {mode, seed});
            const gameId = state.game_id;
            for (let i = 0; i < 400; i++) {
                const tableLen = (state.table || []).length;
                const counts = state.hand_counts || [];
                if (target === 'trump_selection' && state.phase === 'TRUMP_CALL') break;
                if (target === 'empty_table' && state.phase === 'CARD_PLAY' && !tableLen) break;
                if (target === 'mid_trick' && tableLen >= 2 && tableLen <= 3) break;
                if (target === 'late_hand' && state.phase === 'CARD_PLAY' && counts.length && Math.max(...counts) <= 5) break;
                if (target === 'hand_won' && state.phase === 'CARD_PLAY' && [6,7,12].includes((state.tricks_won||[]).reduce((a,b)=>a+b,0))) break;
                if (target === 'game_over' && state.terminal) break;
                if (state.terminal) break;
                if (mode === 'spectate') {
                    state = await post(`/api/games/${gameId}/step`);
                } else if (state.current_seat === 0 && state.legal_actions && state.legal_actions.length) {
                    state = await post(`/api/games/${gameId}/action`, {action: state.legal_actions[0]});
                } else {
                    const response = await fetch(`/api/games/${gameId}`);
                    state = await response.json();
                }
            }
            return state;
        }""",  # noqa: E501
        [mode, seed, target],
    )


def _tiny_png() -> bytes:
    """Return an 8x8 solid-red PNG for endpoint probing."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        header = struct.pack(">I", len(data)) + tag + data
        return header + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes([255, 0, 0] * 8) for _ in range(8))
    ihdr = struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def review_screenshot(image_path: Path, vlm_base_url: str, vlm_model: str, max_tokens: int) -> str:
    """Send one screenshot to the reviewer endpoint; return its verdict text."""
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    body = {
        "model": vlm_model,
        "max_tokens": max_tokens,
        # The reviewer is a reasoning model; disable thinking so the response
        # is the final verdict rather than chain-of-thought.
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": REVIEW_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    request = urllib.request.Request(
        vlm_base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, list):  # some servers return parts
        content = "".join(part.get("text", "") for part in content)
    return content.strip()


def main() -> int:
    """Run capture -> review for every state; exit 0 only if all CLEAN."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("DEEPHOKM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--vlm-base-url", default=os.environ.get("DEEPHOKM_VLM_BASE_URL", ""))
    parser.add_argument("--vlm-model", default=os.environ.get("DEEPHOKM_VLM_MODEL", ""))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()

    if not args.vlm_base_url or not args.vlm_model:
        raise SystemExit("set DEEPHOKM_VLM_BASE_URL and DEEPHOKM_VLM_MODEL (see .env.example)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Probe the endpoint with a tiny generated image first: the spec requires
    # confirming the reviewer accepts image content parts before the loop.
    if args.probe_only:
        tiny = Path(__file__).parent / "_probe.png"
        tiny.write_bytes(_tiny_png())
        verdict = review_screenshot(tiny, args.vlm_base_url, args.vlm_model, 60)
        print("probe response:", verdict[:200])
        tiny.unlink()
        if "red" not in verdict.lower():
            print("WARNING: reviewer did not identify the probe image")
        return 0

    screenshots = capture(args.base_url, args.out_dir, TARGET_STATES)
    print(f"captured {len(screenshots)} screenshots")

    report_path = args.out_dir / "review_report.json"
    findings: dict[str, str] = {}
    all_clean = True
    for path in screenshots:
        print(f"reviewing {path.name} ...", flush=True)
        try:
            verdict = review_screenshot(path, args.vlm_base_url, args.vlm_model, args.max_tokens)
        except urllib.error.URLError as exc:
            print(f"  ERROR reviewing {path.name}: {exc}")
            all_clean = False
            findings[path.name] = f"REVIEW ERROR: {exc}"
            continue
        findings[path.name] = verdict
        is_clean = verdict.strip().endswith("CLEAN") or verdict.strip() == "CLEAN"
        if not is_clean:
            all_clean = False
        print(f"  {'CLEAN' if is_clean else 'FINDINGS'}")
        if not is_clean:
            print(indent(verdict))

    report_path.write_text(json.dumps(findings, indent=2))
    print(f"report: {report_path}")
    print("RESULT:", "CLEAN" if all_clean else "FINDINGS")
    return 0 if all_clean else 1


def indent(text: str, pad: str = "    ") -> str:
    """Indent every line of a verdict for console readability."""
    return "\n".join(pad + line for line in text.splitlines())


if __name__ == "__main__":
    sys.exit(main())
