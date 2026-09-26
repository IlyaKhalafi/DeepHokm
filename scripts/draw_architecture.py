"""Render the RankCNN architecture diagram used in the README.

Generated from the module's real constants rather than hand-drawn, so the
figure cannot drift away from the network it documents: plane count, suit and
rank extents, channel width, block count and head widths are all imported.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from deephokm.cards import NUM_RANKS, NUM_SUITS  # noqa: E402
from deephokm.nn.features import NUM_PLANES, NUM_SCALARS  # noqa: E402
from deephokm.nn.rank_cnn import TRUMP_HIDDEN  # noqa: E402
from deephokm.rules.legality import NUM_ACTIONS  # noqa: E402

CHANNELS = 384
LAYERS = 8
INK = "#1f2328"
MUTED = "#57606a"
LINE = "#d0d7de"
ACCENT = "#2da44e"
TRUMP_C = "#8250df"
CARD_C = "#0969da"
FILL = "#f6f8fa"
W, H = 1180, 470


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x: float, y: float, s: str, size: int = 13, *, fill: str = INK,
         anchor: str = "start", weight: str = "normal", mono: bool = False) -> str:
    family = ("ui-monospace,SFMono-Regular,Menlo,monospace" if mono
              else "-apple-system,Segoe UI,Helvetica,Arial,sans-serif")
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="{family}">{esc(s)}</text>')


def box(x: float, y: float, w: float, h: float, *, stroke: str = LINE,
        fill: str = "#ffffff", rx: int = 8, width: float = 1.4) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def arrow(x1: float, y1: float, x2: float, y2: float, colour: str = MUTED) -> str:
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" '
            f'stroke-width="1.6" marker-end="url(#a)"/>')


def grid(x: float, y: float, cell: float = 13.0) -> str:
    """The 4x13 suit-by-rank plane, drawn to scale."""
    out = []
    for r in range(NUM_SUITS):
        for c in range(NUM_RANKS):
            shade = "#eaf3ea" if r == 0 else "#ffffff"
            out.append(f'<rect x="{x + c * cell}" y="{y + r * cell}" width="{cell}" '
                       f'height="{cell}" fill="{shade}" stroke="{LINE}" stroke-width="0.7"/>')
    return "".join(out)


def build() -> str:
    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif">',
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/>'
        '</marker></defs>',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        text(24, 34, "RankCNN — suit-equivariant action-value network", 17, fill=INK, weight="700"),
        text(24, 56, "Convolutions run along the RANK axis only, with weights "
                     "shared across all four suits.", 12.5, fill=MUTED),
        text(24, 74, "Ranks are ordered, suits are not: a kernel crossing suits "
                     "would assert an adjacency the rules lack.", 12.5, fill=MUTED),
    ]

    # ---- input ----------------------------------------------------------
    gx, gy = 40, 118
    p.append(text(gx, gy - 14, f"input  {NUM_PLANES} planes x {NUM_SUITS} suits x "
                               f"{NUM_RANKS} ranks", 12, fill=INK, weight="600", mono=True))
    for d in (8, 4, 0):  # stacked planes suggesting depth
        p.append(grid(gx + d, gy + d))
    p.append(text(gx - 8, gy + 30, "suits", 10, fill=MUTED, anchor="end"))
    p.append(text(gx + 85, gy + 4 * 13 + 34, "ranks  2 to A", 10, fill=MUTED, anchor="middle"))

    planes = [
        "0  cards in your hand", "1  cards seen so far", "2  cards on the table",
        "3  trump suit", "4  play recency", "5-8  who played each past card",
        "9-12  who played each table card", "13  legal plays right now",
    ]
    for i, label in enumerate(planes):
        p.append(text(gx, gy + 118 + i * 16, label, 10.5, fill=MUTED, mono=True))

    # ---- conv stack -----------------------------------------------------
    bx, by, bw, bh = 300, 108, 250, 190
    p.append(box(bx, by, bw, bh, stroke=ACCENT, fill=FILL))
    p.append(text(bx + bw / 2, by + 26, f"x{LAYERS} blocks", 13,
                  fill=ACCENT, anchor="middle", weight="700"))
    p.append(text(bx + 16, by + 54, "concat( h , mean over suits(h) )", 11.5, fill=INK, mono=True))
    p.append(text(bx + 16, by + 76, f"conv 1x3 along ranks  ->  {CHANNELS} ch",
                  11.5, fill=INK, mono=True))
    p.append(text(bx + 16, by + 98, "GELU", 11.5, fill=INK, mono=True))
    p.append(text(bx + 16, by + 128, "mean over suits is symmetric, so", 10.5, fill=MUTED))
    p.append(text(bx + 16, by + 144, "relabelling suits permutes the output", 10.5, fill=MUTED))
    p.append(text(bx + 16, by + 160, "the same way — equivariance by", 10.5, fill=MUTED))
    p.append(text(bx + 16, by + 176, "construction, not by augmentation.", 10.5, fill=MUTED))
    p.append(arrow(gx + 13 * 13 + 20, gy + 30, bx - 8, gy + 30))

    # ---- scalars --------------------------------------------------------
    sx, sy = 300, 326
    p.append(box(sx, sy, 250, 74, stroke=LINE, fill="#ffffff"))
    p.append(text(sx + 16, sy + 24, f"{NUM_SCALARS} scalars", 12, fill=INK,
                  weight="600", mono=True))
    p.append(text(sx + 16, sy + 44, "phase · tricks won · match points · seat", 10.5, fill=MUTED))
    p.append(text(sx + 16, sy + 62, "projected and added to every cell", 10.5, fill=MUTED))
    p.append(arrow(sx + 125, sy - 6, sx + 125, by + bh + 6))

    # ---- heads ----------------------------------------------------------
    hx = 640
    p.append(box(hx, 112, 250, 104, stroke=CARD_C, fill="#ffffff"))
    p.append(text(hx + 16, 138, "card head", 12.5, fill=CARD_C, weight="700"))
    p.append(text(hx + 16, 160, "conv 1x1  →  1 value per cell", 11, fill=INK, mono=True))
    p.append(text(hx + 16, 182, f"{NUM_SUITS} x {NUM_RANKS}  =  52 card values",
                  11, fill=INK, mono=True))
    p.append(text(hx + 16, 202, "every card keeps its own score", 10.5, fill=MUTED))

    p.append(box(hx, 240, 250, 116, stroke=TRUMP_C, fill="#ffffff"))
    p.append(text(hx + 16, 266, "trump head", 12.5, fill=TRUMP_C, weight="700"))
    p.append(text(hx + 16, 288, "mean + max over the grid", 11, fill=INK, mono=True))
    p.append(text(hx + 16, 310, f"concat scalars -> {TRUMP_HIDDEN} -> {NUM_SUITS}",
                  11, fill=INK, mono=True))
    p.append(text(hx + 16, 330, "pooling discards which suit is which,", 10.5, fill=MUTED))
    p.append(text(hx + 16, 346, "so play defers trump calls to greedy", 10.5, fill=MUTED))

    p.append(arrow(bx + bw + 8, 170, hx - 8, 164))
    p.append(arrow(bx + bw + 8, 240, hx - 8, 292))

    # ---- output ---------------------------------------------------------
    ox = 960
    p.append(box(ox, 150, 190, 124, stroke=INK, fill=FILL))
    p.append(text(ox + 95, 180, f"{NUM_ACTIONS} action values", 13,
                  fill=INK, anchor="middle", weight="700"))
    p.append(text(ox + 95, 204, "52 cards + 4 trump calls", 11, fill=MUTED, anchor="middle"))
    p.append(text(ox + 95, 230, "6.4M parameters", 11.5, fill=INK, anchor="middle", mono=True))
    p.append(text(ox + 95, 252, "30 ms in numpy", 11.5, fill=ACCENT,
                  anchor="middle", weight="700", mono=True))
    p.append(arrow(hx + 250 + 8, 164, ox - 8, 190))
    p.append(arrow(hx + 250 + 8, 298, ox - 8, 232))

    p.append(text(24, H - 22, "The search scores every legal action in sampled "
                              "worlds; the network orders the candidates and "
                              "eliminates only what the evidence rules out.",
                  11.5, fill=MUTED))
    p.append("</svg>")
    return "\n".join(p)


def main() -> None:
    out = pathlib.Path(__file__).resolve().parents[1] / "docs" / "media" / "rankcnn.svg"
    out.write_text(build())
    print(f"wrote {out} ({out.stat().st_size / 1000:.1f} kB)")


if __name__ == "__main__":
    main()
