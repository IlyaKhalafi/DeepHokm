"""Render win-rate curves and key training scalars to ``figures/``.

Reads the TensorBoard event files written by the training run and produces
PNG figures: gauntlet win rates over time and the core PPO training scalars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

FIGURES_DIR = Path("figures")


def main() -> None:
    """Load TensorBoard scalars and render the figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("logs/tb"),
        help="TensorBoard log directory (may contain multiple runs)",
    )
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = sorted(p for p in args.logdir.glob("**") if p.is_dir() and any(p.iterdir()))
    if not runs:
        raise SystemExit(f"no event files under {args.logdir}")

    scalars: dict[str, list[tuple[int, float]]] = {}
    for run in runs:
        acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            points = [(event.step, event.value) for event in acc.Scalars(tag)]
            scalars.setdefault(tag, []).extend(points)

    if not scalars:
        raise SystemExit(f"no scalar tags found under {args.logdir}")

    # Figure 1: gauntlet win rates.
    gauntlet_tags = sorted(t for t in scalars if t.startswith("gauntlet/"))
    if gauntlet_tags:
        fig, ax = plt.subplots(figsize=(8, 5))
        for tag in gauntlet_tags:
            points = sorted(scalars[tag])
            steps = [p[0] for p in points]
            values = [p[1] for p in points]
            ax.plot(steps, values, label=tag.split("/", 1)[1], marker="o")
        ax.set_xlabel("environment steps")
        ax.set_ylabel("win rate")
        ax.set_ylim(0, 1)
        ax.set_title("Gauntlet win rates")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = args.out_dir / "gauntlet_win_rates.png"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")

    # Figure 2: training scalars.
    train_tags = [
        t
        for t in scalars
        if t.startswith(("train/", "rollout/"))
        and not t.startswith(("train/loss", "train/n_updates"))
    ][:8]
    if train_tags:
        fig, axes = plt.subplots(len(train_tags), 1, figsize=(8, 3 * len(train_tags)))
        if len(train_tags) == 1:
            axes = [axes]
        for ax, tag in zip(axes, train_tags, strict=True):
            points = sorted(scalars[tag])
            ax.plot([p[0] for p in points], [p[1] for p in points])
            ax.set_title(tag)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        out = args.out_dir / "training_scalars.png"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
