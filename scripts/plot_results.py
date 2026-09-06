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


ScalarsByRun = dict[str, dict[str, list[tuple[int, float]]]]


def _load_scalars(logdir: Path) -> ScalarsByRun:
    """Read every run under ``logdir``, keyed by run label then tag.

    Keeping runs separate (rather than one flat tag -> points dict) is the
    point: a ``logdir`` holding several distinct runs (the documented,
    intended use of ``--logdir``) must never let one run's points splice
    into another's just because they share a tag and an overlapping step
    range -- each run is always its own series.
    """
    runs = sorted(p for p in logdir.glob("**") if p.is_dir() and any(p.iterdir()))
    if not runs:
        raise SystemExit(f"no event files under {logdir}")

    scalars: ScalarsByRun = {}
    for run in runs:
        acc = EventAccumulator(str(run), size_guidance={"scalars": 0})
        acc.Reload()
        tags = acc.Tags()["scalars"]
        if not tags:
            continue
        run_label = str(run.relative_to(logdir)) if run != logdir else run.name
        run_scalars = scalars.setdefault(run_label, {})
        for tag in tags:
            run_scalars[tag] = sorted((event.step, event.value) for event in acc.Scalars(tag))

    if not scalars:
        raise SystemExit(f"no scalar tags found under {logdir}")
    return scalars


def _plot_gauntlet(scalars: ScalarsByRun, out_dir: Path, multi_run: bool) -> None:
    """Render the gauntlet win-rate figure, one line per (run, opponent)."""
    gauntlet_tags = sorted(
        {
            tag
            for run_scalars in scalars.values()
            for tag in run_scalars
            if tag.startswith("gauntlet/")
        }
    )
    if not gauntlet_tags:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_label, run_scalars in scalars.items():
        for tag in gauntlet_tags:
            points = run_scalars.get(tag)
            if not points:
                continue
            opponent = tag.split("/", 1)[1]
            label = f"{run_label}/{opponent}" if multi_run else opponent
            ax.plot([p[0] for p in points], [p[1] for p in points], label=label, marker="o")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("win rate")
    ax.set_ylim(0, 1)
    ax.set_title("Gauntlet win rates")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / "gauntlet_win_rates.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def _plot_training_scalars(scalars: ScalarsByRun, out_dir: Path, multi_run: bool) -> None:
    """Render the core PPO scalars figure, one subplot per tag."""
    excluded = ("train/loss", "train/n_updates")
    all_tags = {
        tag
        for run_scalars in scalars.values()
        for tag in run_scalars
        if tag.startswith(("train/", "rollout/")) and not tag.startswith(excluded)
    }
    train_tags = sorted(all_tags)[:8]
    if not train_tags:
        return
    fig, axes = plt.subplots(len(train_tags), 1, figsize=(8, 3 * len(train_tags)))
    if len(train_tags) == 1:
        axes = [axes]
    for ax, tag in zip(axes, train_tags, strict=True):
        for run_label, run_scalars in scalars.items():
            points = run_scalars.get(tag)
            if not points:
                continue
            ax.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                label=run_label if multi_run else None,
            )
        ax.set_title(tag)
        ax.grid(alpha=0.3)
        if multi_run:
            ax.legend(fontsize="small")
    fig.tight_layout()
    out = out_dir / "training_scalars.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


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
    scalars = _load_scalars(args.logdir)
    multi_run = len(scalars) > 1
    _plot_gauntlet(scalars, args.out_dir, multi_run)
    _plot_training_scalars(scalars, args.out_dir, multi_run)


if __name__ == "__main__":
    main()
