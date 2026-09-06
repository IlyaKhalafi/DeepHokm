"""Tests for scripts/plot_results.py's TensorBoard aggregation.

Imports the script as a module (it lives outside src/, so it is not a
package); this exercises the exact functions ``make plot`` calls.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

_SPEC = importlib.util.spec_from_file_location(
    "plot_results", Path(__file__).resolve().parents[1] / "scripts" / "plot_results.py"
)
assert _SPEC is not None and _SPEC.loader is not None
plot_results = importlib.util.module_from_spec(_SPEC)
sys.modules["plot_results"] = plot_results
_SPEC.loader.exec_module(plot_results)


def _write_run(run_dir: Path, tag: str, values: list[tuple[int, float]]) -> None:
    writer = SummaryWriter(log_dir=str(run_dir))
    for step, value in values:
        writer.add_scalar(tag, value, global_step=step)
    writer.close()


def test_separate_runs_stay_separate_series(tmp_path: Path) -> None:
    """Two runs sharing a tag and overlapping steps must not splice together.

    Regression test: the loader used to flatten every run's points into one
    list per tag and sort only by step, so two distinct experiments logged
    under the same --logdir (the documented, intended use of the flag) could
    interleave into a single misleading curve.
    """
    _write_run(tmp_path / "run_a", "gauntlet/random", [(0, 0.1), (100, 0.9)])
    _write_run(tmp_path / "run_b", "gauntlet/random", [(0, 0.5), (100, 0.5)])

    scalars = plot_results._load_scalars(tmp_path)

    assert set(scalars) == {"run_a", "run_b"}

    def steps_and_values(points: list[tuple[int, float]]) -> tuple[list[int], list[float]]:
        return [p[0] for p in points], [round(p[1], 4) for p in points]

    # TensorBoard round-trips scalars through float32, so compare rounded
    # (the point under test is which run a point belongs to, not precision).
    assert steps_and_values(scalars["run_a"]["gauntlet/random"]) == ([0, 100], [0.1, 0.9])
    assert steps_and_values(scalars["run_b"]["gauntlet/random"]) == ([0, 100], [0.5, 0.5])


def test_gauntlet_figure_renders_one_series_per_run(tmp_path: Path) -> None:
    _write_run(tmp_path / "run_a", "gauntlet/random", [(0, 0.1), (100, 0.9)])
    _write_run(tmp_path / "run_b", "gauntlet/random", [(0, 0.5), (100, 0.5)])
    out_dir = tmp_path / "figures"
    out_dir.mkdir()

    scalars = plot_results._load_scalars(tmp_path)
    plot_results._plot_gauntlet(scalars, out_dir, multi_run=len(scalars) > 1)

    assert (out_dir / "gauntlet_win_rates.png").is_file()


def test_single_run_series_label_has_no_run_prefix(tmp_path: Path) -> None:
    """A single-run --logdir keeps the plain tag label (no visual regression)."""
    _write_run(tmp_path / "only_run", "gauntlet/random", [(0, 0.4)])

    scalars = plot_results._load_scalars(tmp_path)
    assert len(scalars) == 1
