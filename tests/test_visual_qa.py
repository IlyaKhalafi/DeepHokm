"""Tests for the visual QA harness's verdict gate.

The harness's exit code is the milestone's acceptance signal, so the gate that
decides whether a reviewer verdict counts as clean has to be exact.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "visual_qa", Path(__file__).resolve().parents[1] / "scripts" / "visual_qa.py"
)
assert _SPEC is not None and _SPEC.loader is not None
visual_qa = importlib.util.module_from_spec(_SPEC)
sys.modules["visual_qa"] = visual_qa
_SPEC.loader.exec_module(visual_qa)


@pytest.mark.parametrize("verdict", ["CLEAN", " CLEAN ", "CLEAN\n"])
def test_clean_verdicts_pass(verdict: str) -> None:
    assert visual_qa.verdict_is_clean(verdict)


@pytest.mark.parametrize(
    "verdict",
    [
        "1. BUGS (major): the trump indicator is clipped. The UI is not CLEAN",
        "Output exactly `CLEAN`",
        "CLEAN, apart from the mobile layout",
        "",
        "clean",
    ],
)
def test_non_clean_verdicts_fail(verdict: str) -> None:
    """A suffix or substring match would let a failing pass count as clean."""
    assert not visual_qa.verdict_is_clean(verdict)
