"""Shared pytest configuration.

Caps torch's intra-op thread pool for the whole suite. The models under test
are small, so the default thread count spends more time coordinating than
computing, and on a machine already running other workloads the process gets
starved for cores it never needed — tests that touch torch otherwise take
minutes each instead of seconds.
"""

from __future__ import annotations

import os

import torch as th

th.set_num_threads(int(os.environ.get("DEEPHOKM_TEST_TORCH_THREADS", "2")))
