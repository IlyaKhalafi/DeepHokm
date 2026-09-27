"""Opponent policies implementing the HokmPolicy protocol.

The web UI serves ``PureQNetPolicy`` by default: the numpy action-value
network alone, no live search, ~18 ms per decision. ``NumpyHybridPolicy``
pairs the same network with a live determinized search for higher strength
at a latency cost (see each class's docstring for measured numbers).
Importing this package does not require a deep-learning framework.
"""

from __future__ import annotations

from deephokm.policies.base import HokmPolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.numpy_hybrid import NumpyHybridPolicy
from deephokm.policies.pure_qnet_policy import PureQNetPolicy
from deephokm.policies.random_policy import RandomPolicy

__all__ = [
    "GreedyPolicy",
    "HokmPolicy",
    "NumpyHybridPolicy",
    "PureQNetPolicy",
    "RandomPolicy",
]
