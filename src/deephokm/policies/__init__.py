"""Opponent policies implementing the HokmPolicy protocol.

``NumpyHybridPolicy`` is the policy this project ships: an action-value network
in numpy guiding a determinized search. Importing this package does not require
a deep-learning framework.
"""

from __future__ import annotations

from deephokm.policies.base import HokmPolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.numpy_hybrid import NumpyHybridPolicy
from deephokm.policies.random_policy import RandomPolicy

__all__ = [
    "GreedyPolicy",
    "HokmPolicy",
    "NumpyHybridPolicy",
    "RandomPolicy",
]
