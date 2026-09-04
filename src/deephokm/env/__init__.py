"""Gymnasium environment for Hokm.

Registers the ``Hokm-v0`` entry point on import and re-exports
:class:`~deephokm.env.hokm_env.HokmEnv`.
"""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

from deephokm.env.hokm_env import HokmEnv

_ID = "Hokm-v0"

if _ID not in registry:  # pragma: no cover
    register(
        id=_ID,
        entry_point="deephokm.env.hokm_env:HokmEnv",
    )

__all__ = ["HokmEnv"]
