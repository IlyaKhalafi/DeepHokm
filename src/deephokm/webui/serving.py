"""Model serving for the web UI: the trained policy as a HokmPolicy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch as th

from deephokm.env.spaces import Observation
from deephokm.policies.random_policy import RandomPolicy

DEFAULT_MODEL_PATH = "checkpoints/final.zip"


class ServedPolicy:
    """The trained MaskablePPO checkpoint acting as an opponent.

    Inference runs under ``torch.inference_mode()`` with action masks on the
    faster of CPU/GPU for batch-1 calls (CPU on this machine), pinned to one
    intra-op thread.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        """Load the checkpoint.

        Args:
            model_path: Path to the saved ``MaskablePPO`` zip.
            device: Inference device (CPU is faster for batch-1 calls).
        """
        from sb3_contrib import MaskablePPO  # noqa: PLC0415

        # is_file(), not exists(): a bind mount whose host path is missing
        # arrives as an empty directory, and MaskablePPO.load would fail on it
        # with an opaque IsADirectoryError deep inside torch.
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"model checkpoint not found (or not a file) at {model_path}; "
                "set DEEPHOKM_MODEL"
            )
        th.set_num_threads(1)
        self.model = MaskablePPO.load(model_path, device=device)
        self.device = device

    def act(self, observation: Observation, action_mask: np.ndarray) -> int:
        """Return the policy's action for the observation."""
        obs = {key: np.asarray(value)[None, ...] for key, value in observation.items()}
        mask = np.asarray(action_mask, dtype=bool)[None, ...]
        with th.inference_mode():
            actions, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(np.asarray(actions).reshape(-1)[0])

    def reset(self, seed: int | None = None) -> None:
        """No-op: the served policy is stateless across episodes."""


def resolve_model_path() -> str:
    """Return the checkpoint path from the environment (or the default)."""
    return os.environ.get("DEEPHOKM_MODEL", DEFAULT_MODEL_PATH)


def build_opponents(served: Any | None) -> list[Any]:
    """Opponent list: the served policy at every seat (learner slot included).

    The env ignores the entry for its own seat, so one list serves both
    human and spectate modes. Any object implementing the ``HokmPolicy``
    protocol is accepted, so the network-plus-search policy and the
    reinforcement-learning checkpoint are interchangeable here.
    """
    if served is None:
        return [RandomPolicy(11 + i) for i in range(4)]
    return [served, served, served, served]
