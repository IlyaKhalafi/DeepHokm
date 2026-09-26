"""Neural network components.

Importing this package must not pull in PyTorch. Playing runs on the numpy
inference path, and a deployment installs numpy without a deep-learning
framework; eagerly importing the torch-backed modules here made
``import deephokm.nn.features`` fail in exactly that environment.

The torch-backed names stay importable and are resolved on first access, so
``from deephokm.nn import HokmTransformerExtractor`` still works wherever torch
is installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deephokm.nn.features import NUM_PLANES, NUM_SCALARS, build_features

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from deephokm.nn.extractor import HokmTransformerExtractor
    from deephokm.nn.policy import HokmMaskablePolicy
    from deephokm.nn.tokenizer import (
        TokenizedObservation,
        tokenize,
        tokenize_tensor_batch,
    )

_TORCH_BACKED = {
    "HokmTransformerExtractor": "deephokm.nn.extractor",
    "HokmMaskablePolicy": "deephokm.nn.policy",
    "TokenizedObservation": "deephokm.nn.tokenizer",
    "tokenize": "deephokm.nn.tokenizer",
    "tokenize_tensor_batch": "deephokm.nn.tokenizer",
}


def __getattr__(name: str) -> Any:
    """Resolve torch-backed names on first use (PEP 562)."""
    module_path = _TORCH_BACKED.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module  # noqa: PLC0415

    return getattr(import_module(module_path), name)


__all__ = [
    "NUM_PLANES",
    "NUM_SCALARS",
    "HokmMaskablePolicy",
    "HokmTransformerExtractor",
    "TokenizedObservation",
    "build_features",
    "tokenize",
    "tokenize_tensor_batch",
]
