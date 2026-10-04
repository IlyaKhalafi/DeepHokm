"""Resolve a checkpoint's input schema without importing torch or scratch code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from deephokm.nn import public_features


def resolve_feature_mode(
    weights: Path | dict[str, np.ndarray], input_planes: int, explicit: str | None
) -> str:
    recorded = None
    if isinstance(weights, Path):
        contract = weights.with_suffix(".features.json")
        if contract.exists():
            data = json.loads(contract.read_text())
            if (
                data.get("schema_version") != 1
                or data["weights_sha256"] != hashlib.sha256(weights.read_bytes()).hexdigest()
                or data["public_features_sha256"]
                != hashlib.sha256(Path(public_features.__file__).read_bytes()).hexdigest()
            ):
                raise ValueError("checkpoint feature contract hash/schema mismatch")
            recorded = data["feature_mode"]
            if data.get("input_planes") != input_planes:
                raise ValueError("checkpoint contract channel count mismatch")
        else:
            metadata_path = weights.parent / "best.json"
            if metadata_path.exists():
                data = json.loads(metadata_path.read_text())
                if Path(data.get("numpy_export", "")).name != weights.name:
                    raise ValueError("checkpoint metadata belongs to another NumPy export")
                if data.get("status") != "complete":
                    raise ValueError("checkpoint training is not complete")
                recorded = data["config"].get("feature_mode", "baseline")
    if explicit is not None and recorded is not None and explicit != recorded:
        raise ValueError("explicit feature mode disagrees with checkpoint metadata")
    mode = explicit or recorded
    if mode is None:
        if input_planes != public_features.feature_plane_count("baseline"):
            raise ValueError(
                "expanded checkpoint requires feature metadata or explicit feature_mode"
            )
        mode = "baseline"
    if public_features.feature_plane_count(mode) != input_planes:
        raise ValueError("checkpoint channels do not match feature mode")
    return mode
