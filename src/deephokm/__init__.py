"""DeepHokm: reinforcement learning for Hokm, the Persian trick-taking card game.

The package is organised in layers:

- :mod:`deephokm.cards` — card, suit, rank and deck primitives.
- :mod:`deephokm.rules` — the pure Hokm rules engine (legality, trick
  resolution, scoring, hakem rotation).
- :mod:`deephokm.env` — a Gymnasium environment exposing the engine to
  learning agents.
- :mod:`deephokm.policies` — the policy protocol plus scripted baselines.
- :mod:`deephokm.nn` — the transformer feature extractor and policy wiring.
- :mod:`deephokm.training` — the self-play training pipeline.
- :mod:`deephokm.webui` — the web UI for playing against the trained model.
"""

__version__ = "0.1.0"
