# DeepHokm

Reinforcement learning for **Hokm** (حکم), the Persian trick-taking card game: a
fully tested rules engine, a Gymnasium environment with action masking, a
transformer policy trained by self-play `MaskablePPO`, a web UI for playing
against the trained model, and Docker packaging.

## Status

Under development. Milestones: scaffold, rules engine, environment, network,
training pipeline, training run, web UI, visual QA, containerization.

## Contents

- [Rules](#rules)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Training](#training)
- [Web UI](#web-ui)
- [Docker](#docker)
- [Results](#results)
- [Configuration](#configuration)
- [Known limitations](#known-limitations)

## Rules

Hokm is a four-player partnership trick-taking game. Seats 0-3 play clockwise;
teams A = {0, 2} and B = {1, 3} sit opposite each other. The implementation pins
these deliberate design decisions (no other variants are supported):

- Standard 52-card deck; suits clubs, diamonds, hearts, spades (ids 0-3); ranks
  2..10, J, Q, K, A with 2 lowest and A highest; card id = 13 * suit + rank.
- The first **hakem** (trump caller) is a uniformly random seat. The hakem is
  dealt 5 cards and declares the trump suit, then all 52 cards are dealt so
  every player holds exactly 13 (the hakem 5 + 8).
- The hakem leads the first trick. Players must follow the led suit when able,
  otherwise may play any card. The highest trump wins the trick; absent trump,
  the highest card of the led suit wins. The winner leads next. 13 tricks per
  hand.
- Capturing 7+ of 13 tricks wins the hand for the team and scores 1 game point.
  No kot/bustom bonus in v1.
- First team to 7 game points wins the match. One RL episode = one full match.
  If the hakem's team won the hand the hakem stays; otherwise the hakem passes
  to seat `(old_hakem + 1) % 4`.
- Each player observes only public information plus their own hand.

## Architecture

- `src/deephokm/cards.py` — card, suit, rank and deck primitives.
- `src/deephokm/rules/` — pure rules engine: legality, trick resolution,
  scoring, hakem rotation.
- `src/deephokm/env/` — `HokmEnv` (Gymnasium `Hokm-v0`) with action masking.
- `src/deephokm/policies/` — the `HokmPolicy` protocol and scripted baselines.
- `src/deephokm/nn/` — card embeddings, transformer feature extractor, policy
  wiring for sb3-contrib `MaskablePPO`.
- `src/deephokm/training/` — vectorized self-play pipeline, opponent snapshot
  pool, evaluation gauntlet, CLI.
- `src/deephokm/webui/` — FastAPI app serving a static frontend and the REST
  API; the server owns all game state.
- `scripts/` — environment benchmark, result plotting, visual QA harness.

## Quickstart

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
make test
make lint
```

## Environment

`Hokm-v0` is a single-seat Gymnasium environment: the learner plays one seat,
three opponent policies (any `HokmPolicy` implementation) play the rest inside
`step()`. Observations are a Dict of binary card vectors plus small integer
scalars (see `src/deephokm/env/spaces.py`); the action space is `Discrete(56)`
(52 card plays + 4 trump declarations) with legality masks available via
`env.action_masks()` and `info["action_mask"]`. Illegal actions raise
`ValueError` rather than being resampled.

One note on `gymnasium.utils.env_checker.check_env`: its determinism probe
samples an action *before* its final `reset()`, so for any environment whose
legal-action set changes phase across resets (trump call vs. card play) the
probed action can belong to the previous episode's hand. The suite therefore
runs `check_env` through a small documented shim (`CheckerProbeShim` in
`tests/test_hokm_env.py`) that resamples only that blind probe; every other
checker assertion runs against the bare environment.

Benchmark: `make bench` (mask-respecting random play, ~10k learner steps/s
per worker on this machine).

## Network

`HokmTransformerExtractor` tokenizes the Dict observation into a fixed
34-token layout: up to 13 hand tokens (no positional embedding — the hand is
a set), up to 4 trick tokens in play order, 13 history tokens (completed
tricks, most recent first), and 4 context tokens (trump, phase, tricks,
points) each with its own value embedding. Cards share one `Embedding(53, 128)`
(52 cards + PAD). Learned type embeddings separate the four groups; learned
positional embeddings mark order-sensitive slots. A pre-LayerNorm
`TransformerEncoder` (d_model=128, 4 heads, 3 layers, FFN 512, GELU, dropout
0) runs bidirectional attention with a padding mask, and a single learned
query attention-pools the real tokens. ~640k parameters, far under the 2M cap.

`HokmMaskablePolicy` binds the extractor to sb3-contrib's
`MaskableActorCriticPolicy` with the default `net_arch=[]` so the pooled
features feed the actor and value heads directly; orthogonal init runs with
gain 0.01 on the action head and 1.0 on the value head. Logit masking is
handled entirely by `MaskablePPO`.

## Training

Documented once the training CLI lands.

## Web UI

Documented once the web UI lands.

## Docker

Documented once the container lands.

## Results

To be filled by the training milestone: win rates against the evaluation
gauntlet, training curves, and the final hyperparameters.

## Configuration

All deployment-specific values are environment variables documented in
[`.env.example`](.env.example) — GPU device id, web UI port, served model path,
and the visual QA reviewer endpoint. Defaults live only there; per-machine
values live in a gitignored `.env`.

## Known limitations

To be documented as the project matures.
