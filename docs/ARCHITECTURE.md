# Architecture

The environment, the network, and the search that uses it.

- `src/deephokm/cards.py` — card, suit, rank and deck primitives.
- `src/deephokm/rules/` — pure rules engine: legality, trick resolution,
  scoring, hakem rotation.
- `src/deephokm/env/` — `HokmEnv` (Gymnasium `Hokm-v0`) with action masking.
- `src/deephokm/policies/` — the `HokmPolicy` protocol and scripted baselines.
- `src/deephokm/nn/` — card tokenizer, transformer feature extractor, policy
  wiring for sb3-contrib `MaskablePPO`.
- `src/deephokm/training/` — vectorized self-play pipeline, opponent snapshot
  pool, parallel evaluation gauntlet, CLI.
- `src/deephokm/webui/` — FastAPI app serving a static frontend and the REST
  API; the server owns all game state.
- `scripts/` — environment benchmark, result plotting, Playwright smoke test,
  visual QA harness.

### Environment

`Hokm-v0` is a single-seat Gymnasium environment: the learner plays one seat,
three opponent policies (any `HokmPolicy` implementation) play the rest inside
`step()`. Observations are a Dict of binary card vectors plus small integer
scalars (`hand`, `seen`, `trick`, `trick_play`, `history`, `trump`, `phase`,
`tricks_won`, `game_points`, `seat` — see `src/deephokm/env/spaces.py`); the
action space is `Discrete(56)` (52 card plays + 4 trump declarations) with
legality masks available via `env.action_masks()` and `info["action_mask"]`.
Illegal actions raise `ValueError` rather than being resampled. Rewards are
sparse by default: +1 for winning the match, -1 for losing; optional shaping
via the `trick_reward` (per-trick) and `hand_reward` (per-hand) constructor
arguments — see *Training* for which one the CLI actually uses.

`history` carries the current hand's completed tricks as card ids in reverse
play order (most recent first, `-1` padded, 48 slots). It is redundant with
`seen` as a *set* but not as a *sequence*: a binary vector records which cards
have gone but not when, and reading a trick-taking position needs the order.
Everything in it is public information — the property test in
`tests/test_hokm_env.py` asserts that no observation field ever exposes a card
that is not in the acting seat's own hand or already played.

One note on `gymnasium.utils.env_checker.check_env`: its determinism probe
samples an action *before* its final `reset()`, so for any environment whose
legal-action set changes phase across resets (trump call vs. card play) the
probed action can belong to the previous episode's hand. The suite runs
`check_env` through a small documented shim (`CheckerProbeShim` in
`tests/test_hokm_env.py`) that resamples only that blind probe; every other
checker assertion runs against the bare environment.

Benchmark: `make bench` measures mask-respecting random play
(~10k learner steps/s per worker on the development machine).

### Network

`HokmTransformerExtractor` tokenizes the Dict observation into a fixed
70-token layout: up to 13 hand tokens (no positional embedding — the hand is
a set), up to 4 trick tokens in play order, up to 48 history tokens in
recency order, and 5 context tokens (trump, phase, tricks, points, seat)
each with its own value embedding. The history group spans the whole hand
rather than the 13 most recent plays: which cards are gone is the central
read in a trick-taking game, and 48 tokens cost nothing at this model size. Cards share one `Embedding(53, 128)`
(52 cards + PAD). Learned type embeddings separate the four groups; learned
positional embeddings mark order-sensitive slots. A pre-LayerNorm
`TransformerEncoder` (d_model=128, 4 heads, 3 layers, FFN 512, GELU, dropout
0) runs bidirectional attention with a padding mask, and a single learned
query attention-pools the real tokens. ~640k parameters, far under the 2M cap.

`HokmMaskablePolicy` binds the extractor to sb3-contrib's
`MaskableActorCriticPolicy` with `net_arch=[]` so the pooled features feed the
actor and value heads directly; orthogonal init runs with gain 0.01 on the
action head and 1.0 on the value head. Logit masking is handled entirely by
`MaskablePPO`.


## Search

`src/deephokm/policies/search.py` adds an optional test-time search step on
top of a trained policy for card-play decisions: root-sampled
imperfect-information Monte Carlo (IIMC), the imperfect-information
analogue of the search a chess engine runs against a learned evaluator.
Since opponents' hands are hidden, it samples plausible full deals
consistent with public play (respecting suit voids), then evaluates the
policy's own top candidate actions by rolling each forward with the same
network before picking the best-scoring one. See the module docstring for
the full design and its information-fairness invariant.

```bash
uv run python scripts/evaluate_search.py --model checkpoints/m8_main/final.zip \
  --opponent greedy --n-seeds 25 --n-samples 4 --top-k 2
```

A first 30-paired-game evaluation at `n_samples=4, top_k=2` looked
promising (19/30 vs. the plain policy, 17/30 vs. greedy), but a larger
follow-up (70 and 72 paired games respectively, combining both batches)
did not hold up: 37/70 (0.53) vs. plain — essentially a coin flip — and
32/72 (0.44) vs. greedy — behind the plain policy. At this setting, search
does not reliably improve on the trained network and plausibly hurts
slightly against a disciplined opponent; see `REVIEW_LOG.local.md` for two
untested hypotheses why. Treat this as an unproven experiment, not a
shipped improvement.

