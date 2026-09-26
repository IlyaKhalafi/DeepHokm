# DeepHokm

**Hokm** (حکم), the Persian trick-taking card game, played by a small
convolutional action-value network guiding a determinized search.

Beats a scripted greedy opponent in **85.5%** of matches (399 held-out matches,
95% CI [0.820, 0.889]). Runs on **numpy alone** at **30 ms per decision** —
PyTorch is a development dependency, not a runtime one.

![DeepHokm web UI](docs/media/demo.gif)

## Play it

```bash
uv sync
make webui        # http://localhost:8025
```

Play a seat against the model, or watch it play itself. The engine that serves
the UI is the same one used for evaluation.

## How it works, in one paragraph

Each decision is scored by sampling **determinized worlds** — full deals of the
unseen cards consistent with everything public: your hand, every card played,
and the suit voids revealed when a player fails to follow suit. Every legal card
is played out in each sampled world, and a card is preferred over the scripted
baseline only when it wins an exact one-sided sign test, so the policy deviates
on evidence rather than on noise. The network's job is to make that search
cheap: a **suit-equivariant CNN** (6.4M parameters, `1x3` convolutions along the
*rank* axis with weights shared across all four suits, since the rules are
symmetric under relabelling suits but ranks are ordered) scores all 52 cards in
one pass and orders the candidates, and actions are eliminated only once the
evidence rules them out — never pruned before being scored. It is trained by
supervised distillation from the search itself: 47,599 decisions labelled by a
much stronger search, fitted with a soft-target cross-entropy whose temperature
matches the teacher's sampling error, restricted to the decisions where the
teacher actually has a preference.

## Results

| policy | win rate vs greedy | matches | cost |
|---|---|---|---|
| Scripted greedy baseline | 0.500 | — | — |
| Search alone, K=48 | 0.570 | 200 † | 1/64 of the budget below |
| Network + search, K=48 | 0.713 | 122 † | same budget |
| Search alone, K=3072 | 0.852 | 128 † | no network |
| **Network + elimination search, K=3072** | **0.855** | **399 ‡** | 30 ms network + search |

‡ held-out seeds, disjoint from everything used to choose the model or its
settings; 95% CI [0.820, 0.889].
† measured during development on the seed sets used for tuning, so these are
indicative rather than held-out.

The network does not raise the ceiling — it reaches the search's own level and
makes small-budget search substantially stronger (0.713 against 0.570 at
K=48). Why no
policy in this family can exceed the search it wraps is derived in
[the methods write-up](docs/METHODS_AND_RESULTS.md#19-why-the-hybrid-cannot-beat-the-search-it-wraps).

## Documentation

| | |
|---|---|
| [Methods and results](docs/METHODS_AND_RESULTS.md) | Every idea tried, what it measured, and why the design ended up here — including the approaches that failed |
| [Architecture](docs/ARCHITECTURE.md) | Environment, observation space, network, search |
| [Training](docs/TRAINING.md) | Label generation and the fitting procedure |
| [Rules](docs/RULES.md) | The exact Hokm variant implemented |
| [Deployment](docs/DEPLOYMENT.md) | Web UI, Docker, configuration |
| [Result tables](docs/RESULTS_TABLES.md) | Raw measurements from earlier phases |

## What did not work

Recorded because the negative results took most of the effort and are the
reason the final design looks as it does:

- **Self-play reinforcement learning** plateaued at the level of a greedy
  clone across every variant tried. A one-ply policy cannot represent lines
  that pay off several tricks later.
- **Architecture search moved nothing.** Six variants from 377k to 8M
  parameters landed in one band; a structure-free MLP matched the transformer.
- **Agreement with the teacher was a broken metric.** 60% of decisions have
  tied best actions and 32% have no real choice at all, so most of the metric
  was grading coin flips.
- **Pruning to the network's favourites lost to plain search**, because an
  action that is never scored can never be chosen.

## Development

```bash
make test         # pytest
make lint         # ruff + mypy
make bench        # environment throughput
```

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).
