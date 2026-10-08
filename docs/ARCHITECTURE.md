# Architecture

DeepHokm separates game rules, public information, policy decisions, model
inference, and delivery. This keeps the rules engine authoritative and makes
policy experiments replaceable without duplicating Hokm logic.

## Repository boundaries

| Package | Responsibility |
|---|---|
| `deephokm.rules` | Dealing, turn order, legality, trick resolution, scoring, and match state |
| `deephokm.env` | Gymnasium adapter, observations, rewards, and action masks |
| `deephokm.policies` | Greedy, search, Q-pure, Q-hybrid, and public endgame policies |
| `deephokm.nn` | Feature contracts, RankCNN, NumPy inference, and legacy RL components |
| `deephokm.training` | Self-play and behavioral-cloning workflows |
| `deephokm.webui` | FastAPI endpoints, game lifecycle, model loading, and static UI |
| `scripts` | Reproducible data generation, evaluation, plotting, and QA commands |

The dependency direction is deliberate: policies call the rules engine, the
environment adapts it, and the web application composes both. The frontend
never reimplements legality or scoring.

## Rules and state transitions

`HokmEngine` owns one `MatchState` and is the sole authority for the next
seat and legal actions. A transition follows one path:

1. derive the current seat and legal action set;
2. apply a trump declaration or card play;
3. resolve a completed trick;
4. score and redeal when a hand ends;
5. return an `ActionOutcome` for external consumers.

The pure modules in `deephokm.rules` contain the individual operations.
Keeping legality and trick resolution outside the environment and UI prevents
the same rule from acquiring subtly different implementations.

The rollout hot path may pass an already-derived legal list to the engine and
skip allocating an outcome for an ordinary mid-trick play. That private path is
restricted to simulations; normal callers retain full validation and event
objects. Regression tests compare fast and normal transitions across seeded
games.

## Public-information boundary

The observation contains the acting seat hand plus public facts: trump, phase,
current trick, played-card history, trick totals, match points, and seat. The
action mask is derived from the same live engine state.

Search never reads the true hidden hands. It samples complete hypothetical
deals consistent with:

- the acting seat hand;
- cards already played;
- cards known to remain unseen;
- suit voids publicly proved when a seat fails to follow suit.

Tests mutate opponent hands while holding public state fixed and verify that
public features and policy inputs remain unchanged. This is the central
information-fairness invariant.

## Action-value network

The shipped action-value model is `RankCNN`. Cards form a
`suit x rank` grid. Convolutions run along ordered ranks while weights are
shared across suits, so the architecture does not invent adjacency between
clubs, diamonds, hearts, and spades.

The original model consumes 14 public card planes plus scalar match context.
Expanded feature contracts can add proved voids, trick position, leadership,
remaining actors, and public strength. Each expanded archive has adjacent
metadata that records its feature mode, training K, and weight hash. Serving
validates that contract before accepting the weights.

Training uses PyTorch. Deployment exports the learned parameters to `.npz`
and `NumpyQNet` implements the same forward pass with NumPy. Parity tests
compare both implementations, and a subprocess test blocks Torch imports while
the NumPy policy plays a match.

The older transformer and MaskablePPO modules remain available for reproducing
the experiments documented in [Methods and results](METHODS_AND_RESULTS.md).
They are not the default serving path.

## Policy composition

The main policy layers are intentionally small and composable:

- `GreedyPolicy` provides a deterministic public-information baseline and
  rollout policy.
- `PureQNetPolicy` scores all actions once, masks illegal actions, and picks
  the best legal value.
- `NumpyHybridPolicy` uses network values to order candidates, then evaluates
  them in shared determinized worlds with statistically justified elimination.
- `SearchServedPolicy` adapts either Q policy to the web policy interface and
  rebuilds public suit voids from the engine history.

Fast mode uses Q-pure inference. Hard mode loads the separate network trained
from K=6144 teacher data and adds online verification search. Offline teacher K
and online verification K are separate budgets; [Deployment](DEPLOYMENT.md)
defines both and reports latency.

## Web serving

The FastAPI process loads and validates weight archives once. Each game gets a
fresh policy object because a search policy holds a reference to its live
engine, while immutable weight arrays and the optional worker pool are shared.
`GameStore` owns game records and per-game locks, so concurrent requests
cannot observe a half-applied transition.

The server returns only viewer-safe state. Human actions and spectate steps
both pass through the authoritative engine. Static JavaScript renders state and
sends action identifiers; it does not decide whether a card is legal.

The container runs as an unprivileged user, mounts replacement weights
read-only, and keeps temporary files in memory. NumPy inference runs on CPU, so
the deployment does not require an NVIDIA runtime.

## Quality gates

`make lint` checks Ruff formatting, Ruff lint rules, and strict mypy types.
`make test` runs the complete pytest suite with branch coverage. Tests cover
rules, legal masks, seeded rollouts, public-information isolation, PyTorch to
NumPy parity, web API behavior, and browser-level UI flows.

See [Training](TRAINING.md) for data and fitting workflows and
[Deployment](DEPLOYMENT.md) for runtime configuration.
