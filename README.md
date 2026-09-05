# DeepHokm

Reinforcement learning for **Hokm** (حکم), the Persian trick-taking card game: a
fully tested rules engine, a Gymnasium environment with action masking, a
transformer policy trained by self-play `MaskablePPO`, a web UI for playing
against the trained model, and Docker packaging.

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
sparse by default: +1 for winning the match, -1 for losing; optional per-trick
shaping via the `trick_reward` constructor argument.

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

## Quickstart

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
make test
make lint
```

Key resolved versions (see `uv.lock` for the full set): gymnasium 1.3.0,
stable-baselines3 2.9.0, sb3-contrib 2.9.0, torch 2.14.0 (CUDA 13.0 wheels),
fastapi 0.141.1, numpy 2.4.6.

## Training

```bash
make train                                   # defaults: 1M steps, 8 envs
make train ARGS="--total-timesteps 5000000 --n-envs 8 --seed 0 \
  --out-dir checkpoints/run1 --tensorboard-log logs/tb"
```

The CLI dumps a reproducible `config.json` next to the checkpoints and
resumes via `--resume <checkpoint.zip>`.

**Self-play.** Every environment re-draws its three opponent seats at each
`reset()`: 60% the latest snapshot, 30% uniform over the retained pool, 10% a
fresh random policy. Snapshots are written atomically into the run's
`opponents/` directory and the workers rescan that directory, because the
environments run in forked subprocesses and a pool held as an in-process
Python object would never reach them. Snapshot opponents sample rather than
take their argmax, so the learner meets varied lines instead of one frozen
script.

**Evaluation.** The gauntlet plays the learner in one seat against a table of
the named opponent — the scripted `GreedyPolicy` also fills the learner's own
partner seat, which is the honest reading of "win rate against X". Rungs:
random policy, greedy baseline, first snapshot, latest snapshot; 100 games
each, every `--eval-every` environment steps, sharded across worker
processes. Figures render with `make plot`.

Hyperparameters (logged in every config dump): learning_rate 3e-4 decaying
linearly to 0, n_steps 256 per worker, batch_size 512, n_epochs 10,
target_kl 0.03, gamma 0.95, gae_lambda 0.95, clip_range 0.2, ent_coef 0.01,
vf_coef 0.5, max_grad_norm 0.5, trick_reward 0.05. Four values deviate from
the reference set (learning-rate decay, `target_kl`, `n_steps`, and the
`gamma`/`trick_reward` pair) — each forced by evidence recorded during
development; see *Results* and `REVIEW_LOG.local.md` for the comparison that
produced them. `--gamma 0.997 --trick-reward 0.0` restores the original
sparse-reward reference config for anyone who wants to reproduce it.

## Web UI

```bash
make webui            # serves on ${DEEPHOKM_PORT}
```

Two modes: play seat 0 against the trained policy, or spectate AI-vs-AI one
decision at a time (with auto-play). The UI always shows your hand, the
table with seat attribution and trick order, trump, per-team tricks and game
points, whose turn it is, the current phase, and explicit card counts per
seat; illegal cards are visibly disabled and rejected server-side.

## Docker

```bash
cp .env.example .env    # edit per machine; set DEEPHOKM_GPU_DEVICE_ID
docker compose up --build -d
curl -fs "http://localhost:${DEEPHOKM_PORT}/health"
```

or plain docker (export the configuration first — `--env-file` only feeds the
container's environment, not the shell expansions in these flags):

```bash
set -a; . ./.env; set +a
docker build -t deephokm-web:latest .
docker run --rm -p "${DEEPHOKM_PORT}:${DEEPHOKM_PORT}" --tmpfs /tmp \
  --gpus "device=${DEEPHOKM_GPU_DEVICE_ID}" --env-file .env deephokm-web:latest
```

The image is a multi-stage uv build. The trained checkpoint is baked in from
`checkpoints/final.zip` — copy (or symlink) the checkpoint you want to serve
there before building:

```bash
cp checkpoints/<run>/checkpoints/ppo_<step>_steps.zip checkpoints/final.zip
```

At run time a read-only bind mount can override the baked checkpoint: set
`DEEPHOKM_MODEL_PATH` in `.env` (absolute or relative to the compose file;
it defaults to the repository's own `checkpoints/final.zip`, so the mount is
a no-op override rather than a requirement). All deployment values (GPU
device id, port, model path) come from the environment — changing
`DEEPHOKM_GPU_DEVICE_ID` in `.env` is the only thing needed to re-pin the GPU.
The container serves on CPU by default (batch-1 inference is faster there,
see `src/deephokm/webui/serving.py`), so the GPU reservation is optional. It
runs as an unprivileged user with a RAM-backed `/tmp` and writes nothing
durable, so the only host state it needs is the mounted checkpoint.

## Results

The first full run (5M steps, sparse reward, gamma 0.997) never learned:
`SubprocVecEnv` workers are forked once at startup, and the opponent pool was
an in-process Python object living in the training process — the forked
workers kept the (empty) pool from the moment they were created, so the
entire run was PPO against fixed random opponents rather than self-play, and
a +/-1 match outcome ~150 steps away carries almost no gradient at gamma
0.997 regardless. Both were fixed: opponents are now drawn from a snapshot
*directory* that every worker rescans on each `reset()` (the one channel a
forked process actually shares with its parent), and the reference
hyperparameters gained `target_kl`, a decaying learning rate, and a shorter,
trick-reward-shaped discount (see `train.py:hyperparameters` and
`REVIEW_LOG.local.md` for the controlled comparison behind that choice). A
behaviour-cloning probe against the scripted `GreedyPolicy` (0.79 test
accuracy vs. a 0.46 random-legal baseline) confirmed the network can
represent good play, isolating the original failure to training, not
architecture.

Current run: 8M steps, seed 0, 32 parallel environments with seat rotation,
`trick_reward=0.05`, `gamma=0.95`, gauntlet every 250k steps against random
play, the scripted greedy baseline, the earliest snapshot, and the latest
snapshot (100 games each). Final numbers and the win-rate curves are in
`figures/gauntlet_win_rates.png` (regenerate with `make plot`); the table
below is the final gauntlet round of the run.

| Opponent        | Win rate (100 games) |
| --------------- | -------------------- |
| Random policy   | see figures          |
| Greedy baseline | see figures          |
| First snapshot  | see figures          |
| Latest snapshot | see figures          |

See *Known limitations* for the honest read on where this run landed.

## Configuration

All deployment-specific values are environment variables documented in
[`.env.example`](.env.example) — GPU device id, web UI port, served model
paths, and the visual QA reviewer endpoint. Defaults live only there;
per-machine values live in a gitignored `.env`.

## Known limitations

- Trick-reward shaping trades off long-horizon match strategy for
  learnability: `gamma=0.95` discounts the sparse +/-1 match outcome almost
  to nothing by the time a hand is decided, so the policy optimizes hand-level
  and trick-level play rather than match-level strategy (e.g. deliberately
  losing a hand to keep the hakem). `--gamma 0.997 --trick-reward 0.0`
  restores the original sparse-reward objective for anyone who wants to train
  toward match-level strategy directly, at the cost of a much larger step
  budget.
- No kot/bustom scoring; the web UI plays a single match per game id with no
  reconnection.
- The evaluation gauntlet reports deterministic-policy win rates; stochastic
  evaluation may differ.
