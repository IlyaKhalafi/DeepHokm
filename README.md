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

**Self-play.** Every environment re-draws its opponents at each `reset()` as
one team-coherent pair rather than four independent seats: 45% the latest
snapshot, 20% uniform over the retained pool, 25% the scripted `GreedyPolicy`,
10% a fresh random policy — the same draw fills both seats of one team, so a
table is never a mix of "one partner + two mismatched opponents". Snapshots
are written atomically into the run's `opponents/` directory and the workers
rescan that directory, because the environments run in forked subprocesses
and a pool held as an in-process Python object would never reach them.
Snapshot opponents sample rather than take their argmax, so the learner meets
varied lines instead of one frozen script. `GreedyPolicy` was folded into the
primary mix (rather than only appearing at evaluation time) specifically
because the first completed run's win rate against it never improved — see
*Results* for that experiment.

**Evaluation.** The gauntlet plays the learner in one seat against a table of
the named opponent — the scripted `GreedyPolicy` also fills the learner's own
partner seat, which is the honest reading of "win rate against X". Rungs:
random policy, greedy baseline, first snapshot, latest snapshot; 100 games
each, every `--eval-every` environment steps, sharded across worker
processes. Figures render with `make plot`.

Hyperparameters (logged in every config dump): learning_rate 3e-4 decaying
linearly to a 3e-5 floor, n_steps 256 per worker, batch_size 1024, n_epochs 4,
target_kl 0.02, gamma 0.997, gae_lambda 0.95, clip_range 0.2, ent_coef 0.01,
vf_coef 0.5, max_grad_norm 0.5, trick_reward 0.0, hand_reward 0.15.

`gamma` and `gae_lambda` are the reference values — an earlier run trained at
`gamma=0.95` (paired with `trick_reward=0.05`) on the theory that a +/-1 match
outcome ~150 steps away carries almost no gradient at the reference
`gamma=0.997`. That theory was checked directly and is backwards:
`0.997**150 ≈ 0.64` of the reward is retained over that horizon, versus
`0.95**150 ≈ 0.0005` — 0.95 is the value that destroys almost all long-horizon
credit, not 0.997. `trick_reward` is retired in favor of `hand_reward`: it
pays the same amount whether a trick decided a close hand or mopped up one
already settled, which can teach the policy to chase tricks that no longer
matter; `hand_reward` only pays out when a hand's outcome is actually decided.
`n_steps`, `batch_size`, `n_epochs`, `target_kl` and the learning-rate floor
still deviate from the reference set, forced by evidence recorded during
development (the vec-env rollout-size rationale in `train.py:DEFAULT_N_STEPS`,
and a KL blowup during the first full run that motivated fewer, larger-batch
epochs plus a hard trust-region cap — see `train.py:hyperparameters` and
`REVIEW_LOG.local.md`). `ent_coef` is not annealed: sb3-contrib's
`MaskablePPO` consumes it as a plain float in the loss, not through a
schedule like `learning_rate`, so annealing it would need a training-loop
patch rather than a constructor argument.

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
  --env-file .env deephokm-web:latest
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
a no-op override rather than a requirement). The container never reserves a
GPU device at all: `ServedPolicy` always runs inference on CPU (batch-1
calls are faster there than a GPU round-trip; see
`src/deephokm/webui/serving.py`), so a mandatory device reservation would
only break `docker compose up` on a machine with no nvidia container
runtime for no benefit. `DEEPHOKM_GPU_DEVICE_ID` still pins bare-metal
training and the dev web UI server (`make train`, `make webui`) to GPU 6 via
`CUDA_VISIBLE_DEVICES`; it plays no role in the container. The container
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

Completed run: 8M steps (~3.4 hours on GPU 6), seed 0, 32 parallel
environments with seat rotation, `trick_reward=0.05`, `gamma=0.95`, gauntlet
every 250k steps against random play, the scripted greedy baseline, the
earliest snapshot, and the latest snapshot (100 games each). Win-rate curves
are in `figures/gauntlet_win_rates.png` and the core PPO scalars in
`figures/training_scalars.png` (regenerate both with `make plot`); the table
below is the final gauntlet round of the run.

| Opponent        | Win rate (100 games) |
| --------------- | --------------------- |
| Random policy   | 0.70                  |
| Greedy baseline | 0.15                  |
| First snapshot  | 0.58                  |
| Latest snapshot | 0.50                  |

Against the >= 80% (random) / >= 55% (first snapshot) targets: the first
snapshot target is met; the random-policy target is not.
`figures/gauntlet_win_rates.png` shows the full picture: win rate vs. random
climbs from ~0.58 to a ~0.65-0.81 band by 2M steps and stays there for the
rest of the run — real early improvement, then noisy fluctuation around a
plateau that sits mostly below 0.80 rather than a further climb past it. The
0.70 in the table is the literal final round; two nearby rounds against the
same fully-trained 8M-step policy read 0.78 and 0.81 (`GauntletCallback`
runs one round on its normal 250k-step cadence and then one more,
unconditionally, when training ends, so the last two rounds in the log
evaluate the identical policy on two different 100-game samples) — noise
around the same plateau, not evidence the target was actually met. Win rate
vs. the latest snapshot hovering near 0.50 throughout is expected for
self-play: the learner and its most recent past self are closely matched by
construction.

Win rate vs. the scripted greedy baseline never climbed out of a 0.09-0.23
band across the entire run (see the figure), despite random and snapshot
performance clearly improving over the same period. A follow-up experiment
during training narrows down why: loading a 4.5M-step checkpoint and
fine-tuning it for 150k further steps against an opponent mix that always
included `GreedyPolicy` (rather than the main run's random/self-play mix,
which never includes it) moved the win rate against greedy by less than one
percentage point (0.180 -> 0.187, a shift a 150-game sample can't
distinguish from noise) at a properly pinned, non-decaying learning rate.
That rules out the cheapest explanation (the policy has simply never seen
this opponent style) and points at something that needs either much longer
exposure or a redesigned self-play mix from the start of a run, not a late
graft onto an already-converged policy — the next experiment worth running
is a fresh run with `GreedyPolicy` folded into the primary self-play
opponent pool from step zero.

## Configuration

All deployment-specific values are environment variables documented in
[`.env.example`](.env.example) — GPU device id, web UI port, served model
paths, and the visual QA reviewer endpoint. Defaults live only there;
per-machine values live in a gitignored `.env`.

## Known limitations

- The trained model does not reliably beat a disciplined, non-learning
  opponent: win rate against the scripted greedy baseline stayed in a
  0.09-0.23 band for the entire 8M-step run (see *Results*). It plays a
  clearly-better-than-random game (0.70 vs. random, up from ~0.58 early in
  training) but has not learned the deeper tactical play that separates it
  from a simple heuristic. The self-play population's opponent mix (60%
  latest snapshot, 30% pool, 10% random) never includes a disciplined,
  non-self-play style during training, only at evaluation time; a targeted
  fine-tune experiment (see *Results*) suggests this is not a quick fix.
- The results above were produced under the earlier `gamma=0.95` /
  `trick_reward=0.05` configuration, which (see *Training*) rested on a
  mathematically backwards discounting argument. The current default
  configuration (`gamma=0.997`, `hand_reward=0.15`, `trick_reward=0.0`, the
  team-coherent self-play mix with `GreedyPolicy` folded in from step zero,
  and the revised PPO schedule) has not yet completed a full run at the time
  of writing; this section will be replaced with that run's numbers once it
  finishes. `--trick-reward 0.05 --hand-reward 0.0` reproduces the retired
  per-trick shaping for comparison.
- No kot/bustom scoring; the web UI plays a single match per game id with no
  reconnection.
- The evaluation gauntlet reports deterministic-policy win rates; stochastic
  evaluation may differ.
