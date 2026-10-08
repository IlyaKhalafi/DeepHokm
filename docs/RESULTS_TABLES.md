# Result tables

Raw measurements from the earlier phases of the project. The narrative
explaining what was tried, what failed, and why is in METHODS_AND_RESULTS.md.

The first full run (5M steps, sparse reward, gamma 0.997) never learned:
`SubprocVecEnv` workers are forked once at startup, and the opponent pool was
an in-process Python object living in the training process. The forked workers
kept the (empty) pool from the moment they were created, so the
entire run was PPO against fixed random opponents rather than self-play, and
a long-horizon credit-assignment problem was also suspected. The latter
explanation was incorrect: gamma 0.997 retains about 64% of a reward after
150 steps. Opponents are now drawn from a snapshot
*directory* that every worker rescans on each `reset()` (the one channel a
forked process actually shares with its parent), and the reference
hyperparameters gained `target_kl`, a decaying learning rate, and a shorter,
trick-reward-shaped discount in the initial follow-up; the later run restored
gamma 0.997 (see `train.py:hyperparameters`). A
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
rest of the run. This shows real early improvement followed by noisy
fluctuation around a plateau that sits mostly below 0.80. The
0.70 in the table is the literal final round; two nearby rounds against the
same fully-trained 8M-step policy read 0.78 and 0.81 (`GauntletCallback`
runs one round on its normal 250k-step cadence and then one more,
unconditionally, when training ends, so the last two rounds in the log
evaluate the identical policy on two different 100-game samples). This is
noise around the same plateau, not evidence the target was actually met. Win rate
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
graft onto an already-converged policy.

**Second completed PPO run (historical).** 8M steps (~3.4 hours on GPU 6),
seed 0, 32 parallel environments, the corrected hyperparameters described
under *Training* (`gamma=0.997`, `hand_reward=0.15`, `trick_reward=0.0`,
team-coherent self-play with `GreedyPolicy` folded into the primary mix).
An intermediate attempt at `P_GREEDY=0.25` was aborted at 4.75M/8M steps
after the gauntlet showed no improving trend against either random or
greedy for the entire first 60% of the run. Cutting `P_GREEDY` to 0.10 and
relaunching produced a stable,
non-stalled run; the table below is the mean of all 33 gauntlet rounds
logged across the full run (`figures/gauntlet_win_rates.png` /
`figures/training_scalars.png`, regenerate with `make plot`).

| Opponent        | Win rate (mean of 33 rounds) | Range       |
| --------------- | ----------------------------- | ----------- |
| Random policy   | 0.61                          | 0.50 - 0.70 |
| Greedy baseline | 0.13                          | 0.06 - 0.24 |
| First snapshot  | 0.61                          | 0.52 - 0.72 |
| Latest snapshot | 0.51                          | 0.43 - 0.60 |

Against the >= 80% (random) / >= 55% (first snapshot) targets: the first
snapshot target is met on average; the random-policy target is not, and the
run never approached it. Win rate vs. random oscillates in a 0.50-0.70
band for the entire 8M steps with no visible upward trend past the first
250k-step evaluation, a materially lower ceiling than the earlier run's
0.65-0.81 plateau. Win rate vs. greedy (mean 0.13, range 0.06-0.24) is
statistically indistinguishable from the earlier run's 0.09-0.23 band:
folding `GreedyPolicy` into training at a 10% share avoided the outright
stall seen at 25%, but did not produce the hoped-for improvement against it
either. Correcting the mathematically-backwards gamma and replacing
trick-level shaping with hand-level shaping did not, on this evidence,
raise the ceiling on win rate against either a random or a disciplined
opponent. It produced a run that trains stably (no stall, no collapse)
but plateaus at a similar level to the run it replaced. RL from a
from-scratch random initialization may simply need far more than 8M steps,
or a materially different self-play curriculum, to discover disciplined
play on its own. See *Training* for the behavioral-cloning warm start
built to test that directly (a supervised fit to `GreedyPolicy` before any
RL, rather than more self-play tuning); its own run and results are not yet
recorded here.


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
