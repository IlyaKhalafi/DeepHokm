# Training

How the label data is generated and how the network is fitted to it.
For why the objective looks the way it does, see METHODS_AND_RESULTS.md.

```bash
make train                                   # defaults: 1M steps, 8 envs
make train ARGS="--total-timesteps 5000000 --n-envs 8 --seed 0 \
  --out-dir checkpoints/run1 --tensorboard-log logs/tb"
```

The CLI dumps a reproducible `config.json` next to the checkpoints and
resumes via `--resume <checkpoint.zip>`.

**Behavioral-cloning warm start.** Two straight from-scratch self-play runs
(see *Results*) plateaued at essentially the same win rate against
`GreedyPolicy` regardless of gamma, reward shaping, or that opponent's
training-time share — evidence that RL from a random initialization may
simply need a better starting point against a disciplined, non-learning
style, not just more self-play tuning. `deephokm.training.behavioral_cloning`
collects a dataset from `GreedyPolicy`-vs-`GreedyPolicy`-vs-`GreedyPolicy`-vs-
`GreedyPolicy` rollouts (every seat played by a fresh, independently seeded
instance) and fits a fresh policy to imitate it by supervised cross-entropy,
saving a normal `MaskablePPO` checkpoint:

```bash
make pretrain-bc ARGS="--n-matches 3000 --epochs 8 --out-path checkpoints/bc_pretrained.zip"
make train ARGS="--resume checkpoints/bc_pretrained.zip --total-timesteps 8000000 --n-envs 32 \
  --out-dir checkpoints/run2 --tensorboard-log logs/tb"
```

The saved checkpoint's own hyperparameters (from the throwaway model used
only to hold a correctly-constructed policy) are discarded on `--resume`:
`MaskablePPO.load()` applies the loaded weights on top of the real training
run's hyperparameters, so PPO fine-tuning starts from a policy that already
plays a disciplined game rather than from random initialization.

**Self-play.** Every environment re-draws its opponents at each `reset()` as
one team-coherent pair rather than four independent seats: 55% the latest
snapshot, 25% uniform over the retained pool, 10% the scripted `GreedyPolicy`,
10% a fresh random policy — the same draw fills both seats of one team, so a
table is never a mix of "one partner + two mismatched opponents". Snapshots
are written atomically into the run's `opponents/` directory and the workers
rescan that directory, because the environments run in forked subprocesses
and a pool held as an in-process Python object would never reach them.
Snapshot opponents sample rather than take their argmax, so the learner meets
varied lines instead of one frozen script. `GreedyPolicy` is folded into the
primary mix (rather than only appearing at evaluation time) specifically
because the first completed run's win rate against it never improved — see
*Results* for that experiment. Its share started at 25% in an aborted
follow-up run and was cut to 10%: at 25%, gauntlet win rate against *both*
random and greedy sat flat (no upward trend across the first 4.75M of 8M
steps, worse than the original run's early ramp), consistent with a quarter
of every training table being a disciplined, non-exploitable opponent
diluting the learning signal before the policy has any competence at all.
10% keeps some greedy exposure from step zero without dominating the
curriculum the way 25% did.

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

