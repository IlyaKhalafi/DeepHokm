# Methods and results

DeepHokm combines a rules engine, public-information policies, and a small
action-value network. A match ends when a team wins seven hands; a hand ends
when a team wins seven tricks. See [Rules](RULES.md) for the exact variant.

## Information available to a policy

A policy may use its own hand, trump, the current trick, public card history,
scores, and suit voids inferred from failures to follow suit. It must not
inspect opponents' actual hands.

Search samples complete deals consistent with that information. These are
hypothetical worlds, not observations of hidden cards. Tests check that changing
actual hidden hands cannot change the public network inputs.

## Policies

| Policy | Decision method | Tradeoff |
|---|---|---|
| Greedy | Public card tracking and tactical rules | Fast, no training needed |
| Legal depth search | Determinized rollouts with statistical action comparisons | Strength costs CPU time |
| Pure Q-net | Highest-scoring legal action; greedy trump declaration | NumPy-only inference |
| Q-hybrid | Network ordering plus determinized search | More compute than pure inference |

### K notation: training versus live search

This project uses K for a count of sampled determinized worlds, but K appears
at two separate stages:

- **Teacher K=6144** is an offline data-generation budget. For every recorded
  decision, the teacher samples worlds to estimate legal-action values. Those
  estimates become training labels. This cost ends when the dataset is built.
- **Verification K=384** is the default online Q-hybrid budget. At each
  non-forced decision, sequential elimination samples fresh worlds, scores the
  surviving legal actions in the same worlds, and removes actions only when
  the paired evidence puts them sufficiently behind.

In the current six-round elimination implementation, K=384 means up to 64 new
shared worlds per round, or 384 shared worlds across elimination when all six
rounds run. If the best survivor differs from greedy, a separate K=384 paired
sign test verifies that final choice. Each world may therefore execute several
rollouts, one for every surviving action; K is a sampling budget, not the total
number of simulated action rollouts or a latency in milliseconds.

A network trained from K=6144 teacher labels can be served without live search,
but that is Q-pure inference, not Q-hybrid. Q-hybrid is the trained network plus
the online verification stage. Raising the online K spends more time on the
current move; it does not improve or retrain the stored weights. The deployment
controls and latency measurements are documented in
[Deployment](DEPLOYMENT.md#two-different-k-values).

The greedy policy conserves winning cards when it cannot win a trick, avoids
unnecessary partner overcalls, remembers proved suit voids, and handles
position-specific tactics. In third hand, it considers whether the fourth
player can still beat the partnership; blindly playing high is not always
correct. The public endgame solver can reason over the cards that remain
consistent with public information.

The search is a practical approximation. Determinization and rollout-policy
bias remain limitations; sampling more worlds does not remove every source
of error.

## Network and training objective

RankCNN treats the suit axis symmetrically and convolves along the ordered
rank axis. The original input has 14 card planes and 10 scalar values.
Additional feature modes explicitly encode proved voids, trick position,
current leadership, remaining actors, and publicly inferable card strength.

Training uses legal-action soft targets derived from search values. Decisions
where every legal action is equally optimal are excluded from the decisive
accuracy metric. Near-ties receive similar probability, avoiding an arbitrary
single-card classification target.

Validation membership is deterministic at the match level. New shards cannot
move an existing match between partitions, and duplicate match seeds are
rejected. Early stopping retains the best validation checkpoint; an optional
linear warmup/cosine schedule controls the learning rate.

Exports contain NumPy weights and a feature contract. Expanded input widths
are not sufficient to identify a feature mode: a matched zero-feature control
has the same width as its corresponding treatment. Serving checks metadata,
channel counts, and hashes before using expanded weights.

See [Training](TRAINING.md) for reproducible commands and
[Architecture](ARCHITECTURE.md) for implementation details.

## Shipped-model benchmarks

These results describe the bundled 14-plane checkpoint and the greedy baseline
used in those experiments. They are not measurements against every subsequent
version of the greedy policy.

| Policy | Wins vs greedy | Matches | Evaluation |
|---|---|---|---|
| Pure network | 66.25% | 80 | Held-out |
| Network + elimination search, K=3072 | 85.5% | 399 | Held-out |
| Search only, K=3072 | 85.2% | 128 | Development |
| Network + search, K=48 | 71.3% | 122 | Development |
| Search only, K=48 | 57.0% | 200 | Development |

Reported 95% intervals were [55.9%, 76.6%] for pure inference and
[82.0%, 88.9%] for the held-out search-paired result. Historical pure-inference
median latency was 18 ms; live search adds seconds and is not part of that
latency claim. Hardware and search budgets affect these costs.

The web UI defaults to pure inference. Search is an explicit deployment option;
see [Deployment](DEPLOYMENT.md).

## Public-feature experiments

These development results use K=6144 teacher labels. Validation agreement
means choosing an action in the teacher's optimal set, not winning a game.

| Matched experiment | Control agreement | Added-feature agreement | Seeds |
|---|---|---|---|
| Public trick context | 66.64% | 69.15% | 3 |
| Public strength beyond trick context | About 68.8% | About 70.6% | 3 |

The context experiment used the same 243 games in both arms and improved
agreement in all three initialization seeds. Mean teacher-value regret
decreased by approximately 24%. The strength experiment likewise used
equal-width models, the same games, and paired seeds. Both experiments use
validation-selected checkpoints; neither is an untouched final-test result.

Expanded-data strength retraining reached 70.49% decisive validation agreement.
It does not replace the bundled model automatically.

A separate paired match benchmark tested the same K=6144-trained context
weights in both serving configurations against the updated greedy policy:

| Serving configuration | Online K | Wins | Matches | Median | p95 |
|---|---:|---:|---:|---:|---:|
| Q-pure | 0 | 69 | 100 | 11 ms | 20 ms |
| Q-hybrid | 384 | 86 | 100 | 2.43 s | 8.05 s |

Deals were played on both team sides; the paired bootstrap intervals were
[58%, 80%] and [78%, 93%]. These are development measurements, not updates to
the shipped-model table. The comparison isolates the online verification
stage: both rows use the same weights trained from teacher K=6144 labels.

No completed head-to-head evaluation establishes that the K=6144-trained
student is superior to legal search. Partial runs are not used to claim a final
win rate.

## What did not work

- Self-play PPO and behavioral cloning did not consistently beat the scripted
  baseline in the measured configurations.
- Raw-value regression and single-best-action agreement obscured the many
  tied or nearly tied search labels.
- Restricting search to the network's top candidates discarded useful actions.
  Ordering and statistically justified elimination worked better.
- Larger models and more epochs alone did not resolve the validation gap.
  Confidence-weighted training also failed to improve the matched experiment.

These are observations about the tested setups, not impossibility results.
Teacher agreement, teacher repeatability, and match win rate are different
metrics; none alone establishes a universal ceiling on a student's strength.

## Reproducibility and repository scope

The repository contains reusable code, tests, documentation, and the bundled
inference weights. Generated datasets, run checkpoints, benchmark logs,
machine-specific launchers, and exploratory scratch files are local artifacts,
not source releases. Keep new experimental results separate from the shipped
model until their evaluation protocol is fixed and completed.
