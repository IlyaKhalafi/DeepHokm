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

A separate paired match benchmark of the context model against the updated
greedy policy yielded 69/100 wins for pure inference and 86/100 for Q-hybrid.
Its median decision costs were about 11 ms and 2.43 seconds respectively.
Deals were played on both team sides; the paired bootstrap intervals were
[58%, 80%] and [78%, 93%]. These are separate development measurements, not
updates to the shipped-model table.

No completed K=6144 head-to-head evaluation establishes superiority over legal
search. Partial runs are not used to claim a final win rate.

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
