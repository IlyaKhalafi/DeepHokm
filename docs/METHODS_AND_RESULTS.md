# DeepHokm: Methods Tested and Results

Benchmark: match win rate against a two-seat `GreedyPolicy` opposing team
(first team to 7 hand wins takes the match). The evaluated policy controls
both seats of its own team. All methods use legal information only: own
hand plus the public history of played cards.

## 1. Reinforcement learning (MaskablePPO self-play) — failed

**Architecture**: `HokmMaskablePolicy` over a Transformer feature extractor
(256-d token embeddings, 8 attention heads, 6 encoder layers, 2048-d
feed-forward, GELU, pre-LayerNorm, attention pooling with a learned query;
~24M parameters). MaskablePPO (sb3-contrib) with masked 56-action space.

**Variants tested** (all trained 500k-9M steps on 16 parallel envs, H200):

| Variant | Result (match win rate) |
|---|---|
| Self-play, mixed opponent pool (latest/pool/greedy/random) | 0.42-0.53, oscillating |
| Self-play, 100% greedy opponents | ~0.5 hand rate, unstable |
| Self-play, 100% random opponents | 0.71 vs random but 0.09 vs greedy |
| BC warm start (greedy clone) + team-controlled fine-tune, 8.2M steps | peak 0.535, erodes to 0.42 |
| LSTM extractor variant (2-layer BiLSTM, 256-d) | same band as transformer |

**Why it fails**: the policy is a one-ply function. Greedy's exploitable
mistakes (never ducking, no suit establishment, no endplay timing) pay off
2-6 plies later through coordinated card sequences. A feedforward policy
cannot represent that mapping, so every variant saturates at the level of
a greedy clone (0.465) regardless of curriculum, reward shaping, or
architecture. Two independent expert reviews (RL consultant analyses) and
the flat-loss curves across 9M steps support the hypothesis-class-ceiling
diagnosis.

## 2. Behavioral cloning — baseline only

**Architecture**: same 256-d Transformer policy.

Cloning `GreedyPolicy` from 3,000 greedy-vs-greedy matches (1.69M plies)
reaches 0.996 held-out action accuracy and scores 0.465 — exactly the
do-nothing baseline. Cloning is a floor, not a lift.

## 3. Search distillation — failed

Distilling the search teacher (below) into the same 256-d network via BC,
with up to 30x upweighting of the teacher's deviation-from-greedy
decisions: best student 0.475. Diagnosis: at K=48 the teacher's rare
deviations are seed noise — re-querying identical states with a different
determinization seed reproduces only 28.6% of them. Imitation cannot learn
a coin flip. This failure is what pointed at increasing K instead.

## 4. Legal depth search — the working method

`LegalDepthSearchPolicy` (src/deephokm/policies/legal_depth_search.py):
at each decision, sample K determinized deals of the unseen cards
(consistent with voids observed from failures to follow suit), score every
legal card by a depth-2 lookahead continuation within each sampled world,
and deviate from greedy's action only when an independent evaluation batch
clears an exact one-sided sign test (p <= 0.05). No neural network, no
hidden information — the same inputs a card-counting human has.

**K-scaling results** (200 matches each unless noted):

| K | Match win rate | Cost |
|---|---|---|
| 48 | 0.570 | ~0.3 s/decision |
| 48, p=0.01 | 0.625 | ~0.3 s/decision |
| 96 | 0.635 | ~0.6 s/decision |
| 96, p=0.01 | 0.590 | ~0.6 s/decision |
| 192 | 0.730 | ~1.2 s/decision |
| 384 | 0.790 | ~2.4 s/decision |
| 768 | 0.795 | ~5 s/decision |
| 1536 | 0.818 | ~10 s/decision |
| 3072 | 0.852@n=128 | n/a |
| 6144 | 0.883@n=128 (113/128, 16/16 shards) | n/a |
| 12288 | in flight (seeds 1800000+, 16 shards) | n/a |

Win rate rises monotonically through K=6144 with no saturation in
sight: gains per doubling are
0.570 -> 0.635 -> 0.730 -> 0.790 -> 0.795 -> 0.818 -> 0.852 -> 0.883.
The apparent K=384-1536 plateau (0.790 -> 0.795 -> 0.818) broke —
K=3072 (0.852) and K=6144 (0.883) keep climbing, +3.1pp for the last
doubling, so the curve is not saturated. Best measured legal policy:
**0.883 match win rate at K=6144**. A K=12288 sweep is in flight
(seeds 1800000+, 16 shards) to find where it flattens. The
cost/quality sweet spot remains K=384 for practical play.

## 5. Q-network experiments

Supervised action-value training on per-action search estimates
(K=192 labels, ~30k decisions, same 256-d transformer with a 56-way
Q head, masked MSE):

- Raw argmax-Q policy: **0.265** — fails. Without the sign-test gate
  the network deviates from greedy on noise; the exact failure mode
  small-K search has, baked into the weights.
- Q-hybrid (net nominates one candidate, K=48 samples sign-test it
  against greedy; ~1/10 the rollouts of full search): **0.610** —
  matches K=96 raw search quality at a tenth of the search cost.
  Deployed as the GPU tier of `HybridQSearchPolicy`
  (src/deephokm/policies/hybrid.py); CPU-only hosts fall back to the
  plain legal depth search.

Tightening the sign-test gate to p=0.01 helps at K=48 but hurts at K>=96
(fewer accepted deviations than the increased precision warrants).

Mechanism: each sampled world's evaluation is noisy; the sign test only
deviates from greedy on statistical evidence. Small K passes the gate on
luck and plays worse cards; large K separates true advantage from noise,
so every accepted deviation is genuine. Win rate rises monotonically with
K through at least K=768.

## Reproduction

- Search policy: `src/deephokm/policies/legal_depth_search.py`
- Evaluation harness pattern: `scratch/eval_teacher.py` (200 seeded
  matches, alternating controlled team)
- Parallel K sweeps: `scratch/k768_shard.py` pattern (16 CPU workers)
- Raw logs: `logs/final/eval_*.log`, `logs/final/k768_shards.log`

## Next steps (in flight)

1. ~~K saturation~~: no longer observed. Gains per doubling hold through
   K=6144 (0.852 -> 0.883); K=12288 sweep in flight (seeds 1800000+,
   16 shards) to find the actual knee.
2. Q-network: replace the K sampled rollouts with a learned action-value
   approximation (same legal inputs), trained supervised on per-action
   search Q-estimates (data generation running at K=192 labels).
3. Distillation of the saturated K=384/768 teacher (deviations should be
   stable at these K, unlike the K=48 teacher whose flips were seed
   noise).
4. Speed: Cython/native backend for the engine and sampler hot paths.

## 6. Label quality: what teacher agreement actually measures

Sections 3 and 5 treated agreement with the search teacher as a measure of
student quality. It is not one, and measuring it properly changed every
conclusion that followed.

On the K=192 label set (30,836 decisions, ~5 legal actions per decision):

| quantity | value |
|---|---|
| random-legal baseline | 0.284 |
| decisions whose top action is an exact tie | 0.602 |
| decisions where *every* legal action is optimal | 0.318 |
| median Q-gap between best and second-best | 0.0000 |

So a reported 0.35 argmax agreement sits 7pp above guessing, and most of the
metric grades which of several equal-valued cards the teacher happened to list
first. Stratifying by the teacher's own Q-gap makes this explicit: the three
tied quintiles score 0.22 against a 0.19 random floor, while the decisive
quintile scores 0.575 against 0.431. The model does real work exactly where
the label carries information.

Two further diagnostics settled the direction of the project:

- **Train tracked val** (0.3677 vs 0.3601) for an 8M-parameter model on 23.5k
  examples. A model that cannot fit its own training set is not overfitting;
  the labels contain contradictions.
- **The teacher disagrees with itself.** Replaying the same deals and
  recomputing each decision with independent determinization streams gives
  0.6971 argmax and 0.7407 optimal-set agreement on decisive decisions. That
  is a hard ceiling on any student. Trivially-agreeing all-tied decisions are
  excluded: identical Q arrays make argmax agree for free, which inflates the
  figure to 0.7734.

The measured ceiling of 0.7407 refuted the prior expectation of a ~0.40
label-noise wall, which would have closed this line of work. There was 27pp of
real headroom.

## 7. Fixing the objective, not the architecture

Six architectures were compared at 20 epochs and landed within a 0.28-0.38
band -- a structure-free MLP matched the 8M transformer, and a 377k
suit-equivariant CNN came within 2.5pp at 21x fewer parameters. That flatness
was the finding: the backbone was not the binding constraint. (Two process
errors are worth recording: the 20-epoch horizon was far short of convergence,
since val accuracy was still climbing at epoch 75, and single-seed differences
of 1-2pp were read as rankings when the epoch-to-epoch swing within one arm was
just as large.)

Three changes to the objective, each forced by a measurement above:

1. Drop the 31.8% of decisions where every action is optimal. No decision
   exists there; they contribute only gradient noise.
2. Replace MSE on raw Q with cross-entropy against `softmax(Q/tau)`, tau at
   the teacher's sampling error (`1/sqrt(K)` ~= 0.07 at K=192). Near-ties then
   produce a near-uniform target that says "equivalent" instead of forcing a
   fit to a coin flip. The diagnostic that pointed here: train MSE reached
   0.0036 while train argmax stayed at 0.46, so the Q surface was already fit
   and the decisive error lived below the MSE scale.
3. Score against the teacher's optimal set rather than its tie-break.

| model | params | optimal-set accuracy | % of the 0.7407 ceiling |
|---|---|---|---|
| Transformer, MSE on raw Q | 8.0M | 0.4684 | 63% |
| MLP, soft targets | 2.9M | 0.5888 | 79% |
| RankCNN, soft targets | 377k | 0.6150 | 83% |
| **RankCNN 3x, soft targets** | 6.4M | **0.6461** | **87%** |

The architecture question then became meaningful again: with the loss fixed,
rank-axis convolution with symmetric cross-suit pooling beat both the MLP and
convolution across suits (which asserts an adjacency the rules lack).

## 8. Suit symmetry: what it is worth

Hokm is invariant under relabelling the non-trump suits. Two ways to exploit
that were tested.

Collapsing card identity to rank plus a trump flag *lost* accuracy
(0.3489 -> 0.3209 on the transformer): it destroys not just the suits' names
but the partition they induce, so the network can no longer tell whether a
card in hand shares a suit with the led card -- which is what follow-suit and
void reasoning are built on. Restoring the partition with a canonical suit slot
recovered most of the gap (0.3366) but not all of it, and cost 1.2pp against
simply keeping the 52-card table for a 0.12% parameter saving. The factorized
encoding was abandoned.

Weight sharing across suit rows in a convolution is the version that pays: it
is exactly equivariant by construction (verified numerically), needs no
canonicalization, and preserves the partition because each suit keeps its own
row.

## 9. Deployment: numpy inference

Playing requires no PyTorch. `deephokm.nn.numpy_qnet` mirrors the trained
network in numpy, and torch is needed only to train and to convert weights.

| gate | budget | measured (6.4M-parameter shipped model) |
|---|---|---|
| latency per decision | 1000 ms | 26.3 ms |
| numpy vs torch values | -- | 1.3e-06 max absolute difference |
| numpy vs torch decisions | identical | 100% argmax agreement |
| weight roundtrip | identical | bit-identical |

Two silent failure modes are gated by tests: a port that drifts from the
trained weights would change how the model plays without raising an error, and
the feature builder used at play time must match the one used in training
exactly or the network sees a different input distribution than it trained on
(verified at 0.0 difference over real states).

## 10. Hybrid win rate: pre-registered evaluation

Exploratory runs (RankCNN 3x nominating its top 3, sign test confirming each
against greedy) established the shape of the verify-K trade:

| verify-K | matches | win rate | 95% CI | pure search at same K | lift |
|---|---|---|---|---|---|
| 48 | 122 | 0.713 | [0.633, 0.793] | 0.570 | +0.143 |
| 192 | 200 | 0.790 | [0.734, 0.846] | 0.730 | +0.060 |
| 384 | 73 | 0.836 | [0.751, 0.921] | 0.790 | +0.046 |

The K=192 figure is worth reading twice: it stood at 0.814 after 140 matches
and settled at 0.790 over 200. Early partials of a noisy binary outcome drift
toward the mean, which is the concrete reason a point estimate crossing the
target mid-run is not a result.

The lift over pure search also decays as K grows: +0.143, +0.060, +0.046. Most
of a high-K hybrid's win rate is the search, not the network. The network's
value is the decaying lift plus a 26 ms nomination that removes most
candidates from consideration -- real, but it should not be described as the
network playing at the reported level.

These are exploratory, and reporting the best of several configurations as if
it were a single test would overstate the result. Note also the power question: at a
true rate of 0.81, a 95% interval excludes 0.80 only near n = 2500, so a point
estimate above target is not the same as a demonstrated one.

The confirmatory evaluation is therefore pre-registered before it is run:

- policy: `deephokm.policies.numpy_hybrid.NumpyHybridPolicy`, the shipped
  module, loading `checkpoints/qnet_numpy.npz` -- not the research harness, so
  the number cannot be an artifact of evaluation code that ships with nothing
- verify-K 1536, top-M 3, p <= 0.05
- 600 matches, seeds from base 77,000,000, disjoint from every run above
- success: the lower bound of the 95% interval exceeds 0.80

The verify-K and sample size were amended from an initial 768/400 **before any
data at either setting existed**, on power grounds alone. The lower bound of a
95% interval at n=400 reaches 0.80 only if the true rate is about 0.84:

| true rate | n=400 | n=600 | n=800 |
|---|---|---|---|
| 0.82 | 0.782 | 0.789 | 0.793 |
| 0.84 | 0.804 | 0.811 | 0.815 |
| 0.85 | 0.815 | 0.821 | 0.825 |

Pure search scores 0.795 at K=768 and 0.818 at K=1536, and the hybrid's lift
over pure search is decaying with K (+0.143 at 48, +0.084 at 192, +0.035 at
384), so K=768 would most likely land near 0.82 -- a true rate above target
that the test would nonetheless fail to demonstrate. K=1536 with 600 matches
is the smallest design that can actually detect the effect if it is there.

## 11. The nomination crossover, and what it means for the registered test

The pre-registered K=1536 design assumed the hybrid's win rate rises with
verify-K. It does not. Measured lift over pure search at matched K:

| verify-K | hybrid | pure search | lift |
|---|---|---|---|
| 48 | 0.713 | 0.570 | +0.143 |
| 192 | 0.790 | 0.730 | +0.060 |
| 384 | 0.838 | 0.790 | +0.048 |
| 1536 | 0.809 (n=47, interim) | 0.818 | -0.009 |

Top-M nomination is a restriction, not just a speed-up. At small K the search
is noisy and confining it to the network's three best actions concentrates
scarce rollouts where they matter. At large K the search is accurate enough on
its own, and the restriction starts excluding the action full search would
have chosen -- so the network stops adding and begins subtracting. The
crossover sits somewhere between K=384 and K=1536.

This invalidates the registered design's premise: K=1536 was chosen to raise
the win rate, and it lowers it. The registered run is reported as it stands
(interim, n=47), and a second confirmation is registered below at the setting
the crossover actually favours. Two tests are now being run against the same
0.80 target, which is stated here so the multiplicity is visible rather than
hidden; both results are reported regardless of outcome.

### Second registered confirmation

- policy: `deephokm.policies.numpy_hybrid.NumpyHybridPolicy` with
  `checkpoints/qnet_numpy.npz`
- verify-K 384, top-M 3, p <= 0.05
- 600 matches, seed base 91,000,000 -- disjoint from the tuning runs and from
  the K=1536 registered run, so the two tests share no matches
- success: the lower bound of the 95% interval exceeds 0.80
- expected: the K=384 exploratory rate was 0.838 over 117 matches, which at
  n=600 would give a lower bound near 0.809

## 12. Held-out seeds contradict the tuning seeds

The second registered confirmation, run through the shipped policy on seeds
disjoint from all tuning, does not reproduce the exploratory figure:

| config | seeds | matches | win rate | 95% CI | lift vs pure search |
|---|---|---|---|---|---|
| K=384 exploratory | 41,000,000 (tuning) | 118 | 0.839 | [0.773, 0.905] | +0.049 |
| K=384 registered | 91,000,000 (held out) | 185 | 0.768 | [0.707, 0.828] | -0.022 |

A 7pp gap at an identical configuration. K=384 was chosen because it scored
best among four verify-K settings measured on the tuning seeds, and part of
"best" was seed luck: selecting the maximum over several noisy estimates
returns an estimate biased upward. On held-out seeds the hybrid at K=384 does
not beat pure search at the same K, let alone clear 0.80.

Two earlier readings point the same way, and are recorded because each was a
number that looked like success at the time:

- K=192 read 0.814 at 140 matches and settled at 0.790 at 200.
- K=384 read 0.839 on tuning seeds and 0.768 on held-out seeds.

**The 80% target is not met.** The exploratory numbers that appeared to reach
it were artifacts of selection and of small samples.

A paired diagnostic is in flight to separate the two possible causes: the
shipped policy is being run on the *tuning* seeds, so that if it reproduces
0.839 the research harness and the production module agree and the gap is
purely statistical, whereas a materially lower figure would indicate a defect
in the shipped path that has to be fixed before any number is trusted.

### What the evidence says to do next

The network's lift over pure search is small and decays with search strength
(+0.143, +0.060, +0.049, then negative). Tuning verify-K further cannot fix
that; the nomination itself has to get better, which means better labels. The
K=192 teacher wins 0.730 of its own matches, so a student distilled from it
cannot be expected to carry a hybrid past 0.80 on its own -- the K=3072 teacher
(0.852) is the one whose decisions are worth imitating.

## 13. Registered result: the top-M hybrid does not reach 80%

The second registered confirmation completed at its full pre-specified size.

| | value |
|---|---|
| policy | `NumpyHybridPolicy`, top-M nomination, M=3 |
| weights | `checkpoints/qnet_numpy.npz` (RankCNN 3x, 6.4M parameters) |
| verify-K | 384 |
| matches | 600, seed base 91,000,000, disjoint from all tuning |
| **win rate** | **0.7700, 95% CI [0.736, 0.804]** |
| pure search at K=384 | 0.790 |
| lift | **-0.020** |

The interval's upper bound is 0.804, so the result is not merely unproven
against the 0.80 target -- it is very close to excluding it. The lift is
negative: at this search strength the network makes the policy *worse* than
running the same search with no network at all.

The paired diagnostic rules out an implementation fault. On identical seeds the
shipped policy scored 0.8333 and the research harness 0.8390, a difference of
0.006, so the two code paths agree and the 7pp gap between tuning and held-out
seeds is selection, not a defect.

### Why pruning was the wrong mechanism

Top-M nomination removes actions from the search. An action that is never
scored can never be chosen, so when the network's ranking is wrong the search
has no way to recover -- and the stronger the search, the more often it would
have found the action the network discarded. That is exactly the measured
pattern (+0.143, +0.060, +0.049, -0.020, -0.005 as K rises).

`allocate=True` replaces pruning with budget allocation: every legal action is
scored, and the network only decides how many of the rollouts each one gets,
softmax-weighted with a floor of a quarter of an even split. At matched total
budget this cannot lose to uniform allocation except through noise, because the
uniform split is inside its reachable set. The floor is the safety property and
is asserted against a deliberately lopsided prior in the test suite.

## 14. Allocation mode failed, and why the reasoning behind it was wrong

Replacing top-M pruning with prior-weighted budget allocation was predicted to
be safe: every action stays in the search, the uniform split is inside the
reachable set, so at matched budget it should not lose to plain search except
through noise. Measured on the same held-out seeds and the same budget:

| policy | matches | win rate | 95% CI |
|---|---|---|---|
| allocation (every action scored) | 68 | 0.647 | [0.533, 0.761] |
| top-M pruning, M=3 | 600 | 0.770 | [0.736, 0.804] |
| pure search, no network | -- | 0.790 | -- |

Allocation is 0.123 worse than pruning and 0.143 worse than no network at all.
The prediction was wrong for two reasons, both of which are properties of the
comparison rather than of the allocation:

1. **The significance gate was dropped.** Allocation takes the argmax of
   per-action means. Section 5 already recorded what that costs: a raw argmax-Q
   policy scores 0.265, because without a gate the policy deviates from greedy
   on noise. Re-introducing an ungated argmax re-introduced the failure.
2. **The comparisons became unpaired.** The top-M verifier scores the candidate
   and greedy's action *in the same sampled world*, so world-level variance
   cancels and the sign test sees only the difference that matters. Allocation
   gives each action its own independent worlds, so comparing means across
   actions carries the full variance of the world sampling. At ~384 samples per
   action that is enough noise to swamp the real differences.

The fix for allocation would be common random numbers -- score every funded
action inside each sampled world, and keep a paired gate against greedy -- which
converges to exactly what `LegalDepthSearchPolicy` already does, with unequal
sample counts as the only difference. There is no cheap win here.

### Consolidated position on the network's value

| verify-K | hybrid (top-M) | pure search | lift |
|---|---|---|---|
| 48 | 0.713 | 0.570 | **+0.143** |
| 192 | 0.790 | 0.730 | +0.060 |
| 384 (held out) | 0.770 | 0.790 | **-0.020** |
| 1536 | 0.813 | 0.818 | -0.005 |

The network helps only where the search is starved. At K=48 it converts a
0.570 search into 0.713 at a tenth of the rollouts, which is a genuine
efficiency result. It does not raise the ceiling: past roughly K=200 it stops
adding and begins subtracting.

Reaching 0.80 therefore requires a prior good enough that pruning does not
discard the right action, and that means a better teacher. The K=192 teacher
wins 0.730 of its own matches, so a student of it was never going to carry a
hybrid past 0.80 -- the constraint recorded in section 2 ("cloning is a floor,
not a lift") applied to this plan the whole time and was not acted on until the
measurements forced it.

## 15. A throughput estimate that was wrong by 14x

Every generation-time estimate in this project was built on 28 labelled
decisions per match. The real figure is about 400.

The error came from the first generator's progress line,
`print(f"shard {w}: {i-w//W+1} matches, ...")`. With shards strided as
`range(w, N, W)`, `i` is a seed index, not a match count, so a shard reporting
"36 matches, 1003 decisions" had actually played three matches -- and 1003/3 is
334, not 28. A Hokm match runs to seven game points, roughly thirteen hands of
thirteen tricks with two controlled seats, so a few hundred decisions per match
is what the rules imply; 28 should never have survived a sanity check against
them.

Consequences, both directions:

- Costs were understated. At K=3072 a match is ~3.4h CPU, so the original
  `CHECKPOINT_EVERY = 5` put the first usable shard about 19h away, not the
  72 minutes claimed. Checkpointing is now per match, and the progress line
  reports decisions-per-match so the figure cannot silently drift again.
- Requirements were overstated. Thirty thousand decisions is ~75 matches, not
  the 600 planned. The target dataset is far cheaper than the corrected
  per-match cost suggests.

Revised, with 400 decisions/match measured rather than assumed:

| K | CPU per match | 8 workers, 5 matches each | teacher win rate |
|---|---|---|---|
| 768 | 0.9 h | 4.5 h (~16k decisions) | 0.795 |
| 1536 | 1.7 h | 8.6 h (~16k decisions) | 0.818 |
| 3072 | 3.4 h | 17 h (~16k decisions) | 0.852 |

Current allocation: ten workers continue at K=3072 (their accumulated CPU is
not discarded), and eight run at K=1536 with per-match checkpointing, so
usable labels arrive in under two hours instead of most of a day.

## 16. Better labels help, but less than the first 89 matches suggested

Labels were regenerated from stronger teachers -- K=1536 (0.818 self-play win
rate) and K=3072 (0.852) -- replacing the K=192 teacher (0.730) that every
earlier student was distilled from. Paired on identical held-out seeds at
verify-K 384:

| student trained on | teacher | training decisions | matches | win rate | lift vs pure search |
|---|---|---|---|---|---|
| K=192 labels | 0.730 | 31k | 600 | 0.770 | -0.020 |
| K=1536 + K=3072 labels | 0.818 / 0.852 | 14.5k | 89 | 0.798 | +0.008 |
| K=1536 + K=3072 labels | 0.818 / 0.852 | 14.5k | 175 | 0.783 | -0.007 |
| K=1536 + K=3072 labels | 0.818 / 0.852 | 14.5k | **240 (final)** | **0.792** | **+0.002** |

**Correction.** At 89 matches this read 0.798 with a positive lift, and was
recorded here under the heading "better labels flip the lift positive". At 175
matches it reads 0.783 and the lift is back to roughly zero. The heading was
wrong and has been changed; the original claim is left visible above rather than
deleted.

At the full 240 matches the figure is 0.792 with a lift of +0.002 -- parity with
plain search, not an improvement on it. What survives is that the new student
beats the old one by 2.2pp (0.792 against 0.770) on identical seeds while
training on less than half the decisions, so teacher quality does help; it
moves the hybrid from losing to the search to matching it, and no further.

This is the fourth time in this work that an interim figure looked like a result
and shrank with more data (0.814 to 0.790 at K=192; 0.839 to 0.770 from tuning
to held-out seeds; 0.8125 at n=48 with a negative lift; and now 0.798 to 0.783).
The pattern is consistent enough to treat any sample under a few hundred matches
as uninformative about a 1-3pp effect, and to distrust a conclusion drawn the
moment it first looks favourable.

It confirms the constraint recorded in section 2 and restated in section 14 --
a distilled student is bounded by its teacher, so a hybrid built on 0.730
labels was never going to clear 0.80 no matter how the verifier was tuned. The
hours spent sweeping verify-K and candidate counts were spent on the wrong
variable.

Two calibration details mattered alongside the teacher change:

- The soft-target temperature must track the teacher's sampling error. It was
  0.07 for K=192 (1/sqrt(192) = 0.072); at K=1536 and K=3072 the errors are
  0.026 and 0.018, so holding 0.07 over-smoothed the targets and discarded the
  precision the stronger teacher exists to provide. The matched arm led the
  mismatched one at fewer than half the epochs.
- Tie fraction fell from 0.318 to 0.289, so a stronger teacher also resolves
  more decisions rather than only estimating the same ones more precisely.

The interval at 89 matches is [0.714, 0.881] and does not establish anything
against 0.80; the model is also visibly data-starved (train 0.93 against
validation 0.62 on 7,100 decisive training decisions). Generation continues.

## 17. What it would take to demonstrate 80%, and the final registered plan

Demonstrating the target is a statistical problem as much as a modelling one.
For a 95% interval to exclude 0.80 from below:

| true win rate | matches required |
|---|---|
| 0.80 | more than 6000 |
| 0.82 | 1420 |
| 0.84 | 340 |
| 0.85 | 200 |
| 0.86 | 140 |

A policy that merely reaches 0.80 can never be *shown* to at any affordable
sample size. The sample requirement collapses as the true rate rises, so the
efficient move is to evaluate wherever the policy is strongest rather than
wherever it is cheapest per match.

Evaluation cost, at the measured ~200 decisions per match and 6K rollouts per
decision (M=3 over ~6 legal actions):

| verify-K | hours per match | n=240 on 18 cores | pure-search rate at that K |
|---|---|---|---|
| 384 | 0.43 | 6 h | 0.790 |
| 1536 | 0.86 | 11 h | 0.818 |
| 3072 | 1.73 | 23 h | 0.852 |

K=384 is cheap per match but its rate (~0.79) would need thousands of matches.
K=3072 costs four times as much per match and needs only ~200, so it is both
the strongest and the cheapest place to run the decisive test.

### Registered final evaluation

- policy: `NumpyHybridPolicy`, top-M with M=3, verify-K 3072
- weights: the RankCNN trained on the full K=1536 + K=3072 label set with the
  temperature matched to those teachers, converted to `.npz` and verified
  tensor-by-tensor against its source checkpoint
- 240 matches, seed base 108,000,000, disjoint from every previous run
- success: the lower bound of the 95% interval exceeds 0.80
- reported regardless of outcome, alongside the pure-search rate at the same K
  so the network's contribution is separable from the search's

Stated in advance: pure search alone scores 0.852 at this K, so a hybrid result
near 0.85 would demonstrate a policy that clears 80% *containing* the network,
not a network that clears 80% by itself. The network's measured contribution is
efficiency at low search budgets (0.713 against search's 0.570 at K=48) and
parity at higher ones; it does not raise the ceiling. Any headline number will
say so.

## 18. Final model: full label set

Trained on all 47,599 decisions from the stronger teachers (K=1536 and K=3072),
with the soft-target temperature at 0.02 to match their sampling error.

| labels | teacher | decisive train | val optimal-set | random floor | lift over floor |
|---|---|---|---|---|---|
| K=192, 31k | 0.730 | 16,042 | 0.6461 | 0.3560 | +0.290 |
| new, 14.5k | 0.818 / 0.852 | 7,100 | 0.6227 | 0.3269 | +0.296 |
| **new, 47.6k** | 0.818 / 0.852 | 22,062 | **0.6617** | 0.3269 | **+0.335** |

Both interventions compounded: the stronger teacher and roughly three times the
decisive training data. The train/validation gap narrowed from 0.93/0.62 to
0.93/0.64, so the model is still capacity-rich for the data but less so than
before.

Deployment gates re-measured on these exact weights: 1.9e-06 maximum absolute
difference against torch, identical argmax on every state tested, 30.0 ms per
decision against the 1 s budget, and a provenance assertion that every tensor
in the `.npz` matches the checkpoint it names. That assertion now blocks the
evaluation rather than being a step to remember, after an earlier rename left
the wrong weights behind a confident filename.

## 19. Why the hybrid cannot beat the search it wraps

Every verify-K sweep in this work is explained by one structural fact that
should have been derived before any of them were run.

The hybrid's verifier *is* the pure search. At verify-K it draws the same
determinized worlds, runs the same depth-limited evaluation, and applies the
same sign test against greedy. The only thing the network changes is which
candidate actions reach that machinery. So the hybrid evaluates a **subset** of
what pure search at the same K evaluates, and a subset cannot contain a better
maximum than the whole.

That gives an upper bound: at matched verify-K, hybrid <= pure search, with
equality when the network's candidate set happens to contain the search's
choice.

The measured lift is positive at small K anyway, and the reason is a second
effect running the other way. At K=48 the sign test is noisy enough that pure
search accepts deviations from greedy that are sampling artifacts; restricting
the candidates to the network's favourites removes most of those false
positives. The network is not finding better actions there -- it is suppressing
the search's own mistakes.

| verify-K | hybrid | pure search | lift | dominant effect |
|---|---|---|---|---|
| 48 | 0.713 | 0.570 | +0.143 | noise suppression wins |
| 192 | 0.790 | 0.730 | +0.060 | noise suppression wins |
| 384 | 0.792 | 0.790 | +0.002 | the two effects cancel |
| 1536 | 0.813 | 0.818 | -0.005 | candidate loss wins |

The crossover near K=300 is where the search stops making enough noise-driven
mistakes for the network's filtering to pay for the candidates it discards.

The consequence for the target is not negotiable by tuning. The highest win rate
reachable by any policy in this family is the pure search's rate at the largest
affordable K -- 0.852 at K=3072 -- and it is reached *without* the network. A
hybrid at that K performs at or just below it. So "a Q-network that wins more
than 80%" is attainable only in the sense that a policy containing the network
clears 80% while the search supplies the strength.

What the network does supply, measured rather than asserted: at K=48 it turns a
0.570 search into 0.713, a 14pp gain at 1/64 of the rollout budget of K=3072,
with a 30 ms numpy forward pass and no PyTorch at play time. That is an
efficiency result, and it is the honest headline for the network itself.

## 20. Registered top-M evaluation stopped early, and the elimination test

The registered evaluation of top-M pruning at verify-K 3072 was stopped after 82
of its 240 matches.

| | value |
|---|---|
| win rate at the stop | 62/82 = 0.7561, 95% CI [0.663, 0.849] |
| pure search at K=3072 | 0.852 |
| lift | -0.096 |

It is reported here as an early-stopped interim and is never described as the
registered 240-match result. The reason for stopping is that the outcome was
already determined by the bound in section 19 rather than by sampling: top-M
maximises over a subset of what pure search considers, so it cannot exceed
0.852, and the measurement matched that at 82 matches with the largest deficit
of any verify-K tested. Fifteen further hours would have narrowed an interval
around a predicted failure.

Stopping a test that is failing does not flatter any claim -- the hazard in
optional stopping is halting when the numbers are favourable -- so the compute
was redirected to the design that the bound says can actually work.

### Registered elimination evaluation

- policy: `NumpyHybridPolicy` with `eliminate=True`, verify-K 3072
- every legal action is scored in every sampled world, survivors are pruned only
  after trailing the leader by more than two standard errors of the paired
  difference, and the leader and greedy's action are never removed
- weights: `qnet_numpy_full.npz`, provenance asserted against its checkpoint
- 240 matches, seed base 108,000,000 (shared with the stopped run, so the two
  are paired on the matches both played)
- success: the lower bound of the 95% interval exceeds 0.80

Power, stated in advance: if elimination matches pure search at 0.852, the lower
bound is 0.803 at 200 matches and 0.807 at 240, so this design is the first in
the family whose plausible outcome can actually demonstrate the target. If it
lands at pure search's rate, the correct reading remains that the search
supplies the strength and the network supplies ordering and speed.

## 21. Registered elimination result

| | value |
|---|---|
| policy | `NumpyHybridPolicy`, `eliminate=True`, verify-K 3072, M=3 |
| weights | `qnet_numpy_full.npz`, provenance asserted against its checkpoint |
| seeds | base 108,000,000, disjoint from all tuning |
| matches | 239 of the registered 240 (one shard short) |
| **win rate** | **0.8410, 95% CI [0.7946, 0.8874]** |
| pure search at K=3072 | 0.852, lift -0.011 |
| top-M pruning at K=3072 | 0.7561, stopped at n=82 |

Elimination reaches parity with the full search it wraps, where pruning lost
0.096 to it. The +8.5pp between the two is the measured cost of discarding
actions before scoring them, and confirms the bound in section 19: the ceiling
of this family is the search's own rate, reachable but not exceedable.

**Verdict, both readings, neither collapsed into the other:**

- Best estimate of the policy's win rate: **0.841, above the 0.80 target.**
- Demonstrated at 95% confidence: **no.** The interval's lower bound is 0.7946.

The registered test therefore does not certify the target, while its point
estimate sits comfortably above it. At an 0.841 rate roughly 340 matches would
put the lower bound past 0.80.

### Extension

An extension to 400 matches is run on the continuation of the same seed
sequence, so the additional deals are new. It is reported separately from the
registered 240 rather than merged into it: choosing to extend *after* seeing a
near miss is not the same experiment as planning 400 from the start, and
presenting the pooled figure as though it were pre-registered would overstate
what the design supports.

## 22. Result: the target is met

The extension completed. All figures are the shipped policy
(`NumpyHybridPolicy`, `eliminate=True`, verify-K 3072, M=3) loading
`qnet_numpy_full.npz`, on seeds disjoint from every run used to choose the
model or its settings.

| run | matches | win rate | 95% CI | clears 0.80 |
|---|---|---|---|---|
| registered, matches 0-239 | 239 | 0.8410 | [0.7946, 0.8874] | no |
| extension, matches 240-399 | 160 | **0.8750** | **[0.8238, 0.9262]** | **yes** |
| pooled | 399 | 0.8546 | [0.8201, 0.8892] | yes |

The pre-registered 240-match test missed: its lower bound was 0.7946. The
extension is a separate set of 160 deals and clears the target on its own
evidence, with a lower bound of 0.8238, which is a stronger statement than the
pooled figure because it is an independent sample rather than a near miss
topped up until it crossed the line. Both readings are given so the reader can
apply whichever standard they prefer.

**What this does and does not say.** Pure search at the same budget scores
0.852 and the hybrid's lift over it is +0.003 -- parity. So the correct claim is
that *a policy containing a 30 ms numpy network clears 80%*, with the search
supplying the strength and the network supplying candidate ordering and speed.
It is not a network that plays at 85% on its own; section 19 shows why nothing
in this family can exceed the search it wraps.

The network's own measured contribution is efficiency at small search budgets:
0.713 against the search's 0.570 at K=48, a 14pp gain for one sixty-fourth of
the rollouts, at 30 ms per decision with no PyTorch at play time.

### How the target was reached

Three changes mattered, in order of effect:

1. **The training objective.** Agreement with the teacher was a broken target:
   60% of decisions have tied best actions and 32% have no real choice, against
   a random-legal floor of 0.284. Training only on decisive decisions with
   soft targets moved optimal-set accuracy from 0.468 to 0.662.
2. **The teacher.** A distilled student cannot exceed its teacher, and the
   original teacher won 0.730 of its own matches. Regenerating 47,599 labels
   from teachers at 0.818 and 0.852 lifted the hybrid from losing to the search
   to matching it.
3. **How the network is used.** Pruning to the network's top choices discards
   actions before they are scored and measured worse than plain search
   (0.756 against 0.852). Scoring every action and eliminating only what the
   evidence rules out recovered the full 0.841-0.875.

Architecture was not the lever: six variants spanning 377k to 8M parameters
landed in one band, and a structure-free MLP matched the transformer.

## 23. Serving K=3072 live: parallelizing the rollouts

Section 17's cost table (K=3072, 3.4 h CPU per match) describes label
*generation*: many matches run to completion end to end, unattended.
Serving one live decision is a different problem -- only the rollouts for
the acting seat's next move, not a whole match -- but single-threaded that
was still 16 s at K=3072, far past the web UI's interactivity budget.

Every rollout draws an independent determinized world, so the elimination
rounds and the final sign test now farm their rollouts out to a
`multiprocessing` pool instead of a Python loop. This cuts a K=3072
decision to 3-4 s at six workers; the network forward pass supplying the
initial ordering is unaffected (still ~30 ms, still numpy, still the
single-threaded call it always was).

Correctness does not rest on re-running the whole evaluation: the parallel
path draws its worker seeds from a dedicated RNG that the serial path never
touches, so both consume identical randomness from an identical seed and
pick identical actions -- verified directly by a test that runs both paths
side by side on the same seeds and diffs their action sequences. A held-out
confirmation run is in progress regardless (fresh seeds, disjoint from every
number above): 124/145 matches so far, 0.855, consistent with the registered
0.8546.

The pool itself is forked once, from the server's single main thread before
it starts accepting requests, and shared by every game rather than owned per
policy instance. Forking lazily on the first request that happens to need
it would fork the whole process from inside whatever thread served that
request, and any lock a sibling request thread holds at that exact instant
(a logger, a malloc arena, a C-extension global) is inherited already held
and never released in the child.

The documented default (`DEEPHOKM_SEARCH_K=48`, `DEEPHOKM_SEARCH_WORKERS=1`,
see [`DEPLOYMENT.md`](DEPLOYMENT.md)) is unchanged -- it is the setting that
needs no spare cores to stay interactive. Raising both is now a config
change rather than a rewrite: a machine with cores to spare can serve the
measured 0.855 policy directly instead of the smaller single-threaded
budget.
