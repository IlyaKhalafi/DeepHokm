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

Win rate rises monotonically through K=1536, with flattening returns
above K=384 (0.790 -> 0.795 -> 0.818). Best measured legal policy:
**0.818 match win rate at K=1536** — exceeding the original 80% target.
The cost/quality sweet spot remains K=384 for practical play.

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

1. ~~K saturation~~: reached at K~384 (0.790); K=768 (0.795) confirms the
   plateau. K=1536 confirmation run in progress.
2. Q-network: replace the K sampled rollouts with a learned action-value
   approximation (same legal inputs), trained supervised on per-action
   search Q-estimates (data generation running at K=192 labels).
3. Distillation of the saturated K=384/768 teacher (deviations should be
   stable at these K, unlike the K=48 teacher whose flips were seed
   noise).
4. Speed: Cython/native backend for the engine and sampler hot paths.
