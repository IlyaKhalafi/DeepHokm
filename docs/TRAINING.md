# Training

The shipped model is a search-value network, not a PPO checkpoint. PyTorch is
used for training and export; the web UI's default inference path uses NumPy.

## Generate legal teacher labels

Run from the repository root after `uv sync`:

```bash
uv run python -m scripts.generate_q_data \
  --worker 0 --workers 1 --matches 40 --k 6144 \
  --seed-base 100000 --prefix qdata --out-dir data
```

Each completed match is written atomically as `data/qdata_<index>.pkl`.
Interruptions retain periodic hidden resume files; rerunning the identical
command validates and skips completed matches, then resumes unfinished ones.
Changing seeds, search budgets, or collector code requires a new prefix/output
directory. Pickle files must be trusted: never load arbitrary downloaded shards.

For parallel collection, run one process per worker index, all with identical
`--workers`, `--matches`, `--seed-base`, `--prefix`, and `--k`. Worker indices
range from zero to `workers - 1`; their match indices are disjoint. Keep each
new collection's seed range disjoint from prior collections and evaluations.

K=6144 is expensive. For a pipeline smoke test, use a small K and few matches;
do not interpret those labels as equivalent to the full-budget teacher.

## Fit and export RankCNN

```bash
uv run python -m scripts.train_qnet \
  rank_cnn 80 1.5 0.013 0 'qdata_*.pkl' \
  --data-dir data --seed 0 --feature-mode public_strength \
  --batch-size 128 --device cpu --cpu-threads 2 \
  --lr-schedule warmup_cosine --warmup-epochs 5 \
  --min-learning-rate 0.000003 \
  --early-stopping-patience 20 --early-stopping-min-delta 0.001 \
  --checkpoint checkpoints/my-run/best.pt \
  --export-numpy checkpoints/my-run/qnet_numpy.npz
```

The positional arguments are architecture, maximum epochs, size multiplier,
soft-target temperature, suit augmentation flag, and shard glob. For RankCNN,
suit symmetry is already built into the architecture.

The shard glob is relative to `--data-dir`; it may contain comma-separated
patterns. Match-level validation membership is stable as more files arrive.
The run requires decisive examples in both partitions. Duplicate match seeds
are rejected; hold out a separate, untouched seed range for final evaluation.

The trainer retains the best validation weights and writes `best.json` with
dataset hashes, source hashes, configuration, training history, and stop reason.
NumPy export checks forward-pass parity and creates a sibling
`qnet_numpy.features.json` describing the feature mode and hashes. Existing
checkpoints and exports are never silently overwritten.

Use `--feature-mode baseline` for the original 14-plane input. Other modes are
`voids`, `trick_context`, and `public_strength`; their `_zero` controls retain
the same model width while zeroing only the newly added feature layer.

On GPU, select the intended device explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts.train_qnet \
  rank_cnn 80 1.5 0.013 0 'qdata_*.pkl' \
  --data-dir data --device cuda --gpu-memory-limit-mib 2048 \
  --step-pause 0.1 --checkpoint checkpoints/gpu-run/best.pt
```

The memory limit applies to the PyTorch allocator; it is not a guarantee of
exclusive GPU access. Coordinate GPU use with other workloads.

## Evaluate before deployment

Validation optimal-set agreement is not match win rate. Use fresh seeds,
swap team sides, report search budgets and latency, and retain the matched
greedy baseline version when comparing models. Optional `--test-reservation`
accepts a JSON object with `schema_version: 1`, `seed_start`,
`seed_stop_exclusive`, and the corresponding `matches` count; matching teacher
shards are rejected before training.

New weights are not deployed automatically. Configure the web UI only after
evaluation; see [Deployment](DEPLOYMENT.md).

## Alternative self-play training

The earlier PPO and greedy-cloning workflows remain available for comparison:

```bash
make train ARGS="--total-timesteps 1000000 --n-envs 8 --seed 0"
make pretrain-bc ARGS="--n-matches 3000 --epochs 8 --out-path checkpoints/bc.zip"
```

These produce Stable-Baselines3 checkpoints, not NumPy Q-network exports.
See [Methods and results](METHODS_AND_RESULTS.md) for the limitations observed
in these experiments.
