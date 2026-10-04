"""Train action-value networks from legal search labels.

Use soft-target cross-entropy on decisive decisions and retain the best
validation checkpoint. Optional early stopping, warmup/cosine scheduling and
shared-GPU controls are recorded in per-run metadata.

Run with ``python -m scripts.train_qnet --help``. Pickle training data must be
trusted; do not load arbitrary downloaded shards.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch as th
from scripts import qnet_features
from scripts.qnet_features import (
    ARMS,
    NUM_RANKS,
    NUM_SUITS,
    _build,
    load_features,
    matching_files,
)

NUM_SUITS_, NUM_RANKS_ = NUM_SUITS, NUM_RANKS

TIE_EPS = 1e-9
# Teacher Q is a mean over K worlds of outcomes in [-1, 1]; at K=192 the
# standard error is ~1/sqrt(192) ~= 0.07. Gaps below this are noise, so tau
# at that scale flattens the target exactly where the label is uninformative.
DEFAULT_TAU = 0.07
TRUMP_PLANE = 3  # planes[:, TRUMP_PLANE] marks the trump suit's row
DEFAULT_PATIENCE = 20
DEFAULT_MIN_DELTA = 0.001  # 0.1 percentage points in accuracy, not loss units.
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_WARMUP_START_FACTOR = 0.1
LABEL_MATRIX_DIMENSIONS = 2


@dataclass
class EarlyStopping:
    """Count epochs without a meaningful gain, independently of best weights.

    Small gains can accumulate past min_delta before resetting patience.
    A patience of zero disables stopping, but still tracks progress.
    """

    patience: int
    min_delta: float
    meaningful_best: float = -math.inf
    stale_epochs: int = 0

    def update(self, score: float) -> bool:
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("validation accuracy must be finite and in [0, 1]")
        if score > self.meaningful_best + self.min_delta:
            self.meaningful_best = score
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return self.patience > 0 and self.stale_epochs >= self.patience


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Retain the original positional interface and add sharing controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("epochs", type=int)
    parser.add_argument("scale", type=float)
    parser.add_argument("tau", type=float, nargs="?", default=DEFAULT_TAU)
    parser.add_argument("augment", type=int, choices=(0, 1), nargs="?", default=0)
    parser.add_argument("data_glob", nargs="?", default="qdata_*.pkl")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--gpu-memory-limit-mib", type=float, default=None)
    parser.add_argument("--step-pause", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--export-numpy", type=Path, default=None)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="epochs without meaningful validation gain; 0 disables stopping",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=DEFAULT_MIN_DELTA,
        help="minimum meaningful gain in accuracy units (0.001 = 0.1 pp)",
    )
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine"), default="constant")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--warmup-start-factor", type=float, default=DEFAULT_WARMUP_START_FACTOR)
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    parser.add_argument(
        "--feature-mode",
        choices=(
            "baseline",
            "voids",
            "voids_zero",
            "trick_context",
            "trick_context_zero",
            "public_strength",
            "public_strength_zero",
        ),
        default="baseline",
    )
    parser.add_argument("--loss-weighting", choices=("none", "q_gap"), default="none")
    parser.add_argument("--confidence-gap-scale", type=float, default=0.01)
    parser.add_argument("--confidence-weight-floor", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch size must be positive")
    if any(not math.isfinite(x) or x <= 0 for x in (args.scale, args.tau)):
        parser.error("scale and temperature must be finite and positive")
    if not math.isfinite(args.step_pause) or args.step_pause < 0:
        parser.error("step pause must be finite and nonnegative")
    if args.cpu_threads is not None and args.cpu_threads <= 0:
        parser.error("CPU thread count must be positive")
    if args.gpu_memory_limit_mib is not None and (
        not math.isfinite(args.gpu_memory_limit_mib) or args.gpu_memory_limit_mib <= 0
    ):
        parser.error("GPU memory limit must be finite and positive")
    if args.export_numpy is not None and args.arm != "rank_cnn":
        parser.error("numpy export supports rank_cnn only")
    if args.feature_mode != "baseline" and args.arm != "rank_cnn":
        parser.error("extra feature experiments support rank_cnn only")
    if args.early_stopping_patience < 0:
        parser.error("early-stopping patience must be nonnegative")
    if (
        not math.isfinite(args.early_stopping_min_delta)
        or not 0 <= args.early_stopping_min_delta <= 1
    ):
        parser.error("early-stopping min delta must be finite and in [0, 1]")
    validate_schedule_args(parser, args)
    validate_confidence_args(parser, args)
    validate_seed(parser, args.seed)
    return args


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Resolve datasets independently of the caller's machine or home directory."""
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--test-reservation", type=Path, default=None)


def validate_seed(parser, seed: int) -> None:
    if not 0 <= seed < 2**32:
        parser.error("seed must be in [0, 2**32)")


def validate_confidence_args(parser, args) -> None:
    if not math.isfinite(args.confidence_gap_scale) or args.confidence_gap_scale <= 0:
        parser.error("confidence gap scale must be finite and positive")
    if not math.isfinite(args.confidence_weight_floor) or not 0 < args.confidence_weight_floor <= 1:
        parser.error("confidence weight floor must be finite and in (0, 1]")


def validate_schedule_args(parser, args) -> None:
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if (
        not math.isfinite(args.min_learning_rate)
        or not 0 <= args.min_learning_rate <= args.learning_rate
    ):
        parser.error("minimum learning rate must be finite and between zero and peak")
    if not math.isfinite(args.warmup_start_factor) or not 0 < args.warmup_start_factor <= 1:
        parser.error("warmup start factor must be finite and in (0, 1]")
    if args.warmup_epochs < 0:
        parser.error("warmup epochs must be nonnegative")
    if args.lr_schedule == "warmup_cosine" and args.warmup_epochs >= args.epochs:
        parser.error("warmup must leave at least one epoch for cosine decay")


def warmup_cosine_factor(
    step: int, *, total_steps: int, warmup_steps: int, start_factor: float, min_factor: float
) -> float:
    """Multiplier for an optimizer update: exact endpoints, no cosine rebound."""
    step = max(0, step)
    if step < warmup_steps:
        fraction = step / (warmup_steps - 1) if warmup_steps > 1 else 1.0
        return start_factor + (1 - start_factor) * fraction
    decay_steps = total_steps - warmup_steps
    fraction = min(1.0, (step - warmup_steps) / max(1, decay_steps - 1))
    return min_factor + (1 - min_factor) * (1 + math.cos(math.pi * fraction)) / 2


def build_scheduler(optimizer, args, training_examples: int):
    if args.lr_schedule == "constant":
        return None
    steps_per_epoch = math.ceil(training_examples / args.batch_size)
    factor = partial(
        warmup_cosine_factor,
        total_steps=args.epochs * steps_per_epoch,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
        start_factor=args.warmup_start_factor,
        min_factor=args.min_learning_rate / args.learning_rate,
    )
    return th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def configure_device(args: argparse.Namespace) -> str:
    """Constrain only this process, without changing another GPU workload."""
    if args.cpu_threads is not None:
        th.set_num_threads(args.cpu_threads)
    device = args.device
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    if device == "cuda":
        if not th.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        if args.gpu_memory_limit_mib is not None:
            total = th.cuda.get_device_properties(0).total_memory
            limit = int(args.gpu_memory_limit_mib * 1024**2)
            if limit > total:
                raise ValueError("GPU memory limit exceeds device capacity")
            free, _ = th.cuda.mem_get_info(0)
            if free < limit + 1024**3:
                raise RuntimeError("insufficient free GPU memory for the limit and 1 GiB headroom")
            th.cuda.set_per_process_memory_fraction(limit / total, device=0)
            # Avoid expensive autotuning and its transient workspace allocations.
            th.backends.cudnn.benchmark = False
    elif args.gpu_memory_limit_mib is not None:
        raise ValueError("GPU memory limits require a CUDA device")
    return device


def yield_device(device: str, pause: float) -> None:
    """Finish this batch before yielding; queued CUDA work must not spill over."""
    if pause > 0:
        if device == "cuda":
            th.cuda.synchronize()
        time.sleep(pause)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def save_checkpoint(model, path: Path) -> None:
    """Publish a complete CPU state dict; never leave a half-written best model."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    th.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary)
    temporary.replace(path)


def relabel_suits(
    p_: th.Tensor, m_: th.Tensor, t_: th.Tensor
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Randomly relabel the three non-trump suits of each sample.

    Trump is held fixed (it is not exchangeable with the others), the
    plane rows are permuted, and the 52 card actions are permuted in
    lockstep so masks and targets keep pointing at the same physical
    cards. Action values are invariant under this relabelling, so every
    permuted sample is a genuine extra training example.
    """
    b = p_.shape[0]
    dev = p_.device
    trump = p_[:, TRUMP_PLANE].amax(dim=2).argmax(dim=1)  # (b,)
    # Per row: the 3 non-trump suits in ascending order, shuffled.
    others = th.argsort(
        (th.arange(NUM_SUITS_, device=dev).view(1, -1) == trump.view(-1, 1)).to(th.int64),
        dim=1,
        stable=True,
    )[:, :3]
    shuffled = th.gather(others, 1, th.argsort(th.rand(b, 3, device=dev), dim=1))
    perm = th.empty((b, NUM_SUITS_), dtype=th.int64, device=dev)
    perm.scatter_(1, others, shuffled)
    perm.scatter_(1, trump.view(-1, 1), trump.view(-1, 1))
    # perm[i, s] = which original suit now sits in row s.
    pp = th.gather(p_, 2, perm.view(b, 1, NUM_SUITS_, 1).expand(-1, p_.shape[1], -1, p_.shape[3]))
    card_idx = (
        perm.view(b, NUM_SUITS_, 1) * NUM_RANKS_ + th.arange(NUM_RANKS_, device=dev).view(1, 1, -1)
    ).reshape(b, -1)
    full = th.cat([card_idx, th.arange(52, 56, device=dev).expand(b, -1)], dim=1)
    return pp, th.gather(m_, 1, full), th.gather(t_, 1, full)


def _optimal_sets(masks: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-decision optimal-action set, and which decisions are decisive.

    A decision is decisive when at least one legal action is strictly worse
    than the best: if every action ties there is nothing to learn or score.
    """
    legal_mask = masks > 0
    q = np.where(legal_mask, targets, -np.inf)
    opt = legal_mask & (q >= q.max(axis=1, keepdims=True) - TIE_EPS)
    n_legal = legal_mask.sum(axis=1)
    decisive = (opt.sum(axis=1) < n_legal) & (n_legal > 1)
    return opt, decisive


def _optset_acc(
    model,
    idx: np.ndarray,
    tensors: tuple,
    device: str,
    *,
    batch_size: int = 512,
    step_pause: float = 0.0,
) -> tuple[float, float]:
    """Fraction of ``idx`` where the model's pick is in the teacher's optimal set.

    Also returns the random-legal rate over the same rows, so the score is
    always read against its own floor.
    """
    P, S, M, OPT = tensors
    hit = chance = 0.0
    with th.no_grad():
        for s_ in range(0, len(idx), batch_size):
            ix = th.as_tensor(idx[s_ : s_ + batch_size])
            p_, sc_, m_, o_ = (x[ix].to(device) for x in (P, S, M, OPT))
            pick = model(p_, sc_).masked_fill(m_ == 0, -1e9).argmax(1)
            hit += o_.gather(1, pick[:, None]).sum().item()
            chance += (o_.sum(1).float() / m_.sum(1).float()).sum().item()
            yield_device(device, step_pause)
    return hit / len(idx), chance / len(idx)


def training_manifest(args, counts: tuple[int, int, int, int]) -> dict:
    """Record the exact dataset and implementation used by this run."""
    n_train, n_validation, decisive_train, decisive_validation = counts
    return {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "torch_version": str(th.__version__),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training_examples": n_train,
        "validation_examples": n_validation,
        "decisive_training_examples": decisive_train,
        "decisive_validation_examples": decisive_validation,
        "training_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "feature_source_sha256": hashlib.sha256(
            Path(sys.modules[_build.__module__].__file__).read_bytes()
        ).hexdigest(),
        "public_feature_source_sha256": hashlib.sha256(
            Path(sys.modules[_build.__module__].PUBLIC_FEATURE_SOURCE).read_bytes()
        ).hexdigest()
        if hasattr(sys.modules[_build.__module__], "PUBLIC_FEATURE_SOURCE")
        else None,
        "dataset": [
            {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
            for path in matching_files(args.data_glob)
        ],
        "status": "running",
    }


def export_numpy(model, checkpoint: Path, output: Path, features: tuple, validation) -> float:
    """Export best weights only after checking values and legal-action choices."""
    from deephokm.nn.numpy_qnet import NumpyQNet  # noqa: PLC0415

    planes, scalars, masks = features
    state = th.load(checkpoint, map_location="cpu", weights_only=True)
    params = {key: value.numpy() for key, value in state.items()}
    model.cpu().load_state_dict(state)
    model.eval()
    check = validation[:8]
    with th.no_grad():
        expected = model(th.as_tensor(planes[check]), th.as_tensor(scalars[check])).numpy()
    actual = NumpyQNet(params)(planes[check], scalars[check])
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    if not np.array_equal(
        np.where(masks[check] > 0, actual, -np.inf).argmax(1),
        np.where(masks[check] > 0, expected, -np.inf).argmax(1),
    ):
        raise AssertionError("numpy export changed a legal-action decision")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **params)
    temporary.replace(output)
    print(f"verified numpy export -> {output}", flush=True)
    return float(np.abs(actual - expected).max())


def training_confidence_weights(masks, targets, training_indices, *, gap_scale, weight_floor):
    """Fixed, positive teacher-gap weights; normalize over TRAINING rows only.

    Confidence is a heuristic, not a calibrated probability. Compare the best
    Q with the best *strictly suboptimal* legal action: several equal best
    actions are an informative optimal set, not automatically a noisy example.
    Forced/all-tied rows must have already been removed. Validation examples
    never affect normalization and are never scored with these weights.
    """
    if not math.isfinite(gap_scale) or gap_scale <= 0:
        raise ValueError("confidence gap scale must be finite and positive")
    if not math.isfinite(weight_floor) or not 0 < weight_floor <= 1:
        raise ValueError("confidence weight floor must be finite and in (0, 1]")
    masks, targets, indices = np.asarray(masks), np.asarray(targets), np.asarray(training_indices)
    if masks.shape != targets.shape or masks.ndim != LABEL_MATRIX_DIMENSIONS:
        raise ValueError("masks and targets must be matching matrices")
    if (
        indices.ndim != 1
        or not len(indices)
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= len(targets))
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("training indices must be nonempty, unique and in range")
    selected_masks, selected_targets = masks[indices], targets[indices]
    if not np.isin(selected_masks, (0, 1)).all() or not np.isfinite(selected_targets).all():
        raise ValueError("training masks/targets must be binary/finite")
    optimal, decisive = _optimal_sets(selected_masks, selected_targets)
    if not decisive.all():
        raise ValueError("confidence weighting requires decisive training examples only")
    legal_q = np.where(selected_masks > 0, selected_targets, -np.inf)
    best_worse = np.where(~optimal, legal_q, -np.inf).max(1)
    gap = legal_q.max(1).astype(np.float64) - best_worse.astype(np.float64)
    raw = weight_floor + (1 - weight_floor) * gap / (gap + gap_scale)
    weights = np.ones(len(targets), dtype=np.float32)
    weights[indices] = (raw / raw.mean()).astype(np.float32)
    return weights


def distillation_loss(logits, masks, targets, tau, *, weights=None):
    """Legal-action soft-label cross entropy, optionally weighted per example."""
    logits = logits.masked_fill(masks == 0, -1e9)
    soft_targets = th.softmax(targets.masked_fill(masks == 0, -1e9) / tau, dim=1)
    per_example = -(soft_targets * th.log_softmax(logits, dim=1)).sum(1)
    if weights is not None:
        if weights.shape != per_example.shape:
            raise ValueError("weights must have one value per training example")
        per_example = per_example * weights
    return per_example.mean()


def train_epoch(
    model, optimizer, ordered_indices, tensors: tuple, args, *, scheduler=None, sample_weights=None
) -> float:
    """Train one epoch, respecting the same sharing limits as validation."""
    model.train()
    losses = []
    for start in range(0, len(ordered_indices), args.batch_size):
        indices = th.as_tensor(ordered_indices[start : start + args.batch_size])
        planes, scalars, masks, targets = (x[indices].to(args.device) for x in tensors)
        if args.augment:
            planes, masks, targets = relabel_suits(planes, masks, targets)
        weights = sample_weights[indices].to(args.device) if sample_weights is not None else None
        loss = distillation_loss(model(planes, scalars), masks, targets, args.tau, weights=weights)
        if not th.isfinite(loss).item():
            raise FloatingPointError("non-finite training loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
        yield_device(args.device, args.step_pause)
    return float(np.mean(losses))


def prepare_outputs(args, counts: tuple[int, int, int, int]) -> tuple[Path, Path, dict]:
    """Choose an isolated artifact name and record the run before training."""
    tag = "".join(ch for ch in args.data_glob if ch.isalnum())[:24]
    schedule_tag = (
        f"{args.lr_schedule}_lr{args.learning_rate:g}_warm{args.warmup_epochs}"
        f"_start{args.warmup_start_factor:g}_min{args.min_learning_rate:g}"
    )
    checkpoint = args.checkpoint or Path(
        f"checkpoints/"
        f"soft_{args.arm}_x{args.scale:g}_tau{args.tau:g}_aug{args.augment}"
        f"_features{args.feature_mode}"
        f"_weight{args.loss_weighting}_gap{args.confidence_gap_scale:g}"
        f"_floor{args.confidence_weight_floor:g}"
        f"_{schedule_tag}_epochs{args.epochs}_seed{args.seed}_{tag}.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint.with_suffix(".json")
    if checkpoint.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {checkpoint}")
    manifest = training_manifest(args, counts)
    write_json_atomic(metadata_path, manifest)
    return checkpoint, metadata_path, manifest


def fit(
    model, optimizer, tensors, *, tr, va, args, ckpt, metadata_path, manifest, sample_weights=None
) -> float:
    """Keep every strict best checkpoint, but stop on significant stagnation."""
    P, S, M, T, OPT = tensors
    stopper = EarlyStopping(args.early_stopping_patience, args.early_stopping_min_delta)
    scheduler = build_scheduler(optimizer, args, len(tr))
    best = -math.inf
    rng = np.random.default_rng(args.seed)
    manifest.update(
        stopped_early=False,
        stop_reason="epochs_exhausted",
        parameters=sum(p.numel() for p in model.parameters()),
        training_monitor_examples=min(len(tr), len(va)),
    )
    for ep in range(args.epochs):
        order = rng.permutation(len(tr))
        lr_start = optimizer.param_groups[0]["lr"] if optimizer is not None else args.learning_rate
        schedule_options = {"scheduler": scheduler} if scheduler is not None else {}
        if sample_weights is not None:
            schedule_options["sample_weights"] = sample_weights
        loss = train_epoch(model, optimizer, tr[order], (P, S, M, T), args, **schedule_options)
        model.eval()
        scoring = {"batch_size": args.batch_size, "step_pause": args.step_pause}
        va_acc, va_chance = _optset_acc(model, va, (P, S, M, OPT), args.device, **scoring)
        should_stop = stopper.update(va_acc)
        if args.lr_schedule == "warmup_cosine" and ep + 1 <= args.warmup_epochs:
            stopper.stale_epochs = 0
            should_stop = False
        tr_acc, _ = _optset_acc(model, tr[: len(va)], (P, S, M, OPT), args.device, **scoring)
        star = ""
        if va_acc > best:
            best = va_acc
            save_checkpoint(model, ckpt)
            manifest.update(
                best_epoch=ep + 1, best_validation_optset=best, best_training_monitor_optset=tr_acc
            )
            star = " *saved"
        manifest.update(
            completed_epochs=ep + 1,
            stale_epochs=stopper.stale_epochs,
            meaningful_best_validation_optset=stopper.meaningful_best,
            latest_validation_optset=va_acc,
            latest_training_monitor_optset=tr_acc,
            learning_rate_start=lr_start,
            scheduler_optimizer_steps=scheduler.last_epoch if scheduler else 0,
        )
        if should_stop and ep + 1 < args.epochs:
            manifest.update(stopped_early=True, stop_reason="validation_plateau")
        write_json_atomic(metadata_path, manifest)
        print(
            f"epoch {ep + 1}: ce={loss:.4f} train_optset={tr_acc:.4f} "
            f"val_optset={va_acc:.4f} (random-legal={va_chance:.4f}) "
            f"lr_start={lr_start:.6g}{star}",
            flush=True,
        )
        if should_stop:
            print(
                f"early stopping: {stopper.stale_epochs} epochs without a gain "
                f"> {stopper.min_delta:g}; keeping best epoch {manifest['best_epoch']}",
                flush=True,
            )
            break
    return best


def validate_test_reservation(args) -> None:
    """Reject reserved games before building inputs, including incidental shards."""
    if args.test_reservation is None:
        return
    reservation = json.loads(args.test_reservation.read_text())
    start, stop = reservation["seed_start"], reservation["seed_stop_exclusive"]
    if (
        reservation.get("schema_version") != 1
        or type(start) is not int
        or type(stop) is not int
        or not 0 <= start < stop
        or reservation.get("matches") != stop - start
    ):
        raise ValueError("invalid final-test reservation")
    excluded = range(start, stop)
    for path in matching_files(args.data_glob):
        with open(path, "rb") as stream:
            seed = pickle.load(stream).get("seed")
        if seed is None or seed in excluded:
            raise ValueError(f"missing seed or reserved final-test game in training: {path}")
    args.test_reservation_sha256 = hashlib.sha256(args.test_reservation.read_bytes()).hexdigest()
    args.test_reservation_source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    qnet_features.ROOT = str(args.data_dir.resolve())
    arm, epochs, scale = args.arm, args.epochs, args.scale
    tau, augment, data_glob = args.tau, bool(args.augment), args.data_glob
    if args.checkpoint is not None and args.checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {args.checkpoint}")
    if args.export_numpy is not None and args.export_numpy.exists():
        raise FileExistsError(f"refusing to overwrite an existing export: {args.export_numpy}")
    validate_test_reservation(args)
    device = configure_device(args)
    args.device = device

    feature_options = {"feature_mode": args.feature_mode} if args.feature_mode != "baseline" else {}
    planes, scalars, masks, targets, n_train = load_features(data_glob, **feature_options)
    n = planes.shape[0]
    opt, decisive = _optimal_sets(masks, targets)
    tr = np.flatnonzero(decisive[:n_train])
    va = np.flatnonzero(decisive[n_train:]) + n_train
    if not len(tr) or not len(va):
        raise ValueError("need decisive examples in both training and validation partitions")
    if not np.isfinite(targets).all():
        raise ValueError("teacher targets must be finite")
    print(
        f"arm={arm} scale={scale} tau={tau} epochs={epochs} data={data_glob}\n"
        f"decisive train={len(tr)}/{n_train}  val={len(va)}/{n - n_train}  "
        f"(dropped {1 - decisive.mean():.3f} of all decisions as all-tied)  "
        f"augment={augment}",
        flush=True,
    )

    th.manual_seed(args.seed)
    model_options = {"input_planes": planes.shape[1]} if args.feature_mode != "baseline" else {}
    model = _build(arm, ARMS[arm], scale, **model_options).to(device)
    print(
        f"params={sum(p.numel() for p in model.parameters()):,} device={device} "
        f"batch_size={args.batch_size} gpu_memory_limit_mib={args.gpu_memory_limit_mib} "
        f"step_pause={args.step_pause}",
        flush=True,
    )
    opt_alg = th.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    P, S, M, T, OPT = (th.as_tensor(x) for x in (planes, scalars, masks, targets, opt))

    ckpt, metadata_path, manifest = prepare_outputs(args, (n_train, n - n_train, len(tr), len(va)))
    sample_weights = None
    if args.loss_weighting == "q_gap":
        weights = training_confidence_weights(
            masks,
            targets,
            tr,
            gap_scale=args.confidence_gap_scale,
            weight_floor=args.confidence_weight_floor,
        )
        sample_weights = th.as_tensor(weights)
        manifest["confidence_weights"] = {
            "normalization": "training-only dataset mean, not per minibatch",
            "minimum": float(weights[tr].min()),
            "maximum": float(weights[tr].max()),
            "mean": float(weights[tr].mean()),
            "effective_examples": float(
                weights[tr].sum(dtype=np.float64) ** 2
                / np.square(weights[tr].astype(np.float64)).sum()
            ),
        }
        print("confidence weights: " + json.dumps(manifest["confidence_weights"]), flush=True)
        write_json_atomic(metadata_path, manifest)
    best = fit(
        model,
        opt_alg,
        (P, S, M, T, OPT),
        tr=tr,
        va=va,
        args=args,
        ckpt=ckpt,
        metadata_path=metadata_path,
        manifest=manifest,
        **({"sample_weights": sample_weights} if sample_weights is not None else {}),
    )

    print(f"best val_optset={best:.4f} -> {ckpt}", flush=True)
    model.load_state_dict(th.load(ckpt, map_location=device, weights_only=True))
    if args.export_numpy is not None:
        error = export_numpy(model, ckpt, args.export_numpy, (planes, scalars, masks), va)
        manifest["numpy_export"] = str(args.export_numpy)
        manifest["numpy_max_abs_error"] = error
        manifest["numpy_export_sha256"] = hashlib.sha256(args.export_numpy.read_bytes()).hexdigest()
        feature_module = sys.modules[_build.__module__]
        if hasattr(feature_module, "PUBLIC_FEATURE_SOURCE"):
            write_json_atomic(
                args.export_numpy.with_suffix(".features.json"),
                {
                    "schema_version": 1,
                    "feature_mode": args.feature_mode,
                    "input_planes": planes.shape[1],
                    "public_features_sha256": manifest["public_feature_source_sha256"],
                    "weights_sha256": manifest["numpy_export_sha256"],
                },
            )
    manifest["status"] = "complete"
    write_json_atomic(metadata_path, manifest)


if __name__ == "__main__":
    main()
