"""Q-net trained on the quantity the deployment actually uses.

Three corrections to the MSE/argmax setup, each motivated by a measurement:

1. DROP ALL-TIED DECISIONS. 31.8% of decisions have every legal action at the
   teacher's maximum -- there is no decision to make, and grading the model on
   which equal-valued card the teacher listed first is grading a coin flip.
   They contribute only gradient noise.

2. SOFT-TARGET CROSS-ENTROPY instead of MSE on raw Q. The target is
   softmax(Q_legal / tau) with tau set to the teacher's own sampling noise
   scale, so near-ties produce a near-uniform target that correctly says "these
   are equivalent" rather than forcing the model to fit which one won a
   coin flip. Measured: train_mse reached 0.0036 while train argmax stayed at
   0.46, i.e. the Q surface was already fit and argmax was still wrong -- the
   error that matters lives below the MSE scale.

3. REPORT OPTIMAL-SET ACCURACY. Did the model pick an action the teacher
   values at its maximum, rather than the one it happened to rank first. This
   is directly comparable to the measured teacher-vs-itself ceiling
   (0.7407 on decisive decisions).

argv: arm epochs scale [tau] [augment] [data_glob]

Setting augment=1 exploits suit exchangeability as DATA, not as an
architectural constraint. The three non-trump suits can be relabelled without
changing any action value, so each training sample has 3! = 6 valid
presentations. This multiplies the effective dataset by 6 at zero teacher
cost, which matters because the measured bottleneck is data, not capacity
(train optimal-set accuracy reaches the teacher self-agreement ceiling while
val trails it by ~16pp). Note it is a no-op for an exactly suit-equivariant
arm like rank_cnn, which already maps all six presentations to one output --
so the comparison also tells us whether equivariance-by-construction or
equivariance-by-augmentation is the better deal.
"""
import sys

import numpy as np
import torch as th

sys.path.insert(0, "/home/ubuntu8/ilya/research/DeepHokm/scratch")
from arch_sweep import ARMS, NUM_RANKS, NUM_SUITS, _build, load_features  # noqa: E402

NUM_SUITS_, NUM_RANKS_ = NUM_SUITS, NUM_RANKS

TIE_EPS = 1e-9
# Teacher Q is a mean over K worlds of outcomes in [-1, 1]; at K=192 the
# standard error is ~1/sqrt(192) ~= 0.07. Gaps below this are noise, so tau
# at that scale flattens the target exactly where the label is uninformative.
DEFAULT_TAU = 0.07
ARG_TAU = 3      # argv index past which an explicit tau was given
ARG_AUG = 4      # ... and an augmentation flag
ARG_GLOB = 5     # ... and a dataset glob (comma-separated patterns allowed)
TRUMP_PLANE = 3  # planes[:, TRUMP_PLANE] marks the trump suit's row


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
    trump = p_[:, TRUMP_PLANE].amax(dim=2).argmax(dim=1)          # (b,)
    # Per row: the 3 non-trump suits in ascending order, shuffled.
    others = th.argsort(
        (th.arange(NUM_SUITS_, device=dev).view(1, -1) == trump.view(-1, 1)).to(th.int64),
        dim=1, stable=True,
    )[:, :3]
    shuffled = th.gather(others, 1, th.argsort(th.rand(b, 3, device=dev), dim=1))
    perm = th.empty((b, NUM_SUITS_), dtype=th.int64, device=dev)
    perm.scatter_(1, others, shuffled)
    perm.scatter_(1, trump.view(-1, 1), trump.view(-1, 1))
    # perm[i, s] = which original suit now sits in row s.
    pp = th.gather(
        p_, 2, perm.view(b, 1, NUM_SUITS_, 1).expand(-1, p_.shape[1], -1, p_.shape[3])
    )
    card_idx = (perm.view(b, NUM_SUITS_, 1) * NUM_RANKS_
                + th.arange(NUM_RANKS_, device=dev).view(1, 1, -1)).reshape(b, -1)
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
    model, idx: np.ndarray, tensors: tuple, device: str
) -> tuple[float, float]:
    """Fraction of ``idx`` where the model's pick is in the teacher's optimal set.

    Also returns the random-legal rate over the same rows, so the score is
    always read against its own floor.
    """
    P, S, M, OPT = tensors
    hit = chance = 0.0
    with th.no_grad():
        for s_ in range(0, len(idx), 512):
            ix = th.as_tensor(idx[s_ : s_ + 512])
            p_, sc_, m_, o_ = (x[ix].to(device) for x in (P, S, M, OPT))
            pick = model(p_, sc_).masked_fill(m_ == 0, -1e9).argmax(1)
            hit += o_.gather(1, pick[:, None]).sum().item()
            chance += (o_.sum(1).float() / m_.sum(1).float()).sum().item()
    return hit / len(idx), chance / len(idx)


def main() -> None:
    argv = sys.argv[1:]
    arm, epochs, scale = argv[0], int(argv[1]), float(argv[2])
    tau = float(argv[ARG_TAU]) if len(argv) > ARG_TAU else DEFAULT_TAU
    augment = bool(int(argv[ARG_AUG])) if len(argv) > ARG_AUG else False
    data_glob = argv[ARG_GLOB] if len(argv) > ARG_GLOB else "qdata_*.pkl"

    planes, scalars, masks, targets, n_train = load_features(data_glob)
    n = planes.shape[0]
    opt, decisive = _optimal_sets(masks, targets)
    tr = np.flatnonzero(decisive[:n_train])
    va = np.flatnonzero(decisive[n_train:]) + n_train
    print(
        f"arm={arm} scale={scale} tau={tau} epochs={epochs} data={data_glob}\n"
        f"decisive train={len(tr)}/{n_train}  val={len(va)}/{n - n_train}  "
        f"(dropped {1 - decisive.mean():.3f} of all decisions as all-tied)  "
        f"augment={augment}",
        flush=True,
    )

    device = "cuda" if th.cuda.is_available() else "cpu"
    th.manual_seed(0)
    model = _build(arm, ARMS[arm], scale).to(device)
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt_alg = th.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    P, S, M, T, OPT = (
        th.as_tensor(x) for x in (planes, scalars, masks, targets, opt)
    )

    def soft_target(t: th.Tensor, m: th.Tensor) -> th.Tensor:
        return th.softmax(t.masked_fill(m == 0, -1e9) / tau, dim=1)

    # Every hyperparameter that distinguishes an arm belongs in the filename.
    # Keying on the dataset alone let two concurrent tau arms write the same
    # file, leaving a checkpoint whose provenance could not be established.
    tag = "".join(ch for ch in data_glob if ch.isalnum())[:24]
    ckpt = (
        f"/home/ubuntu8/ilya/research/DeepHokm/checkpoints/"
        f"soft_{arm}_x{scale:g}_tau{tau:g}_aug{int(augment)}_{tag}.pt"
    )
    best = 0.0
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr))
        losses = []
        for s in range(0, len(tr), 512):
            ix = th.as_tensor(tr[order[s : s + 512]])
            p_, sc_, m_, t_ = (x[ix].to(device) for x in (P, S, M, T))
            if augment:
                p_, m_, t_ = relabel_suits(p_, m_, t_)
            logits = model(p_, sc_).masked_fill(m_ == 0, -1e9)
            loss = -(soft_target(t_, m_) * th.log_softmax(logits, dim=1)).sum(1).mean()
            opt_alg.zero_grad()
            loss.backward()
            opt_alg.step()
            losses.append(loss.item())
        model.eval()
        tensors = (P, S, M, OPT)
        va_acc, va_chance = _optset_acc(model, va, tensors, device)
        tr_acc, _ = _optset_acc(model, tr[: len(va)], tensors, device)
        star = ""
        if va_acc > best:
            # Keep the best-on-val weights, not the last epoch's: val peaks
            # then drifts down as the model starts memorising.
            best = va_acc
            th.save(model.state_dict(), ckpt)
            star = " *saved"
        print(
            f"epoch {ep + 1}: ce={np.mean(losses):.4f} "
            f"train_optset={tr_acc:.4f} val_optset={va_acc:.4f} "
            f"(random-legal={va_chance:.4f}){star}",
            flush=True,
        )

    print(f"best val_optset={best:.4f} -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
