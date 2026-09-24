"""Architecture sweep for the Q-network on K=192 search labels.

Every arm consumes the SAME precomputed features, the same train/val split,
the same seed, optimizer, epoch count and batch size; only the network body
differs. Arms are selected by argv[1].

Features (built once, cached):
  planes  (N, 14, 4, 13) float32 -- suit x rank grid:
      0 own hand, 1 seen, 2 on table,
      3 trump suit (broadcast over ranks),
      4 history recency (1.0 = most recent play, decaying, 0 = not played),
      5-8 history role one-hot (who played it: self/next/partner/prev),
      9-12 current-trick role one-hot (who played it),
      13 legality (this card is a legal action right now)
  scalars (N, 10) float32 -- phase(2), tricks_won(2)/13, points(2)/7, seat(4)

The suit axis is exchangeable and the rank axis is ordered, so conv arms
convolve along ranks only and pool over suits; see the CNN docstrings.
"""
import glob
import os
import pickle
import sys

import numpy as np
import torch as th
from torch import nn

sys.path.insert(0, "/home/ubuntu8/ilya/research/DeepHokm/src")

NUM_SUITS, NUM_RANKS, NUM_CARDS, NUM_ACTIONS = 4, 13, 52, 56
CACHE = "/home/ubuntu8/ilya/research/DeepHokm/scratch/arch_features.npz"


def build_features():
    files = sorted(
        glob.glob("/home/ubuntu8/ilya/research/DeepHokm/scratch/qdata_*.pkl"),
        key=lambda p: int(p.split("_")[-1].split(".")[0]),
    )
    data = []
    for f in files:
        with open(f, "rb") as fh:
            data.append(pickle.load(fh))
    obs = {k: np.concatenate([d["obs"][k] for d in data]) for k in data[0]["obs"]}
    masks = np.concatenate([d["masks"] for d in data]).astype(np.float32)
    qvals = [q for d in data for q in d["qvals"]]
    legals = [lg for d in data for lg in d["legals"]]
    n = len(qvals)
    n_train = sum(len(d["qvals"]) for d in data[:6])

    planes = np.zeros((n, 14, NUM_SUITS, NUM_RANKS), dtype=np.float32)
    for i, key in enumerate(("hand", "seen", "trick")):
        planes[:, i] = obs[key].reshape(n, NUM_SUITS, NUM_RANKS)
    planes[:, 3] = obs["trump"].reshape(n, NUM_SUITS, 1)

    hist, hist_role = obs["history"], obs["history_role"]
    n_slots = hist.shape[1]
    rows = np.repeat(np.arange(n), n_slots)
    cards = hist.reshape(-1)
    roles = hist_role.reshape(-1)
    recency = np.tile(1.0 - np.arange(n_slots) / n_slots, n)
    keep = cards >= 0
    r, c, ro, rec = rows[keep], cards[keep], roles[keep], recency[keep]
    planes[r, 4, c // NUM_RANKS, c % NUM_RANKS] = rec
    planes[r, 5 + ro, c // NUM_RANKS, c % NUM_RANKS] = 1.0

    trick_play = obs["trick_play"]
    seat = obs["seat"].argmax(axis=1)
    for s in range(NUM_SUITS):
        card = trick_play[:, s]
        has = card >= 0
        idx = np.flatnonzero(has)
        rel = (s - seat[idx]) % NUM_SUITS
        cc = card[idx]
        planes[idx, 9 + rel, cc // NUM_RANKS, cc % NUM_RANKS] = 1.0

    planes[:, 13] = masks[:, :NUM_CARDS].reshape(n, NUM_SUITS, NUM_RANKS)

    scalars = np.concatenate(
        [
            obs["phase"].astype(np.float32),
            obs["tricks_won"].astype(np.float32) / 13.0,
            obs["game_points"].astype(np.float32) / 7.0,
            obs["seat"].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)

    targets = np.zeros((n, NUM_ACTIONS), dtype=np.float32)
    for i, (lg, q) in enumerate(zip(legals, qvals, strict=True)):
        targets[i, lg] = q
    return planes, scalars, masks, targets, n_train


def load_features():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["planes"], z["scalars"], z["masks"], z["targets"], int(z["n_train"])
    planes, scalars, masks, targets, n_train = build_features()
    np.savez_compressed(
        CACHE, planes=planes, scalars=scalars, masks=masks, targets=targets,
        n_train=n_train,
    )
    return planes, scalars, masks, targets, n_train


def build_row_features(
    obs: dict, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Features for ONE live observation, identical to the batch builder.

    Training reads features out of the label pickles in bulk; play needs them
    one decision at a time. Both must produce byte-identical layouts or the
    net sees a different input distribution at play time than it trained on,
    so this mirrors :func:`build_features` plane for plane and is covered by
    a parity test against it.
    """
    planes = np.zeros((14, NUM_SUITS, NUM_RANKS), dtype=np.float32)
    for i, key in enumerate(("hand", "seen", "trick")):
        planes[i] = np.asarray(obs[key]).reshape(NUM_SUITS, NUM_RANKS)
    planes[3] = np.asarray(obs["trump"]).reshape(NUM_SUITS, 1)

    hist = np.asarray(obs["history"])
    hist_role = np.asarray(obs["history_role"])
    n_slots = len(hist)
    for pos, (card, role) in enumerate(zip(hist, hist_role, strict=True)):
        if card < 0:
            continue
        planes[4, card // NUM_RANKS, card % NUM_RANKS] = 1.0 - pos / n_slots
        planes[5 + int(role), card // NUM_RANKS, card % NUM_RANKS] = 1.0

    trick_play = np.asarray(obs["trick_play"])
    seat = int(np.argmax(np.asarray(obs["seat"])))
    for s in range(NUM_SUITS):
        card = int(trick_play[s])
        if card < 0:
            continue
        rel = (s - seat) % NUM_SUITS
        planes[9 + rel, card // NUM_RANKS, card % NUM_RANKS] = 1.0

    planes[13] = np.asarray(mask)[:NUM_CARDS].reshape(NUM_SUITS, NUM_RANKS)

    scalars = np.concatenate(
        [
            np.asarray(obs["phase"], dtype=np.float32),
            np.asarray(obs["tricks_won"], dtype=np.float32) / 13.0,
            np.asarray(obs["game_points"], dtype=np.float32) / 7.0,
            np.asarray(obs["seat"], dtype=np.float32),
        ]
    ).astype(np.float32)
    return planes, scalars


class MLP(nn.Module):
    """Flatten everything. No structural prior at all -- the control."""

    def __init__(self, width: int = 1024, depth: int = 3) -> None:
        super().__init__()
        d = 14 * NUM_SUITS * NUM_RANKS + 10
        layers: list[nn.Module] = [nn.Linear(d, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, NUM_ACTIONS))
        self.net = nn.Sequential(*layers)

    def forward(self, planes, scalars):
        x = th.cat([planes.flatten(1), scalars], dim=1)
        return self.net(x)


class RankCNN(nn.Module):
    """Convolve along ranks only; suits share weights and mix symmetrically.

    Kernels are (1, k): they span adjacent RANKS, which are genuinely ordered
    (a King neighbours a Queen in strength), and never span suits, which are
    exchangeable labels with no adjacency. Sharing weights across the suit
    axis is exactly the suit-permutation equivariance the rules have.

    Suits still have to talk to each other -- whether a suit is long or void
    matters relative to the others -- so each block appends the MEAN over
    suits, broadcast back. Mean is symmetric, so the mixing keeps
    equivariance rather than breaking it.

    The head decodes per cell with a 1x1 conv, so every card keeps its own
    value; a pooled (permutation-invariant) vector feeds the trump actions.
    """

    def __init__(self, ch: int = 128, layers: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        cin = 14
        for _ in range(layers):
            self.blocks.append(nn.Conv2d(cin * 2, ch, (1, 3), padding=(0, 1)))
            cin = ch
        self.scalar_proj = nn.Linear(10, ch)
        self.card_head = nn.Conv2d(ch, 1, 1)
        self.trump_head = nn.Sequential(
            nn.Linear(ch * 2 + 10, 256), nn.GELU(), nn.Linear(256, NUM_SUITS)
        )

    def forward(self, planes: th.Tensor, scalars: th.Tensor) -> th.Tensor:
        h = planes
        for block in self.blocks:
            ctx = h.mean(dim=2, keepdim=True).expand_as(h)   # symmetric over suits
            h = th.nn.functional.gelu(block(th.cat([h, ctx], dim=1)))
        h = h + self.scalar_proj(scalars).view(h.shape[0], -1, 1, 1)
        card_logits = self.card_head(h).flatten(1)           # (N, 52)
        pooled = th.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        trump_logits = self.trump_head(th.cat([pooled, scalars], dim=1))
        return th.cat([card_logits, trump_logits], dim=1)


class GridCNN(nn.Module):
    """Standard 2D conv over the 4x13 grid -- convolves ACROSS suits too.

    Included deliberately as the wrong-prior baseline: (3, 3) kernels treat
    the suit axis as if adjacent suits were related, which the rules say
    they are not. If this matches RankCNN, the suit-adjacency prior costs
    nothing; if it loses, the exchangeability argument is doing real work.
    """

    def __init__(self, ch=128, layers=4):
        super().__init__()
        blocks, cin = [], 14
        for _ in range(layers):
            blocks += [nn.Conv2d(cin, ch, 3, padding=1), nn.GELU()]
            cin = ch
        self.conv = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(ch * NUM_SUITS * NUM_RANKS + 10, 512),
            nn.GELU(),
            nn.Linear(512, NUM_ACTIONS),
        )

    def forward(self, planes, scalars):
        h = self.conv(planes).flatten(1)
        return self.head(th.cat([h, scalars], dim=1))


class DeepSets(nn.Module):
    """Cards as a set: per-card encoder, symmetric pooling, per-card decoder.

    Each card gets its own feature vector plus a learned RANK embedding (ranks
    are ordered and not exchangeable, so they keep an identity). Cards pool
    within a suit, and suit descriptors pool across suits with mean+max, which
    is permutation-invariant -- relabelling suits cannot change the output.
    The decoder scores every card from its own encoding, its suit's
    descriptor, and the global context.
    """

    def __init__(self, d: int = 256) -> None:
        super().__init__()
        self.rank_emb = nn.Embedding(NUM_RANKS, 32)
        self.enc = nn.Sequential(
            nn.Linear(14 + 32, d), nn.GELU(), nn.Linear(d, d), nn.GELU()
        )
        self.suit_mix = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.ctx = nn.Sequential(nn.Linear(2 * d + 10, d), nn.GELU())
        self.dec = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.trump_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, NUM_SUITS))

    def forward(self, planes: th.Tensor, scalars: th.Tensor) -> th.Tensor:
        n = planes.shape[0]
        feats = planes.permute(0, 2, 3, 1)                              # (N,4,13,14)
        ranks = th.arange(NUM_RANKS, device=planes.device)
        rk = self.rank_emb(ranks).view(1, 1, NUM_RANKS, -1).expand(n, NUM_SUITS, -1, -1)
        h = self.enc(th.cat([feats, rk], dim=-1))                       # (N,4,13,d)

        suit_desc = self.suit_mix(th.cat([h.mean(dim=2), h.amax(dim=2)], dim=-1))  # (N,4,d)
        global_desc = th.cat([suit_desc.mean(dim=1), suit_desc.amax(dim=1)], dim=-1)
        ctx = self.ctx(th.cat([global_desc, scalars], dim=1))           # (N,d)

        per_card = th.cat(
            [
                h,
                suit_desc.unsqueeze(2).expand(-1, -1, NUM_RANKS, -1),
                ctx.view(n, 1, 1, -1).expand(-1, NUM_SUITS, NUM_RANKS, -1),
            ],
            dim=-1,
        )                                                               # (N,4,13,3d)
        card_logits = self.dec(per_card).squeeze(-1)                    # (N,4,13)
        return th.cat([card_logits.flatten(1), self.trump_head(ctx)], dim=1)


ARMS = {"mlp": MLP, "rank_cnn": RankCNN, "grid_cnn": GridCNN, "deepsets": DeepSets}

DEFAULT_EPOCHS = 20
ARG_EPOCHS = 1   # argv index past which an explicit epoch count was given
ARG_SCALE = 2    # ... and a capacity scale


def _build(arm: str, model_cls: type, scale: float) -> nn.Module:
    """Instantiate an arm at a capacity scale (1.0 = the sweep's baseline)."""
    if arm == "mlp":
        return model_cls(width=int(1024 * scale), depth=3 if scale <= 1 else 5)
    if arm in ("rank_cnn", "grid_cnn"):
        return model_cls(ch=int(128 * scale), layers=4 if scale <= 1 else 8)
    return model_cls(d=int(256 * scale))


def main():
    argv = sys.argv[1:]
    arm = argv[0]
    epochs = int(argv[1]) if len(argv) > ARG_EPOCHS else DEFAULT_EPOCHS
    scale = float(argv[2]) if len(argv) > ARG_SCALE else 1.0
    model_cls = ARMS[arm]
    planes, scalars, masks, targets, n_train = load_features()
    n = planes.shape[0]
    print(f"arm={arm} n={n} train={n_train} val={n - n_train}", flush=True)

    device = "cuda" if th.cuda.is_available() else "cpu"
    th.manual_seed(0)
    model = _build(arm, model_cls, scale).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params={n_par:,}", flush=True)
    opt = th.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    P = th.as_tensor(planes)
    S = th.as_tensor(scalars)
    M = th.as_tensor(masks)
    T = th.as_tensor(targets)

    rng = np.random.default_rng(0)
    BATCH = 512
    EPOCHS = epochs
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(n_train)
        losses = []
        for s in range(0, n_train, BATCH):
            idx = th.as_tensor(order[s : s + BATCH])
            p, sc, m, t = (x[idx].to(device) for x in (P, S, M, T))
            pred = model(p, sc)
            loss = (((pred - t) * m) ** 2).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()

        def acc_over(lo: int, hi: int) -> float:
            good = seen = 0
            with th.no_grad():
                for st_ in range(lo, hi, BATCH):
                    ix = th.arange(st_, min(st_ + BATCH, hi))
                    p_, sc_, m_, t_ = (x[ix].to(device) for x in (P, S, M, T))
                    pr = model(p_, sc_).masked_fill(m_ == 0, -1e9)
                    tg = t_.masked_fill(m_ == 0, -1e9)
                    good += (pr.argmax(1) == tg.argmax(1)).sum().item()
                    seen += len(ix)
            return good / seen

        val_acc = acc_over(n_train, n)
        # Same number of rows as val, from train: a like-for-like bias read.
        train_acc = acc_over(0, min(n - n_train, n_train))
        print(
            f"epoch {ep + 1}: train_mse={np.mean(losses):.4f}, "
            f"train_argmax_acc={train_acc:.4f}, val_argmax_acc={val_acc:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
