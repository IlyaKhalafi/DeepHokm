"""Dataset features and model builders for search-value distillation.

Match-level validation membership stays fixed as new shards arrive. Public
feature channels share their implementation with the NumPy serving policies.
Only load trusted, locally generated pickle shards.
"""

import glob
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import torch as th
from torch import nn

from deephokm.nn import public_features as _public_features

NUM_SUITS, NUM_RANKS, NUM_CARDS, NUM_ACTIONS = 4, 13, 52, 56
NUM_SEATS, BASE_PLANES = 4, 14
PARTNER_ROLE = 2
ROOT = str(Path.cwd() / "data")
DEFAULT_GLOB = "qdata_*.pkl"
FEATURE_SCHEMA_VERSION = 4
PUBLIC_FEATURE_SOURCE = _public_features.__file__
FEATURE_MODES = _public_features.FEATURE_MODES
public_voids = _public_features.public_voids
current_trick = _public_features.current_trick
trick_context_planes = _public_features.trick_context_planes
append_void_planes = _public_features.append_void_planes
CONTEXT_NAMES = _public_features.CONTEXT_NAMES
MIN_DATA_SHARDS = 2


# Features are cached per dataset pattern and validated against the matched
# source files. A long-running generator adds shards over time, so the pattern
# alone is not enough to identify the contents of a partial dataset.
def cache_path(pattern: str) -> str:
    slug = hashlib.sha256(pattern.encode()).hexdigest()[:16]
    return f"{ROOT}/features_{slug}.npz"


def matching_files(pattern: str) -> list[str]:
    """Resolve shard patterns in stable match-index order."""
    files: list[str] = []
    for part in pattern.split(","):
        files.extend(
            sorted(
                glob.glob(f"{ROOT}/{part.strip()}"),
                key=lambda p: int(p.split("_")[-1].split(".")[0]),
            )
        )
    if not files:
        raise FileNotFoundError(f"no shards matched {pattern!r} under {ROOT}")
    return list(dict.fromkeys(os.path.realpath(path) for path in files))


def source_signature(files: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify the exact completed shards used by a feature cache."""
    stats = [os.stat(path) for path in files]
    return (
        np.asarray(files),
        np.asarray([stat.st_size for stat in stats], dtype=np.int64),
        np.asarray([stat.st_mtime_ns for stat in stats], dtype=np.int64),
    )


def load_split_shards(files: list[str]):
    """Load disjoint matches and keep validation membership stable."""
    if len(files) < MIN_DATA_SHARDS:
        raise ValueError("at least two completed shards are required for training and validation")
    data = []
    seeds: set[int] = set()
    for f in files:
        with open(f, "rb") as fh:
            shard = pickle.load(fh)
        if "seed" in shard:
            if shard["seed"] in seeds:
                raise ValueError(f"duplicate match seed {shard['seed']} in {f}")
            seeds.add(shard["seed"])
        data.append(shard)
    # Membership stays fixed while a collector adds files. A validation
    # match must never later become training data because a glob grew.
    train, validation = [], []
    for path, shard in zip(files, data, strict=True):
        key = f"seed:{shard['seed']}" if "seed" in shard else f"shard:{Path(path).name}"
        is_validation = hashlib.sha256(key.encode()).digest()[0] % 4 == 0
        (validation if is_validation else train).append(shard)
    if not train or not validation:
        raise ValueError("need completed shards in both stable train and validation partitions")
    data = train + validation
    return data, sum(len(d["qvals"]) for d in train)


def feature_source_hash():
    """Cache identity includes the pinned, shared public-feature implementation."""
    return hashlib.sha256(
        Path(__file__).read_bytes() + Path(PUBLIC_FEATURE_SOURCE).read_bytes()
    ).hexdigest()


def build_features(
    pattern: str = DEFAULT_GLOB, *, files: list[str] | None = None, feature_mode: str = "baseline"
):
    """Build features from every shard matching ``pattern`` (comma-separated ok)."""
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature mode: {feature_mode}")
    files = matching_files(pattern) if files is None else files
    data, n_train = load_split_shards(files)
    obs = {k: np.concatenate([d["obs"][k] for d in data]) for k in data[0]["obs"]}
    masks = np.concatenate([d["masks"] for d in data]).astype(np.float32)
    qvals = [q for d in data for q in d["qvals"]]
    legals = [lg for d in data for lg in d["legals"]]
    n = len(qvals)

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
    if feature_mode != "baseline":
        planes = np.stack(
            [
                append_void_planes(
                    planes[i], {key: value[i] for key, value in obs.items()}, feature_mode
                )
                for i in range(n)
            ]
        )
    return planes, scalars, masks, targets, n_train


def load_features(pattern: str = DEFAULT_GLOB, *, feature_mode: str = "baseline"):
    """Load a cache only while its source shard set is unchanged."""
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature mode: {feature_mode}")
    cache = cache_path(pattern)
    if feature_mode != "baseline":
        cache = cache.removesuffix(".npz") + f"_{feature_mode}.npz"
    files = matching_files(pattern)
    signature = source_signature(files)
    source_hash = feature_source_hash()
    if os.path.exists(cache):
        with np.load(cache) as z:
            if (
                "schema_version" in z
                and int(z["schema_version"]) == FEATURE_SCHEMA_VERSION
                and "feature_source_sha256" in z
                and str(z["feature_source_sha256"]) == source_hash
                and all(
                    key in z and np.array_equal(z[key], value)
                    for key, value in zip(("files", "sizes", "mtimes_ns"), signature, strict=True)
                )
            ):
                return z["planes"], z["scalars"], z["masks"], z["targets"], int(z["n_train"])
    feature_options = {"feature_mode": feature_mode} if feature_mode != "baseline" else {}
    planes, scalars, masks, targets, n_train = build_features(
        pattern, files=files, **feature_options
    )
    temporary = f"{cache}.{os.getpid()}.tmp"
    with open(temporary, "wb") as stream:
        np.savez_compressed(
            stream,
            planes=planes,
            scalars=scalars,
            masks=masks,
            targets=targets,
            n_train=n_train,
            files=signature[0],
            sizes=signature[1],
            mtimes_ns=signature[2],
            schema_version=FEATURE_SCHEMA_VERSION,
            feature_source_sha256=source_hash,
        )
    os.replace(temporary, cache)
    return planes, scalars, masks, targets, n_train


def build_row_features(
    obs: dict, mask: np.ndarray, *, feature_mode: str = "baseline"
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
    return append_void_planes(planes, obs, feature_mode), scalars


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

    def __init__(self, ch: int = 128, layers: int = 4, input_planes: int = 14) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        cin = input_planes
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
            ctx = h.mean(dim=2, keepdim=True).expand_as(h)  # symmetric over suits
            h = th.nn.functional.gelu(block(th.cat([h, ctx], dim=1)))
        h = h + self.scalar_proj(scalars).view(h.shape[0], -1, 1, 1)
        card_logits = self.card_head(h).flatten(1)  # (N, 52)
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
        self.enc = nn.Sequential(nn.Linear(14 + 32, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        self.suit_mix = nn.Sequential(nn.Linear(2 * d, d), nn.GELU())
        self.ctx = nn.Sequential(nn.Linear(2 * d + 10, d), nn.GELU())
        self.dec = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.trump_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, NUM_SUITS))

    def forward(self, planes: th.Tensor, scalars: th.Tensor) -> th.Tensor:
        n = planes.shape[0]
        feats = planes.permute(0, 2, 3, 1)  # (N,4,13,14)
        ranks = th.arange(NUM_RANKS, device=planes.device)
        rk = self.rank_emb(ranks).view(1, 1, NUM_RANKS, -1).expand(n, NUM_SUITS, -1, -1)
        h = self.enc(th.cat([feats, rk], dim=-1))  # (N,4,13,d)

        suit_desc = self.suit_mix(th.cat([h.mean(dim=2), h.amax(dim=2)], dim=-1))  # (N,4,d)
        global_desc = th.cat([suit_desc.mean(dim=1), suit_desc.amax(dim=1)], dim=-1)
        ctx = self.ctx(th.cat([global_desc, scalars], dim=1))  # (N,d)

        per_card = th.cat(
            [
                h,
                suit_desc.unsqueeze(2).expand(-1, -1, NUM_RANKS, -1),
                ctx.view(n, 1, 1, -1).expand(-1, NUM_SUITS, NUM_RANKS, -1),
            ],
            dim=-1,
        )  # (N,4,13,3d)
        card_logits = self.dec(per_card).squeeze(-1)  # (N,4,13)
        return th.cat([card_logits.flatten(1), self.trump_head(ctx)], dim=1)


ARMS = {"mlp": MLP, "rank_cnn": RankCNN, "grid_cnn": GridCNN, "deepsets": DeepSets}

DEFAULT_EPOCHS = 20
ARG_EPOCHS = 1  # argv index past which an explicit epoch count was given
ARG_SCALE = 2  # ... and a capacity scale
ARG_DATA = 3  # ... and an explicit dataset glob


def _build(arm: str, model_cls: type, scale: float, *, input_planes: int = 14) -> nn.Module:
    """Instantiate an arm at a capacity scale (1.0 = the sweep's baseline)."""
    if input_planes != BASE_PLANES:
        if arm != "rank_cnn":
            raise ValueError("extra feature planes require rank_cnn")
        return model_cls(
            ch=int(128 * scale), layers=4 if scale <= 1 else 8, input_planes=input_planes
        )
    if arm == "mlp":
        return model_cls(width=int(1024 * scale), depth=3 if scale <= 1 else 5)
    if arm in ("rank_cnn", "grid_cnn"):
        return model_cls(ch=int(128 * scale), layers=4 if scale <= 1 else 8)
    return model_cls(d=int(256 * scale))
