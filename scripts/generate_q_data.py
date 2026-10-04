"""Generate resumable, full-action K-sample Q labels for the Q-hybrid policy.

Each worker owns match indices ``worker, worker + workers, ...``. A completed
match is one atomic pickle, directly readable by ``scripts.train_qnet``.
Workers can be restarted with the same arguments without duplicating work.

The controlled team's *trajectory* follows the legal depth-two search policy:
the first half of the worlds nominates an action and the independent second
half applies its paired sign-test gate against greedy. Every legal action is
scored in both halves to provide K-sample labels for training.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np

from deephokm.cards import NUM_CARDS, NUM_RANKS
from deephokm.env.spaces import Observation, mask_for, observation_for
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.legal_depth_search import (
    MIN_N_SAMPLES,
    LegalDepthSearchPolicy,
    _sign_test_p_value,
)
from deephokm.policies.search import sample_determinized_hands
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import NUM_SEATS, Phase, team_of

Record = tuple[Observation, np.ndarray, np.ndarray, np.ndarray, int, int, float]
SCHEMA_VERSION = 2
TRAJECTORY = "legal-depth-search-sign-test-v2"
CHECKPOINT_DECISIONS = 10


class MatchPayload(TypedDict):
    obs: dict[str, np.ndarray]
    masks: np.ndarray
    legals: list[np.ndarray]
    qvals: list[np.ndarray]
    seed: int
    k: int
    search_depth: int
    trajectory: str
    schema_version: int
    provenance: dict[str, Any]
    teacher_actions: np.ndarray
    greedy_actions: np.ndarray
    p_values: np.ndarray


class MatchCheckpoint(TypedDict):
    seed: int
    k: int
    provenance: dict[str, Any]
    engine: HokmEngine
    search: LegalDepthSearchPolicy
    records: list[Record]


def collector_provenance(k: int) -> dict[str, Any]:
    """Pin the source defining the labels, their observations, and the rules."""
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__).resolve(),
        *sorted((root / "src/deephokm/policies").glob("*.py")),
        *sorted((root / "src/deephokm/rules").glob("*.py")),
        root / "src/deephokm/cards.py",
        root / "src/deephokm/env/spaces.py",
    ]
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source.relative_to(root)).encode())
        digest.update(source.read_bytes())
    policy = LegalDepthSearchPolicy(n_samples=k, search_depth=2)
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory": TRAJECTORY,
        "source_sha256": digest.hexdigest(),
        "k": k,
        "search_depth": 2,
        "max_p_value": policy.max_p_value,
        "max_rollout_plies": policy.max_rollout_plies,
    }


def score_decision(
    engine: HokmEngine,
    search: LegalDepthSearchPolicy,
    greedy: GreedyPolicy,
    legal: list[int],
) -> tuple[int, Record]:
    """Return the exact seeded teacher action and K-world rollout estimates.

    When the nominee equals greedy, the real teacher stops after selection.
    Extra label worlds then use a copy of its RNG, leaving the next real
    decision's RNG state exactly as ``search.decide()`` would leave it.
    """
    k = search.n_samples
    seat = engine.current_seat()
    hands = engine.state.hands
    team = team_of(seat)
    obs = observation_for(hands, seat, engine.state.game_points)
    mask = mask_for(legal)
    greedy_action = greedy.act(obs, mask)
    own_hand = hands.hands[seat]
    seen = set(hands.played) | set(own_hand)
    unseen = [card for card in range(NUM_CARDS) if card not in seen]
    sizes = [len(hands.hands[s]) for s in range(NUM_SEATS)]
    totals = dict.fromkeys(legal, 0.0)
    select_totals = dict.fromkeys(legal, 0.0)
    clone_rng = random.Random()
    n_select = k // 2
    wins = losses = 0
    best_action = greedy_action
    sampling_rng = search._rng

    for sample_index in range(k):
        sampled_hands = sample_determinized_hands(
            seat, own_hand, unseen, sizes, voids=search.voids.voids, rng=sampling_rng
        )
        values = {
            action: search._score_action(engine, seat, action, team, sampled_hands, clone_rng)
            for action in legal
        }
        for action, value in values.items():
            totals[action] += value
            if sample_index < n_select:
                select_totals[action] += value
        if sample_index + 1 == n_select:
            best_action = max(legal, key=select_totals.__getitem__)
            if best_action == greedy_action or search.max_p_value <= 0.0:
                sampling_rng = random.Random()
                sampling_rng.setstate(search._rng.getstate())
        if sample_index >= n_select and best_action != greedy_action:
            difference = values[best_action] - values[greedy_action]
            wins += difference > 0
            losses += difference < 0

    p_value = _sign_test_p_value(wins, wins + losses)
    switch = (
        best_action != greedy_action and search.max_p_value > 0.0 and p_value <= search.max_p_value
    )
    qvals = np.asarray([totals[action] / k for action in legal], dtype=np.float32)
    action = best_action if switch else greedy_action
    record = (obs, mask, np.asarray(legal, dtype=np.int16), qvals, action, greedy_action, p_value)
    return action, record


def play_match(
    seed: int,
    k: int,
    *,
    checkpoint_path: Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> MatchPayload:
    """Collect every non-forced card decision for one controlled team."""
    greedy = GreedyPolicy()
    provenance = collector_provenance(k) if provenance is None else provenance
    if checkpoint_path is not None and checkpoint_path.exists():
        with checkpoint_path.open("rb") as stream:
            checkpoint: MatchCheckpoint = pickle.load(stream)
        if (checkpoint["seed"], checkpoint["k"], checkpoint["provenance"]) != (seed, k, provenance):
            raise ValueError(f"checkpoint has different provenance: {checkpoint_path}")
        engine, search, records = checkpoint["engine"], checkpoint["search"], checkpoint["records"]
        print(f"seed {seed}: resumed at {len(records)} labeled decisions", flush=True)
    else:
        engine = HokmEngine()
        engine.start_match(seed=seed)
        search = LegalDepthSearchPolicy(n_samples=k, search_depth=2, seed=seed)
        records = []
    team = seed % 2
    checkpointed = len(records)
    start = time.monotonic()
    while engine.state.winner is None:
        seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(seat)
        if hands.phase is Phase.CARD_PLAY and team_of(seat) == team and len(legal) > 1:
            action, record = score_decision(engine, search, greedy, legal)
            records.append(record)
        else:
            obs = observation_for(hands, seat, engine.state.game_points)
            action = greedy.act(obs, mask_for(legal))
        led_suit = hands.current_trick[0][1] // NUM_RANKS if hands.current_trick else None
        outcome = engine.apply_action(action, seat=seat)
        if outcome.card is not None:
            search.observe(seat, outcome.card, led_suit)
        if outcome.hand_complete:
            search.reset_hand()
        if checkpoint_path is not None and (
            len(records) - checkpointed >= CHECKPOINT_DECISIONS or outcome.hand_complete
        ):
            write_atomic(
                checkpoint_path,
                {
                    "seed": seed,
                    "k": k,
                    "provenance": provenance,
                    "engine": engine,
                    "search": search,
                    "records": records,
                },
            )
            checkpointed = len(records)
            print(
                f"seed {seed}: {len(records)} decisions checkpointed, "
                f"points={engine.state.game_points}, elapsed={time.monotonic() - start:.0f}s",
                flush=True,
            )

    if not records:
        raise RuntimeError(f"match {seed} yielded no non-forced decisions")
    observation_keys = dict(records[0][0]).keys()
    return {
        "obs": {
            key: np.stack([cast(np.ndarray, dict(r[0])[key]) for r in records])
            for key in observation_keys
        },
        "masks": np.stack([r[1] for r in records]),
        "legals": [r[2] for r in records],
        "qvals": [r[3] for r in records],
        "seed": seed,
        "k": k,
        "search_depth": 2,
        "trajectory": TRAJECTORY,
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "teacher_actions": np.asarray([r[4] for r in records], dtype=np.int16),
        "greedy_actions": np.asarray([r[5] for r in records], dtype=np.int16),
        "p_values": np.asarray([r[6] for r in records], dtype=np.float64),
    }


def write_atomic(path: Path, payload: object) -> None:
    """Commit a shard or checkpoint without exposing a partially written file."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--matches", type=int, required=True)
    parser.add_argument("--k", type=int, default=6144)
    parser.add_argument("--seed-base", type=int, default=40_000_000)
    parser.add_argument("--prefix", default="qdata")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.workers <= 0 or not 0 <= args.worker < args.workers:
        parser.error("worker must be in [0, workers) and workers must be positive")
    if args.matches <= 0 or args.k < MIN_N_SAMPLES:
        parser.error("matches must be positive and k must be at least 3")
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    if not args.prefix or any(char not in alphabet for char in args.prefix):
        parser.error("prefix must contain only letters, digits, and underscores")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    provenance = collector_provenance(args.k)

    for index in range(args.worker, args.matches, args.workers):
        seed = args.seed_base + index
        output = args.out_dir / f"{args.prefix}_{index}.pkl"
        checkpoint_path = args.out_dir / f".{args.prefix}_{index}.resume"
        lock_path = args.out_dir / f".{args.prefix}_{index}.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if output.exists():
                with output.open("rb") as stream:
                    existing = pickle.load(stream)
                if existing.get("seed") != seed or existing.get("provenance") != provenance:
                    raise ValueError(f"existing shard has different provenance: {output}")
                print(f"worker {args.worker}: skipped completed match {index}", flush=True)
                continue
            print(
                f"worker {args.worker}: starting match {index}, seed={seed}, K={args.k}", flush=True
            )
            payload = play_match(
                seed, args.k, checkpoint_path=checkpoint_path, provenance=provenance
            )
            write_atomic(output, payload)
            checkpoint_path.unlink(missing_ok=True)
        print(
            f"worker {args.worker}: match {index} complete, "
            f"{len(payload['qvals'])} decisions -> {output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
