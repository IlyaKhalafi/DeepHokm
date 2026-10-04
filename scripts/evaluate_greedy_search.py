"""Resume paired full matches of current greedy against K-sample legal search.

``--reference=frozen`` keeps the legal opponent's fallback and rollout policy
at a saved greedy source file. ``--reference=rebased`` uses current greedy in
the opponent too. Every completed match is durable, with policy provenance
and decision latency, so interrupted long benchmarks can resume safely.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from deephokm.env.spaces import mask_for, observation_for
from deephokm.policies import legal_depth_search, oracle_search
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.rules.engine import HokmEngine
from deephokm.rules.state import Phase


def _run_match(index: int, config: dict[str, Any]) -> dict[str, Any]:
    if config["reference"] == "frozen":
        spec = importlib.util.spec_from_file_location(
            "benchmark_frozen_greedy", config["frozen_source"]
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        # These imported constructors are deliberately rebound only in the
        # benchmark worker, not added to either module's public exports.
        legal_depth_search.GreedyPolicy = module.GreedyPolicy  # type: ignore[attr-defined]
        oracle_search.GreedyPolicy = module.GreedyPolicy  # type: ignore[attr-defined]
    seed = config["seed_base"] + index // 2
    greedy_team = index % 2
    engine = HokmEngine()
    engine.start_match(seed=seed)
    greedy = GreedyPolicy()
    search = legal_depth_search.LegalDepthSearchPolicy(
        n_samples=config["k"], search_depth=2, seed=seed
    )
    latencies: list[float] = []
    search_seconds = 0.0
    start = time.monotonic()
    while engine.state.winner is None:
        seat = engine.current_seat()
        hands = engine.state.hands
        legal = engine.legal_actions(seat)
        if seat % 2 == greedy_team:
            obs = observation_for(hands, seat, engine.state.game_points)
            mask = mask_for(legal)
            decision_start = time.perf_counter()
            action = greedy.act(obs, mask)
            if hands.phase is Phase.CARD_PLAY and len(legal) > 1:
                latencies.append(time.perf_counter() - decision_start)
        else:
            decision_start = time.perf_counter()
            action = search.decide(engine)
            search_seconds += time.perf_counter() - decision_start
        led_suit = hands.current_trick[0][1] // 13 if hands.current_trick else None
        outcome = engine.apply_action(action, seat=seat)
        if outcome.card is not None:
            search.observe(seat, outcome.card, led_suit)
        if outcome.hand_complete:
            search.reset_hand()
    return {
        "index": index,
        "seed": seed,
        "greedy_team": greedy_team,
        "greedy_won": engine.state.winner == greedy_team,
        "points": engine.state.game_points,
        "seconds": time.monotonic() - start,
        "greedy_median_ms": float(np.median(latencies)) * 1000,
        "greedy_p95_ms": float(np.percentile(latencies, 95)) * 1000,
        "greedy_max_ms": max(latencies) * 1000,
        "search_seconds": search_seconds,
        "config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=6144)
    parser.add_argument("--matches", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=7_000_000)
    parser.add_argument("--reference", choices=("frozen", "rebased"), default="rebased")
    parser.add_argument("--frozen-source", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.k < legal_depth_search.MIN_N_SAMPLES or args.matches <= 0 or args.workers <= 0:
        parser.error("k must be at least 3; matches and workers must be positive")
    if args.reference == "frozen" and (
        args.frozen_source is None or not args.frozen_source.is_file()
    ):
        parser.error("frozen reference requires an existing --frozen-source")
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for filename in ("greedy_policy.py", "public_endgame.py", "trick_odds.py"):
        digest.update((root / "src/deephokm/policies" / filename).read_bytes())
    config = {
        "benchmark_schema": 2,
        "k": args.k,
        "seed_base": args.seed_base,
        "reference": args.reference,
        "greedy_sha256": digest.hexdigest(),
        "frozen_source": str(args.frozen_source.resolve()) if args.frozen_source else None,
        "frozen_sha256": hashlib.sha256(args.frozen_source.read_bytes()).hexdigest()
        if args.frozen_source
        else None,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    pending = []
    for index in range(args.matches):
        output = args.out_dir / f"match_{index:04d}.json"
        if output.exists():
            result = json.loads(output.read_text())
            if result["config"] != config:
                raise ValueError(f"existing benchmark has different provenance: {output}")
            completed.append(result)
        else:
            pending.append(index)
    print(f"benchmark={config}, remaining={len(pending)}/{args.matches}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_match, index, config): index for index in pending}
        for future in as_completed(futures):
            result = future.result()
            output = args.out_dir / f"match_{result['index']:04d}.json"
            temporary = output.with_suffix(".json.tmp")
            with temporary.open("w") as stream:
                json.dump(result, stream)
            os.replace(temporary, output)
            completed.append(result)
            wins = sum(match["greedy_won"] for match in completed)
            print(
                f"match={result['index']} greedy_won={result['greedy_won']} "
                f"greedy={wins}/{len(completed)} seconds={result['seconds']:.1f} "
                f"greedy_p95_ms={result['greedy_p95_ms']:.3f}",
                flush=True,
            )
    wins = sum(match["greedy_won"] for match in completed)
    print(
        f"FINAL greedy={wins}/{len(completed)} legal={len(completed) - wins}/{len(completed)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
