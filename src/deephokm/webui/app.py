"""The DeepHokm web application: FastAPI server plus static frontend.

The server owns all game state and drives the same rules engine as training;
the frontend never re-implements Hokm rules. The trained model plays the
non-human seats (or all seats in spectate mode).
"""

from __future__ import annotations

import multiprocessing
import os
import random as _random
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.rules.legality import NUM_ACTIONS
from deephokm.webui.search_serving import (
    HARD_TRAINING_K,
    SearchServedPolicy,
    describe_policy,
    describe_search_mode,
    load_shared_weights,
    resolve_hard_search_k,
    resolve_hard_weights_path,
    resolve_weights_path,
    resolve_workers,
)
from deephokm.webui.serving import ServedPolicy, build_opponents, resolve_model_path
from deephokm.webui.state import (
    GameDifficulty,
    GameStore,
    advance_one_ply,
    apply_human_action,
    public_state,
)

_store = GameStore()
_model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the trained checkpoint once at startup.

    A missing checkpoint is a deployment error: fail loudly rather than
    silently serving random policies as "the model". Operators who
    deliberately want random opponents set DEEPHOKM_ALLOW_RANDOM_MODEL=1.
    """
    served = get_served()
    if isinstance(served, SearchServedPolicy) and resolve_workers() > 1:
        # Fork the shared hard-mode pool during single-threaded startup.
        _get_search_pool(resolve_workers())
    allow_random = os.environ.get("DEEPHOKM_ALLOW_RANDOM_MODEL", "") == "1"
    if served is None and not allow_random:
        raise RuntimeError(
            f"no Fast network weights at {resolve_weights_path()} and no checkpoint at "
            f"{resolve_model_path()}; set DEEPHOKM_QNET or DEEPHOKM_MODEL, or set "
            "DEEPHOKM_ALLOW_RANDOM_MODEL=1 for random opponents"
        )
    try:
        yield
    finally:
        pool = getattr(app.state, "search_pool", None)
        if pool is not None:
            pool.close()
            pool.join()
            del app.state.search_pool


app = FastAPI(title="DeepHokm", version="0.1.0", lifespan=lifespan)


class CreateGameRequest(BaseModel):
    """Request body for POST /api/games."""

    mode: str = Field(default="human", pattern="^(human|spectate)$")
    difficulty: GameDifficulty = "fast"
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class ActionRequest(BaseModel):
    """Request body for POST /api/games/{id}/action."""

    action: int = Field(ge=0, le=NUM_ACTIONS - 1)


def _get_search_pool(workers: int) -> multiprocessing.pool.Pool | None:
    """The one process pool shared by every search-policy instance.

    Created once, memoized on ``app.state``. Forking lazily on the first
    search decision would fork from inside whatever request thread
    happened to make that call, and any lock a sibling request thread holds
    at that instant (loggers, malloc arenas, C-extension globals) would be
    inherited already held and never released in the child. Creating it
    here -- called from ``lifespan`` before the server accepts requests --
    forks from the single main thread instead, before that race can exist.
    """
    if workers <= 1:
        return None
    if not hasattr(app.state, "search_pool"):
        app.state.search_pool = multiprocessing.get_context("fork").Pool(workers)
    pool: multiprocessing.pool.Pool = app.state.search_pool
    return pool


def get_served() -> SearchServedPolicy | ServedPolicy | GreedyPolicy | None:
    """Load the served policy once per process (memoized, race-free).

    ``DEEPHOKM_POLICY=greedy`` serves the deterministic public-information
    heuristic directly, without loading network weights. Otherwise the numpy
    action-value network with optional search remains the default.

    The NumPy action-value path is preferred because it is the actively
    evaluated and shipped policy. The reinforcement-learning checkpoint is
    retained only as a compatibility fallback for deployments configured with
    DEEPHOKM_MODEL. Benchmark claims live in the results documentation rather
    than in serving code.
    """
    with _model_lock:
        if (
            not hasattr(app.state, "model")
            and os.environ.get("DEEPHOKM_POLICY", "network").strip().lower() == "greedy"
        ):
            app.state.model = GreedyPolicy()
            app.state.model_disabled = False
        elif not hasattr(app.state, "model") and not getattr(app.state, "model_disabled", False):
            weights = resolve_weights_path()
            checkpoint = resolve_model_path()
            if os.path.isfile(weights):
                hard_weights_path = resolve_hard_weights_path()
                if not os.path.isfile(hard_weights_path):
                    raise FileNotFoundError(
                        f"Hard-mode K={HARD_TRAINING_K} weights not found at "
                        f"{hard_weights_path}; set DEEPHOKM_HARD_QNET"
                    )
                # Cache both archives, not policy instances: each game needs
                # a policy object of its own because it holds the engine it
                # decides for. Fast and Hard deliberately use different
                # networks; the latter was trained on K=6144 teacher data.
                fast_weights = load_shared_weights(weights)
                hard_weights = load_shared_weights(hard_weights_path)
                if hard_weights.training_k != HARD_TRAINING_K:
                    contract_path = hard_weights.path.with_suffix(".features.json")
                    raise ValueError(
                        f"Hard-mode weights must declare training_k={HARD_TRAINING_K} "
                        f"in {contract_path}"
                    )
                app.state.fast_weights = fast_weights
                app.state.hard_weights = hard_weights
                app.state.model = SearchServedPolicy(
                    fast_weights,
                    search_k=0,
                )
            elif os.path.isfile(checkpoint):
                app.state.model = ServedPolicy(checkpoint)
            else:
                app.state.model_disabled = True
        return getattr(app.state, "model", None)


def _fresh_policy(
    difficulty: GameDifficulty = "fast",
) -> SearchServedPolicy | ServedPolicy | GreedyPolicy | None:
    """A policy instance for one game.

    The search policy is rebuilt per game so that each game owns the engine
    reference it decides against; the weights and the scorer pool behind it
    are shared. The reinforcement-learning checkpoint is stateless across
    episodes, so the memoized instance is reused as-is.
    """
    if difficulty not in {"fast", "hard"}:
        raise ValueError(f"unknown difficulty {difficulty!r}")
    served = get_served()
    if isinstance(served, SearchServedPolicy):
        search_k = resolve_hard_search_k() if difficulty == "hard" else 0
        workers = resolve_workers() if difficulty == "hard" else 1
        weights = app.state.hard_weights if difficulty == "hard" else app.state.fast_weights
        return SearchServedPolicy(
            weights,
            search_k=search_k,
            workers=workers,
            pool=_get_search_pool(workers) if difficulty == "hard" else None,
        )
    if isinstance(served, GreedyPolicy):
        return GreedyPolicy()
    return served


def _policy_name(policy: SearchServedPolicy | ServedPolicy | GreedyPolicy | None) -> str:
    """Return the stable public identifier for a per-game policy."""
    if isinstance(policy, SearchServedPolicy):
        name = describe_policy(policy)["policy"]
        assert isinstance(name, str)
        return name
    if isinstance(policy, ServedPolicy):
        return "maskable-ppo"
    if isinstance(policy, GreedyPolicy):
        return "greedy-baseline"
    return "random-baseline"


@app.get("/health")
def health() -> dict[str, Any]:
    """Container healthcheck."""
    model = getattr(app.state, "model", None)
    payload: dict[str, Any] = {
        "status": "ok",
        "model_loaded": model is not None,
        "games": len(_store._games),
    }
    payload.update(describe_policy(model if isinstance(model, SearchServedPolicy) else None))
    if isinstance(model, SearchServedPolicy):
        payload["modes"] = {
            "fast": {
                **describe_search_mode(search_k=0, workers=0),
                "weights_sha256": app.state.fast_weights.sha256,
            },
            "hard": {
                **describe_search_mode(search_k=resolve_hard_search_k(), workers=resolve_workers()),
                "training_k": app.state.hard_weights.training_k,
                "weights_sha256": app.state.hard_weights.sha256,
            },
        }
    return payload


@app.post("/api/games", status_code=201)
def create_game(request: CreateGameRequest) -> JSONResponse:
    """Create a game and return its initial public state."""
    seed = request.seed if request.seed is not None else _random.randrange(2**31)
    policy = _fresh_policy(request.difficulty)
    record = _store.create(
        mode=request.mode,
        seed=seed,
        opponents=build_opponents(policy),
        difficulty=request.difficulty,
        policy=_policy_name(policy),
    )
    return JSONResponse(public_state(record), status_code=201)


@app.get("/api/games/{game_id}")
def get_game(game_id: str) -> dict[str, Any]:
    """Return the full public state for the viewer's seat."""
    record = _store.get(game_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown game id")
    # Render under the game lock: the engine mutates hands mid-step and a
    # concurrent read could observe a torn or mid-transition state.
    with record.lock:
        return public_state(record)


@app.post("/api/games/{game_id}/action")
def submit_action(game_id: str, request: ActionRequest) -> dict[str, Any]:
    """Apply the viewer's action and return the resulting public state."""
    record = _store.get(game_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown game id")
    if record.mode != "human":
        raise HTTPException(status_code=400, detail="spectate games take no actions")
    # One request at a time per game: the engine is shared mutable state.
    with record.lock:
        if record.env.engine.state.winner is not None:
            raise HTTPException(status_code=409, detail="game is over")
        if record.env.engine.current_seat() != record.viewer_seat:
            raise HTTPException(status_code=409, detail="not the viewer's turn")
        try:
            apply_human_action(record, request.action)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return public_state(record)


@app.post("/api/games/{game_id}/step")
def spectate_step(game_id: str) -> dict[str, Any]:
    """Advance the game by exactly one ply and return the state.

    Used by both modes. In a human game the client calls this after playing
    its own card, once per AI seat, so each reply appears on the table
    instead of the whole trick resolving inside a single request.

    Every seat is AI-controlled in spectate mode, so this drives the engine
    directly (``engine.apply_action``) for whichever seat is actually due,
    rather than going through ``HokmEnv.step()``: that call is built around
    the single-learner-seat Gym API and auto-plays every *other* seat before
    returning, so one "advance one play" click could silently resolve up to
    a full trick's worth of plays instead of the single ply the button
    promises.
    """
    record = _store.get(game_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown game id")
    # One request at a time per game: the engine is shared mutable state.
    with record.lock:
        engine = record.env.engine
        if engine.state.winner is not None:
            return public_state(record)
        if record.mode != "spectate" and engine.current_seat() == record.viewer_seat:
            raise HTTPException(
                status_code=409,
                detail="it is your turn; the server does not play your card for you",
            )
        advance_one_ply(record)
        return public_state(record)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the frontend."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    with open(os.path.join(static_dir, "index.html"), encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
