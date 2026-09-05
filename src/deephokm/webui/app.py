"""The DeepHokm web application: FastAPI server plus static frontend.

The server owns all game state and drives the same rules engine as training;
the frontend never re-implements Hokm rules. The trained model plays the
non-human seats (or all seats in spectate mode).
"""

from __future__ import annotations

import os
import random as _random
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from deephokm.rules.legality import NUM_ACTIONS
from deephokm.webui.serving import ServedPolicy, build_opponents, resolve_model_path
from deephokm.webui.state import GameStore, apply_human_action, public_state

DEFAULT_PORT = 8025

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
    allow_random = os.environ.get("DEEPHOKM_ALLOW_RANDOM_MODEL", "") == "1"
    if served is None and not allow_random:
        raise RuntimeError(
            f"no trained model at {resolve_model_path()}; set DEEPHOKM_MODEL or "
            "set DEEPHOKM_ALLOW_RANDOM_MODEL=1 for random opponents"
        )
    yield


app = FastAPI(title="DeepHokm", version="0.1.0", lifespan=lifespan)


class CreateGameRequest(BaseModel):
    """Request body for POST /api/games."""

    mode: str = Field(default="human", pattern="^(human|spectate)$")
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class ActionRequest(BaseModel):
    """Request body for POST /api/games/{id}/action."""

    action: int = Field(ge=0, le=NUM_ACTIONS - 1)


def get_served() -> ServedPolicy | None:
    """Load the served policy once per process (memoized, race-free)."""
    with _model_lock:
        if not hasattr(app.state, "model") and not getattr(app.state, "model_disabled", False):
            path = resolve_model_path()
            if os.path.exists(path):
                app.state.model = ServedPolicy(path)
            else:
                app.state.model_disabled = True
        return getattr(app.state, "model", None)


@app.get("/health")
def health() -> dict[str, Any]:
    """Container healthcheck."""
    return {
        "status": "ok",
        "model_loaded": getattr(app.state, "model", None) is not None,
        "games": len(_store._games),
    }


@app.post("/api/games", status_code=201)
def create_game(request: CreateGameRequest) -> JSONResponse:
    """Create a game and return its initial public state."""
    seed = request.seed if request.seed is not None else _random.randrange(2**31)
    record = _store.create(
        mode=request.mode,
        seed=seed,
        opponents=build_opponents(get_served()),
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
        if record.env._engine.state.winner is not None:
            raise HTTPException(status_code=409, detail="game is over")
        if record.env._engine.current_seat() != record.viewer_seat:
            raise HTTPException(status_code=409, detail="not the viewer's turn")
        try:
            apply_human_action(record, request.action)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return public_state(record)


@app.post("/api/games/{game_id}/step")
def spectate_step(game_id: str) -> dict[str, Any]:
    """Advance a spectate game by one decision and return the state."""
    record = _store.get(game_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown game id")
    if record.mode != "spectate":
        raise HTTPException(status_code=400, detail="only spectate games can step")
    env = record.env
    # One request at a time per game: the engine is shared mutable state.
    with record.lock:
        if env._engine.state.winner is None:
            seat = env._engine.current_seat()
            if seat == record.viewer_seat:
                mask = env.action_masks()
                obs = env._observation_for(seat)
                action = env.opponents[seat].act(obs, mask)
                env.step(np.int64(action))
                record.moves += 1
        return public_state(record)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the frontend."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    with open(os.path.join(static_dir, "index.html"), encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
