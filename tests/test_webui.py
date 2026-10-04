"""Tests for the web UI: REST API, state rendering, model serving."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
import torch as th
from fastapi.testclient import TestClient
from sb3_contrib import MaskablePPO

import deephokm.webui.app as app_module
from deephokm.cards import NUM_CARDS
from deephokm.env import HokmEnv
from deephokm.nn.policy import HokmMaskablePolicy
from deephokm.policies.greedy_policy import GreedyPolicy
from deephokm.policies.random_policy import RandomPolicy
from deephokm.rules.state import HAKEM_FIRST_BATCH, NUM_SEATS
from deephokm.webui.app import app
from deephokm.webui.serving import ServedPolicy
from deephokm.webui.state import GameStore


@pytest.fixture(scope="session")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny trained checkpoint, built once for the whole session.

    Training even 128 steps of MaskablePPO takes minutes on CPU, which is
    fine once but not for every test that needs a served model. Intra-op
    threading is capped because the model is tiny: with the default thread
    count each matrix op pays more in thread coordination than it saves, and
    on a busy machine the process gets starved for cores it never needed.
    """

    path = tmp_path_factory.mktemp("model") / "model.zip"
    threads = th.get_num_threads()
    th.set_num_threads(2)
    try:
        env = HokmEnv(seat=0, opponents=[RandomPolicy(i) for i in range(NUM_SEATS)])
        model = MaskablePPO(
            HokmMaskablePolicy,
            env,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            device="cpu",
        )
        model.learn(total_timesteps=128)
        model.save(str(path))
    finally:
        th.set_num_threads(threads)
    return path


@pytest.fixture()
def client(
    model_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A test client with a real (tiny) trained model served.

    Yields inside a context manager so the app's lifespan (model loading)
    runs exactly as it does under uvicorn.
    """

    monkeypatch.setenv("DEEPHOKM_MODEL", str(model_path))
    # The app prefers the numpy search weights over DEEPHOKM_MODEL when the
    # default archive exists; without this the tests would exercise the full
    # elimination search, which takes minutes per match instead of covering
    # the HTTP contract quickly with the tiny checkpoint above.
    monkeypatch.setenv("DEEPHOKM_QNET", str(tmp_path / "absent.npz"))

    # Reset module-level state so the new model is picked up.

    if hasattr(app_module.app.state, "model"):
        del app_module.app.state.model
    app_module.app.state.model_disabled = False
    app_module._store = GameStore()
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_greedy_policy_mode_skips_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPHOKM_POLICY", "greedy")
    if hasattr(app_module.app.state, "model"):
        del app_module.app.state.model
    app_module.app.state.model_disabled = False
    try:
        served = app_module.get_served()
        assert isinstance(served, GreedyPolicy)
        assert isinstance(app_module._fresh_policy(), GreedyPolicy)
        assert app_module.health()["policy"] == "greedy-baseline"
    finally:
        if hasattr(app_module.app.state, "model"):
            del app_module.app.state.model
        app_module.app.state.model_disabled = False


def test_index_serves_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "DeepHokm" in response.text
    assert "text/html" in response.headers["content-type"]


def test_static_assets_available(client: TestClient) -> None:
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_create_human_game(client: TestClient) -> None:
    """A fresh game's dealt-card total must match its phase.

    Before trump is declared only the hakem holds cards (5); the response
    can also already be in CARD_PLAY if an AI hakem's auto-resolved trump
    call (and possibly a few AI-led plays before the human's turn) already
    happened inside reset() -- in which case every card is accounted for
    between hands and the table.
    """
    response = client.post("/api/games", json={"mode": "human", "seed": 42})
    assert response.status_code == 201
    state = response.json()
    assert state["mode"] == "human"
    assert state["viewer_seat"] == 0
    assert state["phase"] in ("TRUMP_CALL", "CARD_PLAY")
    if state["phase"] == "TRUMP_CALL":
        assert sum(state["hand_counts"]) == HAKEM_FIRST_BATCH
    else:
        assert sum(state["hand_counts"]) + len(state["table"]) == NUM_CARDS
    assert state["game_points"] == [0, 0]


def test_create_game_default_mode(client: TestClient) -> None:
    response = client.post("/api/games", json={})
    assert response.status_code == 201
    assert response.json()["mode"] == "human"


def test_create_game_rejects_bad_mode(client: TestClient) -> None:
    response = client.post("/api/games", json={"mode": "nonsense"})
    assert response.status_code == 422


def test_get_game_roundtrip(client: TestClient) -> None:
    created = client.post("/api/games", json={"mode": "human", "seed": 7}).json()
    fetched = client.get(f"/api/games/{created['game_id']}")
    assert fetched.status_code == 200
    assert fetched.json() == created


def test_get_unknown_game_404(client: TestClient) -> None:
    assert client.get("/api/games/doesnotexist").status_code == 404


def test_full_human_game_playable(client: TestClient) -> None:
    """A full match is playable through the API: trump call through game over."""
    state = client.post("/api/games", json={"mode": "human", "seed": 1}).json()
    game_id = state["game_id"]
    moves = 0
    while not state["terminal"]:
        if state["current_seat"] != 0:
            state = client.post(f"/api/games/{game_id}/step").json()
            continue
        legal = state["legal_actions"]
        assert legal, "viewer's turn but no legal actions"
        action = legal[0]
        response = client.post(f"/api/games/{game_id}/action", json={"action": action})
        assert response.status_code == 200, response.text
        state = response.json()
        moves += 1
        assert moves < 5000, "game did not terminate"
    assert state["winner"] in (0, 1)
    assert max(state["game_points"]) == 7
    # After termination, further actions are rejected.
    response = client.post(f"/api/games/{game_id}/action", json={"action": 52})
    assert response.status_code == 409


def test_illegal_action_rejected(client: TestClient) -> None:
    """Actions off the mask are rejected with 422, never silently applied."""
    state = client.post("/api/games", json={"mode": "human", "seed": 3}).json()
    game_id = state["game_id"]
    # Play until it is the viewer's turn.
    while state["current_seat"] != 0 and not state["terminal"]:
        state = client.get(f"/api/games/{game_id}").json()
    if state["terminal"]:
        pytest.skip("seed produced an immediate terminal state")
    illegal = next(a for a in range(56) if a not in state["legal_actions"])
    response = client.post(f"/api/games/{game_id}/action", json={"action": illegal})
    assert response.status_code == 422


def test_action_out_of_range_rejected(client: TestClient) -> None:
    state = client.post("/api/games", json={"mode": "human", "seed": 5}).json()
    game_id = state["game_id"]
    assert client.post(f"/api/games/{game_id}/action", json={"action": 56}).status_code == 422
    assert client.post(f"/api/games/{game_id}/action", json={"action": -1}).status_code == 422


def test_spectate_game_rejects_actions(client: TestClient) -> None:
    state = client.post("/api/games", json={"mode": "spectate", "seed": 9}).json()
    game_id = state["game_id"]
    response = client.post(f"/api/games/{game_id}/action", json={"action": 52})
    assert response.status_code == 400


def test_spectate_game_steps_to_completion(client: TestClient) -> None:
    state = client.post("/api/games", json={"mode": "spectate", "seed": 11}).json()
    game_id = state["game_id"]
    steps = 0
    while not state["terminal"]:
        response = client.post(f"/api/games/{game_id}/step")
        assert response.status_code == 200
        state = response.json()
        steps += 1
        assert steps < 5000, "spectate game did not terminate"
    assert state["winner"] in (0, 1)
    assert max(state["game_points"]) == 7


def test_spectate_mode_hides_private_hands(client: TestClient) -> None:
    """Spectators must never receive any seat's private hand or actions."""
    state = client.post("/api/games", json={"mode": "spectate", "seed": 31}).json()
    game_id = state["game_id"]
    for _ in range(10):
        state = client.post(f"/api/games/{game_id}/step").json()
        assert state["hand"] == [], "spectator received a private hand"
        assert state["legal_actions"] == [], "spectator received legal actions"
        assert "seed" not in state, "seed exposes the deterministic deal"
        if state["terminal"]:
            break


def test_human_state_omits_seed(client: TestClient) -> None:
    """The seed must never appear in any public state (cheating oracle)."""
    state = client.post("/api/games", json={"mode": "human", "seed": 41}).json()
    assert "seed" not in state


def test_human_game_rejects_step_on_viewer_turn(client: TestClient) -> None:
    """Stepping is allowed for AI seats only; never for the viewer's seat."""
    state = client.post("/api/games", json={"mode": "human", "seed": 13}).json()
    game_id = state["game_id"]
    # Seat 0 is the viewer; step past any AI plies until it is our turn.
    while state["current_seat"] != 0 and not state["terminal"]:
        state = client.post(f"/api/games/{game_id}/step").json()
    assert state["current_seat"] == 0
    response = client.post(f"/api/games/{game_id}/step")
    assert response.status_code == 409


def test_public_state_has_no_hidden_information(client: TestClient) -> None:
    """The public state must not leak other seats' private cards."""
    state = client.post("/api/games", json={"mode": "human", "seed": 17}).json()
    game_id = state["game_id"]
    # Play a few actions to reach a mid-hand state.
    for _ in range(6):
        current = client.get(f"/api/games/{game_id}").json()
        if current["terminal"] or current["current_seat"] != 0:
            break
        client.post(f"/api/games/{game_id}/action", json={"action": current["legal_actions"][0]})
    state = client.get(f"/api/games/{game_id}").json()
    # The response must contain only: the viewer's hand, played cards, counts.
    assert "hand" in state and all(
        isinstance(c, dict) and {"card", "name"} == set(c) for c in state["hand"]
    )
    assert "hand_counts" in state and len(state["hand_counts"]) == 4
    # No other seat's private cards appear: the only card lists are the
    # viewer's hand and the public table.
    hand_cards = {c["card"] for c in state["hand"]}
    table_cards = {entry["card"] for entry in state["table"]}
    assert all(0 <= c < 52 for c in hand_cards | table_cards)
    assert len(state["hand"]) == state["hand_counts"][0]


def test_table_shows_play_order_and_seats(client: TestClient) -> None:
    """The table entries carry seat attribution for each played card."""
    state = client.post("/api/games", json={"mode": "spectate", "seed": 19}).json()
    game_id = state["game_id"]
    seen_mid_trick = False
    for _ in range(40):
        response = client.post(f"/api/games/{game_id}/step")
        state = response.json()
        if 2 <= len(state["table"]) <= 3:
            seen_mid_trick = True
            seats = [entry["seat"] for entry in state["table"]]
            assert len(set(seats)) == len(seats), "a seat appears twice on the table"
            assert all(0 <= s < 4 for s in seats)
            break
        if state["terminal"]:
            break
    assert seen_mid_trick, "never observed a mid-trick table"


def test_trump_selection_flow(client: TestClient) -> None:
    """When the viewer is hakem, the trump-call decision is exposed."""
    for seed in range(40):
        state = client.post("/api/games", json={"mode": "human", "seed": seed}).json()
        if state["phase"] == "TRUMP_CALL" and state["current_seat"] == 0:
            assert set(state["legal_actions"]) == {52, 53, 54, 55}
            assert state["trump"] is None
            response = client.post(f"/api/games/{state['game_id']}/action", json={"action": 54})
            assert response.status_code == 200
            after = response.json()
            assert after["trump"] == 2
            assert after["phase"] == "CARD_PLAY"
            return
    pytest.fail("no seed made the viewer the hakem")


def test_game_store_eviction() -> None:
    store = GameStore(max_games=3)

    for i in range(5):
        store.create("human", seed=i, opponents=[RandomPolicy(i) for i in range(NUM_SEATS)])
    assert len(store._games) == 3
    # The oldest two are gone.
    assert store.get(store._games[list(store._games)[0]].id) is not None


def test_game_store_eviction_closes_evicted_policies() -> None:
    """A policy holding a resource (e.g. a search pool) must be released.

    build_opponents puts the SAME policy object in all four opponent slots,
    so the fix must close each distinct object once, not fail or double-close
    on the repeats.
    """

    class ClosableStub:
        def __init__(self) -> None:
            self.closed = 0

        def act(self, observation: object, action_mask: np.ndarray) -> int:
            del observation
            return int(np.flatnonzero(action_mask)[0])

        def close(self) -> None:
            self.closed += 1

    store = GameStore(max_games=1)
    first = ClosableStub()
    store.create("human", seed=0, opponents=[first, first, first, first])
    second = ClosableStub()
    store.create("human", seed=1, opponents=[second, second, second, second])

    assert first.closed == 1, "evicted game's policy was never closed"
    assert second.closed == 0, "the still-live game's policy must not be closed"


def test_served_policy_acts_legally(tmp_path: Path) -> None:

    env = HokmEnv(seat=0, opponents=[RandomPolicy(i) for i in range(NUM_SEATS)])
    model = MaskablePPO(
        HokmMaskablePolicy,
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        device="cpu",
    )
    model.learn(total_timesteps=128)
    path = tmp_path / "m.zip"
    model.save(str(path))
    served = ServedPolicy(str(path))

    obs, info = env.reset(seed=21)
    legal = set(np.flatnonzero(info["action_mask"]).tolist())
    action = served.act(obs, np.asarray(info["action_mask"], dtype=bool))
    assert action in legal


def test_served_policy_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="DEEPHOKM_MODEL"):
        ServedPolicy(str(tmp_path / "missing.zip"))


def test_seed_bounds_enforced(client: TestClient) -> None:
    assert client.post("/api/games", json={"mode": "human", "seed": -1}).status_code == 422
    response = client.post("/api/games", json={"mode": "human", "seed": 2**31})
    assert response.status_code == 422


def test_model_loaded_once_across_games(client: TestClient) -> None:
    """The served policy must be a single shared object across games."""

    first = client.post("/api/games", json={"mode": "human", "seed": 1})
    second = client.post("/api/games", json={"mode": "human", "seed": 2})
    assert first.status_code == second.status_code == 201
    model = app_module.app.state.model
    again = client.post("/api/games", json={"mode": "human", "seed": 3})
    assert again.status_code == 201
    assert app_module.app.state.model is model


def test_concurrent_actions_do_not_double_advance(client: TestClient) -> None:
    """Two racing action posts must not both apply; exactly one wins.

    Drives two threads against the same game at the viewer's turn; the game
    must remain consistent (one action applied, mask still valid).
    """

    state = client.post("/api/games", json={"mode": "human", "seed": 23}).json()
    game_id = state["game_id"]
    while state["current_seat"] != 0 and not state["terminal"]:
        state = client.get(f"/api/games/{game_id}").json()
    if state["terminal"]:
        pytest.skip("immediate terminal state")
    action = state["legal_actions"][0]

    results: list = []

    def submit() -> None:
        response = client.post(f"/api/games/{game_id}/action", json={"action": action})
        results.append(response.status_code)

    threads = [threading.Thread(target=submit) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # At most one request can apply the action; the rest must be rejected
    # (409 not-your-turn or 422 illegal-after-apply) — never a crash. The
    # applied action lets the opponents respond inside step(), so the total
    # card count can drop by several; what matters is a legal, consistent
    # state afterwards (the state round-trips and the game can continue).
    assert all(code in (200, 409, 422) for code in results), results
    assert sum(code == 200 for code in results) == 1, results
    after = client.get(f"/api/games/{game_id}").json()
    assert 0 <= sum(after["hand_counts"]) <= 52
    assert after["phase"] in ("TRUMP_CALL", "CARD_PLAY", "HAND_OVER", "MATCH_OVER")
