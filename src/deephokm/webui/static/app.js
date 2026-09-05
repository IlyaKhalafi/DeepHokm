/* DeepHokm frontend: a thin renderer and action submitter.
   All rules live on the server; this file only renders state and posts actions. */

const SUITS = ["♣", "♦", "♥", "♠"]; // clubs, diamonds, hearts, spades
const SUIT_NAMES = ["Clubs", "Diamonds", "Hearts", "Spades"];
const RED_SUITS = new Set([1, 2]);

let gameId = null;
let state = null;
let autoTimer = null;

const $ = (id) => document.getElementById(id);

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function updateStatusFromState() {
  if (!state) return;
  const banner = $("mode-banner");
  banner.classList.toggle("hidden", state.mode !== "spectate");
  if (state.mode === "spectate") {
    banner.textContent = "Spectator view — four AI players; no hands are shown.";
    setStatus(
      state.terminal
        ? "Match over."
        : `Next up: seat ${state.current_seat}. Press Advance to watch the play.`
    );
    return;
  }
  if (state.terminal) {
    setStatus("Match over — start a new game.");
  } else if (state.current_seat === state.viewer_seat) {
    setStatus(state.phase === "TRUMP_CALL" ? "Your call: choose trump." : "Your turn: play a card.");
  } else {
    setStatus("The model is thinking…");
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) { /* keep default */ }
    throw new Error(detail);
  }
  return response.json();
}

function cardEl(card, { tiny = false, clickable = false, legal = true } = {}) {
  const suit = Math.floor(card / 13);
  const rank = card % 13;
  const rankName = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"][rank];
  const el = document.createElement("div");
  el.className = `card ${RED_SUITS.has(suit) ? "red" : "black"}` +
    (tiny ? " tiny" : "") +
    (clickable ? (legal ? " legal" : " illegal") : "");
  if (clickable && legal) {
    el.addEventListener("click", () => playAction(card));
    el.title = "Play this card";
  } else if (clickable) {
    el.title = "Not a legal play";
  }
  el.innerHTML = `<span class="rank">${rankName}</span><span class="suit">${SUITS[suit]}</span>`;
  return el;
}

function backEl(tiny = true) {
  const el = document.createElement("div");
  el.className = `card back${tiny ? " tiny" : ""}`;
  return el;
}

function render() {
  if (!state) return;
  const mine = state.viewer_seat;
  const partner = (mine + 2) % 4;

  // Result banner for a finished match: the single place the result is shown.
  let banner = document.querySelector(".banner");
  if (state.terminal) {
    if (!banner) {
      banner = document.createElement("div");
      banner.className = "banner";
      $("game").prepend(banner);
    }
    const won = state.winner === 0;
    banner.className = `banner ${won ? "win" : "lose"}`;
    banner.textContent = state.mode === "spectate"
      ? `Match over — team ${state.winner} wins ${state.game_points[0]}–${state.game_points[1]}`
      : won
        ? `You win the match ${state.game_points[0]}–${state.game_points[1]}!`
        : `The model wins the match ${state.game_points[1]}–${state.game_points[0]}.`;
  } else if (banner) {
    banner.remove();
  }

  // scores
  $("points-us").textContent = state.game_points[0];
  $("tricks-us").textContent = state.tricks_won[0];
  $("points-them").textContent = state.game_points[1];
  $("tricks-them").textContent = state.tricks_won[1];

  // trump / turn / phase
  $("trump").textContent = state.trump === null || state.trump === undefined
    ? "Trump: not declared"
    : `Trump: ${SUIT_NAMES[state.trump]} ${SUITS[state.trump]}`;
  $("turn").textContent = state.terminal
    ? "match complete"
    : state.mode === "human" && state.current_seat === mine
      ? "Your turn"
      : `Seat ${state.current_seat} is playing…`;
  const phaseText = state.terminal
    ? `final hand ${state.hand_number}`
    : {
        TRUMP_CALL: "Trump selection",
        CARD_PLAY: "Card play",
        HAND_OVER: "Hand complete",
      }[state.phase] || state.phase;
  $("phase").textContent = state.terminal
    ? phaseText
    : `${phaseText} · hand ${state.hand_number}`;

  // seats: highlight the active one
  for (let seat = 0; seat < 4; seat++) {
    $(`seat-${seat}`).classList.toggle("active", seat === state.current_seat);
    const holder = $(`cards-${seat}`);
    holder.innerHTML = "";
    if (state.mode === "human" && seat === mine) {
      // The viewer's cards live in the bottom hand row; the seat slot stays
      // visible with a count marker so the table cross reads complete.
      if (!state.terminal && state.hand_counts[seat] > 0) {
        const marker = document.createElement("div");
        marker.className = "seat-count";
        marker.textContent = `${state.hand_counts[seat]} cards`;
        holder.appendChild(marker);
      }
      continue;
    }
    const label = $(`seat-${seat}`).querySelector(".seat-label");
    let seatName;
    if (state.mode === "spectate") {
      seatName = `Seat ${seat}`;
    } else if (seat === mine) {
      seatName = "YOU";
    } else if (seat === partner) {
      seatName = "Partner";
    } else {
      seatName = `Seat ${seat}`;
    }
    const teamTag = state.mode === "spectate"
      ? (seat % 2 === 0 ? "team A" : "team B")
      : (seat % 2 === 0 ? "your team" : "opponent");
    label.textContent = `${seatName}${seat === state.hakem ? " · hakem" : ""} · ${teamTag}`;
    // Compact backs row capped at a glanceable width, with an explicit count
    // so nothing has to be counted by eye.
    const shown = Math.min(state.hand_counts[seat], 8);
    for (let i = 0; i < shown; i++) holder.appendChild(backEl());
    const tag = document.createElement("span");
    tag.className = "seat-count-tag";
    tag.textContent = `${state.hand_counts[seat]}`;
    holder.appendChild(tag);
  }

  // table: seats that have played show their card (plus their remaining
  // count); the first entry led the trick.
  for (const [i, entry] of state.table.entries()) {
    const holder = $(`cards-${entry.seat}`);
    holder.innerHTML = "";
    const el = cardEl(entry.card, { tiny: true });
    if (i === 0) el.classList.add("led");
    if (i === state.table.length - 1) el.classList.add("last-play");
    holder.appendChild(el);
    const tag = document.createElement("span");
    tag.className = "seat-count-tag";
    tag.textContent = `${state.hand_counts[entry.seat]}`;
    holder.appendChild(tag);
  }

  // hand: in spectate mode the API sends no private hand; the bottom row is
  // removed so the layout reads as a spectator view, not a missing player.
  const handEl = $("hand");
  handEl.classList.toggle("spectate", state.mode === "spectate");
  let hint = document.querySelector(".hand-hint");
  if (!hint) {
    hint = document.createElement("div");
    hint.className = "hand-hint";
    handEl.after(hint);
  }
  const playable = state.mode === "human" && !state.terminal
    ? state.legal_actions.filter((a) => a < 52).length
    : 0;
  hint.textContent =
    state.mode === "spectate"
      ? ""
      : state.terminal
        ? ""
        : playable > 0
          ? `${playable} playable — highlighted cards`
          : state.phase === "TRUMP_CALL"
            ? "choose trump above"
            : "waiting for other players";
  handEl.innerHTML = "";
  const myTurn = !state.terminal && state.current_seat === mine;
  const legalSet = new Set(state.legal_actions);
  const choosingTrump = myTurn && state.phase === "TRUMP_CALL";
  for (const c of state.hand) {
    handEl.appendChild(cardEl(c.card, {
      // During trump selection the hand is information, not actions: render
      // every card normally (nothing is playable yet by design).
      clickable: state.mode === "human" && !choosingTrump,
      legal: state.mode === "spectate"
        || choosingTrump
        || (myTurn && legalSet.has(c.card)),
    }));
  }

  // trump picker
  const picker = $("trump-picker");
  const canPick = state.mode === "human" && myTurn && state.phase === "TRUMP_CALL" && legalSet.has(52);
  picker.classList.toggle("hidden", !canPick);

  // controls
  const spectating = state.mode === "spectate";
  $("step").classList.toggle("hidden", !spectating);
  $("auto").classList.toggle("hidden", !spectating);
  $("step").disabled = state.terminal;
  $("auto").disabled = state.terminal;
  if (spectating) {
    $("step").textContent = state.terminal ? "Match over" : "Advance one play";
  }

  updateStatusFromState();
}

async function refresh() {
  try {
    state = await api(`/api/games/${gameId}`);
    render();
    updateStatusFromState();
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function playAction(action) {
  try {
    state = await api(`/api/games/${gameId}/action`, {
      method: "POST",
      body: JSON.stringify({ action }),
    });
    render();
    updateStatusFromState();
  } catch (err) {
    setStatus(err.message, true);
    await refresh();
  }
}

async function startGame() {
  const mode = $("mode").value;
  const seedRaw = $("seed").value.trim();
  const body = { mode };
  if (seedRaw !== "") body.seed = parseInt(seedRaw, 10);
  try {
    state = await api("/api/games", { method: "POST", body: JSON.stringify(body) });
    gameId = state.game_id;
    $("setup").classList.add("hidden");
    $("game").classList.remove("hidden");
    setStatus(mode === "human" ? "You play seat 0; the model plays the rest." : "Spectating AI vs AI.");
    render();
  } catch (err) {
    setStatus(err.message, true);
  }
}

async function spectateStep() {
  try {
    state = await api(`/api/games/${gameId}/step`, { method: "POST" });
    render();
  } catch (err) {
    setStatus(err.message, true);
  }
}

function toggleAuto() {
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
    $("auto").textContent = "Auto-play";
    return;
  }
  $("auto").textContent = "Stop";
  autoTimer = setInterval(async () => {
    await spectateStep();
    if (state && state.terminal) {
      clearInterval(autoTimer);
      autoTimer = null;
      $("auto").textContent = "Auto-play";
    }
  }, 900);
}

// Test/QA hook: render an externally fetched state through the same renderer
// the UI uses. Only set when the page exposes the marker (automated harness).
window.__deephokmRender = (externalState) => {
  if (externalState && externalState.game_id) {
    gameId = externalState.game_id;
    state = externalState;
    $("setup").classList.add("hidden");
    $("game").classList.remove("hidden");
    render();
  }
};

document.addEventListener("DOMContentLoaded", () => {
  $("start").addEventListener("click", startGame);
  $("newgame").addEventListener("click", () => {
    if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
    $("game").classList.add("hidden");
    $("setup").classList.remove("hidden");
    setStatus("Ready");
  });
  $("step").addEventListener("click", spectateStep);
  $("auto").addEventListener("click", toggleAuto);
  document.querySelectorAll(".trump-btn").forEach((btn) => {
    btn.addEventListener("click", () => playAction(52 + parseInt(btn.dataset.suit, 10)));
  });
});
