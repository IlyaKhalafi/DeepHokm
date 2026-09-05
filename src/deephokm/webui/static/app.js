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

  // Result banner for a finished match (created once, then updated).
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
      ? `Match over — team ${state.winner} wins (${state.game_points[0]}–${state.game_points[1]})`
      : won
        ? `You win the match ${state.game_points[0]}–${state.game_points[1]}!`
        : `The model wins ${state.game_points[1]}–${state.game_points[0]}.`;
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
    ? state.mode === "spectate"
      ? `Match over — team ${state.winner} wins`
      : state.winner === 0
        ? "Match over — you win!"
        : "Match over — the model wins."
    : state.mode === "human" && state.current_seat === mine
      ? "Your turn"
      : `Seat ${state.current_seat} is playing…`;
  const phaseText = state.terminal
    ? "Match over"
    : {
        TRUMP_CALL: "Trump selection",
        CARD_PLAY: "Card play",
        HAND_OVER: "Hand complete",
      }[state.phase] || state.phase;
  $("phase").textContent = `${phaseText} · hand ${state.hand_number}`;

  // seats: highlight the active one
  for (let seat = 0; seat < 4; seat++) {
    $(`seat-${seat}`).classList.toggle("active", seat === state.current_seat);
    const holder = $(`cards-${seat}`);
    holder.innerHTML = "";
    const label = $(`seat-${seat}`).querySelector(".seat-label");
    let seatName;
    if (state.mode === "spectate") {
      seatName = `Seat ${seat}`;
    } else if (seat === mine) {
      seatName = "You";
    } else if (seat === partner) {
      seatName = "Partner";
    } else {
      seatName = `Seat ${seat}`;
    }
    label.textContent = `${seatName}${seat === state.hakem ? " · hakem" : ""}`;
    for (let i = 0; i < state.hand_counts[seat]; i++) holder.appendChild(backEl());
  }

  // table: replace backs at seats that have played
  for (const entry of state.table) {
    const holder = $(`cards-${entry.seat}`);
    holder.innerHTML = "";
    holder.appendChild(cardEl(entry.card, { tiny: true }));
  }

  // hand: in spectate mode the API sends no private hand; show the seat-0
  // row's backs only.
  const handEl = $("hand");
  handEl.innerHTML = "";
  const myTurn = !state.terminal && state.current_seat === mine;
  const legalSet = new Set(state.legal_actions);
  for (const c of state.hand) {
    handEl.appendChild(cardEl(c.card, {
      clickable: state.mode === "human",
      legal: state.mode === "spectate" || (myTurn && legalSet.has(c.card)),
    }));
  }

  // trump picker
  const picker = $("trump-picker");
  const canPick = state.mode === "human" && myTurn && state.phase === "TRUMP_CALL" && legalSet.has(52);
  picker.classList.toggle("hidden", !canPick);

  // controls
  $("step").classList.toggle("hidden", state.mode !== "spectate");
  $("auto").classList.toggle("hidden", state.mode !== "spectate");
  $("step").disabled = state.terminal;
  $("auto").disabled = state.terminal;
}

async function refresh() {
  try {
    state = await api(`/api/games/${gameId}`);
    render();
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
    setStatus(" ");
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
