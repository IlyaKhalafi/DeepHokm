/* DeepHokm frontend: a thin renderer and action submitter.
   All rules live on the server; this file only renders state and posts actions. */

const SUITS = ["♣", "♦", "♥", "♠"]; // clubs, diamonds, hearts, spades
const SUIT_NAMES = ["Clubs", "Diamonds", "Hearts", "Spades"];
const RED_SUITS = new Set([1, 2]);
const POLICY_LABELS = {
  "numpy-qnet-only": "Fast",
  "numpy-qnet+elimination-search": "Hard",
  "maskable-ppo": "MaskablePPO fallback",
  "greedy-baseline": "Greedy baseline",
  "random-baseline": "Random baseline",
};

const AUTO_PLAY_MS = 900;

let gameId = null;
let state = null;
let autoTimer = null;

const $ = (id) => document.getElementById(id);

function setStatus(text, isError = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function policyLabel(gameState) {
  return POLICY_LABELS[gameState.policy]
    || (gameState.difficulty === "hard" ? "Hard" : "Fast");
}

function updateStatusFromState() {
  if (!state) return;
  const banner = $("mode-banner");
  banner.classList.toggle("hidden", state.mode !== "spectate");
  if (state.mode === "spectate") {
    const strength = policyLabel(state);
    banner.textContent = `Spectator view — four ${strength} players; no hands are shown.`;
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
    setStatus(state.policy === "numpy-qnet+elimination-search"
      ? "Hard mode is searching…"
      : `${policyLabel(state)} is thinking…`);
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
      // FastAPI returns a string detail for our own HTTPExceptions and an
      // array of error objects for request-validation failures; only the
      // string is safe to show as-is.
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = "Invalid request";
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
      ? `Match over — team ${state.winner === 0 ? "A" : "B"} wins ` +
        `${state.game_points[0]}–${state.game_points[1]}`
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
    ? `final hand #${state.hand_number}`
    : {
        TRUMP_CALL: "Trump selection",
        CARD_PLAY: "Card play",
        HAND_OVER: "Hand complete",
      }[state.phase] || state.phase;
  // "#" disambiguates the hand's ordinal from a card count at a glance.
  $("phase").textContent = state.terminal
    ? phaseText
    : `${phaseText} · hand #${state.hand_number}`;
  $("difficulty-badge").textContent = policyLabel(state);

  // seats: highlight the active one
  for (let seat = 0; seat < 4; seat++) {
    $(`seat-${seat}`).classList.toggle("active", seat === state.current_seat);
    const holder = $(`cards-${seat}`);
    holder.innerHTML = "";
    const label = $(`seat-${seat}`).querySelector(".seat-label");
    if (state.mode === "human" && seat === mine) {
      // The viewer's cards live in the bottom hand row; the seat slot stays
      // visible with a count marker so the table cross reads complete. The
      // label still has to be written here: it carries the hakem marker, and
      // a human hakem must be able to see they hold it.
      label.textContent = `YOU${seat === state.hakem ? " · hakem" : ""} · your team`;
      if (!state.terminal && state.hand_counts[seat] > 0) {
        const marker = document.createElement("div");
        marker.className = "seat-count";
        marker.textContent = `${state.hand_counts[seat]} cards`;
        holder.appendChild(marker);
      }
      continue;
    }
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
    tag.title = `${state.hand_counts[seat]} cards left in this hand`;
    tag.setAttribute("aria-label", tag.title);
    holder.appendChild(tag);
  }

  // table: seats that have played show their card (plus their remaining
  // count); the first entry led the trick.
  for (const [i, entry] of state.table.entries()) {
    const holder = $(`cards-${entry.seat}`);
    holder.innerHTML = "";
    const el = cardEl(entry.card, { tiny: true });
    if (i === 0) {
      el.classList.add("led");
      el.title = "led this trick";
    }
    if (i === state.table.length - 1) {
      el.classList.add("last-play");
      el.title = el.title ? `${el.title} · latest play` : "latest play";
      const caret = document.createElement("span");
      caret.className = "play-caret";
      caret.textContent = "\u25B2";
      caret.setAttribute("aria-hidden", "true");
      el.appendChild(caret);
    }
    holder.appendChild(el);
    const tag = document.createElement("span");
    tag.className = "seat-count-tag";
    tag.textContent = `${state.hand_counts[entry.seat]}`;
    tag.title = `${state.hand_counts[entry.seat]} cards left in this hand`;
    tag.setAttribute("aria-label", tag.title);
    holder.appendChild(tag);
  }

  // trick legend: only meaningful once this trick has a card on the table
  // (leading a trick, trump selection, and match-over all show an empty
  // table, where "led the trick" / "latest play" refer to nothing yet).
  // With an empty table, show the trick that just finished rather than a bare
  // felt: this is the only moment the player can see how the trick was won.
  const showingLastTrick = state.table.length === 0 && (state.last_trick || []).length > 0;
  if (showingLastTrick) {
    for (const entry of state.last_trick) {
      const holder = $(`cards-${entry.seat}`);
      if (!holder) continue;
      holder.innerHTML = "";
      const el = cardEl(entry.card, { tiny: true });
      el.classList.add("resolved");
      if (entry.seat === state.last_trick_winner) {
        el.classList.add("trick-winner");
        el.title = "won the trick";
      }
      holder.appendChild(el);
    }
  }
  $("trick-legend").classList.toggle("hidden", state.table.length === 0 && !showingLastTrick);
  // "TABLE" only helps while the felt is empty; with cards down it is noise
  // sitting in the middle of the play area.
  const centre = $("trick-center");
  if (centre) {
    centre.classList.toggle("hidden", state.table.length > 0 || showingLastTrick);
  }
  // Named distinctly: `banner` is already taken in this scope by the
  // match-result element, and redeclaring it breaks the whole script.
  const trickBanner = $("trick-result");
  if (trickBanner) {
    const winner = state.last_trick_winner;
    trickBanner.classList.toggle("hidden", !showingLastTrick || winner === null);
    if (showingLastTrick && winner !== null) {
      const mine = winner % 2 === state.viewer_seat % 2;
      trickBanner.textContent = mine
        ? "Your team wins the trick"
        : "Opponents win the trick";
      trickBanner.classList.toggle("theirs", !mine);
    }
  }

  // hand: in spectate mode the API sends no private hand; the bottom row is
  // removed so the layout reads as a spectator view, not a missing player.
  // At match end there is nothing left to play, so the row and its hint
  // collapse entirely rather than reserving empty space for them.
  const handEl = $("hand");
  handEl.classList.toggle("spectate", state.mode === "spectate");
  handEl.classList.toggle("hidden", state.terminal);
  let hint = document.querySelector(".hand-hint");
  if (!hint) {
    hint = document.createElement("div");
    hint.className = "hand-hint";
    handEl.after(hint);
  }
  hint.classList.toggle("hidden", state.terminal);
  const playable = state.mode === "human" && !state.terminal
    ? state.legal_actions.filter((a) => a < 52).length
    : 0;
  hint.textContent =
    state.mode === "spectate"
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

// Pauses between plies. The server resolves a ply instantly, so without a
// deliberate pause every reply and the whole trick would land in one frame and
// the player would never see what the other seats played.
const REPLY_PAUSE_MS = 850;
const TRICK_PAUSE_MS = 1400;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

let advancing = false;

async function playAction(action) {
  if (advancing) return;           // ignore clicks while the table is resolving
  try {
    state = await api(`/api/games/${gameId}/action`, {
      method: "POST",
      body: JSON.stringify({ action }),
    });
    render();
    updateStatusFromState();
    await advanceUntilMyTurn();
  } catch (err) {
    setStatus(err.message, true);
    await refresh();
  }
}

/**
 * Step the AI seats one at a time so each reply is visible.
 *
 * The action endpoint applies only the player's own card. Each remaining seat
 * is advanced by its own request with a pause between, and a longer pause once
 * a trick completes so the winning play can be read before the next trick
 * starts.
 */
async function advanceUntilMyTurn() {
  advancing = true;
  try {
    let guard = 0;
    while (
      state && !state.terminal &&
      state.current_seat !== state.viewer_seat &&
      guard++ < 60
    ) {
      const tableBefore = state.table.length;
      await sleep(tableBefore === 0 && state.last_trick.length ? TRICK_PAUSE_MS : REPLY_PAUSE_MS);
      state = await api(`/api/games/${gameId}/step`, { method: "POST" });
      render();
      updateStatusFromState();
    }
    // Hold the completed trick on screen before the player acts again.
    if (state && !state.terminal && state.table.length === 0 && state.last_trick.length) {
      await sleep(TRICK_PAUSE_MS);
      render();
    }
  } finally {
    advancing = false;
    render();
  }
}

async function startGame() {
  // No seed is sent: the server picks one. The API still accepts an explicit
  // seed for reproducing a deal, which is what the evaluation harness uses.
  const mode = $("mode").value;
  const difficulty = $("difficulty").value;
  const body = { mode, difficulty };
  try {
    state = await api("/api/games", { method: "POST", body: JSON.stringify(body) });
    gameId = state.game_id;
    $("setup").classList.add("hidden");
    $("game").classList.remove("hidden");
    const strength = policyLabel(state);
    setStatus(mode === "human"
      ? `You play seat 0; ${strength} plays the rest.`
      : `Spectating ${strength} vs ${strength}.`);
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

function stopAuto() {
  if (autoTimer) clearTimeout(autoTimer);
  autoTimer = null;
  $("auto").textContent = "Auto-play";
}

function toggleAuto() {
  if (autoTimer) {
    stopAuto();
    return;
  }
  $("auto").textContent = "Stop";
  // Chained timeouts, not setInterval: one model decision can take longer
  // than the tick, and overlapping requests would queue up behind the
  // server's per-game lock.
  const tick = async () => {
    await spectateStep();
    if (!autoTimer) return;
    if (state && state.terminal) {
      stopAuto();
      return;
    }
    autoTimer = setTimeout(tick, AUTO_PLAY_MS);
  };
  autoTimer = setTimeout(tick, AUTO_PLAY_MS);
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
    stopAuto();
    $("game").classList.add("hidden");
    $("setup").classList.remove("hidden");
    setStatus("");
  });
  $("step").addEventListener("click", spectateStep);
  $("auto").addEventListener("click", toggleAuto);
  document.querySelectorAll(".trump-btn").forEach((btn) => {
    btn.addEventListener("click", () => playAction(52 + parseInt(btn.dataset.suit, 10)));
  });
});


// Report the live engine in the footer: which policy is playing and how many
// worlds it samples per decision. The UI previously described itself as
// reinforcement-learned, which is not what plays here.
async function showEngine() {
  const el = document.getElementById("engine");
  if (!el) return;
  try {
    const response = await fetch("/health");
    if (!response.ok) return;
    const info = await response.json();
    if (!info.model_loaded) {
      el.textContent = "engine: scripted baseline (no network weights loaded)";
      return;
    }
    if (info.modes) {
      el.textContent = "AI strength: Fast · Hard";
      return;
    }
    const k = info.search_k ? `${info.search_k} sampled worlds/decision` : "";
    el.textContent = info.policy === "numpy-qnet+elimination-search"
      ? `engine: numpy action-value network + elimination search — ${k}`
      : `engine: ${info.policy}`;
  } catch (err) {
    /* the footer badge is decoration; a failure here must not break play */
  }
}
showEngine();
