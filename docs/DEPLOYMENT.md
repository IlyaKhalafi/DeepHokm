# Deployment

Running the web UI locally, in Docker, and the configuration surface.

![DeepHokm web UI](media/board.png)

The served policy is the numpy action-value network guiding the determinized
search. `DEEPHOKM_SEARCH_K` sets how many worlds each decision samples, trading
latency for strength; the default keeps a decision well under a second.
`DEEPHOKM_QNET` points at the weight archive.

```bash
make webui            # serves on ${DEEPHOKM_PORT}
```

Two modes: play seat 0 against the trained policy, or spectate AI-vs-AI one
decision at a time (with auto-play). The UI always shows your hand, the
table with seat attribution and trick order, trump, per-team tricks and game
points, whose turn it is, the current phase, and explicit card counts per
seat; illegal cards are visibly disabled and rejected server-side.


## Docker

```bash
cp .env.example .env    # edit per machine; set DEEPHOKM_GPU_DEVICE_ID
docker compose up --build -d
curl -fs "http://localhost:${DEEPHOKM_PORT}/health"
```

or plain docker (export the configuration first — `--env-file` only feeds the
container's environment, not the shell expansions in these flags):

```bash
set -a; . ./.env; set +a
docker build -t deephokm-web:latest .
docker run --rm -p "${DEEPHOKM_PORT}:${DEEPHOKM_PORT}" --tmpfs /tmp \
  --env-file .env deephokm-web:latest
```

The image is a multi-stage uv build. The numpy Q-network weights
(`checkpoints/qnet_numpy_full.npz`, committed to the repository) are baked
in by default — the same file `make webui` serves locally, so a fresh
`docker compose up --build` plays at the measured 0.855 win rate out of the
box. To bake a different weight archive instead, set `DEEPHOKM_QNET` in
`.env` to its repo-relative path before building.

At run time a read-only bind mount can override the baked weights: set
`DEEPHOKM_MODEL_PATH` in `.env` (absolute or relative to the compose file;
it defaults to the repository's own `checkpoints/qnet_numpy_full.npz`, so
the mount is a no-op override rather than a requirement). The container
never reserves a GPU device at all: numpy inference runs on CPU in ~30 ms
regardless (see `src/deephokm/nn/numpy_qnet.py`), so a mandatory device
reservation would only break `docker compose up` on a machine with no
nvidia container runtime for no benefit. `DEEPHOKM_GPU_DEVICE_ID` still pins
bare-metal training and the dev web UI server (`make train`, `make webui`)
to GPU 6 via `CUDA_VISIBLE_DEVICES`; it plays no role in the container. The
container runs as an unprivileged user with a RAM-backed `/tmp` and writes
nothing durable, so the only host state it needs is the mounted weights.

The earlier MaskablePPO checkpoint (`checkpoints/final.zip`) is not baked
into the image; it plateaued at the level of a greedy clone during
training (see [`METHODS_AND_RESULTS.md`](METHODS_AND_RESULTS.md)) and the
served app only falls back to it if the numpy weights are absent.


## Configuration

All deployment-specific values are environment variables documented in
[`.env.example`](.env.example) — GPU device id, web UI port, served model
paths, and the visual QA reviewer endpoint. Defaults live only there;
per-machine values live in a gitignored `.env`.

