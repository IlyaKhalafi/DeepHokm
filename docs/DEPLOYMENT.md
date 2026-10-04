# Deployment

Running the web UI locally, in Docker, and the configuration surface.

![DeepHokm web UI](media/board.png)

The served policy is selected by `DEEPHOKM_POLICY`: `network` (the default)
uses the numpy action-value network, while `greedy` runs the deterministic
public-information heuristic without loading weights. `DEEPHOKM_QNET` points
at the weight archive; `DEEPHOKM_SEARCH_K` (default `0`) sets how many
determinized worlds each decision samples from a live search paired with
the network, trading latency for strength -- `0` is the network alone (no
search, no rollouts, an 18 ms decision, 0.663 against a greedy opposing
team); raising it past `0` adds live search on top of the same weights (3072
reaches the measured ceiling of 0.855, at seconds per decision --
`DEEPHOKM_SEARCH_WORKERS` parallelizes the rollouts to keep that
interactive). The K used to generate this network's own training labels
(up to 6144, see `METHODS_AND_RESULTS.md`) is a separate, offline,
one-time cost that never runs at serve time regardless of this setting.

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
