# DeepHokm web UI container: uv-based multi-stage build.
#
# The GPU device id, listen port, and model path all come from the
# environment (see .env.example); nothing deployment-specific is baked in
# beyond the optional weight-path build args.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS builder

WORKDIR /app

# Install uv project files first so dependency layers cache independently.
# The readme is deliberately NOT in this layer: it is only needed by the
# project install below, and a docs edit must not reinstall the torch stack.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev


FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Runtime OS deps: libgomp for torch's OpenMP, curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY README.md pyproject.toml ./

# Fast and Hard use distinct archives. The Hard network was trained from
# K=6144 teacher data, and its adjacent feature contract pins the input
# schema, weight hash, and training K. Both can be overridden by read-only
# bind mounts.
ARG QNET_PATH=checkpoints/qnet_numpy_full.npz
ARG HARD_QNET_PATH=checkpoints/qhybrid_k6144.npz
ARG HARD_QNET_CONTRACT_PATH=checkpoints/qhybrid_k6144.features.json
COPY ${QNET_PATH} /app/checkpoints/qnet_numpy_full.npz
COPY ${HARD_QNET_PATH} /app/checkpoints/qhybrid_k6144.npz
COPY ${HARD_QNET_CONTRACT_PATH} /app/checkpoints/qhybrid_k6144.features.json

# The listen port is deployment configuration: it arrives as DEEPHOKM_PORT at
# run time (compose / --env-file / -e) and is never baked in, so the image
# carries no EXPOSE literal either.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DEEPHOKM_QNET=/app/checkpoints/qnet_numpy_full.npz \
    DEEPHOKM_HARD_QNET=/app/checkpoints/qhybrid_k6144.npz

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fs "http://127.0.0.1:${DEEPHOKM_PORT}/health" || exit 1

COPY src/deephokm/webui/docker_entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \
    && useradd --create-home --uid 10001 app \
    && chown -R app:app /app

# The service is network-facing and writes nothing to disk: run unprivileged.
USER app

ENTRYPOINT ["/app/entrypoint.sh"]
