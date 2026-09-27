# DeepHokm web UI container: uv-based multi-stage build.
#
# The GPU device id, listen port, and model path all come from the
# environment (see .env.example); nothing deployment-specific is baked in
# beyond the optional QNET_PATH build arg.

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

# The served weights are baked in via QNET_PATH (a repo-relative path), and
# can be overridden at run time by a read-only bind mount. This is the numpy
# action-value network guiding the determinized search -- the measured
# policy (0.855 win rate against a greedy opposing team) -- not the earlier
# MaskablePPO checkpoint, which plateaued at the level of a greedy clone and
# is kept only as a fallback the served app never reaches for by default.
ARG QNET_PATH=checkpoints/qnet_numpy_full.npz
COPY ${QNET_PATH} /app/checkpoints/qnet_numpy_full.npz

# The listen port is deployment configuration: it arrives as DEEPHOKM_PORT at
# run time (compose / --env-file / -e) and is never baked in, so the image
# carries no EXPOSE literal either.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DEEPHOKM_QNET=/app/checkpoints/qnet_numpy_full.npz

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fs "http://127.0.0.1:${DEEPHOKM_PORT}/health" || exit 1

COPY src/deephokm/webui/docker_entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \
    && useradd --create-home --uid 10001 app \
    && chown -R app:app /app

# The service is network-facing and writes nothing to disk: run unprivileged.
USER app

ENTRYPOINT ["/app/entrypoint.sh"]
