#!/bin/sh
# Launch the web UI. The listen port is deployment configuration and is never
# baked into the image: it must arrive as DEEPHOKM_PORT (compose passes it
# through from .env).
set -e
if [ -z "${DEEPHOKM_PORT}" ]; then
    echo "DEEPHOKM_PORT is not set; pass it via --env-file .env or -e" >&2
    exit 1
fi
exec uvicorn deephokm.webui.app:app --host 0.0.0.0 --port "${DEEPHOKM_PORT}"
