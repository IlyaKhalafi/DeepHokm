# DeepHokm task runner. Every workflow goes through uv.
#
# Configuration comes from .env (if present) plus the ambient environment;
# variables already set in the environment always win.

-include .env

# Export the configuration surface so recipe shells (and child processes) see
# the .env values; variables already set in the environment always win.
# Every value is stripped: `-include` parses .env as a makefile, and Make keeps
# any whitespace between a value and a trailing comment, which would otherwise
# pad paths and URLs with trailing spaces.
DEEPHOKM_PORT := $(strip $(DEEPHOKM_PORT))
DEEPHOKM_MODEL := $(strip $(DEEPHOKM_MODEL))
DEEPHOKM_MODEL_PATH := $(strip $(DEEPHOKM_MODEL_PATH))
DEEPHOKM_GPU_DEVICE_ID := $(strip $(DEEPHOKM_GPU_DEVICE_ID))
DEEPHOKM_ALLOW_RANDOM_MODEL := $(strip $(DEEPHOKM_ALLOW_RANDOM_MODEL))
DEEPHOKM_VLM_BASE_URL := $(strip $(DEEPHOKM_VLM_BASE_URL))
DEEPHOKM_VLM_MODEL := $(strip $(DEEPHOKM_VLM_MODEL))
DEEPHOKM_BASE_URL := $(strip $(DEEPHOKM_BASE_URL))
DEEPHOKM_CUDA_COMPAT_PATH := $(strip $(DEEPHOKM_CUDA_COMPAT_PATH))

export DEEPHOKM_PORT
export DEEPHOKM_MODEL
export DEEPHOKM_MODEL_PATH
export DEEPHOKM_GPU_DEVICE_ID
export DEEPHOKM_ALLOW_RANDOM_MODEL
export DEEPHOKM_VLM_BASE_URL
export DEEPHOKM_VLM_MODEL
export DEEPHOKM_BASE_URL

ifneq ($(DEEPHOKM_GPU_DEVICE_ID),)
export CUDA_VISIBLE_DEVICES ?= $(DEEPHOKM_GPU_DEVICE_ID)
endif
ifneq ($(DEEPHOKM_CUDA_COMPAT_PATH),)
ifdef LD_LIBRARY_PATH
export LD_LIBRARY_PATH := $(DEEPHOKM_CUDA_COMPAT_PATH):$(LD_LIBRARY_PATH)
else
export LD_LIBRARY_PATH := $(DEEPHOKM_CUDA_COMPAT_PATH)
endif
endif

ARGS ?=

.PHONY: help test lint format bench train plot webui smoke visual-qa \
        docker-build docker-up docker-down

help:  ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

test:  ## Run the full pytest suite
	uv run pytest

lint:  ## Run ruff lint and the mypy type check
	uv run ruff check .
	uv run mypy src

format:  ## Apply ruff formatting and lint fixes
	uv run ruff format .
	uv run ruff check --fix .

bench:  ## Benchmark environment throughput
	uv run python scripts/bench_env.py

train:  ## Run self-play training (extra flags via ARGS="...")
	uv run python -m deephokm.training.train $(ARGS)

plot:  ## Render training figures from TensorBoard logs
	uv run python scripts/plot_results.py $(ARGS)

webui:  ## Run the web UI (serves the trained model)
	@test -n "$${DEEPHOKM_PORT}" || \
		{ echo "DEEPHOKM_PORT is unset; copy .env.example to .env" >&2; exit 1; }
	uv run uvicorn deephokm.webui.app:app --host 0.0.0.0 --port "$${DEEPHOKM_PORT}"

smoke:  ## Playwright smoke test against a running web UI
	uv run python scripts/playwright_smoke.py

visual-qa:  ## Run the automated visual QA loop against a running web UI
	uv run python scripts/visual_qa.py $(ARGS)

docker-build:  ## Build the web UI image (bakes the configured checkpoint)
	docker compose build

docker-up:  ## Start the web UI container in the background
	docker compose up -d

docker-down:  ## Stop the web UI container
	docker compose down
