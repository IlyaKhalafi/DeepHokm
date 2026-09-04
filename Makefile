# DeepHokm task runner. Every workflow goes through uv.
#
# Configuration comes from .env (if present) plus the ambient environment;
# variables already set in the environment always win.

-include .env

ifdef DEEPHOKM_GPU_DEVICE_ID
export CUDA_VISIBLE_DEVICES ?= $(DEEPHOKM_GPU_DEVICE_ID)
endif
ifdef DEEPHOKM_CUDA_COMPAT_PATH
ifdef LD_LIBRARY_PATH
export LD_LIBRARY_PATH := $(DEEPHOKM_CUDA_COMPAT_PATH):$(LD_LIBRARY_PATH)
else
export LD_LIBRARY_PATH := $(DEEPHOKM_CUDA_COMPAT_PATH)
endif
endif

ARGS ?=

.PHONY: help test lint format bench

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

test:  ## Run the full pytest suite
	uv run pytest

lint:  ## Run ruff lint
	uv run ruff check .

format:  ## Apply ruff formatting and lint fixes
	uv run ruff format .
	uv run ruff check --fix .

bench:  ## Benchmark environment throughput
	uv run python scripts/bench_env.py
