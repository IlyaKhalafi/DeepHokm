# Contributing

Bug reports, policy improvements, and reproducible experiments are welcome.
Describe the Hokm position or seed that reproduces a problem and include the
expected and observed behavior. Do not attach credentials or private data.

## Development

```bash
uv sync
make test
make lint
```

Keep game-rule changes in `src/deephokm/rules`, policies in
`src/deephokm/policies`, and public network features in `src/deephokm/nn`.
Tests should cover legal actions, partnership behavior, suit symmetry, hand
resets, and independence from opponents' actual hidden cards where applicable.

The reusable Q-network workflow is documented in [Training](docs/TRAINING.md).
Use small CPU runs for smoke tests; do not start expensive training or overwrite
the bundled weights as part of a normal test run.

## What belongs in a commit

- Reusable source, regression tests, documentation, and intentional demo assets.
- Measured claims with evaluation seeds, sample sizes, opponent versions,
  search budgets, and a clear distinction between development and final tests.
- Configuration examples without machine-specific paths or credentials.

Keep datasets, generated models, logs, caches, scratch scripts, and dated
job launchers out of Git. The bundled inference checkpoint is the deliberate
exception; replacing it should be a separately evaluated change.

Use a focused commit and inspect `git diff --cached` before committing.
Do not include unrelated work or machine-local agent instructions.
