# EurekaMesh CCAD (publish)

Minimal, reproducible subset for experiments and paper figures.

## Environment

- Python 3.10+
- RDKit (via `environment.yml`) or `pip install -r requirements.txt`
- Set `OPENAI_API_KEY` to run live (OpenAI connector)
- Optional: `ANTHROPIC_API_KEY` (Anthropic), `LOCAL_LLM_URL` (OpenAI-compatible local server)
- Optional: `MOCK_LLM` to bypass network with inline SMILES (one per line)

```bash
# conda
conda env create -f environment.yml
conda activate eurekamesh-pub

# or pip (no RDKit binary provided)
pip install -r requirements.txt
```

## Quick start

```bash
# run tests
make test

# run baseline (live - OpenAI)
export OPENAI_API_KEY=sk-...
make baseline

# run baseline (offline mock)
export MOCK_LLM=$'CCO\nCCN\nO=C(O)C1=CC=CC=C1\nC1CCCCC1\nC1=CC=CC=C1\nCC(=O)O'
make baseline

# figures from latest A/B/C
make figs

# end-to-end (baseline + A/B/C + figures)
make e2e
```

Outputs are written to `data/` as JSON + CSV with token/cost accounting.
Figures saved to `figures/` (PNG + SVG). Generate HTML report to `report/index.html` with:

```bash
make report
```

## Connectors

- OpenAI (default): `OPENAI_API_KEY`, optional `OPENAI_URL` override.
- Anthropic: `ANTHROPIC_API_KEY` and optional `ANTHROPIC_URL` (messages API). Usage requires wiring in experiments.
- Local: `LOCAL_LLM_URL` to an OpenAI-compatible endpoint.
- All connectors support simple retries/backoff. For offline smoke tests, set `MOCK_LLM`.

## Benchmarks (placeholders)

```bash
# GuacaMol placeholder report
make bench-guacamol

# MOSES placeholder report
make bench-moses
```

These targets write stub JSON with instructions to enable the official benchmark harnesses.

## Configuration

See `code/config.py` for:
- Model selection, temperature, token limits
- Cost per 1K tokens (prompt/completion)
- Seed(s)

Override via environment variables, e.g. `LLM_MODEL`, `PROMPT_COST_PER_1K`.

## CI

This repo includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that:
- Installs the environment (conda + pip)
- Runs tests with `MOCK_LLM`
- Builds figures and the HTML report, uploading `report/` as an artifact

## Notes

- For paper reproduction, prefer `MOCK_LLM` for CI and smoke tests, then run live for final metrics.
- Anthropic/local connectors are available; integrate by swapping connector in experiments.
