# EurekaMesh CCAD (publish)

Minimal, reproducible subset for experiments and paper figures.

## What is CCAD?

CCAD (Cumulative Context Anti‑Duplication) is a modular approach to steer LLM generation towards maximising useful, non‑duplicated hypotheses under cost control. Core ideas:
- Canonicalize domain items (e.g., SMILES with RDKit) and build an exact dedup set in real‑time.
- Inject anti‑duplication context (recent uniques or prototypes via RAG) so the LLM avoids near‑repeats.
- Track token usage and report efficiency as UPT.

Scope in this repo: minimal, reproducible research subset for chemistry; adapters show how to extend to other domains.

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

# CLI quickstart (offline mock)
export MOCK_LLM=$'CCO\nCCN\nO=C(O)C1=CC=CC=C1\nC1CCCCC1\nC1=CC=CC=C1\nCC(=O)O'
python -m eurekamesh.cli run --mode ccad-basic --n 60 --fuzzy-policy count --fuzzy-threshold 0.90
python -m eurekamesh.cli bench --n 60 --seeds 1,2,3
python -m eurekamesh.cli report

# Installable (editable) and entrypoint
pip install -e .
eurekamesh run --mode ccad-basic --n 60
eurekamesh bench --n 60 --seeds 1,2,3
eurekamesh report

# Dashboard (Streamlit)
make dashboard
```

Outputs are written to `data/` as JSON + CSV with token/cost accounting.
Figures saved to `figures/` (PNG + SVG). Generate HTML report to `report/index.html` with:

```bash
make report
```

## Experiment types

- Baseline (naive LLM): `make baseline`
  - No anti‑dup context; validates parser/canonicalizer and provides a raw reference.
  - Output: `data/raw_baseline_*.{json,csv}`, metrics per run.

- A/B/C anti‑dup context: `make abtest` then `make figs`
  - Modes: `none`, `basic` (recent uniques), `rag` (prototype retrieval with embeddings).
  - Always canonicalization ON; chunk‑wise metrics and figures:
    - `figures/abtest_upt_dup.(png|svg)`, `figures/post_filters_summary.(png|svg)`, `figures/abtest_runs_upt.(png|svg)`.

- Benchmarks (real, minimal): `make bench-guacamol` / `make bench-moses` / `make bench-real`
  - Computes validity, uniqueness, UPT; includes hooks for advanced metrics (FCD, SNN, IntDiv) if extras are installed.

- End‑to‑end: `make e2e`
  - Runs baseline → A/B/C → figures → `make report` (HTML summary).

## Metrics

- UPT (%) = 100 × unique_accepted / total_generated (per run or aggregate).
- Validity rate (%) = 100 × valid / total_generated (SMILES via RDKit).
- Duplicate rate (%) = 100 × duplicates / total_generated (after canonicalization).
- Optional: post‑filters acceptance and SA‑score (when enabled).
- Optional (near‑dup químico): ECFP4 + Tanimoto con política `reject|count|allow` y umbral `FUZZY_THRESHOLD`.

## Connectors

- OpenAI (default): `OPENAI_API_KEY`, optional `OPENAI_URL` override.
- Anthropic: `ANTHROPIC_API_KEY` and optional `ANTHROPIC_URL` (messages API). Usage requires wiring in experiments.
- Local: `LOCAL_LLM_URL` to an OpenAI-compatible endpoint.
- All connectors support simple retries/backoff. For offline smoke tests, set `MOCK_LLM`.

## Framework scope

- Domains included: `code/eurekamesh/domains/drug_discovery` and `.../synthetic_biology`
  - Provide `canonicalizer.py`, optional `constraints.py`/`scorer.py` stubs.
- Core modules: `core/interfaces.py`, `core/ccad_engine.py`, `core/metrics.py`, `core/rag_system.py`.
- Observability: JSONL/CSV logs, figures, HTML report; token/cost accounting in `code/config.py`.
- Reproducibilidad:
  - Seeds and config centralizados (`code/config.py`), override por variables de entorno.
  - CI ejecuta tests y genera report con `MOCK_LLM`.

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
- Runs a mini-benchmark in MOCK mode and aggregates results in the report

## Notes

- For paper reproduction, prefer `MOCK_LLM` for CI and smoke tests, then run live for final metrics.
- Anthropic/local connectors are available; integrate by swapping connector in experiments.
