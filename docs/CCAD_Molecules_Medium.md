# CCAD for Molecular Generation (Reference Case Study)

This article applies the general CCAD framework to small‑molecule generation. For the framework overview, see `CCAD_Framework_Medium.md`.

## Pipeline
1) Inputs: SMILES prompted to an LLM (baseline)  
2) Canonicalization: RDKit converts SMILES to a canonical form  
3) Exact dedup: rejects repeated canonical SMILES  
4) Fuzzy chemical dedup: ECFP4/Tanimoto with policies `reject|count|allow`  
5) Anti‑dup context: last K canonical uniques or RAG prototypes (Sentence‑BERT)  
6) Verifier/filters: Lipinski, SA‑Score, optional PAINS  
7) Adaptive policy: adjust K/threshold/temperature based on duplicate/validity rates  
8) Report: JSONL/CSV logs, figures (UPT/dup), HTML summary

## Results (illustrative)
- UPT ≈ 71.6–72.0% on EGFR‑like prompts (preliminary n=3)  
- 100% Lipinski pass; SA‑Score ≈ 3.36–3.54  
- RAG adds +0.4% UPT and 3–5× context compression (benefit increases with N)

## Quickstart (offline)
```bash
pip install -e .
export MOCK_LLM=$'CCO\nCCN\nc1ccccc1\nCC(=O)O\nCCO'
make e2e
open report/index.html
```

## Notes
- Fuzzy thresholds typically 0.90–0.95; tune per domain.  
- Deterministic embedding fallback used in CI; enable Sentence‑BERT for live RAG.

## CTA
Have a domain of interest (materials/proteins/sequences)? Open an issue with ~10 sample outputs to build an adapter.


