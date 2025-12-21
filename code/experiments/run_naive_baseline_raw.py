import os
import json
import csv
import asyncio
from datetime import datetime
from typing import List

from rdkit import Chem
try:
    from .parser import parse_llm_smiles_output
except Exception:
    try:
        from experiments.parser import parse_llm_smiles_output
    except Exception:
        from parser import parse_llm_smiles_output
try:
    from .util_connectors import OpenAIConnector
except Exception:
    try:
        from experiments.util_connectors import OpenAIConnector
    except Exception:
        from util_connectors import OpenAIConnector
try:
    from ..config import MODEL, SEEDS
except Exception:
    try:
        from config import MODEL, SEEDS
    except Exception:
        class _ModelCfg:
            name = os.getenv('LLM_MODEL', 'gpt-4o')
            temperature = 0.8
            max_tokens = 2000
            prompt_cost_per_1k = float(os.getenv('PROMPT_COST_PER_1K', '0.005'))
            completion_cost_per_1k = float(os.getenv('COMPLETION_COST_PER_1K', '0.015'))
        class _Seeds:
            main_seed = int(os.getenv('SEED', '42'))
        MODEL = _ModelCfg()
        SEEDS = _Seeds()

MODEL_NAME = MODEL.name
TARGET_PER_RUN = 150
RUNS = 3
TEMP = MODEL.temperature
MAX_TOKENS = MODEL.max_tokens

SYSTEM_PROMPT = (
    "You are a medicinal chemistry expert.\n"
    "Task: output SMILES candidates in the exact format requested."
)

USER_PROMPT = (
    "Generate diverse, drug-like molecules (SMILES). Requirements:\n"
    "- Valid SMILES syntax\n"
    "- Lipinski-compliant\n"
    "- Some scaffold diversity\n\n"
    "Output EXACTLY one SMILES per line. No numbering, no extra text."
)

connector = OpenAIConnector()

async def call_llm(prompt: str, temperature: float, n: int) -> (List[str], dict):
    prompt_with_n = (
        f"{prompt}\n\n"
        f"Generate EXACTLY {n} lines. One SMILES per line. "
        f"No numbering, bullets or extra text."
    )
    items, usage = await connector.generate(SYSTEM_PROMPT, prompt_with_n, MODEL_NAME, temperature, MAX_TOKENS)
    # Ensure we return at most n items
    items = items[:max(1, n)]
    return items, usage

def is_valid_smiles(s: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(s)
        return mol is not None
    except Exception:
        return False

async def run_one(run_idx: int) -> dict:
    items, usage = await call_llm(USER_PROMPT, TEMP, TARGET_PER_RUN)
    total = len(items)
    valid = [s for s in items if is_valid_smiles(s)]
    invalid = total - len(valid)
    can_set = set()
    for s in valid:
        mol = Chem.MolFromSmiles(s)
        can = Chem.MolToSmiles(mol, canonical=True) if mol else s
        can_set.add(can)
    upt = (len(can_set) / max(1, total)) if total else 0.0
    upt_valid = (len(can_set) / max(1, len(valid))) if valid else 0.0
    valid_rate = len(valid) / max(1, total)
    ptok = usage.get('prompt_tokens', 0)
    ctok = usage.get('completion_tokens', 0)
    cost = (ptok/1000.0)*MODEL.prompt_cost_per_1k + (ctok/1000.0)*MODEL.completion_cost_per_1k
    return {
        "run": run_idx,
        "total": total,
        "valid": len(valid),
        "invalid": invalid,
        "valid_rate": valid_rate,
        "unique_canonical": len(can_set),
        "upt": upt,
        "upt_valid": upt_valid,
        "prompt_tokens": ptok,
        "completion_tokens": ctok,
        "cost_usd": cost,
        "timestamp": datetime.utcnow().isoformat()
    }

async def main():
    results = []
    for i in range(1, RUNS+1):
        r = await run_one(i)
        results.append(r)
    import statistics as st
    upt_mean = st.mean([r["upt"] for r in results])
    upt_std = st.stdev([r["upt"] for r in results]) if len(results) > 1 else 0.0
    vr_mean = st.mean([r["valid_rate"] for r in results])
    vr_std = st.stdev([r["valid_rate"] for r in results]) if len(results) > 1 else 0.0
    cost_total = sum(r['cost_usd'] for r in results)
    out = {
        "model": MODEL_NAME,
        "runs": results,
        "summary": {
            "upt_mean": upt_mean,
            "upt_std": upt_std,
            "valid_rate_mean": vr_mean,
            "valid_rate_std": vr_std,
            "total_cost_usd": cost_total
        }
    }
    os.makedirs("publish/data", exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    json_path = f"publish/data/raw_baseline_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    # CSV per-run metrics
    csv_path = f"publish/data/raw_baseline_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run","total","valid","invalid","unique_canonical","upt","upt_valid","valid_rate","prompt_tokens","completion_tokens","cost_usd","timestamp"]) 
        for r in results:
            w.writerow([
                r['run'], r['total'], r['valid'], r['invalid'], r['unique_canonical'],
                round(r['upt'],4), round(r['upt_valid'],4), round(r['valid_rate'],4),
                r['prompt_tokens'], r['completion_tokens'], round(r['cost_usd'],6), r['timestamp']
            ])
        w.writerow([])
        w.writerow(["summary_upt_mean", round(out['summary']['upt_mean'],4)])
        w.writerow(["summary_upt_std", round(out['summary']['upt_std'],4)])
        w.writerow(["summary_valid_rate_mean", round(out['summary']['valid_rate_mean'],4)])
        w.writerow(["summary_valid_rate_std", round(out['summary']['valid_rate_std'],4)])
        w.writerow(["summary_total_cost_usd", round(out['summary']['total_cost_usd'],6)])
    # JSONL structured log per run
    jsonl_path = f"publish/data/raw_baseline_{ts}.jsonl"
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "event": "baseline_run",
                "model": MODEL_NAME,
                **r
            }) + "\n")
    print(f"Saved: {json_path}\nSaved: {csv_path}\nSaved: {jsonl_path}")

if __name__ == "__main__":
    asyncio.run(main())
