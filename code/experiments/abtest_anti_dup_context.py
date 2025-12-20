import asyncio
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from eurekamesh.core import CCAdEngine
from eurekamesh.core.rag_system import InMemoryRAGSystem
from eurekamesh.domains.drug_discovery import (
    SMILESCanonicalizer,
    DrugLikenessScorer,
    KINASE_INHIBITOR_CONSTRAINTS
)
from .parser import parse_llm_smiles_output

# Config for cost accounting
try:
    from ..config import MODEL
except Exception:
    try:
        from config import MODEL
    except Exception:
        class _ModelCfg:
            prompt_cost_per_1k = float(os.getenv('PROMPT_COST_PER_1K', '0.005'))
            completion_cost_per_1k = float(os.getenv('COMPLETION_COST_PER_1K', '0.015'))
        MODEL = _ModelCfg()

RUNS = int(os.getenv('RUNS', '3'))
TARGET = int(os.getenv('TARGET', '100'))
MAX_CONTEXT_ITEMS = int(os.getenv('MAX_CONTEXT_ITEMS', '12'))
FUZZY_POLICY = os.getenv('FUZZY_POLICY', 'count')  # reject|count|allow
FUZZY_THRESHOLD = float(os.getenv('FUZZY_THRESHOLD', '0.90'))
ENABLE_FUZZY = os.getenv('ENABLE_FUZZY', '0') in ('1', 'true', 'True')

# token accumulator across LLM calls for this script execution
_token_acc = {'prompt_tokens': 0, 'completion_tokens': 0}

async def openai_llm_generator(prompt: str, temperature: float, n: int):
    # MOCK_LLM allows offline runs (one SMILES per line)
    mock = os.getenv('MOCK_LLM')
    if mock:
        items = [s.strip() for s in mock.splitlines() if s.strip()]
        return (items[:max(1, n)], {'input_tokens': 0, 'output_tokens': 0})
    import aiohttp
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    system_prompt = "You are a medicinal chemistry expert. Output SMILES in the requested format."
    model_name = getattr(MODEL, 'name', os.getenv('LLM_MODEL', 'gpt-4o'))
    max_tokens = getattr(MODEL, 'max_tokens', 2000)
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens)
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                raise RuntimeError(await response.text())
            result = await response.json()
            content = result['choices'][0]['message']['content']
            items = parse_llm_smiles_output(content)
            usage = result.get('usage', {})
            # accumulate tokens
            pt = usage.get('prompt_tokens', usage.get('input_tokens', 0))
            ct = usage.get('completion_tokens', usage.get('output_tokens', 0))
            _token_acc['prompt_tokens'] += int(pt or 0)
            _token_acc['completion_tokens'] += int(ct or 0)
            return (items[:max(1, n)], {'input_tokens': pt, 'output_tokens': ct})

async def run_condition(mode: str, target: int = TARGET) -> Dict[str, Any]:
    canon = SMILESCanonicalizer()
    scorer = DrugLikenessScorer()
    retrieval = None
    enable_rag = False
    if mode == 'rag':
        try:
            retrieval = InMemoryRAGSystem(
                embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                similarity_threshold=0.92,
                max_memory_size=10000
            )
            enable_rag = True
        except Exception:
            retrieval = None
            enable_rag = False
    engine = CCAdEngine(
        canonicalizer=canon,
        llm_generator=openai_llm_generator,
        scorer=scorer,
        retrieval_system=retrieval,
        enable_fuzzy_dedup=ENABLE_FUZZY,
        fuzzy_policy=FUZZY_POLICY,
        fuzzy_threshold=FUZZY_THRESHOLD,
        enable_anti_dup_context=(mode in ('basic','rag')),
        enable_rag_context=enable_rag,
        max_context_items=MAX_CONTEXT_ITEMS
    )
    constraints = KINASE_INHIBITOR_CONSTRAINTS
    base_prompt = (
        f"Generate drug-like molecules for {constraints.target_protein}.\n\n"
        f"{constraints.to_prompt_context()}\n\n"
        f"Anti-dup rules (strict):\n"
        f"- Do NOT repeat canonical SMILES seen so far.\n"
        f"- Prefer novel scaffolds over close analogs.\n"
        f"- If unsure, skip (leave line empty).\n\n"
        f"Output EXACTLY one SMILES per line. No numbers/bullets/punctuation."
    )
    before_pt, before_ct = _token_acc['prompt_tokens'], _token_acc['completion_tokens']
    res = await engine.generate(
        prompts=[base_prompt], target_counts=[target], temperature=0.8,
        max_per_call=50, min_per_call=10, max_iterations=60,
        enable_adaptive_temp=True, enable_early_stopping=True
    )
    after_pt, after_ct = _token_acc['prompt_tokens'], _token_acc['completion_tokens']
    used_pt = max(0, after_pt - before_pt)
    used_ct = max(0, after_ct - before_ct)
    # cost estimate
    cost_usd = (used_pt/1000.0)*MODEL.prompt_cost_per_1k + (used_ct/1000.0)*MODEL.completion_cost_per_1k
    m = res['metrics']
    items = res.get('items', [])  # unique accepted items
    # Post-filter metrics
    post_pass = 0
    sa_vals: List[float] = []
    for s in items:
        sc = scorer.score(s)
        if sc.get('valid', 0) and sc.get('lipinski_pass', 0) >= 1.0:
            if 'sa_score' in sc:
                if sc['sa_score'] <= scorer.sa_score_max:
                    post_pass += 1
            else:
                post_pass += 1
        if 'sa_score' in sc and sc.get('valid', 0):
            sa_vals.append(sc['sa_score'])
    post_accept_rate = (post_pass / max(1, len(items))) if items else 0.0
    post_sa_avg = (sum(sa_vals) / len(sa_vals)) if sa_vals else 0.0
    return {
        'mode': mode,
        'total_generated': m.get('total_generated',0),
        'total_valid_generated': m.get('total_valid_generated', 0),
        'unique_accepted': m.get('unique_accepted',0),
        'duplicates_filtered': m.get('duplicates_filtered',0),
        'duplicate_rates': m.get('duplicate_rates',[]),
        'duplicate_rates_raw': m.get('duplicate_rates_raw', []),
        'chunks_processed': m.get('chunks_processed',0),
        'post_accept_rate': post_accept_rate,
        'post_sa_avg': post_sa_avg,
        'prompt_tokens': used_pt,
        'completion_tokens': used_ct,
        'cost_usd': cost_usd
    }

def summarize(r: Dict[str,Any]):
    upt = (r['unique_accepted']/r['total_generated']) if r['total_generated'] else 0.0
    avg_dup = (sum(r['duplicate_rates'])/len(r['duplicate_rates'])) if r['duplicate_rates'] else 0.0
    return upt, avg_dup

BASE = Path('publish') if Path('publish/code').exists() else Path('.')

async def main():
    print(f"\nA/B/C Context + Filters (canon ON; runs={RUNS}; ctx={MAX_CONTEXT_ITEMS}; target={TARGET}; RAG=MiniLM)\n")
    results: List[Dict[str,Any]] = []
    start_pt, start_ct = _token_acc['prompt_tokens'], _token_acc['completion_tokens']
    for i in range(1, RUNS+1):
        print(f"Run {i}/{RUNS} - none")
        results.append({'run': i, **(await run_condition('none'))})
        print(f"Run {i}/{RUNS} - basic")
        results.append({'run': i, **(await run_condition('basic'))})
        print(f"Run {i}/{RUNS} - rag")
        results.append({'run': i, **(await run_condition('rag'))})
    # Aggregate
    modes = ['none','basic','rag']
    agg = {}
    import statistics as st
    for mode in modes:
        subset = [r for r in results if r['mode']==mode]
        upt = st.mean([summarize(r)[0] for r in subset]) if subset else 0.0
        dup = st.mean([summarize(r)[1] for r in subset]) if subset else 0.0
        post_acc = st.mean([r['post_accept_rate'] for r in subset]) if subset else 0.0
        post_sa = st.mean([r['post_sa_avg'] for r in subset]) if subset else 0.0
        pt_sum = sum(r.get('prompt_tokens',0) for r in subset)
        ct_sum = sum(r.get('completion_tokens',0) for r in subset)
        cost_sum = sum(r.get('cost_usd',0.0) for r in subset)
        agg[mode] = {
            'upt': upt,
            'dup_rate': dup,
            'post_accept_rate': post_acc,
            'post_sa_avg': post_sa,
            'prompt_tokens': pt_sum,
            'completion_tokens': ct_sum,
            'cost_usd': cost_sum
        }
    total_pt = _token_acc['prompt_tokens'] - start_pt
    total_ct = _token_acc['completion_tokens'] - start_ct
    total_cost = (total_pt/1000.0)*MODEL.prompt_cost_per_1k + (total_ct/1000.0)*MODEL.completion_cost_per_1k
    print("\nAggregate:")
    for mode in modes:
        a = agg[mode]
        print(f"{mode:>5}: UPT={a['upt']:.1%} dup={a['dup_rate']:.1%} post-accept={a['post_accept_rate']:.1%} SA-avg={a['post_sa_avg']:.2f}")
    # Save
    out = {
        'runs': RUNS,
        'max_context_items': MAX_CONTEXT_ITEMS,
        'target': TARGET,
        'fuzzy_policy': FUZZY_POLICY,
        'fuzzy_threshold': FUZZY_THRESHOLD,
        'enable_fuzzy': ENABLE_FUZZY,
        'per_run': results,
        'aggregate': agg,
        'total_prompt_tokens': total_pt,
        'total_completion_tokens': total_ct,
        'total_cost_usd': total_cost
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = BASE / f"abtest_filters_results_{ts}.json"
    out_jsonl = BASE / f"abtest_filters_results_{ts}.jsonl"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)
    # JSONL structured per-run
    with open(out_jsonl, 'w') as f:
        for r in results:
            f.write(json.dumps({"event":"abtest_run", **r}) + "\n")
    print(f"Saved: {out_json}")

if __name__ == "__main__":
    asyncio.run(main())
