import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from experiments.parser import parse_llm_smiles_output as default_parser  # reuse simple parser

try:
    from ..config import MODEL
except Exception:
    class _ModelCfg:
        prompt_cost_per_1k = float(os.getenv('PROMPT_COST_PER_1K', '0.005'))
        completion_cost_per_1k = float(os.getenv('COMPLETION_COST_PER_1K', '0.015'))
    MODEL = _ModelCfg()

BASE = Path('publish') if Path('publish/code').exists() else Path('.')

def openai_llm_generator(prompt: str) -> Dict[str, Any]:
    mock = os.getenv('MOCK_LLM')
    if mock:
        items = [s.strip() for s in mock.splitlines() if s.strip()]
        return {"items": items, "usage": {"input_tokens": 0, "output_tokens": 0}}
    import requests
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": os.getenv('LLM_MODEL', 'gpt-4o'),
        "messages": [
            {"role": "system", "content": "You output one SMILES per line."},
            {"role": "user", "content": prompt}
        ]
    }
    r = requests.post(url, headers=headers, json=data, timeout=60)
    r.raise_for_status()
    obj = r.json()
    content = obj['choices'][0]['message']['content']
    usage = obj.get('usage', {})
    items = default_parser(content)
    return {"items": items, "usage": {"input_tokens": usage.get('prompt_tokens', 0), "output_tokens": usage.get('completion_tokens', 0)}}

def run_posthoc(target: int = 100, parser=default_parser) -> Dict[str, Any]:
    # Single prompt, no anti-dup context, no policy; post-hoc dedup at the end
    base_prompt = "Generate as many valid SMILES as you can. EXACTLY one SMILES per line. No bullets."
    resp = openai_llm_generator(base_prompt)
    raw_items = resp["items"]
    # Truncate or pad (pad is no-op) to approximate target
    items = raw_items[:max(1, target)]
    # Validate + canonicalize (simple: keep non-empty)
    valid = [s for s in items if s]
    # Post-hoc exact dedup
    seen = set()
    uniques = []
    dups = 0
    for s in valid:
        can = s.strip()
        if can in seen:
            dups += 1
            continue
        seen.add(can)
        uniques.append(s)
    total_generated = len(items)
    total_valid = len(valid)
    unique_accepted = len(uniques)
    dup_rate_raw = (dups / total_generated) if total_generated else 0.0
    dup_rate_valid = (dups / total_valid) if total_valid else 0.0
    upt_raw = (unique_accepted / total_generated) if total_generated else 0.0
    upt_valid = (unique_accepted / total_valid) if total_valid else 0.0
    usage = resp.get("usage", {})
    cost = (usage.get('input_tokens', 0)/1000.0)*MODEL.prompt_cost_per_1k + (usage.get('output_tokens', 0)/1000.0)*MODEL.completion_cost_per_1k
    return {
        "mode": "posthoc_dedup_only",
        "total_generated": total_generated,
        "total_valid_generated": total_valid,
        "unique_accepted": unique_accepted,
        "dup_rate_raw": dup_rate_raw,
        "dup_rate_valid": dup_rate_valid,
        "upt_raw": upt_raw,
        "upt_valid": upt_valid,
        "prompt_tokens": usage.get('input_tokens', 0),
        "completion_tokens": usage.get('output_tokens', 0),
        "cost_usd": cost
    }

def main():
    target = int(os.getenv('TARGET', '100'))
    res = run_posthoc(target=target)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = BASE / f"data/posthoc_dedup_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()


