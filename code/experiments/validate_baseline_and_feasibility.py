"""
Reviewer-requested validation:
1. Naive baseline (n=3 runs, 300 mols each)
2. Impracticability evidence (100 mols each: no-canon vs full-CCAD)

Total: ~1200 molecules, estimated 1-2 hours
"""

import asyncio
import os
import sys
import json
from datetime import datetime
import statistics

sys.path.insert(0, str(__file__))
from .parser import parse_llm_smiles_output

from rdkit import Chem
from eurekamesh.core import CCAdEngine
from eurekamesh.domains.drug_discovery import (
    SMILESCanonicalizer,
    DrugLikenessScorer,
    KINASE_INHIBITOR_CONSTRAINTS
)


class NoOpCanonicalizer:
    """Canonicalizer that does nothing - for impracticability test."""
    def canonicalize(self, smiles: str) -> str:
        return smiles.strip()  # No canonicalization
    
    def is_valid(self, smiles: str) -> bool:
        try:
            return Chem.MolFromSmiles(smiles) is not None
        except Exception:
            return False
    
    def preprocess(self, smiles: str) -> str:
        return smiles.strip()


async def openai_llm_generator(prompt: str, temperature: float, n: int):
    """Call OpenAI API."""
    import aiohttp
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompt = "You are a medicinal chemistry expert. Output SMILES in the requested format."
    
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 2000
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                raise RuntimeError(await response.text())
            
            result = await response.json()
            content = result['choices'][0]['message']['content']
            
            items, usage = parse_llm_smiles_output(content), result.get('usage',{})
            usage_stats = {
                'input_tokens': usage.get('prompt_tokens', 0),
                'output_tokens': usage.get('completion_tokens', 0),
                'model': 'gpt-4o'
            }
            
            return (items, usage_stats)


async def run_experiment(
    config_name: str,
    use_canonicalization: bool,
    use_anti_dup_context: bool,
    target_count: int,
    run_number: int
):
    """Run a single experiment."""
    print(f"\n{'='*70}\nEXPERIMENT: {config_name} (Run {run_number})\n{'='*70}\n")
    canonicalizer = SMILESCanonicalizer() if use_canonicalization else NoOpCanonicalizer()
    scorer = DrugLikenessScorer()
    engine = CCAdEngine(
        canonicalizer=canonicalizer,
        llm_generator=openai_llm_generator,
        scorer=scorer,
        retrieval_system=None,
        enable_fuzzy_dedup=False,
        enable_anti_dup_context=use_anti_dup_context,
        enable_rag_context=False,
        max_context_items=50
    )
    constraints = KINASE_INHIBITOR_CONSTRAINTS
    base_prompt = f"Generate drug-like molecules for {constraints.target_protein}.\n\n{constraints.to_prompt_context()}\n\nOutput EXACTLY one SMILES per line."
    start_time = datetime.now()
    results = await engine.generate(
        prompts=[base_prompt], target_counts=[target_count], temperature=0.8,
        max_per_call=50, min_per_call=10, max_iterations=50,
        enable_adaptive_temp=True, enable_early_stopping=True
    )
    duration = (datetime.now() - start_time).total_seconds()
    metrics = results['metrics']
    total_generated = metrics.get('total_generated', 0)
    unique_accepted = metrics.get('unique_accepted', 0)
    duplicates = metrics.get('total_duplicates', 0) if 'total_duplicates' in metrics else 0
    valid_generated = unique_accepted + duplicates
    invalid_count = max(0, total_generated - valid_generated)
    valid_rate = (valid_generated / max(1, total_generated)) if total_generated else 0
    print(f"Duration: {duration:.1f}s | total={total_generated} valid={valid_generated} ({valid_rate:.1%}) invalid={invalid_count}")
    return {
        'config_name': config_name,
        'run_number': run_number,
        'duration_seconds': duration,
        'use_canonicalization': use_canonicalization,
        'use_anti_dup_context': use_anti_dup_context,
        'total_generated': total_generated,
        'valid_generated': valid_generated,
        'invalid_count': invalid_count,
        'valid_rate': valid_rate,
        'n_unique': results['n_unique'],
        'upt': (unique_accepted / max(1, total_generated)) if total_generated else 0.0,
        'avg_dup_rate': sum(metrics.get('duplicate_rates', [])) / max(1, len(metrics.get('duplicate_rates', [])))
    }


async def main():
    """Run all experiments."""
    
    print("\n" + "🔬"*35 + "\nREVIEWER-REQUESTED VALIDATION\n" + "🔬"*35 + "\n")
    all_results = []
    
    # Naive baseline (n=3)
    for run_num in range(1, 4):
        all_results.append(await run_experiment("Naive_Baseline", False, False, 300, run_num))
    
    # Feasibility
    all_results.append(await run_experiment("No_Canonicalization", False, True, 100, 1))
    all_results.append(await run_experiment("Full_CCAD", True, True, 100, 1))
    
    out = f"baseline_and_feasibility_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved: {out}")
    
    # Generate summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Naive baseline stats
    naive_results = [r for r in all_results if r['config_name'] == 'Naive_Baseline']
    if len(naive_results) == 3:
        upts = [r['upt'] for r in naive_results]
        valid_rates = [r['valid_rate'] for r in naive_results]
        
        print(f"\nNaive Baseline (n=3):")
        print(f"  UPT: {statistics.mean(upts):.1%} ± {statistics.stdev(upts):.1%}")
        print(f"  Valid rate: {statistics.mean(valid_rates):.1%} ± {statistics.stdev(valid_rates):.1%}")
    
    # Impracticability evidence
    no_canon = next((r for r in all_results if r['config_name'] == 'No_Canonicalization'), None)
    full_ccad = next((r for r in all_results if r['config_name'] == 'Full_CCAD'), None)
    
    if no_canon and full_ccad:
        print(f"\nImpracticability Evidence:")
        print(f"  No Canonicalization: {no_canon['valid_rate']:.1%} valid ({no_canon['invalid_count']}/{no_canon['total_generated']} invalid)")
        print(f"  Full CCAD: {full_ccad['valid_rate']:.1%} valid ({full_ccad['invalid_count']}/{full_ccad['total_generated']} invalid)")
        print(f"  → Difference: {(full_ccad['valid_rate'] - no_canon['valid_rate']):.1%}")
    
    print("\n" + "="*70)
    print("✅ All experiments completed!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())

