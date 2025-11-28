import os
import json
from datetime import datetime
from typing import List, Dict, Any

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from eurekamesh.core import CCAdEngine
from eurekamesh.core.rag_system import InMemoryRAGSystem
from eurekamesh.domains.drug_discovery import (
    SMILESCanonicalizer,
    DrugLikenessScorer,
    KINASE_INHIBITOR_CONSTRAINTS
)
try:
    from .util_connectors import OpenAIConnector
except Exception:
    try:
        from experiments.util_connectors import OpenAIConnector
    except Exception:
        from util_connectors import OpenAIConnector

# Config
BENCHMARK = os.getenv('BENCHMARK', 'guacamol')  # or 'moses'
MODE = os.getenv('MODE', 'naive')  # naive | ccad-basic | ccad-rag
N = int(os.getenv('N', '300'))
TEMP = float(os.getenv('TEMP', '0.8'))
MODEL_NAME = os.getenv('LLM_MODEL', 'gpt-4o')
SEED = int(os.getenv('SEED', '42'))
GUACAMOL_REF_SMI = os.getenv('GUACAMOL_REF_SMI')  # optional path to reference SMILES
MOSES_REF_SMI = os.getenv('MOSES_REF_SMI')  # optional path to reference SMILES

# Cost config
try:
    from ..config import MODEL as _MODEL
except Exception:
    try:
        from config import MODEL as _MODEL
    except Exception:
        class _M:
            prompt_cost_per_1k = float(os.getenv('PROMPT_COST_PER_1K', '0.005'))
            completion_cost_per_1k = float(os.getenv('COMPLETION_COST_PER_1K', '0.015'))
        _MODEL = _M()

connector = OpenAIConnector()
_token_acc = {'prompt_tokens': 0, 'completion_tokens': 0}

def is_valid_smiles(s: str) -> bool:
    try:
        return Chem.MolFromSmiles(s) is not None
    except Exception:
        return False

def canonicalize(s: str) -> str:
    try:
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else s
    except Exception:
        return s

def to_ecfp4(smiles: List[str]):
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        fps.append(fp)
    return fps

def internal_diversity(smiles: List[str]) -> float:
    fps = to_ecfp4([s for s in smiles if is_valid_smiles(s)])
    n = len(fps)
    if n < 2:
        return 0.0
    # average (1 - tanimoto) over pairs (sampled for speed if large)
    import random
    pairs = 0
    acc = 0.0
    for _ in range(min(2000, n*(n-1)//2)):
        i, j = random.randrange(n), random.randrange(n)
        if i == j:
            continue
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        acc += (1.0 - sim)
        pairs += 1
    return (acc / pairs) if pairs else 0.0

def snn_to_ref(gen_smiles: List[str], ref_smiles: List[str]) -> float:
    gen_fps = to_ecfp4([s for s in gen_smiles if is_valid_smiles(s)])
    ref_fps = to_ecfp4([s for s in ref_smiles if is_valid_smiles(s)])
    if not gen_fps or not ref_fps:
        return 0.0
    best_sims = []
    for g in gen_fps:
        sims = DataStructs.BulkTanimotoSimilarity(g, ref_fps)
        best_sims.append(max(sims) if sims else 0.0)
    # SNN (GuacaMol) is usually average similarity to nearest neighbor in ref
    return sum(best_sims) / len(best_sims)

def try_fcd(gen_smiles: List[str], ref_smiles: List[str]) -> float:
    try:
        from fcd_torch import FCD
        fcd = FCD(device='cpu')
        return float(fcd(gen_smiles, ref_smiles))
    except Exception:
        return -1.0  # unavailable

async def generate_naive() -> List[str]:
    system = "You are a medicinal chemistry expert."
    constraints = KINASE_INHIBITOR_CONSTRAINTS.to_prompt_context()
    user = (
        f"Generate diverse, drug-like SMILES. Requirements:\n"
        f"- Valid SMILES syntax\n- Lipinski compliant\n- SA-Score <= 4 preferred\n\n"
        f"Context:\n{constraints}\n\n"
        f"Output EXACTLY one SMILES per line. No numbering, no extra text. Target: {N}."
    )
    mock = os.getenv('MOCK_LLM')
    if mock:
        items = [s.strip() for s in mock.splitlines() if s.strip()]
        return items[:N]
    items, usage = await connector.generate(system, user, MODEL_NAME, TEMP, 2000)
    _token_acc['prompt_tokens'] += int(usage.get('prompt_tokens', 0))
    _token_acc['completion_tokens'] += int(usage.get('completion_tokens', 0))
    return items

async def generate_ccad(use_rag: bool) -> List[str]:
    canon = SMILESCanonicalizer()
    scorer = DrugLikenessScorer()
    retrieval = None
    if use_rag:
        try:
            retrieval = InMemoryRAGSystem(
                embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                similarity_threshold=0.92,
                max_memory_size=10000
            )
        except Exception:
            retrieval = None
            use_rag = False
    async def llm(prompt: str, temperature: float, n: int):
        mock = os.getenv('MOCK_LLM')
        if mock:
            items = [s.strip() for s in mock.splitlines() if s.strip()]
            return (items[:max(1, n)], {'prompt_tokens': 0, 'completion_tokens': 0})
        items, usage = await connector.generate(
            "You are a medicinal chemistry expert.", prompt, MODEL_NAME, TEMP, 2000
        )
        _token_acc['prompt_tokens'] += int(usage.get('prompt_tokens', 0))
        _token_acc['completion_tokens'] += int(usage.get('completion_tokens', 0))
        return items, usage
    engine = CCAdEngine(
        canonicalizer=canon,
        llm_generator=llm,
        scorer=scorer,
        retrieval_system=retrieval,
        enable_fuzzy_dedup=False,
        enable_anti_dup_context=True,
        enable_rag_context=use_rag,
        max_context_items=12
    )
    base_prompt = (
        f"Generate drug-like molecules for {KINASE_INHIBITOR_CONSTRAINTS.target_protein}.\n\n"
        f"{KINASE_INHIBITOR_CONSTRAINTS.to_prompt_context()}\n\n"
        f"Anti-dup rules (strict):\n- Do NOT repeat canonical SMILES seen so far.\n- Prefer novel scaffolds.\n\n"
        f"Output EXACTLY one SMILES per line. No numbers/bullets/punctuation."
    )
    res = await engine.generate(
        prompts=[base_prompt], target_counts=[N], temperature=TEMP,
        max_per_call=50, min_per_call=10, max_iterations=60,
        enable_adaptive_temp=True, enable_early_stopping=True
    )
    return res.get('items', [])

def compute_metrics(items: List[str]) -> Dict[str, Any]:
    total = len(items)
    valids = [s for s in items if is_valid_smiles(s)]
    can_set = set(canonicalize(s) for s in valids)
    metrics = {
        'total': total,
        'valid': len(valids),
        'valid_rate': (len(valids)/max(1,total)) if total else 0.0,
        'unique_canonical': len(can_set),
        'upt': (len(can_set)/max(1,total)) if total else 0.0
    }
    # Advanced metrics (optional)
    adv: Dict[str, Any] = {}
    try:
        adv['intdiv'] = internal_diversity(list(can_set))
    except Exception:
        adv['intdiv'] = None
    # SNN and FCD need reference
    ref_path = GUACAMOL_REF_SMI or MOSES_REF_SMI
    if ref_path and os.path.exists(ref_path):
        try:
            with open(ref_path) as f:
                ref = [line.strip() for line in f if line.strip()]
            adv['snn_ref'] = snn_to_ref(list(can_set), ref)
            fcd_val = try_fcd(list(can_set), ref)
            adv['fcd'] = None if fcd_val < 0 else fcd_val
        except Exception:
            adv['snn_ref'] = None
            adv['fcd'] = None
    metrics['advanced'] = adv
    return metrics

async def main():
    has_guacamol = False
    has_moses = False
    try:
        import guacamol  # noqa: F401
        has_guacamol = True
    except Exception:
        pass
    try:
        import moses  # noqa: F401
        has_moses = True
    except Exception:
        pass

    # Generate
    if MODE == 'naive':
        items = await generate_naive()
    elif MODE == 'ccad-basic':
        items = await generate_ccad(use_rag=False)
    elif MODE == 'ccad-rag':
        items = await generate_ccad(use_rag=True)
    else:
        raise ValueError('MODE must be naive|ccad-basic|ccad-rag')

    # Metrics
    metrics = compute_metrics(items)

    # Token/cost
    pt = _token_acc['prompt_tokens']
    ct = _token_acc['completion_tokens']
    cost_usd = (pt/1000.0)*_MODEL.prompt_cost_per_1k + (ct/1000.0)*_MODEL.completion_cost_per_1k

    out = {
        'benchmark': BENCHMARK,
        'mode': MODE,
        'n_target': N,
        'llm_model': MODEL_NAME,
        'seed': SEED,
        'has_guacamol': has_guacamol,
        'has_moses': has_moses,
        'metrics': metrics,
        'prompt_tokens': pt,
        'completion_tokens': ct,
        'cost_usd': cost_usd,
        'notes': 'Advanced metrics computed if refs provided and libs available (GUACAMOL_REF_SMI/MOSES_REF_SMI).'
    }
    os.makedirs('publish/data', exist_ok=True)
    fn = f"publish/data/bench_real_{BENCHMARK}_{MODE}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fn, 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved:', fn)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
