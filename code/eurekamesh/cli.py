import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Any, Dict

from eurekamesh.domains.drug_discovery import (
    SMILESCanonicalizer,
    DrugLikenessScorer,
    KINASE_INHIBITOR_CONSTRAINTS
)
from eurekamesh.core.ccad_engine import CCAdEngine


def _mock_or_env_llm():
    async def _gen(prompt: str, temperature: float, n: int):
        mock = os.getenv('MOCK_LLM')
        if mock:
            items = [s.strip() for s in mock.splitlines() if s.strip()]
            return (items[:max(1, n)], {'prompt_tokens': 0, 'completion_tokens': 0})
        raise RuntimeError("No LLM configured. Set MOCK_LLM for offline mode.")
    return _gen


def cmd_run(args: argparse.Namespace) -> None:
    canon = SMILESCanonicalizer()
    scorer = DrugLikenessScorer()
    retrieval = None
    enable_rag = args.mode == 'ccad-rag'
    if enable_rag:
        try:
            from eurekamesh.core.rag_system import InMemoryRAGSystem
            retrieval = InMemoryRAGSystem(
                embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                similarity_threshold=0.92,
                max_memory_size=10000
            )
        except Exception:
            retrieval = None
            enable_rag = False
    engine = CCAdEngine(
        canonicalizer=canon,
        llm_generator=_mock_or_env_llm(),
        scorer=scorer,
        retrieval_system=retrieval,
        enable_fuzzy_dedup=(args.fuzzy_policy in ('reject','count')),
        fuzzy_policy=args.fuzzy_policy,
        fuzzy_threshold=args.fuzzy_threshold,
        enable_anti_dup_context=(args.mode in ('ccad-basic','ccad-rag')),
        enable_rag_context=enable_rag,
        max_context_items=args.max_context_items
    )
    base_prompt = (
        f"Generate drug-like molecules for {KINASE_INHIBITOR_CONSTRAINTS.target_protein}.\n\n"
        f"{KINASE_INHIBITOR_CONSTRAINTS.to_prompt_context()}\n\n"
        f"Anti-dup rules (strict):\n"
        f"- Do NOT repeat canonical SMILES seen so far.\n"
        f"- Prefer novel scaffolds.\n"
        f"- If unsure, skip (leave line empty).\n\n"
        f"Output EXACTLY one SMILES per line. No numbers/bullets/punctuation."
    )
    import asyncio
    res = asyncio.get_event_loop().run_until_complete(engine.generate(
        prompts=[base_prompt],
        target_counts=[args.n],
        temperature=args.temperature,
        max_per_call=min(50, max(5, args.n)),
        min_per_call=5,
        max_iterations=120,
        enable_adaptive_temp=True,
        enable_early_stopping=True
    ))
    items = res.get('items', [])
    os.makedirs('publish/data', exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out = {
        'mode': args.mode,
        'n_target': args.n,
        'temperature': args.temperature,
        'fuzzy_policy': args.fuzzy_policy,
        'fuzzy_threshold': args.fuzzy_threshold,
        'max_context_items': args.max_context_items,
        'items': items,
        'metrics': res.get('metrics', {}),
        'summary': res.get('summary', '')
    }
    path = f"publish/data/cli_run_{args.mode}_{ts}.json"
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved:', path)


def cmd_bench(args: argparse.Namespace) -> None:
    # mini-benchmark: 3 seeds x (none, basic, rag)
    import asyncio
    seeds = [int(s) for s in (args.seeds.split(','))] if args.seeds else [1, 2, 3]
    modes = ['none', 'basic', 'rag']
    results: List[Dict[str, Any]] = []
    for seed in seeds:
        os.environ['SEED'] = str(seed)
        for mode in modes:
            # Delegate to CCAdEngine quickly (none=anti-dup off)
            canon = SMILESCanonicalizer()
            scorer = DrugLikenessScorer()
            retrieval = None
            enable_rag = (mode == 'rag')
            if enable_rag:
                try:
                    from eurekamesh.core.rag_system import InMemoryRAGSystem
                    retrieval = InMemoryRAGSystem(
                        embedding_model='sentence-transformers/all-MiniLM-L6-v2',
                        similarity_threshold=0.92,
                        max_memory_size=10000
                    )
                except Exception:
                    retrieval = None
                    enable_rag = False
            engine = CCAdEngine(
                canonicalizer=canon,
                llm_generator=_mock_or_env_llm(),
                scorer=scorer,
                retrieval_system=retrieval,
                enable_fuzzy_dedup=(args.fuzzy_policy in ('reject','count')),
                fuzzy_policy=args.fuzzy_policy,
                fuzzy_threshold=args.fuzzy_threshold,
                enable_anti_dup_context=(mode in ('basic','rag')),
                enable_rag_context=enable_rag,
                max_context_items=args.max_context_items
            )
            base_prompt = (
                f"Generate drug-like molecules for {KINASE_INHIBITOR_CONSTRAINTS.target_protein}.\n\n"
                f"{KINASE_INHIBITOR_CONSTRAINTS.to_prompt_context()}\n\n"
                f"Output EXACTLY one SMILES per line. No numbering."
            )
            res = asyncio.get_event_loop().run_until_complete(engine.generate(
                prompts=[base_prompt],
                target_counts=[args.n],
                temperature=args.temperature,
                max_per_call=min(50, max(5, args.n)),
                min_per_call=5,
                max_iterations=120,
                enable_adaptive_temp=True,
                enable_early_stopping=True
            ))
            metrics = res.get('metrics', {})
            results.append({
                'seed': seed,
                'mode': mode,
                'n_target': args.n,
                'metrics': metrics
            })
    os.makedirs('publish/data', exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_path = f"publish/data/mini_bench_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump({'runs': results}, f, indent=2)
    print('Saved:', out_path)


def cmd_report(args: argparse.Namespace) -> None:
    # Reuse existing report pipeline
    from experiments.logs_aggregate import main as logs_main
    from experiments.render_report import main as render_main
    logs_main()
    render_main()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='eurekamesh', description='EurekaMesh CCAD CLI')
    sub = p.add_subparsers(dest='cmd', required=True)
    # run
    pr = sub.add_parser('run', help='Run a single generation session')
    pr.add_argument('--mode', choices=['none','ccad-basic','ccad-rag'], default='ccad-basic')
    pr.add_argument('--n', type=int, default=100)
    pr.add_argument('--temperature', type=float, default=0.8)
    pr.add_argument('--max-context-items', type=int, default=12)
    pr.add_argument('--fuzzy-policy', choices=['reject','count','allow'], default='count')
    pr.add_argument('--fuzzy-threshold', type=float, default=0.90)
    pr.set_defaults(func=cmd_run)
    # bench
    pb = sub.add_parser('bench', help='Run a mini-benchmark (3 seeds x modes)')
    pb.add_argument('--n', type=int, default=60)
    pb.add_argument('--temperature', type=float, default=0.8)
    pb.add_argument('--seeds', type=str, default='1,2,3')
    pb.add_argument('--max-context-items', type=int, default=12)
    pb.add_argument('--fuzzy-policy', choices=['reject','count','allow'], default='count')
    pb.add_argument('--fuzzy-threshold', type=float, default=0.90)
    pb.set_defaults(func=cmd_bench)
    # report
    prep = sub.add_parser('report', help='Aggregate logs and render HTML report')
    prep.set_defaults(func=cmd_report)
    return p


def main(argv: List[str] = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()


