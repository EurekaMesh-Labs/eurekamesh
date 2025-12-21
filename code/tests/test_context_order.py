import asyncio
from eurekamesh.domains.drug_discovery.canonicalizer import SMILESCanonicalizer
from eurekamesh.core.ccad_engine import CCAdEngine


def test_anti_dup_context_uses_last_N_in_order():
    canon = SMILESCanonicalizer()
    engine = CCAdEngine(
        canonicalizer=canon,
        llm_generator=None,  # not used here
        enable_anti_dup_context=True,
        max_context_items=3
    )
    # Simulate accepted items by calling process_batch twice
    # Valid SMILES in known order
    batch1 = ["CCO", "CCN"]
    batch2 = ["CCC", "c1ccccc1"]
    # Run in event loop because process_batch is async-aware called by generate; but safe to call directly
    unique1, _ = asyncio.get_event_loop().run_until_complete(engine.process_batch(batch1))
    unique2, _ = asyncio.get_event_loop().run_until_complete(engine.process_batch(batch2))
    assert len(unique1) == 2
    assert len(unique2) == 2
    # Now build context and ensure last 3 appear in order: CCN, CCC, c1ccccc1
    ctx = engine.build_anti_dup_context(use_rag=False)
    expected_order = ["CCN", "CCC", "c1ccccc1"]
    # Verify increasing indices
    positions = [ctx.find(s) for s in expected_order]
    assert all(p >= 0 for p in positions), f"Not all expected items found in context: {positions}"
    assert positions == sorted(positions), f"Items not in chronological order: {positions}"



