# CCAD: A general framework to control duplication and steer diversity in LLM generation

Duplicate outputs = wasted tokens. CCAD turns duplication into a controllable signal.  
Outcome: +15–18 points absolute UPT (preliminary, n small), with explicit reproducibility.

---

## TL;DR

- CCAD is a general loop for LLM generation that maximizes Unique‑per‑Token (UPT) while maintaining validity and cost control.
- Core moves: canonicalize → exact dedup → optional fuzzy dedup → anti‑dup memory/context → verifier → policy update.
- Adaptive policy (bandit‑like): tighten dedup or increase context when duplication rises; lower temperature or enforce schema if validity falls.
- Domain adapters expose a minimal interface; the molecules adapter in the repo is a reference example only.
- Reproducibility: MOCK_LLM offline mode, deterministic embedding fallback, CI and make targets.

---

## Metrics

We track three families of metrics:

- UPT_raw = unique_accepted / total_generated  
  UPT_valid = unique_accepted / total_valid_generated  
  Interpretation: UPT is “useful diversity per budget” (how many unique items you get per tokens generated).
- Duplicate rates:  
  dup_rate_raw = duplicates / total_generated;  
  dup_rate_valid = duplicates / total_valid_generated.
- Validity rate = valid / total_generated (domain‑specific validation).

Each run logs JSONL with chunk‑level history; a CSV aggregator produces figures and an HTML report. Costs are estimated from token usage.

---

## The CCAD loop

Pseudocode overview:

```text
repeat until N uniques:
  # 1) Build anti‑dup context (last K or prototypes)
  context = render_memory(exemplars)
  # 2) Generate a chunk with LLM
  chunk = LLM_generate(base_prompt + context, temperature, chunk_size)
  # 3) Process
  for item in chunk:
    if not validate(item): continue
    can = canonicalize(item)
    if can in exact_set: mark duplicate; continue
    if fuzzy_policy != 'allow' and is_near_duplicate(can): continue (or count)
    accept(can); add to memory
  # 4) Update policy (adaptive)
  if dup_rate > high: increase K, tighten fuzzy(threshold/policy)
  if valid_rate < low: lower temperature, enforce schema/grammar
```

Diagrammatically: canonicalize → exact dedup → fuzzy dedup → anti‑dup memory/context → verifier → policy update.

---

## Adaptive policy (explore‑exploit)

Simple rules that work in practice:

- If `dup_rate` rises: increase `K` (anti‑dup context), switch `fuzzy_policy` to `reject` or lower the threshold, or vary the prompt.
- If `valid_rate` drops: decrease `temperature`, enable stricter parsing or grammar, or reduce context hardness to avoid over‑constraint.

This is a bandit‑style controller, favoring exploration when duplication is high and tightening constraints when quality is low.

---

## Domain Adapter Interface

An adapter defines how CCAD “understands” items in a domain:

- `canonicalize(x) -> str`
- `signature(x) -> Any` (embedding/fingerprint)
- `similarity(a, b) -> float`
- `validate(x) -> dict | bool` (domain checks and scores)
- `render_memory(exemplars) -> str` (turn memory into compact context)
- Optional: `parse_model_output(text) -> list[item]`

The molecules adapter is just a reference; the same pattern applies to text, code, materials, or fuzzing inputs.

---

## Mini demo (non‑chemical)

Example with generic short strings (2–3 inputs), showing UPT before/after:

```
# Prompt (simplified)
Generate 5 short items, one per line. No numbering.

# Raw LLM output (example)
apple
banana
orange
Apple   # notation variant → canonicalize lowercases
banana  # exact dup
apples  # near‑dup (token‑set similarity)

# Naive baseline (no canonicalization/context/dedup)
# 5 generated, 3 unique → UPT_raw = 3/5 = 60%

# CCAD (canon + context + exact/fuzzy)
- canonicalize → lowercase
- canonicalize → lowercase
- exact dedup removes exact repetition (banana)
- fuzzy dedup (threshold) suppresses "apple"/"apples" if too similar
- anti‑dup memory shows recent uniques → LLM avoids repeats

# After applying CCAD, the next chunk avoids repeats
# 5 generated, 5 unique → UPT_raw ≈ 100% (toy example)
```

Result: higher UPT with the same token budget.

---

## Reproducibility

- Offline: `MOCK_LLM` supplies deterministic lines. `make e2e` runs tests, baseline and A/B/C, then builds an HTML report.
- Deterministic fallback for embeddings ensures CI stability (no reliance on Python’s hash randomness).
- Seeds and configs via environment; CI uploads artifacts.

---

## Limitations

- Preliminary n is small; thresholds and gains vary by domain.
- Fuzzy similarity needs tuning per adapter; embeddings introduce small overhead.
- Tradeoff between diversity and validity requires careful policy.

---

## Call to action

Open an issue with your domain and ~10 example outputs; we’ll guide you to build an adapter using the interface above.


