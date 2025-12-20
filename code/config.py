import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str = os.getenv('LLM_MODEL', 'gpt-4o')
    temperature: float = 0.8
    max_tokens: int = 2000
    # rough costs per 1K tokens (USD) – adjust per provider/pricing
    prompt_cost_per_1k: float = float(os.getenv('PROMPT_COST_PER_1K', '0.005'))
    completion_cost_per_1k: float = float(os.getenv('COMPLETION_COST_PER_1K', '0.015'))

@dataclass
class Seeds:
    main_seed: int = int(os.getenv('SEED', '42'))

MODEL = ModelConfig()
SEEDS = Seeds()


