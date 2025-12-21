import os
import asyncio
import time
import random
from typing import Tuple, List, Optional

try:
    from .parser import parse_llm_smiles_output
except Exception:
    try:
        from experiments.parser import parse_llm_smiles_output
    except Exception:
        from parser import parse_llm_smiles_output

RETRY_STATUS = {429, 500, 502, 503, 504}

async def http_post_with_retry(url: str, headers: dict, data_obj: dict, max_retries: int = 4, initial_backoff: float = 0.6, timeout_s: float = 60.0) -> dict:
    """POST JSON with exponential backoff + jitter, tries aiohttp then urllib."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=data_obj) as response:
                    if response.status in RETRY_STATUS:
                        text = await response.text()
                        raise RuntimeError(f"HTTP {response.status}: {text[:200]}")
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            last_err = e
            # fallback to urllib in a thread
            try:
                import json as _json
                import urllib.request
                import urllib.error
                def _post():
                    req = urllib.request.Request(url, data=_json.dumps(data_obj).encode('utf-8'), headers=headers, method='POST')
                    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                        return _json.loads(resp.read().decode('utf-8'))
                return await asyncio.to_thread(_post)
            except Exception as e2:
                last_err = e2
                if attempt < max_retries - 1:
                    jitter = random.uniform(0.8, 1.2)
                    backoff = initial_backoff * (2 ** attempt) * jitter
                    await asyncio.sleep(backoff)
                else:
                    break
    assert last_err is not None
    raise last_err

class LLMConnector:
    async def generate(self, system_prompt: str, user_prompt: str, model: str, temperature: float, max_tokens: int) -> Tuple[List[str], dict]:
        raise NotImplementedError

class OpenAIConnector(LLMConnector):
    async def generate(self, system_prompt: str, user_prompt: str, model: str, temperature: float, max_tokens: int):
        mock = os.getenv('MOCK_LLM')
        if mock:
            lines = [s.strip() for s in mock.splitlines() if s.strip()]
            return lines[:200], {'prompt_tokens': 0, 'completion_tokens': 0}
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        url = os.getenv('OPENAI_URL', 'https://api.openai.com/v1/chat/completions')
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {"model": model, "messages":[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], "temperature": temperature, "max_tokens": max_tokens}
        result = await http_post_with_retry(url, headers, data)
        content = result['choices'][0]['message']['content']
        items = parse_llm_smiles_output(content)
        usage = result.get('usage', {})
        return items, {'prompt_tokens': usage.get('prompt_tokens',0), 'completion_tokens': usage.get('completion_tokens',0)}

class AnthropicConnector(LLMConnector):
    async def generate(self, system_prompt: str, user_prompt: str, model: str, temperature: float, max_tokens: int):
        mock = os.getenv('MOCK_LLM')
        if mock:
            lines = [s.strip() for s in mock.splitlines() if s.strip()]
            return lines[:200], {'input_tokens': 0, 'output_tokens': 0}
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        url = os.getenv('ANTHROPIC_URL', 'https://api.anthropic.com/v1/messages')
        headers = {
            "x-api-key": api_key,
            "anthropic-version": os.getenv('ANTHROPIC_VERSION', '2023-06-01'),
            "content-type": "application/json"
        }
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        result = await http_post_with_retry(url, headers, data)
        content = ''.join([b.get('text','') for b in result.get('content', [])])
        items = parse_llm_smiles_output(content)
        usage = result.get('usage', {})
        return items, {'prompt_tokens': usage.get('input_tokens',0), 'completion_tokens': usage.get('output_tokens',0)}

class LocalConnector(LLMConnector):
    async def generate(self, system_prompt: str, user_prompt: str, model: str, temperature: float, max_tokens: int):
        url = os.getenv('LOCAL_LLM_URL')
        if url:
            headers = {"Content-Type": "application/json"}
            data = {"model": model, "messages":[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}], "temperature": temperature, "max_tokens": max_tokens}
            result = await http_post_with_retry(url, headers, data)
            if 'choices' in result:
                content = result['choices'][0]['message']['content']
            else:
                content = result.get('text', '')
            items = parse_llm_smiles_output(content)
            return items, {'prompt_tokens': 0, 'completion_tokens': 0}
        raise NotImplementedError("Set LOCAL_LLM_URL or use MOCK_LLM for LocalConnector")
