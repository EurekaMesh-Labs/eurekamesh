import json
import re
from typing import List

def parse_llm_smiles_output(text: str) -> List[str]:
    """Parse LLM output trying JSON array first, then fallback to one-per-line.
    Returns a list of raw candidate strings (not validated).
    """
    # Try JSON array
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            return [s for s in arr if isinstance(s, str)]
        except Exception:
            pass
    # Fallback to line-by-line
    lines = [ln.strip() for ln in text.splitlines()]
    # Drop obvious noise lines
    drop_num = re.compile(r"^\d+\.?$")
    cleaned = []
    for ln in lines:
        if not ln:
            continue
        if ln.startswith(("#","//","[","{")):
            continue
        if drop_num.match(ln):
            continue
        # remove leading numbering like "1. ..." or "- ..."
        if ". " in ln[:4]:
            ln = ln.split(". ",1)[1]
        if ln.startswith("-"):
            ln = ln[1:].strip()
        cleaned.append(ln.strip(", "))
    return cleaned
