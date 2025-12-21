"""
Ensure the repository's `code/` package directory is importable in all contexts
(pytest, scripts, CI) without needing to set PYTHONPATH explicitly.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent
code_dir = repo_root / "code"
if code_dir.is_dir():
    path_str = str(code_dir)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)



