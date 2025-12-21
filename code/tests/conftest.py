import os
import sys
from pathlib import Path

# Ensure package imports work when running tests from repo root or CI
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))



