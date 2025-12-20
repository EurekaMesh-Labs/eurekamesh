import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

"""
E2E runner:
- Runs baseline (live or MOCK_LLM)
- Runs A/B/C experiment
- Generates figures
"""

def run(cmd, env=None):
    print('> ', cmd)
    res = subprocess.run(cmd, shell=True, env=env)
    if res.returncode != 0:
        sys.exit(res.returncode)

def main():
    env = os.environ.copy()
    base = 'publish' if Path('publish/code').exists() else '.'
    py = f"PYTHONPATH={base}/code python"
    # step 1: baseline
    run(f"{py} -m experiments.run_naive_baseline_raw", env)
    # step 2: A/B/C
    run(f"{py} -m experiments.abtest_anti_dup_context", env)
    # step 3: figures
    run(f"{py} -m experiments.plot_abtest_summary", env)
    print('E2E completed at', datetime.utcnow().isoformat())

if __name__ == '__main__':
    main()

