import os
import json
import glob
import csv
from pathlib import Path

BASE = Path('publish') if Path('publish/code').exists() else Path('.')
DATA_DIR = BASE / 'data'
OUT_CSV = DATA_DIR / 'metrics_aggregate.csv'

FIELDS = [
    'source','type','timestamp','model','mode','n_target',
    'run','total','valid','unique_canonical','upt','valid_rate',
]

def row_from_baseline_json(path: str):
    with open(path) as f:
        data = json.load(f)
    model = data.get('model','')
    rows = []
    for r in data.get('runs', []):
        rows.append({
            'source': os.path.basename(path),
            'type': 'baseline',
            'timestamp': r.get('timestamp',''),
            'model': model,
            'mode': 'naive',
            'n_target': '',
            'run': r.get('run',''),
            'total': r.get('total',''),
            'valid': r.get('valid',''),
            'unique_canonical': r.get('unique_canonical',''),
            'upt': r.get('upt',''),
            'valid_rate': r.get('valid_rate',''),
        })
    return rows

def row_from_bench_json(path: str):
    with open(path) as f:
        data = json.load(f)
    m = data.get('metrics', {})
    return [{
        'source': os.path.basename(path),
        'type': 'benchmark',
        'timestamp': '',
        'model': data.get('llm_model',''),
        'mode': data.get('mode',''),
        'n_target': data.get('n_target',''),
        'run': '',
        'total': m.get('total',''),
        'valid': m.get('valid',''),
        'unique_canonical': m.get('unique_canonical',''),
        'upt': m.get('upt',''),
        'valid_rate': m.get('valid_rate',''),
    }]

def row_from_abtest_json(path: str):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for r in data.get('per_run', []):
        upt = (r.get('unique_accepted',0)/max(1, r.get('total_generated',0))) if r.get('total_generated') else 0.0
        rows.append({
            'source': os.path.basename(path),
            'type': 'abtest',
            'timestamp': '',
            'model': '',
            'mode': r.get('mode',''),
            'n_target': data.get('target',''),
            'run': r.get('run',''),
            'total': r.get('total_generated',''),
            'valid': '',
            'unique_canonical': r.get('unique_accepted',''),
            'upt': upt,
            'valid_rate': '',
        })
    return rows

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    # Baseline JSONs
    for p in glob.glob(str(DATA_DIR / 'raw_baseline_*.json')):
        rows.extend(row_from_baseline_json(p))
    # Benchmarks
    for p in glob.glob(str(DATA_DIR / 'bench_real_*.json')):
        rows.extend(row_from_bench_json(p))
    # ABTest
    for p in glob.glob(str(BASE / 'abtest_*results_*.json')):
        rows.extend(row_from_abtest_json(p))
    # Write CSV
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('Saved:', OUT_CSV)

if __name__ == '__main__':
    main()

