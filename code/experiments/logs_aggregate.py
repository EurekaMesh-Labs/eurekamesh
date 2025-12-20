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
    'run',
    'total','valid','unique_canonical',
    'upt','upt_valid','valid_rate',
    'dup_rate','dup_rate_raw',
    'cost_usd'
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
            'upt_valid': r.get('upt_valid',''),
            'valid_rate': r.get('valid_rate',''),
            'dup_rate': '',
            'dup_rate_raw': '',
            'cost_usd': r.get('cost_usd',''),
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
        'upt_valid': m.get('upt_valid',''),
        'valid_rate': m.get('valid_rate',''),
        'dup_rate': m.get('dup_rate',''),
        'dup_rate_raw': m.get('dup_rate_raw',''),
        'cost_usd': m.get('cost_usd',''),
    }]

def row_from_abtest_json(path: str):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for r in data.get('per_run', []):
        total = r.get('total_generated', 0) or 0
        total_valid = r.get('total_valid_generated', 0) or 0
        unique = r.get('unique_accepted', 0) or 0
        upt = (unique / max(1, total)) if total else 0.0
        upt_valid = (unique / max(1, total_valid)) if total_valid else ''
        # duplicate rates (averages if lists available)
        dr_list = r.get('duplicate_rates', []) or []
        drr_list = r.get('duplicate_rates_raw', []) or []
        dup_rate = sum(dr_list)/len(dr_list) if dr_list else ''
        dup_rate_raw = sum(drr_list)/len(drr_list) if drr_list else ''
        rows.append({
            'source': os.path.basename(path),
            'type': 'abtest',
            'timestamp': '',
            'model': '',
            'mode': r.get('mode',''),
            'n_target': data.get('target',''),
            'run': r.get('run',''),
            'total': total,
            'valid': total_valid,
            'unique_canonical': unique,
            'upt': upt,
            'upt_valid': upt_valid,
            'valid_rate': (total_valid/max(1, total)) if total else '',
            'dup_rate': dup_rate,
            'dup_rate_raw': dup_rate_raw,
            'cost_usd': r.get('cost_usd',''),
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
    # Mini-bench CLI outputs
    for p in glob.glob(str(DATA_DIR / 'mini_bench_*.json')):
        try:
            with open(p) as f:
                mb = json.load(f)
            for r in mb.get('runs', []):
                m = r.get('metrics', {})
                rows.append({
                    'source': os.path.basename(p),
                    'type': 'mini_bench',
                    'timestamp': '',
                    'model': '',
                    'mode': r.get('mode',''),
                    'n_target': r.get('n_target',''),
                    'run': r.get('seed',''),
                    'total': m.get('total_generated', m.get('total','')),
                    'valid': m.get('total_valid_generated',''),
                    'unique_canonical': m.get('unique_accepted',''),
                    'upt': ( (m.get('unique_accepted',0)/max(1,m.get('total_generated',0))) if m.get('total_generated') else '' ),
                    'upt_valid': '',
                    'valid_rate': '',
                    'dup_rate': '',
                    'dup_rate_raw': '',
                    'cost_usd': ''
                })
        except Exception:
            pass
    # Write CSV
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('Saved:', OUT_CSV)

if __name__ == '__main__':
    main()

