import json
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path('publish') if Path('publish/code').exists() else Path('.')

def load_latest_results():
    # Prefer files under publish/ if called from repo root
    candidates = sorted(glob.glob(str(BASE / 'abtest_*results_*.json')))
    if not candidates:
        # Fallback: data dir
        candidates = sorted(glob.glob(str(BASE / 'data' / 'abtest_*results_*.json')))
    if not candidates:
        raise FileNotFoundError('No abtest results found.')
    return candidates[-1]

def main():
    res_path = load_latest_results()
    with open(res_path) as f:
        data = json.load(f)
    agg = data.get('aggregate', {})
    # expected keys: none/basic/rag each with upt, dup, post_accept, sa_avg
    labels = ['none', 'basic', 'rag']
    UPT = [agg.get(k, {}).get('upt', 0)*100 for k in labels]
    DUP = [agg.get(k, {}).get('dup_rate', agg.get(k, {}).get('dup', 0))*100 for k in labels]
    POST = [agg.get(k, {}).get('post_accept_rate', agg.get(k, {}).get('post_accept', 0))*100 for k in labels]
    SA = [agg.get(k, {}).get('post_sa_avg', agg.get(k, {}).get('sa_avg', 0)) for k in labels]

    out_dir = BASE / 'figures'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Ablation/Context UPT vs Duplicate Rate
    x = np.arange(len(labels))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(7,4))
    b1 = ax1.bar(x - width/2, UPT, width, color=['#9aa5b1','#3b82f6','#1e40af'], label='UPT (%)')
    ax1.set_ylabel('UPT (%)')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    b2 = ax2.bar(x + width/2, DUP, width, color=['#fca5a5','#ef4444','#7f1d1d'], label='Duplicate rate (%)')
    ax2.set_ylabel('Duplicate rate (%)')
    ax2.set_ylim(0, 100)

    for i, b in enumerate(b1):
        ax1.annotate(f"{UPT[i]:.1f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                     textcoords='offset points', xytext=(0,3), ha='center', fontsize=9)
    for i, b in enumerate(b2):
        ax2.annotate(f"{DUP[i]:.1f}%", (b.get_x()+b.get_width()/2, b.get_height()),
                     textcoords='offset points', xytext=(0,3), ha='center', fontsize=9, color='#7f1d1d')

    fig.suptitle('A/B/C Context Comparison: UPT vs Duplicate Rate')
    fig.tight_layout()
    fig.savefig(out_dir / 'abtest_upt_dup.png', dpi=180)
    fig.savefig(out_dir / 'abtest_upt_dup.svg')
    plt.close(fig)

    # Figure 2: Post-filters summary (UPT + SA-avg with acceptance overlay)
    fig, ax1 = plt.subplots(figsize=(7,4))
    bars = ax1.bar(x, UPT, 0.55, color=['#9aa5b1','#3b82f6','#1e40af'])
    ax1.set_ylim(0,100)
    ax1.set_ylabel('UPT (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    ax2.plot(x, SA, color='#10b981', marker='o', label='SA-avg (lower=better)')
    ax2.set_ylabel('SA-avg (lower=better)')
    ax2.set_ylim(0,6)

    for i,(b,pa) in enumerate(zip(bars, POST)):
        ax1.annotate(f'{UPT[i]:.1f}%', (b.get_x()+b.get_width()/2, b.get_height()),
                     textcoords='offset points', xytext=(0,3), ha='center', fontsize=9)
        ax1.text(b.get_x()+b.get_width()/2, 3, f'Accept {pa:.0f}%', ha='center', va='bottom', fontsize=8, color='#4b5563')

    fig.suptitle('Post-filters Summary: UPT, SA-avg, Acceptance')
    fig.tight_layout()
    fig.savefig(out_dir / 'post_filters_summary.png', dpi=180)
    fig.savefig(out_dir / 'post_filters_summary.svg')
    plt.close(fig)

    # Per-run UPT figure
    per = data.get('per_run', [])
    runs = sorted({r.get('run') for r in per})
    modes = ['none','basic','rag']
    fig, ax = plt.subplots(figsize=(8,4))
    for mode, color in zip(modes, ['#6b7280','#3b82f6','#1e40af']):
        ys = []
        xs = []
        for r in runs:
            for row in per:
                if row.get('run') == r and row.get('mode') == mode:
                    tot = row.get('total_generated', 0)
                    uniq = row.get('unique_accepted', 0)
                    ys.append(100* (uniq / max(1, tot)))
                    xs.append(r)
                    break
        if xs:
            ax.plot(xs, ys, marker='o', color=color, label=mode)
    ax.set_xlabel('Run')
    ax.set_ylabel('UPT (%)')
    ax.set_ylim(0, 100)
    ax.set_xticks(runs)
    ax.legend()
    fig.suptitle('A/B/C Per‑run UPT')
    fig.tight_layout()
    fig.savefig(out_dir / 'abtest_runs_upt.png', dpi=180)
    fig.savefig(out_dir / 'abtest_runs_upt.svg')
    plt.close(fig)

    print('Saved figures to', out_dir)

if __name__ == '__main__':
    main()
