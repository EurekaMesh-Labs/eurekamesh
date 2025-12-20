import csv
from pathlib import Path
from datetime import datetime

# Support both monorepo (publish/ as subdir) and repo-root usage
BASE = Path('publish') if Path('publish/code').exists() else Path('.')

DATA_CSV = BASE / 'data' / 'metrics_aggregate.csv'
FIG_DIR = BASE / 'figures'
OUT_DIR = BASE / 'report'
OUT_HTML = OUT_DIR / 'index.html'

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>EurekaMesh CCAD – Report</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
    h1 { margin-top: 0; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 14px; }
    th { background: #f3f4f6; text-align: left; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px,1fr)); gap: 16px; }
    .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }
    .muted { color: #6b7280; font-size: 12px; }
  </style>
</head>
<body>
  <h1>EurekaMesh CCAD – Report</h1>
  <div class="muted">Generated: __GENERATED__</div>

  <h2>Summary Figures</h2>
  <div class="grid">
    <div class="card">
      <h3>UPT vs Duplicate Rate (A/B/C)</h3>
      <img src="../figures/abtest_upt_dup.png" style="max-width:100%"/>
    </div>
    <div class="card">
      <h3>Post-filters Summary</h3>
      <img src="../figures/post_filters_summary.png" style="max-width:100%"/>
    </div>
  </div>

  <h2>Metrics (Aggregated)</h2>
  <table>
    <thead><tr>__HEAD__</tr></thead>
    <tbody>
      __ROWS__
    </tbody>
  </table>
</body>
</html>
"""

COLUMNS = [
  'source','type','timestamp','model','mode','n_target','run',
  'total','valid','unique_canonical',
  'upt','upt_valid','valid_rate',
  'dup_rate','dup_rate_raw',
  'cost_usd'
]


def load_rows():
    if not DATA_CSV.exists():
        return []
    out = []
    with open(DATA_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(row)
    return out


def render_table_rows(rows):
    cells = []
    for row in rows:
        tds = ''.join(f"<td>{row.get(col,'')}</td>" for col in COLUMNS)
        cells.append(f"<tr>{tds}</tr>")
    return '\n'.join(cells)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    head = ''.join(f"<th>{c}</th>" for c in COLUMNS)
    html = TEMPLATE.replace('__GENERATED__', datetime.utcnow().isoformat()) \
                   .replace('__HEAD__', head) \
                   .replace('__ROWS__', render_table_rows(rows))
    OUT_HTML.write_text(html, encoding='utf-8')
    print('Saved:', OUT_HTML)

if __name__ == '__main__':
    main()
