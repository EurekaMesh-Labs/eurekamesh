import json
import glob
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE = Path('publish') if Path('publish/code').exists() else Path('.')


def load_metrics_csv() -> pd.DataFrame:
    csv_path = BASE / 'data' / 'metrics_aggregate.csv'
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def load_latest_abtest() -> dict:
    candidates = sorted(glob.glob(str(BASE / 'abtest_*results_*.json')))
    if not candidates:
        candidates = sorted(glob.glob(str(BASE / 'data' / 'abtest_*results_*.json')))
    if not candidates:
        return {}
    with open(candidates[-1]) as f:
        return json.load(f)


def main():
    st.set_page_config(page_title="EurekaMesh CCAD Dashboard", layout="wide")
    st.title("EurekaMesh CCAD – Dashboard")
    st.caption("Quick view of runs, metrics, costs and A/B/C aggregates")

    # Metrics CSV
    df = load_metrics_csv()
    if df.empty:
        st.warning("No metrics_aggregate.csv found yet. Run 'make report' first.")
    else:
        st.subheader("Aggregated Metrics (CSV)")
        st.dataframe(df, use_container_width=True, height=320)

        # Simple pivots
        with st.expander("Summary by mode"):
            if 'mode' in df.columns and 'upt' in df.columns:
                pivot = df.groupby('mode', dropna=False)['upt'].mean().sort_values(ascending=False)
                st.bar_chart(pivot)

    # Latest A/B/C
    st.subheader("Latest A/B/C Summary")
    data = load_latest_abtest()
    if not data:
        st.info("No A/B/C results found. Run 'make abtest' first.")
        return

    agg = data.get('aggregate', {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Max context items", data.get('max_context_items', 0))
    col2.metric("Target per mode", data.get('target', 0))
    col3.metric("Runs", data.get('runs', 0))

    # Bars: UPT and Duplicate rate
    labels = ['none', 'basic', 'rag']
    upt = [agg.get(k, {}).get('upt', 0) * 100 for k in labels]
    dup = [agg.get(k, {}).get('dup_rate', agg.get(k, {}).get('dup', 0)) * 100 for k in labels]

    fig, ax1 = plt.subplots(figsize=(6, 3))
    x = range(len(labels))
    ax1.bar([i - 0.2 for i in x], upt, width=0.4, color='#3b82f6', label='UPT (%)')
    ax2 = ax1.twinx()
    ax2.bar([i + 0.2 for i in x], dup, width=0.4, color='#ef4444', label='Dup (%)')
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)
    ax1.set_ylabel('UPT (%)')
    ax2.set_ylabel('Dup (%)')
    fig.tight_layout()
    st.pyplot(fig)

    # Costs per mode
    cost_data = [(k, agg.get(k, {}).get('cost_usd', 0.0)) for k in labels]
    st.write("Estimated cost (USD) per mode:", cost_data)


if __name__ == "__main__":
    main()



