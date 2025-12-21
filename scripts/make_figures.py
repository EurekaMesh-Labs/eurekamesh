import shutil
from pathlib import Path

def main():
    root = Path('publish') if Path('publish/code').exists() else Path('.')
    src = root / 'figures'
    dst = root / 'docs' / 'figures'
    dst.mkdir(parents=True, exist_ok=True)
    to_copy = ['abtest_upt_dup.svg', 'abtest_runs_upt.svg', 'post_filters_summary.svg']
    for name in to_copy:
        s = src / name
        if s.exists():
            shutil.copy2(s, dst / name)
    print(f"Copied figures to {dst}")

if __name__ == "__main__":
    main()


